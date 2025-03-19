use crate::instance::Id as InstanceId;
use crate::runtime::RuntimeError;
use crate::service::{Service, ServiceError};
use crate::utils::IdPool;
use crate::{messaging, runtime, service};
use anyhow::Result;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use rmp_serde::decode;
use serde::{Deserialize, Serialize};
use std::mem;
use std::sync::{Arc, OnceLock};
use thiserror::Error;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task;
use tokio::task::JoinHandle;
use tokio_tungstenite::accept_async;
use tungstenite::Message;
use tungstenite::protocol::Message as WsMessage;
use uuid::Uuid;

pub const CHUNK_SIZE_BYTES: usize = 256 * 1024; // 256 KiB
static SERVICE_ID_SERVER: OnceLock<usize> = OnceLock::new();

/// Define the various errors that can happen while handling messages.
#[derive(Debug, Error)]
pub enum ServerError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("WebSocket accept error: {0}")]
    WsAccept(#[from] tungstenite::Error),

    #[error("MessagePack decode error: {0}")]
    MsgPackDecode(#[from] rmp_serde::decode::Error),

    #[error("Text frames not supported")]
    TextFrameNotSupported,

    #[error("Chunk size {actual} exceeds {limit} bytes limit")]
    ChunkTooLarge { actual: usize, limit: usize },

    #[error("Mismatch in total_chunks: was {was}, now {now}")]
    ChunkCountMismatch { was: usize, now: usize },

    #[error("Out-of-order chunk: expected {expected}, got {got}")]
    OutOfOrderChunk { expected: usize, got: usize },

    #[error("Hash mismatch: expected {expected}, got {found})")]
    HashMismatch { expected: String, found: String },

    #[error("Invalid instance_id: {0}")]
    InvalidInstanceId(String),

    #[error("Instance {instance} not owned by client")]
    NotOwnedInstance { instance: String },

    #[error("No such running instance: {0}")]
    NoSuchRunningInstance(String),

    #[error("Failed to write program: {0}")]
    FileWriteError(#[source] std::io::Error),

    #[error("Failed to start program: {0}")]
    StartProgramFailed(#[from] RuntimeError),
}

/// Messages from client -> server
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "query")]
    Query {
        corr_id: u32,
        subject: String,
        record: String,
    },

    #[serde(rename = "upload_program")]
    UploadProgram {
        corr_id: u32,
        program_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },

    #[serde(rename = "launch_instance")]
    LaunchInstance { corr_id: u32, program_hash: String },

    #[serde(rename = "signal_instance")]
    SignalInstance {
        instance_id: String,
        message: String,
    },

    #[serde(rename = "terminate_instance")]
    TerminateInstance { instance_id: String },
}

/// Messages from server -> client
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "response")]
    Response {
        corr_id: u32,
        successful: bool,
        result: String,
    },

    #[serde(rename = "instance_event")]
    InstanceEvent {
        instance_id: String,
        event: String,
        message: String,
    },

    #[serde(rename = "server_event")]
    ServerEvent { message: String },
}

type ClientId = u32;

#[derive(Debug)]
pub enum Command {
    Send {
        inst_id: InstanceId,
        message: String,
    },
    DetachInstance {
        inst_id: InstanceId,
        reason: String,
    },
}

impl Command {
    pub fn dispatch(self) -> Result<(), ServiceError> {
        let service_id =
            *SERVICE_ID_SERVER.get_or_init(move || service::get_service_id("server").unwrap());
        service::dispatch(service_id, self)
    }
}

struct ServerState {
    client_id_pool: Mutex<IdPool<ClientId>>,
    clients: DashMap<ClientId, JoinHandle<()>>,
    instance_chans: DashMap<InstanceId, mpsc::Sender<ClientCommand>>,
}

pub struct Server {
    state: Arc<ServerState>,
    listener_loop: task::JoinHandle<()>,
}

impl Server {
    pub fn new(addr: &str) -> Self {
        let state = Arc::new(ServerState {
            client_id_pool: Mutex::new(IdPool::new(ClientId::MAX)),
            clients: DashMap::new(),
            instance_chans: DashMap::new(),
        });

        let listener_loop = task::spawn(Self::listener_loop(addr.to_string(), state.clone()));
        Server {
            state,
            listener_loop,
        }
    }

    async fn listener_loop(addr: String, state: Arc<ServerState>) {
        let listener = TcpListener::bind(addr).await.unwrap();
        while let Ok((stream, addr)) = listener.accept().await {
            let id = {
                let mut id_pool = state.client_id_pool.lock().await;
                id_pool.acquire().unwrap()
            };
            if let Ok(mut client) = Client::new(id, stream, state.clone()).await {
                let client_handle = task::spawn(async move {
                    client.serve().await;
                    client.cleanup().await;
                });

                state.clients.insert(id, client_handle);
            }
        }
    }
}

impl Service for Server {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        let inst_id = match cmd {
            Command::Send { inst_id, .. } | Command::DetachInstance { inst_id, .. } => inst_id,
        };

        // send it to the client if it's connected
        if let Some(chan) = self.state.instance_chans.get(&inst_id) {
            chan.send(ClientCommand::Internal(cmd)).await.ok();
        }
    }
}

struct InFlightUpload {
    program_hash: String,
    total_chunks: usize,
    buffer: Vec<u8>,
}

struct Client {
    id: ClientId,
    state: Arc<ServerState>,

    inflight_upload: Option<InFlightUpload>,
    inst_owned: Vec<InstanceId>,

    write_tx: mpsc::Sender<WsMessage>,
    incoming_rx: mpsc::Receiver<ClientCommand>,
    incoming_tx: mpsc::Sender<ClientCommand>,

    writer_task: JoinHandle<()>,
    reader_task: JoinHandle<()>,
}

enum ClientCommand {
    FromClient(ClientMessage),
    Internal(Command),
}

pub const QUERY_PROGRAM_EXISTS: &str = "program_exists";

impl Client {
    async fn new(id: ClientId, stream: TcpStream, state: Arc<ServerState>) -> Result<Self> {
        let (write_tx, mut write_rx) = mpsc::channel(1000);
        let (incoming_tx, incoming_rx) = mpsc::channel(1000);

        let ws_stream = accept_async(stream).await?;
        let (mut ws_writer, mut ws_reader) = ws_stream.split();

        let writer_task = task::spawn(async move {
            while let Some(message) = write_rx.recv().await {
                if let Err(e) = ws_writer.send(message).await {
                    println!("Error writing to ws stream: {:?}", e);
                    break;
                }
            }
        });

        let incoming_tx_ = incoming_tx.clone();
        let reader_task = task::spawn(async move {
            let incoming_tx = incoming_tx_;
            while let Some(Ok(msg)) = ws_reader.next().await {
                match msg {
                    Message::Binary(bin) => {
                        // Decode via rmp-serde
                        match decode::from_slice::<ClientMessage>(&bin) {
                            Ok(client_message) => {
                                incoming_tx
                                    .send(ClientCommand::FromClient(client_message))
                                    .await
                                    .ok();
                            }
                            Err(e) => {
                                eprintln!("Failed to decode client msgpack: {:?}", e);
                            }
                        }
                    }
                    Message::Close(_) => {
                        break;
                    }
                    Message::Text(_) | Message::Ping(_) | Message::Pong(_) => {}
                    _ => {
                        eprintln!("Unexpected message type from client");
                        // ignore
                    }
                }
            }
        });

        Ok(Self {
            id,
            state,
            inflight_upload: None,
            inst_owned: Vec::new(),
            write_tx,
            incoming_rx,
            incoming_tx,
            writer_task,
            reader_task,
        })
    }

    async fn serve(&mut self) {
        while let Some(cmd) = self.incoming_rx.recv().await {
            match cmd {
                ClientCommand::FromClient(message) => match message {
                    ClientMessage::Query {
                        corr_id,
                        subject,
                        record,
                    } => self.handle_query(corr_id, subject, record).await,
                    ClientMessage::UploadProgram {
                        corr_id,
                        program_hash,
                        chunk_index,
                        total_chunks,
                        chunk_data,
                    } => {
                        self.handle_upload_program(
                            corr_id,
                            program_hash,
                            chunk_index,
                            total_chunks,
                            chunk_data,
                        )
                        .await
                    }
                    ClientMessage::LaunchInstance {
                        corr_id,
                        program_hash,
                    } => self.handle_launch_instance(corr_id, program_hash).await,
                    ClientMessage::SignalInstance {
                        instance_id,
                        message,
                    } => self.handle_signal_instance(instance_id, message).await,
                    ClientMessage::TerminateInstance { instance_id } => {
                        self.handle_terminate_instance(instance_id).await
                    }
                },
                ClientCommand::Internal(cmd) => match cmd {
                    Command::Send { inst_id, message } => {
                        self.send_inst_event(inst_id, "message".to_string(), message)
                            .await
                    }
                    Command::DetachInstance { inst_id, reason } => {
                        self.handle_detach_instance(inst_id, reason).await;
                    }
                },
            }
        }
    }

    async fn send(&mut self, msg: ServerMessage) {
        match rmp_serde::to_vec_named(&msg) {
            Ok(encoded) => {
                if let Err(e) = self.write_tx.send(WsMessage::Binary(encoded.into())).await {
                    eprintln!("WS write error: {:?}", e);
                }
            }
            Err(e) => {
                eprintln!("MessagePack encode error: {:?}", e);
            }
        }
    }

    async fn send_response(&mut self, corr_id: u32, successful: bool, result: String) {
        let msg = ServerMessage::Response {
            corr_id,
            successful,
            result,
        };
        self.send(msg).await;
    }

    async fn send_inst_event(&mut self, inst_id: InstanceId, event: String, message: String) {
        self.send(ServerMessage::InstanceEvent {
            instance_id: inst_id.to_string(),
            event,
            message,
        })
        .await;
    }

    async fn handle_detach_instance(&mut self, inst_id: InstanceId, reason: String) {
        self.inst_owned.retain(|&id| id != inst_id);

        if let Some(_) = self.state.instance_chans.remove(&inst_id) {

            self.send_inst_event(inst_id, "terminated".to_string(), reason)
                .await;
        }
    }
    async fn handle_query(&mut self, corr_id: u32, subject: String, record: String) {
        match subject.as_str() {
            QUERY_PROGRAM_EXISTS => {
                let (evt_tx, evt_rx) = oneshot::channel();

                runtime::Command::ProgramExists {
                    hash: record.clone(),
                    event: evt_tx,
                }
                .dispatch()
                .unwrap();

                let exists = evt_rx.await.unwrap();
                self.send_response(corr_id, true, exists.to_string()).await;
            }
            _ => {
                println!("Unknown query subject: {}", subject);
            }
        }
    }

    async fn handle_upload_program(
        &mut self,
        corr_id: u32,
        program_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        mut chunk_data: Vec<u8>,
    ) {
        if self.inflight_upload.is_none() {
            self.inflight_upload = Some(InFlightUpload {
                program_hash: program_hash.clone(),
                total_chunks,
                buffer: Vec::new(),
            });
        }

        if chunk_data.len() > CHUNK_SIZE_BYTES {
            self.send_response(
                corr_id,
                false,
                format!(
                    "chunk size {} exceeds limit {}",
                    chunk_data.len(),
                    CHUNK_SIZE_BYTES
                ),
            )
            .await;

            self.inflight_upload = None;
            return;
        }

        let inflight_upload = self.inflight_upload.as_mut().unwrap();
        inflight_upload.buffer.append(&mut chunk_data);

        // The last chunk
        if chunk_index == total_chunks - 1 {
            let file_hash = blake3::hash(&inflight_upload.buffer).to_hex().to_string();

            if file_hash != inflight_upload.program_hash {
                self.send_response(
                    corr_id,
                    false,
                    format!(
                        "hash mismatch: expected {}, got {}",
                        program_hash, file_hash
                    ),
                )
                .await;
            } else {
                let (evt_tx, evt_rx) = oneshot::channel();
                runtime::Command::UploadProgram {
                    hash: file_hash.clone(),
                    raw: mem::take(&mut inflight_upload.buffer),
                    event: evt_tx,
                }
                .dispatch()
                .unwrap();
                let _ = evt_rx.await.unwrap();

                println!("Uploaded program with hash {}", file_hash);

                self.send_response(corr_id, true, file_hash).await;
            }

            self.inflight_upload = None;
        }
    }

    async fn handle_launch_instance(&mut self, corr_id: u32, program_hash: String) {
        let (evt_tx, evt_rx) = oneshot::channel();
        runtime::Command::LaunchInstance {
            program_hash: program_hash.clone(),
            event: evt_tx,
        }
        .dispatch()
        .unwrap();

        match evt_rx.await.unwrap() {
            Ok(instance_id) => {
                //register
                self.state
                    .instance_chans
                    .insert(instance_id.clone(), self.incoming_tx.clone());

                self.inst_owned.push(instance_id.clone());
                self.send_response(corr_id, true, instance_id.to_string())
                    .await;
            }
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
            }
        }
    }

    async fn handle_signal_instance(&mut self, instance_id: String, message: String) {
        if let Ok(inst_id) = Uuid::parse_str(&instance_id) {
            if self.inst_owned.contains(&inst_id) {
                messaging::Command::Broadcast {
                    topic: inst_id.to_string(),
                    message,
                }
                .dispatch()
                .unwrap();
            }
        }
    }

    async fn handle_terminate_instance(&mut self, instance_id: String) {
        if let Ok(inst_id) = Uuid::parse_str(&instance_id) {
            if self.inst_owned.contains(&inst_id) {
                runtime::trap(inst_id, "user terminated the program");
            }
        }
    }

    async fn cleanup(&mut self) {
        // remove all instances owned by this connection
        for inst_id in self.inst_owned.iter() {
            if let Some(_) = self.state.instance_chans.remove(inst_id) {
                runtime::trap(*inst_id, "socket terminated");
            }
        }

        self.reader_task.abort();
        self.writer_task.abort();

        // remove the connection from the state
        self.state.clients.remove(&self.id);

        self.state
            .client_id_pool
            .lock()
            .await
            .release(self.id)
            .unwrap()
    }
}
