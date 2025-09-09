use crate::instance::InstanceId;
use crate::messaging::dispatch_u2i;
use crate::model::Model;
use crate::runtime::RuntimeError;
use crate::service::{Service, ServiceError, install_service};
use crate::utils::IdPool;
use crate::{auth, messaging, model, runtime, service};
use anyhow::Result;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
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
    #[serde(rename = "authenticate")]
    Authenticate { corr_id: u32, token: String },

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
    LaunchInstance {
        corr_id: u32,
        program_hash: String,
        arguments: Vec<String>,
    },

    #[serde(rename = "launch_server_instance")]
    LaunchServerInstance {
        corr_id: u32,
        port: u32,
        program_hash: String,
        arguments: Vec<String>,
    },

    #[serde(rename = "signal_instance")]
    SignalInstance {
        instance_id: String,
        message: String,
    },

    #[serde(rename = "terminate_instance")]
    TerminateInstance { instance_id: String },

    #[serde(rename = "attach_remote_service")]
    AttachRemoteService {
        corr_id: u32,
        endpoint: String,
        service_type: String,
        service_name: String,
    },
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
        event: EventCode,
        message: String,
    },

    #[serde(rename = "server_event")]
    ServerEvent { message: String },
}
#[derive(Debug, Serialize, Deserialize)]
pub enum EventCode {
    Message = 0,
    Completed = 1,
    Aborted = 2,
    Exception = 3,
    ServerError = 4,
    OutOfResources = 5,
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
        termination_code: u32,
        message: String,
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
    enable_auth: bool,
    client_id_pool: Mutex<IdPool<ClientId>>,
    clients: DashMap<ClientId, JoinHandle<()>>,
    instance_chans: DashMap<InstanceId, mpsc::Sender<ClientCommand>>,
}

pub struct Server {
    state: Arc<ServerState>,
    listener_loop: task::JoinHandle<()>,
}

impl Server {
    pub fn new(addr: &str, enable_auth: bool) -> Self {
        let state = Arc::new(ServerState {
            enable_auth,
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
        while let Ok((stream, _addr)) = listener.accept().await {
            let id = {
                let mut id_pool = state.client_id_pool.lock().await;
                id_pool.acquire().unwrap()
            };
            if let Ok(mut client) = Client::new(id, stream, state.clone()).await {
                let client_handle = task::spawn(async move {
                    client.run().await;
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
    authenticated: bool,

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
pub const QUERY_MODEL_STATUS: &str = "model_status";

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
                        match rmp_serde::decode::from_slice::<ClientMessage>(&bin) {
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
            authenticated: !state.enable_auth,
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

    /// Manages the entire lifecycle of a client connection.
    async fn run(&mut self) {
        loop {
            tokio::select! {
                // To prevent starvation, prioritize processing existing messages
                // over checking for disconnection.
                biased;

                // Branch 1: A new command was received from the client or an internal service.
                Some(cmd) = self.incoming_rx.recv() => {
                    self.handle_command(cmd).await;
                },

                // Branch 2: The reader task has finished. This is a primary signal
                // that the client has disconnected.
                _ = &mut self.reader_task => {
                    // println!("Client {} disconnected (reader task finished).", self.id);
                    break;
                },

                // Branch 3: The writer task has finished, likely due to a
                // "broken pipe" error when trying to send a message.
                _ = &mut self.writer_task => {
                    // println!("Client {} disconnected (writer task finished).", self.id);
                    break;
                }

                // The `else` branch is taken when all other branches are disabled,
                // meaning the channel is closed and tasks are done. This is a clean shutdown.
                else => break,
            }
        }

        // The loop has exited, so we can now guarantee that cleanup will run.
        self.cleanup().await;
    }

    /// Processes a single command. This contains the logic from your old `serve` method.
    async fn handle_command(&mut self, cmd: ClientCommand) {
        match cmd {
            ClientCommand::FromClient(message) => match message {
                ClientMessage::Authenticate { corr_id, token } => {
                    self.handle_authenticate(corr_id, token).await
                }
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
                    arguments,
                } => {
                    self.handle_launch_instance(corr_id, program_hash, arguments)
                        .await
                }
                ClientMessage::LaunchServerInstance {
                    corr_id,
                    port,
                    program_hash,
                    arguments,
                } => {
                    self.handle_launch_server_instance(corr_id, port, program_hash, arguments)
                        .await
                }
                ClientMessage::SignalInstance {
                    instance_id,
                    message,
                } => self.handle_signal_instance(instance_id, message).await,
                ClientMessage::TerminateInstance { instance_id } => {
                    self.handle_terminate_instance(instance_id).await
                }
                ClientMessage::AttachRemoteService {
                    corr_id,
                    endpoint,
                    service_type,
                    service_name,
                } => {
                    self.handle_attach_remote_service(
                        corr_id,
                        endpoint,
                        service_type,
                        service_name,
                    )
                    .await;
                }
            },
            ClientCommand::Internal(cmd) => match cmd {
                Command::Send { inst_id, message } => {
                    self.send_inst_event(inst_id, EventCode::Message, message)
                        .await
                }
                Command::DetachInstance {
                    inst_id,
                    termination_code,
                    message,
                } => {
                    self.handle_detach_instance(inst_id, termination_code, message)
                        .await;
                }
            },
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

    async fn send_inst_event(&mut self, inst_id: InstanceId, event: EventCode, message: String) {
        self.send(ServerMessage::InstanceEvent {
            instance_id: inst_id.to_string(),
            event,
            message,
        })
        .await;
    }

    async fn handle_detach_instance(
        &mut self,
        inst_id: InstanceId,
        termination_code: u32,
        message: String,
    ) {
        if !self.authenticated {
            return;
        }
        self.inst_owned.retain(|&id| id != inst_id);

        if self.state.instance_chans.remove(&inst_id).is_some() {
            let event_code = match termination_code {
                0 => EventCode::Completed,
                1 => EventCode::Aborted,
                2 => EventCode::Exception,
                3 => EventCode::ServerError,
                4 => EventCode::OutOfResources,
                _ => EventCode::ServerError,
            };

            self.send_inst_event(inst_id, event_code, message).await;
        }
    }

    async fn handle_authenticate(&mut self, corr_id: u32, token: String) {
        if !self.authenticated {
            if let Ok(claims) = auth::validate_jwt(&token) {
                self.authenticated = true;
                self.send_response(corr_id, true, claims.sub).await;
            } else {
                self.send_response(corr_id, false, "Invalid token".to_string())
                    .await;
            }
        } else {
            self.send_response(corr_id, true, "Already authenticated".to_string())
                .await;
        }
    }

    async fn handle_query(&mut self, corr_id: u32, subject: String, record: String) {
        if !self.authenticated {
            self.send_response(corr_id, false, "Not authenticated".to_string())
                .await;
        }

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
            QUERY_MODEL_STATUS => {
                // gather model status from all attached backends
                let runtime_stats = model::runtime_stats().await;
                let runtime_stats_json = serde_json::to_string(&runtime_stats).unwrap();

                self.send_response(corr_id, true, runtime_stats_json).await;
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
        if !self.authenticated {
            self.send_response(corr_id, false, "Not authenticated".to_string())
                .await;
        }

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

                self.send_response(corr_id, true, file_hash).await;
            }

            self.inflight_upload = None;
        }
    }

    async fn handle_launch_instance(
        &mut self,
        corr_id: u32,
        program_hash: String,
        arguments: Vec<String>,
    ) {
        if !self.authenticated {
            self.send_response(corr_id, false, "Not authenticated".to_string())
                .await;
        }

        let (evt_tx, evt_rx) = oneshot::channel();
        runtime::Command::LaunchInstance {
            program_hash: program_hash.clone(),
            arguments,
            event: evt_tx,
        }
        .dispatch()
        .unwrap();

        match evt_rx.await.unwrap() {
            Ok(instance_id) => {
                //register
                self.state
                    .instance_chans
                    .insert(instance_id, self.incoming_tx.clone());

                self.inst_owned.push(instance_id);
                self.send_response(corr_id, true, instance_id.to_string())
                    .await;
            }
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
            }
        }
    }

    async fn handle_launch_server_instance(
        &mut self,
        corr_id: u32,
        port: u32,
        program_hash: String,
        arguments: Vec<String>,
    ) {
        if !self.authenticated {
            self.send_response(corr_id, false, "Not authenticated".to_string())
                .await;
        }

        let (evt_tx, evt_rx) = oneshot::channel();
        runtime::Command::LaunchServerInstance {
            program_hash: program_hash.clone(),
            port,
            arguments,
            event: evt_tx,
        }
        .dispatch()
        .unwrap();

        match evt_rx.await.unwrap() {
            Ok(_) => {
                self.send_response(corr_id, true, "server launched".to_string())
                    .await;
            }
            Err(e) => {
                self.send_response(corr_id, false, e.to_string()).await;
            }
        }
    }

    async fn handle_signal_instance(&mut self, instance_id: String, message: String) {
        if !self.authenticated {
            return;
        }
        if let Ok(inst_id) = Uuid::parse_str(&instance_id) {
            if self.inst_owned.contains(&inst_id) {
                dispatch_u2i(messaging::PushPullCommand::Push {
                    topic: inst_id.to_string(),
                    message,
                });
            }
        }
    }

    async fn handle_terminate_instance(&mut self, instance_id: String) {
        if !self.authenticated {
            return;
        }
        if let Ok(inst_id) = Uuid::parse_str(&instance_id) {
            if self.inst_owned.contains(&inst_id) {
                runtime::trap(inst_id, runtime::TerminationCause::Signal);
            }
        }
    }

    async fn handle_attach_remote_service(
        &mut self,
        corr_id: u32,
        endpoint: String,
        service_type: String,
        service_name: String,
    ) {
        if !self.authenticated {
            self.send_response(corr_id, false, "Not authenticated".into())
                .await;
            return;
        }

        match service_type.as_str() {
            "model" => {
                // Try to create the model; fail fast on error.
                let model_service = match Model::new(&endpoint).await {
                    Ok(m) => m,
                    Err(e) => {

                        println!("Failed to create model backend: {:?}", e);
                        self.send_response(
                            corr_id,
                            false,
                            "Failed to attach to model backend server".into(),
                        )
                        .await;
                        return;
                    }
                };

                // Try to install; fail fast on error.
                let Some(service_id) = install_service(&service_name, model_service) else {
                    self.send_response(
                        corr_id,
                        false,
                        "Failed to register the model service".into(),
                    )
                    .await;
                    return;
                };

                // Success path.
                model::register_model(service_name, service_id);
                self.send_response(
                    corr_id,
                    true,
                    "Model service registration successful".into(),
                )
                .await;
            }

            other => {
                self.send_response(corr_id, false, format!("Unknown service type: {other}"))
                    .await;
            }
        }
    }

    /// The cleanup logic is now guaranteed to run.
    async fn cleanup(&mut self) {
        // Terminate all instances owned by this client.
        for inst_id in self.inst_owned.drain(..) {
            if self.state.instance_chans.remove(&inst_id).is_some() {
                runtime::trap_exception(inst_id, "socket terminated");
            }
        }

        // Abort the tasks to ensure they are stopped. It's safe to abort
        // tasks that have already completed.
        self.reader_task.abort();
        self.writer_task.abort();

        // Remove the client from the central map.
        self.state.clients.remove(&self.id);

        // Release the client ID back to the pool.
        self.state
            .client_id_pool
            .lock()
            .await
            .release(self.id)
            .expect("Failed to release client ID");
    }
}
