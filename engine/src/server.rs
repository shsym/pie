use crate::instance::Id as InstanceId;
use crate::runtime::RuntimeError;
use crate::service::{Service, ServiceError};
use crate::utils::IdPool;
use crate::{messaging, runtime, service};
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task;
use tokio_tungstenite::accept_async;
use tungstenite::protocol::Message as WsMessage;
use uuid::Uuid;

// mod pb_bindings {
//     include!(concat!(env!("OUT_DIR"), "/client.rs"));
// }
const CHUNK_SIZE_BYTES: usize = 64 * 1024; // 64 KiB

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
    #[serde(rename = "query_existence")]
    QueryExistence { hash: String },

    #[serde(rename = "upload_program")]
    UploadProgram {
        hash: String,
        chunk_index: usize,
        total_chunks: usize,
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },

    #[serde(rename = "start_program")]
    StartProgram { hash: String },

    #[serde(rename = "send_event")]
    SendEvent {
        instance_id: String,
        event_data: String,
    },

    #[serde(rename = "terminate_program")]
    TerminateProgram { instance_id: String },
}

/// Messages from server -> client
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "query_response")]
    QueryResponse { hash: String, exists: bool },

    #[serde(rename = "upload_ack")]
    UploadAck { hash: String, chunk_index: usize },

    #[serde(rename = "upload_complete")]
    UploadComplete { hash: String },

    #[serde(rename = "program_launched")]
    ProgramLaunched { hash: String, instance_id: String },

    #[serde(rename = "program_event")]
    ProgramEvent {
        instance_id: String,
        event_data: String,
    },

    #[serde(rename = "program_terminated")]
    ProgramTerminated { instance_id: String, reason: String },

    #[serde(rename = "error")]
    Error { error: String },
}

type ConnectionId = u32;

#[derive(Debug)]
pub enum Command {
    Send { inst: InstanceId, message: String },

    Terminate { inst: InstanceId, reason: String },
}

impl Command {
    pub fn dispatch(self) -> Result<(), ServiceError> {
        service::dispatch(service::SERVICE_SERVER, self)
    }
}

struct ServerState {
    connection_id_pool: Mutex<IdPool<ConnectionId>>,
    connections: DashMap<ConnectionId, Connection>,
    instance_chans: DashMap<InstanceId, mpsc::Sender<ConnectionCommand>>,
}

pub struct Server {
    state: Arc<ServerState>,
    listener_loop: task::JoinHandle<()>,
}

impl Server {
    pub fn new(addr: &str) -> Self {
        let state = Arc::new(ServerState {
            connection_id_pool: Mutex::new(IdPool::new(ConnectionId::MAX)),
            connections: DashMap::new(),
            instance_chans: DashMap::new(),
        });

        let listener_loop = task::spawn(Self::listener_loop(addr, state.clone()));
        Server {
            state,
            listener_loop,
        }
    }

    async fn listener_loop(addr: &str, state: Arc<ServerState>) {
        let listener = TcpListener::bind(addr).await.unwrap();

        while let Ok((stream, addr)) = listener.accept().await {
            let id = {
                let mut id_pool = state.connection_id_pool.lock().await;
                id_pool.acquire().unwrap()
            };
            let connection = Connection::new(id, stream, state.clone());

            state.connections.insert(id, connection);
        }
    }
}

impl Service for Server {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            Command::Send { inst, message } => {
                if let Some(sender) = self.state.instance_chans.get(&inst) {
                    sender
                        .send(ConnectionCommand::Send(ServerMessage::ProgramEvent {
                            instance_id: inst.to_string(),
                            event_data: message,
                        }))
                        .await
                        .ok();
                }
            }

            Command::Terminate { inst, reason } => {
                if let Some(sender) = self.state.instance_chans.get(&inst) {
                    sender
                        .send(ConnectionCommand::DetachInstance {
                            instance_id: inst,
                            reason,
                        })
                        .await
                        .ok();
                }
            }
        }
    }
}

enum ConnectionCommand {
    Send(ServerMessage),
    DetachInstance {
        instance_id: InstanceId,
        reason: String,
    },
}

struct Connection {
    id: ConnectionId,

    //sender: mpsc::Sender<ConnectionCommand>,
    handler_loop: task::JoinHandle<()>,
}

impl Connection {
    fn new(id: ConnectionId, stream: TcpStream, state: Arc<ServerState>) -> Self {
        let handler_loop = task::spawn(Self::handler_loop(stream, id, state));
        Connection {
            id,
            //sender: tx,
            handler_loop,
        }
    }

    async fn temp() {}

    async fn handler_loop(stream: TcpStream, id: ConnectionId, state: Arc<ServerState>) {
        let (writer_tx, mut writer_rx) = mpsc::channel(1000);

        let ws_stream = accept_async(stream).await?;
        let (mut ws_writer, mut ws_reader) = ws_stream.split();
        let mut upload_buffer = Vec::new();

        let mut owned_instance_ids = Vec::new();

        loop {
            tokio::select! {
                // Process outgoing messages from the server to the client.
                Some(cmd) = writer_rx.recv() => {

                    match cmd {
                        ConnectionCommand::Send(msg) => {
                            match rmp_serde::to_vec_named(&msg) {
                                Ok(encoded) => {
                                    if let Err(e) = ws_writer.send(WsMessage::Binary(encoded.into())).await {
                                        eprintln!("WS write error: {:?}", e);
                                        break;
                                    }
                                }
                                Err(e) => {
                                    eprintln!("MessagePack encode error: {:?}", e);
                                    break;
                                }
                            }
                        }
                        ConnectionCommand::DetachInstance { instance_id, reason } => {
                            // remove from owned_instances_ids
                            if let Some(_) = state.instance_chans.remove(&instance_id) {
                                owned_instance_ids.retain(|id| id != &instance_id);

                                let cmd = ConnectionCommand::Send(ServerMessage::ProgramTerminated { instance_id:instance_id.to_string(), reason: "User request".to_string() })

                                writer_tx.send(cmd).await?;
                            }
                        }
                    }

                },

                // Process incoming messages from the client.
                Some(result) = ws_reader.next() => {

                    let msg = result?;
                    if msg.is_binary() {
                        // Try to decode the client message
                        let client_msg = rmp_serde::from_slice::<ClientMessage>(&msg.into_data())?;

                        match client_msg {
                            ClientMessage::QueryExistence { hash } => {
                                let (evt_tx, evt_rx) = oneshot::channel();

                                runtime::Command::ProgramExists {
                                    hash: hash.clone(),
                                    event: evt_tx,
                                }
                                .dispatch()?;

                                let exists = evt_rx.await?;

                                writer_tx.send(ConnectionCommand::Send(ServerMessage::QueryResponse { hash, exists })).await?;
                            }
                            ClientMessage::UploadProgram {
                                hash,
                                chunk_index,
                                total_chunks,
                                mut chunk_data,
                            } => {
                                if chunk_data.len() > CHUNK_SIZE_BYTES {
                                    writer_tx.send(ConnectionCommand::Send(ServerMessage::Error {
                                        error: ServerError::ChunkTooLarge {
                                            actual: chunk_data.len(),
                                            limit: CHUNK_SIZE_BYTES,
                                        }
                                        .to_string(),
                                    }))?;
                                } else {
                                    // First chunk
                                    if chunk_index == 0 {
                                        upload_buffer.clear();
                                    }

                                    upload_buffer.append(&mut chunk_data);

                                    // The last chunk
                                    if chunk_index == total_chunks - 1 {
                                        let file_hash = blake3::hash(&upload_buffer).to_hex().to_string();

                                        if file_hash != hash {
                                            writer_tx.send(ConnectionCommand::Send(ServerMessage::Error {
                                                error: ServerError::HashMismatch {
                                                    expected: hash,
                                                    found: file_hash,
                                                }
                                                .to_string(),
                                            }))?;
                                        } else {
                                            let (evt_tx, evt_rx) = oneshot::channel();
                                            runtime::Command::UploadProgram {
                                                hash: file_hash.clone(),
                                                raw: upload_buffer.clone(),
                                                event: evt_tx,
                                            }
                                            .dispatch()?;
                                            let _ = evt_rx.await?;

                                            writer_tx.send(ConnectionCommand::Send(ServerMessage::UploadComplete { hash: file_hash })).await?;
                                        }
                                    }
                                }


                            }
                            ClientMessage::StartProgram { hash } => {
                                let (evt_tx, evt_rx) = oneshot::channel();
                                runtime::Command::LaunchInstance {
                                    hash: hash.clone(),
                                    event: evt_tx,
                                }
                                .dispatch()?;

                                if let Ok(instance_id) = evt_rx.await? {

                                    //register
                                    state.instance_chans.insert(instance_id.clone(), writer_tx.clone());
                                    owned_instance_ids.push(instance_id.clone());
                                    writer_tx.send(ConnectionCommand::Send(ServerMessage::ProgramLaunched { hash, instance_id:instance_id.to_string() })).await?;
                                } else {
                                    writer_tx.send(ConnectionCommand::Send(ServerMessage::Error {
                                        error: "Failed to launch program".to_string(),
                                    })).await?;
                                }

                            }
                            ClientMessage::SendEvent {
                                instance_id,
                                event_data,
                            } => messaging::Command::Broadcast {
                                topic: instance_id.clone(),
                                message: event_data.clone(),
                            }
                            .dispatch()?,
                            ClientMessage::TerminateProgram { instance_id } => {

                                let inst_id = Uuid::parse_str(&instance_id)
                                    .map_err(|_| ServerError::InvalidInstanceId(instance_id.clone()))?;

                                runtime::trap(
                                    inst_id,
                                    "user terminated the program"
                                );

                            }
                        }
                    } else if msg.is_text() {
                        // Return an error message for text frames
                        writer_tx.send(ServerError::TextFrameNotSupported.into()).await.unwrap();
                    } else if msg.is_close() {
                        break;
                    }

                },

                // If both streams are exhausted, exit the loop.
                else => break,
            }
        }

        // remove all instances owned by this connection
        for instance_id in owned_instance_ids {
            if let Some(_) = state.instance_chans.remove(&instance_id) {
                runtime::trap(instance_id, "socket terminated");
            }
        }

        // remove the connection from the state
        state.connections.remove(&id);
    }
}
