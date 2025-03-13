use crate::instance_old::Id as InstanceId;
use crate::runtime::{Runtime, RuntimeError};
use anyhow::Result;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{
    mpsc::{channel, Sender},
    Mutex,
};
use tokio::task;
use tokio_tungstenite::accept_async;
use tungstenite::protocol::Message as WsMessage;
use uuid::Uuid;

/// The maximum chunk size we allow in an upload.
const CHUNK_SIZE_BYTES: usize = 64 * 1024; // 64 KiB

/// Per-client ID
pub type Id = Uuid;

/// In-progress upload
struct InFlightUpload {
    total_chunks: usize,
    collected_data: Vec<u8>,
    received_chunks: usize,
}

struct ClientHandle {
    to_origin: Sender<ServerMessage>,
    instances: Vec<InstanceId>,
}

/// Our WebSocket server's global state:
///  - each connection can look up the controller’s `RuntimeState`
///  - track file uploads in progress
///  - track connected clients, etc.
pub struct ServerState {
    /// The “controller” or “runtime” data we share with the controlling logic
    runtime: Arc<Runtime>,

    /// Tracks partial uploads in progress (hash -> InFlightUpload)
    uploads_in_flight: DashMap<String, Arc<Mutex<InFlightUpload>>>,

    /// Map of clients
    clients: DashMap<Id, ClientHandle>,
}

impl ServerState {
    pub fn new(runtime: Arc<Runtime>) -> Self {
        Self {
            runtime,
            uploads_in_flight: DashMap::new(),
            clients: DashMap::new(),
        }
    }
}

/// The actual server.
/// This struct is fairly minimal here—just holds an `Arc<ServerState>`.
pub struct WebSocketServer {
    state: Arc<ServerState>,
}

impl WebSocketServer {
    pub fn new(state: Arc<ServerState>) -> Self {
        Self { state }
    }

    /// Start listening on a TCP address, accept new websockets, etc.
    pub async fn run(self, addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;

        loop {
            let (stream, _) = listener.accept().await?;
            let state_clone = self.state.clone();
            task::spawn(async move {
                if let Err(e) = handle_connection(stream, state_clone).await {
                    eprintln!("Connection error: {:?}", e);
                }
            });
        }
    }
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

    #[error("Hash mismatch: expected {expected}, got {found} (chunks={chunks})")]
    HashMismatch {
        expected: String,
        found: String,
        chunks: usize,
    },

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

/// Convert a `ServerError` into a single `ServerMessage::Error`.
impl From<ServerError> for ServerMessage {
    fn from(err: ServerError) -> Self {
        ServerMessage::Error {
            error: err.to_string(),
        }
    }
}

/// A handy alias: we return a list of `ServerMessage` on success, or a single `ServerError` on error.
type ServerResult = std::result::Result<Vec<ServerMessage>, ServerError>;

async fn handle_connection(stream: TcpStream, state: Arc<ServerState>) -> Result<()> {
    let ws_stream = accept_async(stream).await?;
    let (mut ws_writer, mut ws_reader) = ws_stream.split();

    let (server2client_tx, mut server2client_rx) = channel::<ServerMessage>(128);

    let client_id = Uuid::new_v4();
    let client_handle = ClientHandle {
        to_origin: server2client_tx.clone(),
        instances: Vec::new(),
    };
    state.clients.insert(client_id, client_handle);

    // Writer task (Server -> Client)
    let writer_task = task::spawn(async move {
        while let Some(msg) = server2client_rx.recv().await {
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
    });

    // Reader loop
    while let Some(msg) = ws_reader.next().await {
        // If we got an error from the WS stream itself, bail out
        let msg = msg?;

        if msg.is_binary() {
            // Try to decode the client message
            let client_msg = rmp_serde::from_slice::<ClientMessage>(&msg.into_data())?;

            // Dispatch & handle
            let responses = handle_client_message(state.clone(), client_id, client_msg).await;
            match responses {
                Ok(msgs) => {
                    for m in msgs {
                        let _ = server2client_tx.send(m).await;
                    }
                }
                Err(err) => {
                    // Convert the error to a single error message
                    let _ = server2client_tx.send(err.into()).await;
                }
            }
        } else if msg.is_text() {
            // Return an error message for text frames
            let _ = server2client_tx
                .send(ServerError::TextFrameNotSupported.into())
                .await;
        } else if msg.is_close() {
            break;
        }
    }

    // Cleanup
    writer_task.abort();
    state.clients.remove(&client_id);
    Ok(())
}

/// Dispatch a `ClientMessage` to the correct handler
async fn handle_client_message(
    state: Arc<ServerState>,
    client_id: Id,
    msg: ClientMessage,
) -> ServerResult {
    match msg {
        ClientMessage::QueryExistence { hash } => handle_query_existence(state, hash).await,
        ClientMessage::UploadProgram {
            hash,
            chunk_index,
            total_chunks,
            chunk_data,
        } => handle_upload_program(state, hash, chunk_index, total_chunks, chunk_data).await,
        ClientMessage::StartProgram { hash } => handle_start_program(state, client_id, hash).await,
        ClientMessage::SendEvent {
            instance_id,
            event_data,
        } => handle_send_event(state, client_id, instance_id, event_data).await,
        ClientMessage::TerminateProgram { instance_id } => {
            handle_terminate_program(state, client_id, instance_id).await
        }
    }
}

async fn handle_query_existence(state: Arc<ServerState>, hash: String) -> ServerResult {
    let exists = state.runtime.programs_in_disk.contains_key(&hash);
    Ok(vec![ServerMessage::QueryResponse { hash, exists }])
}

async fn handle_upload_program(
    state: Arc<ServerState>,
    hash: String,
    chunk_index: usize,
    total_chunks: usize,
    chunk_data: Vec<u8>,
) -> ServerResult {
    if chunk_data.len() > CHUNK_SIZE_BYTES {
        return Err(ServerError::ChunkTooLarge {
            actual: chunk_data.len(),
            limit: CHUNK_SIZE_BYTES,
        });
    }

    let upload_entry = state
        .uploads_in_flight
        .entry(hash.clone())
        .or_insert_with(|| {
            Arc::new(Mutex::new(InFlightUpload {
                total_chunks,
                collected_data: Vec::with_capacity(total_chunks * CHUNK_SIZE_BYTES),
                received_chunks: 0,
            }))
        })
        .clone();

    let mut inflight = upload_entry.lock().await;

    if inflight.total_chunks != total_chunks {
        return Err(ServerError::ChunkCountMismatch {
            was: inflight.total_chunks,
            now: total_chunks,
        });
    }

    if inflight.received_chunks != chunk_index {
        return Err(ServerError::OutOfOrderChunk {
            expected: inflight.received_chunks,
            got: chunk_index,
        });
    }

    inflight.collected_data.extend_from_slice(&chunk_data);
    inflight.received_chunks += 1;

    let mut replies = vec![ServerMessage::UploadAck {
        hash: hash.clone(),
        chunk_index,
    }];

    // If last chunk:
    if inflight.received_chunks == inflight.total_chunks {
        let final_data = std::mem::take(&mut inflight.collected_data);
        let total_chunks_stored = inflight.total_chunks;
        drop(inflight);
        state.uploads_in_flight.remove(&hash);

        let actual_hash = blake3::hash(&final_data).to_hex().to_string();
        if actual_hash != hash {
            return Err(ServerError::HashMismatch {
                expected: hash,
                found: actual_hash,
                chunks: total_chunks_stored,
            });
        }

        // Write to disk if not yet present
        if state.runtime.programs_in_disk.get(&hash).is_none() {
            let file_path = std::path::Path::new(super::PROGRAM_CACHE_DIR).join(&hash);
            // If writing fails, return an error
            std::fs::write(&file_path, &final_data).map_err(ServerError::FileWriteError)?;
            state
                .runtime
                .programs_in_disk
                .insert(hash.clone(), file_path);
        }
        replies.push(ServerMessage::UploadComplete { hash });
    }

    Ok(replies)
}

async fn handle_start_program(
    state: Arc<ServerState>,
    client_id: Id,
    hash: String,
) -> ServerResult {
    let tx = state
        .clients
        .get(&client_id)
        .expect("Client must exist.")
        .to_origin
        .clone();

    let instance_id = state
        .runtime
        .start_program(&hash, tx)
        .await
        // Convert anyhow::Error to our custom error variant
        .map_err(|e| ServerError::StartProgramFailed(e))?;

    // Track the instance in the client handle
    if let Some(mut client) = state.clients.get_mut(&client_id) {
        client.instances.push(instance_id);
    }

    Ok(vec![ServerMessage::ProgramLaunched {
        hash,
        instance_id: instance_id.to_string(),
    }])
}

async fn handle_send_event(
    state: Arc<ServerState>,
    client_id: Id,
    instance_id: String,
    event_data: String,
) -> ServerResult {
    let inst_id = Uuid::parse_str(&instance_id)
        .map_err(|_| ServerError::InvalidInstanceId(instance_id.clone()))?;

    // Check if the instance is owned by the client
    let handle = state.clients.get(&client_id).unwrap(); // or expect

    if !handle.instances.contains(&inst_id) {
        return Err(ServerError::NotOwnedInstance {
            instance: instance_id,
        });
    }

    if let Some(runtime_handle) = state.runtime.running_instances.get(&inst_id) {
        // Send event to the instance
        runtime_handle
            .evt_from_origin
            .send(event_data)
            .await
            .unwrap();
        // If no immediate server messages are needed, return an empty list
        Ok(vec![])
    } else {
        Err(ServerError::NoSuchRunningInstance(inst_id.to_string()))
    }
}

async fn handle_terminate_program(
    state: Arc<ServerState>,
    client_id: Id,
    instance_id: String,
) -> ServerResult {
    let inst_id = Uuid::parse_str(&instance_id)
        .map_err(|_| ServerError::InvalidInstanceId(instance_id.clone()))?;

    let handle = state.clients.get(&client_id).unwrap();
    if !handle.instances.contains(&inst_id) {
        return Err(ServerError::NotOwnedInstance {
            instance: instance_id,
        });
    }

    let was_terminated = state.runtime.terminate_program(inst_id);
    if was_terminated {
        Ok(vec![ServerMessage::ProgramTerminated {
            instance_id,
            reason: "User requested".to_string(),
        }])
    } else {
        Err(ServerError::NoSuchRunningInstance(inst_id.to_string()))
    }
}
