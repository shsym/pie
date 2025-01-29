use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

use parking_lot::Mutex;

use tokio::net::{TcpListener, TcpStream};
use tokio::task;
use tokio_tungstenite::accept_async;
use tungstenite::protocol::Message as WsMessage;

use blake3::Hasher;
use futures::{SinkExt, StreamExt};
use uuid::Uuid;

// Wasmtime imports
use wasmtime::{Engine, Module, Store};

// For MessagePack serialization/deserialization.
use rmp_serde::{decode::from_slice, encode::to_vec_named};
use serde::{Deserialize, Serialize};

/// Directory where we store (and load) program binaries:
const PROGRAM_CACHE_DIR: &str = "./program_cache";
/// For demonstration, we define a maximum chunk size. The client
/// is expected to respect this, though in practice, you could
/// negotiate or override it.
const CHUNK_SIZE_BYTES: usize = 64 * 1024; // 64 KiB

// ---------------------------
// Server State
// ---------------------------

/// Global server state:
/// - Program cache on disk
/// - In-flight uploads
/// - Running program instances
#[derive(Default)]
struct ServerState {
    /// Maps a BLAKE3 hash -> Path to the cached binary on disk
    program_cache: HashMap<String, PathBuf>,

    /// Tracks partial uploads in progress
    in_flight_uploads: HashMap<String, InFlightUpload>,

    /// Running program instances (instance_id -> handle)
    running_instances: HashMap<String, ProgramInstanceHandle>,
}

/// In-progress upload info
struct InFlightUpload {
    total_chunks: usize,
    collected_data: Vec<u8>,
    received_chunks: usize,
}

/// Minimal handle for a running program instance
struct ProgramInstanceHandle {
    hash: String,
    store: Store<()>,
}

// ---------------------------
// Message Definitions
// ---------------------------

/// Messages from client -> server (in MessagePack)
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ClientMessage {
    /// Query if a program with a given hash exists in the server cache
    #[serde(rename = "query_existence")]
    QueryExistence { hash: String },

    /// Upload a program in chunks
    #[serde(rename = "upload_program")]
    UploadProgram {
        hash: String,
        chunk_index: usize,
        total_chunks: usize,
        /// Raw chunk data (no base64, since we use MessagePack binary)
        chunk_data: Vec<u8>,
    },

    /// Start running a cached program
    #[serde(rename = "start_program")]
    StartProgram {
        hash: String,
        #[serde(default)]
        configuration: serde_json::Value,
    },

    /// Send an event (arbitrary data) to a running program instance
    #[serde(rename = "send_event")]
    SendEvent {
        hash: String,
        instance_id: String,
        event_data: serde_json::Value,
    },

    /// Terminate a running program instance
    #[serde(rename = "terminate_program")]
    TerminateProgram { hash: String, instance_id: String },
}

/// Messages from server -> client (in MessagePack)
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ServerMessage<'a> {
    #[serde(rename = "query_response")]
    QueryResponse { hash: &'a str, exists: bool },

    #[serde(rename = "upload_ack")]
    UploadAck { hash: &'a str, chunk_index: usize },

    #[serde(rename = "upload_complete")]
    UploadComplete { hash: &'a str },

    #[serde(rename = "program_launched")]
    ProgramLaunched { hash: &'a str, instance_id: &'a str },

    #[serde(rename = "program_event")]
    ProgramEvent {
        hash: &'a str,
        instance_id: &'a str,
        event_data: serde_json::Value,
    },

    #[serde(rename = "program_terminated")]
    ProgramTerminated { hash: &'a str, instance_id: &'a str },

    #[serde(rename = "error")]
    Error { error: String },
}

// ---------------------------
// Main
// ---------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Ensure the cache directory exists
    fs::create_dir_all(PROGRAM_CACHE_DIR)?;

    // Create default server state
    let mut server_state = ServerState::default();
    // Scan the existing cache_dir and load the programs
    load_existing_programs(Path::new(PROGRAM_CACHE_DIR), &mut server_state)?;

    let state = Arc::new(Mutex::new(server_state));

    // Start listening on port 9000
    let addr = "127.0.0.1:9000";
    let listener = TcpListener::bind(addr).await?;
    println!("Symphony server listening on ws://{}", addr);

    // Accept incoming connections
    loop {
        let (stream, _) = listener.accept().await?;
        let peer_state = state.clone();
        task::spawn(async move {
            if let Err(e) = handle_connection(stream, peer_state).await {
                eprintln!("Connection error: {}", e);
            }
        });
    }
}

/// Load existing program binaries from disk into server cache
fn load_existing_programs(cache_dir: &Path, state: &mut ServerState) -> anyhow::Result<()> {
    let entries = fs::read_dir(cache_dir)?;
    for entry in entries {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            let path = entry.path();
            // Read the file contents
            let data = fs::read(&path)?;
            // Compute BLAKE3
            let mut hasher = Hasher::new();
            hasher.update(&data);
            let hash = hasher.finalize().to_hex().to_string();

            // Insert into cache map
            state.program_cache.insert(hash, path);
        }
    }
    Ok(())
}

// ---------------------------
// Connection Handler
// ---------------------------

async fn handle_connection(
    stream: TcpStream,
    state: Arc<Mutex<ServerState>>,
) -> anyhow::Result<()> {
    let ws_stream = accept_async(stream).await?;
    println!("New WebSocket connection established.");

    let (mut write, mut read) = ws_stream.split();

    // Read messages in a loop
    while let Some(msg) = read.next().await {
        let msg = msg?;
        if msg.is_binary() {
            // Attempt to decode from MessagePack
            let data = msg.into_data();
            match from_slice::<ClientMessage>(&data) {
                Ok(parsed) => {
                    let responses = handle_client_message(parsed, state.clone()).await;
                    for resp in responses {
                        let encoded = to_vec_named(&resp)?;
                        // Send as binary
                        write.send(WsMessage::Binary(encoded.into())).await?;
                    }
                }
                Err(err) => {
                    let error_msg = ServerMessage::Error {
                        error: format!("MessagePack decode error: {}", err),
                    };
                    let encoded = to_vec_named(&error_msg)?;
                    write.send(WsMessage::Binary(encoded.into())).await?;
                }
            }
        } else if msg.is_text() {
            // We do not support text frames. Send an error back.
            let error_msg = ServerMessage::Error {
                error: "Text frames not supported. Please send MessagePack binary.".to_string(),
            };
            let encoded = to_vec_named(&error_msg)?;
            write.send(WsMessage::Binary(encoded.into())).await?;
        } else if msg.is_close() {
            println!("Client closed the connection.");
            break;
        }
    }

    Ok(())
}

// ---------------------------
// Message Dispatch
// ---------------------------

async fn handle_client_message(
    msg: ClientMessage,
    state: Arc<Mutex<ServerState>>,
) -> Vec<ServerMessage<'static>> {
    match msg {
        ClientMessage::QueryExistence { hash } => {
            let exists = {
                let guard = state.lock();
                guard.program_cache.contains_key(&hash)
            };
            vec![ServerMessage::QueryResponse {
                hash: Box::leak(hash.into_boxed_str()),
                exists,
            }]
        }

        ClientMessage::UploadProgram {
            hash,
            chunk_index,
            total_chunks,
            chunk_data,
        } => {
            let mut guard = state.lock();

            if chunk_data.len() > CHUNK_SIZE_BYTES {
                return vec![ServerMessage::Error {
                    error: format!(
                        "Chunk size {} exceeds server policy of {} bytes",
                        chunk_data.len(),
                        CHUNK_SIZE_BYTES
                    ),
                }];
            }

            let entry = guard
                .in_flight_uploads
                .entry(hash.clone())
                .or_insert(InFlightUpload {
                    total_chunks,
                    collected_data: Vec::with_capacity(total_chunks * CHUNK_SIZE_BYTES),
                    received_chunks: 0,
                });

            // Validate total chunk count is consistent
            if entry.total_chunks != total_chunks {
                return vec![ServerMessage::Error {
                    error: "Mismatch in total_chunks from earlier upload messages.".to_string(),
                }];
            }

            // Check chunk ordering
            if chunk_index != entry.received_chunks {
                return vec![ServerMessage::Error {
                    error: format!(
                        "Out-of-order chunk. Expected {}, got {}",
                        entry.received_chunks, chunk_index
                    ),
                }];
            }

            // Accumulate
            entry.collected_data.extend_from_slice(&chunk_data);
            entry.received_chunks += 1;

            // Build response
            let mut replies = vec![ServerMessage::UploadAck {
                hash: Box::leak(hash.clone().into_boxed_str()),
                chunk_index,
            }];

            // Check if all chunks have arrived
            if entry.received_chunks == entry.total_chunks {
                let InFlightUpload { collected_data, .. } =
                    guard.in_flight_uploads.remove(&hash).unwrap();

                // Compute BLAKE3 to verify correctness
                let mut hasher = Hasher::new();
                hasher.update(&collected_data);
                let actual_hash = hasher.finalize().to_hex().to_string();

                if actual_hash != hash {
                    // Hash mismatch, discard
                    drop(guard);
                    return vec![ServerMessage::Error {
                        error: format!(
                            "BLAKE3 mismatch. Expected {}, computed {}. Discarding upload.",
                            hash, actual_hash
                        ),
                    }];
                }

                // If not already in the cache, persist the file
                if !guard.program_cache.contains_key(&hash) {
                    let file_path = Path::new(PROGRAM_CACHE_DIR).join(&hash);
                    if let Err(e) = fs::write(&file_path, &collected_data) {
                        drop(guard);
                        return vec![ServerMessage::Error {
                            error: format!("Failed to write program to disk: {}", e),
                        }];
                    }
                    guard.program_cache.insert(hash.clone(), file_path);
                }

                replies.push(ServerMessage::UploadComplete {
                    hash: Box::leak(hash.into_boxed_str()),
                });
            }

            replies
        }

        ClientMessage::StartProgram {
            hash,
            configuration,
        } => {
            // Acquire path from the cache
            let (path, engine) = {
                let guard = state.lock();
                let path = match guard.program_cache.get(&hash) {
                    Some(p) => p.clone(),
                    None => {
                        return vec![ServerMessage::Error {
                            error: format!("No cached program found for hash {}", hash),
                        }]
                    }
                };
                // For demonstration, we create a default engine.
                // You could parse `configuration` to set memory/cpu limits.
                let engine = Engine::default();
                (path, engine)
            };

            let data = match fs::read(&path) {
                Ok(d) => d,
                Err(e) => {
                    return vec![ServerMessage::Error {
                        error: format!("Failed to read cached program from disk: {}", e),
                    }]
                }
            };

            let module = match Module::new(&engine, &data) {
                Ok(m) => m,
                Err(e) => {
                    return vec![ServerMessage::Error {
                        error: format!("Failed to compile program: {}", e),
                    }]
                }
            };

            let store = Store::new(&engine, ());

            // Generate a unique instance_id
            let instance_id = Uuid::new_v4().to_string();

            {
                let mut guard = state.lock();
                let handle = ProgramInstanceHandle {
                    hash: hash.clone(),
                    store,
                };
                guard.running_instances.insert(instance_id.clone(), handle);
            }

            vec![ServerMessage::ProgramLaunched {
                hash: Box::leak(hash.into_boxed_str()),
                instance_id: Box::leak(instance_id.into_boxed_str()),
            }]
        }

        ClientMessage::SendEvent {
            hash,
            instance_id,
            event_data,
        } => {
            let guard = state.lock();
            let handle = match guard.running_instances.get(&instance_id) {
                Some(h) => h,
                None => {
                    return vec![ServerMessage::Error {
                        error: format!("No running instance with ID {}", instance_id),
                    }]
                }
            };

            if handle.hash != hash {
                return vec![ServerMessage::Error {
                    error: "Program hash mismatch for the given instance.".to_string(),
                }];
            }

            // In a real scenario, you’d call an exported function or
            // otherwise pass `event_data` into the Program instance.
            // Here we simply echo back an event notification.
            vec![ServerMessage::ProgramEvent {
                hash: Box::leak(hash.into_boxed_str()),
                instance_id: Box::leak(instance_id.into_boxed_str()),
                event_data,
            }]
        }

        ClientMessage::TerminateProgram { hash, instance_id } => {
            let mut guard = state.lock();
            let handle = match guard.running_instances.remove(&instance_id) {
                Some(h) => h,
                None => {
                    return vec![ServerMessage::Error {
                        error: format!("No running instance with ID {}", instance_id),
                    }]
                }
            };

            if handle.hash != hash {
                return vec![ServerMessage::Error {
                    error: "Program hash mismatch for the given instance.".to_string(),
                }];
            }

            // Drop the instance handle
            vec![ServerMessage::ProgramTerminated {
                hash: Box::leak(hash.into_boxed_str()),
                instance_id: Box::leak(instance_id.into_boxed_str()),
            }]
        }
    }
}
