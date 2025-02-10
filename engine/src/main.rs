mod spi;
mod cmd_buffer;
mod remote_obj;
mod handler;
//mod state_old;

use crate::spi::{App, InstanceMessage, InstanceState};
use anyhow::Context;
use blake3::Hasher;
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use uuid::Uuid;

use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{channel, Receiver, Sender};
use tokio::sync::Mutex; // async mutex
use tokio::task::{self, JoinHandle};
use tokio_tungstenite::accept_async;
use tungstenite::protocol::Message as WsMessage;

// Wasmtime imports
use wasmtime::component::{Component, Linker};
use wasmtime::{Config, Engine, Store};
use wasmtime_wasi::{WasiImpl, WasiView};

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

pub type ClientId = Uuid;
pub type InstanceId = Uuid;
pub type ProgramHash = String;

/// Global server state:
/// - Program cache on disk
/// - In-flight uploads
/// - Running program instances
struct ServerState {
    /// Maps a BLAKE3 hash -> Path to the binary on disk
    programs_in_disk: DashMap<ProgramHash, PathBuf>,

    // Compiled WASM components in memory
    programs_in_memory: DashMap<ProgramHash, Component>,

    /// Tracks partial uploads in progress
    programs_in_flight: DashMap<ProgramHash, Arc<Mutex<InFlightUpload>>>,

    // Client WebSocket channels (server -> client)
    clients: DashMap<ClientId, ClientHandle>,

    /// Running program instances (instance_id -> handle)
    running_instances: DashMap<InstanceId, InstanceHandle>,

    // wasmtime engine
    engine: Engine,

    // The "global" sender (instances -> server)
    inst2server: Sender<InstanceMessage>,
}

/// In-progress upload info
struct InFlightUpload {
    total_chunks: usize,
    collected_data: Vec<u8>,
    received_chunks: usize,
}

struct ClientHandle {
    server2client: Sender<ServerMessage>,
}

/// Minimal handle for a running program instance
struct InstanceHandle {
    client_id: ClientId,
    hash: String,
    server2inst: Sender<InstanceMessage>,
    join_handle: JoinHandle<()>,
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
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },

    /// Start running a cached program
    #[serde(rename = "start_program")]
    StartProgram { hash: String },

    /// Send an event (arbitrary data) to a running program instance
    #[serde(rename = "send_event")]
    SendEvent {
        instance_id: String,
        event_data: String,
    },

    /// Terminate a running program instance
    #[serde(rename = "terminate_program")]
    TerminateProgram { instance_id: String },
}

/// Messages from server -> client (in MessagePack)
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ServerMessage {
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
    ProgramTerminated { instance_id: String },

    #[serde(rename = "error")]
    Error { error: String },
}

// ---------------------------
// Main
// ---------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Ensure the cache directory exists
    fs::create_dir_all(PROGRAM_CACHE_DIR).context("Failed to create program cache directory")?;

    // Create default server state

    let mut config = Config::default();
    config.async_support(true);

    // create channel
    let (inst2server_tx, mut inst2server_rx) = channel(1024);

    let mut server_state = ServerState {
        programs_in_disk: DashMap::new(),
        programs_in_memory: DashMap::new(),
        programs_in_flight: DashMap::new(),
        clients: DashMap::new(),
        running_instances: DashMap::new(),
        engine: Engine::new(&config)?,
        inst2server: inst2server_tx,
    };

    // Scan the existing cache_dir and load the programs
    load_existing_programs(Path::new(PROGRAM_CACHE_DIR), &mut server_state)
        .context("Failed to load existing programs")?;

    let state = Arc::new(server_state);

    // Start listening on port 9000
    let addr = "127.0.0.1:9000";
    let listener = TcpListener::bind(addr).await?;
    println!("Symphony server listening on ws://{}", addr);

    //
    let state_ = state.clone();

    tokio::spawn(async move {
        // Global loop reading from the global MPSC receiver
        while let Some(instance_msg) = inst2server_rx.recv().await {
            let InstanceMessage {
                instance_id,
                dest_id,
                message,
            } = instance_msg;

            // get handle
            let instance_handle = state_.running_instances.get(&instance_id).unwrap();
            let client_id = instance_handle.client_id;

            // dest_id = 0: to symphony server.
            // dest_id = 1: to client.
            // dest_id = 2: to LLM server.
            // dest_id > 4: to other instances.

            // Construct a ProgramEvent message for the client
            if dest_id == 1 {
                // (Just parse or wrap the `message` into JSON)

                let server_msg = ServerMessage::ProgramEvent {
                    instance_id: instance_id.to_string(),
                    event_data: message,
                };

                // get client handle
                let client_handle = state_.clients.get(&client_id).unwrap();
                client_handle.server2client.send(server_msg).await.unwrap();
            } else {
                // Currently do nothing for other channels,
            }
        }

        // This is the end of the global loop
    });

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
            state.programs_in_disk.insert(hash, path);
        }
    }
    Ok(())
}

// ---------------------------
// Connection Handler
// ---------------------------

async fn handle_connection(stream: TcpStream, state: Arc<ServerState>) -> anyhow::Result<()> {
    let ws_stream = accept_async(stream).await?;

    // print the detailed information of the connection
    println!("New connection: {}", ws_stream.get_ref().peer_addr()?);

    let (mut write, mut read) = ws_stream.split();

    // create external channel
    let (server2client_tx, mut server2client_rx) = channel(1024);

    let client_id = Uuid::new_v4();
    let client_handle = ClientHandle {
        server2client: server2client_tx.clone(),
    };

    // insert the client handle into the global state
    state.clients.insert(client_id, client_handle);

    // server -> client
    let writer_task = {
        // The 'write' half is not cloneable, so we move it into the task.
        tokio::spawn(async move {
            while let Some(msg) = server2client_rx.recv().await {
                match to_vec_named(&msg) {
                    Ok(encoded) => {
                        // Send the encoded message
                        if let Err(e) = write.send(WsMessage::Binary(encoded.into())).await {
                            eprintln!("Failed to write to ws: {}", e);
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to encode server_msg: {}", e);
                        break;
                    }
                }
            }
        })
    };

    // reader task
    // Read messages in a loop
    while let Some(msg) = read.next().await {
        let msg = msg?;
        if msg.is_binary() {
            // Attempt to decode from MessagePack
            let data = msg.into_data();
            match from_slice::<ClientMessage>(&data) {
                Ok(parsed) => {
                    let responses = handle_client_message(state.clone(), client_id, parsed).await;
                    for resp in responses {
                        server2client_tx.send(resp).await?;
                    }
                }
                Err(err) => {
                    let error_msg = ServerMessage::Error {
                        error: format!("MessagePack decode error: {}", err),
                    };
                    server2client_tx.send(error_msg).await?;
                }
            }
        } else if msg.is_text() {
            // We do not support text frames. Send an error back.
            let error_msg = ServerMessage::Error {
                error: "Text frames not supported. Please send MessagePack binary.".to_string(),
            };
            server2client_tx.send(error_msg).await?;
        } else if msg.is_close() {
            println!("Client closed the connection.");
            break;
        }
    }
    writer_task.abort();
    Ok(())
}

// ---------------------------
// Message Dispatch
// ---------------------------

async fn handle_client_message(
    state: Arc<ServerState>,
    client_id: ClientId,
    msg: ClientMessage,
) -> Vec<ServerMessage> {
    match msg {
        ClientMessage::QueryExistence { hash } => {
            handle_client_message_query_existence(state, hash).await
        }

        ClientMessage::UploadProgram {
            hash,
            chunk_index,
            total_chunks,
            chunk_data,
        } => {
            handle_client_message_upload_program(state, hash, chunk_index, total_chunks, chunk_data)
                .await
        }

        ClientMessage::StartProgram { hash } => {
            handle_client_message_start_program(state, hash, client_id).await
        }

        ClientMessage::SendEvent {
            instance_id,
            event_data,
        } => handle_client_message_send_message(state, instance_id, event_data).await,

        ClientMessage::TerminateProgram { instance_id } => {
            handle_client_message_terminate_program(state, instance_id).await
        }
    }
}

async fn handle_client_message_query_existence(
    state: Arc<ServerState>,
    hash: String,
) -> Vec<ServerMessage> {
    let exists = { state.programs_in_disk.contains_key(&hash) };
    vec![ServerMessage::QueryResponse { hash, exists }]
}

async fn handle_client_message_upload_program(
    state: Arc<ServerState>,
    hash: String,
    chunk_index: usize,
    total_chunks: usize,
    chunk_data: Vec<u8>,
) -> Vec<ServerMessage> {
    // 1) Basic validation before locking
    if chunk_data.len() > CHUNK_SIZE_BYTES {
        return vec![ServerMessage::Error {
            error: format!(
                "Chunk size {} exceeds server policy of {} bytes",
                chunk_data.len(),
                CHUNK_SIZE_BYTES
            ),
        }];
    }

    // 2) Obtain or initialize the per-hash upload entry
    //    (we don't need a global lock, just dashmap + a per-hash mutex)
    let upload_entry = state
        .programs_in_flight
        .entry(hash.clone())
        .or_insert_with(|| {
            // Allocate a new InFlightUpload if not present
            Arc::new(Mutex::new(InFlightUpload {
                total_chunks,
                collected_data: Vec::with_capacity(total_chunks * CHUNK_SIZE_BYTES),
                received_chunks: 0,
            }))
        })
        .clone(); // Arc<AsyncMutex<...>>

    // 3) Lock just this one hash's upload data
    let mut upload_data = upload_entry.lock().await;

    // Validate total_chunks consistency across calls
    if upload_data.total_chunks != total_chunks {
        return vec![ServerMessage::Error {
            error: format!(
                "Mismatch in total_chunks: previously {}, now {}",
                upload_data.total_chunks, total_chunks
            ),
        }];
    }

    // 4) Check chunk ordering
    if chunk_index != upload_data.received_chunks {
        return vec![ServerMessage::Error {
            error: format!(
                "Out-of-order chunk. Expected {}, got {}",
                upload_data.received_chunks, chunk_index
            ),
        }];
    }

    // 5) Accumulate
    upload_data.collected_data.extend_from_slice(&chunk_data);
    upload_data.received_chunks += 1;

    // We'll build our responses in a single vector
    let mut replies = vec![ServerMessage::UploadAck {
        hash: hash.clone(),
        chunk_index,
    }];

    // 6) Check if this was the last chunk
    if upload_data.received_chunks == upload_data.total_chunks {
        // Extract the collected data now that it's complete
        let final_data = std::mem::take(&mut upload_data.collected_data);
        let total_chunks_stored = upload_data.total_chunks;

        // We must drop the upload_data lock before removing from DashMap
        drop(upload_data);

        // Remove the entry so future uploads for this hash can start fresh
        state.programs_in_flight.remove(&hash);

        // Verify BLAKE3 hash
        let mut hasher = Hasher::new();
        hasher.update(&final_data);
        let actual_hash = hasher.finalize().to_hex().to_string();

        if actual_hash != hash {
            return vec![ServerMessage::Error {
                error: format!(
                    "BLAKE3 mismatch. Expected {}, computed {} (total chunks={})",
                    hash, actual_hash, total_chunks_stored
                ),
            }];
        }

        // If not already in the server's disk map, persist it
        if state.programs_in_disk.get(&hash).is_none() {
            let file_path = Path::new(PROGRAM_CACHE_DIR).join(&hash);
            if let Err(e) = fs::write(&file_path, &final_data) {
                return vec![ServerMessage::Error {
                    error: format!("Failed to write program to disk: {}", e),
                }];
            }

            // Record in dashmap
            state.programs_in_disk.insert(hash.clone(), file_path);
        }

        // Announce completion
        replies.push(ServerMessage::UploadComplete { hash: hash.clone() });
    }

    // 7) Return the accumulated replies
    replies
}

async fn handle_client_message_start_program(
    state: Arc<ServerState>,
    hash: String,
    client_id: ClientId,
) -> Vec<ServerMessage> {
    // Load WASM component from disk if not already in memory
    if state.programs_in_memory.get(&hash).is_none() {
        if let Some(path) = state.programs_in_disk.get(&hash) {
            println!("Loading component from path: {hash}");
            let component = Component::from_file(&state.engine, path.value())
                .with_context(|| format!("Failed to compile program: {hash}"));

            // Handle the error explicitly and return ServerMessage::Error
            match component {
                Ok(comp) => {
                    // Add the component to the in-memory cache
                    state.programs_in_memory.insert(hash.clone(), comp);
                }
                Err(e) => {
                    return vec![ServerMessage::Error {
                        error: format!("Failed to read program from disk: {}", e),
                    }];
                }
            }
        } else {
            return vec![ServerMessage::Error {
                error: format!("No program found for hash {}", hash),
            }];
        }
    }

    // get the component from in-memory
    let component = state.programs_in_memory.get(&hash).unwrap().clone();
    let instance_id = Uuid::new_v4();

    // create a channel
    let (server2inst_tx, server2inst_rx) = channel(32);

    let inst_state = InstanceState::new(instance_id, state.inst2server.clone(), server2inst_rx);

    // linker and store

    let engine_clone = state.engine.clone();
    // 2) Spawn a background task to do the heavy lifting
    let join_handle = tokio::spawn({
        // We clone references so the closure can move them in

        async move {
            // Lock the store for instantiation
            let mut store = Store::new(&engine_clone, inst_state);

            let mut linker: Linker<InstanceState> = Linker::new(&engine_clone);

            if let Err(e) = App::add_to_linker(&mut linker, |s| s) {
                eprintln!("Error adding App to linker: {}", e);
                return; // or handle more gracefully
            }

            // Maybe we can do more fine-grained linking by submodules in the future
            if let Err(e) = wasmtime_wasi::add_to_linker_async(&mut linker) {
                eprintln!("Failed to link WASI bindings: {}", e);
                return;
            }

            // Instantiate
            let instance = match linker.instantiate_async(&mut store, &component).await {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("Failed to instantiate: {}", e);
                    return;
                }
            };

            let run_interface = match instance.get_export(&mut store, None, "spi:app/run") {
                Some(r) => r,
                None => {
                    eprintln!("No spi:app/run found");
                    return;
                }
            };
            let run_func_export = match instance.get_export(&mut store, Some(&run_interface), "run")
            {
                Some(r) => r,
                None => {
                    eprintln!("No run export found");
                    return;
                }
            };
            let run_func = match instance
                .get_typed_func::<(), (Result<(), ()>,)>(&mut store, &run_func_export)
            {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("Failed to get run function: {}", e);
                    return;
                }
            };

            println!("Running instance_id={}", instance_id);

            // Actually run
            match run_func.call_async(&mut store, ()).await {
                Ok((Ok(()),)) => {
                    println!("Finished normally for instance_id={}", instance_id);
                }
                Ok((Err(()),)) => {
                    eprintln!("Returned an error for instance_id={}", instance_id);
                }
                Err(call_err) => {
                    eprintln!("Call error: {}", call_err);
                }
            }

            // If we get here, the WASM has finished or errored out
            // -- do any necessary cleanup / signals to the outside.
        }
    });

    // 3) Insert an entry into running_instances so we can reference this instance_id
    // later in SendEvent or TerminateProgram, etc.
    // Create a new instance handle
    let handle = InstanceHandle {
        client_id,
        hash: hash.clone(),
        server2inst: server2inst_tx,
        join_handle,
    };

    state.running_instances.insert(instance_id, handle);

    // 4) Return ProgramLaunched *immediately*, without blocking
    vec![ServerMessage::ProgramLaunched {
        hash,
        instance_id: instance_id.to_string(),
    }]
}

async fn handle_client_message_send_message(
    state: Arc<ServerState>,
    instance_id: String,
    event_data: String,
) -> Vec<ServerMessage> {
    let instance_id = Uuid::parse_str(&instance_id).expect("Invalid UUID format");

    let instance_handle = match state.running_instances.get(&instance_id) {
        Some(e) => e,
        None => {
            return vec![ServerMessage::Error {
                error: format!("No running instance with ID {}", instance_id),
            }]
        }
    };

    if let Err(e) = instance_handle
        .server2inst
        .send(InstanceMessage {
            instance_id,
            dest_id: 0,
            message: event_data,
        })
        .await
    {
        return vec![ServerMessage::Error {
            error: format!("Failed to send event to instance: {}", e),
        }];
    }

    vec![]
}

async fn handle_client_message_terminate_program(
    state: Arc<ServerState>,
    instance_id: String,
) -> Vec<ServerMessage> {
    let instance_id = Uuid::parse_str(&instance_id).expect("Invalid UUID format");

    let instance_handle = match state.running_instances.get(&instance_id) {
        Some(e) => e,
        None => {
            return vec![ServerMessage::Error {
                error: format!("No running instance with ID {}", instance_id),
            }]
        }
    };

    // abort
    instance_handle.join_handle.abort();

    // remove the instance from the running_instances
    state.running_instances.remove(&instance_id);

    // Drop the instance handle
    vec![ServerMessage::ProgramTerminated {
        instance_id: instance_id.to_string(),
    }]
}
