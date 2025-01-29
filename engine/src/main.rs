mod spi;

use anyhow::Context;
use blake3::Hasher;
use futures::{SinkExt, StreamExt};
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};
use uuid::Uuid;

use crate::spi::{App, InstanceMessage, InstanceState};

use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task::{self, JoinHandle};
use tokio_tungstenite::accept_async;
use tungstenite::protocol::Message as WsMessage;

// Wasmtime imports
use wasmtime::component::{Component, Instance, Linker};
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

/// Global server state:
/// - Program cache on disk
/// - In-flight uploads
/// - Running program instances
struct ServerState {
    /// Maps a BLAKE3 hash -> Path to the binary on disk
    programs_in_disk: HashMap<String, PathBuf>,

    // Compiled WASM components in memory
    programs_in_memory: HashMap<String, Component>,

    /// Tracks partial uploads in progress
    programs_in_flight: HashMap<String, InFlightUpload>,

    // wasmtime engine
    engine: Engine,

    /// Running program instances (instance_id -> handle)
    running_instances: HashMap<Uuid, InstanceHandle>,

    // "The" receiver for all instance messages.
    // Sender per instance is stored in the handle.
    receiver: Receiver<InstanceMessage>,
    sender: Sender<InstanceMessage>,
}

/// In-progress upload info
struct InFlightUpload {
    total_chunks: usize,
    collected_data: Vec<u8>,
    received_chunks: usize,
}

/// Minimal handle for a running program instance
struct InstanceHandle {
    hash: String,
    sender: Sender<InstanceMessage>,
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
    fs::create_dir_all(PROGRAM_CACHE_DIR).context("Failed to create program cache directory")?;

    // Create default server state

    let mut config = Config::default();
    config.async_support(true);

    // create channel
    let (sender, receiver) = tokio::sync::mpsc::channel(1024);

    let mut server_state = ServerState {
        programs_in_disk: HashMap::new(),
        programs_in_memory: HashMap::new(),
        programs_in_flight: HashMap::new(),
        engine: Engine::new(&config)?,
        running_instances: HashMap::new(),
        receiver,
        sender,
    };

    // Scan the existing cache_dir and load the programs
    load_existing_programs(Path::new(PROGRAM_CACHE_DIR), &mut server_state)
        .context("Failed to load existing programs")?;

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
            state.programs_in_disk.insert(hash, path);
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

    // print the detailed information of the connection
    println!("New connection: {}", ws_stream.get_ref().peer_addr()?);

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
                guard.programs_in_disk.contains_key(&hash)
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
                .programs_in_flight
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
                    guard.programs_in_flight.remove(&hash).unwrap();

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
                if !guard.programs_in_disk.contains_key(&hash) {
                    let file_path = Path::new(PROGRAM_CACHE_DIR).join(&hash);
                    if let Err(e) = fs::write(&file_path, &collected_data) {
                        drop(guard);
                        return vec![ServerMessage::Error {
                            error: format!("Failed to write program to disk: {}", e),
                        }];
                    }
                    guard.programs_in_disk.insert(hash.clone(), file_path);
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
            // Load WASM component from disk if not already in memory
            let mut guard = state.lock();
            if guard.programs_in_memory.get(&hash).is_none() {
                if let Some(path) = guard.programs_in_disk.get(&hash) {
                    println!("Loading component from path: {hash}");
                    let component = Component::from_file(&guard.engine, path)
                        .with_context(|| format!("Failed to compile program: {hash}"));

                    // Handle the error explicitly and return ServerMessage::Error
                    match component {
                        Ok(comp) => {
                            // Add the component to the in-memory cache
                            guard.programs_in_memory.insert(hash.clone(), comp);
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
            let component = guard.programs_in_memory.get(&hash).unwrap().clone();
            let instance_id = Uuid::new_v4();

            // create a channel
            let (sender, receiver) = tokio::sync::mpsc::channel(32);

            let state = InstanceState::new(instance_id, guard.sender.clone(), receiver);

            // linker and store

            let engine_clone = guard.engine.clone();
            // 2) Spawn a background task to do the heavy lifting
            let join_handle = tokio::spawn({
                // We clone references so the closure can move them in

                async move {
                    // Lock the store for instantiation
                    let mut store = Store::new(&engine_clone, state);

                    let mut linker: Linker<InstanceState> = Linker::new(&engine_clone);

                    App::add_to_linker(&mut linker, |s| s).expect("Failed to add App to linker");
                    link_wasi_bindings(&mut linker)
                        .context("Failed to link WASI bindings")
                        .expect("Failed to link WASI bindings");

                    // Instantiate
                    let instance = linker
                        .instantiate_async(&mut store, &component)
                        .await
                        .expect("Failed to instantiate");

                    // Optionally store the Instance somewhere (Arc<Instance>, etc.)
                    // Then run the "run" entry point
                    let run_interface = match instance.get_export(&mut store, None, "spi:app/run") {
                        Some(r) => r,
                        None => {
                            eprintln!("No spi:app/run found");
                            return;
                        }
                    };
                    let run_func_export =
                        match instance.get_export(&mut store, Some(&run_interface), "run") {
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

                    println!("entering wasm run for instance_id={}", instance_id);

                    // Actually run
                    match run_func.call_async(&mut store, ()).await {
                        Ok((Ok(()),)) => {
                            println!("WASM finished normally for instance_id={}", instance_id);
                        }
                        Ok((Err(()),)) => {
                            eprintln!("WASM run returned an error for instance_id={}", instance_id);
                        }
                        Err(call_err) => {
                            eprintln!("WASM call error: {}", call_err);
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
                hash: hash.clone(),
                sender,
                join_handle,
            };

            guard.running_instances.insert(instance_id, handle);

            // 4) Return ProgramLaunched *immediately*, without blocking
            vec![ServerMessage::ProgramLaunched {
                hash: Box::leak(hash.into_boxed_str()),
                instance_id: Box::leak(instance_id.to_string().into_boxed_str()),
            }]
            ////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////
            // The problematic code block ends here
            ////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////////////////////////////////////////////////////////////
        }

        ClientMessage::SendEvent {
            hash,
            instance_id,
            event_data,
        } => {
            // let guard = state.lock();
            // let handle = match guard.running_instances.get(&instance_id) {
            //     Some(h) => h,
            //     None => {
            //         return vec![ServerMessage::Error {
            //             error: format!("No running instance with ID {}", instance_id),
            //         }]
            //     }
            // };
            //
            // if handle.hash != hash {
            //     return vec![ServerMessage::Error {
            //         error: "Program hash mismatch for the given instance.".to_string(),
            //     }];
            // }
            //
            // // In a real scenario, you’d call an exported function or
            // // otherwise pass `event_data` into the Program instance.
            // // Here we simply echo back an event notification.
            // vec![ServerMessage::ProgramEvent {
            //     hash: Box::leak(hash.into_boxed_str()),
            //     instance_id: Box::leak(instance_id.into_boxed_str()),
            //     event_data,
            // }]
            vec![]
        }

        ClientMessage::TerminateProgram { hash, instance_id } => {
            // let mut guard = state.lock();
            // let handle = match guard.running_instances.remove(&instance_id) {
            //     Some(h) => h,
            //     None => {
            //         return vec![ServerMessage::Error {
            //             error: format!("No running instance with ID {}", instance_id),
            //         }]
            //     }
            // };
            //
            // if handle.hash != hash {
            //     return vec![ServerMessage::Error {
            //         error: "Program hash mismatch for the given instance.".to_string(),
            //     }];
            // }

            // // Drop the instance handle
            // vec![ServerMessage::ProgramTerminated {
            //     hash: Box::leak(hash.into_boxed_str()),
            //     instance_id: Box::leak(instance_id.into_boxed_str()),
            // }]
            vec![ServerMessage::ProgramTerminated {
                hash: Box::leak(hash.into_boxed_str()),
                instance_id: "test_dummy",
            }]
        }
    }
}

/// Copied from [wasmtime_wasi::type_annotate]
pub fn type_annotate<T: WasiView, F>(val: F) -> F
where
    F: Fn(&mut T) -> WasiImpl<&mut T>,
{
    val
}
pub fn link_wasi_bindings<T: WasiView>(l: &mut Linker<T>) -> Result<(), wasmtime::Error> {
    let closure = type_annotate::<T, _>(|t| WasiImpl(t));
    let options = wasmtime_wasi::bindings::sync::LinkOptions::default();
    wasmtime_wasi::bindings::sync::filesystem::types::add_to_linker_get_host(l, closure)?;
    wasmtime_wasi::bindings::filesystem::preopens::add_to_linker_get_host(l, closure)?;
    wasmtime_wasi::bindings::io::error::add_to_linker_get_host(l, closure)?;
    wasmtime_wasi::bindings::sync::io::streams::add_to_linker_get_host(l, closure)?;
    wasmtime_wasi::bindings::cli::exit::add_to_linker_get_host(l, &options.into(), closure)?;
    wasmtime_wasi::bindings::cli::environment::add_to_linker_get_host(l, closure)?;
    wasmtime_wasi::bindings::cli::stdin::add_to_linker_get_host(l, closure)?;
    wasmtime_wasi::bindings::cli::stdout::add_to_linker_get_host(l, closure)?;
    wasmtime_wasi::bindings::cli::stderr::add_to_linker_get_host(l, closure)?;

    Ok(())
}
