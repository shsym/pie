use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;
use wasmtime::{component::Component, component::Linker, Engine, Store};
use wasmtime_wasi;

use crate::instance::{App, Command, Id as InstanceId, InstanceState};
use crate::server::ServerMessage;
use crate::tokenizer;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RuntimeError {
    /// Wrap general I/O errors
    #[error("I/O error occurred: {0}")]
    Io(#[from] std::io::Error),

    /// Wrap Wasmtime errors
    #[error("Wasmtime error occurred: {0}")]
    Wasmtime(#[from] wasmtime::Error),

    /// No program found for a given hash
    #[error("No such program with hash={0}")]
    MissingProgram(String),

    /// Failed to compile a WASM component from disk
    #[error("Failed to compile program at path {path:?}: {source}")]
    CompileWasm {
        path: std::path::PathBuf,
        #[source]
        source: wasmtime::Error,
    },

    /// Fallback for unexpected cases
    #[error("Runtime error: {0}")]
    Other(String),
}

/// Holds the “global” or “runtime” data that the controller needs to manage
/// instances, compiled programs, etc.
pub struct Runtime {
    /// The Wasmtime engine (global)
    pub engine: Engine,

    /// Pre-compiled WASM components, keyed by BLAKE3 hex string
    pub programs_in_memory: DashMap<String, Component>,

    /// Paths to compiled modules on disk
    pub programs_in_disk: DashMap<String, std::path::PathBuf>,

    /// Running instances
    pub running_instances: DashMap<InstanceId, InstanceHandle>,

    /// The channel for (Instance -> controller) messages
    pub inst2server: Sender<(InstanceId, Command)>,

    /// For demonstration: a tokenizer or other utility
    pub tokenizer: Arc<tokenizer::BytePairEncoder>,
}

pub struct InstanceHandle {
    pub hash: String,
    pub to_origin: Sender<ServerMessage>,
    pub evt_from_origin: Sender<String>,
    pub evt_from_peers: Sender<(String, String)>,
    pub join_handle: tokio::task::JoinHandle<()>,
}

impl Runtime {
    /// Create a new `Runtime`
    pub fn new(engine: Engine, inst2server: Sender<(InstanceId, Command)>) -> Self {
        // Here you might also load your tokenizer file or do other initialization.
        let tokenizer = Arc::new(
            tokenizer::llama3_tokenizer(super::TOKENIZER_MODEL).expect("Tokenizer load failed"),
        );

        Self {
            engine,
            programs_in_memory: DashMap::new(),
            programs_in_disk: DashMap::new(),
            running_instances: DashMap::new(),
            inst2server,
            tokenizer,
        }
    }

    /// If desired, you can add your “load programs from disk” logic here.
    pub fn load_existing_programs(&self, cache_dir: &std::path::Path) -> Result<(), RuntimeError> {
        let entries = std::fs::read_dir(cache_dir)?; // Will map to RuntimeError::Io automatically
        for entry in entries {
            let entry = entry?; // same here, auto-converted to RuntimeError::Io
            if entry.file_type()?.is_file() {
                let path = entry.path();
                let data = std::fs::read(&path)?; // also auto Io
                let hash = blake3::hash(&data).to_hex().to_string();
                self.programs_in_disk.insert(hash, path);
            }
        }
        Ok(())
    }

    /// Actually start a program instance
    pub async fn start_program(
        &self,
        hash: &str,
        to_origin: Sender<ServerMessage>,
    ) -> Result<InstanceId, RuntimeError> {
        // 1) Make sure the `Component` is loaded in memory
        if self.programs_in_memory.get(hash).is_none() {
            // load from disk if possible
            if let Some(path_entry) = self.programs_in_disk.get(hash) {
                // Use a custom error variant for compile errors
                let component =
                    Component::from_file(&self.engine, path_entry.value()).map_err(|err| {
                        RuntimeError::CompileWasm {
                            path: path_entry.value().to_path_buf(),
                            source: err,
                        }
                    })?;
                self.programs_in_memory.insert(hash.to_string(), component);
            } else {
                // If not on disk either, return a custom error
                return Err(RuntimeError::MissingProgram(hash.to_string()));
            }
        }

        // 2) Now we have a compiled component
        let component = match self.programs_in_memory.get(hash) {
            Some(c) => c.clone(),
            None => {
                return Err(RuntimeError::Other(
                    "Failed to get component from memory".into(),
                ))
            }
        };

        let instance_id = Uuid::new_v4();

        // 3) Create channels for event-sending
        let (server2inst_tx, server2inst_rx) = tokio::sync::mpsc::channel::<String>(32);
        let (peer_tx, peer_rx) = tokio::sync::mpsc::channel::<(String, String)>(32);

        // 4) Build the InstanceState
        let inst_state = InstanceState::new(
            instance_id,
            self.inst2server.clone(),
            server2inst_rx,
            peer_rx,
            crate::instance::InstanceUtils {
                tokenizer: self.tokenizer.clone(),
            },
        );

        // 5) Instantiate and run in a task
        let engine_clone = self.engine.clone();
        let join_handle = tokio::spawn(async move {
            // Wrap everything in a closure returning a Result,
            // so we can capture errors more systematically if desired:
            let result = async {
                let mut store = Store::new(&engine_clone, inst_state);
                let mut linker = Linker::new(&engine_clone);

                // Add to linker
                App::add_to_linker(&mut linker, |s| s)
                    .map_err(|e| RuntimeError::Other(format!("Error adding to linker: {e}")))?;

                wasmtime_wasi::add_to_linker_async(&mut linker)
                    .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))?;

                let instance = linker
                    .instantiate_async(&mut store, &component)
                    .await
                    .map_err(|e| RuntimeError::Other(format!("Instantiation error: {e}")))?;

                // Attempt to call “run”
                let run_export = instance.get_export(&mut store, None, "spi:app/run");
                let run_iface = run_export
                    .ok_or_else(|| RuntimeError::Other("No spi:app/run in the module".into()))?;

                let run_func_export = instance.get_export(&mut store, Some(&run_iface), "run");
                let run_func_export = run_func_export
                    .ok_or_else(|| RuntimeError::Other("No 'run' function found".into()))?;

                let run_func = instance
                    .get_typed_func::<(), (Result<(), ()>,)>(&mut store, &run_func_export)
                    .map_err(|e| {
                        RuntimeError::Other(format!("Failed to get 'run' function: {e}"))
                    })?;

                match run_func.call_async(&mut store, ()).await {
                    Ok((Ok(()),)) => {
                        println!("Instance {instance_id} finished normally");
                    }
                    Ok((Err(()),)) => {
                        eprintln!("Instance {instance_id} returned an error");
                    }
                    Err(call_err) => {
                        eprintln!("Instance {instance_id} call error: {call_err}");
                    }
                }
                Ok::<(), RuntimeError>(())
            }
            .await;

            if let Err(err) = result {
                eprintln!("Instance {instance_id} error: {err}");
            }
        });

        // 6) Record in the “running_instances” so we can manage it later
        let instance_handle = InstanceHandle {
            hash: hash.to_string(),
            to_origin,
            evt_from_origin: server2inst_tx,
            evt_from_peers: peer_tx,
            join_handle,
        };
        self.running_instances.insert(instance_id, instance_handle);

        Ok(instance_id)
    }

    /// Terminate (abort) a running instance
    pub fn terminate_program(&self, instance_id: InstanceId) -> bool {
        if let Some((_, handle)) = self.running_instances.remove(&instance_id) {
            handle.join_handle.abort();
            true
        } else {
            false
        }
    }
}
