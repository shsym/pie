use dashmap::DashMap;
use std::sync::{Arc, OnceLock};
use uuid::Uuid;
use wasmtime::{Config, Engine, Store, component::Component, component::Linker};
use wasmtime_wasi;

use crate::instance::{Id as InstanceId, InstanceState};
use crate::{bindings, server, service};

use crate::service::{Service, ServiceError};
use thiserror::Error;
use tokio::sync::oneshot;
const VERSION: &str = env!("CARGO_PKG_VERSION");

static SERVICE_ID_RUNTIME: OnceLock<usize> = OnceLock::new();

pub fn trap<T>(instance_id: InstanceId, message: T)
where
    T: ToString,
{
    Command::Trap {
        instance_id,
        message: message.to_string(),
    }
    .dispatch()
    .unwrap();
}

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

#[derive(Debug)]
pub enum Command {
    GetVersion {
        event: oneshot::Sender<String>,
    },

    ProgramExists {
        hash: String,
        event: oneshot::Sender<bool>,
    },

    UploadProgram {
        hash: String,
        raw: Vec<u8>,
        event: oneshot::Sender<Result<String, RuntimeError>>,
    },

    LaunchInstance {
        program_hash: String,
        event: oneshot::Sender<Result<InstanceId, RuntimeError>>,
    },

    Trap {
        instance_id: InstanceId,
        message: String,
    },

    Warn {
        instance_id: InstanceId,
        message: String,
    },
}

impl Command {
    pub fn dispatch(self) -> Result<(), ServiceError> {
        let service_id =
            *SERVICE_ID_RUNTIME.get_or_init(move || service::get_service_id("runtime").unwrap());

        service::dispatch(service_id, self)
    }
}

/// Holds the “global” or “runtime” data that the controller needs to manage
/// instances, compiled programs, etc.
pub struct Runtime {
    /// The Wasmtime engine (global)
    engine: Engine,
    linker: Arc<Linker<InstanceState>>,

    /// Pre-compiled WASM components, keyed by BLAKE3 hex string
    programs_in_memory: DashMap<String, Component>,

    /// Paths to compiled modules on disk
    programs_in_disk: DashMap<String, std::path::PathBuf>,

    /// Running instances
    running_instances: DashMap<InstanceId, InstanceHandle>,
}

pub struct InstanceHandle {
    pub hash: String,
    //pub to_origin: Sender<ServerMessage>,
    // pub evt_from_system: Sender<String>,
    // pub evt_from_origin: Sender<String>,
    // pub evt_from_peers: Sender<(String, String)>,
    pub join_handle: tokio::task::JoinHandle<()>,
}
//#[async_trait]
impl Service for Runtime {
    type Command = Command;

    async fn handle(&mut self, cmd: Self::Command) {
        match cmd {
            Command::ProgramExists { hash, event } => {
                let exists = self.programs_in_memory.contains_key(&hash)
                    || self.programs_in_disk.contains_key(&hash);
                event.send(exists).unwrap();
            }

            Command::UploadProgram { hash, raw, event } => {
                if self.programs_in_memory.contains_key(&hash) {
                    event.send(Ok(hash)).unwrap();
                } else {
                    if let Ok(component) = Component::from_binary(&self.engine, raw.as_slice()) {
                        self.programs_in_memory.insert(hash.to_string(), component);

                        // Write to disk
                        let file_path = std::path::Path::new(super::PROGRAM_CACHE_DIR).join(&hash);
                        std::fs::write(&file_path, &raw).unwrap();
                        self.programs_in_disk.insert(hash.clone(), file_path);
                        event.send(Ok(hash)).unwrap();
                    } else {
                        event
                            .send(Err(RuntimeError::Other("Failed to compile".into())))
                            .unwrap();
                    }
                }
            }

            Command::LaunchInstance {
                program_hash: hash,
                event,
            } => {
                let instance_id = self.start_program(&hash).await.unwrap();
                event.send(Ok(instance_id)).unwrap();
            }

            Command::Trap {
                instance_id,
                message,
            } => {
                self.terminate_program(instance_id, message).await;
            }

            Command::Warn {
                instance_id,
                message,
            } => server::Command::Send {
                inst_id: instance_id.clone(),
                message: message.clone(),
            }
            .dispatch()
            .unwrap(),
            Command::GetVersion { event } => {
                event.send(VERSION.to_string()).unwrap();
            }
        }
    }
}

impl Runtime {
    pub fn new() -> Self {
        // Configure Wasmtime engine
        let mut config = Config::default();
        config.async_support(true);
        let engine = Engine::new(&config).unwrap();

        let mut linker = Linker::<InstanceState>::new(&engine);

        // Add to linker
        wasmtime_wasi::add_to_linker_async(&mut linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))
            .unwrap();
        wasmtime_wasi_http::add_only_http_to_linker_async(&mut linker)
            .map_err(|e| RuntimeError::Other(format!("Failed to link WASI: {e}")))
            .unwrap();

        bindings::add_to_linker(&mut linker).unwrap();

        Self {
            engine,
            linker: Arc::new(linker),
            programs_in_memory: DashMap::new(),
            programs_in_disk: DashMap::new(),
            running_instances: DashMap::new(),
        }
    }

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
    pub async fn start_program(&self, hash: &str) -> Result<InstanceId, RuntimeError> {
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
                ));
            }
        };

        let instance_id = Uuid::new_v4();

        // 4) Build the InstanceState

        // 5) Instantiate and run in a task
        let engine = self.engine.clone();
        let linker = self.linker.clone();

        let join_handle = tokio::spawn(Self::launch(instance_id, component, engine, linker));

        // 6) Record in the “running_instances” so we can manage it later
        let instance_handle = InstanceHandle {
            hash: hash.to_string(),
            join_handle,
        };
        self.running_instances.insert(instance_id, instance_handle);

        Ok(instance_id)
    }

    /// Terminate (abort) a running instance
    pub async fn terminate_program(&self, instance_id: InstanceId, reason: String) {
        if let Some((_, handle)) = self.running_instances.remove(&instance_id) {
            handle.join_handle.abort();
            server::Command::DetachInstance {
                inst_id: instance_id.clone(),
                reason,
            }
            .dispatch()
            .ok();

            // TODO: cleanup other resources (l4m, etc.)
        }
    }

    async fn launch(
        instance_id: InstanceId,
        component: Component,
        engine: Engine,
        linker: Arc<Linker<InstanceState>>,
    ) {
        let inst_state = InstanceState::new(instance_id).await;

        // Wrap everything in a closure returning a Result,
        // so we can capture errors more systematically if desired:
        let result = async {
            let mut store = Store::new(&engine, inst_state);

            let instance = linker
                .instantiate_async(&mut store, &component)
                .await
                .map_err(|e| RuntimeError::Other(format!("Instantiation error: {e}")))?;

            // Attempt to call “run”
            let run_export = instance.get_export(&mut store, None, "symphony:app/run");
            let run_iface = run_export
                .ok_or_else(|| RuntimeError::Other("No symphony:app/run in the module".into()))?;

            let run_func_export = instance.get_export(&mut store, Some(&run_iface), "run");
            let run_func_export = run_func_export
                .ok_or_else(|| RuntimeError::Other("No 'run' function found".into()))?;

            let run_func = instance
                .get_typed_func::<(), (Result<(), String>,)>(&mut store, &run_func_export)
                .map_err(|e| RuntimeError::Other(format!("Failed to get 'run' function: {e}")))?;

            match run_func.call_async(&mut store, ()).await {
                Ok((Ok(()),)) => {
                    //println!("Instance {instance_id} finished normally");
                    Ok(())
                }
                Ok((Err(runtime_err),)) => {
                    //eprintln!("Instance {instance_id} returned an error");
                    Err(RuntimeError::Other(runtime_err))
                }
                Err(call_err) => {
                    //eprintln!("Instance {instance_id} call error: {call_err}");
                    Err(RuntimeError::Other(format!("Call error: {call_err}")))
                }
            }
        }
        .await;

        if let Err(err) = result {
            println!("Instance {instance_id} failed: {err}");
            server::Command::DetachInstance {
                inst_id: instance_id.clone(),
                reason: format!("{err}"),
            }
            .dispatch()
            .ok();
        } else {
            server::Command::DetachInstance {
                inst_id: instance_id.clone(),
                reason: format!("instance normally finished"),
            }
            .dispatch()
            .ok();
        }
    }
}
