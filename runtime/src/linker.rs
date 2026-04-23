//! Linker service.
//!
//! Singleton service that owns the pre-configured wasmtime Engine.
//! Creates per-instance linkers with WASI, WASI HTTP, Pie API host bindings,
//! and dynamically linked library dependencies.

mod dynamic_linking;
mod output;
mod state;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use anyhow::{Result, anyhow};
use tokio::sync::oneshot;
use wasmtime::component::{Component, Instance as WasmInstance, Linker as WasmLinker};
use wasmtime::{Engine, Module, Store};

use crate::api;
use crate::program::{self, ProgramName};
use crate::service::{Service, ServiceHandler};

pub use state::InstanceState;

use crate::process::ProcessId;

// ---- Singleton Actor --------------------------------------------------------

static SERVICE: LazyLock<Service<Message>> = LazyLock::new(Service::new);

/// Spawns the linker service with the given engine.
pub fn spawn(engine: &Engine, allow_filesystem: bool) {
    SERVICE
        .spawn(|| Linker::new(engine, allow_filesystem))
        .expect("linker already spawned");
}

// ---- Public API (message wrappers) ------------------------------------------

/// Link and instantiate a program with its dependencies.
pub async fn instantiate(
    process_id: ProcessId,
    username: String,
    program_name: &ProgramName,
    capture_outputs: bool,
    token_budget: Option<usize>,
) -> Result<(Store<InstanceState>, WasmInstance)> {
    let (tx, rx) = oneshot::channel();
    SERVICE.send(Message::Instantiate {
        process_id,
        username,
        program_name: program_name.clone(),
        capture_outputs,
        token_budget,
        response: tx,
    })?;
    rx.await?
}

// ---- State ------------------------------------------------------------------

struct Linker {
    engine: Engine,
    allow_filesystem: bool,
    /// Shared core modules (e.g. CPython interpreter) loaded from py-runtime/shared/.
    /// Registered with every per-instance WasmLinker so component imports can resolve.
    shared_modules: Arc<Vec<(String, Module)>>,
    /// Path to the py-runtime directory (~/.pie/py-runtime), if it exists.
    /// Passed to InstanceState so Python inferlets get PYTHONHOME and preopened dirs.
    py_runtime_dir: Option<PathBuf>,
}

impl Linker {
    fn new(engine: &Engine, allow_filesystem: bool) -> Self {
        let py_runtime_dir = {
            let dir = crate::path::get_py_runtime_dir();
            if dir.is_dir() {
                tracing::info!("Python runtime directory: {}", dir.display());
                Some(dir)
            } else {
                tracing::info!("No Python runtime directory found at {}", dir.display());
                None
            }
        };

        let shared_modules = if let Some(ref dir) = py_runtime_dir {
            let shared_dir = dir.join("shared");
            if shared_dir.is_dir() {
                load_shared_modules(engine, &shared_dir)
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        if !shared_modules.is_empty() {
            tracing::info!("Loaded {} shared core module(s)", shared_modules.len());
        }

        Linker {
            engine: engine.clone(),
            allow_filesystem,
            shared_modules: Arc::new(shared_modules),
            py_runtime_dir,
        }
    }

    async fn instantiate(
        &self,
        process_id: ProcessId,
        username: String,
        program_name: &ProgramName,
        capture_outputs: bool,
        token_budget: Option<usize>,
    ) -> Result<(Store<InstanceState>, WasmInstance)> {
        // 1. Get the main component
        let component = program::get_wasm_component(program_name)
            .await
            .ok_or_else(|| anyhow!("Component not found for program: {}", program_name))?;

        // 2. Get dependency components
        let dependency_components = self.resolve_dependency_components(program_name).await?;

        // 3. Create instance state and store
        let inst_state = InstanceState::new(
            process_id,
            username,
            capture_outputs,
            self.allow_filesystem,
            token_budget,
            self.py_runtime_dir.as_deref(),
        );
        let mut store = Store::new(&self.engine, inst_state);

        // 4. Create and configure linker
        let mut linker = WasmLinker::<InstanceState>::new(&self.engine);

        wasmtime_wasi::p2::add_to_linker_async(&mut linker)
            .expect("Failed to link WASI");
        wasmtime_wasi_http::add_only_http_to_linker_async(&mut linker)
            .expect("Failed to link WASI HTTP");

        api::add_to_linker(&mut linker)?;

        // Register shared core modules (e.g. CPython interpreter) so Python
        // inferlets can dynamically import the runtime instead of bundling it.
        for (name, module) in self.shared_modules.iter() {
            linker
                .root()
                .module(name, module)
                .unwrap_or_else(|e| panic!("Failed to register shared module '{name}': {e}"));
        }

        // 5. Instantiate library dependencies (dynamic linking)
        if !dependency_components.is_empty() {
            dynamic_linking::instantiate_libraries(
                &self.engine,
                &mut linker,
                &mut store,
                dependency_components,
            )
            .await?;
        }

        // 6. Instantiate the main component
        let instance = linker
            .instantiate_async(&mut store, &component)
            .await
            .map_err(|e| anyhow!("Instantiation error: {e}"))?;

        Ok((store, instance))
    }

    /// Resolve and fetch all dependency components for a program.
    async fn resolve_dependency_components(
        &self,
        program_name: &ProgramName,
    ) -> Result<Vec<Component>> {
        let manifest = program::fetch_manifest(program_name)
            .await
            .ok_or_else(|| anyhow!("Manifest not found for: {}", program_name))?;

        let dep_names = manifest.dependency_names();
        let mut components = Vec::with_capacity(dep_names.len());

        for dep_name in dep_names {
            let component = program::get_wasm_component(&dep_name)
                .await
                .ok_or_else(|| anyhow!("Dependency component not found: {}", dep_name))?;
            components.push(component);
        }

        Ok(components)
    }
}

/// Loads shared core modules (.wasm files) from a directory.
fn load_shared_modules(engine: &Engine, shared_dir: &Path) -> Vec<(String, Module)> {
    let mut modules = Vec::new();
    let entries = match fs::read_dir(shared_dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(
                "Failed to read shared modules dir {}: {e}",
                shared_dir.display()
            );
            return modules;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Failed to read shared module entry: {e}");
                continue;
            }
        };
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "wasm") {
            let import_name = path.file_stem().unwrap().to_str().unwrap().to_string();
            tracing::info!(
                "Loading shared module: {} -> {}",
                path.display(),
                import_name
            );
            match Module::from_file(engine, &path) {
                Ok(module) => modules.push((import_name, module)),
                Err(e) => tracing::error!("Failed to load shared module {}: {e}", path.display()),
            }
        }
    }
    modules
}

// ---- Messages ---------------------------------------------------------------

enum Message {
    Instantiate {
        process_id: ProcessId,
        username: String,
        program_name: ProgramName,
        capture_outputs: bool,
        token_budget: Option<usize>,
        response: oneshot::Sender<Result<(Store<InstanceState>, WasmInstance)>>,
    },
}

impl ServiceHandler for Linker {
    type Message = Message;

    async fn handle(&mut self, msg: Message) {
        match msg {
            Message::Instantiate { process_id, username, program_name, capture_outputs, token_budget, response } => {
                let _ = response.send(
                    self.instantiate(process_id, username, &program_name, capture_outputs, token_budget).await
                );
            }
        }
    }
}
