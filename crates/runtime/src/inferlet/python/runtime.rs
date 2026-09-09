use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use wasmtime::{Engine, Module};

use super::snapshot;

type SharedModules = (Vec<(String, Module)>, Vec<(String, Module)>);

struct State {
    engine: Engine,
    py_runtime_dir: Option<PathBuf>,
    shared_modules: OnceLock<SharedModules>,
    snapshot_enabled: bool,
}

static STATE: OnceLock<State> = OnceLock::new();

pub fn init(engine: &Engine, py_runtime_dir: &Path, snapshot_enabled: bool) {
    if STATE.get().is_some() {
        return;
    }

    let py_runtime_dir = if py_runtime_dir.is_dir() {
        tracing::info!("Python runtime directory: {}", py_runtime_dir.display());
        Some(py_runtime_dir.to_path_buf())
    } else {
        tracing::info!(
            "No Python runtime directory found at {}",
            py_runtime_dir.display()
        );
        None
    };

    let _ = STATE.set(State {
        engine: engine.clone(),
        py_runtime_dir,
        shared_modules: OnceLock::new(),
        snapshot_enabled,
    });
}

fn state() -> &'static State {
    STATE
        .get()
        .expect("python::runtime::init must be called before use")
}

pub fn dir() -> Option<&'static Path> {
    state().py_runtime_dir.as_deref()
}

pub fn full_modules() -> &'static [(String, Module)] {
    &loaded_modules().0
}

pub fn stripped_modules() -> &'static [(String, Module)] {
    &loaded_modules().1
}

pub fn is_snapshot_enabled() -> bool {
    state().snapshot_enabled
}

pub fn is_available() -> bool {
    state().py_runtime_dir.is_some() && !full_modules().is_empty()
}

fn loaded_modules() -> &'static SharedModules {
    let state = state();
    state.shared_modules.get_or_init(|| {
        let Some(dir) = state.py_runtime_dir.as_ref() else {
            return (Vec::new(), Vec::new());
        };
        let shared_dir = dir.join("shared");
        if !shared_dir.is_dir() {
            return (Vec::new(), Vec::new());
        }

        let loaded = load_shared_modules(&state.engine, &shared_dir);
        if !loaded.0.is_empty() {
            tracing::info!(
                "Loaded {} shared core module(s); snapshot {}",
                loaded.0.len(),
                if state.snapshot_enabled {
                    "enabled"
                } else {
                    "disabled"
                },
            );
        }
        loaded
    })
}

fn load_shared_modules(engine: &Engine, shared_dir: &Path) -> SharedModules {
    let mut full = Vec::new();
    let mut stripped = Vec::new();

    let entries = match fs::read_dir(shared_dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!(
                "Failed to read shared modules dir {}: {e}",
                shared_dir.display()
            );
            return (full, stripped);
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

            let bytes = match fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    tracing::error!("Failed to read shared module {}: {e}", path.display());
                    continue;
                }
            };

            match Module::new(engine, &bytes) {
                Ok(module) => full.push((import_name.clone(), module)),
                Err(e) => {
                    tracing::error!("Failed to compile shared module {}: {e}", path.display());
                    continue;
                }
            }

            match snapshot::strip_module_data(&bytes) {
                Ok(stripped_bytes) => match Module::new(engine, &stripped_bytes) {
                    Ok(module) => stripped.push((import_name, module)),
                    Err(e) => tracing::error!(
                        "Failed to compile stripped shared module {}: {e}",
                        path.display()
                    ),
                },
                Err(e) => tracing::error!("Failed to strip shared module {}: {e}", path.display()),
            }
        }
    }

    (full, stripped)
}
