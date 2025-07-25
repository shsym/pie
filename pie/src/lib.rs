use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use tokio::sync::oneshot;

// Public modules
pub mod auth;
pub mod client;

// Internal modules
mod service;
mod backend;
mod batching;
mod bindings;
mod instance;
mod l4m;
mod messaging;
mod object;
mod ping;
mod runtime;
pub mod server;
mod tokenizer;
mod utils;
mod bindings2;

// Re-export core components from internal modules
use crate::auth::{create_jwt, init_secret};
use crate::messaging::{PubSub, PushPull};
use crate::ping::Ping;
use crate::runtime::Runtime;
use crate::server::Server;
use crate::service::install_service;

/// Configuration for the PIE engine.
#[derive(Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub enable_auth: bool,
    pub auth_secret: String,
    pub cache_dir: PathBuf,
    pub verbose: bool,
    pub log: Option<PathBuf>,
}

/// Runs the PIE server logic within an existing Tokio runtime.
///
/// This async function sets up all the engine's services and listens for a shutdown
/// signal to terminate gracefully.
pub async fn run_server(config: Config, mut shutdown_rx: oneshot::Receiver<()>) -> Result<()> {
    // Ensure the cache directory exists
    fs::create_dir_all(&config.cache_dir).with_context(|| {
        let err_msg = format!("Setup failure: could not create cache dir at {:?}", &config.cache_dir);
        tracing::error!(error = %err_msg);
        err_msg
    })?;

    if config.enable_auth {
        tracing::info!("Authentication is enabled.");
        init_secret(&config.auth_secret);
        let token = create_jwt("default", auth::Role::User)?;
        tracing::info!("Use this token to authenticate: {}", token);
    } else {
        tracing::info!("Authentication is disabled.");
    }

    // Set up core services
    let runtime = Runtime::new(&config.cache_dir);
    runtime.load_existing_programs()?;

    let server_url = format!("{}:{}", config.host, config.port);

    let server = Server::new(&server_url, config.enable_auth);
    let messaging_inst2inst = PubSub::new();
    let messaging_user2inst = PushPull::new();
    let dummy_ping_backend = backend::SimulatedBackend::new(ping::Simulator::new()).await;
    let ping = Ping::new(dummy_ping_backend.clone()).await;

    install_service("runtime", runtime);
    install_service("server", server);
    install_service("messaging-inst2inst", messaging_inst2inst);
    install_service("messaging-user2inst", messaging_user2inst);
    install_service("ping", ping);

    tracing::info!("✅ PIE runtime started successfully on {}", server_url);

    // Wait for either a Ctrl+C signal or the shutdown signal from the parent task
    tokio::select! {
        res = tokio::signal::ctrl_c() => {
            if let Err(e) = res {
                tracing::error!("Failed to listen for Ctrl+C: {}", e);
            }
            tracing::info!("Ctrl+C received, shutting down.");
        }
        _ = &mut shutdown_rx => {
            tracing::info!("Shutdown signal received, shutting down.");
        }
    }

    Ok(())
}