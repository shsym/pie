use anyhow::{Context, Result, anyhow};
// use ring::rand::{SecureRandom, SystemRandom};
use std::fs;
use std::path::PathBuf;
use tokio::sync::oneshot;

use crate::auth::AuthorizedUsers;
use crate::kvs;
use crate::messaging;
use crate::runtime;
use crate::server;

/// Configuration for the PIE engine.
#[derive(Debug)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub enable_auth: bool,
    pub cache_dir: PathBuf,
    pub verbose: bool,
    pub log_dir: Option<PathBuf>,
    pub registry: String,
}

/// Runs the PIE server logic within an existing Tokio runtime.
///
/// This async function sets up all the engine's services and listens for a shutdown
/// signal to terminate gracefully.
pub async fn run_server(
    config: Config,
    authorized_users: AuthorizedUsers,
    ready_tx: oneshot::Sender<String>,
    shutdown_rx: oneshot::Receiver<()>,
) -> Result<()> {
    // Ensure the cache directory exists
    fs::create_dir_all(&config.cache_dir).with_context(|| {
        let err_msg = format!(
            "Setup failure: could not create cache dir at {:?}",
            &config.cache_dir
        );
        tracing::error!(error = %err_msg);
        err_msg
    })?;


    let server_url = format!("{}:{}", config.host, config.port);

    // Generate a random 64-character string for internal client connection authentication.
    let internal_auth_token = crate::auth::generate_internal_auth_token()?;

    runtime::start_service(&config.cache_dir);
    server::start_service(
        &server_url,
        config.enable_auth,
        authorized_users,
        internal_auth_token.clone(),
        config.registry.clone(),
        config.cache_dir.clone(),
    );
    kvs::start_service();
    messaging::start_service();

    ready_tx.send(internal_auth_token).unwrap();

    shutdown_rx.await?;

    Ok(())
}
