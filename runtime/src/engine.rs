use anyhow::{Context, Result, anyhow};
// use ring::rand::{SecureRandom, SystemRandom};
use std::fs;
use std::path::PathBuf;
use tokio::sync::oneshot;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

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
    // Initialize tracing with file logging if log_dir is specified
    init_tracing(&config.log_dir, config.verbose)?;

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

/// Initialize the tracing subscriber with optional file logging.
fn init_tracing(log_dir: &Option<PathBuf>, verbose: bool) -> Result<()> {
    use tracing_subscriber::fmt;
    use tracing_subscriber::EnvFilter;

    let filter = if verbose {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("debug"))
    } else {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info"))
    };

    match log_dir {
        Some(dir) => {
            // Ensure log directory exists
            fs::create_dir_all(dir).with_context(|| {
                format!("Failed to create log directory: {:?}", dir)
            })?;

            // Create a rolling file appender that rotates daily
            let file_appender = tracing_appender::rolling::daily(dir, "pie.log");
            let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

            // Keep the guard alive for the lifetime of the process
            // by leaking it (this is intentional for logging)
            std::mem::forget(_guard);

            tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer().with_writer(non_blocking).with_ansi(false))
                .init();
        }
        None => {
            // Log to stdout if no log directory specified
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer())
                .init();
        }
    }

    Ok(())
}

