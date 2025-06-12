use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use engine_manager::{
    handlers::{health_handler, heartbeat_handler, list_backends_handler, register_backend_handler,
               controller_status_handler, controller_start_handler, controller_stop_handler,
               shutdown_handler, terminate_backend_handler},
    state::AppState,
};
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};

#[derive(Parser)]
#[command(name = "pie_engine_manager")]
#[command(about = "Symphony Engine Manager Service")]
struct Args {
    /// Bind to all interfaces (0.0.0.0) instead of localhost only (127.0.0.1)
    #[arg(long = "bind-all", help = "Bind to all interfaces (0.0.0.0) instead of localhost only")]
    bind_all: bool,

    /// Port to listen on
    #[arg(short = 'p', long = "port", default_value = "3000")]
    port: u16,

    /// Path to configuration file
    #[arg(short = 'c', long = "config", default_value = "config.json")]
    config: String,

    /// Disable colored output (useful when redirecting to files)
    #[arg(long = "no-color", help = "Disable colored output")]
    no_color: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Initialize tracing with info level by default
    if args.no_color {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    let shared_state = Arc::new(RwLock::new(AppState::new_with_config(args.config.clone())));

    // Set the manager port in the state
    {
        let mut state = shared_state.write().unwrap();
        state.manager_port = Some(args.port);
    }

    // Start timeout monitoring task
    let timeout_state = shared_state.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(10));
        interval.tick().await; // First tick is immediate

        loop {
            interval.tick().await;

            let timeout_backends = {
                let mut state = timeout_state.write().unwrap();
                state.check_for_timeouts(35)
            };

            if !timeout_backends.is_empty() {
                for (backend_id, backend_name) in timeout_backends {
                    tracing::warn!("Backend {} ({}) marked as unresponsive due to timeout", backend_name, backend_id);
                }
            }
        }
    });

    // Build our application with routes
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/backends", get(list_backends_handler))
        .route("/backends/register", post(register_backend_handler))
        .route("/backends/:backend_id/heartbeat", post(heartbeat_handler))
        .route("/backends/:backend_id/terminate", post(terminate_backend_handler))
        .route("/controller/status", get(controller_status_handler))
        .route("/controller/start", post(controller_start_handler))
        .route("/controller/stop", post(controller_stop_handler))
        .route("/shutdown", post(shutdown_handler))
        .with_state(shared_state);

    // Select address based on command line argument
    let ip = if args.bind_all {
        [0, 0, 0, 0]  // 0.0.0.0 - bind to all interfaces
    } else {
        [127, 0, 0, 1]  // 127.0.0.1 - localhost only
    };

    let addr = SocketAddr::from((ip, args.port));
    println!("Starting engine-management-service on {}", addr);
    tracing::info!("engine-management-service listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
