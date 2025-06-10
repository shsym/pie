use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use engine_manager::{
    handlers::{heartbeat_handler, list_backends_handler, register_backend_handler},
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
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Initialize tracing
    tracing_subscriber::fmt::init();

    let shared_state = Arc::new(RwLock::new(AppState::new()));

    // Build our application with routes
    let app = Router::new()
        .route("/backends", get(list_backends_handler))
        .route("/backends/register", post(register_backend_handler))
        .route("/backends/:backend_id/heartbeat", post(heartbeat_handler))
        .with_state(shared_state);

    // Select address based on command line argument
    let ip = if args.bind_all {
        [0, 0, 0, 0]  // 0.0.0.0 - bind to all interfaces
    } else {
        [127, 0, 0, 1]  // 127.0.0.1 - localhost only
    };

    let addr = SocketAddr::from((ip, args.port));
    tracing::info!("engine-management-service listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
