use axum::{
    routing::{get, post},
    Router,
};
use engine_manager::{
    handlers::{heartbeat_handler, list_backends_handler, register_backend_handler},
    state::{AppState, SharedState},
};
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let shared_state = Arc::new(RwLock::new(AppState::new()));

    // Build our application with routes
    let app = Router::new()
        .route("/backends", get(list_backends_handler))
        .route("/backends/register", post(register_backend_handler))
        .route("/backends/:backend_id/heartbeat", post(heartbeat_handler))
        .with_state(shared_state);

    // Run it with hyper on localhost:3000
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000)); // TODO: Make configurable
    tracing::info!("engine-management-service listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
