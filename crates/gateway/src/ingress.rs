pub mod http;
pub mod identity;
pub mod ws;

use axum::{
    Router,
    routing::{get, post},
};

use crate::GatewayState;

pub fn router(state: GatewayState) -> Router {
    Router::new()
        .route("/v1/generate", post(http::generate)) // REST + SSE, one-shot
        .route("/v1/ws", get(ws::ws)) // WebSocket, multi-turn
        .with_state(state)
}
