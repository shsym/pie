use crate::message::{ClientMessage, ServerMessage};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct GatewayFrame {
    pub message: ClientMessage,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WorkerFrame {
    pub message: ServerMessage,
}
