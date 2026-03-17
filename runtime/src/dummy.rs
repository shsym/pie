//! Dummy backend implementation for testing purposes.
//!
//! This module provides a minimal backend that:
//! - Connects to the Pie engine via WebSocket
//! - Authenticates using internal authentication
//! - Registers as a remote service (for testing registration flow)
//! - Returns minimal/error responses for handler requests
//!
//! The dummy backend is primarily used for CI testing to verify engine behavior
//! without requiring a full model backend.

use anyhow::{Context, Result};
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use pie_client::message::{ClientMessage, ServerMessage};
use rand::Rng;
use rmp_serde::encode;
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message as WsMessage};

/// Configuration for the dummy backend.
#[derive(Debug, Clone)]
pub struct DummyBackendConfig {
    pub controller_host: String,
    pub controller_port: u16,
    pub internal_auth_token: String,
}

/// Starts the dummy backend.
///
/// This function connects to the engine, authenticates, and registers as a service.
/// It runs indefinitely until cancelled.
pub async fn start_dummy_backend(config: DummyBackendConfig) -> Result<()> {
    println!("[Dummy Backend] Starting...");

    // Connect to the engine via WebSocket
    let controller_url = format!("ws://{}:{}", config.controller_host, config.controller_port);
    let (ws_stream, _) = connect_async(&controller_url)
        .await
        .context(format!("Failed to connect to engine at {}", controller_url))?;

    let (mut ws_write, mut ws_read) = ws_stream.split();

    // Authenticate with the engine
    let auth_msg = ClientMessage::InternalAuthenticate {
        corr_id: 0,
        token: config.internal_auth_token.clone(),
    };
    let auth_bytes = Bytes::from(
        encode::to_vec_named(&auth_msg).context("Failed to encode authentication message")?,
    );
    ws_write
        .send(WsMessage::Binary(auth_bytes))
        .await
        .context("Failed to send authentication message")?;

    // Wait for authentication response
    let response = ws_read
        .next()
        .await
        .context("Connection closed before authentication response")??;
    let response_data = response.into_data();
    let server_msg: ServerMessage = rmp_serde::from_slice(&response_data)
        .context("Failed to decode authentication response")?;

    match server_msg {
        ServerMessage::Response {
            successful, result, ..
        } => {
            if !successful {
                anyhow::bail!("Authentication failed: {}", result);
            }
        }
        _ => anyhow::bail!("Unexpected response type for authentication"),
    }
    println!("[Dummy Backend] Connected to engine at {}", controller_url);

    // Generate unique service name
    let unique_id: u32 = rand::rng().random_range(100000..=999999);
    let service_name = format!("pie-dummy-backend-{}", unique_id);

    // Note: In FFI mode (the only mode now), remote service attachment is not supported.
    // This registration will be rejected by the server, but is useful for testing the
    // registration flow and error handling.

    // Register with the engine
    let register_msg = ClientMessage::AttachRemoteService {
        corr_id: 0,
        endpoint: service_name.clone(),
        service_type: "model".to_string(),
        service_name: "dummy-model".to_string(),
    };
    let register_bytes = Bytes::from(
        encode::to_vec_named(&register_msg).context("Failed to encode registration message")?,
    );
    ws_write
        .send(WsMessage::Binary(register_bytes))
        .await
        .context("Failed to send registration message")?;

    // Wait for registration response
    let response = ws_read
        .next()
        .await
        .context("Connection closed before registration response")??;
    let response_data = response.into_data();
    let server_msg: ServerMessage =
        rmp_serde::from_slice(&response_data).context("Failed to decode registration response")?;

    match server_msg {
        ServerMessage::Response {
            successful, result, ..
        } => {
            if !successful {
                anyhow::bail!("Registration failed: {}", result);
            }
        }
        _ => anyhow::bail!("Unexpected response type for registration"),
    }
    println!(
        "[Dummy Backend] Registered with engine (service: {})",
        service_name
    );

    // Keep the WebSocket connection alive
    // Note: Remote service registration is no longer supported in FFI mode.
    // The server will reject the registration, but this is useful for testing error handling.
    println!("[Dummy Backend] Note: Remote service attachment is not supported in FFI mode.");
    println!("[Dummy Backend] This is only useful for registration testing.");

    while let Some(result) = ws_read.next().await {
        match result {
            Ok(msg) => {
                eprintln!("[Dummy Backend] Unexpected WebSocket message: {:?}", msg);
            }
            Err(e) => {
                eprintln!("[Dummy Backend] WebSocket error: {}", e);
                break;
            }
        }
    }

    Ok(())
}
