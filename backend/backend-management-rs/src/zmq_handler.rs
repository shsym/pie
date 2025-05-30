use crate::error::{ManagementError, Result};
use crate::proto::{handshake, ping};
use crate::types::{ManagementCommand, ManagementResponse};
use tokio::sync::mpsc;
use tracing::{info, warn, error, debug};
use prost::Message;
use bytes::Bytes;
use std::collections::HashMap;

/// ZMQ message handler for client handshakes and CLI management
pub struct ZmqHandler {
    /// ZMQ context
    context: zmq::Context,
    /// Client handshake router socket
    client_router: Option<zmq::Socket>,
    /// CLI management router socket  
    cli_router: Option<zmq::Socket>,
    /// Client handshake endpoint
    client_endpoint: String,
    /// CLI management endpoint
    cli_endpoint: String,
    /// Shutdown signal receiver
    shutdown_rx: Option<mpsc::Receiver<()>>,
}

impl ZmqHandler {
    /// Create a new ZMQ handler
    pub fn new(client_endpoint: String, cli_endpoint: String) -> Self {
        Self {
            context: zmq::Context::new(),
            client_router: None,
            cli_router: None,
            client_endpoint,
            cli_endpoint,
            shutdown_rx: None,
        }
    }

    /// Initialize ZMQ sockets
    pub fn init(&mut self) -> Result<()> {
        info!("Initializing ZMQ sockets");

        // Create client handshake socket
        let client_socket = self.context.socket(zmq::ROUTER)?;
        client_socket.bind(&self.client_endpoint)?;
        info!("Client handshake socket bound to: {}", self.client_endpoint);
        self.client_router = Some(client_socket);

        // Create CLI management socket
        let cli_socket = self.context.socket(zmq::ROUTER)?;
        cli_socket.bind(&self.cli_endpoint)?;
        info!("CLI management socket bound to: {}", self.cli_endpoint);
        self.cli_router = Some(cli_socket);

        Ok(())
    }

    /// Start the main message handling loop
    pub async fn run(
        &mut self,
        mut command_tx: mpsc::Sender<(ManagementCommand, mpsc::Sender<ManagementResponse>)>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) -> Result<()> {
        info!("Starting ZMQ message handler");

        let client_socket = self.client_router.as_ref()
            .ok_or_else(|| ManagementError::Service { 
                message: "Client socket not initialized".to_string() 
            })?;

        let cli_socket = self.cli_router.as_ref()
            .ok_or_else(|| ManagementError::Service { 
                message: "CLI socket not initialized".to_string() 
            })?;

        // Set up polling
        let mut items = [
            client_socket.as_poll_item(zmq::POLLIN),
            cli_socket.as_poll_item(zmq::POLLIN),
        ];

        loop {
            // Check for shutdown signal
            if let Ok(()) = shutdown_rx.try_recv() {
                info!("Received shutdown signal, stopping ZMQ handler");
                break;
            }

            // Poll sockets with timeout
            match zmq::poll(&mut items, 100) {
                Ok(_) => {
                    // Handle client handshake messages
                    if items[0].is_readable() {
                        if let Err(e) = self.handle_client_message(client_socket).await {
                            error!("Error handling client message: {}", e);
                        }
                    }

                    // Handle CLI management messages
                    if items[1].is_readable() {
                        if let Err(e) = self.handle_cli_message(cli_socket, &mut command_tx).await {
                            error!("Error handling CLI message: {}", e);
                        }
                    }
                }
                Err(zmq::Error::EAGAIN) => {
                    // Timeout, continue loop
                    continue;
                }
                Err(e) => {
                    error!("ZMQ polling error: {}", e);
                    return Err(ManagementError::Zmq(e));
                }
            }
        }

        info!("ZMQ handler shutdown complete");
        Ok(())
    }

    /// Handle client handshake messages
    async fn handle_client_message(&self, socket: &zmq::Socket) -> Result<()> {
        debug!("Handling client handshake message");

        // Read multipart message
        let msg = socket.recv_multipart(zmq::DONTWAIT)?;
        if msg.len() < 2 {
            warn!("Received invalid client message with {} parts", msg.len());
            return Ok(());
        }

        let client_id = &msg[0];
        let payload = &msg[1];

        // Try to parse as handshake request
        match handshake::Request::decode(&payload[..]) {
            Ok(_request) => {
                debug!("Received handshake request from client");
                
                // Create handshake response with supported protocols
                let response = handshake::Response {
                    protocols: vec![
                        "l4m".to_string(),
                        "ping".to_string(),
                        "l4m-vision".to_string(),
                    ],
                };

                let mut response_buf = Vec::new();
                response.encode(&mut response_buf)?;

                // Send response back to client
                socket.send_multipart(&[client_id, &response_buf], 0)?;
                debug!("Sent handshake response to client");
            }
            Err(e) => {
                warn!("Failed to decode handshake request: {}", e);
            }
        }

        Ok(())
    }

    /// Handle CLI management messages
    async fn handle_cli_message(
        &self,
        socket: &zmq::Socket,
        command_tx: &mut mpsc::Sender<(ManagementCommand, mpsc::Sender<ManagementResponse>)>,
    ) -> Result<()> {
        debug!("Handling CLI management message");

        // Read multipart message
        let msg = socket.recv_multipart(zmq::DONTWAIT)?;
        if msg.len() < 2 {
            warn!("Received invalid CLI message with {} parts", msg.len());
            return Ok(());
        }

        let client_id = &msg[0];
        let payload = &msg[1];

        // Try to parse as JSON command
        let command_str = String::from_utf8_lossy(payload);
        match serde_json::from_str::<ManagementCommand>(&command_str) {
            Ok(command) => {
                debug!("Received CLI command: {}", command.command);

                // Create response channel
                let (response_tx, mut response_rx) = mpsc::channel::<ManagementResponse>(1);

                // Send command to service
                if let Err(e) = command_tx.send((command, response_tx)).await {
                    error!("Failed to send command to service: {}", e);
                    return Ok(());
                }

                // Wait for response
                if let Some(response) = response_rx.recv().await {
                    let response_json = serde_json::to_string(&response)?;
                    socket.send_multipart(&[client_id, response_json.as_bytes()], 0)?;
                    debug!("Sent CLI response");
                } else {
                    warn!("No response received from service");
                }
            }
            Err(e) => {
                warn!("Failed to decode CLI command: {}", e);
                
                // Send error response
                let error_response = ManagementResponse::error(
                    "unknown".to_string(),
                    format!("Invalid command format: {}", e)
                );
                let response_json = serde_json::to_string(&error_response)?;
                socket.send_multipart(&[client_id, response_json.as_bytes()], 0)?;
            }
        }

        Ok(())
    }

    /// Close ZMQ sockets
    pub fn close(&mut self) {
        if let Some(socket) = self.client_router.take() {
            let _ = socket.disconnect(&self.client_endpoint);
        }
        if let Some(socket) = self.cli_router.take() {
            let _ = socket.disconnect(&self.cli_endpoint);
        }
        info!("ZMQ sockets closed");
    }
}

impl Drop for ZmqHandler {
    fn drop(&mut self) {
        self.close();
    }
}

impl std::fmt::Debug for ZmqHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZmqHandler")
            .field("client_endpoint", &self.client_endpoint)
            .field("cli_endpoint", &self.cli_endpoint)
            .field("client_router", &"<ZMQ Socket>")
            .field("cli_router", &"<ZMQ Socket>")
            .field("context", &"<ZMQ Context>")
            .finish()
    }
}
