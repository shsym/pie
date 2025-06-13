use crate::error::{ManagementError, Result};
use crate::proto::handshake;
use crate::types::{ManagementCommand, ManagementResponse, ModelInstance};
use tokio::sync::mpsc;
use tracing::{info, warn, error, debug};
use prost::Message;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use zeromq::{RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage};
use tokio::time::{timeout, Duration};
use prost::bytes::Bytes;

/// ZMQ message handler for client handshakes and CLI management
pub struct ZmqHandler {
    /// Client handshake router socket
    client_router: Option<RouterSocket>,
    /// CLI management router socket  
    cli_router: Option<RouterSocket>,
    /// Client handshake endpoint
    client_endpoint: String,
    /// CLI management endpoint
    cli_endpoint: String,
    /// Reference to model instances registry to compute available protocols
    model_instances: Arc<RwLock<HashMap<String, ModelInstance>>>,
    /// Shutdown signal receiver
    _shutdown_rx: Option<mpsc::Receiver<()>>,
}

impl ZmqHandler {
    /// Create a new ZMQ handler
    pub fn new(client_endpoint: String, cli_endpoint: String, model_instances: Arc<RwLock<HashMap<String, ModelInstance>>>) -> Self {
        Self {
            client_router: None,
            cli_router: None,
            client_endpoint,
            cli_endpoint,
            model_instances,
            _shutdown_rx: None,
        }
    }

    /// Initialize ZMQ sockets
    pub async fn init(&mut self) -> Result<()> {
        info!("Initializing ZMQ sockets");

        // Create client handshake socket
        let mut client_socket = RouterSocket::new();
        client_socket.bind(&self.client_endpoint).await
            .map_err(|e| ManagementError::Service { 
                message: format!("Failed to bind client socket to {}: {}", self.client_endpoint, e)
            })?;
        info!("Client handshake socket bound to: {}", self.client_endpoint);
        self.client_router = Some(client_socket);

        // Create CLI management socket
        let mut cli_socket = RouterSocket::new();
        cli_socket.bind(&self.cli_endpoint).await
            .map_err(|e| ManagementError::Service { 
                message: format!("Failed to bind CLI socket to {}: {}", self.cli_endpoint, e)
            })?;
        info!("CLI management socket bound to: {}", self.cli_endpoint);
        self.cli_router = Some(cli_socket);

        Ok(())
    }

    /// Compute available protocols from all running model instances
    fn get_available_protocols(&self) -> Vec<String> {
        let instances = self.model_instances.read().unwrap();
        let mut available_protocols = std::collections::HashSet::new();
        
        for instance in instances.values() {
            for protocol in &instance.supported_protocols {
                available_protocols.insert(protocol.clone());
            }
        }
        
        let protocols: Vec<String> = available_protocols.into_iter().collect();
        debug!("Current available protocols: {:?}", protocols);
        protocols
    }

    /// Start the main message handling loop
    pub async fn run(
        &mut self,
        mut command_tx: mpsc::Sender<(ManagementCommand, mpsc::Sender<ManagementResponse>)>,
        mut shutdown_rx: mpsc::Receiver<()>,
    ) -> Result<()> {
        info!("Starting ZMQ message handler");

        let mut client_socket = self.client_router.take()
            .ok_or_else(|| ManagementError::Service { 
                message: "Client socket not initialized".to_string() 
            })?;

        let mut cli_socket = self.cli_router.take()
            .ok_or_else(|| ManagementError::Service { 
                message: "CLI socket not initialized".to_string() 
            })?;

        loop {
            tokio::select! {
                // Check for shutdown signal
                _ = shutdown_rx.recv() => {
                    info!("Received shutdown signal, stopping ZMQ handler");
                    break;
                }

                // Handle client handshake messages
                result = timeout(Duration::from_millis(100), client_socket.recv()) => {
                    match result {
                        Ok(Ok(msg)) => {
                            if let Err(e) = self.handle_client_message(&mut client_socket, msg).await {
                                error!("Error handling client message: {}", e);
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Client socket receive error: {}", e);
                        }
                        Err(_) => {
                            // Timeout, continue loop
                        }
                    }
                }

                // Handle CLI management messages
                result = timeout(Duration::from_millis(100), cli_socket.recv()) => {
                    match result {
                        Ok(Ok(msg)) => {
                            if let Err(e) = self.handle_cli_message(&mut cli_socket, msg, &mut command_tx).await {
                                error!("Error handling CLI message: {}", e);
                            }
                        }
                        Ok(Err(e)) => {
                            error!("CLI socket receive error: {}", e);
                        }
                        Err(_) => {
                            // Timeout, continue loop
                        }
                    }
                }
            }
        }

        info!("ZMQ handler shutdown complete");
        Ok(())
    }

    /// Handle client handshake messages
    async fn handle_client_message(&self, socket: &mut RouterSocket, msg: ZmqMessage) -> Result<()> {
        debug!("Handling client handshake message");

        // Extract frames from ZMQ message
        if msg.len() < 2 {
            warn!("Received invalid client message with {} parts", msg.len());
            return Ok(());
        }

        let client_id = msg.get(0).unwrap();
        let payload = msg.get(1).unwrap();

        // Try to parse as handshake request
        match handshake::Request::decode(&payload[..]) {
            Ok(_request) => {
                debug!("Received handshake request from client");
                
                // Get currently available protocols from all running backends
                let protocols = self.get_available_protocols();
                
                // Create handshake response with supported protocols
                let response = handshake::Response {
                    protocols,
                };

                let mut response_buf = Vec::new();
                response.encode(&mut response_buf)
                    .map_err(|e| ManagementError::Service { 
                        message: format!("Failed to encode handshake response: {}", e)
                    })?;

                // Create response message with client ID and payload
                let mut response_frames = VecDeque::new();
                response_frames.push_back(Bytes::from(client_id.to_vec()));
                response_frames.push_back(Bytes::from(response_buf));
                
                let response_msg = ZmqMessage::try_from(response_frames)
                    .map_err(|e| ManagementError::Service { 
                        message: format!("Failed to create response message: {}", e)
                    })?;

                // Send response back to client
                socket.send(response_msg).await
                    .map_err(|e| ManagementError::Service { 
                        message: format!("Failed to send handshake response: {}", e)
                    })?;
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
        socket: &mut RouterSocket,
        msg: ZmqMessage,
        command_tx: &mut mpsc::Sender<(ManagementCommand, mpsc::Sender<ManagementResponse>)>,
    ) -> Result<()> {
        debug!("Handling CLI management message");

        // Extract frames from ZMQ message
        if msg.len() < 2 {
            warn!("Received invalid CLI message with {} parts", msg.len());
            return Ok(());
        }

        let client_id = msg.get(0).unwrap();
        let payload = msg.get(1).unwrap();

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
                    let response_json = serde_json::to_string(&response)
                        .map_err(|e| ManagementError::Service { 
                            message: format!("Failed to serialize response: {}", e)
                        })?;

                    // Create response message with client ID and payload
                    let mut response_frames = VecDeque::new();
                    response_frames.push_back(Bytes::from(client_id.to_vec()));
                    response_frames.push_back(Bytes::from(response_json.into_bytes()));
                    
                    let response_msg = ZmqMessage::try_from(response_frames)
                        .map_err(|e| ManagementError::Service { 
                            message: format!("Failed to create CLI response message: {}", e)
                        })?;

                    match socket.send(response_msg).await {
                        Ok(()) => {
                            debug!("Sent CLI response");
                        },
                        Err(e) => {
                            // Check if this is a broken pipe (client disconnected)
                            let error_msg = e.to_string();
                            if error_msg.contains("Broken pipe") || error_msg.contains("Connection reset") {
                                debug!("CLI client disconnected before receiving response (likely timeout)");
                            } else {
                                return Err(ManagementError::Service { 
                                    message: format!("Failed to send CLI response: {}", e)
                                });
                            }
                        }
                    }
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
                let response_json = serde_json::to_string(&error_response)
                    .map_err(|e| ManagementError::Service { 
                        message: format!("Failed to serialize error response: {}", e)
                    })?;

                // Create error response message
                let mut error_frames = VecDeque::new();
                error_frames.push_back(Bytes::from(client_id.to_vec()));
                error_frames.push_back(Bytes::from(response_json.into_bytes()));
                
                let error_msg = ZmqMessage::try_from(error_frames)
                    .map_err(|e| ManagementError::Service { 
                        message: format!("Failed to create error response message: {}", e)
                    })?;

                socket.send(error_msg).await
                    .map_err(|e| ManagementError::Service { 
                        message: format!("Failed to send error response: {}", e)
                    })?;
            }
        }

        Ok(())
    }

    /// Close ZMQ sockets
    pub fn close(&mut self) {
        // Sockets will be automatically closed when dropped
        self.client_router = None;
        self.cli_router = None;
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
            .field("client_router", &"<Router Socket>")
            .field("cli_router", &"<Router Socket>")
            .finish()
    }
}
