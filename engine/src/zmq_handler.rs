use anyhow::{anyhow, Context};
use std::collections::HashMap;
use tokio::time::{sleep, timeout, Duration};
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};
use backend_management_rs::types::{ManagementCommand, ManagementResponse};

/// Configuration for management service connection
pub struct ManagementConfig {
    pub endpoint: String,
}

/// Helper to connect to ZMQ socket with timeout
async fn connect_with_timeout(socket: &mut DealerSocket, endpoint: &str, timeout_seconds: u64) -> anyhow::Result<()> {
    timeout(Duration::from_secs(timeout_seconds), socket.connect(endpoint))
        .await
        .map_err(|_| anyhow!("Connection to {} timed out after {} seconds", endpoint, timeout_seconds))?
        .context("Failed to connect to service")?;
    Ok(())
}

/// Helper to send commands to management service via ZMQ
pub async fn send_management_command(
    socket: &mut DealerSocket,
    command: &str,
    params: HashMap<String, serde_json::Value>,
) -> anyhow::Result<ManagementResponse> {
    let command_obj = ManagementCommand::new(command.to_string(), params);
    
    let command_json = serde_json::to_string(&command_obj)
        .context("Failed to serialize command")?;

    let message = ZmqMessage::from(command_json.as_bytes().to_vec());
    
    // Send with timeout
    timeout(Duration::from_secs(10), socket.send(message))
        .await
        .map_err(|_| anyhow!("Command send timed out after 10 seconds"))?
        .context("Failed to send command")?;

    // Receive with timeout
    let response_msg = timeout(Duration::from_secs(10), socket.recv())
        .await
        .map_err(|_| anyhow!("Response receive timed out after 10 seconds"))?
        .context("Failed to receive response")?;
        
    let response_bytes = response_msg.get(0)
        .context("Empty response")?;

    let response: ManagementResponse = serde_json::from_slice(response_bytes)
        .context("Failed to parse response")?;

    Ok(response)
}

/// Find a model in the status response
pub fn find_model_in_status<'a>(response: &'a ManagementResponse, model_name: &str) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    response.data
        .as_ref()?
        .get("models")?
        .as_array()?
        .iter()
        .find_map(|model| {
            let model_obj = model.as_object()?;
            if model_obj.get("model_name")?.as_str()? == model_name {
                Some(model_obj)
            } else {
                None
            }
        })
}

/// Check if a model is available via management service
pub async fn check_model_availability(model_name: &str, config: &ManagementConfig) -> anyhow::Result<bool> {
    let mut socket = DealerSocket::new();
    connect_with_timeout(&mut socket, &config.endpoint, 5).await
        .context("Failed to connect to management service")?;

    let status_response = send_management_command(&mut socket, "status", HashMap::new()).await?;

    if !status_response.success {
        return Err(anyhow!(
            "Management service status check failed: {}",
            status_response.error.unwrap_or_else(|| "Unknown error".to_string())
        ));
    }

    Ok(find_model_in_status(&status_response, model_name).is_some())
}

/// Get model endpoint from management service, loading if necessary
pub async fn get_model_endpoint(model_name: &str, config: &ManagementConfig) -> anyhow::Result<String> {
    let mut socket = DealerSocket::new();
    connect_with_timeout(&mut socket, &config.endpoint, 5).await
        .context("Failed to connect to management service")?;

    // Check if model is already loaded
    let status_response = send_management_command(&mut socket, "status", HashMap::new()).await?;

    if let Some(model) = find_model_in_status(&status_response, model_name) {
        let endpoint = model.get("endpoint")
            .and_then(|ep| ep.as_str())
            .context("No endpoint in loaded model info")?;
        
        return Ok(endpoint.to_string());
    }

    // Model not loaded, attempt to load it
    let mut load_params = HashMap::new();
    load_params.insert("model_name".to_string(), serde_json::Value::String(model_name.to_string()));
    
    let load_response = send_management_command(&mut socket, "load-model", load_params).await?;

    if !load_response.success {
        return Err(anyhow!(
            "Failed to load model {}: {}",
            model_name,
            load_response.error.unwrap_or_else(|| "Unknown error".to_string())
        ));
    }

    let endpoint = load_response.data
        .as_ref()
        .and_then(|data| data.get("endpoint"))
        .and_then(|ep| ep.as_str())
        .context("No endpoint in load response")?;

    // Wait for the backend process to start and create the IPC socket
    wait_for_endpoint_available(endpoint, 30).await
        .context("Model loaded but backend endpoint did not become available")?;
    
    Ok(endpoint.to_string())
}

/// Check management service status
pub async fn check_management_service_status(config: &ManagementConfig) -> anyhow::Result<()> {
    let mut socket = DealerSocket::new();
    connect_with_timeout(&mut socket, &config.endpoint, 5).await
        .context("Failed to connect to management service")?;

    let response = send_management_command(&mut socket, "status", HashMap::new()).await?;

    if !response.success {
        return Err(anyhow!(
            "Management service status check failed: {}",
            response.error.unwrap_or_else(|| "Unknown error".to_string())
        ));
    }

    Ok(())
}

/// Helper function to wait for an IPC endpoint to become available
async fn wait_for_endpoint_available(endpoint: &str, max_wait_seconds: u64) -> anyhow::Result<()> {
    use std::path::Path;
    use tokio::time::timeout;
    use std::io::{self, Write};
    
    // Extract the file path from the IPC endpoint
    let ipc_path = if endpoint.starts_with("ipc://") {
        &endpoint[6..] // Remove "ipc://" prefix
    } else {
        return Err(anyhow!("Invalid IPC endpoint format: {}", endpoint));
    };
    
    let wait_duration = Duration::from_secs(max_wait_seconds);
    let check_interval = Duration::from_millis(100);
    let spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'];
    let mut spinner_index = 0;
    
    timeout(wait_duration, async {
        loop {
            if Path::new(ipc_path).exists() {
                // Clear the spinner line
                print!("\r{}", " ".repeat(50));
                print!("\r");
                io::stdout().flush().unwrap();
                return Ok(());
            }
            
            // Show spinner
            print!("\r{} Waiting for backend to start...", spinner_chars[spinner_index]);
            io::stdout().flush().unwrap();
            spinner_index = (spinner_index + 1) % spinner_chars.len();
            
            sleep(check_interval).await;
        }
    }).await
    .map_err(|_| anyhow!("Timeout waiting for IPC socket to become available: {}", ipc_path))?
}

/// Connect to ZMQ backend with retry logic and spinner
pub async fn connect_to_backend_with_retry(endpoint: &str) -> anyhow::Result<DealerSocket> {
    use std::io::{self, Write};
    
    let mut socket = DealerSocket::new();
    
    // Retry connection with exponential backoff and spinner
    let mut retry_count = 0;
    let max_retries = 10;
    let initial_delay = Duration::from_millis(100);
    let spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'];
    let mut spinner_index = 0;
    
    loop {
        match timeout(Duration::from_secs(2), socket.connect(endpoint)).await {
            Ok(Ok(_)) => {
                // Clear the spinner line
                print!("\r{}", " ".repeat(50));
                print!("\r");
                io::stdout().flush().unwrap();
                break;
            }
            Ok(Err(_)) | Err(_) => {
                if retry_count >= max_retries {
                    // Clear the spinner line before showing error
                    print!("\r{}", " ".repeat(50));
                    print!("\r");
                    io::stdout().flush().unwrap();
                    return Err(anyhow!(
                        "Failed to connect to {} after {} retries", 
                        endpoint, max_retries
                    ));
                }
                
                // Show spinner
                print!("\r{} Connecting to backend...", spinner_chars[spinner_index]);
                io::stdout().flush().unwrap();
                spinner_index = (spinner_index + 1) % spinner_chars.len();
                
                let delay = initial_delay * (1 << retry_count); // Exponential backoff
                sleep(delay).await;
                retry_count += 1;
            }
        }
    }
    
    Ok(socket)
}
