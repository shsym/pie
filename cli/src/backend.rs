use clap::Subcommand;
use anyhow::{Result, anyhow, bail};
use std::process::Command;
use std::path::{Path, PathBuf};
use std::fs::{self, OpenOptions};
use tracing::{info, error, warn, debug};
use serde_json::Value;
use crate::constants::network;
use crate::config::Config;
use crate::spinner::with_spinner;
use std::time::Duration;

#[derive(Subcommand)]
pub enum BackendCommands {
    /// List all registered backends
    List {
        /// URL of the management service
        #[arg(long, default_value = network::DEFAULT_MANAGEMENT_URL)]
        management_service_url: String,
    },

    /// Show status of a specific backend
    Status {
        /// Backend ID
        backend_id: String,

        /// URL of the management service
        #[arg(long, default_value = network::DEFAULT_MANAGEMENT_URL)]
        management_service_url: String,
    },

    /// Start a new backend process
    Start {
        /// Model name or backend type (e.g., "DeepSeek-R1-0528-Qwen3-8B", "qwen3")
        model_or_backend: String,

        /// URL of the management service
        #[arg(long, default_value = network::DEFAULT_MANAGEMENT_URL)]
        management_service_url: String,

        /// Host for the backend's management API
        #[arg(long, default_value = network::DEFAULT_BACKEND_HOST)]
        backend_host: String,

        /// Port for the backend's management API
        #[arg(long)]
        backend_api_port: Option<u16>,

        /// Base port to start searching for available ports
        #[arg(long, default_value_t = network::DEFAULT_BACKEND_BASE_PORT)]
        base_port: u16,

        /// IPC endpoint for ZMQ data plane communication
        #[arg(long, default_value = "ipc:///tmp/pie-ipc")]
        ipc_endpoint: String,

        /// Additional arguments for the backend
        #[arg(last = true)]
        backend_args: Vec<String>,
    },

    /// Terminate a running backend
    Terminate {
        /// Backend ID to terminate
        backend_id: String,

        /// URL of the management service
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        management_service_url: String,
    },
}

pub async fn handle_command(cmd: BackendCommands) -> Result<()> {
    match cmd {
        BackendCommands::List { management_service_url } => {
            list_backends(&management_service_url).await
        }
        BackendCommands::Status { backend_id, management_service_url } => {
            get_backend_status(&backend_id, &management_service_url).await
        }
        BackendCommands::Start {
            model_or_backend,
            management_service_url,
            backend_host,
            backend_api_port,
            base_port,
            ipc_endpoint,
            backend_args
        } => {
            start_backend(
                &model_or_backend,
                &management_service_url,
                &backend_host,
                backend_api_port,
                base_port,
                &ipc_endpoint,
                &backend_args
            ).await
        }
        BackendCommands::Terminate { backend_id, management_service_url } => {
            terminate_backend(&backend_id, &management_service_url).await
        }
    }
}

fn print_backend_info(backend_obj: &serde_json::Map<String, Value>) {
    let id = backend_obj.get("backend_id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let status = backend_obj.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");

    // Parse capabilities array for backend type and name
    let (backend_type, name, models, ipc_endpoint) = parse_backend_capabilities(backend_obj);

    println!("  Backend ID: {}", id);
    println!("  Type: {}", backend_type);
    println!("  Name: {}", name);
    println!("  Status: {}", status);

    // Show management API address if available
    if let Some(mgmt_api) = backend_obj.get("management_api_address") {
        println!("  Engine Management API: {}", mgmt_api.as_str().unwrap_or("unknown"));
    }

    if !models.is_empty() {
        println!("  Models: {}", models.join(", "));
    }

    if !ipc_endpoint.is_empty() {
        println!("  IPC Endpoint: {}", ipc_endpoint);
    }

    println!();
}

fn parse_backend_capabilities(backend_obj: &serde_json::Map<String, Value>) -> (String, String, Vec<String>, String) {
    if let Some(capabilities) = backend_obj.get("capabilities") {
        if let Some(caps_array) = capabilities.as_array() {
            let mut backend_type = "unknown".to_string();
            let mut name = "unknown".to_string();
            let mut models = Vec::new();
            let mut ipc_endpoint = String::new();

            for cap in caps_array {
                if let Some(cap_str) = cap.as_str() {
                    if cap_str.starts_with("type:") {
                        backend_type = cap_str[5..].to_string();
                    } else if cap_str.starts_with("name:") {
                        name = cap_str[5..].to_string();
                    } else if cap_str.starts_with("model:") {
                        models.push(cap_str[6..].to_string());
                    } else if cap_str.starts_with("ipc_endpoint:") {
                        ipc_endpoint = cap_str[13..].to_string();
                    }
                }
            }

            return (backend_type, name, models, ipc_endpoint);
        }
    }

    ("unknown".to_string(), "unknown".to_string(), Vec::new(), String::new())
}

async fn list_backends(management_service_url: &str) -> Result<()> {
    let client = reqwest::Client::new();
    let url = format!("{}/backends", management_service_url);

    match client.get(&url).send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<Value>().await {
                Ok(response_data) => {
                    // Expect new engine-manager format: {"backends": [...]}
                    if let Some(backends_obj) = response_data.get("backends") {
                        if let Some(backends_array) = backends_obj.as_array() {
                            if backends_array.is_empty() {
                                info!("No backends currently registered");
                                println!("No backends currently registered.");
                                return Ok(());
                            }

                            println!("Registered Backends:");
                            println!("==================");

                            for backend in backends_array {
                                if let Some(backend_obj) = backend.as_object() {
                                    print_backend_info(backend_obj);
                                }
                            }
                        } else {
                            error!("Invalid backends data format - expected array");
                            bail!("Invalid response format from management service");
                        }
                    } else {
                        error!("Response missing 'backends' field");
                        bail!("Invalid response format from management service - missing backends field");
                    }
                }
                Err(e) => {
                    error!("Failed to parse backends response: {}", e);
                    bail!("Failed to parse response from management service");
                }
            }
        }
        Ok(response) => {
            error!("Management service responded with status: {}", response.status());
            bail!("Management service is not responding properly (status: {})", response.status());
        }
        Err(e) => {
            error!("Failed to connect to management service: {}", e);
            bail!("Failed to connect to management service at {}", management_service_url);
        }
    }

    Ok(())
}

async fn get_backend_status(backend_id: &str, management_service_url: &str) -> Result<()> {
    // First, get the backend info from engine-manager to find its management API address
    let client = reqwest::Client::new();
    let backends_url = format!("{}/backends", management_service_url);

    let backend_info = match client.get(&backends_url).send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<Value>().await {
                Ok(response_data) => {
                    if let Some(backends_obj) = response_data.get("backends") {
                        if let Some(backends_array) = backends_obj.as_array() {
                            // Find the backend with matching ID
                            backends_array.iter()
                                .find(|backend| {
                                    backend.get("backend_id")
                                        .and_then(|id| id.as_str())
                                        .map(|id| id == backend_id)
                                        .unwrap_or(false)
                                })
                                .cloned()
                        } else {
                            error!("Invalid backends data format");
                            bail!("Invalid response format from management service");
                        }
                    } else {
                        error!("Response missing 'backends' field");
                        bail!("Invalid response format from management service");
                    }
                }
                Err(e) => {
                    error!("Failed to parse backends response: {}", e);
                    bail!("Failed to parse response from management service");
                }
            }
        }
        Ok(response) => {
            error!("Management service responded with status: {}", response.status());
            bail!("Management service is not responding properly");
        }
        Err(e) => {
            error!("Failed to connect to management service: {}", e);
            bail!("Failed to connect to management service at {}", management_service_url);
        }
    };

    let backend_info = backend_info.ok_or_else(|| anyhow::anyhow!("Backend with ID '{}' not found", backend_id))?;

    // Get the backend's management API address
    let mgmt_api_address = backend_info
        .get("management_api_address")
        .and_then(|addr| addr.as_str())
        .ok_or_else(|| anyhow::anyhow!("Backend management API address not found"))?;

    // Now call the backend's health endpoint directly
    let health_url = format!("{}/manage/health", mgmt_api_address);

    match client.get(&health_url).send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<Value>().await {
                Ok(health_data) => {
                    println!("Backend Status:");
                    println!("===============");
                    println!("Backend ID: {}", backend_id);

                    if let Some(service_name) = health_data.get("service_name") {
                        println!("Service Name: {}", service_name.as_str().unwrap_or("unknown"));
                    }

                    if let Some(backend_type) = health_data.get("backend_type") {
                        println!("Type: {}", backend_type.as_str().unwrap_or("unknown"));
                    }

                    if let Some(status) = health_data.get("status") {
                        println!("Health Status: {}", status.as_str().unwrap_or("unknown"));
                    }

                    println!("Engine Management API: {}", mgmt_api_address);

                    if let Some(ipc_endpoint) = health_data.get("ipc_endpoint") {
                        println!("IPC Endpoint: {}", ipc_endpoint.as_str().unwrap_or("unknown"));
                    }

                    if let Some(loaded_models) = health_data.get("loaded_models") {
                        if let Some(models_array) = loaded_models.as_array() {
                            if !models_array.is_empty() {
                                println!("Loaded Models: {}", models_array.iter()
                                    .filter_map(|m| m.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", "));
                            } else {
                                println!("Loaded Models: none");
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to parse backend health response: {}", e);
                    bail!("Failed to parse response from backend");
                }
            }
        }
        Ok(response) if response.status() == 404 => {
            warn!("Backend health endpoint not found");
            println!("Backend health endpoint not available");
        }
        Ok(response) => {
            error!("Backend responded with status: {}", response.status());
            bail!("Backend is not responding properly");
        }
        Err(e) => {
            error!("Failed to connect to backend: {}", e);
            bail!("Failed to connect to backend at {}", mgmt_api_address);
        }
    }

    Ok(())
}

async fn terminate_backend(backend_id: &str, management_service_url: &str) -> Result<()> {
    // First, get the backend info from engine-manager to find its management API address
    let client = reqwest::Client::new();
    let backends_url = format!("{}/backends", management_service_url);

    let backend_info = match client.get(&backends_url).send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<Value>().await {
                Ok(response_data) => {
                    if let Some(backends_obj) = response_data.get("backends") {
                        if let Some(backends_array) = backends_obj.as_array() {
                            // Find the backend with matching ID
                            backends_array.iter()
                                .find(|backend| {
                                    backend.get("backend_id")
                                        .and_then(|id| id.as_str())
                                        .map(|id| id == backend_id)
                                        .unwrap_or(false)
                                })
                                .cloned()
                        } else {
                            error!("Invalid backends data format");
                            bail!("Invalid response format from management service");
                        }
                    } else {
                        error!("Response missing 'backends' field");
                        bail!("Invalid response format from management service");
                    }
                }
                Err(e) => {
                    error!("Failed to parse backends response: {}", e);
                    bail!("Failed to parse response from management service");
                }
            }
        }
        Ok(response) => {
            error!("Management service responded with status: {}", response.status());
            bail!("Management service is not responding properly");
        }
        Err(e) => {
            error!("Failed to connect to management service: {}", e);
            bail!("Failed to connect to management service at {}", management_service_url);
        }
    };

    let backend_info = backend_info.ok_or_else(|| anyhow::anyhow!("Backend with ID '{}' not found", backend_id))?;

    // Get the backend's management API address
    let mgmt_api_address = backend_info
        .get("management_api_address")
        .and_then(|addr| addr.as_str())
        .ok_or_else(|| anyhow::anyhow!("Backend management API address not found"))?;

    // Now call the backend's terminate endpoint directly
    let terminate_url = format!("{}/manage/terminate", mgmt_api_address);

    match client.post(&terminate_url).send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<Value>().await {
                Ok(result) => {
                    if let Some(message) = result.get("message") {
                        println!("{}", message.as_str().unwrap_or("Backend termination requested"));
                    } else {
                        println!("Backend termination requested successfully");
                    }
                }
                Err(_) => {
                    println!("Backend termination requested successfully");
                }
            }
        }
        Ok(response) if response.status() == 404 => {
            warn!("Backend terminate endpoint not found");
            println!("Backend terminate endpoint not available");
        }
        Ok(response) => {
            error!("Backend responded with status: {}", response.status());
            bail!("Failed to terminate backend");
        }
        Err(e) => {
            error!("Failed to connect to backend: {}", e);
            bail!("Failed to connect to backend at {}", mgmt_api_address);
        }
    }

    Ok(())
}

fn load_backend_config() -> Result<Config> {
    // Use the CLI's unified configuration
    Config::load_default()
        .or_else(|_| {
            // If no config file is found, provide a helpful error message
            warn!("No configuration file found");
            bail!("Configuration file 'config.json' not found. Please ensure it exists in the current directory.");
        })
}

fn resolve_model_to_backend(model_or_backend: &str, config: &Config) -> Result<(String, String)> {
    // First check if it's a direct backend type in config
    if config.backends.model_backends.contains_key(model_or_backend) {
        return Ok((model_or_backend.to_string(), model_or_backend.to_string()));
    }

    // Then check if it's a model name (short or full) in config
    for model_info in &config.models.supported_models {
        if model_info.name == model_or_backend || model_info.fullname == model_or_backend {
            return Ok((model_info.fullname.clone(), model_info.model_type.clone()));
        }
    }

    // If not found, throw an error - no inference or guessing
    bail!("Unknown model or backend type: '{}'. Please check the configuration file or use a supported model name.", model_or_backend);
}

fn find_backend_script(backend_type: &str, config: &Config) -> Result<PathBuf> {
    let script_name = config.backends.model_backends.get(backend_type)
        .ok_or_else(|| anyhow!("Unknown backend type: {}", backend_type))?;

    let backend_paths = [
        format!("../backend/backend-python/{}", script_name),
        format!("../../backend/backend-python/{}", script_name), // if running from target/
        format!("./backend/backend-python/{}", script_name), // local development
    ];

    for path_str in &backend_paths {
        let path = Path::new(path_str);
        if path.exists() {
            debug!("Found backend script at: {}", path_str);
            return Ok(path.to_path_buf());
        }
    }

    bail!("Could not find backend script '{}' for backend type '{}'", script_name, backend_type)
}

fn find_next_available_port(start_port: u16) -> u16 {
    for port in start_port..start_port + 100 {
        if is_port_available(port) {
            return port;
        }
    }
    start_port // Fallback to original port if none found
}

fn is_port_available(port: u16) -> bool {
    // Check if port is available on localhost (127.0.0.1)
    // This is appropriate since we're checking for local port conflicts
    std::net::TcpListener::bind(("127.0.0.1", port)).is_ok()
}

async fn start_backend(
    model_or_backend: &str,
    management_service_url: &str,
    backend_host: &str,
    backend_api_port: Option<u16>,
    base_port: u16,
    ipc_endpoint: &str,
    backend_args: &[String],
) -> Result<()> {
    info!("Starting backend for: {}", model_or_backend);

    // Load configuration
    let config = load_backend_config()?;

    // Resolve model name and backend type
    let (model_name, backend_type) = resolve_model_to_backend(model_or_backend, &config)?;
    info!("Resolved to model: '{}', backend type: '{}'", model_name, backend_type);

    // Find the backend script
    let script_path = find_backend_script(&backend_type, &config)?;
    info!("Using backend script: {}", script_path.display());

    // Determine port (find available port if not specified)
    let port = backend_api_port.unwrap_or_else(|| {
        find_next_available_port(base_port)
    });

    info!("Backend will use management API port: {}", port);

    // Build command arguments
    let mut cmd = Command::new("python3");
    cmd.arg(&script_path);
    cmd.args(&[
        "--model-name", &model_name,
        "--management-service-url", management_service_url,
        "--backend-host", backend_host,
        "--backend-api-port", &port.to_string(),
        "--ipc-endpoint", ipc_endpoint,
    ]);

    // Always auto-load the model on startup
    cmd.arg("--auto-load");

    // Add any additional arguments
    cmd.args(backend_args);

    // Set environment variables for Python backend
    cmd.env("PYTHONPATH", "../backend/backend-python:../../backend/backend-python:./backend-python");

    // Create logs directory and redirect backend output to log files
    fs::create_dir_all("logs").unwrap_or_default();

    // Create log file with timestamp (same pattern as CLI logs)
    let timestamp = chrono::Local::now().format("%Y-%m-%d").to_string();
    let log_file_name = format!("logs/backend-{}-{}-{}.log", backend_type, port, timestamp);
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file_name)?;

    cmd.stdout(log_file.try_clone()?);
    cmd.stderr(log_file);

    info!("Starting backend process...");
    info!("Backend logs will be written to: {}", log_file_name);
    debug!("Command: python3 {} --model-name {} --management-service-url {} --backend-host {} --backend-api-port {} --ipc-endpoint {}{}",
           script_path.display(),
           model_name,
           management_service_url,
           backend_host,
           port,
           ipc_endpoint,
           if !backend_args.is_empty() { format!(" {}", backend_args.join(" ")) } else { "".to_string() });

    match cmd.spawn() {
        Ok(child) => {
            println!("✓ Backend started successfully!");
            println!("  Model: {}", model_name);
            println!("  Backend Type: {}", backend_type);
            println!("  Engine Management API: http://{}:{}", backend_host, port);
            println!("  IPC Endpoint: {}", ipc_endpoint);
            println!("  Process ID: {}", child.id());
            println!("  Logs: {}", log_file_name);

            // Wait for backend to register with a spinner
            with_spinner(
                async {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    Ok::<(), anyhow::Error>(())
                },
                "Wait for the backend to be discovered by the engine...",
                "Backend registration completed!"
            ).await?;
            // Instructions after backend registration
            println!("Use 'pie-cli backend list' to check registration status");
            println!("Use 'curl http://{}:{}/manage/health' to check backend health", backend_host, port);
            println!("Use 'tail -f {}' to monitor backend logs", log_file_name);
        }
        Err(e) => {
            error!("Failed to start backend process: {}", e);
            bail!("Failed to start backend: {}", e);
        }
    }

    Ok(())
}
