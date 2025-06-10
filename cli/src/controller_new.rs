use clap::Subcommand;
use anyhow::{Result, anyhow, bail};
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, error, warn, debug};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Subcommand)]
pub enum ControllerCommands {
    /// Start the pie controller and its child processes (engine-manager and engine)
    Start {
        /// Host address for the controller services
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port for the engine management service
        #[arg(long, default_value = "8080")]
        port: u16,

        /// Engine WebSocket port
        #[arg(long, default_value = "9123")]
        engine_port: u16,
    },

    /// Check the status of controller and managed processes
    Status {
        /// Host address for the controller services
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port for the engine management service
        #[arg(long, default_value = "8080")]
        port: u16,
    },

    /// Stop the controller and all managed processes
    Stop {
        /// Host address for the controller services
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port for the engine management service
        #[arg(long, default_value = "8080")]
        port: u16,
    },
}

pub async fn handle_command(cmd: ControllerCommands) -> Result<()> {
    match cmd {
        ControllerCommands::Start { host, port, engine_port } => {
            start_controller(&host, port, engine_port).await
        }
        ControllerCommands::Status { host, port } => {
            check_status(&host, port).await
        }
        ControllerCommands::Stop { host, port } => {
            stop_controller(&host, port).await
        }
    }
}

async fn start_controller(host: &str, port: u16, _engine_port: u16) -> Result<()> {
    info!("Starting Pie controller...");

    // Check if engine-manager is already running
    if is_engine_manager_running(host, port).await {
        warn!("Engine-manager is already running on {}:{}", host, port);
        return Ok(());
    }

    // Start engine-manager process in detached mode
    info!("Starting engine-manager on {}:{}", host, port);
    start_engine_manager_detached(host, port)?;

    // Wait a bit for engine-manager to start
    sleep(Duration::from_secs(2)).await;

    // Verify engine-manager is responding
    if !is_engine_manager_running(host, port).await {
        error!("Engine-manager failed to start properly");
        bail!("Failed to start engine-manager");
    }

    info!("Engine-manager started successfully");

    // Start the controller engine process via HTTP API
    start_controller_engine(host, port).await?;

    info!("Pie controller started successfully!");
    info!("Engine management service available at: http://{}:{}", host, port);
    info!("Use 'pie-cli controller status' to check status");
    info!("Use 'pie-cli controller stop' to stop all services");

    Ok(())
}

async fn check_status(host: &str, port: u16) -> Result<()> {
    let client = reqwest::Client::new();
    let status_url = format!("http://{}:{}/controller/status", host, port);

    match client.get(&status_url).send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<Value>().await {
                Ok(status) => {
                    println!("Pie Controller Status:");
                    println!("=====================");
                    println!("Engine Manager: Running");
                    println!("  - Service: Healthy");
                    println!("  - Timestamp: {}", status["timestamp"].as_str().unwrap_or("Unknown"));

                    // Get backends status
                    let backends_url = format!("http://{}:{}/backends", host, port);
                    match client.get(&backends_url).send().await {
                        Ok(response) if response.status().is_success() => {
                            match response.json::<Value>().await {
                                Ok(backends) => {
                                    if let Some(backends_array) = backends.as_array() {
                                        println!("  - Registered backends: {}", backends_array.len());
                                    }
                                }
                                Err(_) => println!("  - Registered backends: Unknown"),
                            }
                        }
                        _ => println!("  - Registered backends: Unable to fetch"),
                    }

                    // Controller engine status
                    if let Some(controller) = status["controller"].as_object() {
                        let engine_running = controller["engine_process_running"].as_bool().unwrap_or(false);
                        println!("Engine Process: {}", if engine_running { "Running" } else { "Not running" });
                    } else {
                        println!("Engine Process: Unknown");
                    }
                }
                Err(e) => {
                    error!("Failed to parse status response: {}", e);
                    bail!("Invalid status response from engine-manager");
                }
            }
        }
        Ok(response) => {
            println!("Engine Manager: Not responding (HTTP {})", response.status());
        }
        Err(_) => {
            println!("Controller is not running");
        }
    }

    Ok(())
}

async fn stop_controller(host: &str, port: u16) -> Result<()> {
    info!("Stopping Pie controller...");

    let client = reqwest::Client::new();
    let stop_url = format!("http://{}:{}/controller/stop", host, port);

    match client.post(&stop_url).send().await {
        Ok(response) if response.status().is_success() => {
            info!("Controller stopped successfully");
        }
        Ok(response) => {
            warn!("Stop request returned status: {}", response.status());
        }
        Err(_) => {
            warn!("Could not connect to engine-manager, it may already be stopped");
        }
    }

    // Try to kill the engine-manager process if it's still running
    // This is a fallback in case the HTTP stop didn't work
    if is_engine_manager_running(host, port).await {
        warn!("Engine-manager still running, attempting to kill process...");
        let _ = Command::new("pkill")
            .args(&["-f", "pie_engine_manager"])
            .output();
    }

    info!("Pie controller stopped");
    Ok(())
}

async fn is_engine_manager_running(host: &str, port: u16) -> bool {
    let client = reqwest::Client::new();
    let health_url = format!("http://{}:{}/health", host, port);

    match client.get(&health_url).send().await {
        Ok(response) if response.status().is_success() => true,
        _ => false,
    }
}

async fn start_controller_engine(host: &str, port: u16) -> Result<()> {
    let client = reqwest::Client::new();
    let start_url = format!("http://{}:{}/controller/start", host, port);

    match client.post(&start_url).send().await {
        Ok(response) if response.status().is_success() => {
            info!("Controller engine process started");
            Ok(())
        }
        Ok(response) => {
            warn!("Engine start request returned status: {}", response.status());
            Ok(()) // Non-fatal for now since engine startup is Phase 3
        }
        Err(e) => {
            error!("Failed to start controller engine: {}", e);
            bail!("Could not start controller engine process");
        }
    }
}

fn find_engine_manager_binary() -> Result<PathBuf> {
    // First, try to find it in PATH (works when installed/bundled)
    if let Ok(output) = Command::new("which").arg("pie_engine_manager").output() {
        if output.status.success() {
            let path_output = String::from_utf8_lossy(&output.stdout);
            let path_str = path_output.trim();
            if !path_str.is_empty() {
                debug!("Found pie_engine_manager in PATH: {}", path_str);
                return Ok(PathBuf::from(path_str));
            }
        }
    }

    // Fallback: try relative paths for development
    let relative_paths = [
        "../engine-manager/target/release/pie_engine_manager",
        "../engine-manager/target/debug/pie_engine_manager",
        "../../engine-manager/target/release/pie_engine_manager", // if running from cli/target/
        "../../engine-manager/target/debug/pie_engine_manager",
    ];

    for path_str in &relative_paths {
        let path = Path::new(path_str);
        if path.exists() {
            debug!("Found pie_engine_manager at relative path: {}", path_str);
            return Ok(path.to_path_buf());
        }
    }

    bail!("Could not find pie_engine_manager binary. Please ensure it's compiled or in PATH.")
}

fn start_engine_manager_detached(host: &str, port: u16) -> Result<()> {
    let binary_path = find_engine_manager_binary()?;

    let mut cmd = Command::new(&binary_path);
    cmd.args(&[
        "--port", &port.to_string(),
    ]);

    // Add bind-all if not localhost
    if host != "127.0.0.1" && host != "localhost" {
        cmd.arg("--bind-all");
    }

    // Detach the process completely
    cmd.stdout(Stdio::null())
       .stderr(Stdio::null())
       .stdin(Stdio::null());

    info!("Starting engine-manager from: {}", binary_path.display());

    let _child = cmd.spawn()
        .map_err(|e| anyhow!("Failed to start engine-manager from {}: {}", binary_path.display(), e))?;

    // Don't wait for the child - let it run detached
    debug!("Engine-manager started in detached mode");

    Ok(())
}
