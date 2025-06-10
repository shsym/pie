use clap::Subcommand;
use anyhow::{Result, anyhow, bail};
use std::process::{Command, Stdio};
use std::time::Duration;
use std::path::{Path, PathBuf};
use tokio::time::sleep;
use tracing::{info, error, warn, debug};
use serde_json::Value;

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
    if check_engine_manager_health(&format!("http://{}:{}", host, port)).await {
        println!("Engine-manager is already running at http://{}:{}", host, port);
        println!("Use 'pie-cli controller status' to check status");
        return Ok(());
    }

    // Start engine-manager process in detached mode
    info!("Starting engine-manager on {}:{}", host, port);
    start_engine_manager_detached(host, port)?;

    // Wait a bit for engine-manager to start
    sleep(Duration::from_secs(2)).await;

    // Verify engine-manager is responding
    if !check_engine_manager_health(&format!("http://{}:{}", host, port)).await {
        error!("Engine-manager failed to start properly");
        bail!("Failed to start engine-manager");
    }

    info!("Engine-manager started successfully");

    info!("Pie controller started successfully!");
    info!("Engine management service available at: http://{}:{}", host, port);
    info!("Use 'pie-cli controller status' to check status");
    info!("Use 'pie-cli controller stop' to stop all services");

    Ok(())
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

    // Detach the process - don't capture stdout/stderr
    cmd.stdout(Stdio::null())
       .stderr(Stdio::null())
       .stdin(Stdio::null());

    info!("Starting engine-manager from: {}", binary_path.display());

    cmd.spawn()
        .map_err(|e| anyhow!("Failed to start engine-manager from {}: {}", binary_path.display(), e))?;

    Ok(())
}

async fn check_engine_manager_health(url: &str) -> bool {
    let client = reqwest::Client::new();
    let health_url = format!("{}/health", url);

    for attempt in 1..=5 {
        match client.get(&health_url).send().await {
            Ok(response) if response.status().is_success() => {
                return true;
            }
            Ok(response) => {
                warn!("Health check attempt {}: got status {}", attempt, response.status());
            }
            Err(e) => {
                warn!("Health check attempt {}: {}", attempt, e);
            }
        }
        sleep(Duration::from_secs(1)).await;
    }

    false
}

async fn check_status(host: &str, port: u16) -> Result<()> {
    println!("Pie Controller Status:");
    println!("=====================");

    let client = reqwest::Client::new();
    let base_url = format!("http://{}:{}", host, port);
    let status_url = format!("{}/controller/status", base_url);

    match client.get(&status_url).send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<Value>().await {
                Ok(status) => {
                    println!("Engine Manager: Running");
                    println!("  - Service: Healthy");

                    if let Some(timestamp) = status.get("timestamp") {
                        println!("  - Last seen: {}", timestamp.as_str().unwrap_or("Unknown"));
                    }

                    if let Some(controller) = status.get("controller") {
                        if let Some(engine_running) = controller.get("engine_process_running") {
                            if engine_running.as_bool().unwrap_or(false) {
                                println!("  - Engine process: Running");
                            } else {
                                println!("  - Engine process: Not running");
                            }
                        }
                    }

                    // Get backends status
                    let backends_url = format!("{}/backends", base_url);
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
                }
                Err(e) => {
                    println!("Engine Manager: Error parsing status response: {}", e);
                }
            }
        }
        Ok(response) => {
            println!("Engine Manager: Service responding but not healthy (status: {})", response.status());
        }
        Err(_) => {
            println!("Engine Manager: Not running or not accessible at {}:{}", host, port);
        }
    }

    Ok(())
}

async fn stop_controller(host: &str, port: u16) -> Result<()> {
    info!("Stopping Pie controller...");

    let client = reqwest::Client::new();
    let base_url = format!("http://{}:{}", host, port);
    let shutdown_url = format!("{}/shutdown", base_url);

    match client.post(&shutdown_url).send().await {
        Ok(response) if response.status().is_success() => {
            match response.json::<Value>().await {
                Ok(result) => {
                    if let Some(message) = result.get("message") {
                        println!("{}", message.as_str().unwrap_or("Engine-manager is shutting down"));
                    } else {
                        println!("Engine-manager is shutting down");
                    }
                }
                Err(_) => {
                    println!("Engine-manager is shutting down");
                }
            }
        }
        Ok(response) => {
            warn!("Shutdown request returned status: {}", response.status());
            println!("Controller may not have stopped properly");
        }
        Err(e) => {
            warn!("Failed to send shutdown request: {}", e);
            println!("Controller may not be running or not accessible at {}:{}", host, port);
        }
    }

    info!("Pie controller stop command completed");
    Ok(())
}
