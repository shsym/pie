use clap::Subcommand;
use anyhow::{Result, anyhow, bail, Context};
use std::process::{Command, Stdio};
use std::time::Duration;
use std::path::{Path, PathBuf};
use std::fs::{OpenOptions, create_dir_all};
use tokio::time::sleep;
use tracing::{info, warn, debug};
use serde_json::Value;
use crate::spinner::{with_spinner, with_dynamic_spinner};
use crate::constants::{network, spinner as spinner_constants};
use crate::config::Config;

/// Resolve a path to an absolute path
fn resolve_absolute_path<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
    let path = path.as_ref();
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        let current_dir = std::env::current_dir()
            .context("Failed to get current directory")?;
        Ok(current_dir.join(path))
    }
}

#[derive(Subcommand)]
pub enum ControllerCommands {
    /// Start the pie controller and its child processes (engine-manager and engine)
    Start {
        /// Host address for the controller services
        #[arg(long, default_value = network::DEFAULT_HOST)]
        host: String,

        /// Port for the engine management service
        #[arg(long, default_value_t = network::DEFAULT_HTTP_PORT)]
        port: u16,

        /// Engine WebSocket port
        #[arg(long, default_value_t = network::DEFAULT_GRPC_PORT)]
        engine_port: u16,
    },

    /// Check the status of controller and managed processes
    Status {
        /// Host address for the controller services
        #[arg(long, default_value = network::DEFAULT_HOST)]
        host: String,

        /// Port for the engine management service
        #[arg(long, default_value_t = network::DEFAULT_HTTP_PORT)]
        port: u16,
    },

    /// Stop the controller and all managed processes
    Stop {
        /// Host address for the controller services
        #[arg(long, default_value = network::DEFAULT_HOST)]
        host: String,

        /// Port for the engine management service
        #[arg(long, default_value_t = network::DEFAULT_HTTP_PORT)]
        port: u16,
    },
}

pub async fn handle_command(cmd: ControllerCommands) -> Result<()> {
    // Load the unified configuration
    let config = Config::load_default()
        .context("Failed to load configuration file. Please ensure config.json exists in the project root.")?;

    match cmd {
        ControllerCommands::Start { host, port, engine_port } => {
            start_controller(&config, &host, port, engine_port).await
        }
        ControllerCommands::Status { host, port } => {
            check_status(&host, port).await
        }
        ControllerCommands::Stop { host, port } => {
            stop_controller(&host, port).await
        }
    }
}

async fn start_engine_via_manager(management_url: &str) -> Result<()> {
    let client = reqwest::Client::new();
    let start_url = format!("{}/controller/start", management_url);

    let response = client.post(&start_url)
        .send()
        .await
        .context("Failed to send start engine request")?;

    if response.status().is_success() {
        let response_json: Value = response.json().await
            .context("Failed to parse start engine response")?;

        if let Some(status) = response_json.get("status").and_then(|s| s.as_str()) {
            if status == "started" || status == "already_running" {
                debug!("Engine start response: {:?}", response_json);
                return Ok(());
            }
        }

        bail!("Engine start returned unexpected status: {:?}", response_json);
    } else {
        let status_code = response.status();
        let error_text = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
        bail!("Failed to start engine: HTTP {} - {}", status_code, error_text);
    }
}

async fn start_controller(config: &Config, host: &str, port: u16, _engine_port: u16) -> Result<()> {
    // Check if engine-manager is already running
    if check_engine_manager_health(&format!("http://{}:{}", host, port)).await {
        println!("✓ Engine-manager is already running at http://{}:{}", host, port);
        println!("  Use 'pie-cli controller status' to check status");
        return Ok(());
    }

    // Start the spinner immediately and run the entire startup process
    let startup_future = async {
        // Start engine-manager process in detached mode
        start_engine_manager_detached(config, host, port).await?;

        // Wait for engine-manager to start with a spinning indicator
        // Instead of a single 2-second sleep, we'll do smaller sleeps
        // and update the spinner message to show progress
        for i in 1..=spinner_constants::CONTROLLER_STARTUP_CYCLES {
            tokio::time::sleep(spinner_constants::STARTUP_SLEEP_DURATION).await;

            // The spinner will keep spinning during this time automatically
            // because it's running in the tokio::select! loop

            // After 1 second, we can start trying health checks
            if i >= spinner_constants::HEALTH_CHECK_START_CYCLES {
                let management_url = format!("http://{}:{}", host, port);
                if check_engine_manager_health(&management_url).await {
                    return Ok(());
                }
            }
        }

        // If we get here, do additional health check retries
        let management_url = format!("http://{}:{}", host, port);
        for attempt in 1..=5 {
            if check_engine_manager_health(&management_url).await {
                return Ok(());
            }
            if attempt < 5 {
                sleep(Duration::from_millis(500)).await;
            }
        }

        Err(anyhow!("Engine-manager did not become healthy within reasonable time"))
    };

    // Run with dynamic spinner that shows progress during waiting
    let message_updates = vec![
        (Duration::from_millis(500), "Starting engine-manager process...".to_string()),
        (Duration::from_millis(1000), "Waiting for service initialization...".to_string()),
        (Duration::from_millis(1500), "Checking service health...".to_string()),
    ];

    with_dynamic_spinner(
        startup_future,
        &format!("Starting engine-manager on {}:{}...", host, port),
        "Engine-manager started successfully!",
        message_updates
    ).await?;

    info!("Engine-manager started successfully");

    // Now start the engine process through the engine-manager
    let start_engine_future = async {
        let management_url = format!("http://{}:{}", host, port);
        start_engine_via_manager(&management_url).await
    };

    with_spinner(
        start_engine_future,
        "Starting engine process...",
        "Engine started successfully!"
    ).await?;

    info!("Pie controller started successfully!");

    println!("🚀 Engine management service available at: http://{}:{}", host, port);
    println!("🚀 Engine process started and available via WebSocket");
    println!("   Use 'pie-cli controller status' to check status");
    println!("   Use 'pie-cli controller stop' to stop all services");

    Ok(())
}

fn find_binary_from_config(binary_name: &str, search_paths: &[String]) -> Result<PathBuf> {
    // First, try to find it in PATH (works when installed/bundled)
    if let Ok(output) = Command::new("which").arg(binary_name).output() {
        if output.status.success() {
            let path_output = String::from_utf8_lossy(&output.stdout);
            let path_str = path_output.trim();
            if !path_str.is_empty() {
                let absolute_path = resolve_absolute_path(path_str)?;
                info!("Found {} in PATH: {}", binary_name, absolute_path.display());
                return Ok(absolute_path);
            }
        }
    }

    // Try the configured search paths
    for path_str in search_paths {
        let path = Path::new(path_str);
        if path.exists() {
            let absolute_path = resolve_absolute_path(path)?;
            info!("Found {} at configured path: {}", binary_name, absolute_path.display());
            return Ok(absolute_path);
        }
    }

    bail!("Could not find {} binary. Please ensure it's compiled or in PATH.", binary_name)
}

fn find_engine_manager_binary(config: &Config) -> Result<PathBuf> {
    find_binary_from_config(
        &config.services.engine_manager.binary_name,
        &config.paths.engine_manager_binary_search
    )
}


async fn start_engine_manager_detached(config: &Config, host: &str, port: u16) -> Result<()> {
    let binary_path = find_engine_manager_binary(config)?;

    // Log the exact binary we're about to start
    info!("Starting engine-manager from: {}", binary_path.display());

    // Create logs directory if it doesn't exist
    create_dir_all(&config.logging.directory).unwrap_or_default();

    // Create log file for engine-manager
    let timestamp = chrono::Local::now().format("%Y-%m-%d").to_string();
    let log_file_name = format!("{}/engine-manager-{}-{}.log", config.logging.directory, port, timestamp);
    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_file_name)?;

    // Resolve absolute config path to pass to engine-manager
    let config_path = resolve_absolute_path("config.json")?;

    let mut cmd = Command::new(&binary_path);
    cmd.args(&[
        "--port", &port.to_string(),
        "--config", &config_path.to_string_lossy(),
        "--no-color",  // Disable ANSI colors for clean log files
    ]);

    // Add bind-all if not localhost
    if host != "127.0.0.1" && host != "localhost" {
        cmd.arg("--bind-all");
    }

    // Redirect output to log file
    cmd.stdout(Stdio::from(log_file.try_clone()?))
       .stderr(Stdio::from(log_file))
       .stdin(Stdio::null());

    info!("Starting engine-manager from: {}", binary_path.display());
    info!("Engine-manager logs will be written to: {}", log_file_name);

    let child = cmd.spawn()
        .map_err(|e| anyhow!("Failed to start engine-manager from {}: {}", binary_path.display(), e))?;

    info!("Engine-manager started with PID: {}", child.id());

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
                        if let Some(engine_info) = controller.get("engine_process") {
                            if let Some(running) = engine_info.get("running") {
                                if running.as_bool().unwrap_or(false) {
                                    println!("  - Engine process: Running");
                                    if let Some(port) = engine_info.get("port") {
                                        println!("    Port: {}", port);
                                    }
                                    if let Some(url) = engine_info.get("url") {
                                        println!("    WebSocket URL: {}", url.as_str().unwrap_or("Unknown"));
                                    }
                                } else {
                                    println!("  - Engine process: Not running");
                                }
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
    let shutdown_future = async {
        info!("Stopping Pie controller...");

        let client = reqwest::Client::new();
        let base_url = format!("http://{}:{}", host, port);
        let shutdown_url = format!("{}/shutdown", base_url);

        match client.post(&shutdown_url).send().await {
            Ok(response) if response.status().is_success() => {
                match response.json::<Value>().await {
                    Ok(result) => {
                        let message = result.get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("Engine-manager service is shutting down");
                        Ok(message.to_string())
                    }
                    Err(_) => {
                        Ok("Engine-manager service is shutting down".to_string())
                    }
                }
            }
            Ok(response) => {
                Err(anyhow!("Failed to shutdown: HTTP {}", response.status()))
            }
            Err(e) => {
                Err(anyhow!("Cannot connect to engine-manager at {}: {}", base_url, e))
            }
        }
    };

    // Run with spinner
    with_spinner(
        shutdown_future,
        "Stopping Pie controller...",
        ""
    ).await?;

    info!("Pie controller stop command completed");
    Ok(())
}
