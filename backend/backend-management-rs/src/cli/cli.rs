//! CLI command definitions and parsing logic.

use clap::Parser;
use super::zmq_client;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct CliArgs {
    #[clap(subcommand)]
    pub command: Commands,
    
    /// Output responses in JSON format
    #[clap(long, global = true)]
    pub json: bool,
}

#[derive(Parser, Debug)]
pub enum Commands {
    /// Start the management service (daemon)
    StartService,
    /// Stop the management service
    StopService,
    /// Get the status of the management service
    Status,
    /// Load a model
    LoadModel {
        model_name: String,
        #[clap(long)]
        config_path: Option<String>,
    },
    /// Unload a model
    UnloadModel {
        model_name: String,
    },
    /// List loaded models
    ListModels,
    /// Install a model from HuggingFace Hub
    InstallModel {
        /// Model name or path on HuggingFace Hub (e.g., meta-llama/Llama-3.1-8B-Instruct)
        model_name: String,
        /// Local name to use for the model (optional, defaults to last part of model_name)
        #[clap(long)]
        local_name: Option<String>,
        /// Force reinstall even if model already exists
        #[clap(long, action)]
        force: bool,
    },
    /// Uninstall a model from local storage
    UninstallModel {
        /// Model name to uninstall
        model_name: String,
        /// Force uninstall even if model is currently loaded
        #[clap(long, action)]
        force: bool,
    },
    /// Transform an installed model (run index and model transformations)
    TransformModel {
        /// Model name to transform (must be already installed)
        model_name: String,
        /// Force transformation even if already processed
        #[clap(long, action)]
        force: bool,
    },
    /// Show service logs
    Log {
        /// Number of lines to show from the end of the log file
        #[clap(short, long, default_value = "50")]
        lines: usize,
        /// Follow the log file for new entries (like tail -f)
        #[clap(short, long, action)]
        follow: bool,
        /// Model name to show logs for (if not specified, shows management service logs)
        #[clap(short, long)]
        model: Option<String>,
        /// List available backend log files
        #[clap(long, action)]
        list: bool,
    },
}

pub async fn process_cli_command(args: CliArgs) {
    match args.command {
        Commands::StartService => {
            handle_start_service(args.json).await;
        }
        Commands::Log { lines, follow, model, list } => {
            handle_log_command(lines, follow, model, list, args.json).await;
        }
        other_command => {
            // Use ZMQ client for all other commands
            match zmq_client::send_command_to_service(other_command, args.json).await {
                Ok(response) => println!("{}", response),
                Err(e) => {
                    if args.json {
                        println!("{}", serde_json::json!({"error": e}));
                    } else {
                        eprintln!("Error: {}", e);
                    }
                }
            }
        }
    }
}

async fn handle_start_service(json: bool) {
    // First check if service is already running
    match zmq_client::send_command_to_service(Commands::Status, json).await {
        Ok(_) => {
            if json {
                println!("{}", serde_json::json!({"status": "already_running", "message": "Service is already running"}));
            } else {
                println!("Service is already running.");
            }
            return;
        }
        Err(_) => {
            // Service is not running, we can try to start it
        }
    }
    
    // Find the symphony-management-service binary
    let binary_path = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.join("symphony-management-service")))
        .filter(|path| path.exists())
        .unwrap_or_else(|| std::path::PathBuf::from("./target/release/symphony-management-service"));
    
    if !binary_path.exists() {
        let error_msg = format!("Could not find symphony-management-service binary at: {}", binary_path.display());
        if json {
            println!("{}", serde_json::json!({"status": "error", "message": error_msg}));
        } else {
            eprintln!("Error: {}", error_msg);
        }
        return;
    }
    
    // Start the service (always daemonized)
    let mut command = std::process::Command::new(&binary_path);
    
    // Always run as daemon - redirect output to null
    command.stdout(std::process::Stdio::null())
           .stderr(std::process::Stdio::null())
           .stdin(std::process::Stdio::null());
    
    match command.spawn() {
        Ok(child) => {
            // For daemonized mode, we don't wait for the child
            if json {
                println!("{}", serde_json::json!({
                    "status": "started", 
                    "message": "Service started as daemon",
                    "pid": child.id()
                }));
            } else {
                println!("✓ Service started as daemon (PID: {})", child.id());
                println!("Logs are written to symphony-service.log");
            }
        }
        Err(e) => {
            let error_msg = format!("Failed to start service: {}", e);
            if json {
                println!("{}", serde_json::json!({"status": "error", "message": error_msg}));
            } else {
                eprintln!("Error: {}", error_msg);
            }
        }
    }
}

async fn handle_log_command(lines: usize, follow: bool, model: Option<String>, list: bool, json: bool) {
    if list {
        list_available_logs(json).await;
        return;
    }

    let log_file_path = if let Some(ref model_name) = model {
        // Show backend logs for specific model
        std::path::Path::new(&format!("symphony-backend-{}.log", model_name)).to_path_buf()
    } else {
        // Show management service logs (default)
        std::path::Path::new("symphony-service.log").to_path_buf()
    };
    
    if !log_file_path.exists() {
        let error_msg = if model.is_some() {
            format!("Backend log file '{}' not found. Is the model loaded?", log_file_path.display())
        } else {
            "Management service log file 'symphony-service.log' not found. Is the service running?".to_string()
        };
        
        if json {
            println!("{}", serde_json::json!({"error": error_msg}));
        } else {
            eprintln!("Error: {}", error_msg);
        }
        return;
    }

    if follow {
        // Implement tail -f functionality
        if let Err(e) = tail_follow_log(&log_file_path, lines, json).await {
            if json {
                println!("{}", serde_json::json!({"error": format!("Failed to follow log: {}", e)}));
            } else {
                eprintln!("Error: Failed to follow log: {}", e);
            }
        }
    } else {
        // Show last N lines
        match read_last_lines(&log_file_path, lines).await {
            Ok(log_lines) => {
                if json {
                    println!("{}", serde_json::json!({
                        "lines": log_lines,
                        "count": log_lines.len(),
                        "file": log_file_path.display().to_string()
                    }));
                } else {
                    if model.is_some() {
                        println!("=== Backend logs for model '{}' ===", model.as_ref().unwrap());
                    } else {
                        println!("=== Management service logs ===");
                    }
                    for line in log_lines {
                        println!("{}", line);
                    }
                }
            }
            Err(e) => {
                if json {
                    println!("{}", serde_json::json!({"error": format!("Failed to read log: {}", e)}));
                } else {
                    eprintln!("Error: Failed to read log: {}", e);
                }
            }
        }
    }
}

async fn list_available_logs(json: bool) {
    use std::fs;
    
    let mut log_files = Vec::new();
    
    // Add management service log if it exists
    if std::path::Path::new("symphony-service.log").exists() {
        log_files.push(("management".to_string(), "symphony-service.log".to_string()));
    }
    
    // Find backend log files
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            if let Some(file_name) = entry.file_name().to_str() {
                if file_name.starts_with("symphony-backend-") && file_name.ends_with(".log") {
                    // Extract model name from filename
                    let model_name = file_name
                        .strip_prefix("symphony-backend-")
                        .and_then(|s| s.strip_suffix(".log"))
                        .unwrap_or("unknown");
                    log_files.push((model_name.to_string(), file_name.to_string()));
                }
            }
        }
    }
    
    if json {
        println!("{}", serde_json::json!({
            "available_logs": log_files.iter().map(|(model, file)| {
                serde_json::json!({
                    "model": model,
                    "file": file
                })
            }).collect::<Vec<_>>()
        }));
    } else {
        if log_files.is_empty() {
            println!("No log files found.");
        } else {
            println!("Available log files:");
            for (model, file) in log_files {
                if model == "management" {
                    println!("  {} (management service): {}", model, file);
                } else {
                    println!("  {} (backend): {}", model, file);
                }
            }
            println!("\nUsage:");
            println!("  symphony-management log                    # Show management service logs");
            println!("  symphony-management log -m <model_name>   # Show backend logs for specific model");
            println!("  symphony-management log --follow          # Follow logs in real-time");
        }
    }
}

async fn read_last_lines(log_path: &std::path::Path, num_lines: usize) -> Result<Vec<String>, std::io::Error> {
    use std::io::{BufRead, BufReader};
    use std::fs::File;
    use std::collections::VecDeque;

    let file = File::open(log_path)?;
    let reader = BufReader::new(file);
    
    let mut lines = VecDeque::new();
    for line in reader.lines() {
        let line = line?;
        lines.push_back(line);
        if lines.len() > num_lines {
            lines.pop_front();
        }
    }
    
    Ok(lines.into_iter().collect())
}

async fn tail_follow_log(log_path: &std::path::Path, initial_lines: usize, json: bool) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{BufRead, BufReader, Seek, SeekFrom};
    use std::fs::File;
    use tokio::time::{sleep, Duration};

    // First, show the last N lines
    if let Ok(lines) = read_last_lines(log_path, initial_lines).await {
        if json {
            println!("{}", serde_json::json!({
                "type": "initial",
                "lines": lines
            }));
        } else {
            for line in lines {
                println!("{}", line);
            }
        }
    }

    // Then follow the file
    let mut file = File::open(log_path)?;
    file.seek(SeekFrom::End(0))?;
    let mut reader = BufReader::new(file);

    if !json {
        println!("--- Following log (Press Ctrl+C to exit) ---");
    }

    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => {
                // No new data, sleep and try again
                sleep(Duration::from_millis(100)).await;
            }
            Ok(_) => {
                // Remove trailing newline for consistent formatting
                if line.ends_with('\n') {
                    line.pop();
                }
                
                if json {
                    println!("{}", serde_json::json!({
                        "type": "new_line",
                        "line": line
                    }));
                } else {
                    println!("{}", line);
                }
            }
            Err(e) => {
                if json {
                    println!("{}", serde_json::json!({"error": format!("Error reading log: {}", e)}));
                } else {
                    eprintln!("Error reading log: {}", e);
                }
                break;
            }
        }
    }

    Ok(())
}
