//! Symphony Management Service - Rust Implementation
//!
//! A long-running daemon that manages backend model instances and handles
//! client handshakes, providing dynamic routing to model-specific endpoints.

use backend_management_rs::service::{ManagementServiceFactory, ManagementServiceTrait};
use backend_management_rs::ManagementServiceImpl;
use std::path::PathBuf;
use tracing::{error, info};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with file output to avoid terminal interference
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("symphony-service.log")
        .expect("Failed to create log file");
    
    tracing_subscriber::fmt()
        .with_writer(log_file)
        .with_ansi(false)
        .init();
    
    println!("Symphony Management Service starting... Logs will be written to symphony-management.log");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let mut config_path: Option<PathBuf> = None;
    let mut backend_path: Option<PathBuf> = None;

    // Simple argument parsing (can be improved with clap later)
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                if i + 1 < args.len() {
                    config_path = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    eprintln!("Error: --config requires a path argument");
                    std::process::exit(1);
                }
            }
            "--backend-path" => {
                if i + 1 < args.len() {
                    backend_path = Some(PathBuf::from(&args[i + 1]));
                    i += 2;
                } else {
                    eprintln!("Error: --backend-path requires a path argument");
                    std::process::exit(1);
                }
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {
                eprintln!("Error: Unknown argument: {}", args[i]);
                print_help();
                std::process::exit(1);
            }
        }
    }

    info!("Starting Symphony Management Service");
    
    // Clean up any leftover IPC sockets from previous runs
    backend_management_rs::cleanup_all_symphony_sockets();

    // Set up signal handling for the main process
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);
    
    // Spawn signal handler task
    let shutdown_tx_signal = shutdown_tx.clone();
    tokio::spawn(async move {
        setup_main_signal_handlers(shutdown_tx_signal).await;
    });

    // Create and start the service
    match ManagementServiceImpl::create(config_path, backend_path) {
        Ok(mut service) => {
            if let Err(e) = service.start().await {
                error!("Failed to start service: {}", e);
                std::process::exit(1);
            }
            
            // Wait for either the service to stop naturally or a shutdown signal
            tokio::select! {
                _ = async {
                    while service.is_running() {
                        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    }
                } => {
                    // Service stopped naturally
                }
                _ = shutdown_rx.recv() => {
                    // Received shutdown signal
                    println!("Shutting down... (check symphony-management.log for details)");
                    info!("Service shutdown detected, stopping...");
                }
            }
            
            // Ensure clean shutdown
            if let Err(e) = service.stop().await {
                error!("Error during shutdown: {}", e);
            }
        }
        Err(e) => {
            error!("Failed to create service: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}

async fn setup_main_signal_handlers(shutdown_tx: tokio::sync::mpsc::Sender<()>) {
    use tokio::signal::unix::{signal, SignalKind};
    
    let mut sigterm = signal(SignalKind::terminate()).expect("Failed to set up SIGTERM handler");
    let mut sigint = signal(SignalKind::interrupt()).expect("Failed to set up SIGINT handler");
    
    tokio::select! {
        _ = sigterm.recv() => {
            println!("Received SIGTERM, shutting down...");
            info!("Received SIGTERM, initiating graceful shutdown");
            let _ = shutdown_tx.send(()).await;
        }
        _ = sigint.recv() => {
            println!("Received SIGINT (Ctrl+C), shutting down...");
            info!("Received SIGINT (Ctrl+C), initiating graceful shutdown");
            let _ = shutdown_tx.send(()).await;
        }
    }
}

fn print_help() {
    println!("Symphony Management Service");
    println!();
    println!("USAGE:");
    println!("    backend-management-rs [OPTIONS]");
    println!();
    println!("OPTIONS:");
    println!("    --config <PATH>        Path to configuration file");
    println!("    --backend-path <PATH>  Path to backend scripts directory");
    println!("    -h, --help             Print this help message");
}
