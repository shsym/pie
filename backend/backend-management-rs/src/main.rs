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
    // Initialize tracing
    tracing_subscriber::fmt::init();

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

    // Create and start the service
    match ManagementServiceImpl::create(config_path, backend_path) {
        Ok(mut service) => {
            if let Err(e) = service.start().await {
                error!("Failed to start service: {}", e);
                std::process::exit(1);
            }
            
            // Keep the service running until shutdown signal
            while service.is_running() {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            
            info!("Service shutdown detected, stopping...");
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
