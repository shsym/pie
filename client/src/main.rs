mod client;

use anyhow::{Context, Result};
use clap::{Arg, Command, ArgMatches};
use colored::Colorize;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::time::sleep;

use client::{Client, hash_program};

// Define a simple macro for client-side logging
macro_rules! log_client {
    ($($arg:tt)*) => {
        println!("{}", format!("[Symphony Launcher] {}", format_args!($($arg)*)).bright_blue());
    }
}

/// Check if the Symphony engine is available
async fn check_engine_available(server_uri: &str) -> Result<()> {
    let timeout = Duration::from_secs(5);
    let start = std::time::Instant::now();

    while start.elapsed() < timeout {
        match Client::connect(server_uri).await {
            Ok(mut client) => {
                client.close().await?;
                return Ok(());
            }
            Err(_) => {
                sleep(Duration::from_millis(500)).await;
            }
        }
    }

    anyhow::bail!("Symphony engine is not available at {}", server_uri)
}

/// Run a WebAssembly program as an interactive instance
async fn run_interactive_instance(
    program_path: &Path,
    server_uri: &str,
    num_instances: usize,
) -> Result<()> {
    // Check if the file exists
    if !program_path.exists() {
        anyhow::bail!("Error: Program file not found at path: {:?}", program_path);
    }

    log_client!("Running program: {}", program_path.display());
    log_client!("Connecting to Symphony engine at: {}", server_uri);

    // Check if engine is available
    check_engine_available(server_uri).await
        .context("Failed to connect to Symphony engine")?;

    let mut client = Client::connect(server_uri).await?;
    let program_blob = fs::read(program_path)?;
    let program_hash = hash_program(&program_blob);

    log_client!("Program file hash: {}", program_hash);

    // If program is not present, upload it
    if !client.program_exists(&program_hash).await? {
        log_client!("Program not found on server, uploading now...");
        client.upload_program(&program_blob).await?;
        log_client!("Program uploaded successfully!");
    } else {
        log_client!("Program already exists on server");
    }

    // Launch instances
    let mut instances = Vec::new();
    for i in 0..num_instances {
        let instance = client.launch_instance(&program_hash).await?;
        log_client!("Instance {} launched (ID: {})", i + 1, instance.id());
        instances.push(instance);
    }

    // Spawn a task for each instance to handle sending and receiving concurrently
    let mut handles = Vec::new();
    for (i, mut instance) in instances.into_iter().enumerate() {
        let handle = tokio::spawn(async move {

            // Handle incoming events
            while let Ok((event, message)) = instance.recv().await {
                match event.as_str() {
                    "terminated" => {
                        log_client!("Instance {} terminated. Reason: {}", instance.id(), message);
                        break;
                    }
                    _ => {
                        log_client!("Instance {} received: {} - {}", instance.id(), event, message);
                    }
                }
            }
            anyhow::Result::<()>::Ok(())
        });
        handles.push(handle);
    }

    // Wait for all instance tasks to complete
    for handle in handles {
        handle.await??;
    }

    client.close().await?;
    log_client!("Client connection closed");
    Ok(())
}

/// Run a WebAssembly program as a server instance on a specific port
async fn run_server_instance(
    program_path: &Path,
    server_uri: &str,
    port: u16,
) -> Result<()> {
    // Check if the file exists
    if !program_path.exists() {
        anyhow::bail!("Error: Program file not found at path: {:?}", program_path);
    }

    log_client!("Running server program: {}", program_path.display());
    log_client!("Connecting to Symphony engine at: {}", server_uri);
    log_client!("Will bind server to port: {}", port);

    // Check if engine is available
    check_engine_available(server_uri).await
        .context("Failed to connect to Symphony engine")?;

    let mut client = Client::connect(server_uri).await?;
    let program_blob = fs::read(program_path)?;
    let program_hash = hash_program(&program_blob);

    log_client!("Program file hash: {}", program_hash);

    // If program is not present, upload it
    if !client.program_exists(&program_hash).await? {
        log_client!("Program not found on server, uploading now...");
        client.upload_program(&program_blob).await?;
        log_client!("Program uploaded successfully!");
    } else {
        log_client!("Program already exists on server");
    }

    // Launch server instance
    client.launch_server_instance(&program_hash, port as u32).await?;
    log_client!("Server instance launched on port {}", port);

    // Keep the client connected until interrupted
    log_client!("Server is running. Press Ctrl+C to stop...");
    tokio::signal::ctrl_c().await?;

    client.close().await?;
    log_client!("Client connection closed");
    Ok(())
}

/// Resolve program path with common extensions and search paths
fn resolve_program_path(program_name: &str) -> Result<PathBuf> {
    let mut candidates = Vec::new();

    // If it's already a path with extension, use it directly
    if program_name.ends_with(".wasm") {
        candidates.push(PathBuf::from(program_name));
    } else {
        // Try various common locations and extensions
        candidates.extend([
            PathBuf::from(format!("{}.wasm", program_name)),
            PathBuf::from(format!("../example-apps/target/wasm32-wasip2/release/{}.wasm", program_name)),
            PathBuf::from(format!("./target/wasm32-wasip2/release/{}.wasm", program_name)),
            PathBuf::from(format!("./examples/{}.wasm", program_name)),
            PathBuf::from(format!("./{}.wasm", program_name)),
        ]);
    }

    // Find the first existing file
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    anyhow::bail!("Could not find WebAssembly program: {}", program_name)
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let matches = Command::new("Symphony Launcher")
        .version("1.0")
        .author("Symphony Team")
        .about("Application launcher for Symphony WebAssembly programs")
        .arg(
            Arg::new("program")
                .help("Path or name of the WebAssembly program to run")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("server")
                .short('s')
                .long("server")
                .value_name("URI")
                .help("Symphony engine WebSocket URI")
                .default_value("ws://127.0.0.1:9123"),
        )
        .arg(
            Arg::new("instances")
                .short('n')
                .long("instances")
                .value_name("COUNT")
                .help("Number of instances to launch (for interactive mode)")
                .default_value("1"),
        )
        .arg(
            Arg::new("server_mode")
                .long("server-mode")
                .action(clap::ArgAction::SetTrue)
                .help("Run as server instance instead of interactive instance"),
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("Port for server mode")
                .default_value("8080"),
        )
        .get_matches();

    let program_name = matches.get_one::<String>("program").unwrap();
    let server_uri = matches.get_one::<String>("server").unwrap();
    let num_instances: usize = matches.get_one::<String>("instances").unwrap().parse()
        .context("Invalid number of instances")?;
    let server_mode = matches.get_flag("server_mode");
    let port: u16 = matches.get_one::<String>("port").unwrap().parse()
        .context("Invalid port number")?;

    // Resolve the program path
    let program_path = resolve_program_path(program_name)
        .context("Failed to resolve program path")?;

    log_client!("Symphony Application Launcher v1.0");
    log_client!("Program path: {}", program_path.display());

    if server_mode {
        run_server_instance(&program_path, server_uri, port).await
    } else {
        run_interactive_instance(&program_path, server_uri, num_instances).await
    }
}
