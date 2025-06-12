// #![allow(unused)]
//
mod client;
mod service;
mod zmq_handler;
mod backend_discovery;

mod backend;
mod batching;
mod bindings;
mod instance;
mod l4m;
mod messaging;
mod object;
mod ping;
mod runtime;
mod server;
mod tokenizer;
mod utils;

//
use anyhow::Context;
use std::path::{Path, PathBuf};

use crate::client::{Client, hash_program};
use crate::l4m::L4m;
use crate::messaging::{PubSub, PushPull};
use crate::ping::Ping;
use crate::runtime::Runtime;
use crate::backend_discovery::discover_backend_for_model;
use crate::server::Server;
use crate::service::Controller;
use clap::{Arg, Command};
use colored::Colorize;
use std::fs;

const PROGRAM_CACHE_DIR: &str = "./program_cache";

// Define a simple macro for client-side logging.
#[macro_export]
macro_rules! log_user {
    ($($arg:tt)*) => {
        println!("{}", format!("[User] {}", format_args!($($arg)*)).bright_blue());
    }
}

/// Check if a model is available by attempting to load it via management service
/// This function is used for on-demand backend discovery
async fn check_model_available(model_name: &str, engine_manager_endpoint: &str) -> anyhow::Result<String> {
    // Use the new HTTP-based backend discovery
    match discover_backend_for_model(engine_manager_endpoint, model_name).await {
        Ok(endpoint) => {
            log_user!("Model {} is available at endpoint: {}", model_name, endpoint);
            Ok(endpoint)
        }
        Err(e) => {
            log_user!("Model {} is not available: {}", model_name, e);
            Err(anyhow::anyhow!("Model {} is not available: {}", model_name, e))
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create log directory if it doesn't exist
    std::fs::create_dir_all("logs").unwrap_or(());

    // Create log file with current date
    let current_date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let log_filename = format!("engine-9123-{}.log", current_date);

    // Initialize tracing subscriber to write to file
    let file_appender = tracing_appender::rolling::never("logs", &log_filename);
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive("pie_rt=debug".parse().unwrap()))
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .with_ansi(false)  // Disable ANSI color codes for clean log files
        .with_writer(non_blocking)
        .init();

    tracing::info!("Symphony Engine starting up");

    //console_subscriber::init();

    // Parse command line arguments
    let matches = Command::new("Symphony Engine")
        .version("1.0")
        .author("Symphony Team")
        .about("Symphony Engine for WebAssembly programs")
        .arg(
            Arg::new("program")
                .help("Name of the program to run (without extension)")
                .default_value("simple_decoding"),
        )
        .arg(
            Arg::new("engine_manager")
                .short('e')
                .long("engine-manager")
                .value_name("ENDPOINT")
                .help("Engine-manager endpoint URL")
                .default_value("http://127.0.0.1:8080"),
        )
        .arg(
            Arg::new("http")
                .short('H')
                .long("http")
                .action(clap::ArgAction::SetTrue)
                .help("Run the HTTP server")
                .default_value("false"),
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("Port to run the HTTP server on")
                .default_value("9123"),
        )
        .arg(
            Arg::new("dummy")
                .short('d')
                .long("dummy")
                .action(clap::ArgAction::SetTrue)
                .help("Run the dummy client")
                .default_value("false"),
        )
        .get_matches();

    let _program_name = matches.get_one::<String>("program").unwrap();
    let _is_http = matches.get_one::<bool>("http").unwrap();
    let port = matches.get_one::<String>("port").unwrap();
    let _port: u16 = port.parse().unwrap_or(9123);
    let use_dummy = matches.get_one::<bool>("dummy").unwrap();
    let use_dummy = *use_dummy;

    // Get engine-manager endpoint from command line or environment variable
    let management_endpoint = matches.get_one::<String>("engine_manager")
        .map(|s| s.clone())
        .or_else(|| std::env::var("SYMPHONY_ENGINE_MANAGER_ENDPOINT").ok())
        .unwrap_or_else(|| "http://127.0.0.1:8080".to_string());

    // Store management endpoint for runtime backend discovery
    // We don't check for backends on startup - they will be discovered when needed
    log_user!("Engine starting - backend discovery will happen on-demand");
    log_user!("Engine-manager endpoint: {}", management_endpoint);

    // 1) Ensure the cache directory exists
    fs::create_dir_all(PROGRAM_CACHE_DIR).context("Failed to create program cache directory")?;

    let runtime = Runtime::new();
    runtime.load_existing_programs(Path::new(PROGRAM_CACHE_DIR))?;

    let server = Server::new("127.0.0.1:9123");
    let messaging_inst2inst = PubSub::new();
    let messaging_user2inst = PushPull::new();

    let ctrl = Controller::new()
            .add("runtime", runtime)
            .add("server", server)
            .add("messaging-inst2inst", messaging_inst2inst)
            .add("messaging-user2inst", messaging_user2inst);

    // Setup services based on mode
    let ctrl = if use_dummy {
        log_user!("Running in dummy mode");

        let dummy_l4m_backend = backend::SimulatedBackend::new(l4m::Simulator::new()).await;
        let dummy_ping_backend = backend::SimulatedBackend::new(ping::Simulator::new()).await;

        let l4m = L4m::new(dummy_l4m_backend.clone()).await;
        let ping = Ping::new(dummy_ping_backend.clone()).await;

        let models = l4m::available_models();
        let default_model = "dummy_model".to_string();
        let model_name = models.first().unwrap_or(&default_model);

        ctrl
            .add(model_name, l4m)
            .add("ping", ping)
    } else {
        // On-demand mode: ping uses real L4M backend when available, L4M services added dynamically
        log_user!("Starting engine in on-demand backend mode");

        // Return controller without any pre-registered services - they'll all be added dynamically
        ctrl
    };

    // Install all services
    ctrl.install();

    // TODO: Add periodic stats printing for connected backends when they are connected
    // This should be done dynamically based on which backends are actually connected

    // Wait forever - applications will be loaded via WebSocket API when requested
    tokio::signal::ctrl_c().await?;

    Ok(())
}

async fn dummy_client2(program_name: String, port: u16) -> anyhow::Result<()> {
    let program_path = PathBuf::from(format!(
        "../example-apps/target/wasm32-wasip2/release/{}.wasm",
        program_name
    ));

    let server_uri = "ws://127.0.0.1:9123";

    let mut client = Client::connect(server_uri).await?;
    let program_blob = fs::read(&program_path)?;
    let program_hash = hash_program(&program_blob);

    log_user!("Program file hash: {}", program_hash);

    // If program is not present, upload it
    if !client.program_exists(&program_hash).await? {
        log_user!("Program not found on server, uploading now...");
        client.upload_program(&program_blob).await?;
        log_user!("Program uploaded successfully!");
    }

    client
        .launch_server_instance(&program_hash, port as u32)
        .await?;

    Ok(())
}

async fn dummy_client(program_name: String) -> anyhow::Result<()> {
    let program_path = PathBuf::from(format!(
        "../example-apps/target/wasm32-wasip2/release/{}.wasm",
        program_name
    ));

    // Check if the file exists
    if !program_path.exists() {
        log_user!("Error: Program file not found at path: {:?}", program_path);
        return Ok(());
    }

    let server_uri = "ws://127.0.0.1:9123";

    log_user!("Using program: {}", program_name);

    let mut client = Client::connect(server_uri).await?;
    let program_blob = fs::read(&program_path)?;
    let program_hash = hash_program(&program_blob);

    log_user!("Program file hash: {}", program_hash);

    // If program is not present, upload it
    if !client.program_exists(&program_hash).await? {
        log_user!("Program not found on server, uploading now...");
        client.upload_program(&program_blob).await?;
        log_user!("Program uploaded successfully!");
    }

    const NUM_INSTANCES: usize = 1;

    // Launch 1 instances sequentially
    let mut instances = Vec::new();
    for i in 0..NUM_INSTANCES {
        let instance = client.launch_instance(&program_hash).await?;
        log_user!("Instance {} launched.", instance.id());
        instances.push(instance);
    }

    // Spawn a task for each instance to handle sending and receiving concurrently.
    let mut handles = Vec::new();
    for mut instance in instances {
        let handle = tokio::spawn(async move {
            instance.send("event #1: Hello from Rust client").await?;
            instance.send("event #2: Another event").await?;
            instance.send("event #3: Another event").await?;

            while let Ok((event, message)) = instance.recv().await {
                match event.as_str() {
                    "terminated" => {
                        log_user!("Instance {} terminated. Reason: {}", instance.id(), message);
                        break;
                    }
                    _ => {
                        log_user!("Instance {} received message: {}", instance.id(), message);
                    }
                }
            }
            anyhow::Result::<()>::Ok(())
        });
        handles.push(handle);
    }

    // Wait for all instance tasks to complete.
    for handle in handles {
        handle.await??;
    }

    client.close().await?;
    log_user!("Client connection closed.");
    Ok(())
}
