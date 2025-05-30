// #![allow(unused)]
//
mod client;
mod service;

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
use serde::{Deserialize, Serialize};
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};
use std::collections::HashMap;

use crate::client::{Client, hash_program};
use crate::l4m::L4m;
use crate::messaging::{PubSub, PushPull};
use crate::ping::Ping;
use crate::runtime::Runtime;
use crate::server::Server;
use crate::service::Controller;
use clap::{Arg, Command};
use colored::Colorize;
use std::fs;

const PROGRAM_CACHE_DIR: &str = "./program_cache";

//
// Engine configuration structures
#[derive(Serialize, Deserialize, Debug)]
struct EngineConfig {
    management_service: ManagementServiceConfig,
    models: Vec<String>,
    default_model: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ManagementServiceConfig {
    endpoint: String,
}

// Data structures for management service communication
#[derive(Serialize, Deserialize, Debug)]
struct ManagementCommand {
    command: String,
    params: HashMap<String, serde_json::Value>,
    correlation_id: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ManagementResponse {
    correlation_id: String,
    success: bool,
    data: Option<serde_json::Value>,
    error: Option<String>,
}

//
// Define a simple macro for client-side logging.
macro_rules! log_user {
    ($($arg:tt)*) => {
        println!("{}", format!("[User] {}", format_args!($($arg)*)).bright_blue());
    }
}

//use console_subscriber;

/// Load engine configuration from JSON file
fn load_config(config_path: Option<&str>) -> anyhow::Result<EngineConfig> {
    let config_path = config_path.unwrap_or("./config.json");

    let config_content = std::fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read config file: {}", config_path))?;

    let config: EngineConfig = serde_json::from_str(&config_content)
        .with_context(|| format!("Failed to parse config file: {}", config_path))?;

    log_user!("Loaded engine config from: {}", config_path);
    log_user!("Available models: {:?}", config.models);
    log_user!("Default model: {}", config.default_model);

    Ok(config)
}

/// Query the management service to get the IPC endpoint for a specific model
async fn get_model_endpoint(model_name: &str, config: &EngineConfig) -> anyhow::Result<String> {
    let mut socket = DealerSocket::new();
    socket.connect(&config.management_service.endpoint).await
        .context("Failed to connect to management service")?;

    // Create load-model command
    let correlation_id = uuid::Uuid::new_v4().to_string();
    let mut params = HashMap::new();
    params.insert("model_name".to_string(), serde_json::Value::String(model_name.to_string()));

    let command = ManagementCommand {
        command: "load-model".to_string(),
        params,
        correlation_id: correlation_id.clone(),
    };

    // Send command to management service
    let command_json = serde_json::to_string(&command)
        .context("Failed to serialize management command")?;
    let message = ZmqMessage::from(command_json.as_bytes().to_vec());

    socket.send(message).await
        .context("Failed to send command to management service")?;

    // Receive response
    let response_msg = socket.recv().await
        .context("Failed to receive response from management service")?;
    let response_bytes = response_msg.get(0)
        .context("Empty response from management service")?;

    let response: ManagementResponse = serde_json::from_slice(response_bytes)
        .context("Failed to parse management service response")?;

    if !response.success {
        return Err(anyhow::anyhow!(
            "Management service error: {}",
            response.error.unwrap_or_else(|| "Unknown error".to_string())
        ));
    }

    // Extract endpoint from response data
    let endpoint = response.data
        .as_ref()
        .and_then(|data| data.get("endpoint"))
        .and_then(|ep| ep.as_str())
        .context("No endpoint in management service response")?
        .to_string();

    log_user!("Got model endpoint from management service: {}", endpoint);
    Ok(endpoint)
}

/// Check all models in config and return the first available one
async fn find_first_available_model(config: &EngineConfig) -> anyhow::Result<String> {
    let mut socket = DealerSocket::new();
    socket.connect(&config.management_service.endpoint).await
        .context("Failed to connect to management service")?;

    for model_name in &config.models {
        log_user!("Checking availability of model: {}", model_name);

        let correlation_id = uuid::Uuid::new_v4().to_string();
        let mut params = HashMap::new();
        params.insert("model_name".to_string(), serde_json::Value::String(model_name.clone()));

        let command = ManagementCommand {
            command: "load-model".to_string(),
            params,
            correlation_id: correlation_id.clone(),
        };

        // Send command to management service
        let command_json = serde_json::to_string(&command)
            .context("Failed to serialize management command")?;
        let message = ZmqMessage::from(command_json.as_bytes().to_vec());

        if let Err(e) = socket.send(message).await {
            log_user!("Failed to send command for model {}: {}", model_name, e);
            continue;
        }

        // Receive response
        let response_msg = match socket.recv().await {
            Ok(msg) => msg,
            Err(e) => {
                log_user!("Failed to receive response for model {}: {}", model_name, e);
                continue;
            }
        };

        let response_bytes = match response_msg.get(0) {
            Some(bytes) => bytes,
            None => {
                log_user!("Empty response for model {}", model_name);
                continue;
            }
        };

        let response: ManagementResponse = match serde_json::from_slice(response_bytes) {
            Ok(resp) => resp,
            Err(e) => {
                log_user!("Failed to parse response for model {}: {}", model_name, e);
                continue;
            }
        };

        if response.success {
            log_user!("Model {} is available", model_name);
            return Ok(model_name.clone());
        } else {
            log_user!("Model {} is not available: {}", model_name,
                response.error.unwrap_or_else(|| "Unknown error".to_string()));
        }
    }

    Err(anyhow::anyhow!("No available models found from the configured list: {:?}", config.models))
}

/// Check if the management service is running and responsive
async fn check_management_service_status(config: &EngineConfig) -> anyhow::Result<()> {
    let mut socket = DealerSocket::new();
    socket.connect(&config.management_service.endpoint).await
        .context("Failed to connect to management service - is it running?")?;

    let correlation_id = uuid::Uuid::new_v4().to_string();
    let command = ManagementCommand {
        command: "status".to_string(),
        params: HashMap::new(),
        correlation_id: correlation_id.clone(),
    };

    let command_json = serde_json::to_string(&command)
        .context("Failed to serialize status command")?;
    let message = ZmqMessage::from(command_json.as_bytes().to_vec());

    socket.send(message).await
        .context("Failed to send status command")?;

    let response_msg = socket.recv().await
        .context("Failed to receive status response")?;
    let response_bytes = response_msg.get(0)
        .context("Empty status response")?;

    let response: ManagementResponse = serde_json::from_slice(response_bytes)
        .context("Failed to parse status response")?;

    if !response.success {
        return Err(anyhow::anyhow!(
            "Management service status check failed: {}",
            response.error.unwrap_or_else(|| "Unknown error".to_string())
        ));
    }

    log_user!("Management service is running");
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("CONFIG_PATH")
                .help("Path to engine config file")
                .default_value("./config.json"),
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

    let program_name = matches.get_one::<String>("program").unwrap();
    let config_path = matches.get_one::<String>("config").unwrap();
    let is_http = matches.get_one::<bool>("http").unwrap();
    let port = matches.get_one::<String>("port").unwrap();
    let port: u16 = port.parse().unwrap_or(9123);
    let use_dummy = matches.get_one::<bool>("dummy").unwrap();
    let use_dummy = *use_dummy;

    // Load engine configuration
    let config = load_config(Some(config_path))?;

    // Check if management service is running first
    check_management_service_status(&config).await
        .context("Management service is not available")?;

    // Find the first available model from the config
    let model_name = find_first_available_model(&config).await
        .context("Failed to find any available models")?;

    log_user!("Using model: {}", model_name);

    // 1) Ensure the cache directory exists
    fs::create_dir_all(PROGRAM_CACHE_DIR).context("Failed to create program cache directory")?;

    //return Ok(());

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

    // Setup with dummy
    let ctrl = if use_dummy {
        log_user!("Running in dummy mode");

        let dummy_l4m_backend = backend::SimulatedBackend::new(l4m::Simulator::new()).await;
        let dummy_ping_backend = backend::SimulatedBackend::new(ping::Simulator::new()).await;

        let l4m = L4m::new(dummy_l4m_backend.clone()).await;
        let ping = Ping::new(dummy_ping_backend.clone()).await;

        ctrl
            .add(l4m::available_models().first().unwrap(), l4m)
            .add("ping", ping)

    } else {
        // Get the model endpoint from management service
        let model_endpoint = get_model_endpoint(&model_name, &config).await
            .context("Failed to get model endpoint from management service")?;

        // Connect to the model backend endpoint
        let l4m_backend = backend::ZmqBackend::bind(&model_endpoint).await?;

        // Setup with real backend
        let l4m = L4m::new(l4m_backend.clone()).await;
        let ping = Ping::new(l4m_backend.clone()).await;

        // Install all services
        ctrl
            .add(l4m::available_models().first().unwrap(), l4m)
            .add("ping", ping)
    };

    // Install all services
    ctrl.install();

    // periodically print stats
    tokio::spawn(async {
        loop {
            let service_id = service::get_service_id(l4m::available_models().first().unwrap()).unwrap();
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            l4m::Command::PrintStats.dispatch(service_id).unwrap();
        }
    });

    // TEST: spawn a dummy client with the program name
    if *is_http {
        tokio::spawn(dummy_client2(program_name.to_string(), port));
    } else {
        tokio::spawn(dummy_client(program_name.to_string()));
    }
    // wait forever
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
