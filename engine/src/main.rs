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
// Define a simple macro for client-side logging.
macro_rules! log_user {
    ($($arg:tt)*) => {
        println!("{}", format!("[User] {}", format_args!($($arg)*)).bright_blue());
    }
}

//use console_subscriber;

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
    let is_http = matches.get_one::<bool>("http").unwrap();
    let port = matches.get_one::<String>("port").unwrap();
    let port: u16 = port.parse().unwrap_or(9123);
    let use_dummy = matches.get_one::<bool>("dummy").unwrap();
    let use_dummy = *use_dummy;

    // 1) Ensure the cache directory exists
    fs::create_dir_all(PROGRAM_CACHE_DIR).context("Failed to create program cache directory")?;

    let dummy_l4m_backend = backend::SimulatedBackend::new(l4m::Simulator::new()).await;
    let dummy_ping_backend = backend::SimulatedBackend::new(ping::Simulator::new()).await;

    let l4m_backend = backend::ZmqBackend::bind("ipc:///tmp/symphony-ipc").await?;

    //return Ok(());

    let runtime = Runtime::new();
    runtime.load_existing_programs(Path::new(PROGRAM_CACHE_DIR))?;

    let server = Server::new("127.0.0.1:9123");
    let messaging_inst2inst = PubSub::new();
    let messaging_user2inst = PushPull::new();

    // Setup with dummy
    if use_dummy {
        log_user!("Running in dummy mode");
        let l4m = L4m::new(dummy_l4m_backend.clone()).await;
        let ping = Ping::new(dummy_ping_backend.clone()).await;

        l4m::set_available_models(["llama3"]);

        // Install all services
        let _ = Controller::new()
            .add("runtime", runtime)
            .add("server", server)
            .add("messaging-inst2inst", messaging_inst2inst)
            .add("messaging-user2inst", messaging_user2inst)
            .add("llama3", l4m)
            .add("ping", ping)
            .install();
    } else {
        // Setup with real backend
        let l4m = L4m::new(l4m_backend.clone()).await;
        let ping = Ping::new(dummy_ping_backend.clone()).await;
        let avail_models = l4m::available_models();

        // Install all services
        let _ = Controller::new()
            .add("runtime", runtime)
            .add("server", server)
            .add("messaging-inst2inst", messaging_inst2inst)
            .add("messaging-user2inst", messaging_user2inst)
            .add(avail_models.first().unwrap(), l4m)
            .add("ping", ping)
            .install();
    }

    // TEST: spawn a dummy client with the program name
    // if *is_http {
    //     tokio::spawn(dummy_client2(program_name.to_string(), port));
    // } else {
    //     tokio::spawn(dummy_client(program_name.to_string()));
    // }
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

    const NUM_INSTANCES: usize = 200;

    // Launch 32 instances sequentially
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
