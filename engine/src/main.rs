#![allow(unused)]

mod backend;
mod client;
mod controller;
mod driver_l4m;
mod instance;
mod lm;
mod object;
mod runtime;
mod server;
mod tokenizer;
mod utils;

use anyhow::Context;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc::{channel, unbounded_channel};

use crate::controller::Controller;
use crate::instance::{Command, Id as InstanceId};
use crate::server::{ServerMessage, ServerState, WebSocketServer};
use wasmtime::{Config, Engine};

use crate::client::Client;
use crate::driver_l4m::DummyBackend;
use crate::runtime::Runtime;
use std::fs;
use std::time::Duration;

/// Directory for cached programs
const PROGRAM_CACHE_DIR: &str = "./program_cache";
const TOKENIZER_MODEL: &str = "../test-tokenizer/tokenizer.model";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1) Ensure the cache directory exists
    fs::create_dir_all(PROGRAM_CACHE_DIR).context("Failed to create program cache directory")?;

    // 2) Configure Wasmtime engine
    let mut config = Config::default();
    config.async_support(true);
    let engine = Engine::new(&config)?;

    // 3) Build a channel for (instance -> controller) commands
    let (inst2server_tx, mut inst2server_rx) = unbounded_channel::<(InstanceId, Command)>();

    // 4) Create our “runtime” state.
    //    (This holds everything the controller and server might share: engine,
    //     references to compiled programs, running instances, etc.)
    let runtime = Arc::new(Runtime::new(engine, inst2server_tx));

    // 5) Scan the existing cache_dir and load (or just record) programs on disk
    runtime.load_existing_programs(Path::new(PROGRAM_CACHE_DIR))?;

    // 6) Spawn the controller loop (which manages commands coming from instances)
    let backend = DummyBackend::new(Duration::ZERO).await;
    // let backend = ZmqBackend::bind("tcp://127.0.0.1:5555")
    //     .await
    //     .context("Failed to bind backend")
    //     .unwrap();

    let mut controller = Controller::new(runtime.clone(), backend).await;

    let controller_handle = tokio::spawn(async move {
        while let Some((inst_id, cmd)) = inst2server_rx.recv().await {
            if let Err(e) = controller.exec(inst_id, cmd).await {
                eprintln!("Controller exec error: {}", e);
            }
        }
    });

    // 7) Create the WebSocket server state.
    //    This example keeps *file uploads* & client connections in WebSocketState,
    //    but references the same Arc<RuntimeState> for controlling programs.
    let ws_state = Arc::new(ServerState::new(runtime.clone()));

    // 8) Start listening for WebSocket connections
    let listen_addr = "127.0.0.1:9000";
    println!("Listening on ws://{}", listen_addr);
    let server_handle = tokio::spawn(async move {
        let server = WebSocketServer::new(ws_state);
        server.run(listen_addr).await
    });

    // TEST: spawn a dummy client
    tokio::spawn(dummy_client());

    // 9) Await both tasks
    let _ = tokio::join!(controller_handle, server_handle);

    Ok(())
}

async fn dummy_client() -> anyhow::Result<()> {
    // Adjust path as needed:
    let wasm_path = PathBuf::from("../example-apps/target/wasm32-wasip2/release/simple_decoding.wasm");
    let server_uri = "ws://127.0.0.1:9000";

    // 1) Create and connect the client
    let mut client = Client::connect(server_uri).await?;

    // 2) Read local file and compute BLAKE3
    let wasm_bytes = fs::read(&wasm_path)?;
    let file_hash = blake3::hash(&wasm_bytes).to_hex().to_string();
    println!("[User] Program file hash: {}", file_hash);

    // 3) Query existence
    match client.query_existence(&file_hash).await? {
        ServerMessage::QueryResponse { hash, exists } => {
            println!(
                "[User] query_existence response: hash={}, exists={}",
                hash, exists
            );

            // 4) If not present, upload
            if !exists {
                println!("[User] Program not found on server, uploading now...");
                client.upload_program(&wasm_bytes, &file_hash).await?;
            } else {
                println!("[User] Program already exists on server, skipping upload.");
            }
        }
        ServerMessage::Error { error } => {
            eprintln!("[User] query_existence got error: {}", error);
        }
        _ => {}
    }

    // 5) Start the program
    if let Some(instance_id) = client.start_program(&file_hash).await? {
        println!("[User] Program launched with instance_id = {}", instance_id);

        // 6) Send a couple of events
        client.send_event(
            &instance_id,
            "Hello from Rust client - event #1".to_string(),
        )?;
        client.send_event(&instance_id, "Another event #2".to_string())?;

        // Wait a bit to let any "program_event" messages come back
        //tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Drain the queue of messages
        while let Ok(Some(msg)) = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            client.wait_for_next_message(),
        )
        .await
        {
            println!("[User] Received async event: {:?}", msg);
        }

        // 7) Terminate the program
        //client.terminate_program(&instance_id).await?;
    } else {
        println!("[User] Program launch failed or was not recognized.");
    }

    // 8) Close the connection
    client.close().await?;
    println!("[User] Client connection closed.");
    Ok(())
}
