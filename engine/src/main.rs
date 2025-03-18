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
// use std::sync::Arc;
// use tokio::sync::mpsc::{channel, unbounded_channel};
//
// use crate::server_old::{ServerMessage, ServerState, WebSocketServer};
// use wasmtime::{Config, Engine};
//
// use crate::backend::SimulatedBackend;
// use crate::client::Client;
// use crate::runtime::{ExceptionDispatcher, Runtime};
use crate::client::Client;
use crate::l4m::L4m;
use crate::messaging::Messaging;
use crate::ping::Ping;
use crate::runtime::Runtime;
use crate::server::{Server, ServerMessage};
use crate::service::ServiceInstaller;
use std::fs;
use std::time::Duration;

// use std::time::Duration;
// use tokio::time::timeout;
//
// /// Directory for cached programs
const PROGRAM_CACHE_DIR: &str = "./program_cache";
//
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1) Ensure the cache directory exists
    fs::create_dir_all(PROGRAM_CACHE_DIR).context("Failed to create program cache directory")?;

    // fn add_builtin_services(mut self, listen_addr: &str) -> Self {
    //     self.add("runtime", Runtime::new())
    //         .add("server", Server::new(listen_addr))
    //         .add("messaging", Messaging::new())
    // }

    let l4m_backend = backend::SimulatedBackend::new(l4m::Simulator::new()).await;
    let ping_backend = backend::SimulatedBackend::new(ping::Simulator::new()).await;

    let installer = ServiceInstaller::new();

    let runtime = Runtime::new();
    runtime.load_existing_programs(Path::new(PROGRAM_CACHE_DIR))?;

    let server = Server::new("127.0.0.1:9000");
    let messaging = Messaging::new();
    let l4m = L4m::new(l4m_backend).await;
    let ping = Ping::new(ping_backend);

    installer
        .add("runtime", runtime)
        .add("server", server)
        .add("messaging", messaging)
        .add("l4m", l4m)
        .add("ping", ping)
        .install();

    // TEST: spawn a dummy client
    tokio::spawn(dummy_client());

    // wait forever
    tokio::signal::ctrl_c().await?;

    Ok(())
}

async fn dummy_client() -> anyhow::Result<()> {
    // Adjust path as needed:
    let wasm_path = PathBuf::from("../example-apps/target/wasm32-wasip2/release/multimodal.wasm");
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
        while let Ok(Some(msg)) =
            tokio::time::timeout(Duration::from_millis(100), client.wait_for_next_message()).await
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
