mod backend;
mod controller;
mod driver_l4m;
mod dummy;
mod instance;
mod lm;
mod object;
mod runtime;
mod server;
mod tokenizer;
mod utils;

use anyhow::Context;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc::channel;

use crate::controller::Controller;
use crate::instance::{Command, Id as InstanceId};
use crate::server::{ServerState, WebSocketServer};
use wasmtime::{Config, Engine};

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
    let (inst2server_tx, mut inst2server_rx) = channel::<(InstanceId, Command)>(1024);

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

    // 9) Await both tasks
    let _ = tokio::join!(controller_handle, server_handle);

    Ok(())
}
