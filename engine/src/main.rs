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

    // 1) Ensure the cache directory exists
    fs::create_dir_all(PROGRAM_CACHE_DIR).context("Failed to create program cache directory")?;

    //let l4m_backend = backend::SimulatedBackend::new(l4m::Simulator::new()).await;
    //let ping_backend = backend::SimulatedBackend::new(ping::Simulator::new()).await;

    let backend = backend::ZmqBackend::bind("tcp://gimlab.org:8888").await?;

    //return Ok(());

    let runtime = Runtime::new();
    runtime.load_existing_programs(Path::new(PROGRAM_CACHE_DIR))?;

    let server = Server::new("127.0.0.1:9123");
    let messaging_inst2inst = PubSub::new();
    let messaging_user2inst = PushPull::new();
    let l4m = L4m::new(backend.clone()).await;
    let ping = Ping::new(backend).await;

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

    // TEST: spawn a dummy client
    tokio::spawn(dummy_client());

    // wait forever
    tokio::signal::ctrl_c().await?;

    Ok(())
}

async fn dummy_client() -> anyhow::Result<()> {
    let program_path =
        PathBuf::from("../example-apps/target/wasm32-wasip2/release/simple_decoding.wasm");
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

    const num_instances: usize = 48;

    // Launch 32 instances sequentially
    let mut instances = Vec::new();
    for i in 0..num_instances {
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
