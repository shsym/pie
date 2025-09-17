// mod service;
//
// mod auth;
// mod backend;
// mod batching;
// mod bindings;
// mod client;
// mod instance;
// mod l4m;
// mod messaging;
// mod object;
// mod ping;
// mod runtime;
// mod server;
// mod tokenizer;
// mod utils;
//
// //
// use anyhow::Context;
// use std::path::{Path, PathBuf};
//
// use crate::auth::{create_jwt, init_secret};
// use crate::client::{Client, hash_program};
// use crate::messaging::{PubSub, PushPull};
// use crate::ping::Ping;
// use crate::runtime::Runtime;
// use crate::server::Server;
// use crate::service::install_service;
// use clap::Parser;
// use serde::Deserialize;
// use std::{env, fs};
// use tracing_subscriber::layer::SubscriberExt;
// use tracing_subscriber::util::SubscriberInitExt;
// use tracing_subscriber::{EnvFilter, Layer};
//
// const DEFAULT_PORT: u16 = 9123; // Default port for the server
// const DEFAULT_HOST: &str = "0.0.0.0"; // Default host for the server
//
// #[derive(Parser, Debug)]
// #[command(author, version, about)]
// struct Cli {
//     /// Path to the TOML configuration file.
//     #[arg(long, help = "Path to a TOML configuration file")]
//     config: Option<PathBuf>,
//
//     /// The network host to connect to.
//     #[arg(long, help = "Network host")]
//     host: Option<String>,
//
//     /// The network port to use.
//     #[arg(short, long, help = "Network port")]
//     port: Option<u16>,
//
//     #[arg(long, help = "Whether to use authentication")]
//     enable_auth: bool,
//
//     #[arg(long, help = "Secret for signing JWTs")]
//     auth_secret: Option<String>,
//
//     /// The directory for caching data.
//     /// Defaults to $PIE_HOME, or $HOME/.cache/pie if not set.
//     #[arg(long, help = "Cache directory")]
//     cache_dir: Option<PathBuf>,
//
//     /// Enable verbose output.
//     #[arg(long, action = clap::ArgAction::SetTrue, help = "Enable verbose logging")]
//     verbose: bool,
//
//     /// A log file to write to.
//     #[arg(long, help = "Log file path")]
//     log: Option<PathBuf>,
// }
//
// /// A struct that represents the structure of your TOML configuration file.
// /// The field names here use snake_case to match TOML conventions.
// #[derive(Deserialize, Debug, Default)]
// struct ConfigFile {
//     host: Option<String>,
//     port: Option<u16>,
//     enable_auth: Option<bool>,
//     auth_secret: Option<String>,
//     cache_dir: Option<PathBuf>,
//     verbose: Option<bool>,
//     log: Option<PathBuf>,
// }
//
// /// The final, merged configuration struct that the application will use.
// #[derive(Debug)]
// struct Config {
//     host: String,
//     port: u16,
//     enable_auth: bool,
//     auth_secret: String,
//     cache_dir: PathBuf,
//     verbose: bool,
//     log: Option<PathBuf>,
// }
//
// /// Determines the default cache directory path.
// ///
// /// It follows this logic:
// /// 1. Check for the `PIE_HOME` environment variable.
// /// 2. If not found, fall back to the user's cache directory (`$HOME/.cache` on Linux).
// /// 3. Appends a "pie" subdirectory to the chosen path.
// fn get_default_cache_dir() -> PathBuf {
//     // Try to get PIE_HOME from environment variables.
//     if let Ok(pie_home) = env::var("PIE_HOME") {
//         return PathBuf::from(pie_home);
//     }
//
//     // Fallback to the system's cache directory if PIE_HOME is not set.
//     // `dirs::cache_dir()` provides the conventional location for cache files.
//     // e.g., ~/.cache on Linux, ~/Library/Caches on macOS, %LOCALAPPDATA% on Windows.
//     if let Some(mut cache_path) = dirs::cache_dir() {
//         cache_path.push("pie");
//         return cache_path;
//     }
//
//     // A final fallback if even the home directory can't be found.
//     // On most systems, this part of the code is unlikely to be reached.
//     let mut fallback = PathBuf::from(".cache");
//     fallback.push("pie");
//     fallback
// }
//
// fn main() -> anyhow::Result<()> {
//     // 1. Parse CLI and TOML config
//     let cli = Cli::parse();
//     let file_config = if let Some(config_path) = &cli.config {
//         let file_contents = fs::read_to_string(config_path)
//             .with_context(|| format!("Could not read config file at {:?}", config_path))?;
//         toml::from_str(&file_contents)
//             .with_context(|| format!("Could not parse TOML from {:?}", config_path))?
//     } else {
//         ConfigFile::default()
//     };
//
//     // 2. Merge configurations
//     let config = Config {
//         host: cli
//             .host
//             .or(file_config.host)
//             .unwrap_or_else(|| DEFAULT_HOST.to_string()),
//         port: cli.port.or(file_config.port).unwrap_or(DEFAULT_PORT),
//         enable_auth: cli.enable_auth || file_config.enable_auth.unwrap_or(false),
//         auth_secret: cli
//             .auth_secret
//             .or(file_config.auth_secret)
//             .unwrap_or("dummy".to_string()),
//         cache_dir: cli
//             .cache_dir
//             .or(file_config.cache_dir)
//             .unwrap_or_else(get_default_cache_dir),
//         verbose: cli.verbose || file_config.verbose.unwrap_or(true),
//         log: cli.log.or(file_config.log),
//     };
//
//     // 3. Setup logging
//     // This guard must be kept alive for the duration of the program.
//     // If it's dropped, the background logging thread will shut down.
//     let _guard;
//
//     let stdout_filter = if config.verbose {
//         EnvFilter::new("info")
//     } else {
//         EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"))
//     };
//
//     let stdout_layer = tracing_subscriber::fmt::layer()
//         .with_writer(std::io::stdout)
//         .with_filter(stdout_filter);
//
//     let file_layer = if let Some(log_path) = &config.log {
//         // Ensure the parent directory exists.
//         if let Some(parent) = log_path.parent() {
//             fs::create_dir_all(parent)
//                 .with_context(|| format!("Failed to create log directory at {:?}", parent))?;
//         }
//
//         // Use tracing_appender to create a non-blocking file writer.
//         // We use `rolling::never` to create a single, non-rotating log file.
//         let file_appender = tracing_appender::rolling::never(
//             log_path.parent().unwrap_or_else(|| Path::new(".")),
//             log_path.file_name().unwrap_or_else(|| "pie.log".as_ref()),
//         );
//
//         // The 'non_blocking' function spawns a dedicated thread for writing.
//         let (non_blocking_writer, guard) = tracing_appender::non_blocking(file_appender);
//         _guard = guard; // Store the guard to keep the thread alive.
//
//         let layer = tracing_subscriber::fmt::layer()
//             .with_writer(non_blocking_writer) // Use the non-blocking writer
//             .with_ansi(false)
//             .with_filter(EnvFilter::new("trace")); // Log everything to the file
//         Some(layer)
//     } else {
//         None
//     };
//
//     tracing_subscriber::registry()
//         .with(stdout_layer)
//         .with(file_layer)
//         .init();
//
//     tracing::debug!("{:#?}", config);
//
//     // 4. Build the Tokio runtime and start the runtime.
//     let rt = tokio::runtime::Builder::new_multi_thread()
//         .enable_all()
//         .build()?;
//
//     rt.block_on(start(config))
// }
//
// async fn start(config: Config) -> anyhow::Result<()> {
//     // Ensure the cache directory exists
//     fs::create_dir_all(&config.cache_dir).map_err(|e| {
//         tracing::error!(error = %e,"Setup failure: could not create cache dir");
//         e
//     })?;
//
//     if config.enable_auth {
//         tracing::info!("Authentication is enabled.");
//         init_secret(&config.auth_secret);
//         let token = create_jwt("default", auth::Role::User)?;
//         tracing::info!("Use this token to authenticate: {}", token);
//     } else {
//         tracing::info!("Authentication is disabled.");
//     }
//
//     // Set up core services
//     let runtime = Runtime::new(&config.cache_dir);
//     runtime.load_existing_programs()?;
//
//     let server_url = format!("{}:{}", config.host, config.port);
//
//     let server = Server::new(&server_url, config.enable_auth);
//     let messaging_inst2inst = PubSub::new();
//     let messaging_user2inst = PushPull::new();
//
//     // Set up test services
//     let dummy_l4m_backend = backend::SimulatedBackend::new(l4m::Simulator::new()).await;
//     let dummy_ping_backend = backend::SimulatedBackend::new(ping::Simulator::new()).await;
//     let ping = Ping::new(dummy_ping_backend.clone()).await;
//
//     install_service("runtime", runtime);
//     install_service("server", server);
//     install_service("messaging-inst2inst", messaging_inst2inst);
//     install_service("messaging-user2inst", messaging_user2inst);
//     install_service("ping", ping);
//
//     //l4m::attach_new_backend("model-test", dummy_l4m_backend).await;
//
//     tracing::info!("Runtime started successfully.");
//
//     spawn_client("simple_decoding".to_string()).await?;
//
//     tokio::signal::ctrl_c().await?;
//     tracing::info!("Ctrl+C received, shutting down.");
//
//     Ok(())
// }
//
// async fn spawn_client(program_name: String) -> anyhow::Result<()> {
//     // Wait until more than one model is available, checking every second.
//     loop {
//         // NOTE: Assuming `l4m::available_models()` is an async function
//         // that returns a collection (e.g., Vec) that has a .len() method.
//         let models = l4m::available_models();
//         if models.len() > 0 {
//             tracing::info!("Found {} models. Proceeding to spawn client.", models.len());
//             break;
//         }
//         tracing::info!(
//             "Waiting for more than one model to become available... Checking again in 1s."
//         );
//         tokio::time::sleep(std::time::Duration::from_secs(1)).await;
//     }
//
//     let program_path = PathBuf::from(format!(
//         "../example-apps/target/wasm32-wasip2/release/{}.wasm",
//         program_name
//     ));
//
//     // Check if the file exists
//     if !program_path.exists() {
//         tracing::error!("Error: Program file not found at path: {:?}", program_path);
//         return Ok(());
//     }
//
//     let server_uri = "ws://127.0.0.1:9123";
//
//     tracing::info!("Using program: {}", program_name);
//
//     let mut client = Client::connect(server_uri).await?;
//     let program_blob = fs::read(&program_path)?;
//     let program_hash = hash_program(&program_blob);
//
//     tracing::info!("Program file hash: {}", program_hash);
//
//     // If program is not present, upload it
//     if !client.program_exists(&program_hash).await? {
//         tracing::info!("Program not found on server, uploading now...");
//         client.upload_program(&program_blob).await?;
//         tracing::info!("Program uploaded successfully!");
//     }
//
//     const NUM_INSTANCES: usize = 1;
//
//     // Launch 1 instances sequentially
//     let mut instances = Vec::new();
//     for i in 0..NUM_INSTANCES {
//         let instance = client.launch_instance(&program_hash).await?;
//         tracing::info!("Instance {} launched.", instance.id());
//         instances.push(instance);
//     }
//
//     // Spawn a task for each instance to handle sending and receiving concurrently.
//     let mut handles = Vec::new();
//     for mut instance in instances {
//         let handle = tokio::spawn(async move {
//             instance.send("event #1: Hello from Rust client").await?;
//             instance.send("event #2: Another event").await?;
//             instance.send("event #3: Another event").await?;
//
//             while let Ok((event, message)) = instance.recv().await {
//                 match event.as_str() {
//                     "terminated" => {
//                         tracing::info!(
//                             "Instance {} terminated. Reason: {}",
//                             instance.id(),
//                             message
//                         );
//                         break;
//                     }
//                     _ => {
//                         tracing::info!("Instance {} received message: {}", instance.id(), message);
//                     }
//                 }
//             }
//             anyhow::Result::<()>::Ok(())
//         });
//         handles.push(handle);
//     }
//
//     // Wait for all instance tasks to complete.
//     for handle in handles {
//         handle.await??;
//     }
//
//     client.close().await?;
//     tracing::info!("Client connection closed.");
//     Ok(())
// }
