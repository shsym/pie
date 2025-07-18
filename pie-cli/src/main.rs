use anyhow::{Context, Result, anyhow};
use clap::{Args, Parser, Subcommand};
use comfy_table::{Table, presets::UTF8_FULL};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use pie::{
    Config as EngineConfig,
    auth::{create_jwt, init_secret},
    client::{self, Client},
};
use rand::{Rng, distributions::Alphanumeric};
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use std::{
    env,
    fs::{self},
    io::{self, Write},
    path::PathBuf,
};
use tokio::io::{AsyncBufReadExt, BufReader, stdin as tokio_stdin};
use tokio::sync::oneshot;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, FmtSubscriber, Layer};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
//================================================================================//
// SECTION: CLI Command & Config Structs
//================================================================================//

#[derive(Parser, Debug)]
#[command(author, version, about = "A CLI for the PIE system")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the PIE engine and enter an interactive session.
    Start(StartArgs),
    #[command(subcommand)]
    /// Manage local models.
    Model(ModelCommands),
}

#[derive(Args, Debug)]
/// Arguments for starting the PIE engine.
pub struct StartArgs {
    /// Path to a custom TOML configuration file.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// The network host to bind to.
    #[arg(long)]
    pub host: Option<String>,
    /// The network port to use.
    #[arg(long)]
    pub port: Option<u16>,
    /// Disable authentication.
    #[arg(long)]
    pub no_auth: bool,
    /// A log file to write to.
    #[arg(long)]
    pub log: Option<PathBuf>,
    /// Enable verbose console logging.
    #[arg(long, short)]
    pub verbose: bool,
}

#[derive(Args, Debug, Default)]
/// Arguments to submit an inferlet (Wasm program) to the engine.
pub struct RunArgs {
    /// Path to the .wasm inferlet file.
    pub wasm_path: PathBuf,
    /// Run the inferlet in the background and print its instance ID.
    #[arg(long, short)]
    pub detach: bool,
}

// Other command structs (ModelCommands, etc.) remain the same
#[derive(Subcommand, Debug)]
pub enum ModelCommands {
    List,
    Add(AddModelArgs),
    Remove(RemoveModelArgs),
    Serve(ModelServeArgs),
}
#[derive(Args, Debug)]
pub struct AddModelArgs {
    pub model_name: String,
}
#[derive(Args, Debug)]
pub struct RemoveModelArgs {
    pub model_name: String,
}
#[derive(Args, Debug)]
pub struct ModelServeArgs {
    pub model_name: String,
    /// Port to serve the model on
    #[arg(long, default_value = "8080")]
    pub port: u16,
}

// Helper struct for parsing the TOML config file
#[derive(Deserialize, Debug, Default)]
struct ConfigFile {
    host: Option<String>,
    port: Option<u16>,
    enable_auth: Option<bool>,
    auth_secret: Option<String>,
    cache_dir: Option<PathBuf>,
    verbose: Option<bool>,
    log: Option<PathBuf>,
}

// Helper struct for what client commands need to know
struct ClientConfig {
    host: String,
    port: u16,
    auth_secret: String,
}

//================================================================================//
// SECTION: Main Entrypoint
//================================================================================//

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Start(args) => {
            // 1. Build the config first
            let engine_config = build_engine_config(&args)?;

            // 2. Initialize logging based on the config and get the file-writer guard
            let _guard = init_logging(&engine_config)?;

            // 3. Start the interactive session, passing the already-built config
            start_interactive_session(engine_config).await?;
        }
        Commands::Model(cmd) => {
            // Model commands don't start the engine, so they can use a simple logger
            let subscriber = FmtSubscriber::builder().with_max_level(tracing::Level::INFO).finish();
            tracing::subscriber::set_global_default(subscriber)?;
            handle_model_command(cmd).await?;
        }
    }
    Ok(())
}
//================================================================================//
// SECTION: Command Handlers
//================================================================================//

/// Starts the engine and drops into a client command-prompt session.
async fn start_interactive_session(engine_config: EngineConfig) -> Result<()> {
    // Config is already built and logging is already initialized.
    let client_config = ClientConfig {
        host: engine_config.host.clone(),
        port: engine_config.port,
        auth_secret: engine_config.auth_secret.clone(),
    };

    let (shutdown_tx, shutdown_rx) = oneshot::channel();

    println!("🚀 Starting PIE engine in background...");
    let server_handle = tokio::spawn(async move {
        if let Err(e) = pie::run_server(engine_config, shutdown_rx).await {
            eprintln!("\n[Engine Error] Engine failed: {}", e);
        }
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    println!("✅ Engine started. Entering interactive session. Type 'help' for commands.");

    let mut reader = BufReader::new(tokio_stdin());
    let mut line = String::new();
    let mut should_exit = false;

    while !should_exit {
        print!("pie> ");
        io::stdout().flush()?;

        line.clear();
        if reader.read_line(&mut line).await? == 0 {
            println!("\nExiting...");
            break; // Exit on Ctrl+D
        }

        let parts: Vec<&str> = line.trim().split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        // Updated loop logic to handle exit correctly
        match handle_interactive_command(parts[0], &parts[1..], &client_config).await {
            Ok(exit_signal) => {
                should_exit = exit_signal;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }

    // --- GRACEFUL SHUTDOWN ---
    println!("Shutting down PIE engine...");
    // Send the shutdown signal. We don't care if the receiver has already been dropped.
    let _ = shutdown_tx.send(());
    // Wait for the server task to finish its cleanup.
    server_handle.await?;
    println!("✅ Shutdown complete.");

    Ok(())
}

/// Parses and executes commands entered in the interactive session.
async fn handle_interactive_command(
    command: &str,
    args: &[&str],
    client_config: &ClientConfig,
) -> Result<bool> {
    match command {
        "run" => {
            let mut wasm_path = None;
            let mut detach = false;
            for arg in args {
                if *arg == "-d" || *arg == "--detach" {
                    detach = true;
                } else if wasm_path.is_none() {
                    wasm_path = Some(PathBuf::from(arg));
                }
            }

            if let Some(path) = wasm_path {
                let run_args = RunArgs {
                    wasm_path: path,
                    detach,
                };
                if let Err(e) = handle_run(run_args, client_config).await {
                    eprintln!("Failed to run inferlet: {}", e);
                }
            } else {
                println!("Usage: run [--detach] <path_to_wasm_file>");
            }
        }
        "query" => {
            println!("(Query functionality not yet implemented)");
        }
        "help" => {
            println!("Available commands:");
            println!("  run [--detach] <path>  - Run a .wasm inferlet");
            println!("  query                  - (Placeholder) Query the engine state");
            println!("  exit                   - Exit the PIE session");
            println!("  help                   - Show this help message");
        }
        "exit" => {
            println!("Exiting...");
            return Ok(true); // Signal to the main loop to exit.
        }
        _ => {
            println!(
                "Unknown command: '{}'. Type 'help' for a list of commands.",
                command
            );
        }
    }
    Ok(false) // Do not exit the loop.
}

/// Connects to the engine and executes a Wasm inferlet.
async fn handle_run(args: RunArgs, client_config: &ClientConfig) -> Result<()> {
    let url = format!("ws://{}:{}", client_config.host, client_config.port);
    let mut client = match Client::connect(&url).await {
        Ok(c) => c,
        Err(_) => {
            return Err(anyhow!(
                "Could not connect to engine at {}. Is it running?",
                url
            ));
        }
    };

    init_secret(&client_config.auth_secret);
    let token = create_jwt("default", pie::auth::Role::User)?;
    client.authenticate(&token).await?;

    let wasm_blob = fs::read(&args.wasm_path)
        .with_context(|| format!("Failed to read Wasm file at {:?}", args.wasm_path))?;
    let hash = client::hash_program(&wasm_blob);
    println!("Program hash: {}", hash);

    if !client.program_exists(&hash).await? {
        client.upload_program(&wasm_blob).await?;
        println!("✅ Program upload successful.");
    }

    println!("🚀 Launching instance...");
    let mut instance = client.launch_instance(&hash).await?;
    println!("✅ Instance launched with ID: {}", instance.id());

    // If not detached, spawn a task to listen for output asynchronously.
    // This prevents blocking the main command prompt.
    if !args.detach {
        println!("Streaming output for instance {}...", instance.id());
        let instance_id = instance.id().to_string();
        tokio::spawn(async move {
            while let Ok((event, message)) = instance.recv().await {
                // Printing asynchronously can interfere with the prompt. Prepending a newline helps.
                if event == "terminated" {
                    println!(
                        "\n[Instance {}] Terminated. Reason: {}",
                        instance_id, message
                    );
                    break;
                } else {
                    println!("\n[Instance {}] {}: {}", instance_id, event, message);
                }
            }
            // Let the user know the stream is done and they can get a clean prompt.
            print!(
                "(Output stream for {} ended). Press Enter to refresh prompt.",
                instance_id
            );
            let _ = io::stdout().flush();
        });
    }

    Ok(())
}

async fn handle_model_command(command: ModelCommands) -> Result<()> {
    match command {
        ModelCommands::List => {
            println!("📚 Available local models:");
            let models_dir = get_pie_home()?.join("models");
            if !models_dir.exists() {
                println!("  No models found.");
                return Ok(());
            }
            for entry in fs::read_dir(models_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    println!("  - {}", entry.file_name().to_string_lossy());
                }
            }
        }
        ModelCommands::Add(args) => {
            println!("➕ Adding model: {}", args.model_name);

            let models_root = get_pie_home()?.join("models");
            let model_files_dir = models_root.join(&args.model_name);
            let metadata_path = models_root.join(format!("{}.toml", args.model_name));

            if metadata_path.exists() || model_files_dir.exists() {
                print!(
                    "⚠️ Model '{}' already exists. Overwrite? [y/N] ",
                    args.model_name
                );
                io::stdout().flush().context("Failed to flush stdout")?;

                let mut confirmation = String::new();
                io::stdin()
                    .read_line(&mut confirmation)
                    .context("Failed to read user input")?;

                if confirmation.trim().to_lowercase() != "y" {
                    println!("Aborted by user.");
                    return Ok(());
                }
            }

            fs::create_dir_all(&model_files_dir)?;
            println!("Parameters will be stored at {:?}", model_files_dir);

            let model_index_base =
                "https://raw.githubusercontent.com/pie-project/model-index/refs/heads/main";
            let metadata_url = format!("{}/{}.toml", model_index_base, args.model_name);

            let metadata_raw =
                match download_file_with_progress(&metadata_url, "Downloading metadata...").await {
                    Ok(data) => data,
                    Err(e) => {
                        if let Some(req_err) = e.downcast_ref::<reqwest::Error>() {
                            if req_err.status() == Some(reqwest::StatusCode::NOT_FOUND) {
                                anyhow::bail!(
                                    "Model '{}' not found in the official index.",
                                    args.model_name
                                );
                            }
                        }
                        return Err(e.context("Failed to download model metadata"));
                    }
                };

            let metadata_str = String::from_utf8(metadata_raw)
                .context("Failed to parse model metadata as UTF-8")?;
            let metadata: toml::Value = toml::from_str(&metadata_str)?;

            fs::write(&metadata_path, &metadata_str)?;

            if let Some(source) = metadata.get("source").and_then(|s| s.as_table()) {
                for (name, url_val) in source {
                    if let Some(url) = url_val.as_str() {
                        let file_data =
                            download_file_with_progress(url, &format!("Downloading {}...", name))
                                .await?;
                        fs::write(model_files_dir.join(name), file_data)?;
                    }
                }
            }
            println!("✅ Model '{}' added successfully!", args.model_name);
        }
        ModelCommands::Remove(args) => {
            println!("🗑️ Removing model: {}", args.model_name);
            let models_root = get_pie_home()?.join("models");
            let model_files_dir = models_root.join(&args.model_name);
            let metadata_path = models_root.join(format!("{}.toml", args.model_name));
            let mut was_removed = false;
            if model_files_dir.exists() {
                fs::remove_dir_all(&model_files_dir)?;
                was_removed = true;
            }
            if metadata_path.exists() {
                fs::remove_file(&metadata_path)?;
                was_removed = true;
            }
            if was_removed {
                println!("✅ Model '{}' removed.", args.model_name);
            } else {
                anyhow::bail!("Model '{}' not found locally.", args.model_name);
            }
        }
        ModelCommands::Serve(_args) => {
            return Err(anyhow!("`model serve` is not yet implemented."));
        }
    }
    Ok(())
}

//================================================================================//
// SECTION: Helpers
//================================================================================//

/// Merges config from file and CLI args to create the final EngineConfig.
fn build_engine_config(args: &StartArgs) -> Result<EngineConfig> {
    let file_config = if let Some(path) = &args.config {
        let content = fs::read_to_string(path)?;
        toml::from_str(&content)?
    } else {
        ConfigFile::default()
    };

    let enable_auth = if args.no_auth {
        false
    } else {
        file_config.enable_auth.unwrap_or(true)
    };
    let auth_secret = file_config.auth_secret.unwrap_or_else(|| {
        rand::thread_rng()
            .sample_iter(&Alphanumeric)
            .take(32)
            .map(char::from)
            .collect()
    });

    Ok(EngineConfig {
        host: args
            .host
            .clone()
            .or(file_config.host)
            .unwrap_or_else(|| "127.0.0.1".to_string()),
        port: args.port.or(file_config.port).unwrap_or(8080),
        enable_auth,
        auth_secret,
        cache_dir: file_config
            .cache_dir
            .unwrap_or_else(|| get_pie_home().unwrap()),
        verbose: args.verbose || file_config.verbose.unwrap_or(false),
        log: args.log.clone().or(file_config.log),
    })
}

fn get_pie_home() -> Result<PathBuf> {
    if let Ok(path) = env::var("PIE_HOME") {
        Ok(PathBuf::from(path))
    } else {
        dirs::cache_dir()
            .map(|p| p.join("pie"))
            .ok_or_else(|| anyhow!("Failed to find home dir"))
    }
}

async fn download_file_with_progress(url: &str, message: &str) -> Result<Vec<u8>> {
    let client = HttpClient::new();
    let res = client.get(url).send().await?.error_for_status()?;
    let total_size = res.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")?
            .progress_chars("##-"),
    );
    pb.set_message(message.to_string());

    let mut downloaded = 0;
    let mut stream = res.bytes_stream();
    let mut content = Vec::with_capacity(total_size as usize);

    while let Some(item) = stream.next().await {
        let chunk = item?;
        downloaded += chunk.len();
        content.extend_from_slice(&chunk);
        pb.set_position(downloaded as u64);
    }

    let file_name = url
        .rsplit('/')
        .next()
        .unwrap_or("file")
        .split('?')
        .next()
        .unwrap_or("file");

    pb.finish_with_message(format!("✅ Downloaded {}", file_name));
    Ok(content)
}


fn init_logging(config: &EngineConfig) -> Result<Option<WorkerGuard>> {
    let mut guard = None;

    // Console logger setup
    let console_filter = if config.verbose {
        // If -v is passed, show info for everything
        EnvFilter::new("info")
    } else {
        // Otherwise, use RUST_LOG or default to "warn"
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn"))
    };
    let console_layer = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stdout)
        .with_filter(console_filter);

    // File logger setup
    let file_layer = if let Some(log_path) = &config.log {
        let parent = log_path.parent().context("Log path has no parent directory")?;
        fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create log directory at {:?}", parent))?;

        let file_appender = tracing_appender::rolling::never(parent, log_path.file_name().unwrap());
        let (non_blocking_writer, worker_guard) = tracing_appender::non_blocking(file_appender);

        // Save the guard to be returned
        guard = Some(worker_guard);

        let layer = tracing_subscriber::fmt::layer()
            .with_writer(non_blocking_writer)
            .with_ansi(false) // No colors in files
            .with_filter(EnvFilter::new("trace")); // Log everything to the file
        Some(layer)
    } else {
        None
    };

    // Register the layers
    tracing_subscriber::registry()
        .with(console_layer)
        .with(file_layer)
        .init();

    Ok(guard)
}