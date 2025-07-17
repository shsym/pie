use anyhow::{anyhow, Context, Result};
use chrono::{Duration, Utc};
use clap::{Args, Parser, Subcommand};
use comfy_table::{presets::UTF8_FULL, Table};
use daemonize::Daemonize;
use futures_util::{stream::StreamExt, SinkExt};
use hex::ToHex;
use indicatif::{ProgressBar, ProgressStyle};
use jsonwebtoken::{encode, EncodingKey, Header};
use rand::{distributions::Alphanumeric, Rng};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    process::Command as StdCommand,
};
use sysinfo::{Pid, System};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use url::Url;

//================================================================================//
// SECTION: WebSocket Client & Messaging
//================================================================================//

/// Messages sent from the CLI client to the engine server.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ClientMessage<'a> {
    #[serde(rename = "authenticate")]
    Authenticate { corr_id: u32, token: String },
    #[serde(rename = "query")]
    Query {
        corr_id: u32,
        subject: &'a str,
    },
    #[serde(rename = "upload_program")]
    UploadProgram {
        corr_id: u32,
        #[serde(with = "serde_bytes")]
        program_blob: &'a [u8],
    },
    #[serde(rename = "program_exists")]
    ProgramExists { corr_id: u32, hash: String },
    #[serde(rename = "launch_instance")]
    LaunchInstance { corr_id: u32, hash: String },
}

/// Messages received from the engine server.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ServerMessage {
    #[serde(rename = "response")]
    Response {
        corr_id: u32,
        successful: bool,
        result: serde_json::Value,
    },
    #[serde(rename = "instance_event")]
    InstanceEvent {
        instance_id: String,
        event: String,
        message: String,
    },
}

//================================================================================//
// SECTION: Core Structs (Config & State)
//================================================================================//

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EngineConfig {
    pub host: String,
    pub port: u16,
    pub enable_auth: bool,
    pub auth_secret: String,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct PidState {
    pub engine: Option<u32>,
    pub backends: Vec<BackendProcess>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BackendProcess {
    pub pid: u32,
    pub model: String,
    pub backend_type: String,
}

//================================================================================//
// SECTION: CLI Command Definitions
//================================================================================//

#[derive(Parser, Debug)]
#[command(author, version, about = "A CLI for the PIE system", long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Parser, Debug)]
enum Commands {
    Start(StartArgs),
    Stop,
    Status,
    #[command(subcommand)]
    Model(ModelCommands),
    Run(RunArgs),
}

#[derive(Args, Debug)]
/// Start the PIE engine as a background process.
pub struct StartArgs {
    /// Path to a custom TOML configuration file for the engine.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
/// Manage models (add, list, serve, etc.).
pub enum ModelCommands {
    /// List locally available models.
    List,
    /// Download a new model from the official model index.
    Add(AddModelArgs),
    /// Remove a local model from the cache.
    Remove(RemoveModelArgs),
    /// Serve a local model with a specified backend.
    Serve(ServeModelArgs),
}

#[derive(Args, Debug)]
pub struct AddModelArgs {
    /// The name of the model to add (e.g., "llama-3.2-1b-instruct").
    pub model_name: String,
}

#[derive(Args, Debug)]
pub struct RemoveModelArgs {
    /// The name of the local model to remove.
    pub model_name: String,
}

#[derive(Args, Debug)]
pub struct ServeModelArgs {
    /// The name of the model to serve.
    pub model_name: String,
    /// The backend to use for serving (e.g., "python", "cuda").
    #[arg(long, short)]
    pub backend: String,
    /// Optional path to a config file for the backend.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

#[derive(Args, Debug)]
/// Submit an inferlet (Wasm program) to the engine.
pub struct RunArgs {
    /// Path to the .wasm inferlet file or a previously uploaded program hash.
    pub wasm_path_or_hash: String,
    /// Run the inferlet in the background and print its instance ID.
    #[arg(long, short)]
    pub detach: bool,
    /// Custom arguments to pass to the inferlet instance.
    #[arg(last = true)]
    pub inferlet_args: Vec<String>,
}

//================================================================================//
// SECTION: Main Entrypoint
//================================================================================//

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Start(args) => handle_start(args).await,
        Commands::Stop => handle_stop().await,
        Commands::Status => handle_status().await,
        Commands::Model(model_cmd) => handle_model_command(model_cmd).await,
        Commands::Run(args) => handle_run(args).await,
    }
}

//================================================================================//
// SECTION: Command Handlers
//================================================================================//

async fn handle_start(args: StartArgs) -> Result<()> {
    if let Ok(pid) = get_running_engine_pid() {
        return Err(anyhow!("Engine is already running with PID: {}", pid));
    }

    println!("🚀 Starting PIE engine...");

    let config = if let Some(config_path) = args.config {
        let content = fs::read_to_string(config_path)?;
        toml::from_str(&content)?
    } else {
        EngineConfig {
            host: "127.0.0.1".to_string(),
            port: 9123,
            enable_auth: true,
            auth_secret: rand::thread_rng().sample_iter(&Alphanumeric).take(32).map(char::from).collect(),
        }
    };
    save_engine_config(&config)?;
    let engine_config_path = get_engine_config_path()?;

    let pie_home = get_pie_home()?;
    let logs_dir = pie_home.join("logs");
    fs::create_dir_all(&logs_dir)?;
    let stdout = File::create(logs_dir.join("engine.out"))?;
    let stderr = File::create(logs_dir.join("engine.err"))?;

    let engine_path = env::current_exe()?.parent().ok_or(anyhow!("Cannot find parent dir"))?.join("../engine/target/release/engine");

    if !engine_path.exists() {
        return Err(anyhow!(
            "Engine executable not found at {:?}.\nPlease build the engine crate first with `cargo build --release` in the `engine` directory.",
            engine_path
        ));
    }

    let daemonize = Daemonize::new()
        .pid_file(get_pids_path()?.join("engine.pid"))
        .working_directory(pie_home)
        .stdout(stdout)
        .stderr(stderr);

    match daemonize.start() {
        Ok(_) => {
            println!("✅ Engine process launched in background.");
            let mut cmd = StdCommand::new(engine_path);
            cmd.arg("--config").arg(engine_config_path);
            let child = cmd.spawn().context("Failed to spawn engine process")?;

            let mut state = load_pid_state().unwrap_or_default();
            state.engine = Some(child.id());
            save_pid_state(&state)?;

            println!("Engine PID: {}. Waiting for it to initialize...", child.id());
            tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

            if is_engine_responsive().await {
                println!("✨ Engine is up and running!");
            } else {
                println!("⚠️ Engine process started but is not responsive. Check logs at: {:?}/logs/", get_pie_home()?);
            }
        }
        Err(e) => return Err(anyhow!("Failed to daemonize: {}", e)),
    }
    Ok(())
}

async fn handle_stop() -> Result<()> {
    println!("🔌 Stopping PIE services...");
    let state = load_pid_state()?;
    let mut s = System::new_all();
    s.refresh_all();

    if let Some(pid_val) = state.engine {
        if let Some(process) = s.process(Pid::from_u32(pid_val)) {
            println!("- Stopping engine (PID: {})", pid_val);
            process.kill();
        }
    }
    for backend in state.backends {
        if let Some(process) = s.process(Pid::from_u32(backend.pid)) {
            println!("- Stopping backend for '{}' (PID: {})", backend.model, backend.pid);
            process.kill();
        }
    }

    fs::remove_file(get_pids_path()?)?;
    fs::remove_file(get_engine_config_path()?)?;
    println!("✅ All services stopped and state cleared.");
    Ok(())
}

async fn handle_status() -> Result<()> {
    let pid = match get_running_engine_pid() {
        Ok(pid) => pid.to_string(),
        Err(_) => "Not Running".to_string(),
    };

    let mut table = Table::new();
    table.load_preset(UTF8_FULL).set_header(vec!["Service", "Status", "Details"]);
    table.add_row(vec!["PIE Engine", &pid, ""]);
    println!("{table}");
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
            let model_parts: Vec<&str> = args.model_name.splitn(2, '-').collect();
            let model_group = model_parts[0];

            let model_index_base = "https://raw.githubusercontent.com/pie-project/model-index/main";
            let metadata_url = format!("{}/{}/{}.toml", model_index_base, model_group, args.model_name);
            let vocab_url_base = format!("{}/{}/", model_index_base, model_group);

            let metadata_raw = download_file_with_progress(&metadata_url, "Downloading metadata...").await?;
            let metadata_str = String::from_utf8(metadata_raw)
                .context("Failed to parse model metadata as UTF-8")?;
            let metadata: toml::Value = toml::from_str(&metadata_str)?;

            let model_dir = get_pie_home()?.join("models").join(&args.model_name);
            fs::create_dir_all(&model_dir)?;
            fs::write(model_dir.join(format!("{}.toml", args.model_name)), &metadata_str)?;

            if let Some(tokenizer) = metadata.get("tokenizer").and_then(|t| t.as_table()) {
                if let Some(vocab_file) = tokenizer.get("vocabulary_file").and_then(|v| v.as_str()) {
                    let vocab_url = format!("{}{}", vocab_url_base, vocab_file);
                    let vocab_data = download_file_with_progress(&vocab_url, &format!("Downloading {}...", vocab_file)).await?;
                    fs::write(model_dir.join(vocab_file), vocab_data)?;
                }
            }

            if let Some(source) = metadata.get("source").and_then(|s| s.as_table()) {
                for (name, url_val) in source {
                    if let Some(url) = url_val.as_str() {
                        let file_data = download_file_with_progress(url, &format!("Downloading {}...", name)).await?;
                        fs::write(model_dir.join(name), file_data)?;
                    }
                }
            }
            println!("✅ Model '{}' added successfully!", args.model_name);
        }
        ModelCommands::Remove(args) => {
            println!("🗑️ Removing model: {}", args.model_name);
            let model_dir = get_pie_home()?.join("models").join(&args.model_name);
            if model_dir.exists() {
                fs::remove_dir_all(model_dir)?;
                println!("✅ Model '{}' removed.", args.model_name);
            } else {
                return Err(anyhow!("Model '{}' not found locally.", args.model_name));
            }
        }
        ModelCommands::Serve(args) => return Err(anyhow!("`model serve` is not yet implemented.")),
    }
    Ok(())
}

async fn handle_run(args: RunArgs) -> Result<()> {
    let engine_config = load_engine_config()?;
    let url = format!("ws://{}:{}", engine_config.host, engine_config.port);
    let (ws_stream, _) = connect_async(url).await.context("Failed to connect to engine WebSocket")?;
    let (mut write, mut read) = ws_stream.split();

    // Authenticate
    let token = generate_jwt(&engine_config.auth_secret)?;
    let auth_msg = ClientMessage::Authenticate { corr_id: 1, token };
    write.send(Message::Text(serde_json::to_string(&auth_msg)?)).await?;

    let auth_response = read.next().await.ok_or(anyhow!("Did not receive auth response"))??;
    if let Message::Text(text) = auth_response {
        let server_msg: ServerMessage = serde_json::from_str(&text)?;
        if let ServerMessage::Response { successful, .. } = server_msg {
            if !successful {
                return Err(anyhow!("Authentication failed"));
            }
            println!("🔐 Authenticated successfully.");
        }
    }

    // Check/Upload Wasm
    let wasm_blob = fs::read(&args.wasm_path_or_hash).context("Failed to read Wasm file")?;
    let hash: String = Sha256::digest(&wasm_blob).encode_hex();
    println!("Program hash: {}", hash);

    let exists_msg = ClientMessage::ProgramExists { corr_id: 2, hash: hash.clone() };
    write.send(Message::Text(serde_json::to_string(&exists_msg)?)).await?;

    let exists_response = read.next().await.ok_or(anyhow!("No response for program_exists"))??;
    let exists = if let Message::Text(text) = exists_response {
        let server_msg: ServerMessage = serde_json::from_str(&text)?;
        if let ServerMessage::Response { successful, result, .. } = server_msg {
            successful && result.as_bool().unwrap_or(false)
        } else { false }
    } else { false };

    if !exists {
        println!("Program not found on server, uploading...");
        let upload_msg = ClientMessage::UploadProgram { corr_id: 3, program_blob: &wasm_blob };
        write.send(Message::Text(serde_json::to_string(&upload_msg)?)).await?;
        let upload_response = read.next().await.ok_or(anyhow!("No response for upload"))??;
        if let Message::Text(text) = upload_response {
            let server_msg: ServerMessage = serde_json::from_str(&text)?;
            if let ServerMessage::Response { successful, .. } = server_msg {
                if !successful { return Err(anyhow!("Upload failed")); }
                println!("✅ Upload successful.");
            }
        }
    } else {
        println!("Program already exists on server.");
    }

    // Launch instance
    println!("🚀 Launching instance...");
    let launch_msg = ClientMessage::LaunchInstance { corr_id: 4, hash: hash.clone() };
    write.send(Message::Text(serde_json::to_string(&launch_msg)?)).await?;

    let launch_response = read.next().await.ok_or(anyhow!("No response for launch"))??;
    let instance_id = if let Message::Text(text) = launch_response {
        let server_msg: ServerMessage = serde_json::from_str(&text)?;
        if let ServerMessage::Response { successful, result, .. } = server_msg {
            if !successful { return Err(anyhow!("Launch failed: {}", result)); }
            result.get("instance_id").and_then(|v| v.as_str()).map(String::from).ok_or(anyhow!("Missing instance_id"))?
        } else { return Err(anyhow!("Unexpected launch response")); }
    } else { return Err(anyhow!("Unexpected launch response format")); };

    println!("✅ Instance launched with ID: {}", instance_id);

    if args.detach {
        return Ok(());
    }

    println!("Attaching to instance output... (Ctrl+C to exit)");
    while let Some(msg) = read.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                let event: ServerMessage = serde_json::from_str(&text)?;
                if let ServerMessage::InstanceEvent { event, message, .. } = event {
                    if event == "terminated" {
                        println!("Instance terminated. Reason: {}", message);
                        break;
                    } else {
                        println!("[{}] {}", event, message);
                    }
                }
            },
            Err(e) => {
                eprintln!("WebSocket error: {}", e);
                break;
            }
            _ => {}
        }
    }
    Ok(())
}


//================================================================================//
// SECTION: Helpers
//================================================================================//

fn get_pie_home() -> Result<PathBuf> {
    if let Ok(path) = env::var("PIE_HOME") {
        Ok(PathBuf::from(path))
    } else {
        dirs::cache_dir()
            .map(|p| p.join("pie"))
            .ok_or_else(|| anyhow!("Failed to find user cache directory"))
    }
}

fn get_pids_path() -> Result<PathBuf> { Ok(get_pie_home()?.join("runtime/pids.json")) }
fn get_engine_config_path() -> Result<PathBuf> { Ok(get_pie_home()?.join("runtime/engine.toml")) }

fn load_pid_state() -> Result<PidState> {
    let path = get_pids_path()?;
    if !path.exists() { return Ok(PidState::default()); }
    let file = File::open(path)?;
    Ok(serde_json::from_reader(file)?)
}

fn save_pid_state(state: &PidState) -> Result<()> {
    let dir = get_pie_home()?.join("runtime");
    fs::create_dir_all(&dir)?;
    let file = File::create(dir.join("pids.json"))?;
    serde_json::to_writer_pretty(file, state)?;
    Ok(())
}

fn load_engine_config() -> Result<EngineConfig> {
    let path = get_engine_config_path()?;
    let content = fs::read_to_string(path).context("Engine not started or config missing. Run `pie start`.")?;
    Ok(toml::from_str(&content)?)
}

fn save_engine_config(config: &EngineConfig) -> Result<()> {
    let dir = get_pie_home()?.join("runtime");
    fs::create_dir_all(&dir)?;
    fs::write(dir.join("engine.toml"), toml::to_string(config)?)?;
    Ok(())
}

fn get_running_engine_pid() -> Result<u32> {
    let pid = load_pid_state()?.engine.ok_or(anyhow!("Engine not running"))?;
    let mut s = System::new_all();
    s.refresh_process(Pid::from_u32(pid));
    if s.process(Pid::from_u32(pid)).is_some() {
        Ok(pid)
    } else {
        Err(anyhow!("Stale PID found. Engine is not running."))
    }
}

async fn is_engine_responsive() -> bool {
    if let Ok(config) = load_engine_config() {
        let url = format!("ws://{}:{}", config.host, config.port);
        return tokio::time::timeout(tokio::time::Duration::from_secs(2), connect_async(url)).await.is_ok();
    }
    false
}

async fn download_file_with_progress(url: &str, message: &str) -> Result<Vec<u8>> {
    let client = Client::new();
    let res = client.get(url).send().await?.error_for_status()?;
    let total_size = res.content_length().unwrap_or(0);

    let pb = ProgressBar::new(total_size);
    pb.set_style(ProgressStyle::default_bar()
        .template("{msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec})")?
        .progress_chars("##-"));
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
    pb.finish_with_message("Downloaded.");
    Ok(content)
}

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    role: String,
    exp: usize,
}

fn generate_jwt(secret: &str) -> Result<String> {
    let expiration = Utc::now().checked_add_signed(Duration::hours(1)).expect("valid timestamp").timestamp();
    let claims = Claims {
        sub: "cli_user".to_owned(),
        role: "User".to_owned(),
        exp: expiration as usize,
    };
    encode(&Header::default(), &claims, &EncodingKey::from_secret(secret.as_ref()))
        .map_err(|e| anyhow!("JWT generation failed: {}", e))
}