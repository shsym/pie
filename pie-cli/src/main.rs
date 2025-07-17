use anyhow::{anyhow, Context, Result};
use chrono::{Duration, Utc};
use clap::{Args, Parser, Subcommand};
use comfy_table::{presets::UTF8_FULL, Table};
use daemonize::Daemonize;
use dashmap::DashMap;
use futures_util::{
    stream::{SplitSink, SplitStream},
    SinkExt, StreamExt,
};
use hex::ToHex;
use indicatif::{ProgressBar, ProgressStyle};
use jsonwebtoken::{encode, EncodingKey, Header};
use rand::{distributions::Alphanumeric, Rng};
use reqwest::Client;
use rmp_serde as rmps;
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    env,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    process::Command as StdCommand,
    sync::{Arc, Mutex},
};
use sysinfo::{Pid, System};
use tokio::{
    net::TcpStream,
    sync::{mpsc, oneshot},
    task::JoinHandle,
};
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tungstenite::protocol::Message as WsMessage;
use url::Url;
use uuid::Uuid;

//================================================================================//
// SECTION: WebSocket Client & Messaging
//================================================================================//

type CorrId = u32;
type InstanceId = Uuid;

/// Messages from client -> server
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "authenticate")]
    Authenticate { corr_id: u32, token: String },

    #[serde(rename = "query")]
    Query {
        corr_id: u32,
        subject: String,
        record: String,
    },

    #[serde(rename = "upload_program")]
    UploadProgram {
        corr_id: u32,
        program_hash: String,
        chunk_index: usize,
        total_chunks: usize,
        #[serde(with = "serde_bytes")]
        chunk_data: Vec<u8>,
    },

    #[serde(rename = "launch_instance")]
    LaunchInstance { corr_id: u32, program_hash: String },

    #[serde(rename = "launch_server_instance")]
    LaunchServerInstance {
        corr_id: u32,
        port: u32,
        program_hash: String,
    },

    #[serde(rename = "signal_instance")]
    SignalInstance {
        instance_id: String,
        message: String,
    },

    #[serde(rename = "terminate_instance")]
    TerminateInstance { instance_id: String },

    #[serde(rename = "attach_remote_service")]
    AttachRemoteService {
        corr_id: u32,
        endpoint: String,
        service_type: String,
        service_name: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "response")]
    Response {
        corr_id: u32,
        successful: bool,
        result: String,
    },

    #[serde(rename = "instance_event")]
    InstanceEvent {
        instance_id: String,
        event: String,
        message: String,
    },

    #[serde(rename = "server_event")]
    ServerEvent { message: String },
}

struct IdPool<T>(Mutex<VecDeque<T>>);
impl IdPool<CorrId> {
    fn new() -> Self {
        Self(Mutex::new((1..=CorrId::MAX).collect()))
    }
    fn acquire(&self) -> Result<CorrId> {
        self.0.lock().unwrap().pop_front().ok_or(anyhow!("Correlation ID pool exhausted"))
    }
    fn release(&self, id: CorrId) {
        self.0.lock().unwrap().push_back(id);
    }
}

/// A client for interacting with the PIE engine.
struct EngineClient {
    writer_tx: mpsc::UnboundedSender<WsMessage>,
    corr_id_pool: Arc<IdPool<CorrId>>,
    pending_requests: Arc<DashMap<CorrId, oneshot::Sender<(bool, String)>>>,
    inst_event_listeners: Arc<DashMap<InstanceId, mpsc::UnboundedSender<(String, String)>>>,
    _reader_handle: JoinHandle<()>,
    _writer_handle: JoinHandle<()>,
}

impl EngineClient {
    /// Connects to the engine and spawns background tasks for communication.
    async fn connect(url: &str) -> Result<Self> {
        let (ws_stream, _) = connect_async(url).await.context("Failed to connect to engine WebSocket")?;
        let (ws_writer, ws_reader) = ws_stream.split();

        let (writer_tx, writer_rx) = mpsc::unbounded_channel();
        let pending_requests = Arc::new(DashMap::new());
        let inst_event_listeners = Arc::new(DashMap::new());
        let corr_id_pool = Arc::new(IdPool::new());

        let writer_handle = tokio::spawn(client_writer_task(ws_writer, writer_rx));
        let reader_handle = tokio::spawn(client_reader_task(ws_reader, pending_requests.clone(), inst_event_listeners.clone()));

        Ok(Self {
            writer_tx,
            corr_id_pool,
            pending_requests,
            inst_event_listeners,
            _reader_handle: reader_handle,
            _writer_handle: writer_handle,
        })
    }

    /// Sends a message and waits for a corresponding response from the server.
    async fn send_and_wait(&self, msg_builder: Box<dyn Fn(CorrId) -> ClientMessage + Send + 'static>) -> Result<(bool, String)> {
        let corr_id = self.corr_id_pool.acquire()?;
        let (tx, rx) = oneshot::channel();
        self.pending_requests.insert(corr_id, tx);

        let msg = msg_builder(corr_id);
        let encoded = rmps::to_vec_named(&msg)?;
        self.writer_tx.send(WsMessage::Binary(encoded))?;

        let result = rx.await.context("Server did not respond to request");
        self.corr_id_pool.release(corr_id);
        result
    }

    async fn authenticate(&self, token: String) -> Result<()> {
        let builder = Box::new(move |corr_id| ClientMessage::Authenticate { corr_id, token: token.clone() });
        let (success, result) = self.send_and_wait(builder).await?;
        if success { Ok(()) } else { Err(anyhow!("Authentication failed: {}", result)) }
    }

    async fn program_exists(&self, hash: &str) -> Result<bool> {
        let hash_clone = hash.to_string();
        let builder = Box::new(move |corr_id| ClientMessage::Query { corr_id, subject: "program_exists".to_string(), record: hash_clone.clone() });
        let (success, result) = self.send_and_wait(builder).await?;
        if success { Ok(result == "true") } else { Err(anyhow!("program_exists query failed: {}", result)) }
    }

    async fn upload_program(&self, blob: &[u8]) -> Result<()> {
        const CHUNK_SIZE: usize = 256 * 1024;
        let program_hash = blake3::hash(blob).to_hex().to_string();

        let (tx, rx) = oneshot::channel();
        let corr_id = self.corr_id_pool.acquire()?;
        self.pending_requests.insert(corr_id, tx);

        let total_chunks = (blob.len() as f64 / CHUNK_SIZE as f64).ceil() as usize;

        for i in 0..total_chunks {
            let start = i * CHUNK_SIZE;
            let end = (start + CHUNK_SIZE).min(blob.len());
            let chunk_data = blob[start..end].to_vec();

            let msg = ClientMessage::UploadProgram { corr_id, program_hash: program_hash.clone(), chunk_index: i, total_chunks, chunk_data };
            let encoded = rmps::to_vec_named(&msg)?;
            self.writer_tx.send(WsMessage::Binary(encoded))?;
        }

        let result = rx.await.context("Server did not respond to upload completion");
        self.corr_id_pool.release(corr_id);
        let (success, result_str) = result?;
        if success { Ok(()) } else { Err(anyhow!("Program upload failed: {}", result_str)) }
    }

    async fn launch_instance(&self, hash: &str) -> Result<(InstanceId, mpsc::UnboundedReceiver<(String, String)>)> {
        let hash_clone = hash.to_string();
        let builder = Box::new(move |corr_id| ClientMessage::LaunchInstance { corr_id, program_hash: hash_clone.clone() });
        let (success, result) = self.send_and_wait(builder).await?;

        if success {
            let instance_id = Uuid::parse_str(&result).context("Failed to parse instance_id from server response")?;
            let (tx, rx) = mpsc::unbounded_channel();
            self.inst_event_listeners.insert(instance_id, tx);
            Ok((instance_id, rx))
        } else {
            Err(anyhow!("Launch instance failed: {}", result))
        }
    }
}

/// Background task to read messages from the server and dispatch them.
async fn client_reader_task(mut ws_reader: SplitStream<WebSocketStream<MaybeTlsStream<TcpStream>>>, pending_requests: Arc<DashMap<CorrId, oneshot::Sender<(bool, String)>>>, inst_event_listeners: Arc<DashMap<InstanceId, mpsc::UnboundedSender<(String, String)>>>) {
    while let Some(Ok(msg)) = ws_reader.next().await {
        if let WsMessage::Binary(bin) = msg {
            if let Ok(server_msg) = rmps::from_slice::<ServerMessage>(&bin) {
                match server_msg {
                    ServerMessage::Response { corr_id, successful, result } => {
                        if let Some((_, sender)) = pending_requests.remove(&corr_id) {
                            let _ = sender.send((successful, result));
                        }
                    },
                    ServerMessage::InstanceEvent { instance_id, event, message } => {
                        if let Ok(id) = Uuid::parse_str(&instance_id) {
                            if let Some(sender) = inst_event_listeners.get(&id) {
                                let _ = sender.send((event, message));
                            }
                        }
                    }
                    _ => {

                    }
                }
            }
        }
    }
}

/// Background task to write messages to the server.
async fn client_writer_task(mut ws_writer: SplitSink<WebSocketStream<MaybeTlsStream<TcpStream>>, WsMessage>, mut writer_rx: mpsc::UnboundedReceiver<WsMessage>) {
    while let Some(msg) = writer_rx.recv().await {
        if ws_writer.send(msg).await.is_err() {
            break; // Connection closed
        }
    }
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

            let models_root = get_pie_home()?.join("models");
            let model_files_dir = models_root.join(&args.model_name);
            let metadata_path = models_root.join(format!("{}.toml", args.model_name));

            // Check if the model already exists and ask for confirmation to overwrite.
            if metadata_path.exists() || model_files_dir.exists() {
                print!("⚠️ Model '{}' already exists. Overwrite? [y/N] ", args.model_name);
                std::io::stdout().flush().context("Failed to flush stdout")?;

                let mut confirmation = String::new();
                std::io::stdin().read_line(&mut confirmation).context("Failed to read user input")?;

                if confirmation.trim().to_lowercase() != "y" {
                    println!("Aborted by user.");
                    return Ok(());
                }
            }

            // --- Set up directories and save files with the new structure ---
            fs::create_dir_all(&model_files_dir)?;

            println!("Parameters will be stored at {:?}", model_files_dir);

            let model_index_base = "https://raw.githubusercontent.com/pie-project/model-index/refs/heads/main";
            let metadata_url = format!("{}/{}.toml", model_index_base, args.model_name);

            // --- Download metadata with specific 404 error handling ---
            let metadata_raw = match download_file_with_progress(&metadata_url, "Downloading metadata...").await {
                Ok(data) => data,
                Err(e) => {
                    if let Some(req_err) = e.downcast_ref::<reqwest::Error>() {
                        if req_err.status() == Some(reqwest::StatusCode::NOT_FOUND) {
                            anyhow::bail!("Model '{}' not found in the official index.", args.model_name);
                        }
                    }
                    return Err(e.context("Failed to download model metadata"));
                }
            };

            let metadata_str = String::from_utf8(metadata_raw)
                .context("Failed to parse model metadata as UTF-8")?;
            let metadata: toml::Value = toml::from_str(&metadata_str)?;

            // Store the .toml metadata file
            fs::write(&metadata_path, &metadata_str)?;

            // Download all source files listed in the metadata
            if let Some(source) = metadata.get("source").and_then(|s| s.as_table()) {
                for (name, url_val) in source {
                    if let Some(url) = url_val.as_str() {
                        let file_data = download_file_with_progress(url, &format!("Downloading {}...", name)).await?;
                        fs::write(model_files_dir.join(name), file_data)?;
                    }
                }
            }
            println!("✅ Model '{}' added successfully!", args.model_name);
        }
        ModelCommands::Remove(args) => {
            println!("🗑️ Removing model: {}", args.model_name);

            // Define the base path for all models
            let models_root = get_pie_home()?.join("models");
            // Path to the subdirectory containing the model's files
            let model_files_dir = models_root.join(&args.model_name);
            // Path to the model's metadata .toml file
            let metadata_path = models_root.join(format!("{}.toml", args.model_name));

            let mut was_removed = false;

            // Attempt to remove the model's file directory if it exists
            if model_files_dir.exists() {
                fs::remove_dir_all(&model_files_dir)?;
                was_removed = true;
            }

            // Attempt to remove the model's metadata file if it exists
            if metadata_path.exists() {
                fs::remove_file(&metadata_path)?;
                was_removed = true;
            }

            if was_removed {
                println!("✅ Model '{}' removed.", args.model_name);
            } else {
                // If neither the directory nor the file was found, return an error
                anyhow::bail!("Model '{}' not found locally.", args.model_name);
            }
        }
        ModelCommands::Serve(args) => return Err(anyhow!("`model serve` is not yet implemented.")),
    }
    Ok(())
}
async fn handle_run(args: RunArgs) -> Result<()> {
    let engine_config = load_engine_config()?;
    let url = format!("ws://{}:{}", engine_config.host, engine_config.port);
    let mut client = EngineClient::connect(&url).await?;

    // Authenticate
    let token = generate_jwt(&engine_config.auth_secret)?;
    client.authenticate(token).await?;
    println!("🔐 Authenticated successfully.");

    // Check/Upload Wasm
    let wasm_blob = fs::read(&args.wasm_path_or_hash).context("Failed to read Wasm file")?;
    let hash: String = blake3::hash(&wasm_blob).to_hex().to_string();
    println!("Program hash: {}", hash);

    if !client.program_exists(&hash).await? {
        println!("Program not found on server, uploading...");
        let pb = ProgressBar::new_spinner();
        pb.set_message("Uploading program...");
        pb.enable_steady_tick(std::time::Duration::from_millis(100));
        client.upload_program(&wasm_blob).await?;
        pb.finish_with_message("✅ Upload successful.");
    } else {
        println!("Program already exists on server.");
    }

    // Launch instance
    println!("🚀 Launching instance...");
    let (instance_id, mut event_rx) = client.launch_instance(&hash).await?;
    println!("✅ Instance launched with ID: {}", instance_id);

    if args.detach {
        return Ok(());
    }

    println!("Attaching to instance output... (Ctrl+C to exit)");
    while let Some((event, message)) = event_rx.recv().await {
        if event == "terminated" {
            println!("Instance terminated. Reason: {}", message);
            break;
        } else {
            println!("[{}] {}", event, message);
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

    // Extract the filename from the URL for a more informative message
    let file_name = url
        .rsplit('/')
        .next()
        .unwrap_or("file") // Fallback for unusual URLs
        .split('?')
        .next()
        .unwrap_or("file"); // Remove query parameters

    pb.finish_with_message(format!("Downloaded {}", file_name));
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