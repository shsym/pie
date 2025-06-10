use clap::{Parser, Subcommand};
use anyhow::Result;

mod controller;
mod backend;
mod model;
mod application;

use controller::ControllerCommands;
use backend::BackendCommands;
use model::ModelCommands;
use application::ApplicationCommands;

#[derive(Parser)]
#[command(name = "pie-cli")]
#[command(about = "Pie CLI - Unified interface for managing Pie system components")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Controller management commands
    #[command(subcommand)]
    Controller(ControllerCommands),

    /// Backend management commands
    #[command(subcommand)]
    Backend(BackendCommands),

    /// Model management commands
    #[command(subcommand)]
    Model(ModelCommands),

    /// Application management commands
    #[command(subcommand)]
    Application(ApplicationCommands),
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Controller(cmd) => controller::handle_command(cmd).await,
        Commands::Backend(cmd) => backend::handle_command(cmd).await,
        Commands::Model(cmd) => model::handle_command(cmd).await,
        Commands::Application(cmd) => application::handle_command(cmd).await,
    }
}
