use clap::{Parser, Subcommand};
use anyhow::Result;
use std::fs;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt};
use crate::spinner::Spinner;
use crate::constants::{spinner as spinner_constants, paths};
use std::io::{self, Write};

mod controller;
mod backend;
mod model;
mod application;
mod spinner;
mod constants;

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
    // Create logs directory if it doesn't exist
    fs::create_dir_all(paths::LOGS_DIR).unwrap_or_default();

    // Set up file logging
    let file_appender = tracing_appender::rolling::never(paths::LOGS_DIR, paths::CLI_LOG_FILE);
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::registry()
        .with(
            fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false)  // No ANSI colors in log files
        )
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    // Start an immediate spinner for long-running commands
    match &cli.command {
        Commands::Controller(ControllerCommands::Start { .. }) => {
            let mut spinner = Spinner::new("Initializing controller startup...");

            // Show immediate feedback with a few spinner ticks
            for _ in 0..spinner_constants::CONTROLLER_INITIAL_TICKS {
                tokio::time::sleep(spinner_constants::TICK_DURATION).await;
                spinner.tick();
            }

            // Instead of finishing, just leave the spinner on the screen
            // The controller command will take over with its own spinner
            print!("\r"); // Just move to beginning of line, don't clear
            io::stdout().flush().unwrap();
        },
        Commands::Controller(ControllerCommands::Stop { .. }) => {
            let mut spinner = Spinner::new("Initializing controller shutdown...");

            // Show immediate feedback
            for _ in 0..spinner_constants::CONTROLLER_INITIAL_TICKS {
                tokio::time::sleep(spinner_constants::TICK_DURATION).await;
                spinner.tick();
            }

            // Don't clear the line, let the stop command take over
            print!("\r");
            io::stdout().flush().unwrap();
        },
        Commands::Backend(BackendCommands::Start { .. }) => {
            let mut spinner = Spinner::new("Initializing backend startup...");

            // Show immediate feedback
            for _ in 0..spinner_constants::BACKEND_INITIAL_TICKS {
                tokio::time::sleep(spinner_constants::TICK_DURATION).await;
                spinner.tick();
            }

            print!("\r");
            io::stdout().flush().unwrap();
        },
        _ => {}
    };

    match cli.command {
        Commands::Controller(cmd) => controller::handle_command(cmd).await,
        Commands::Backend(cmd) => backend::handle_command(cmd).await,
        Commands::Model(cmd) => model::handle_command(cmd).await,
        Commands::Application(cmd) => application::handle_command(cmd).await,
    }
}
