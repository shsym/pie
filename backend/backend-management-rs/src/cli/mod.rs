//! CLI module - main interface for command-line interaction.

pub mod cli;
pub mod zmq_client;
pub mod spinner;

// Re-export the main types from cli.rs
pub use cli::{CliArgs, Commands, process_cli_command};
