#![allow(unused_imports)] // Allow unused imports for now, will be used later

//! Management CLI binary entry point.

use backend_management_rs::cli::{CliArgs, process_cli_command};
use clap::Parser;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = CliArgs::parse();

    // TODO: Initialize ZMQ client, send command, handle response
    // For now, we'll call a placeholder processing function.
    process_cli_command(args).await;

    Ok(())
}
