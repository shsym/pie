use std::path::Path;

use ::client::client::Client;
use anyhow::{Context, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: {} <ws_host> <inferlet_name@version> <wasm_path> <manifest_path> [input_json]",
            args[0]
        );
        std::process::exit(2);
    }
    let ws_host = &args[1];
    let inferlet = &args[2];
    let wasm_path = Path::new(&args[3]);
    let manifest_path = Path::new(&args[4]);
    let input = args.get(5).cloned().unwrap_or_else(|| "{}".to_string());

    let result = submit_inferlet(ws_host, inferlet, wasm_path, manifest_path, &input).await?;
    println!("{result}");
    Ok(())
}

pub async fn submit_inferlet(
    ws_host: &str,
    inferlet: &str,
    wasm_path: &Path,
    manifest_path: &Path,
    input: &str,
) -> Result<String> {
    let identity = std::env::var("PIE_IDENTITY").unwrap_or_else(|_| "test-user".to_string());
    let client = Client::connect_with_identity(ws_host, &identity)
        .await
        .with_context(|| format!("connect to engine at {ws_host}"))?;

    client
        .authenticate("test-user", &None)
        .await
        .context("authenticate")?;

    client
        .add_program(wasm_path, manifest_path, true)
        .await
        .with_context(|| format!("add_program {inferlet}"))?;

    let mut proc = client
        .launch_process(inferlet.to_string(), input.to_string(), true)
        .await
        .with_context(|| format!("launch_process {inferlet}"))?;

    proc.wait_for_return().await
}
