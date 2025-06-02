use backend_management_rs::model_installer::ModelInstaller;
use std::path::PathBuf;
use tempfile::TempDir;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🧪 Testing model download with huggingface-cli");

    // Create a temporary directory for testing
    let temp_dir = TempDir::new()?;
    println!("📁 Using temporary directory: {:?}", temp_dir.path());

    // Create model installer with custom huggingface-cli path
    let installer = ModelInstaller::new(Some(temp_dir.path().to_path_buf()))
        .with_hf_cli_path("/home/sslee/Workspace/symphony/.venv/bin/huggingface-cli".to_string());

    // Test downloading a very small model
    let model_name = "gpt2"; // This is a small model that should download quickly
    println!("⬇️  Attempting to download model: {}", model_name);

    match installer.install_model(model_name).await {
        Ok(model_path) => {
            println!("✅ Successfully downloaded model to: {:?}", model_path);

            // Check if key files exist
            let config_file = model_path.join("config.json");
            let model_file = model_path.join("pytorch_model.bin");
            let safetensors_file = model_path.join("model.safetensors");
            let tokenizer_file = model_path.join("tokenizer.json");

            println!("📋 Checking downloaded files:");
            println!("  config.json: {}", config_file.exists());
            println!("  pytorch_model.bin: {}", model_file.exists());
            println!("  model.safetensors: {}", safetensors_file.exists());
            println!("  tokenizer.json: {}", tokenizer_file.exists());

            // Check if our symphony info file was created
            let symphony_info = model_path.join("symphony_model_info.json");
            println!("  symphony_model_info.json: {}", symphony_info.exists());

            if symphony_info.exists() {
                let content = tokio::fs::read_to_string(&symphony_info).await?;
                println!("📄 Symphony model info:");
                println!("{}", content);
            }

            // Test model info retrieval
            match installer.get_model_info(model_name).await {
                Ok(info) => {
                    println!("📊 Model info retrieved:");
                    println!("  Model name: {}", info.model_name);
                    println!("  Local name: {}", info.local_name);
                    println!("  Model type: {}", info.model_type);
                    println!("  Architectures: {:?}", info.architectures);
                    println!("  Installed at: {:?}", info.installed_at);
                }
                Err(e) => {
                    println!("⚠️  Failed to get model info: {}", e);
                }
            }

        }
        Err(e) => {
            println!("❌ Failed to download model: {}", e);
            return Err(e.into());
        }
    }

    println!("🎉 Test completed successfully!");
    Ok(())
}
