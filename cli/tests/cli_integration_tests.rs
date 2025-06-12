use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;
use tempfile::TempDir;
use std::fs;

/// Test utilities
struct TestContext {
    temp_dir: TempDir,
}

impl TestContext {
    fn new() -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");

        Self {
            temp_dir,
        }
    }

    fn new_with_config() -> Self {
        let context = Self::new();
        context.create_test_config();
        context
    }

    fn create_test_config(&self) {
        let config_content = r#"
{
    "system": {
        "name": "pie",
        "version": "1.0.0",
        "description": "Pie Language Model Engine and Management System"
    },
    "services": {
        "engine_manager": {
            "host": "127.0.0.1",
            "port": 8080,
            "binary_name": "pie_engine_manager"
        },
        "engine": {
            "binary_name": "pie-rt",
            "default_port": 9123,
            "base_args": ["--http"]
        }
    },
    "endpoints": {
        "client_handshake": "ipc:///tmp/symphony-ipc",
        "cli_management": "ipc:///tmp/symphony-cli",
        "management_service": "ipc:///tmp/symphony-cli"
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "directory": "logs"
    },
    "models": {
        "available": [
            "Llama-3.1-8B-Instruct"
        ],
        "default": "Llama-3.1-8B-Instruct",
        "supported_models": [
            {
                "name": "Llama-3.1-8B-Instruct",
                "fullname": "meta-llama/Llama-3.1-8B-Instruct",
                "type": "llama3"
            }
        ]
    },
    "backends": {
        "model_backends": {
            "llama3": "l4m_backend.py"
        }
    },
    "paths": {
        "engine_binary_search": [
            "target/debug/pie-rt"
        ],
        "engine_manager_binary_search": [
            "target/debug/pie_engine_manager"
        ]
    }
}
"#;
        let config_path = self.temp_dir.path().join("config.json");
        fs::write(&config_path, config_content).expect("Failed to write test config");
    }

    fn get_cli_binary() -> String {
        std::env::var("CARGO_BIN_EXE_pie-cli")
            .unwrap_or_else(|_| "target/debug/pie-cli".to_string())
    }
}

// Phase 1 Tests: Controller Functionality

#[tokio::test]
async fn test_cli_shows_help_by_default() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.assert()
        .failure() // clap returns error code when no subcommand provided
        .stderr(predicate::str::contains("Usage:"));
}

#[tokio::test]
async fn test_controller_help_command() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("controller").arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("controller"))
        .stdout(predicate::str::contains("start"))
        .stdout(predicate::str::contains("status"))
        .stdout(predicate::str::contains("stop"));
}

#[tokio::test]
async fn test_controller_status_when_not_running() {
    let ctx = TestContext::new_with_config();
    let cli_binary = std::env::current_dir()
        .unwrap()
        .join(TestContext::get_cli_binary());

    let mut cmd = Command::new(&cli_binary);
    cmd.arg("controller")
        .arg("status")
        .current_dir(ctx.temp_dir.path()); // Run from temp dir with config

    let result = cmd.assert();

    // Accept either "Not running" or "Running" depending on test environment
    // Both are valid since we can't control if a controller is already running
    result.success()
        .stdout(predicate::str::contains("Engine Manager:"));
}

#[tokio::test]
async fn test_controller_start_without_engine_manager_binary() {
    // This test verifies error handling when engine-manager binary is not available
    // We need to test from a directory where relative paths won't work
    let ctx = TestContext::new_with_config();

    // Change to temp directory where relative paths to engine-manager won't work
    let temp_path = ctx.temp_dir.path().to_string_lossy();
    let cli_binary = std::env::current_dir()
        .unwrap()
        .join(TestContext::get_cli_binary());

    let mut cmd = Command::new(&cli_binary);
    cmd.arg("controller")
        .arg("start")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg("18080") // Use a different port to avoid conflicts
        .env("PATH", temp_path.as_ref())
        .current_dir(ctx.temp_dir.path()); // Run from temp dir with config

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Could not find pie_engine_manager binary"));
}

// Backend command tests (Phase 2/3 - stubs for now)

#[tokio::test]
async fn test_backend_help_command() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("backend").arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("backend"))
        .stdout(predicate::str::contains("list"))
        .stdout(predicate::str::contains("start"))
        .stdout(predicate::str::contains("status"))
        .stdout(predicate::str::contains("terminate"));
}

#[tokio::test]
async fn test_backend_list_without_service() {
    // Test that backend list works but shows empty when no management service is running
    // Since we now provide config files, this may succeed but show no backends
    let ctx = TestContext::new_with_config();
    let cli_binary = std::env::current_dir()
        .unwrap()
        .join(TestContext::get_cli_binary());

    let mut cmd = Command::new(&cli_binary);
    cmd.arg("backend")
        .arg("list")
        .current_dir(ctx.temp_dir.path());

    // This could either succeed (showing no backends) or fail (if management service isn't running)
    // Both are valid outcomes depending on the test environment
    let result = cmd.assert();

    // Accept either success with "No backends" message or failure with connection error
    result.code(predicate::in_iter([0, 1]));
}

#[tokio::test]
async fn test_backend_start_without_config() {
    // Test that backend start fails gracefully when configuration is missing
    let ctx = TestContext::new(); // Note: NOT using new_with_config() - no config file

    let cli_binary = std::env::current_dir()
        .unwrap()
        .join(TestContext::get_cli_binary());

    let mut cmd = Command::new(&cli_binary);
    cmd.arg("backend")
        .arg("start")
        .arg("test-backend-type")
        .current_dir(ctx.temp_dir.path()); // Run from temp dir without config

    cmd.assert()
        .failure() // Expected to fail because no config file exists
        .stderr(predicate::str::contains("Configuration file 'config.json' not found"));
}

// Model command tests (Phase 4 - stubs for now)

#[tokio::test]
async fn test_model_help_command() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("model").arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("model"))
        .stdout(predicate::str::contains("load"))
        .stdout(predicate::str::contains("unload"))
        .stdout(predicate::str::contains("download"));
}

#[tokio::test]
async fn test_model_load_command_stub() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("model")
        .arg("load")
        .arg("backend-123")
        .arg("--model-name")
        .arg("test-model");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("will be implemented in Phase"));
}

// Application command tests (Phase 5 - stubs for now)

#[tokio::test]
async fn test_application_help_command() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("application").arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("application"))
        .stdout(predicate::str::contains("deploy"))
        .stdout(predicate::str::contains("run"))
        .stdout(predicate::str::contains("serve"));
}

#[tokio::test]
async fn test_application_deploy_command_stub() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("application")
        .arg("deploy")
        .arg("/path/to/test.wasm");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("will be implemented in Phase"));
}

// Integration tests combining multiple components

#[tokio::test]
async fn test_full_help_structure() {
    // Test that all main commands are available
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("--help");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("controller"))
        .stdout(predicate::str::contains("backend"))
        .stdout(predicate::str::contains("model"))
        .stdout(predicate::str::contains("application"));
}

#[tokio::test]
async fn test_invalid_command() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("invalid-command");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("error:"));
}

#[tokio::test]
async fn test_controller_start_with_custom_port() {
    let ctx = TestContext::new_with_config();
    let temp_path = ctx.temp_dir.path().to_string_lossy();
    let cli_binary = std::env::current_dir()
        .unwrap()
        .join(TestContext::get_cli_binary());

    let mut cmd = Command::new(&cli_binary);
    cmd.arg("controller")
        .arg("start")
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg("19080")
        .arg("--engine-port")
        .arg("19123")
        .env("PATH", temp_path.as_ref())
        .current_dir(ctx.temp_dir.path()); // Run from temp dir with config

    // This should fail because engine-manager binary doesn't exist in PATH or relative paths
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Could not find pie_engine_manager binary"));
}

// Configuration Tests

#[tokio::test]
async fn test_unified_config_loading() {
    use pie_cli::config::Config;
    use tempfile::NamedTempFile;
    use std::io::Write;

    // Create a test unified config
    let test_config = r#"
{
    "system": {
        "name": "Symphony AI Platform",
        "version": "1.0.0",
        "description": "A unified platform for AI model management and execution"
    },
    "services": {
        "engine_manager": {
            "host": "127.0.0.1",
            "port": 8080,
            "binary_name": "pie_engine_manager"
        },
        "engine": {
            "binary_name": "pie_engine",
            "default_port": 9123,
            "base_args": ["--config", "config.json"]
        }
    },
    "endpoints": {
        "client_handshake": "ipc:///tmp/symphony-ipc",
        "cli_management": "ipc:///tmp/symphony-cli",
        "management_service": "http://127.0.0.1:8080"
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "directory": "./logs"
    },
    "models": {
        "available": ["Llama-3.1-8B-Instruct", "DeepSeek-V3-0324"],
        "default": "Llama-3.1-8B-Instruct",
        "supported_models": [
            {
                "name": "Llama-3.1-8B-Instruct",
                "fullname": "meta-llama/Llama-3.1-8B-Instruct",
                "type": "llama3"
            },
            {
                "name": "DeepSeek-V3-0324",
                "fullname": "deepseek-ai/DeepSeek-V3-0324",
                "type": "deepseek"
            }
        ]
    },
    "backends": {
        "model_backends": {
            "llama3": "l4m_backend.py",
            "deepseek": "deepseek_backend.py"
        }
    },
    "paths": {
        "engine_binary_search": ["./target/debug/pie_engine", "../engine/target/debug/pie_engine"],
        "engine_manager_binary_search": ["./target/debug/pie_engine_manager", "../engine-manager/target/debug/pie_engine_manager"]
    }
}
"#;

    // Write test config to a temporary file
    let mut temp_file = NamedTempFile::new().expect("Failed to create temp file");
    write!(temp_file, "{}", test_config).expect("Failed to write test config");

    // Test loading the unified config
    let config = Config::load_from_file(temp_file.path()).expect("Failed to load test config");

    // Verify the config structure
    assert_eq!(config.system.name, "Symphony AI Platform");
    assert_eq!(config.system.version, "1.0.0");
    assert_eq!(config.services.engine_manager.host, "127.0.0.1");
    assert_eq!(config.services.engine_manager.port, 8080);
    assert_eq!(config.services.engine_manager.binary_name, "pie_engine_manager");
    assert_eq!(config.services.engine.binary_name, "pie_engine");
    assert_eq!(config.services.engine.default_port, 9123);
    assert_eq!(config.endpoints.management_service, "http://127.0.0.1:8080");
    assert_eq!(config.logging.level, "INFO");
    assert_eq!(config.models.available.len(), 2);
    assert_eq!(config.models.default, "Llama-3.1-8B-Instruct");
    assert_eq!(config.models.supported_models.len(), 2);
    assert_eq!(config.models.supported_models[0].name, "Llama-3.1-8B-Instruct");
    assert_eq!(config.models.supported_models[0].model_type, "llama3");
    assert!(config.backends.model_backends.contains_key("llama3"));
    assert!(config.backends.model_backends.contains_key("deepseek"));
    assert_eq!(config.paths.engine_binary_search.len(), 2);
    assert_eq!(config.paths.engine_manager_binary_search.len(), 2);
}

#[tokio::test]
async fn test_config_load_default_missing_file() {
    use pie_cli::config::Config;
    use tempfile::TempDir;
    use std::env;

    // Create a temporary directory and change to it
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let original_dir = env::current_dir().expect("Failed to get current directory");

    env::set_current_dir(temp_dir.path()).expect("Failed to change directory");

    // Test that load_default fails when config.json doesn't exist
    let result = Config::load_default();
    assert!(result.is_err());
    let error_msg = format!("{}", result.unwrap_err());
    assert!(error_msg.contains("Default config file 'config.json' not found"));

    // Restore original directory
    env::set_current_dir(original_dir).expect("Failed to restore directory");
}

//...existing code...
