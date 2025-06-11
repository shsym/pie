use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::process::Command;
use tempfile::TempDir;

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
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("controller").arg("status");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Engine Manager: Not running or not accessible"));
}

#[tokio::test]
async fn test_controller_start_without_engine_manager_binary() {
    // This test verifies error handling when engine-manager binary is not available
    // We need to test from a directory where relative paths won't work
    let ctx = TestContext::new();

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
        .current_dir(ctx.temp_dir.path()); // Run from temp dir where relative paths fail

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
async fn test_backend_list_command_stub() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("backend").arg("list");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("will be implemented in Phase"));
}

#[tokio::test]
async fn test_backend_start_command_stub() {
    let mut cmd = Command::new(TestContext::get_cli_binary());
    cmd.arg("backend")
        .arg("start")
        .arg("test-backend-type");
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("will be implemented in Phase"));
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
    let ctx = TestContext::new();
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
        .current_dir(ctx.temp_dir.path()); // Run from temp dir where relative paths fail

    // This should fail because engine-manager binary doesn't exist in PATH or relative paths
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Could not find pie_engine_manager binary"));
}

// Additional tests that would be useful for end-to-end testing
// (These would be implemented as the actual functionality is built)

#[tokio::test]
#[ignore] // Ignored until Phase 2/3 implementation
async fn test_controller_with_mock_engine_manager() {
    // This test would create a mock engine-manager binary and test the full flow
    // Implementation would be added in Phase 2
}

#[tokio::test]
#[ignore] // Ignored until Phase 3 implementation
async fn test_backend_registration_flow() {
    // This test would verify backend registration with the engine-manager
    // Implementation would be added in Phase 3
}

#[tokio::test]
#[ignore] // Ignored until Phase 4 implementation
async fn test_model_management_flow() {
    // This test would verify model loading/unloading through the CLI
    // Implementation would be added in Phase 4
}

#[tokio::test]
#[ignore] // Ignored until Phase 5 implementation
async fn test_application_deployment_flow() {
    // This test would verify application deployment and execution
    // Implementation would be added in Phase 5
}
