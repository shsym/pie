/// Constants for the CLI application
/// This module contains all the hardcoded values used throughout the CLI
/// to improve maintainability and make configuration easier.

/// Spinner and timing constants
pub mod spinner {
    use std::time::Duration;

    /// Number of initial spinner ticks for controller commands (start/stop)
    /// This provides immediate visual feedback to the user
    pub const CONTROLLER_INITIAL_TICKS: usize = 5;

    /// Number of initial spinner ticks for backend commands
    /// Slightly fewer ticks as backend operations are typically faster
    pub const BACKEND_INITIAL_TICKS: usize = 3;

    /// Duration between spinner ticks (100ms)
    /// This creates a smooth spinning animation without being too fast or slow
    pub const TICK_DURATION: Duration = Duration::from_millis(100);

    /// Duration for each sleep cycle during startup wait (100ms)
    /// Used for non-blocking waits that allow the spinner to keep animating
    pub const STARTUP_SLEEP_DURATION: Duration = Duration::from_millis(100);

    /// Number of sleep cycles for controller startup wait (20 * 100ms = 2 seconds)
    /// Provides sufficient buffer time for the engine-manager process to start
    pub const CONTROLLER_STARTUP_CYCLES: u32 = 20;

    /// Number of cycles before health checks start (10 * 100ms = 1 second)
    /// Allows process to initialize before attempting health checks
    pub const HEALTH_CHECK_START_CYCLES: u32 = 10;

    /// Clear line padding for spinner messages (5 characters)
    /// Extra space to ensure the spinner line is fully cleared
    pub const CLEAR_LINE_PADDING: usize = 5;
}

/// Network and connection constants
pub mod network {
    /// Default localhost IP address
    pub const DEFAULT_HOST: &str = "127.0.0.1";

    /// Default controller HTTP port
    pub const DEFAULT_HTTP_PORT: u16 = 8080;

    /// Default backend gRPC port
    pub const DEFAULT_GRPC_PORT: u16 = 9123;

    /// Default management service URL
    pub const DEFAULT_MANAGEMENT_URL: &str = "http://127.0.0.1:8080";

    /// Default backend bind address (all interfaces)
    pub const DEFAULT_BACKEND_HOST: &str = "0.0.0.0";

    /// Default base port for backend services
    pub const DEFAULT_BACKEND_BASE_PORT: u16 = 8081;
}

/// File and directory constants
pub mod paths {
    /// Logs directory name
    pub const LOGS_DIR: &str = "logs";

    /// CLI log file name
    pub const CLI_LOG_FILE: &str = "cli.log";
}
