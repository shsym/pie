//! Pretty spinning wheel/loading indicator for CLI operations.

use std::io::{self, Write};
use tokio::time::Duration;

/// A simple spinner utility for showing progress during long-running operations
pub struct Spinner {
    chars: Vec<char>,
    index: usize,
    message: String,
    is_running: bool,
}

impl Spinner {
    /// Create a new spinner with a default set of characters
    pub fn new(message: &str) -> Self {
        Self {
            chars: vec!['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'],
            index: 0,
            message: message.to_string(),
            is_running: false,
        }
    }

    /// Create a new spinner with custom characters
    pub fn with_chars(message: &str, chars: Vec<char>) -> Self {
        Self {
            chars,
            index: 0,
            message: message.to_string(),
            is_running: false,
        }
    }

    /// Update the spinner message while it's running
    pub fn update_message(&mut self, message: &str) {
        self.message = message.to_string();
    }

    /// Advance the spinner and show the current frame
    pub fn tick(&mut self) {
        if !self.is_running {
            self.is_running = true;
        }
        
        print!("\r{} {}", self.chars[self.index], self.message);
        io::stdout().flush().unwrap();
        self.index = (self.index + 1) % self.chars.len();
    }

    /// Clear the spinner line and optionally show a completion message
    pub fn finish(&mut self, completion_message: Option<&str>) {
        if self.is_running {
            // Clear the spinner line
            print!("\r{}", " ".repeat(self.message.len() + 5));
            print!("\r");
            
            if let Some(msg) = completion_message {
                println!("{}", msg);
            }
            
            io::stdout().flush().unwrap();
            self.is_running = false;
        }
    }

    /// Finish with a success message (shows ✓)
    pub fn finish_success(&mut self, message: &str) {
        if message.is_empty() {
            // Just clear the spinner without showing any message
            self.finish(None);
        } else {
            self.finish(Some(&format!("✓ {}", message)));
        }
    }

    /// Finish with an error message (shows ✗)
    pub fn finish_error(&mut self, message: &str) {
        self.finish(Some(&format!("✗ {}", message)));
    }
}

/// Run a future while showing a spinner
pub async fn with_spinner<F, T, E>(
    future: F,
    initial_message: &str,
    success_message: &str,
) -> Result<T, E>
where
    F: std::future::Future<Output = Result<T, E>>,
{
    let mut spinner = Spinner::new(initial_message);
    let mut interval = tokio::time::interval(Duration::from_millis(100));
    
    tokio::select! {
        result = future => {
            match result {
                Ok(value) => {
                    spinner.finish_success(success_message);
                    Ok(value)
                }
                Err(e) => {
                    spinner.finish(None);
                    Err(e)
                }
            }
        }
        _ = async {
            loop {
                interval.tick().await;
                spinner.tick();
            }
        } => {
            unreachable!("Spinner loop should never complete")
        }
    }
}

/// Run a future while showing a spinner with custom completion handling
pub async fn with_spinner_custom<F, T, E>(
    future: F,
    initial_message: &str,
    on_success: impl FnOnce(&T) -> String,
    on_error: impl FnOnce(&E) -> String,
) -> Result<T, E>
where
    F: std::future::Future<Output = Result<T, E>>,
{
    let mut spinner = Spinner::new(initial_message);
    let mut interval = tokio::time::interval(Duration::from_millis(100));
    
    tokio::select! {
        result = future => {
            match result {
                Ok(ref value) => {
                    let success_msg = on_success(value);
                    spinner.finish_success(&success_msg);
                    result
                }
                Err(ref e) => {
                    let error_msg = on_error(e);
                    spinner.finish_error(&error_msg);
                    result
                }
            }
        }
        _ = async {
            loop {
                interval.tick().await;
                spinner.tick();
            }
        } => {
            unreachable!("Spinner loop should never complete")
        }
    }
}
