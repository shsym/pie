//! Pretty spinning wheel/loading indicator for CLI operations.

use std::io::{self, Write};
use tokio::time::Duration;
use crate::constants::spinner as constants;

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
    }    /// Clear the spinner line and optionally show a completion message
    pub fn finish(&mut self, completion_message: Option<&str>) {
        if self.is_running {
            if let Some(msg) = completion_message {
                // Clear the spinner line and show completion message
                print!("\r{}", " ".repeat(self.message.len() + constants::CLEAR_LINE_PADDING));
                print!("\r");
                println!("{}", msg);
            } else {
                // Just move to the beginning of the line without clearing
                // This allows the next spinner to take over seamlessly
                print!("\r");
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
    let mut interval = tokio::time::interval(constants::TICK_DURATION);

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

/// Run a future while showing a spinner with dynamic message updates
pub async fn with_dynamic_spinner<F, T, E>(
    future: F,
    initial_message: &str,
    success_message: &str,
    message_updates: Vec<(Duration, String)>, // (after_duration, new_message)
) -> Result<T, E>
where
    F: std::future::Future<Output = Result<T, E>>,
{
    let mut spinner = Spinner::new(initial_message);
    let mut interval = tokio::time::interval(constants::TICK_DURATION);
    let start_time = std::time::Instant::now();
    let mut update_index = 0;

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

                // Check if we should update the message
                let elapsed = start_time.elapsed();
                if update_index < message_updates.len() {
                    let (update_time, ref new_message) = message_updates[update_index];
                    if elapsed >= update_time {
                        spinner.update_message(new_message);
                        update_index += 1;
                    }
                }
            }
        } => {
            unreachable!("Spinner loop should never complete")
        }
    }
}
