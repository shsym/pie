//! Utility functions for the Management Service

use anyhow::{Error, Result};
use num_traits::PrimInt;
use std::collections::BTreeSet;
use std::hash::Hash;
use std::sync::atomic::{AtomicIsize, Ordering};
use uuid::Uuid;
use std::path::{Path, PathBuf};
use tracing::{info, warn};


/// Expands a path that may contain a tilde prefix (~/) to the user's home directory.
///
/// # Arguments
/// * `path` - A path string that may start with ~/
///
/// # Returns
/// A PathBuf with the tilde expanded to the actual home directory path
///
/// # Examples
/// ```
/// use backend_management_rs::path_utils::expand_home_dir;
///
/// let expanded = expand_home_dir("~/.cache/symphony/models");
/// // On Unix: "/home/username/.cache/symphony/models"
///
/// let unchanged = expand_home_dir("/absolute/path");
/// // Returns: "/absolute/path"
/// ```
pub fn expand_home_dir<P: AsRef<Path>>(path: P) -> PathBuf {
    let path_str = path.as_ref().to_string_lossy();

    if path_str.starts_with("~/") {
        if let Some(home_dir) = dirs::home_dir() {
            home_dir.join(&path_str[2..])
        } else {
            // Fallback to environment variable if dirs crate fails
            if let Ok(home) = std::env::var("HOME") {
                PathBuf::from(home).join(&path_str[2..])
            } else {
                // Last resort: return the path as-is
                path.as_ref().to_path_buf()
            }
        }
    } else {
        path.as_ref().to_path_buf()
    }
}

/// Convenience function to expand a string path and return it as a String
pub fn expand_home_dir_str<P: AsRef<Path>>(path: P) -> String {
    expand_home_dir(path).to_string_lossy().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_expand_home_dir_with_tilde() {
        let test_path = "~/.cache/symphony/models";
        let expanded = expand_home_dir(test_path);

        // Should not start with ~ anymore
        assert!(!expanded.to_string_lossy().starts_with("~"));

        // Should contain the home directory
        if let Some(home) = dirs::home_dir() {
            assert!(expanded.starts_with(&home));
            assert!(expanded.to_string_lossy().contains(".cache/symphony/models"));
        }
    }

    #[test]
    fn test_expand_home_dir_without_tilde() {
        let test_path = "/absolute/path/to/file";
        let expanded = expand_home_dir(test_path);

        // Should remain unchanged
        assert_eq!(expanded.to_string_lossy(), test_path);
    }

    #[test]
    fn test_expand_home_dir_str() {
        let test_path = "~/.bashrc";
        let expanded = expand_home_dir_str(test_path);

        // Should not start with ~ anymore
        assert!(!expanded.starts_with("~"));
    }

    #[test]
    fn test_expand_home_dir_just_tilde() {
        let test_path = "~/";
        let expanded = expand_home_dir(test_path);

        // Should be the home directory
        if let Some(home) = dirs::home_dir() {
            assert_eq!(expanded, home);
        }
    }
}


#[derive(Debug, Copy, Clone, Default, Eq, PartialEq, Hash)]
pub struct Stream {
    pub inst_id: u128,
    pub local_stream_id: u32,
}

impl Stream {
    pub fn new(inst_id: &Uuid, local_stream_id: Option<u32>) -> Self {
        Self {
            inst_id: inst_id.as_u128(),
            local_stream_id: local_stream_id.unwrap_or(0),
        }
    }
}

/// A fast, thread-safe counter.
#[derive(Debug)]
pub struct Counter {
    count: AtomicIsize,
}

impl Counter {
    /// Creates a new counter starting at the given initial value.
    pub fn new(initial: isize) -> Self {
        Self {
            count: AtomicIsize::new(initial),
        }
    }

    /// Increments the counter by 1.
    pub fn inc(&self) -> isize {
        // Using relaxed ordering because we only care about the counter's value.
        self.count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Decrements the counter by 1.
    pub fn dec(&self) -> isize {
        self.count.fetch_sub(1, Ordering::Relaxed) - 1
    }

    /// Returns the current count.
    pub fn get(&self) -> isize {
        self.count.load(Ordering::Relaxed)
    }
}

/// A very fast bounded ID pool that always returns the smallest available ID.
/// The pool is created with a maximum capacity.
#[derive(Debug)]
pub struct IdPool<T> {
    /// The next (never‑allocated) ID.
    next: T,
    /// The set of freed IDs.
    free: BTreeSet<T>,
    /// The maximum number of IDs that can be allocated.
    max_capacity: T,
}

impl<T> IdPool<T>
where
    T: PrimInt,
{
    /// Create a new ID pool with the given maximum capacity.
    pub fn new(max_capacity: T) -> Self {
        Self {
            next: T::zero(),
            free: BTreeSet::new(),
            max_capacity,
        }
    }

    pub fn set_capacity(&mut self, capacity: T) -> Result<()> {
        if capacity < self.next {
            return Err(Error::msg("Cannot set capacity lower than the next ID"));
        }
        self.max_capacity = capacity;
        Ok(())
    }

    pub fn capacity(&self) -> T {
        self.max_capacity
    }

    /// Allocate and return the smallest available ID.
    ///
    /// Returns `Some(id)` if an ID is available, or `None` if the pool is exhausted.
    pub fn acquire(&mut self) -> Result<T> {
        if let Some(&id) = self.free.iter().next() {
            // There is a freed ID available. Remove and return it.
            self.free.remove(&id);
            Ok(id)
        } else if self.next < self.max_capacity {
            // No freed IDs available; allocate a fresh one.
            let addr = self.next;
            self.next = self.next + T::one();
            Ok(addr)
        } else {
            // Pool is exhausted.
            Err(Error::msg("ID pool exhausted"))
        }
    }

    pub fn acquire_many(&mut self, count: usize) -> Result<Vec<T>> {
        // check if we have enough available ids
        if self.available() < count {
            return Err(Error::msg("ID pool exhausted"));
        }

        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            result.push(self.acquire()?);
        }
        Ok(result)
    }

    /// Release an ID back into the pool so it can be re-used.

    pub fn release(&mut self, addr: T) -> Result<()> {
        // Only allow releasing IDs that were allocated.
        if addr >= self.next {
            return Err(Error::msg("ID was never allocated"));
        }

        // Insert the id into the free set.
        self.free.insert(addr);

        if T::from(self.free.len()).unwrap() > T::from(1000).unwrap() {
            self.tail_optimization();
        }

        Ok(())
    }

    pub fn release_many(&mut self, addrs: &[T]) -> Result<()> {
        for &addr in addrs {
            if addr >= self.next {
                return Err(Error::msg("ID was never allocated"));
            }
            self.free.insert(addr);
        }
        if T::from(self.free.len()).unwrap() > T::from(1000).unwrap() {
            self.tail_optimization();
        }
        Ok(())
    }

    /// Returns the number of IDs that are available for allocation.
    ///
    /// This equals the number of IDs that have been freed plus the difference
    /// between the maximum capacity and the next never‑allocated ID.
    pub fn available(&self) -> usize {
        self.free.len() + (self.max_capacity - self.next).to_usize().unwrap()
    }

    fn tail_optimization(&mut self) {
        // Tail optimization: if the largest freed id is exactly next-1,
        // collapse the free block by decrementing `next`.
        while let Some(&last) = self.free.iter().next_back() {
            if last == self.next - T::one() {
                self.free.remove(&last);
                self.next = self.next - T::one();
            } else {
                break;
            }
        }
    }
}

/// Extract IPC socket path from endpoint URL
pub fn extract_ipc_path(endpoint: &str) -> Option<&str> {
    if endpoint.starts_with("ipc://") {
        Some(&endpoint[6..]) // Remove "ipc://" prefix
    } else {
        None
    }
}

/// Clean up IPC socket file if it exists
pub fn cleanup_ipc_socket(endpoint: &str) {
    if let Some(socket_path) = extract_ipc_path(endpoint) {
        if Path::new(socket_path).exists() {
            match std::fs::remove_file(socket_path) {
                Ok(()) => info!("Cleaned up IPC socket: {}", socket_path),
                Err(e) => warn!("Failed to remove IPC socket {}: {}", socket_path, e),
            }
        }
    }
}

/// Clean up all Symphony-related IPC sockets
pub fn cleanup_all_symphony_sockets() {
    info!("Cleaning up all Symphony IPC sockets");

    // Common patterns for Symphony IPC sockets
    let socket_patterns = [
        "/tmp/pie-ipc",
        "/tmp/pie-cli",
        "/tmp/symphony-test-client",
        "/tmp/symphony-test-cli",
    ];

    // Clean up known socket files
    for pattern in &socket_patterns {
        if Path::new(pattern).exists() {
            match std::fs::remove_file(pattern) {
                Ok(()) => info!("Cleaned up IPC socket: {}", pattern),
                Err(e) => warn!("Failed to remove IPC socket {}: {}", pattern, e),
            }
        }
    }

    // Clean up model instance sockets (symphony-model-*)
    if let Ok(entries) = std::fs::read_dir("/tmp") {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("symphony-model-") {
                    let path = entry.path();
                    match std::fs::remove_file(&path) {
                        Ok(()) => info!("Cleaned up model IPC socket: {:?}", path),
                        Err(e) => warn!("Failed to remove model IPC socket {:?}: {}", path, e),
                    }
                }
            }
        }
    }
}
