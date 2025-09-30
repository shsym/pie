use std::time::{SystemTime, UNIX_EPOCH};

/// Generate a random-looking string based on the current timestamp
pub fn generate_random_string(len: usize) -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    
    // Create a simple alphanumeric string from the timestamp
    let mut result = timestamp.to_string();
    
    // Ensure length is at least the requested length
    while result.len() < len {
        result = format!("{}{}", result, timestamp);
    }
    
    // Trim to the exact requested length
    result[..len].to_string()
}

/// Get the current UNIX timestamp in seconds
pub fn get_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}