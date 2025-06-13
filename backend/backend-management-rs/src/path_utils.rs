use std::path::{Path, PathBuf};

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
