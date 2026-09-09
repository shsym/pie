use std::path::PathBuf;

pub fn pie_home() -> PathBuf {
    if let Ok(dir) = std::env::var("PIE_HOME")
        && !dir.trim().is_empty()
    {
        return PathBuf::from(dir);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".pie")
}

pub fn pie_home_file(name: &str) -> PathBuf {
    pie_home().join(name)
}
