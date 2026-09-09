use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};

const RUNTIME_URL: &str =
    "https://registry.pie-project.org/api/v1/runtimes/python3.14/0.4.0/download";

pub fn runtime_dir() -> PathBuf {
    bootstrap::paths::pie_home().join("py-runtime")
}

fn sentinel() -> PathBuf {
    runtime_dir()
        .join("shared")
        .join("componentize-py-runtime.wasm")
}

pub fn is_installed() -> bool {
    sentinel().is_file()
}

pub fn ensure_installed(quiet: bool) -> Result<PathBuf> {
    let dir = runtime_dir();
    if is_installed() {
        return Ok(dir);
    }

    let pie_home = bootstrap::paths::pie_home();
    std::fs::create_dir_all(&pie_home).map_err(|e| anyhow!("create {pie_home:?}: {e}"))?;

    if !quiet {
        eprintln!("Downloading Python WASM runtime from {RUNTIME_URL}…");
    }
    let blob = fetch()?;

    if !quiet {
        eprintln!("Extracting runtime to {}…", pie_home.display());
    }
    extract(&blob, &pie_home)?;

    if !is_installed() {
        bail!(
            "Python runtime download completed but {dir:?} is incomplete \
             (missing {})",
            sentinel().display()
        );
    }
    Ok(dir)
}

pub fn ensure_installed_best_effort(enabled: bool) {
    let dir = runtime_dir();
    if !enabled {
        tracing::debug!(
            "python runtime download skipped: `python_runtime = false` under [sandbox]"
        );
        return;
    }
    if is_installed() {
        tracing::debug!("python runtime already installed at {}", dir.display());
        return;
    }
    tracing::info!(
        "fetching the Python WASM runtime from {RUNTIME_URL} into {}",
        dir.display()
    );
    if let Err(e) = ensure_installed(/*quiet=*/ true) {
        tracing::warn!(
            "could not fetch the Python WASM runtime from {RUNTIME_URL}: {e}. \
             Python inferlets will not run until this download succeeds; set \
             `python_runtime = false` under [sandbox] in the config to skip it."
        );
    }
}

fn fetch() -> Result<Vec<u8>> {
    let resp = reqwest::blocking::Client::new()
        .get(RUNTIME_URL)
        .send()
        .map_err(|e| anyhow!("GET {RUNTIME_URL}: {e}"))?
        .error_for_status()
        .map_err(|e| anyhow!("GET {RUNTIME_URL}: {e}"))?;
    resp.bytes()
        .map(|b| b.to_vec())
        .map_err(|e| anyhow!("read response body: {e}"))
}

fn extract(blob: &[u8], dest: &Path) -> Result<()> {
    let mut decoder = xz2::read::XzDecoder::new(blob);
    let mut tar = tar::Archive::new(&mut decoder);
    tar.unpack(dest)
        .map_err(|e| anyhow!("extract tarball into {dest:?}: {e}"))?;
    Ok(())
}
