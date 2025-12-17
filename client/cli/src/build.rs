use anyhow::{Context, Result, bail};
use clap::Args;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;
use std::env;

#[derive(Args, Debug)]
pub struct BuildArgs {
    /// Input file (.js, .ts) or directory with package.json
    #[arg(value_parser = crate::path::expand_tilde)]
    pub input: PathBuf,

    /// Output .wasm file path
    #[arg(short, long)]
    pub output: PathBuf,

    /// Enable debug build (include source maps)
    #[arg(long, default_value = "false")]
    pub debug: bool,
}

/// Check if a command is available in PATH
fn command_exists(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Get the path to the inferlet-js library bundled with pie-cli
fn get_inferlet_js_path() -> Result<PathBuf> {
    // First try relative to the executable (for installed version)
    if let Ok(exe_path) = env::current_exe() {
        let inferlet_js_path = exe_path
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("lib").join("inferlet-js"));

        if let Some(path) = inferlet_js_path {
            if path.exists() {
                return Ok(path);
            }
        }
    }

    // Try PIE_HOME environment variable
    if let Ok(pie_home) = env::var("PIE_HOME") {
        let path = PathBuf::from(pie_home).join("inferlet-js");
        if path.exists() {
            return Ok(path);
        }
    }

    // Try to find it in the development tree
    if let Ok(current_dir) = env::current_dir() {
        // Walk up the directory tree looking for inferlet-js
        let mut dir = current_dir.as_path();
        loop {
            let inferlet_js_path = dir.join("inferlet-js");
            if inferlet_js_path.exists() && inferlet_js_path.join("package.json").exists() {
                return Ok(inferlet_js_path);
            }

            match dir.parent() {
                Some(parent) => dir = parent,
                None => break,
            }
        }
    }

    bail!("Could not find inferlet-js library. Please set PIE_HOME environment variable.")
}

/// Get the path to the WIT directory
fn get_wit_path() -> Result<PathBuf> {
    // First try relative to the executable
    if let Ok(exe_path) = env::current_exe() {
        let wit_path = exe_path
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.join("lib").join("wit"));

        if let Some(path) = wit_path {
            if path.exists() {
                return Ok(path);
            }
        }
    }

    // Try PIE_HOME environment variable
    if let Ok(pie_home) = env::var("PIE_HOME") {
        let path = PathBuf::from(pie_home).join("inferlet").join("wit");
        if path.exists() {
            return Ok(path);
        }
    }

    // Try to find it in the development tree
    if let Ok(current_dir) = env::current_dir() {
        let mut dir = current_dir.as_path();
        loop {
            let wit_path = dir.join("inferlet").join("wit");
            if wit_path.exists() {
                return Ok(wit_path);
            }

            match dir.parent() {
                Some(parent) => dir = parent,
                None => break,
            }
        }
    }

    bail!("Could not find WIT directory. Please set PIE_HOME environment variable.")
}

/// Detect whether the input is a single file or a package directory
fn detect_input_type(input: &Path) -> Result<(&'static str, PathBuf)> {
    if input.is_file() {
        let ext = input.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match ext {
            "js" | "ts" => Ok(("file", input.to_path_buf())),
            _ => bail!("Unsupported file type: .{}. Expected .js or .ts", ext),
        }
    } else if input.is_dir() {
        let package_json = input.join("package.json");
        if !package_json.exists() {
            bail!("Directory '{}' does not contain package.json", input.display());
        }

        // Read package.json to find entry point
        let package_content = fs::read_to_string(&package_json)
            .context("Failed to read package.json")?;

        let package: serde_json::Value = serde_json::from_str(&package_content)
            .context("Failed to parse package.json")?;

        let main = package.get("main")
            .and_then(|v| v.as_str())
            .unwrap_or("index.js");

        let entry = input.join(main);
        if !entry.exists() {
            bail!("Entry point '{}' specified in package.json does not exist", entry.display());
        }

        Ok(("package", entry))
    } else {
        bail!("Input '{}' does not exist", input.display());
    }
}

/// Create esbuild configuration for bundling
fn run_esbuild(
    entry_point: &Path,
    output_file: &Path,
    inferlet_js_path: &Path,
    debug: bool,
) -> Result<()> {
    println!("📦 Bundling with esbuild...");

    // Build the esbuild command
    let mut cmd = Command::new("npx");
    cmd.arg("esbuild")
        .arg(entry_point)
        .arg("--bundle")
        .arg("--format=esm")
        .arg("--platform=neutral")
        .arg("--target=es2020")
        .arg("--main-fields=module,main")  // Resolve package entry points on neutral platform
        .arg(format!("--outfile={}", output_file.display()))
        .arg(format!("--alias:inferlet={}", inferlet_js_path.join("src").join("index.ts").display()));

    if debug {
        cmd.arg("--sourcemap=inline");
    } else {
        cmd.arg("--minify");
    }

    // Add external markers for WIT imports (provided by componentize-js at runtime)
    cmd.arg("--external:wasi:*");
    cmd.arg("--external:inferlet:*");

    // Use nunjucks browser build to avoid Node.js dependencies
    // Note: Must use full nunjucks.js (not slim) for runtime template compilation
    let nunjucks_browser = inferlet_js_path
        .join("node_modules")
        .join("nunjucks")
        .join("browser")
        .join("nunjucks.js");
    cmd.arg(format!("--alias:nunjucks={}", nunjucks_browser.display()));

    // Mark Node.js built-ins as external (not used in browser build)
    for builtin in ["fs", "path", "events", "os", "crypto", "child_process", "net", "http", "https", "stream", "util", "url", "buffer", "process", "domain"] {
        cmd.arg(format!("--external:{}", builtin));
    }

    let output = cmd.output().context("Failed to run esbuild")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("esbuild failed:\n{}", stderr);
    }

    Ok(())
}

/// Run componentize-js to compile bundled JS to WASM
fn run_componentize_js(
    input_js: &Path,
    output_wasm: &Path,
    wit_path: &Path,
    debug: bool,
) -> Result<()> {
    println!("🔧 Compiling to WebAssembly component...");

    let mut cmd = Command::new("npx");
    cmd.arg("@bytecodealliance/componentize-js")
        .arg(input_js)
        .arg("-o")
        .arg(output_wasm)
        .arg("--wit")
        .arg(wit_path)
        .arg("--world-name")
        .arg("exec");

    if debug {
        cmd.arg("--debug");
    }

    let output = cmd.output().context("Failed to run componentize-js")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!("componentize-js failed:\nstdout: {}\nstderr: {}", stdout, stderr);
    }

    Ok(())
}

/// Check for Node.js-specific imports that won't work in WASM
fn check_for_nodejs_imports(bundled_js: &Path) -> Result<()> {
    let content = fs::read_to_string(bundled_js)
        .context("Failed to read bundled JS")?;

    let nodejs_modules = ["fs", "path", "child_process", "net", "os", "http", "https", "crypto"];
    let mut warnings = Vec::new();

    for module in nodejs_modules {
        let patterns = [
            format!("require(\"{}\")", module),
            format!("require('{}')", module),
            format!("from \"{}\"", module),
            format!("from '{}'", module),
            format!("import \"{}\"", module),
            format!("import '{}'", module),
        ];

        for pattern in &patterns {
            if content.contains(pattern) {
                warnings.push(format!("  - '{}'", module));
                break;
            }
        }
    }

    if !warnings.is_empty() {
        println!("⚠️  Warning: The following Node.js modules were detected and will not work in WASM:");
        for warning in warnings {
            println!("{}", warning);
        }
        println!("   Consider using pure JavaScript alternatives or Pie WIT APIs instead.\n");
    }

    Ok(())
}

pub async fn handle_build_command(args: BuildArgs) -> Result<()> {
    // Check prerequisites
    if !command_exists("node") {
        bail!("Node.js is required but not found. Please install Node.js (v18+).");
    }

    if !command_exists("npx") {
        bail!("npx is required but not found. Please install Node.js (v18+).");
    }

    // Resolve paths
    let inferlet_js_path = get_inferlet_js_path()?;
    let wit_path = get_wit_path()?;

    println!("🏗️  Building JS inferlet...");
    println!("   Input: {}", args.input.display());
    println!("   Output: {}", args.output.display());

    // Detect input type and entry point
    let (input_type, entry_point) = detect_input_type(&args.input)?;
    println!("   Type: {}", if input_type == "file" { "Single file" } else { "Package" });

    // Create temp directory for intermediate files
    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;
    let bundled_js = temp_dir.path().join("bundled.js");

    // Step 1: Bundle with esbuild
    run_esbuild(&entry_point, &bundled_js, &inferlet_js_path, args.debug)?;

    // Step 2: Check for Node.js imports (warning only)
    check_for_nodejs_imports(&bundled_js)?;

    // Step 3: Compile to WASM with componentize-js
    run_componentize_js(&bundled_js, &args.output, &wit_path, args.debug)?;

    // Success!
    let wasm_size = fs::metadata(&args.output)
        .map(|m| m.len())
        .unwrap_or(0);

    println!("✅ Build successful!");
    println!("   Output: {} ({:.1} KB)", args.output.display(), wasm_size as f64 / 1024.0);

    Ok(())
}
