use anyhow::{Context, Result, bail};
use clap::Args;
use regex::Regex;
use std::io::{self, Write};
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

/// Run npm install in a directory if node_modules doesn't exist
fn ensure_npm_dependencies(package_dir: &Path) -> Result<()> {
    let node_modules = package_dir.join("node_modules");
    if node_modules.exists() {
        return Ok(());
    }

    println!("📦 npm dependencies not found in {}", package_dir.display());
    print!("   Run 'npm install'? (Y/n) ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim().to_lowercase();

    if input == "n" || input == "no" {
        bail!("npm install cancelled. Please run 'npm install' manually in {}", package_dir.display());
    }

    println!("   Installing...");

    let output = Command::new("npm")
        .arg("install")
        .current_dir(package_dir)
        .output()
        .context("Failed to run npm install")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("npm install failed in {}:\n{}", package_dir.display(), stderr);
    }

    Ok(())
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

    // Validate the inferlet alias target exists
    let inferlet_entry = inferlet_js_path.join("src").join("index.ts");
    if !inferlet_entry.is_file() {
        bail!(
            "inferlet entry not found at '{}', ensure inferlet-js/src/index.ts exists",
            inferlet_entry.display()
        );
    }
    let inferlet_entry = inferlet_entry
        .canonicalize()
        .with_context(|| format!("Failed to canonicalize inferlet entry path: {}", inferlet_entry.display()))?;

    // Build the esbuild command
    let mut cmd = Command::new("npx");
    cmd.arg("esbuild")
        .arg(entry_point)
        .arg("--bundle")
        .arg("--format=esm")
        .arg("--platform=neutral")
        .arg("--target=es2022")
        .arg("--main-fields=module,main")  // Resolve package entry points on neutral platform
        .arg(format!("--outfile={}", output_file.display()))
        .arg(format!("--alias:inferlet={}", inferlet_entry.display()));

    if debug {
        cmd.arg("--sourcemap=inline");
    } else {
        cmd.arg("--minify");
    }

    // Add external markers for WIT imports (provided by componentize-js at runtime)
    cmd.arg("--external:wasi:*");
    cmd.arg("--external:inferlet:*");

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
        cmd.arg("--use-debug-build");
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

/// Check for forbidden patterns in user code that conflict with auto-wrapping
fn validate_user_code(bundled_js: &Path) -> Result<()> {
    let content = fs::read_to_string(bundled_js)
        .context("Failed to read bundled JS for validation")?;

    // Check for export const run (various forms)
    // Patterns to catch: export const run, export { run }, export { x as run }
    let export_run_patterns = [
        r"export\s+const\s+run\s*=",
        r"export\s+let\s+run\s*=",
        r"export\s+var\s+run\s*=",
        r"export\s+function\s+run\s*\(",
        r"export\s*\{\s*run\s*\}",
        r"export\s*\{\s*\w+\s+as\s+run\s*\}",
    ];

    for pattern in &export_run_patterns {
        let re = Regex::new(pattern).unwrap();
        if re.is_match(&content) {
            bail!(
                "User code must not export 'run' - it is auto-generated.\n\n\
                 To fix: Remove the 'export const run = {{ ... }}' block from your code.\n\
                 The WIT interface is now automatically created by pie-cli build."
            );
        }
    }

    // Check for function main() at module level
    // This is tricky - we want top-level main, not class methods or nested functions
    // Simple heuristic: look for "function main(" or "async function main("
    // at the start of a line or after export
    let main_patterns = [
        r"(?m)^function\s+main\s*\(",
        r"(?m)^async\s+function\s+main\s*\(",
        r"(?m)^export\s+function\s+main\s*\(",
        r"(?m)^export\s+async\s+function\s+main\s*\(",
        r"(?m)^const\s+main\s*=\s*(async\s*)?\(",
        r"(?m)^let\s+main\s*=\s*(async\s*)?\(",
    ];

    for pattern in &main_patterns {
        let re = Regex::new(pattern).unwrap();
        if re.is_match(&content) {
            bail!(
                "User code must not define a 'main()' function - use top-level code instead.\n\n\
                 To fix: Move your code from inside main() to the top level:\n\n\
                 Before:\n\
                   async function main() {{\n\
                     const model = getAutoModel();\n\
                     // ...\n\
                   }}\n\n\
                 After:\n\
                   const model = getAutoModel();\n\
                   // ..."
            );
        }
    }

    Ok(())
}

/// Generate the WIT interface wrapper that dynamically imports user code
fn generate_wrapper(user_bundle_path: &Path, output_path: &Path) -> Result<()> {
    // Get just the filename for the dynamic import
    let user_bundle_name = user_bundle_path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("user-bundle.js");

    // Generate wrapper that uses dynamic import - no need to extract imports!
    // The user code (with all its imports) runs when dynamically imported.
    let wrapper_content = format!(r#"// Auto-generated by pie-cli build
// This wrapper provides the WIT interface for the inferlet

// WIT interface export (inferlet:core/run)
export const run = {{
  run: async () => {{
    try {{
      await import('./{user_bundle_name}');
      return {{ tag: 'ok' }};
    }} catch (e) {{
      const msg = e instanceof Error ? `${{e.message}}\n${{e.stack}}` : String(e);
      console.log(`\nERROR: ${{msg}}\n`);
      return {{ tag: 'err', val: msg }};
    }}
  }},
}};
"#);

    fs::write(output_path, wrapper_content)
        .with_context(|| format!("Failed to write wrapper to {}", output_path.display()))?;

    Ok(())
}

/// Run esbuild to bundle user code only (keeps inferlet imports as-is)
/// NOTE: We intentionally do NOT minify here so validation regex patterns work.
/// Minification happens in the second pass on the final wrapper.
fn run_esbuild_user_code(
    entry_point: &Path,
    output_file: &Path,
) -> Result<()> {
    let mut cmd = Command::new("npx");
    cmd.arg("esbuild")
        .arg(entry_point)
        .arg("--bundle")
        .arg("--format=esm")
        .arg("--platform=neutral")
        .arg("--target=es2022")
        .arg("--main-fields=module,main")
        .arg(format!("--outfile={}", output_file.display()));

    // Never minify user code - we need readable output for validation
    // Minification happens in the second pass on the final wrapper

    // Keep inferlet imports external - they'll be resolved in the wrapper pass
    cmd.arg("--external:inferlet");
    // Also keep WIT imports external
    cmd.arg("--external:wasi:*");
    cmd.arg("--external:inferlet:*");

    // Mark Node.js built-ins as external
    for builtin in ["fs", "path", "events", "os", "crypto", "child_process", "net", "http", "https", "stream", "util", "url", "buffer", "process", "domain"] {
        cmd.arg(format!("--external:{}", builtin));
    }

    let output = cmd.output().context("Failed to run esbuild for user code")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("esbuild user code bundling failed:\n{}", stderr);
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

    // Ensure npm dependencies are installed for inferlet-js
    ensure_npm_dependencies(&inferlet_js_path)?;

    println!("🏗️  Building JS inferlet...");
    println!("   Input: {}", args.input.display());
    println!("   Output: {}", args.output.display());

    // Detect input type and entry point
    let (input_type, entry_point) = detect_input_type(&args.input)?;
    println!("   Type: {}", if input_type == "file" { "Single file" } else { "Package" });

    // Create temp directory for intermediate files
    let temp_dir = tempfile::tempdir().context("Failed to create temp directory")?;
    let user_bundle = temp_dir.path().join("user-bundle.js");
    let wrapper_js = temp_dir.path().join("wrapper.js");
    let final_bundle = temp_dir.path().join("final-bundle.js");

    // Step 1: Bundle user code with esbuild (resolves user imports, NOT inferlet)
    println!("📦 Bundling user code...");
    run_esbuild_user_code(&entry_point, &user_bundle)?;

    // Step 2: Check for Node.js imports (warning only)
    check_for_nodejs_imports(&user_bundle)?;

    // Step 3: Validate user code (no export run, no main function)
    println!("🔍 Validating user code...");
    validate_user_code(&user_bundle)?;

    // Step 4: Generate wrapper with user code embedded + WIT interface
    println!("🔧 Generating WIT wrapper...");
    generate_wrapper(&user_bundle, &wrapper_js)?;

    // Step 5: Bundle wrapper (resolves inferlet import)
    println!("📦 Bundling final output...");
    run_esbuild(&wrapper_js, &final_bundle, &inferlet_js_path, args.debug)?;

    // Step 6: Compile to WASM with componentize-js
    run_componentize_js(&final_bundle, &args.output, &wit_path, args.debug)?;

    // Success!
    let wasm_size = fs::metadata(&args.output)
        .map(|m| m.len())
        .unwrap_or(0);

    println!("✅ Build successful!");
    println!("   Output: {} ({:.1} KB)", args.output.display(), wasm_size as f64 / 1024.0);

    Ok(())
}
