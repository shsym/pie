use anyhow::{Context, Result, bail};
use clap::Args;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::fs;
use std::env;
use swc_common::{sync::Lrc, SourceMap, FileName};
use swc_ecma_parser::{lexer::Lexer, Parser, StringInput, Syntax};
use swc_ecma_ast::{Decl, ExportDecl, ModuleDecl, ModuleItem, Pat};

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
        .arg("--ignore-scripts")
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

/// Check for Node.js-specific imports that won't work in WASM.
/// Note: Some modules have Web API equivalents (url->URL, crypto->crypto.subtle,
/// stream->ReadableStream, buffer->Uint8Array) that work as globals, but
/// importing the Node.js module directly will always fail.
fn check_for_nodejs_imports(bundled_js: &Path) -> Result<()> {
    let content = fs::read_to_string(bundled_js)
        .context("Failed to read bundled JS")?;

    let nodejs_modules = ["fs", "path", "events", "os", "crypto", "child_process", "net", "http", "https", "stream", "util", "url", "buffer", "process", "domain"];
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

/// Check for forbidden patterns in user code using AST analysis
fn validate_user_code(bundled_js: &Path) -> Result<()> {
    let content = fs::read_to_string(bundled_js)
        .context("Failed to read bundled JS for validation")?;

    // Parse JavaScript into AST
    let cm: Lrc<SourceMap> = Default::default();
    let fm = cm.new_source_file(
        FileName::Custom(bundled_js.display().to_string()).into(),
        content
    );

    let lexer = Lexer::new(
        Syntax::Es(Default::default()),
        Default::default(),
        StringInput::from(&*fm),
        None,
    );

    let mut parser = Parser::new_from(lexer);

    let module = parser.parse_module().map_err(|e| {
        anyhow::anyhow!("Failed to parse JavaScript: {:?}", e)
    })?;

    // Walk AST looking for forbidden exports
    for item in &module.body {
        if let ModuleItem::ModuleDecl(decl) = item {
            check_module_decl(decl)?;
        }
    }

    Ok(())
}

/// Check a module declaration for forbidden exports
fn check_module_decl(decl: &ModuleDecl) -> Result<()> {
    match decl {
        // export const run = ..., export function run(), etc.
        ModuleDecl::ExportDecl(ExportDecl { decl, .. }) => {
            check_decl_for_forbidden_names(decl)?;
        }
        // export { run }, export { foo as run }
        ModuleDecl::ExportNamed(named) => {
            for spec in &named.specifiers {
                if let swc_ecma_ast::ExportSpecifier::Named(n) = spec {
                    let exported_name = n.exported.as_ref().unwrap_or(&n.orig);
                    if let swc_ecma_ast::ModuleExportName::Ident(ident) = exported_name {
                        check_forbidden_export_name(&ident.sym)?;
                    }
                }
            }
        }
        // export default function run() - check if named 'run' or 'main'
        ModuleDecl::ExportDefaultDecl(default_decl) => {
            if let swc_ecma_ast::DefaultDecl::Fn(fn_expr) = &default_decl.decl {
                if let Some(ident) = &fn_expr.ident {
                    // Default export with explicit name - rare but check it
                    check_forbidden_export_name(&ident.sym)?;
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Check a declaration for forbidden export names
fn check_decl_for_forbidden_names(decl: &Decl) -> Result<()> {
    match decl {
        Decl::Fn(fn_decl) => {
            check_forbidden_export_name(&fn_decl.ident.sym)?;
        }
        Decl::Var(var_decl) => {
            for decl in &var_decl.decls {
                check_pattern_for_forbidden_names(&decl.name)?;
            }
        }
        Decl::Class(class_decl) => {
            check_forbidden_export_name(&class_decl.ident.sym)?;
        }
        _ => {}
    }
    Ok(())
}

/// Recursively check a pattern for forbidden names (handles destructuring)
fn check_pattern_for_forbidden_names(pat: &Pat) -> Result<()> {
    match pat {
        // Simple identifier: export const run = ...
        Pat::Ident(ident) => {
            check_forbidden_export_name(&ident.id.sym)?;
        }
        // Array destructuring: export const [run] = arr
        Pat::Array(array_pat) => {
            for elem in &array_pat.elems {
                if let Some(elem_pat) = elem {
                    check_pattern_for_forbidden_names(elem_pat)?;
                }
            }
        }
        // Object destructuring: export const { run } = obj
        Pat::Object(object_pat) => {
            for prop in &object_pat.props {
                match prop {
                    swc_ecma_ast::ObjectPatProp::KeyValue(kv) => {
                        // Check the value pattern (e.g., { foo: run })
                        check_pattern_for_forbidden_names(&kv.value)?;
                    }
                    swc_ecma_ast::ObjectPatProp::Assign(assign) => {
                        // Check the key (e.g., { run })
                        check_forbidden_export_name(&assign.key.sym)?;
                    }
                    swc_ecma_ast::ObjectPatProp::Rest(rest) => {
                        // Check rest pattern (e.g., { ...run })
                        check_pattern_for_forbidden_names(&rest.arg)?;
                    }
                }
            }
        }
        // Rest pattern: export const [...run] = arr
        Pat::Rest(rest_pat) => {
            check_pattern_for_forbidden_names(&rest_pat.arg)?;
        }
        // Assignment pattern: export const [run = 1] = arr
        Pat::Assign(assign_pat) => {
            check_pattern_for_forbidden_names(&assign_pat.left)?;
        }
        // Invalid or expression patterns - skip
        Pat::Invalid(_) | Pat::Expr(_) => {}
    }
    Ok(())
}

/// Check if an export name is forbidden
fn check_forbidden_export_name(name: &str) -> Result<()> {
    if name == "run" {
        bail!(
            "User code must not export 'run' - it is auto-generated.\n\n\
             To fix: Remove the 'export const run = {{ ... }}' block from your code.\n\
             The WIT interface is now automatically created by pie-cli build."
        );
    }
    if name == "main" {
        bail!(
            "User code must not export 'main' - use top-level code instead.\n\n\
             To fix: Move your code from inside main() to the top level."
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn validate_code_str(code: &str) -> Result<()> {
        let mut file = NamedTempFile::new()?;
        file.write_all(code.as_bytes())?;
        validate_user_code(file.path())
    }

    #[test]
    fn test_rejects_export_run() {
        let result = validate_code_str("export const run = () => {};");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("run"));
    }

    #[test]
    fn test_allows_run_in_string() {
        // This should NOT be rejected - it's just a string
        let result = validate_code_str(r#"console.log("export const run = 1");"#);
        assert!(result.is_ok());
    }

    #[test]
    fn test_allows_run_in_comment() {
        // This should NOT be rejected - it's just a comment
        let result = validate_code_str("// export const run = 1\nconst x = 1;");
        assert!(result.is_ok());
    }

    #[test]
    fn test_rejects_unicode_escape_bypass() {
        // Unicode escape for 'e' in export (\u0065 = 'e') - should still be rejected.
        // Use escaped backslash so JS parser receives: \u0065xport const run = () => {};
        let result = validate_code_str("\\u0065xport const run = () => {};");
        assert!(result.is_err());
    }

    #[test]
    fn test_rejects_export_main() {
        let result = validate_code_str("export function main() {}");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("main"));
    }

    #[test]
    fn test_allows_main_in_object() {
        // main as object property should be allowed
        let result = validate_code_str("const obj = { main: () => {} };");
        assert!(result.is_ok());
    }

    #[test]
    fn test_allows_main_as_method() {
        // main as class method should be allowed
        let result = validate_code_str("class Foo { main() {} }");
        assert!(result.is_ok());
    }

    #[test]
    fn test_rejects_object_destructuring() {
        // Object destructuring should be caught
        let result = validate_code_str("export const { run } = obj;");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("run"));
    }

    #[test]
    fn test_rejects_array_destructuring() {
        // Array destructuring should be caught
        let result = validate_code_str("export const [run] = arr;");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("run"));
    }

    #[test]
    fn test_rejects_nested_object_destructuring() {
        // Nested object destructuring: { foo: run }
        let result = validate_code_str("export const { foo: run } = obj;");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("run"));
    }

    #[test]
    fn test_rejects_rest_destructuring() {
        // Rest pattern: { ...run }
        let result = validate_code_str("export const { ...run } = obj;");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("run"));
    }

    #[test]
    fn test_rejects_array_rest_destructuring() {
        // Array rest pattern: [...run]
        let result = validate_code_str("export const [...run] = arr;");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("run"));
    }

    #[test]
    fn test_rejects_assignment_destructuring() {
        // Assignment pattern with default: [run = 1]
        let result = validate_code_str("export const [run = 1] = arr;");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("run"));
    }

    #[test]
    fn test_allows_nested_destructuring_without_run() {
        // Nested destructuring without forbidden names should be allowed
        let result = validate_code_str("export const { foo: bar, baz } = obj;");
        assert!(result.is_ok());
    }

    #[test]
    fn test_allows_complex_destructuring() {
        // Complex destructuring without forbidden names should be allowed
        let result = validate_code_str("export const [a, { b, c: d }, ...rest] = data;");
        assert!(result.is_ok());
    }
}
