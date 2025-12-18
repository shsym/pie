use anyhow::{Context, Result, bail};
use clap::Args;
use std::fs;
use std::path::{Path, PathBuf};
use std::env;

#[derive(Args, Debug)]
pub struct CreateArgs {
    /// Name of the inferlet project
    pub name: String,

    /// Use JavaScript instead of TypeScript
    #[arg(long)]
    pub js: bool,

    /// Output directory (default: current directory)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Get the path to the inferlet-js library
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

    // For development: walk up from current directory
    let current_dir = env::current_dir().context("Failed to get current directory")?;
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

    bail!(
        "Could not find inferlet-js directory.\n\
         Searched from: {}\n\
         Make sure you're running from within the pie repository or that pie-cli is properly installed.",
        current_dir.display()
    )
}

/// Validate that a project path is safe (no path traversal)
fn validate_project_name(name: &str) -> Result<()> {
    // Check for path traversal sequences
    if name.contains("..") {
        bail!("Project name cannot contain '..' (path traversal). Got: '{}'", name);
    }

    // Check for empty or whitespace-only name
    if name.trim().is_empty() {
        bail!("Project name cannot be empty or whitespace-only");
    }

    // Extract the final component (actual project name) for additional checks
    let project_name = Path::new(name)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(name);

    // Check that the final component doesn't start with a dot (hidden files)
    if project_name.starts_with('.') {
        bail!("Project name cannot start with '.' (hidden file). Got: '{}'", project_name);
    }

    Ok(())
}

fn generate_index(dir: &Path, name: &str, ext: &str) -> Result<()> {
    let content = format!(
        r#"// {} - JavaScript/TypeScript Inferlet
// Write your inferlet logic here using top-level code!

const model = getAutoModel();
const ctx = new Context(model);

// Example: Simple text generation
ctx.fillSystem('You are a helpful assistant.');
ctx.fillUser('Hello!');

const sampler = Sampler.topP(0.6, 0.95);
const eosTokens = model.eosTokens().map((arr) => [...arr]);
const stopCond = maxLen(256).or(endsWithAny(eosTokens));

const result = await ctx.generate(sampler, stopCond);
send(result);
send('\n');
"#,
        name
    );

    fs::write(dir.join(format!("index.{}", ext)), content)?;
    Ok(())
}

fn generate_package_json(dir: &Path, name: &str, ext: &str) -> Result<()> {
    let content = format!(
        r#"{{
  "name": "{}",
  "version": "0.1.0",
  "type": "module",
  "main": "index.{}"
}}
"#,
        name, ext
    );

    fs::write(dir.join("package.json"), content)?;
    Ok(())
}

fn generate_tsconfig(dir: &Path, inferlet_path: &Path) -> Result<()> {
    let path_str = inferlet_path.display().to_string().replace('\\', "/");
    let content = format!(
        r#"{{
  "compilerOptions": {{
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "moduleDetection": "force",
    "strict": true,
    "skipLibCheck": true,
    "noEmit": true,
    "baseUrl": ".",
    "paths": {{
      "inferlet": ["{}/src/index.ts"],
      "inferlet/*": ["{}/src/*"]
    }}
  }},
  "include": ["*.ts", "{}/src/globals.d.ts"]
}}
"#,
        path_str, path_str, path_str
    );

    fs::write(dir.join("tsconfig.json"), content)?;
    Ok(())
}

pub async fn handle_create_command(args: CreateArgs) -> Result<()> {
    // Validate project name first
    validate_project_name(&args.name)?;

    // If -o is provided, join it with the name; otherwise use name directly as path
    let project_dir = match args.output {
        Some(output) => output.join(&args.name),
        None => PathBuf::from(&args.name),
    };

    // Extract just the project name (last component) for display/package.json
    let project_name = Path::new(&args.name)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(&args.name);

    // Check if directory already exists
    if project_dir.exists() {
        bail!("Directory '{}' already exists", project_dir.display());
    }

    // Create project directory
    fs::create_dir_all(&project_dir)
        .with_context(|| format!("Failed to create directory '{}'", project_dir.display()))?;

    // Canonicalize project_dir to get absolute path for relative path calculation
    let project_dir_abs = project_dir.canonicalize()
        .with_context(|| format!("Failed to canonicalize path '{}'", project_dir.display()))?;

    // Determine inferlet-js path relative to project
    let inferlet_js_path = get_inferlet_js_path()?;
    let relative_path = pathdiff::diff_paths(&inferlet_js_path, &project_dir_abs)
        .unwrap_or_else(|| inferlet_js_path.clone());

    // Generate files
    let ext = if args.js { "js" } else { "ts" };
    generate_index(&project_dir, project_name, ext)?;
    generate_package_json(&project_dir, project_name, ext)?;
    if !args.js {
        generate_tsconfig(&project_dir, &relative_path)?;
    }

    println!("✅ Created inferlet project: {}", project_name);
    println!("   {}/index.{}", project_dir.display(), ext);
    println!("   {}/package.json", project_dir.display());
    if !args.js {
        println!("   {}/tsconfig.json", project_dir.display());
    }
    println!();
    println!("Next steps:");
    println!("   cd {}", project_dir.display());
    println!("   pie-cli build . -o {}.wasm", project_name);

    Ok(())
}
