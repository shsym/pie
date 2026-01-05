"""Build command implementation for Bakery.

This module implements the `bakery build` subcommand for building
JavaScript/TypeScript and Rust inferlets into WebAssembly components.
"""

import json
import os
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path
from typing import Optional

from rich.panel import Panel
from .console import console
import typer

# Try to import esprima for JS parsing
try:
    import esprima
    HAS_ESPRIMA = True
except ImportError:
    HAS_ESPRIMA = False


def command_exists(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    return shutil.which(cmd) is not None


def detect_platform(input_path: Path) -> str:
    """Auto-detect project platform (rust or javascript).
    
    Args:
        input_path: Path to file or directory.
    
    Returns:
        "rust" or "javascript"
    
    Raises:
        ValueError: If platform cannot be determined.
    """
    if input_path.is_dir():
        if (input_path / "Cargo.toml").exists():
            return "rust"
        if (input_path / "package.json").exists():
            return "javascript"
        raise ValueError(
            f"Cannot detect platform for '{input_path}'. "
            "Expected Cargo.toml (Rust) or package.json (JavaScript)."
        )
    
    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext == ".rs":
            return "rust"
        if ext in (".js", ".ts"):
            return "javascript"
        raise ValueError(
            f"Unsupported file type: {ext}. Expected .rs, .js, or .ts"
        )
    
    raise ValueError(f"Input '{input_path}' does not exist")


def ensure_npm_dependencies(package_dir: Path) -> None:
    """Run npm install if node_modules doesn't exist.
    
    Prompts the user for confirmation before running npm install.
    """
    node_modules = package_dir / "node_modules"
    if node_modules.exists():
        return
    
    console.print(f"[yellow]📦 npm dependencies not found in {package_dir}[/yellow]")
    
    if not typer.confirm("   Run 'npm install'?", default=True):
        raise RuntimeError(
            f"npm install cancelled. Please run 'npm install' manually in {package_dir}"
        )
    
    with console.status("[bold green]Installing npm dependencies...[/bold green]"):
        result = subprocess.run(
            ["npm", "install", "--ignore-scripts"],
            cwd=package_dir,
            capture_output=True,
            text=True,
        )
    
    if result.returncode != 0:
        raise RuntimeError(f"npm install failed in {package_dir}:\n{result.stderr}")


def get_inferlet_js_path() -> Path:
    """Get the path to the inferlet-js library."""
    # Try PIE_SDK environment variable
    if pie_sdk := os.environ.get("PIE_SDK"):
        path = Path(pie_sdk) / "javascript"
        if path.exists():
            return path
    
    # Walk up from current directory
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        inferlet_js_path = parent / "sdk" / "javascript"
        if inferlet_js_path.exists() and (inferlet_js_path / "package.json").exists():
            return inferlet_js_path
    
    raise FileNotFoundError(
        "Could not find inferlet-js library. Please set PIE_SDK environment variable."
    )


def get_wit_path() -> Path:
    """Get the path to the WIT directory."""
    # Try PIE_SDK environment variable
    if pie_sdk := os.environ.get("PIE_SDK"):
        path = Path(pie_sdk) / "interfaces"
        if path.exists():
            return path
    
    # Walk up from current directory
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        wit_path = parent / "sdk" / "interfaces"
        if wit_path.exists():
            return wit_path
    
    raise FileNotFoundError(
        "Could not find WIT directory. Please set PIE_SDK environment variable."
    )


def detect_js_input_type(input_path: Path) -> tuple[str, Path]:
    """Detect whether JS input is a single file or package directory.
    
    Returns:
        Tuple of (type, entry_point) where type is "file" or "package".
    """
    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext in (".js", ".ts"):
            return ("file", input_path)
        raise ValueError(f"Unsupported file type: {ext}. Expected .js or .ts")
    
    if input_path.is_dir():
        package_json = input_path / "package.json"
        if not package_json.exists():
            raise ValueError(f"Directory '{input_path}' does not contain package.json")
        
        # Read package.json to find entry point
        package_data = json.loads(package_json.read_text())
        main = package_data.get("main", "index.js")
        
        entry = input_path / main
        if not entry.exists():
            raise ValueError(
                f"Entry point '{entry}' specified in package.json does not exist"
            )
        
        return ("package", entry)
    
    raise ValueError(f"Input '{input_path}' does not exist")


def run_esbuild_user_code(entry_point: Path, output_file: Path) -> None:
    """Bundle user code with esbuild (keeps inferlet imports external)."""
    cmd = [
        "npx", "esbuild",
        str(entry_point),
        "--bundle",
        "--format=esm",
        "--platform=neutral",
        "--target=es2022",
        "--main-fields=module,main",
        f"--outfile={output_file}",
        # Keep inferlet imports external
        "--external:inferlet",
        "--external:wasi:*",
        "--external:inferlet:*",
    ]
    
    # Mark Node.js built-ins as external
    nodejs_builtins = [
        "fs", "path", "events", "os", "crypto", "child_process",
        "net", "http", "https", "stream", "util", "url", "buffer",
        "process", "domain"
    ]
    for builtin in nodejs_builtins:
        cmd.append(f"--external:{builtin}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"esbuild user code bundling failed:\n{result.stderr}")


def run_esbuild(
    entry_point: Path,
    output_file: Path,
    inferlet_js_path: Path,
    debug: bool,
) -> None:
    """Bundle with esbuild, resolving inferlet imports."""
    
    # Validate inferlet alias target
    inferlet_entry = inferlet_js_path / "src" / "index.ts"
    if not inferlet_entry.is_file():
        raise FileNotFoundError(
            f"inferlet entry not found at '{inferlet_entry}', "
            "ensure inferlet-js/src/index.ts exists"
        )
    
    inferlet_entry = inferlet_entry.resolve()
    
    cmd = [
        "npx", "esbuild",
        str(entry_point),
        "--bundle",
        "--format=esm",
        "--platform=neutral",
        "--target=es2022",
        "--main-fields=module,main",
        f"--outfile={output_file}",
        f"--alias:inferlet={inferlet_entry}",
    ]
    
    if debug:
        cmd.append("--sourcemap=inline")
    else:
        cmd.append("--minify")
    
    # External WIT imports
    cmd.extend(["--external:wasi:*", "--external:inferlet:*"])
    
    # External Node.js built-ins
    nodejs_builtins = [
        "fs", "path", "events", "os", "crypto", "child_process",
        "net", "http", "https", "stream", "util", "url", "buffer",
        "process", "domain"
    ]
    for builtin in nodejs_builtins:
        cmd.append(f"--external:{builtin}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"esbuild failed:\n{result.stderr}")


def run_componentize_js(
    input_js: Path,
    output_wasm: Path,
    wit_path: Path,
    debug: bool,
) -> None:
    """Compile bundled JS to WASM component."""
    
    cmd = [
        "npx", "@bytecodealliance/componentize-js",
        str(input_js),
        "-o", str(output_wasm),
        "--wit", str(wit_path),
        "--world-name", "exec",
    ]
    
    if debug:
        cmd.append("--use-debug-build")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(
            f"componentize-js failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


def check_for_nodejs_imports(bundled_js: Path) -> None:
    """Check for Node.js-specific imports that won't work in WASM."""
    content = bundled_js.read_text()
    
    nodejs_modules = [
        "fs", "path", "events", "os", "crypto", "child_process",
        "net", "http", "https", "stream", "util", "url", "buffer",
        "process", "domain"
    ]
    
    warnings = []
    
    for module in nodejs_modules:
        patterns = [
            f'require("{module}")',
            f"require('{module}')",
            f'from "{module}"',
            f"from '{module}'",
            f'import "{module}"',
            f"import '{module}'",
        ]
        
        for pattern in patterns:
            if pattern in content:
                warnings.append(f"  - '{module}'")
                break
    
    if warnings:
        console.print("[yellow]⚠️  Warning: The following Node.js modules were detected and will not work in WASM:[/yellow]")
        for warning in warnings:
            console.print(f"[yellow]{warning}[/yellow]")
        console.print("[yellow]   Consider using pure JavaScript alternatives or Pie WIT APIs instead.\n[/yellow]")


def validate_user_code(bundled_js: Path) -> None:
    """Validate user code for forbidden exports using AST analysis.
    
    Checks that user code doesn't export 'run' or 'main'.
    """
    if not HAS_ESPRIMA:
        # Fall back to basic string search if esprima not available
        console.print("[yellow]⚠️ esprima not installed, skipping AST validation[/yellow]")
        return
    
    content = bundled_js.read_text()
    
    try:
        tree = esprima.parseModule(content)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JavaScript: {e}")
    
    forbidden_names = {"run", "main"}
    
    def check_pattern(pattern: dict) -> set[str]:
        """Recursively check a pattern for identifiers."""
        names = set()
        if pattern["type"] == "Identifier":
            names.add(pattern["name"])
        elif pattern["type"] == "ObjectPattern":
            for prop in pattern["properties"]:
                if prop["type"] == "Property":
                    names.update(check_pattern(prop["value"]))
                elif prop["type"] == "RestElement":
                    names.update(check_pattern(prop["argument"]))
        elif pattern["type"] == "ArrayPattern":
            for elem in pattern["elements"]:
                if elem:
                    names.update(check_pattern(elem))
        elif pattern["type"] == "RestElement":
            names.update(check_pattern(pattern["argument"]))
        elif pattern["type"] == "AssignmentPattern":
            names.update(check_pattern(pattern["left"]))
        return names
    
    for node in tree.body:
        exported_names = set()
        
        if node["type"] == "ExportNamedDeclaration":
            if node.get("declaration"):
                decl = node["declaration"]
                if decl["type"] == "FunctionDeclaration":
                    exported_names.add(decl["id"]["name"])
                elif decl["type"] == "VariableDeclaration":
                    for var_decl in decl["declarations"]:
                        exported_names.update(check_pattern(var_decl["id"]))
                elif decl["type"] == "ClassDeclaration":
                    exported_names.add(decl["id"]["name"])
            
            if node.get("specifiers"):
                for spec in node["specifiers"]:
                    exported_name = spec.get("exported", spec.get("local"))
                    if exported_name:
                        exported_names.add(exported_name["name"])
        
        elif node["type"] == "ExportDefaultDeclaration":
            decl = node["declaration"]
            if decl.get("id"):
                exported_names.add(decl["id"]["name"])
        
        # Check for forbidden names
        for name in exported_names:
            if name == "run":
                raise RuntimeError(
                    "User code must not export 'run' - it is auto-generated.\n\n"
                    "To fix: Remove the 'export const run = { ... }' block from your code.\n"
                    "The WIT interface is now automatically created by bakery build."
                )
            if name == "main":
                raise RuntimeError(
                    "User code must not export 'main' - use top-level code instead.\n\n"
                    "To fix: Move your code from inside main() to the top level."
                )


def generate_wrapper(user_bundle_path: Path, output_path: Path) -> None:
    """Generate the WIT interface wrapper."""
    user_bundle_name = user_bundle_path.name
    
    wrapper_content = f'''// Auto-generated by bakery build
// This wrapper provides the WIT interface for the inferlet

// WIT interface export (inferlet:core/run)
export const run = {{
  run: async () => {{
    try {{
      await import('./{user_bundle_name}');
      return {{ tag: 'ok' }};
    }} catch (e) {{
      const msg = e instanceof Error ? `${{e.message}}\\n${{e.stack}}` : String(e);
      console.log(`\\nERROR: ${{msg}}\\n`);
      return {{ tag: 'err', val: msg }};
    }}
  }},
}};
'''
    
    output_path.write_text(wrapper_content)


def handle_rust_build(input_path: Path, output: Path) -> None:
    """Build a Rust inferlet to WASM.
    
    Args:
        input_path: Path to the Rust project directory (containing Cargo.toml).
        output: Output path for the .wasm file.
    """
    # Check prerequisites
    if not command_exists("cargo"):
        raise RuntimeError(
            "cargo is required but not found. Please install Rust: https://rustup.rs"
        )
    
    # Ensure input is a directory with Cargo.toml
    if not input_path.is_dir():
        raise ValueError(f"Rust build requires a directory, got file: {input_path}")
    
    cargo_toml = input_path / "Cargo.toml"
    if not cargo_toml.exists():
        raise ValueError(f"No Cargo.toml found in {input_path}")
    
    console.print(f"[bold]🏗️  Building Rust inferlet...[/bold]")
    console.print(f"   Input: [blue]{input_path}[/blue]")
    console.print(f"   Output: [blue]{output}[/blue]")
    
    # Run cargo build
    with console.status("[bold green]Running cargo build...[/bold green]"):
        cmd = [
            "cargo", "build",
            "--target", "wasm32-wasip2",
            "--release",
        ]
        
        result = subprocess.run(
            cmd,
            cwd=input_path,
            capture_output=True,
            text=True,
        )
    
    if result.returncode != 0:
        raise RuntimeError(
            f"cargo build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    
    # Find the output wasm file
    # Parse Cargo.toml to get the package name
    cargo_data = tomllib.loads(cargo_toml.read_text())
    package_name = cargo_data.get("package", {}).get("name", input_path.name)
    # Cargo replaces hyphens with underscores in output file names
    wasm_name = package_name.replace("-", "_")
    
    wasm_path = input_path / "target" / "wasm32-wasip2" / "release" / f"{wasm_name}.wasm"
    
    if not wasm_path.exists():
        raise RuntimeError(
            f"Expected output not found at {wasm_path}\n"
            "Build may have succeeded but output location is different."
        )
    
    # Copy to output location
    shutil.copy2(wasm_path, output)
    
    # Success
    wasm_size = output.stat().st_size if output.exists() else 0
    console.print(Panel(
        f"Output: [bold]{output}[/bold] ({wasm_size / 1024:.1f} KB)",
        title="[green]✅ Build successful![/green]",
        border_style="green"
    ))


def handle_js_build(input_path: Path, output: Path, debug: bool = False) -> None:
    """Build a JavaScript/TypeScript inferlet to WASM.
    
    Build process:
    1. Check prerequisites (Node.js, npx)
    2. Find inferlet-js and WIT paths
    3. Ensure npm dependencies installed
    4. Detect input type (file or package)
    5. Bundle user code with esbuild
    6. Check for Node.js imports (warnings)
    7. Validate user code (no export run/main)
    8. Generate WIT wrapper
    9. Bundle wrapper with esbuild
    10. Compile to WASM with componentize-js
    """
    # Check prerequisites
    if not command_exists("node"):
        raise RuntimeError("Node.js is required but not found. Please install Node.js (v18+).")
    
    if not command_exists("npx"):
        raise RuntimeError("npx is required but not found. Please install Node.js (v18+).")
    
    # Resolve paths
    with console.status("[bold green]Resolving paths...[/bold green]"):
        inferlet_js_path = get_inferlet_js_path()
        wit_path = get_wit_path()
    
    # Ensure npm dependencies
    ensure_npm_dependencies(inferlet_js_path)
    
    console.print("[bold]🏗️  Building JS inferlet...[/bold]")
    console.print(f"   Input: [blue]{input_path}[/blue]")
    console.print(f"   Output: [blue]{output}[/blue]")
    
    # Detect input type
    input_type, entry_point = detect_js_input_type(input_path)
    console.print(f"   Type: [dim]{'Single file' if input_type == 'file' else 'Package'}[/dim]")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        user_bundle = temp_path / "user-bundle.js"
        wrapper_js = temp_path / "wrapper.js"
        final_bundle = temp_path / "final-bundle.js"
        
        with console.status("[bold green]Building split stages...[/bold green]") as status:
            # Step 1: Bundle user code
            status.update("[bold green]📦 Bundling user code...[/bold green]")
            run_esbuild_user_code(entry_point, user_bundle)
            
            # Step 2: Check for Node.js imports
            check_for_nodejs_imports(user_bundle)
            
            # Step 3: Validate user code
            status.update("[bold green]🔍 Validating user code...[/bold green]")
            validate_user_code(user_bundle)
            
            # Step 4: Generate wrapper
            status.update("[bold green]🔧 Generating WIT wrapper...[/bold green]")
            generate_wrapper(user_bundle, wrapper_js)
            
            # Step 5: Bundle wrapper
            status.update("[bold green]📦 Bundling final output...[/bold green]")
            run_esbuild(wrapper_js, final_bundle, inferlet_js_path, debug)
            
            # Step 6: Compile to WASM
            status.update("[bold green]🔧 Compiling to WebAssembly component...[/bold green]")
            run_componentize_js(final_bundle, output, wit_path, debug)
    
    # Success
    wasm_size = output.stat().st_size if output.exists() else 0
    console.print(Panel(
        f"Output: [bold]{output}[/bold] ({wasm_size / 1024:.1f} KB)",
        title="[green]✅ Build successful![/green]",
        border_style="green"
    ))


def handle_build_command(
    input_path: Path,
    output: Path,
    debug: bool = False,
) -> None:
    """Handle the `bakery build` command.
    
    Auto-detects project platform (Rust or JavaScript) and dispatches
    to the appropriate build handler.
    
    Args:
        input_path: Path to the project directory or source file.
        output: Output path for the .wasm file.
        debug: Enable debug mode (JS only: inline source maps).
    """
    # Auto-detect platform
    platform = detect_platform(input_path)
    
    if platform == "rust":
        handle_rust_build(input_path, output)
    else:
        handle_js_build(input_path, output, debug)
