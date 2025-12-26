"""Build command implementation for the Pie CLI.

This module implements the `pie-cli build` subcommand for building
JavaScript/TypeScript inferlets into WebAssembly components.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Try to import esprima for JS parsing
try:
    import esprima
    HAS_ESPRIMA = True
except ImportError:
    HAS_ESPRIMA = False


def command_exists(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    return shutil.which(cmd) is not None


def ensure_npm_dependencies(package_dir: Path) -> None:
    """Run npm install if node_modules doesn't exist.
    
    Prompts the user for confirmation before running npm install.
    """
    node_modules = package_dir / "node_modules"
    if node_modules.exists():
        return
    
    print(f"📦 npm dependencies not found in {package_dir}")
    print("   Run 'npm install'? (Y/n) ", end="")
    sys.stdout.flush()
    
    response = input().strip().lower()
    if response in ("n", "no"):
        raise RuntimeError(
            f"npm install cancelled. Please run 'npm install' manually in {package_dir}"
        )
    
    print("   Installing...")
    
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
    # Try PIE_HOME environment variable
    if pie_home := os.environ.get("PIE_HOME"):
        path = Path(pie_home) / "inferlet-js"
        if path.exists():
            return path
    
    # Walk up from current directory
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        inferlet_js_path = parent / "inferlet-js"
        if inferlet_js_path.exists() and (inferlet_js_path / "package.json").exists():
            return inferlet_js_path
    
    raise FileNotFoundError(
        "Could not find inferlet-js library. Please set PIE_HOME environment variable."
    )


def get_wit_path() -> Path:
    """Get the path to the WIT directory."""
    # Try PIE_HOME environment variable
    if pie_home := os.environ.get("PIE_HOME"):
        path = Path(pie_home) / "inferlet" / "wit"
        if path.exists():
            return path
    
    # Walk up from current directory
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        wit_path = parent / "inferlet" / "wit"
        if wit_path.exists():
            return wit_path
    
    raise FileNotFoundError(
        "Could not find WIT directory. Please set PIE_HOME environment variable."
    )


def detect_input_type(input_path: Path) -> tuple[str, Path]:
    """Detect whether input is a single file or package directory.
    
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
    print("📦 Bundling with esbuild...")
    
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
    print("🔧 Compiling to WebAssembly component...")
    
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
        print("⚠️  Warning: The following Node.js modules were detected and will not work in WASM:")
        for warning in warnings:
            print(warning)
        print("   Consider using pure JavaScript alternatives or Pie WIT APIs instead.\n")


def validate_user_code(bundled_js: Path) -> None:
    """Validate user code for forbidden exports using AST analysis.
    
    Checks that user code doesn't export 'run' or 'main'.
    """
    if not HAS_ESPRIMA:
        # Fall back to basic string search if esprima not available
        print("⚠️ esprima not installed, skipping AST validation")
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
                    "The WIT interface is now automatically created by pie-cli build."
                )
            if name == "main":
                raise RuntimeError(
                    "User code must not export 'main' - use top-level code instead.\n\n"
                    "To fix: Move your code from inside main() to the top level."
                )


def generate_wrapper(user_bundle_path: Path, output_path: Path) -> None:
    """Generate the WIT interface wrapper."""
    user_bundle_name = user_bundle_path.name
    
    wrapper_content = f'''// Auto-generated by pie-cli build
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


def handle_build_command(
    input_path: Path,
    output: Path,
    debug: bool = False,
) -> None:
    """Handle the `pie-cli build` command.
    
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
    inferlet_js_path = get_inferlet_js_path()
    wit_path = get_wit_path()
    
    # Ensure npm dependencies
    ensure_npm_dependencies(inferlet_js_path)
    
    print("🏗️  Building JS inferlet...")
    print(f"   Input: {input_path}")
    print(f"   Output: {output}")
    
    # Detect input type
    input_type, entry_point = detect_input_type(input_path)
    print(f"   Type: {'Single file' if input_type == 'file' else 'Package'}")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        user_bundle = temp_path / "user-bundle.js"
        wrapper_js = temp_path / "wrapper.js"
        final_bundle = temp_path / "final-bundle.js"
        
        # Step 1: Bundle user code
        print("📦 Bundling user code...")
        run_esbuild_user_code(entry_point, user_bundle)
        
        # Step 2: Check for Node.js imports
        check_for_nodejs_imports(user_bundle)
        
        # Step 3: Validate user code
        print("🔍 Validating user code...")
        validate_user_code(user_bundle)
        
        # Step 4: Generate wrapper
        print("🔧 Generating WIT wrapper...")
        generate_wrapper(user_bundle, wrapper_js)
        
        # Step 5: Bundle wrapper
        print("📦 Bundling final output...")
        run_esbuild(wrapper_js, final_bundle, inferlet_js_path, debug)
        
        # Step 6: Compile to WASM
        run_componentize_js(final_bundle, output, wit_path, debug)
    
    # Success
    wasm_size = output.stat().st_size if output.exists() else 0
    print("✅ Build successful!")
    print(f"   Output: {output} ({wasm_size / 1024:.1f} KB)")
