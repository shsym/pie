"""Create command implementation for the Pie CLI.

This module implements the `pie-cli create` subcommand for creating
new JavaScript/TypeScript inferlet projects.
"""

import os
from pathlib import Path
from typing import Optional


def get_inferlet_js_path() -> Path:
    """Get the path to the inferlet-js library.
    
    Searches in order:
    1. Relative to the executable (for installed version)
    2. PIE_HOME environment variable
    3. Walk up from current directory (for development)
    
    Raises:
        FileNotFoundError: If inferlet-js cannot be found.
    """
    # Try PIE_HOME environment variable
    if pie_home := os.environ.get("PIE_HOME"):
        path = Path(pie_home) / "inferlet-js"
        if path.exists():
            return path
    
    # Walk up from current directory (development mode)
    current_dir = Path.cwd()
    for parent in [current_dir] + list(current_dir.parents):
        inferlet_js_path = parent / "inferlet-js"
        if inferlet_js_path.exists() and (inferlet_js_path / "package.json").exists():
            return inferlet_js_path
    
    raise FileNotFoundError(
        f"Could not find inferlet-js directory.\n"
        f"Searched from: {current_dir}\n"
        f"Make sure you're running from within the pie repository or that pie-cli is properly installed."
    )


def validate_project_name(name: str) -> None:
    """Validate that a project name is safe.
    
    Raises:
        ValueError: If the project name is invalid.
    """
    # Check for path traversal
    if ".." in name:
        raise ValueError(f"Project name cannot contain '..' (path traversal). Got: '{name}'")
    
    # Check for empty or whitespace-only
    if not name.strip():
        raise ValueError("Project name cannot be empty or whitespace-only")
    
    # Extract final component
    project_name = Path(name).name
    
    # Check for hidden files
    if project_name.startswith("."):
        raise ValueError(f"Project name cannot start with '.' (hidden file). Got: '{project_name}'")


def generate_index(project_dir: Path, name: str, ext: str) -> None:
    """Generate the index.ts or index.js file."""
    content = f'''// {name} - JavaScript/TypeScript Inferlet
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
send('\\n');
'''
    (project_dir / f"index.{ext}").write_text(content)


def generate_package_json(project_dir: Path, name: str, ext: str) -> None:
    """Generate the package.json file."""
    content = f'''{{
  "name": "{name}",
  "version": "0.1.0",
  "type": "module",
  "main": "index.{ext}"
}}
'''
    (project_dir / "package.json").write_text(content)


def generate_tsconfig(project_dir: Path, inferlet_path: Path) -> None:
    """Generate the tsconfig.json file."""
    # Convert path to forward slashes for JSON
    path_str = str(inferlet_path).replace("\\", "/")
    content = f'''{{
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
      "inferlet": ["{path_str}/src/index.ts"],
      "inferlet/*": ["{path_str}/src/*"]
    }}
  }},
  "include": ["*.ts", "{path_str}/src/globals.d.ts"]
}}
'''
    (project_dir / "tsconfig.json").write_text(content)


def handle_create_command(
    name: str,
    js: bool = False,
    output: Optional[Path] = None,
) -> None:
    """Handle the `pie-cli create` command.
    
    Creates a new inferlet project with:
    - index.ts or index.js
    - package.json
    - tsconfig.json (TypeScript only)
    
    Args:
        name: Name of the inferlet project.
        js: Use JavaScript instead of TypeScript.
        output: Output directory (default: current directory).
    """
    # Validate project name
    validate_project_name(name)
    
    # Determine project directory
    if output:
        project_dir = output / name
    else:
        project_dir = Path(name)
    
    # Extract just the project name for display/package.json
    project_name = Path(name).name
    
    # Check if directory already exists
    if project_dir.exists():
        raise FileExistsError(f"Directory '{project_dir}' already exists")
    
    # Create project directory
    project_dir.mkdir(parents=True)
    
    # Get inferlet-js path for tsconfig
    try:
        inferlet_js_path = get_inferlet_js_path()
        # Make relative to project directory
        project_dir_abs = project_dir.resolve()
        try:
            relative_path = inferlet_js_path.resolve().relative_to(project_dir_abs.parent)
        except ValueError:
            # Not a relative path, use absolute
            relative_path = inferlet_js_path.resolve()
    except FileNotFoundError:
        # Use a placeholder path
        relative_path = Path("../inferlet-js")
    
    # Generate files
    ext = "js" if js else "ts"
    generate_index(project_dir, project_name, ext)
    generate_package_json(project_dir, project_name, ext)
    if not js:
        generate_tsconfig(project_dir, relative_path)
    
    # Print success message
    print(f"✅ Created inferlet project: {project_name}")
    print(f"   {project_dir}/index.{ext}")
    print(f"   {project_dir}/package.json")
    if not js:
        print(f"   {project_dir}/tsconfig.json")
    print()
    print("Next steps:")
    print(f"   cd {project_dir}")
    print(f"   pie-cli build . -o {project_name}.wasm")
