#!/usr/bin/env python3
"""
Debug Framework CLI Setup Script

Sets up CLI tools for production deployment and creates
executable wrappers for easy command-line access.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional


class CLISetup:
    """Setup manager for debug framework CLI tools."""

    def __init__(self):
        self.framework_root = Path(__file__).parent.parent
        self.cli_dir = self.framework_root / "cli"
        self.deployment_dir = Path(__file__).parent

    def detect_install_location(self) -> Optional[Path]:
        """Detect appropriate installation location for CLI tools."""

        # Try common user install locations
        potential_locations = [
            Path.home() / ".local" / "bin",
            Path.home() / "bin",
            Path("/usr/local/bin") if os.access("/usr/local/bin", os.W_OK) else None,
            Path("/opt/homebrew/bin") if Path("/opt/homebrew/bin").exists() and os.access("/opt/homebrew/bin", os.W_OK) else None
        ]

        for location in potential_locations:
            if location and location.exists() and os.access(location, os.W_OK):
                return location

        # Create user local bin if it doesn't exist
        local_bin = Path.home() / ".local" / "bin"
        local_bin.mkdir(parents=True, exist_ok=True)
        return local_bin

    def create_executable_wrapper(self, command_name: str, script_name: str, install_dir: Path) -> bool:
        """Create executable wrapper for a CLI command."""

        try:
            wrapper_path = install_dir / command_name
            cli_script_path = self.cli_dir / script_name

            # Create wrapper script
            wrapper_content = f'''#!/usr/bin/env python3
"""
{command_name} - Debug Framework CLI Tool
Executable wrapper for production deployment.
"""

import sys
import os
from pathlib import Path

# Add debug framework to Python path
framework_root = Path(__file__).resolve().parent.parent / "pie" / "backend" / "backend-python"
if framework_root.exists():
    sys.path.insert(0, str(framework_root))

# Alternative: Use current working directory approach
if not framework_root.exists():
    # Assume we're running from pie project root
    current_dir = Path.cwd()
    framework_paths = [
        current_dir / "backend" / "backend-python",
        current_dir.parent / "backend" / "backend-python",
    ]

    for path in framework_paths:
        if path.exists():
            sys.path.insert(0, str(path))
            break

try:
    if __name__ == "__main__":
        from debug_framework.cli.{script_name.replace('.py', '')} import {command_name.replace('-', '_').title()}CLI
        cli = {command_name.replace('-', '_').title()}CLI()
        cli.main()

except ImportError as e:
    print(f"Error: Could not import debug framework: {{e}}", file=sys.stderr)
    print(f"Please ensure you're running from the pie project directory or set PYTHONPATH correctly.", file=sys.stderr)
    print(f"Current working directory: {{Path.cwd()}}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    sys.exit(1)
'''

            # Write wrapper script
            with open(wrapper_path, 'w') as f:
                f.write(wrapper_content)

            # Make executable
            wrapper_path.chmod(0o755)

            print(f"✅ Created {command_name} executable at {wrapper_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to create {command_name} executable: {e}", file=sys.stderr)
            return False

    def create_python_module_access(self) -> None:
        """Create instructions for Python module access."""

        instructions = f"""
# Debug Framework CLI Access

## Option 1: Python Module (Recommended)
From the pie project root directory:

```bash
# Navigate to backend-python directory
cd backend/backend-python

# Run CLI commands via Python module
python -m debug_framework.cli debug-validate --help
python -m debug_framework.cli plugin-compile --help
python -m debug_framework.cli session-report --help
```

## Option 2: Direct Script Execution
```bash
# From pie/backend/backend-python directory
python debug_framework/cli/debug_validate.py --help
python debug_framework/cli/plugin_compile.py --help
python debug_framework/cli/session_report.py --help
```

## Environment Variables
```bash
export PIE_DEBUG_ENABLED=true          # Enable debug output
export PIE_DEBUG_LEVEL=INFO           # Set debug level
export PIE_DEBUG_DATABASE=/path/to/db  # Custom database location
export PIE_METAL_PATH=/path/to/metal   # Custom Metal backend path
```

## Examples
```bash
# Validate Metal kernels
python -m debug_framework.cli debug-validate --backend metal

# Compile all backend projects
python -m debug_framework.cli plugin-compile --search . --output-dir ./build

# Generate session report
python -m debug_framework.cli session-report --list --days 7
```
"""

        instructions_path = self.deployment_dir / "CLI_USAGE.md"
        with open(instructions_path, 'w') as f:
            f.write(instructions.strip())

        print(f"📄 Created usage instructions at {instructions_path}")

    def setup_development_environment(self) -> bool:
        """Setup development environment for CLI testing."""

        try:
            # Create .env file for development
            env_file = self.framework_root.parent / ".env"

            env_content = """# Debug Framework Environment Variables
PIE_DEBUG_ENABLED=true
PIE_DEBUG_LEVEL=INFO
PIE_DEBUG_DATABASE=~/.pie/debug/debug_framework.db
# PIE_METAL_PATH=/path/to/metal/backend  # Uncomment and set if needed
"""

            if not env_file.exists():
                with open(env_file, 'w') as f:
                    f.write(env_content)
                print(f"✅ Created development .env file at {env_file}")
            else:
                print(f"ℹ️ Development .env file already exists at {env_file}")

            return True

        except Exception as e:
            print(f"❌ Failed to setup development environment: {e}", file=sys.stderr)
            return False

    def validate_installation(self) -> bool:
        """Validate that the CLI tools can be imported and work correctly."""

        print("🔍 Validating CLI tool installation...")

        try:
            # Test imports
            sys.path.insert(0, str(self.framework_root.parent))

            from debug_framework.cli.debug_validate import DebugValidateCLI
            from debug_framework.cli.plugin_compile import PluginCompileCLI
            from debug_framework.cli.session_report import SessionReportCLI

            print("✅ All CLI tools can be imported successfully")

            # Test basic functionality
            debug_cli = DebugValidateCLI()
            if hasattr(debug_cli, 'initialize_validation_engine'):
                print("✅ debug-validate CLI initialized successfully")

            compile_cli = PluginCompileCLI()
            if hasattr(compile_cli, 'initialize_services'):
                print("✅ plugin-compile CLI initialized successfully")

            report_cli = SessionReportCLI()
            if hasattr(report_cli, 'initialize_services'):
                print("✅ session-report CLI initialized successfully")

            return True

        except ImportError as e:
            print(f"❌ Import validation failed: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"❌ CLI validation failed: {e}", file=sys.stderr)
            return False

    def main(self):
        """Main setup process."""
        print("🚀 Setting up Debug Framework CLI Tools")
        print()

        # Validate current installation
        if not self.validate_installation():
            print("❌ Installation validation failed. Please check your Python environment.")
            sys.exit(1)

        print()
        print("📋 Setup Options:")
        print("1. Create Python module access instructions (Recommended)")
        print("2. Create executable wrappers (Advanced)")
        print("3. Setup development environment")
        print("4. All of the above")

        try:
            choice = input("\nSelect option (1-4): ").strip()
        except KeyboardInterrupt:
            print("\n❌ Setup cancelled by user")
            sys.exit(1)

        print()

        if choice in ["1", "4"]:
            # Create Python module access instructions
            self.create_python_module_access()
            print()

        if choice in ["2", "4"]:
            # Create executable wrappers
            install_dir = self.detect_install_location()
            if install_dir:
                print(f"📁 Installing CLI tools to {install_dir}")

                commands = [
                    ("debug-validate", "debug_validate.py"),
                    ("plugin-compile", "plugin_compile.py"),
                    ("session-report", "session_report.py")
                ]

                success_count = 0
                for cmd_name, script_name in commands:
                    if self.create_executable_wrapper(cmd_name, script_name, install_dir):
                        success_count += 1

                if success_count == len(commands):
                    print(f"✅ All CLI tools installed successfully to {install_dir}")
                    print(f"🔧 Make sure {install_dir} is in your PATH environment variable")
                else:
                    print(f"⚠️ {success_count}/{len(commands)} CLI tools installed successfully")
            else:
                print("❌ Could not determine installation directory")
            print()

        if choice in ["3", "4"]:
            # Setup development environment
            self.setup_development_environment()
            print()

        print("🎉 Debug Framework CLI setup completed!")
        print()
        print("Next steps:")
        print("1. Navigate to your pie project root directory")
        print("2. Try running: cd backend/backend-python")
        print("3. Test CLI: python -m debug_framework.cli debug-validate --help")
        print("4. Enable debug mode: export PIE_DEBUG_ENABLED=true")


if __name__ == "__main__":
    setup = CLISetup()
    setup.main()