#!/usr/bin/env python3
"""
Debug Framework CLI Main Entry Point

Provides a unified entry point for all debug framework CLI tools.
"""

import sys
import subprocess
from pathlib import Path


def main():
    """Main CLI dispatcher."""
    if len(sys.argv) < 2:
        print("Debug Framework CLI Tools", file=sys.stderr)
        print("", file=sys.stderr)
        print("Available commands:", file=sys.stderr)
        print("  debug-validate      - Validate kernel implementations", file=sys.stderr)
        print("  plugin-compile      - Compile backend plugins", file=sys.stderr)
        print("  session-report      - Generate session reports", file=sys.stderr)
        print("  generate-references - Generate tensor reference files", file=sys.stderr)
        print("  verify-references   - Verify tensor reference files against fresh inference", file=sys.stderr)
        print("", file=sys.stderr)
        print("Usage:", file=sys.stderr)
        print("  python -m debug_framework.cli <command> [args...]", file=sys.stderr)
        print("", file=sys.stderr)
        print("Examples:", file=sys.stderr)
        print("  python -m debug_framework.cli debug-validate --backend metal", file=sys.stderr)
        print("  python -m debug_framework.cli plugin-compile --search .", file=sys.stderr)
        print("  python -m debug_framework.cli session-report --list", file=sys.stderr)
        print("  python -m debug_framework.cli generate-references --output-dir refs", file=sys.stderr)
        print("  python -m debug_framework.cli verify-references --reference-dir refs/session_123", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    # Get the directory containing the CLI tools
    cli_dir = Path(__file__).parent

    # Map commands to their script files
    command_map = {
        "debug-validate": cli_dir / "debug_validate.py",
        "plugin-compile": cli_dir / "plugin_compile.py",
        "session-report": cli_dir / "session_report.py",
        "generate-references": cli_dir / "generate_references.py",
        "verify-references": cli_dir / "verify_references.py"
    }

    if command not in command_map:
        print(f"Unknown command: {command}", file=sys.stderr)
        print(f"Available commands: {', '.join(command_map.keys())}", file=sys.stderr)
        sys.exit(1)

    script_path = command_map[command]

    if not script_path.exists():
        print(f"Command script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    # Execute the command script
    try:
        result = subprocess.run([sys.executable, str(script_path)] + args)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Failed to execute command: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()