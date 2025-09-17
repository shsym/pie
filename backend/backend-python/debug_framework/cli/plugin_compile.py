#!/usr/bin/env python3
"""
Plugin Compile CLI Tool

Manages compilation workflows for debug framework plugins
across different backend types (Metal, CUDA, C++).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from debug_framework.services.compilation_engine import CompilationEngine
from debug_framework.services.plugin_registry import PluginRegistry
from debug_framework.services.database_manager import DatabaseManager


class PluginCompileCLI:
    """CLI tool for compiling debug framework plugins."""

    def __init__(self):
        self.compilation_engine: Optional[CompilationEngine] = None
        self.plugin_registry: Optional[PluginRegistry] = None
        self.compilation_results: List[Dict[str, Any]] = []

    def initialize_services(self, output_directory: str, database_path: Optional[str] = None) -> bool:
        """Initialize compilation engine and plugin registry."""
        try:
            self.compilation_engine = CompilationEngine(output_directory=output_directory)

            if database_path:
                db_manager = DatabaseManager(database_path)
                self.plugin_registry = PluginRegistry(database_manager=db_manager)
            else:
                self.plugin_registry = PluginRegistry()

            return True
        except Exception as e:
            print(f"❌ Failed to initialize services: {e}", file=sys.stderr)
            return False

    def detect_backend_projects(self, search_path: str) -> List[Dict[str, Any]]:
        """Detect backend projects with CMakeLists.txt."""
        print(f"🔍 Scanning for backend projects in {search_path}...")

        projects = []
        search_root = Path(search_path)

        # Look for CMakeLists.txt files
        for cmake_file in search_root.rglob("CMakeLists.txt"):
            project_dir = cmake_file.parent

            # Determine backend type based on directory structure and contents
            backend_type = self._detect_backend_type(project_dir)
            if not backend_type:
                continue

            project_name = project_dir.name

            project_info = {
                "name": project_name,
                "path": str(project_dir),
                "backend_type": backend_type,
                "cmake_file": str(cmake_file),
                "relative_path": str(project_dir.relative_to(search_root))
            }

            projects.append(project_info)
            print(f"  Found {backend_type} project: {project_name} ({project_info['relative_path']})")

        return projects

    def _detect_backend_type(self, project_dir: Path) -> Optional[str]:
        """Detect backend type based on project structure and files."""
        project_files = [f.name.lower() for f in project_dir.rglob("*") if f.is_file()]

        # Metal backend indicators
        if any(f.endswith('.metal') for f in project_files) or 'metal' in project_dir.name.lower():
            return "metal"

        # CUDA backend indicators
        if any(f.endswith('.cu') or f.endswith('.cuh') for f in project_files) or 'cuda' in project_dir.name.lower():
            return "cuda"

        # C++ backend (generic)
        if any(f.endswith('.cpp') or f.endswith('.hpp') for f in project_files):
            return "cpp"

        return None

    def validate_toolchain(self, backend_type: str) -> Dict[str, Any]:
        """Validate toolchain availability for backend type."""
        if not self.compilation_engine:
            return {"status": "error", "message": "Compilation engine not initialized"}

        is_valid = self.compilation_engine.validate_toolchain(backend_type)
        supported_platforms = self.compilation_engine.get_supported_platforms()

        return {
            "backend_type": backend_type,
            "is_valid": is_valid,
            "supported_platforms": supported_platforms,
            "toolchain_paths": self.compilation_engine.toolchain_paths
        }

    def compile_plugin(self, plugin_spec: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        """Compile a single plugin."""
        plugin_name = plugin_spec.get("name", "unknown")
        backend_type = plugin_spec.get("backend_type", "unknown")

        print(f"🔨 Compiling {backend_type} plugin: {plugin_name}")

        if not self.compilation_engine:
            return {
                "plugin_name": plugin_name,
                "status": "error",
                "message": "Compilation engine not initialized"
            }

        # Prepare compilation spec
        compile_spec = {
            "name": plugin_name,
            "backend_dir": plugin_spec["path"],
            "backend_type": backend_type
        }

        start_time = time.perf_counter()

        try:
            # Run compilation
            result = self.compilation_engine.compile_plugin(compile_spec)

            compilation_time = time.perf_counter() - start_time

            # Enhance result with timing and plugin info
            result.update({
                "plugin_name": plugin_name,
                "backend_type": backend_type,
                "compilation_time": compilation_time,
                "source_path": plugin_spec["path"]
            })

            if result["status"] == "success":
                print(f"  ✅ Compilation successful ({compilation_time:.2f}s)")
                print(f"     Output: {result.get('output_path', 'N/A')}")
            else:
                print(f"  ❌ Compilation failed ({compilation_time:.2f}s)")
                if verbose and result.get("error_message"):
                    print(f"     Error: {result['error_message']}")

            return result

        except Exception as e:
            compilation_time = time.perf_counter() - start_time
            return {
                "plugin_name": plugin_name,
                "backend_type": backend_type,
                "status": "error",
                "compilation_time": compilation_time,
                "error_message": str(e),
                "source_path": plugin_spec["path"]
            }

    def compile_all_projects(self, projects: List[Dict[str, Any]], verbose: bool = False) -> List[Dict[str, Any]]:
        """Compile all detected projects."""
        print(f"🚀 Starting compilation of {len(projects)} projects...")
        print()

        results = []

        for i, project in enumerate(projects, 1):
            print(f"[{i}/{len(projects)}] ", end="")
            result = self.compile_plugin(project, verbose)
            results.append(result)

            # Register successful compilations
            if result["status"] == "success" and self.plugin_registry:
                try:
                    # Create plugin definition for registry
                    plugin_def = {
                        "name": result["plugin_name"],
                        "backend_type": result["backend_type"],
                        "version": "1.0.0",
                        "binary_path": result.get("output_path"),
                        "source_path": result["source_path"]
                    }

                    # This would register in the plugin registry
                    # self.plugin_registry.register_plugin(plugin_def)

                except Exception as e:
                    if verbose:
                        print(f"     Warning: Failed to register plugin: {e}")

            print()

        return results

    def generate_compilation_report(self, results: List[Dict[str, Any]], output_format: str = "json",
                                  output_file: Optional[str] = None) -> None:
        """Generate compilation report."""
        successful = [r for r in results if r["status"] == "success"]
        failed = [r for r in results if r["status"] == "error"]

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "compilation_summary": {
                "total_projects": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results) * 100 if results else 0
            },
            "results": results,
            "backend_breakdown": {}
        }

        # Calculate backend-specific statistics
        backends = set(r["backend_type"] for r in results)
        for backend in backends:
            backend_results = [r for r in results if r["backend_type"] == backend]
            backend_successful = [r for r in backend_results if r["status"] == "success"]

            report["backend_breakdown"][backend] = {
                "total": len(backend_results),
                "successful": len(backend_successful),
                "success_rate": len(backend_successful) / len(backend_results) * 100 if backend_results else 0
            }

        if output_format == "json":
            report_content = json.dumps(report, indent=2)
        else:
            # Text format
            lines = [
                f"Plugin Compilation Report",
                f"Generated: {report['timestamp']}",
                f"",
                f"Summary:",
                f"  Total projects: {report['compilation_summary']['total_projects']}",
                f"  Successful: {report['compilation_summary']['successful']}",
                f"  Failed: {report['compilation_summary']['failed']}",
                f"  Success rate: {report['compilation_summary']['success_rate']:.1f}%",
                f"",
                f"Backend Breakdown:"
            ]

            for backend, stats in report["backend_breakdown"].items():
                lines.append(f"  {backend.upper()}:")
                lines.append(f"    Total: {stats['total']}")
                lines.append(f"    Successful: {stats['successful']}")
                lines.append(f"    Success rate: {stats['success_rate']:.1f}%")

            lines.extend([f"", f"Detailed Results:"])

            for result in results:
                status_emoji = "✅" if result["status"] == "success" else "❌"
                lines.append(f"  {status_emoji} {result['plugin_name']} ({result['backend_type']})")
                lines.append(f"    Time: {result.get('compilation_time', 0):.2f}s")

                if result["status"] == "success":
                    lines.append(f"    Output: {result.get('output_path', 'N/A')}")
                else:
                    lines.append(f"    Error: {result.get('error_message', 'Unknown error')}")
                lines.append("")

            report_content = '\n'.join(lines)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            print(f"📄 Compilation report saved to {output_file}")
        else:
            print(report_content)

    def main(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description="Debug Framework Plugin Compilation Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Auto-discover and compile all backend projects
  plugin-compile --search /path/to/backends --output-dir ./compiled

  # Compile specific project
  plugin-compile --project /path/to/metal-backend --output-dir ./compiled

  # Check toolchain availability
  plugin-compile --check-toolchain metal

  # Generate compilation report
  plugin-compile --search . --output-dir ./build --report build_report.json
            """
        )

        parser.add_argument(
            "--search", "-s",
            help="Search path for backend projects (auto-discovery mode)"
        )

        parser.add_argument(
            "--project", "-p",
            help="Specific project directory to compile"
        )

        parser.add_argument(
            "--output-dir", "-o",
            default="./compiled_plugins",
            help="Output directory for compiled plugins"
        )

        parser.add_argument(
            "--backend-type", "-t",
            choices=["metal", "cuda", "cpp"],
            help="Backend type (required for --project mode)"
        )

        parser.add_argument(
            "--check-toolchain",
            choices=["metal", "cuda", "cpp"],
            help="Check toolchain availability for backend type"
        )

        parser.add_argument(
            "--database", "-d",
            help="Path to plugin registry database"
        )

        parser.add_argument(
            "--report", "-r",
            help="Generate compilation report file"
        )

        parser.add_argument(
            "--format", "-f",
            choices=["json", "text"],
            default="text",
            help="Report output format"
        )

        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose output"
        )

        args = parser.parse_args()

        # Initialize services
        if not self.initialize_services(args.output_dir, args.database):
            sys.exit(1)

        # Handle toolchain check mode
        if args.check_toolchain:
            toolchain_info = self.validate_toolchain(args.check_toolchain)

            print(f"🔧 Toolchain validation for {args.check_toolchain.upper()}:")
            print(f"   Status: {'✅ Available' if toolchain_info['is_valid'] else '❌ Not available'}")
            print(f"   Supported platforms: {', '.join(toolchain_info['supported_platforms'])}")

            if args.verbose:
                print(f"   Toolchain paths:")
                for tool, path in toolchain_info['toolchain_paths'].items():
                    print(f"     {tool}: {path}")

            sys.exit(0 if toolchain_info['is_valid'] else 1)

        # Determine compilation mode and projects
        projects = []

        if args.search:
            # Auto-discovery mode
            projects = self.detect_backend_projects(args.search)

        elif args.project:
            # Single project mode
            if not args.backend_type:
                # Try to auto-detect backend type
                backend_type = self._detect_backend_type(Path(args.project))
                if not backend_type:
                    print("❌ Could not detect backend type. Please specify --backend-type", file=sys.stderr)
                    sys.exit(1)
            else:
                backend_type = args.backend_type

            projects = [{
                "name": os.path.basename(args.project),
                "path": args.project,
                "backend_type": backend_type
            }]

        else:
            print("❌ Either --search or --project must be specified", file=sys.stderr)
            sys.exit(1)

        if not projects:
            print("❌ No compilable projects found")
            sys.exit(1)

        print(f"📦 Found {len(projects)} projects to compile")
        print(f"📁 Output directory: {args.output_dir}")
        print()

        # Compile projects
        self.compilation_results = self.compile_all_projects(projects, args.verbose)

        # Generate summary
        successful = len([r for r in self.compilation_results if r["status"] == "success"])
        failed = len([r for r in self.compilation_results if r["status"] == "error"])

        print(f"📊 Compilation Summary:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📈 Success rate: {successful / len(self.compilation_results) * 100:.1f}%")
        print()

        # Generate report if requested
        if args.report:
            self.generate_compilation_report(self.compilation_results, args.format, args.report)

        # Exit with appropriate code
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    cli = PluginCompileCLI()
    cli.main()