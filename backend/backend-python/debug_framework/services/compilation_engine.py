"""
Compilation engine service for automated plugin builds.

This service is a simple wrapper around CMake that delegates all compilation
tracking, caching, and state management to the underlying build system.
"""

import os
import subprocess
import platform
import shutil
from typing import Dict, Optional, Any


class CompilationEngine:
    """
    Simplified compilation engine that delegates to CMake.

    This engine acts as a thin wrapper around CMake-based builds for different
    backend types (Metal, CUDA, C++). All caching, dependency tracking, and
    state management is handled by CMake.
    """

    def __init__(self, output_directory: str, toolchain_paths: Dict[str, str] = None):
        """
        Initialize the compilation engine.

        Args:
            output_directory: Directory for compiled plugin binaries
            toolchain_paths: Paths to build toolchains (cmake, xcrun, nvcc)
        """
        self.output_directory = output_directory or "/tmp/compiled_plugins"
        self.toolchain_paths = toolchain_paths or self.detect_available_toolchains()

        # Ensure output directory exists
        os.makedirs(self.output_directory, exist_ok=True)

    def detect_available_toolchains(self) -> Dict[str, str]:
        """Auto-detect available build toolchains."""
        toolchains = {}

        # Essential build tools
        build_tools = ["cmake"]

        # Platform-specific tools
        if platform.system() == "Darwin":
            build_tools.append("xcrun")

        # Optional CUDA support
        if os.environ.get("CUDA_HOME") or shutil.which("nvcc"):
            build_tools.append("nvcc")

        for tool in build_tools:
            path = shutil.which(tool)
            if path:
                toolchains[tool] = path

        return toolchains

    def compile_plugin(self, plugin_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compile a plugin using CMake.

        Args:
            plugin_spec: Plugin specification with keys:
                - name: Plugin name
                - backend_dir: Path to backend directory containing CMakeLists.txt
                - backend_type: Backend type (metal, cuda, cpp)

        Returns:
            Dict with compilation result:
                - status: "success" or "error"
                - output_path: Path to compiled artifact (if successful)
                - error_message: Error description (if failed)
        """
        try:
            # Validate CMake availability
            if "cmake" not in self.toolchain_paths:
                return {
                    "status": "error",
                    "output_path": None,
                    "error_message": "CMake not found. Please install CMake."
                }

            # Validate backend directory
            backend_dir = plugin_spec.get("backend_dir")
            if not backend_dir or not os.path.exists(backend_dir):
                return {
                    "status": "error",
                    "output_path": None,
                    "error_message": f"Backend directory not found: {backend_dir}"
                }

            cmake_lists_path = os.path.join(backend_dir, "CMakeLists.txt")
            if not os.path.exists(cmake_lists_path):
                return {
                    "status": "error",
                    "output_path": None,
                    "error_message": f"CMakeLists.txt not found in {backend_dir}"
                }

            # Create build directory
            plugin_name = plugin_spec.get("name", "unknown_plugin")
            build_dir = os.path.join(self.output_directory, f"{plugin_name}_build")

            # Step 1: Configure with CMake
            configure_cmd = [
                self.toolchain_paths["cmake"],
                "-B", build_dir,
                "-S", backend_dir,
                "-DCMAKE_BUILD_TYPE=Release"
            ]

            configure_result = subprocess.run(
                configure_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )

            if configure_result.returncode != 0:
                return {
                    "status": "error",
                    "output_path": None,
                    "error_message": f"CMake configure failed: {configure_result.stderr}"
                }

            # Step 2: Build with CMake
            build_cmd = [
                self.toolchain_paths["cmake"],
                "--build", build_dir
            ]

            build_result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if build_result.returncode != 0:
                return {
                    "status": "error",
                    "output_path": None,
                    "error_message": f"CMake build failed: {build_result.stderr}"
                }

            # Determine output path based on backend type
            output_path = self._get_output_path(build_dir, plugin_spec)

            if not os.path.exists(output_path):
                return {
                    "status": "error",
                    "output_path": None,
                    "error_message": f"Expected output not found: {output_path}"
                }

            return {
                "status": "success",
                "output_path": output_path
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "output_path": None,
                "error_message": "Compilation timed out"
            }
        except Exception as e:
            return {
                "status": "error",
                "output_path": None,
                "error_message": f"Compilation error: {str(e)}"
            }

    def _get_output_path(self, build_dir: str, plugin_spec: Dict[str, Any]) -> str:
        """Get expected output path based on backend type."""
        backend_type = plugin_spec.get("backend_type", "cpp")
        plugin_name = plugin_spec.get("name", "plugin")

        if backend_type == "metal":
            return os.path.join(build_dir, "lib", "pie_metal_kernels.metallib")
        elif backend_type == "cuda":
            return os.path.join(build_dir, "bin", "pie_cuda_be")
        else:  # cpp or generic
            if platform.system() == "Darwin":
                return os.path.join(build_dir, "lib", f"lib{plugin_name}.dylib")
            else:
                return os.path.join(build_dir, "lib", f"lib{plugin_name}.so")

    def get_supported_platforms(self) -> list:
        """Get list of supported compilation platforms based on available toolchains."""
        supported = []

        if "cmake" in self.toolchain_paths:
            supported.append("cpp")

        if "cmake" in self.toolchain_paths and "nvcc" in self.toolchain_paths:
            supported.append("cuda")

        if ("cmake" in self.toolchain_paths and "xcrun" in self.toolchain_paths and
            platform.system() == "Darwin"):
            supported.append("metal")

        return supported

    def validate_toolchain(self, target_platform: str) -> bool:
        """Validate that required toolchain is available for target platform."""
        if target_platform == "metal":
            return ("cmake" in self.toolchain_paths and "xcrun" in self.toolchain_paths and
                   platform.system() == "Darwin")
        elif target_platform == "cuda":
            return "cmake" in self.toolchain_paths and "nvcc" in self.toolchain_paths
        else:  # cpp or generic
            return "cmake" in self.toolchain_paths