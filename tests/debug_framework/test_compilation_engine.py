"""
Test module for CompilationEngine service.

This test module validates the CompilationEngine service which handles
automated compilation of validation plugins for different platforms,
including Metal (macOS), CUDA (Linux), and generic C++ backends.

TDD: This test MUST FAIL until the CompilationEngine service is implemented.
"""

import pytest
import os
import tempfile
import platform
import subprocess
import time
import hashlib
from unittest.mock import patch, MagicMock, mock_open, call
from pathlib import Path

# Proper TDD pattern - use try/except for conditional loading
try:
    from debug_framework.services.compilation_engine import CompilationEngine
    COMPILATION_ENGINE_AVAILABLE = True
except ImportError:
    CompilationEngine = None
    COMPILATION_ENGINE_AVAILABLE = False


class TestCompilationEngine:
    """Test suite for CompilationEngine service functionality."""

    def test_compilation_engine_import_fails(self):
        """Test that CompilationEngine import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.services.compilation_engine import CompilationEngine

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_compilation_engine_initialization(self):
        """Test CompilationEngine service initialization."""
        engine = CompilationEngine(
            output_directory="/tmp/compiled_plugins",
            cache_directory="/tmp/compilation_cache",
            toolchain_paths={"clang": "/usr/bin/clang", "nvcc": "/usr/local/cuda/bin/nvcc"}
        )

        assert engine.output_directory == "/tmp/compiled_plugins"
        assert engine.cache_directory == "/tmp/compilation_cache"
        assert engine.toolchain_paths["clang"] == "/usr/bin/clang"
        assert engine.compilation_history == []
        assert engine.active_compilations == {}

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_toolchain_detection(self):
        """Test automatic detection of available toolchains."""
        engine = CompilationEngine("/tmp/compiled_plugins")

        with patch('shutil.which') as mock_which:
            # Mock different toolchain availability per platform
            def mock_which_behavior(cmd):
                toolchain_map = {
                    "clang": "/usr/bin/clang",
                    "clang++": "/usr/bin/clang++",
                }

                if platform.system() == "Darwin":
                    toolchain_map["xcrun"] = "/usr/bin/xcrun"

                if cmd == "nvcc":
                    # Only available if CUDA is installed
                    return "/usr/local/cuda/bin/nvcc" if os.environ.get("CUDA_HOME") else None

                return toolchain_map.get(cmd)

            mock_which.side_effect = mock_which_behavior

            toolchains = engine.detect_available_toolchains()

            # Always available
            assert "cpp" in toolchains
            assert toolchains["cpp"]["compiler"] == "/usr/bin/clang++"

            # Platform-specific
            if platform.system() == "Darwin":
                assert "metal" in toolchains
                assert toolchains["metal"]["compiler"] == "/usr/bin/xcrun"

            # Environment-dependent
            if os.environ.get("CUDA_HOME"):
                assert "cuda" in toolchains
                assert toolchains["cuda"]["compiler"] == "/usr/local/cuda/bin/nvcc"

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    @pytest.mark.skipif(platform.system() != "Darwin", reason="Metal compilation only available on macOS")
    def test_metal_compilation_macos(self):
        """Test Metal plugin compilation on macOS with proper two-step pipeline."""
        engine = CompilationEngine("/tmp/compiled_plugins")

        plugin_spec = {
            "name": "metal_attention_plugin",
            "source_file": "/path/to/metal_attention.cpp",
            "backend_type": "metal",
            "compilation_flags": ["-std=c++17", "-O3"],
            "frameworks": ["Metal", "Foundation"],
            "include_paths": ["/usr/local/include"],
            "library_paths": ["/usr/local/lib"]
        }

        with patch('subprocess.run') as mock_subprocess, \
             patch('shutil.which') as mock_which, \
             patch('os.path.exists') as mock_exists:

            # Mock xcrun availability
            mock_which.side_effect = lambda cmd: "/usr/bin/xcrun" if cmd == "xcrun" else None
            mock_exists.return_value = True

            # Mock two-step Metal compilation process
            mock_subprocess.side_effect = [
                # Step 1: xcrun metal -c source.cpp -> source.air
                MagicMock(returncode=0, stdout=b"Metal frontend compilation successful"),
                # Step 2: xcrun metallib source.air -> library.metallib
                MagicMock(returncode=0, stdout=b"Metal library creation successful")
            ]

            result = engine.compile_plugin(plugin_spec)

            assert result["status"] == "success"
            assert result["output_path"].endswith(".metallib")
            assert "metal_attention_plugin" in result["output_path"]

            # Verify two-step Metal compilation pipeline
            assert mock_subprocess.call_count == 2

            # First call: xcrun metal compilation to .air
            first_call_cmd = mock_subprocess.call_args_list[0][0][0]
            assert "xcrun" in first_call_cmd[0]
            assert "metal" in first_call_cmd
            assert "-c" in first_call_cmd  # Compile only, don't link
            assert any(arg.endswith(".air") for arg in first_call_cmd)  # Output .air file

            # Second call: xcrun metallib to create .metallib
            second_call_cmd = mock_subprocess.call_args_list[1][0][0]
            assert "xcrun" in second_call_cmd[0]
            assert "metallib" in second_call_cmd
            assert any(arg.endswith(".metallib") for arg in second_call_cmd)  # Output .metallib

            # Verify Metal-specific frameworks
            combined_cmd = " ".join(first_call_cmd + second_call_cmd)
            assert "-framework" in combined_cmd
            assert "Metal" in combined_cmd

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_cuda_compilation_with_nvcc_gating(self):
        """Test CUDA plugin compilation with proper nvcc gating."""
        engine = CompilationEngine("/tmp/compiled_plugins")

        plugin_spec = {
            "name": "cuda_softmax_plugin",
            "source_file": "/path/to/cuda_softmax.cu",
            "backend_type": "cuda",
            "compilation_flags": ["-O3", "-arch=sm_75", "--shared"],
            "include_paths": ["/usr/local/cuda/include"],
            "library_paths": ["/usr/local/cuda/lib64"],
            "libraries": ["cuda", "cudart"],
            "host_compiler_flags": ["-Wall", "-Wextra"],  # Host compiler specific flags
            "host_compiler_path": "/usr/bin/gcc-9"  # Specific host compiler selection
        }

        with patch('subprocess.run') as mock_subprocess, \
             patch('shutil.which') as mock_which, \
             patch('os.path.exists') as mock_exists:

            # Test with nvcc available
            mock_which.side_effect = lambda cmd: "/usr/local/cuda/bin/nvcc" if cmd == "nvcc" else None
            mock_exists.return_value = True
            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stdout=b"CUDA compilation successful",
                stderr=b""
            )

            result = engine.compile_plugin(plugin_spec)

            assert result["status"] == "success"
            assert result["output_path"].endswith(".so")

            # Verify CUDA-specific compilation command construction
            compile_cmd = mock_subprocess.call_args[0][0]
            assert "nvcc" in compile_cmd[0]
            assert "-arch=sm_75" in compile_cmd
            assert "--shared" in compile_cmd

            # Verify host compiler flags are passed through --compiler-options
            cmd_str = " ".join(compile_cmd)
            assert "--compiler-options" in cmd_str or "-Xcompiler" in cmd_str
            if "--compiler-options" in cmd_str:
                # Flags should be grouped after --compiler-options
                assert "-Wall,-Wextra" in cmd_str or "\"-Wall -Wextra\"" in cmd_str

            # Verify host compiler selection via -ccbin or --compiler-bindir
            assert any(flag in cmd_str for flag in ["-ccbin", "--compiler-bindir"])
            if "-ccbin" in cmd_str:
                assert "/usr/bin/gcc-9" in cmd_str
            elif "--compiler-bindir" in cmd_str:
                assert "/usr/bin" in cmd_str  # Directory containing gcc-9

        # Test fallback when nvcc unavailable
        with patch('shutil.which') as mock_which:
            mock_which.side_effect = lambda cmd: None  # nvcc not found

            result = engine.compile_plugin(plugin_spec)

            assert result["status"] == "error"
            assert "CUDA compiler not found" in result["error_message"]

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_cpp_compilation_generic(self):
        """Test generic C++ plugin compilation with proper toolchain detection."""
        engine = CompilationEngine("/tmp/compiled_plugins")

        plugin_spec = {
            "name": "cpp_reference_plugin",
            "source_file": "/path/to/reference.cpp",
            "backend_type": "cpp",
            "compilation_flags": ["-std=c++17", "-O3", "-fPIC", "-shared"],
            "include_paths": ["/usr/local/include", "/opt/homebrew/include"],
            "library_paths": ["/usr/local/lib"],
            "libraries": ["pthread"]
        }

        with patch('subprocess.run') as mock_subprocess, \
             patch('shutil.which') as mock_which, \
             patch('os.path.exists') as mock_exists:

            # Mock compiler availability (prefer clang++ over g++)
            mock_which.side_effect = lambda cmd: {
                "clang++": "/usr/bin/clang++",
                "g++": "/usr/bin/g++"
            }.get(cmd)

            mock_exists.return_value = True
            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stdout=b"C++ compilation successful"
            )

            result = engine.compile_plugin(plugin_spec)

            assert result["status"] == "success"
            output_ext = ".dylib" if platform.system() == "Darwin" else ".so"
            assert result["output_path"].endswith(output_ext)

            # Verify C++ compilation command
            compile_cmd = mock_subprocess.call_args[0][0]
            assert any(compiler in compile_cmd[0] for compiler in ["clang++", "g++"])
            assert "-std=c++17" in compile_cmd
            assert "-fPIC" in compile_cmd
            assert "-shared" in compile_cmd

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_compilation_error_handling(self):
        """Test handling of compilation errors and failures."""
        engine = CompilationEngine("/tmp/compiled_plugins")

        plugin_spec = {
            "name": "broken_plugin",
            "source_file": "/path/to/broken.cpp",
            "backend_type": "cpp",
            "compilation_flags": ["-std=c++17"]
        }

        # Test compilation failure
        with patch('subprocess.run') as mock_subprocess, \
             patch('os.path.exists') as mock_exists:

            mock_exists.return_value = True
            mock_subprocess.return_value = MagicMock(
                returncode=1,
                stdout=b"",
                stderr=b"error: 'undeclared_function' was not declared in this scope"
            )

            result = engine.compile_plugin(plugin_spec)

            assert result["status"] == "error"
            assert "undeclared_function" in result["error_message"]
            assert result["output_path"] is None
            assert "stderr" in result

        # Test missing source file
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False

            result = engine.compile_plugin(plugin_spec)

            assert result["status"] == "error"
            assert "Source file not found" in result["error_message"]

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_content_based_compilation_caching(self):
        """Test compilation result caching based on content hashing."""
        engine = CompilationEngine("/tmp/compiled_plugins", "/tmp/compilation_cache")

        plugin_spec = {
            "name": "cacheable_plugin",
            "source_file": "/path/to/plugin.cpp",
            "backend_type": "cpp",
            "compilation_flags": ["-std=c++17", "-O3"]
        }

        source_content = """
        #include <iostream>
        extern "C" void test_function() {
            std::cout << "Test function" << std::endl;
        }
        """

        # Generate content hash
        content_hash = hashlib.sha256(source_content.encode()).hexdigest()

        with patch('builtins.open', mock_open(read_data=source_content)) as mock_file, \
             patch('os.path.exists') as mock_exists, \
             patch('os.path.getmtime') as mock_getmtime:

            # First compilation - no cache
            mock_exists.side_effect = lambda x: "cache" not in x  # Cache doesn't exist yet
            mock_getmtime.return_value = 1000.0

            with patch('subprocess.run') as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0, stdout=b"Compilation successful")

                result1 = engine.compile_plugin(plugin_spec)

                assert result1["status"] == "success"
                assert result1.get("from_cache") is False
                assert result1["content_hash"] == content_hash

            # Second compilation with same content - should use cache
            mock_exists.side_effect = lambda x: True  # Cache exists

            with patch.object(engine, '_load_cached_result') as mock_load_cache:
                cached_result = {
                    "status": "success",
                    "output_path": "/tmp/compilation_cache/cacheable_plugin.so",
                    "compilation_time": 2.5,
                    "content_hash": content_hash,
                    "from_cache": True
                }
                mock_load_cache.return_value = cached_result

                result2 = engine.compile_plugin(plugin_spec)

                assert result2["status"] == "success"
                assert result2["from_cache"] is True
                assert result2["content_hash"] == content_hash
                mock_load_cache.assert_called_once()

        # Test cache invalidation with content change
        modified_content = source_content.replace("Test function", "Modified function")
        modified_hash = hashlib.sha256(modified_content.encode()).hexdigest()

        with patch('builtins.open', mock_open(read_data=modified_content)):
            with patch('subprocess.run') as mock_subprocess:
                mock_subprocess.return_value = MagicMock(returncode=0, stdout=b"Recompilation successful")

                result3 = engine.compile_plugin(plugin_spec)

                assert result3["status"] == "success"
                assert result3.get("from_cache") is False  # Content changed, cache invalidated
                assert result3["content_hash"] == modified_hash

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_dependency_resolution_with_mocked_pkg_config(self):
        """Test resolution of plugin compilation dependencies with mocked pkg-config."""
        engine = CompilationEngine("/tmp/compiled_plugins")

        plugin_spec = {
            "name": "dependent_plugin",
            "source_file": "/path/to/dependent.cpp",
            "backend_type": "cpp",
            "dependencies": [
                {"name": "base_lib", "type": "static", "path": "/path/to/libbase.a"},
                {"name": "math_lib", "type": "dynamic", "path": "/usr/lib/libm.so"}
            ],
            "pkg_config_deps": ["eigen3", "boost"]
        }

        with patch('subprocess.run') as mock_subprocess, \
             patch('os.path.exists') as mock_exists:

            mock_exists.return_value = True

            # Mock pkg-config calls with proper sequencing
            pkg_config_responses = [
                # pkg-config --cflags eigen3
                MagicMock(returncode=0, stdout=b"-I/usr/include/eigen3 -DEIGEN_NO_DEBUG"),
                # pkg-config --libs eigen3
                MagicMock(returncode=0, stdout=b"-leigen3"),
                # pkg-config --cflags boost
                MagicMock(returncode=0, stdout=b"-I/usr/include/boost -DBOOST_ALL_NO_LIB"),
                # pkg-config --libs boost
                MagicMock(returncode=0, stdout=b"-lboost_system -lboost_filesystem"),
                # Actual compilation with assembled flags
                MagicMock(returncode=0, stdout=b"Compilation with dependencies successful")
            ]

            mock_subprocess.side_effect = pkg_config_responses

            result = engine.compile_plugin(plugin_spec)

            assert result["status"] == "success"

            # Verify pkg-config was called for each dependency
            pkg_config_calls = [call for call in mock_subprocess.call_args_list
                              if len(call[0]) > 0 and call[0][0][0] == "pkg-config"]
            assert len(pkg_config_calls) >= 4  # 2 deps * 2 calls each (cflags + libs)

            # Verify final compilation includes dependency flags
            final_compilation_call = mock_subprocess.call_args_list[-1]
            final_cmd = " ".join(final_compilation_call[0][0])

            # Should include pkg-config derived flags
            assert "-I/usr/include/eigen3" in final_cmd
            assert "-leigen3" in final_cmd
            assert "-I/usr/include/boost" in final_cmd
            assert "-lboost_system" in final_cmd

            # Should include direct dependencies
            assert "/path/to/libbase.a" in final_cmd
            assert "/usr/lib/libm.so" in final_cmd or "-lm" in final_cmd

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_parallel_compilation_with_concurrency_isolation(self):
        """Test parallel compilation of multiple plugins with proper concurrency."""
        engine = CompilationEngine("/tmp/compiled_plugins")

        plugin_specs = [
            {
                "name": f"parallel_plugin_{i}",
                "source_file": f"/path/to/plugin_{i}.cpp",
                "backend_type": "cpp",
                "compilation_flags": ["-std=c++17", "-O3"],
                "unique_id": i  # For tracking in mocks
            }
            for i in range(3)
        ]

        with patch('subprocess.run') as mock_subprocess, \
             patch('os.path.exists') as mock_exists:

            mock_exists.return_value = True

            # Track call order and arguments for concurrency validation
            call_tracker = []

            def track_subprocess_calls(*args, **kwargs):
                cmd = args[0]
                # Extract plugin ID from command arguments
                plugin_id = next((spec["unique_id"] for spec in plugin_specs
                                if spec["source_file"] in " ".join(cmd)), None)
                call_tracker.append({
                    "plugin_id": plugin_id,
                    "timestamp": time.perf_counter(),
                    "command": cmd
                })
                return MagicMock(returncode=0, stdout=b"Compilation successful")

            mock_subprocess.side_effect = track_subprocess_calls

            results = engine.compile_plugins_parallel(plugin_specs, max_workers=2)

            assert len(results) == 3
            assert all(result["status"] == "success" for result in results)
            assert mock_subprocess.call_count == 3

            # Verify parallel execution characteristics
            assert len(call_tracker) == 3

            # Verify each plugin was compiled with correct arguments
            for i, spec in enumerate(plugin_specs):
                matching_calls = [call for call in call_tracker if call["plugin_id"] == i]
                assert len(matching_calls) == 1

                cmd = matching_calls[0]["command"]
                assert f"plugin_{i}.cpp" in " ".join(cmd)
                assert "-std=c++17" in cmd
                assert "-O3" in cmd

            # Verify concurrency: with 2 workers, calls should be made (deterministic check)
            if len(call_tracker) >= 2:
                # Verify all plugins were processed (deterministic)
                plugin_ids_processed = {call["plugin_id"] for call in call_tracker}
                assert plugin_ids_processed == {0, 1, 2}

                # Verify each plugin had exactly one compilation call
                for plugin_id in plugin_ids_processed:
                    plugin_calls = [call for call in call_tracker if call["plugin_id"] == plugin_id]
                    assert len(plugin_calls) == 1

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_cross_compilation_support(self):
        """Test cross-compilation for different target architectures."""
        engine = CompilationEngine("/tmp/compiled_plugins")

        plugin_spec = {
            "name": "cross_compiled_plugin",
            "source_file": "/path/to/plugin.cpp",
            "backend_type": "cpp",
            "target_arch": "arm64",
            "cross_compile_toolchain": "/opt/cross/bin/aarch64-linux-gnu-g++",
            "compilation_flags": ["-std=c++17", "-O3"]
        }

        with patch('subprocess.run') as mock_subprocess, \
             patch('os.path.exists') as mock_exists, \
             patch('shutil.which') as mock_which:

            # Mock cross-compiler availability
            mock_which.side_effect = lambda cmd: {
                "/opt/cross/bin/aarch64-linux-gnu-g++": "/opt/cross/bin/aarch64-linux-gnu-g++",
                "aarch64-linux-gnu-g++": "/opt/cross/bin/aarch64-linux-gnu-g++"
            }.get(cmd)

            mock_exists.return_value = True
            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stdout=b"Cross-compilation successful"
            )

            result = engine.compile_plugin(plugin_spec)

            assert result["status"] == "success"
            assert result["target_architecture"] == "arm64"

            # Verify cross-compilation toolchain was used
            compile_cmd = mock_subprocess.call_args[0][0]
            assert "aarch64-linux-gnu-g++" in compile_cmd[0]

            # Verify target-specific flags
            cmd_str = " ".join(compile_cmd)
            # Should include architecture-specific optimizations
            assert any(flag in cmd_str for flag in ["-march=", "-mcpu=", "-mtune="])

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_compilation_with_monotonic_progress_tracking(self):
        """Test progress tracking using monotonic clock for deterministic timing."""
        engine = CompilationEngine("/tmp/compiled_plugins")

        plugin_specs = [
            {
                "name": f"progress_plugin_{i}",
                "source_file": f"/path/to/plugin_{i}.cpp",
                "backend_type": "cpp"
            }
            for i in range(5)
        ]

        progress_updates = []

        def progress_callback(completed, total, current_plugin):
            # Use monotonic time for consistent progress tracking
            timestamp = time.perf_counter()
            progress_updates.append({
                "completed": completed,
                "total": total,
                "current_plugin": current_plugin,
                "timestamp": timestamp
            })

        with patch('subprocess.run') as mock_subprocess, \
             patch('os.path.exists') as mock_exists:

            mock_exists.return_value = True
            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stdout=b"Compilation successful"
            )

            # Mock time.perf_counter to return predictable monotonic values
            with patch('time.perf_counter') as mock_perf_counter:
                mock_perf_counter.side_effect = [i * 1.0 for i in range(20)]  # Monotonic sequence

                results = engine.compile_plugins_with_progress(
                    plugin_specs,
                    progress_callback=progress_callback
                )

                assert len(results) == 5
                assert len(progress_updates) == 5

                # Verify progress is monotonic and correct
                for i, update in enumerate(progress_updates):
                    assert update["completed"] == i + 1
                    assert update["total"] == 5
                    assert update["current_plugin"] == f"progress_plugin_{i}"

                    # Verify timestamps are monotonic (non-decreasing)
                    if i > 0:
                        assert update["timestamp"] >= progress_updates[i-1]["timestamp"]

                # Final update should show completion
                assert progress_updates[-1]["completed"] == 5
                assert progress_updates[-1]["total"] == 5

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_compilation_warnings_with_nvcc_flag_routing(self):
        """Test handling of compilation warnings with proper nvcc flag routing."""
        engine = CompilationEngine("/tmp/compiled_plugins")

        # Test C++ plugin with strict warnings
        cpp_plugin_spec = {
            "name": "strict_cpp_plugin",
            "source_file": "/path/to/plugin.cpp",
            "backend_type": "cpp",
            "warning_level": "strict",
            "treat_warnings_as_errors": True,
            "warning_flags": ["-Wall", "-Wextra", "-Wpedantic"]
        }

        with patch('subprocess.run') as mock_subprocess, \
             patch('os.path.exists') as mock_exists, \
             patch('shutil.which') as mock_which:

            mock_which.side_effect = lambda cmd: "/usr/bin/clang++" if cmd == "clang++" else None
            mock_exists.return_value = True
            mock_subprocess.return_value = MagicMock(
                returncode=1,
                stdout=b"",
                stderr=b"warning: unused variable 'x' [-Wunused-variable]"
            )

            result = engine.compile_plugin(cpp_plugin_spec)

            assert result["status"] == "error"
            assert "unused variable" in result["error_message"]

            # Verify strict warning flags were used
            compile_cmd = mock_subprocess.call_args[0][0]
            assert "-Wall" in compile_cmd
            assert "-Wextra" in compile_cmd
            assert "-Werror" in compile_cmd

        # Test CUDA plugin with host compiler flag routing
        cuda_plugin_spec = {
            "name": "strict_cuda_plugin",
            "source_file": "/path/to/plugin.cu",
            "backend_type": "cuda",
            "warning_level": "strict",
            "host_compiler_flags": ["-Wall", "-Wextra"],
            "host_compiler_path": "/usr/bin/g++-10"  # Specific host compiler
        }

        with patch('subprocess.run') as mock_subprocess, \
             patch('os.path.exists') as mock_exists, \
             patch('shutil.which') as mock_which:

            mock_which.side_effect = lambda cmd: "/usr/local/cuda/bin/nvcc" if cmd == "nvcc" else None
            mock_exists.return_value = True
            mock_subprocess.return_value = MagicMock(returncode=0, stdout=b"CUDA compilation successful")

            result = engine.compile_plugin(cuda_plugin_spec)

            assert result["status"] == "success"

            # Verify host compiler flags are routed through --compiler-options
            compile_cmd = mock_subprocess.call_args[0][0]
            cmd_str = " ".join(compile_cmd)

            # nvcc should be used
            assert "nvcc" in compile_cmd[0]

            # Host compiler flags should be passed through --compiler-options or -Xcompiler
            assert ("--compiler-options" in cmd_str and ("-Wall" in cmd_str and "-Wextra" in cmd_str)) or \
                   ("-Xcompiler" in cmd_str)

            # Verify host compiler selection
            assert any(flag in cmd_str for flag in ["-ccbin", "--compiler-bindir"])
            if "-ccbin" in cmd_str:
                assert "/usr/bin/g++-10" in cmd_str
            elif "--compiler-bindir" in cmd_str:
                assert "/usr/bin" in cmd_str

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_controlled_environment_propagation(self):
        """Test proper environment setup for compilation with controlled variables."""
        engine = CompilationEngine("/tmp/compiled_plugins")

        plugin_spec = {
            "name": "env_plugin",
            "source_file": "/path/to/plugin.cpp",
            "backend_type": "cpp",
            "environment_vars": {
                "CUDA_HOME": "/usr/local/cuda",
                "LD_LIBRARY_PATH": "/usr/local/lib:/opt/lib",
                "PKG_CONFIG_PATH": "/usr/local/lib/pkgconfig"
            }
        }

        with patch('subprocess.run') as mock_subprocess, \
             patch('os.path.exists') as mock_exists:

            mock_exists.return_value = True
            mock_subprocess.return_value = MagicMock(
                returncode=0,
                stdout=b"Compilation successful"
            )

            result = engine.compile_plugin(plugin_spec)

            assert result["status"] == "success"

            # Verify environment variables were set in subprocess call
            _, kwargs = mock_subprocess.call_args
            env = kwargs.get("env", {})

            # Should inherit current environment plus specified variables
            assert env.get("CUDA_HOME") == "/usr/local/cuda"
            assert "/usr/local/lib" in env.get("LD_LIBRARY_PATH", "")
            assert "/opt/lib" in env.get("LD_LIBRARY_PATH", "")
            assert env.get("PKG_CONFIG_PATH") == "/usr/local/lib/pkgconfig"

            # Should preserve existing PATH
            assert "PATH" in env

    @pytest.mark.skipif(not COMPILATION_ENGINE_AVAILABLE, reason="CompilationEngine not implemented")
    def test_deterministic_cleanup_with_controlled_temp_files(self):
        """Test proper cleanup of temporary files with controlled temp directory."""
        engine = CompilationEngine("/tmp/compiled_plugins")

        plugin_spec = {
            "name": "cleanup_plugin",
            "source_file": "/path/to/plugin.cpp",
            "backend_type": "cpp"
        }

        temp_files_created = []
        temp_dirs_created = []

        def mock_tempfile_factory(suffix="", prefix="", dir=None):
            _ = dir  # Unused parameter for signature compatibility
            temp_file = MagicMock()
            temp_path = f"/tmp/{prefix}temp_{len(temp_files_created)}{suffix}"
            temp_file.name = temp_path
            temp_files_created.append(temp_path)
            return temp_file

        def mock_tempdir_factory(prefix="", suffix="", dir=None):
            _ = dir  # Unused parameter for signature compatibility
            temp_path = f"/tmp/{prefix}tempdir_{len(temp_dirs_created)}{suffix}"
            temp_dirs_created.append(temp_path)
            return temp_path

        with patch('tempfile.NamedTemporaryFile', side_effect=mock_tempfile_factory), \
             patch('tempfile.mkdtemp', side_effect=mock_tempdir_factory), \
             patch('subprocess.run') as mock_subprocess, \
             patch('os.path.exists') as mock_exists, \
             patch('os.remove') as mock_remove, \
             patch('shutil.rmtree') as mock_rmtree:

            mock_subprocess.return_value = MagicMock(returncode=0, stdout=b"Compilation successful")
            mock_exists.return_value = True

            result = engine.compile_plugin(plugin_spec)

            assert result["status"] == "success"

            # Verify temporary files were created and cleaned up
            assert len(temp_files_created) > 0 or len(temp_dirs_created) > 0

            # Verify cleanup was performed
            cleanup_calls = mock_remove.call_count + mock_rmtree.call_count
            total_temp_items = len(temp_files_created) + len(temp_dirs_created)

            # Should clean up at least some temporary items
            assert cleanup_calls >= min(total_temp_items, 1)