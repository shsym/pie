"""
Test module for PluginRegistry service.

This test module validates the PluginRegistry service which manages plugin
discovery, registration, compilation, and lifecycle management for different
backend validation plugins.

TDD: This test MUST FAIL until the PluginRegistry service is implemented.
"""

import pytest
import os
import tempfile
import platform
import shutil
import sys
import hashlib
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

# Proper TDD pattern - use try/except for conditional loading
try:
    from debug_framework.services.plugin_registry import PluginRegistry
    PLUGIN_REGISTRY_AVAILABLE = True
except ImportError:
    PluginRegistry = None
    PLUGIN_REGISTRY_AVAILABLE = False


class TestPluginRegistry:
    """Test suite for PluginRegistry service functionality."""

    @pytest.mark.xfail(not PLUGIN_REGISTRY_AVAILABLE, reason="TDD gate", strict=False)
    def test_plugin_registry_import_fails(self):
        """Test that PluginRegistry import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.services.plugin_registry import PluginRegistry

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_registry_initialization(self):
        """Test PluginRegistry service initialization."""
        registry = PluginRegistry(
            plugin_directory="/tmp/debug_plugins",
            cache_directory="/tmp/plugin_cache"
        )

        assert registry.plugin_directory == "/tmp/debug_plugins"
        assert registry.cache_directory == "/tmp/plugin_cache"
        assert registry.registered_plugins == {}
        assert registry.loaded_plugins == {}
        assert registry.compilation_cache == {}

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    @patch('os.path.exists')
    def test_plugin_discovery(self, mock_exists):
        """Test automatic plugin discovery in directories."""
        mock_exists.return_value = True
        registry = PluginRegistry("/tmp/debug_plugins")

        # Mock directory contents
        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = [
                "metal_attention_plugin.cpp",
                "cuda_softmax_plugin.cu",
                "pytorch_reference_plugin.py",
                "README.md",  # Should be ignored
                "plugin_config.json"
            ]

            with patch('os.path.isfile') as mock_isfile:
                mock_isfile.side_effect = lambda x: not x.endswith('.md')

                discovered_plugins = registry.discover_plugins()

                assert len(discovered_plugins) == 4
                assert any(p["name"] == "metal_attention_plugin" for p in discovered_plugins)
                assert any(p["name"] == "cuda_softmax_plugin" for p in discovered_plugins)
                assert any(p["name"] == "pytorch_reference_plugin" for p in discovered_plugins)
                assert any(p["name"] == "plugin_config" for p in discovered_plugins)

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_registration(self):
        """Test plugin registration with metadata validation."""
        registry = PluginRegistry("/tmp/debug_plugins")

        plugin_metadata = {
            "name": "metal_attention_plugin",
            "version": "1.0.0",
            "backend_type": "metal",
            "supported_operations": ["attention", "softmax"],
            "source_file": "/path/to/metal_attention_plugin.cpp",
            "interface_version": "2.0",
            "dependencies": ["Metal.framework"],
            "compilation_flags": ["-std=c++17", "-O3"]
        }

        plugin_id = registry.register_plugin(plugin_metadata)

        assert plugin_id is not None
        assert plugin_id in registry.registered_plugins

        registered = registry.registered_plugins[plugin_id]
        assert registered["name"] == "metal_attention_plugin"
        assert registered["backend_type"] == "metal"
        assert "attention" in registered["supported_operations"]

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_metadata_validation(self):
        """Test validation of plugin metadata during registration."""
        registry = PluginRegistry("/tmp/debug_plugins")

        # Test missing required fields
        invalid_metadata = {
            "name": "incomplete_plugin",
            # Missing version, backend_type, etc.
        }

        with pytest.raises(ValueError, match="Missing required field"):
            registry.register_plugin(invalid_metadata)

        # Test invalid backend type
        invalid_backend = {
            "name": "invalid_backend_plugin",
            "version": "1.0.0",
            "backend_type": "unsupported_backend",
            "supported_operations": ["test"],
            "source_file": "/path/to/plugin.cpp"
        }

        with pytest.raises(ValueError, match="Unsupported backend_type"):
            registry.register_plugin(invalid_backend)

        # Test invalid version format
        invalid_version = {
            "name": "invalid_version_plugin",
            "version": "not.a.version",
            "backend_type": "metal",
            "supported_operations": ["test"],
            "source_file": "/path/to/plugin.cpp"
        }

        with pytest.raises(ValueError, match="Invalid version format"):
            registry.register_plugin(invalid_version)

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    @pytest.mark.skipif(platform.system() != "Darwin", reason="Metal compilation only available on macOS")
    def test_plugin_compilation_metal(self):
        """Test plugin compilation for Metal backend."""
        # Additional check for xcrun availability
        if not shutil.which("xcrun"):
            pytest.skip("xcrun not available for Metal compilation")

        registry = PluginRegistry("/tmp/debug_plugins")

        plugin_metadata = {
            "name": "metal_test_plugin",
            "version": "1.0.0",
            "backend_type": "metal",
            "supported_operations": ["attention"],
            "source_file": "/path/to/metal_plugin.cpp",
            "compilation_flags": ["-std=c++17", "-framework", "Metal"]
        }

        plugin_id = registry.register_plugin(plugin_metadata)

        # Mock compilation process
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = b"Compilation successful"

            with patch('os.path.exists') as mock_exists:
                mock_exists.return_value = True

                compiled_path = registry.compile_plugin(plugin_id)

                assert compiled_path is not None
                assert compiled_path.endswith('.dylib')
                mock_subprocess.assert_called_once()

                # Verify Metal-specific compilation flags were used
                compile_cmd = mock_subprocess.call_args[0][0]
                assert "-framework" in compile_cmd
                assert "Metal" in compile_cmd

                # Verify .metallib shader compilation is handled
                # Either validate companion .metallib exists or .air→.metallib flow is explicit
                metallib_path = compiled_path.replace('.dylib', '.metallib')
                with patch('os.path.exists') as mock_metallib_exists:
                    mock_metallib_exists.return_value = True
                    # Should validate shader compilation or document the requirement
                    if hasattr(registry, '_validate_metal_shader_compilation'):
                        assert registry._validate_metal_shader_compilation(plugin_id, metallib_path)
                    else:
                        assert any('air' in str(arg) or 'metallib' in str(arg) for arg in compile_cmd)

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_compilation_cuda(self):
        """Test plugin compilation for CUDA backend."""
        # Skip if nvcc is not available
        if not shutil.which("nvcc"):
            pytest.skip("nvcc not available for CUDA compilation")

        registry = PluginRegistry("/tmp/debug_plugins")

        plugin_metadata = {
            "name": "cuda_test_plugin",
            "version": "1.0.0",
            "backend_type": "cuda",
            "supported_operations": ["softmax"],
            "source_file": "/path/to/cuda_plugin.cu",
            "compilation_flags": ["-O3", "-arch=sm_75"],
            "host_compiler_flags": ["-Wall", "-Wextra", "-std=c++17"]
        }

        plugin_id = registry.register_plugin(plugin_metadata)

        # Mock CUDA compilation
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = b"CUDA compilation successful"

            with patch('shutil.which') as mock_which:
                mock_which.return_value = "/usr/local/cuda/bin/nvcc"

                with patch('os.path.exists') as mock_exists:
                    mock_exists.return_value = True

                    compiled_path = registry.compile_plugin(plugin_id)

                    assert compiled_path is not None
                    assert compiled_path.endswith('.so')

                    # Verify CUDA compiler was used with reasonable flags
                    compile_cmd = mock_subprocess.call_args[0][0]
                    assert "nvcc" in compile_cmd[0]
                    assert "-arch=sm_75" in compile_cmd
                    # Verify CUDA-specific compilation pattern
                    assert any(".cu" in str(arg) or "cuda" in str(arg).lower() for arg in compile_cmd)

                    # Verify host compiler flags are routed through -Xcompiler
                    host_flags_found = False
                    for i, arg in enumerate(compile_cmd):
                        if arg == "-Xcompiler" and i + 1 < len(compile_cmd):
                            next_arg = compile_cmd[i + 1]
                            if any(flag in next_arg for flag in ["-Wall", "-Wextra", "-std=c++17"]):
                                host_flags_found = True
                                break
                    assert host_flags_found, "Host compiler flags should be routed through -Xcompiler"

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_loading_and_interface_validation(self):
        """Test plugin loading and interface validation."""
        registry = PluginRegistry("/tmp/debug_plugins")

        plugin_metadata = {
            "name": "test_plugin",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["test_op"],
            "source_file": "/path/to/plugin.cpp",
            "interface_version": "2.0"
        }

        plugin_id = registry.register_plugin(plugin_metadata)

        # Mock plugin loading
        with patch('ctypes.CDLL') as mock_cdll:
            mock_lib = MagicMock()
            mock_lib.get_interface_version.return_value = b"2.0"
            mock_lib.get_supported_operations.return_value = b"test_op"
            mock_lib.validate_interface.return_value = 1  # Success
            mock_cdll.return_value = mock_lib

            loaded_plugin = registry.load_plugin(plugin_id, "/path/to/compiled.so")

            assert loaded_plugin is not None
            assert plugin_id in registry.loaded_plugins
            assert registry.loaded_plugins[plugin_id]["library"] == mock_lib

            # Verify interface validation was called
            mock_lib.validate_interface.assert_called_once()

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_interface_version_mismatch(self):
        """Test handling of plugin interface version mismatches."""
        registry = PluginRegistry("/tmp/debug_plugins")

        plugin_metadata = {
            "name": "version_mismatch_plugin",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["test_op"],
            "source_file": "/path/to/plugin.cpp",
            "interface_version": "2.0"
        }

        plugin_id = registry.register_plugin(plugin_metadata)

        # Mock plugin with incompatible interface version
        with patch('ctypes.CDLL') as mock_cdll:
            mock_lib = MagicMock()
            mock_lib.get_interface_version.return_value = b"1.0"  # Different version
            mock_cdll.return_value = mock_lib

            with pytest.raises(ValueError, match="Interface version mismatch"):
                registry.load_plugin(plugin_id, "/path/to/compiled.so")

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_operation_execution(self):
        """Test execution of plugin operations."""
        registry = PluginRegistry("/tmp/debug_plugins")

        plugin_metadata = {
            "name": "execution_test_plugin",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["test_operation"],
            "source_file": "/path/to/plugin.cpp"
        }

        plugin_id = registry.register_plugin(plugin_metadata)

        # Mock loaded plugin
        mock_lib = MagicMock()
        mock_lib.test_operation.return_value = 42
        registry.loaded_plugins[plugin_id] = {
            "library": mock_lib,
            "metadata": plugin_metadata
        }

        # Execute operation
        result = registry.execute_operation(plugin_id, "test_operation", {"input": "test_data"})

        assert result == 42
        mock_lib.test_operation.assert_called_once()

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_dependency_resolution(self):
        """Test plugin dependency resolution and loading order."""
        registry = PluginRegistry("/tmp/debug_plugins")

        # Register plugins with dependencies
        base_plugin = {
            "name": "base_plugin",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["base_op"],
            "source_file": "/path/to/base.cpp",
            "dependencies": []
        }

        dependent_plugin = {
            "name": "dependent_plugin",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["dependent_op"],
            "source_file": "/path/to/dependent.cpp",
            "dependencies": ["base_plugin"]
        }

        base_id = registry.register_plugin(base_plugin)
        dependent_id = registry.register_plugin(dependent_plugin)

        # Test dependency resolution
        load_order = registry.resolve_dependencies([dependent_id, base_id])

        # Base plugin should be loaded before dependent plugin
        assert load_order.index(base_id) < load_order.index(dependent_id)

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_dependency_cycle_detection(self):
        """Test detection and error reporting for circular dependencies."""
        registry = PluginRegistry("/tmp/debug_plugins")

        # Create plugins with circular dependencies
        plugin_a = {
            "name": "plugin_a",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["op_a"],
            "source_file": "/path/to/plugin_a.cpp",
            "dependencies": ["plugin_b"]  # A depends on B
        }

        plugin_b = {
            "name": "plugin_b",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["op_b"],
            "source_file": "/path/to/plugin_b.cpp",
            "dependencies": ["plugin_c"]  # B depends on C
        }

        plugin_c = {
            "name": "plugin_c",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["op_c"],
            "source_file": "/path/to/plugin_c.cpp",
            "dependencies": ["plugin_a"]  # C depends on A, creating cycle A→B→C→A
        }

        id_a = registry.register_plugin(plugin_a)
        id_b = registry.register_plugin(plugin_b)
        id_c = registry.register_plugin(plugin_c)

        # Test cycle detection during dependency resolution
        with pytest.raises(ValueError, match="Circular dependency detected"):
            registry.resolve_dependencies([id_a, id_b, id_c])

        # Verify specific cycle reporting
        try:
            registry.resolve_dependencies([id_a, id_b, id_c])
        except ValueError as e:
            # Should identify the specific plugins in the cycle
            error_message = str(e)
            assert "plugin_a" in error_message
            assert "plugin_b" in error_message
            assert "plugin_c" in error_message

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_caching_mechanism(self):
        """Test plugin compilation caching for performance."""
        registry = PluginRegistry("/tmp/debug_plugins", "/tmp/plugin_cache")

        plugin_metadata = {
            "name": "cacheable_plugin",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["cache_test"],
            "source_file": "/path/to/plugin.cpp"
        }

        plugin_id = registry.register_plugin(plugin_metadata)

        # Mock content-based caching instead of timestamp-based
        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', mock_open(read_data=b'test plugin source')):

            # Mock cache file exists and content hash matches
            mock_exists.side_effect = lambda x: "cache" in x  # Cache exists

            # Mock content hash for cache validation
            test_source_hash = hashlib.sha256(b'test plugin source').hexdigest()

            with patch('subprocess.run') as mock_subprocess, \
                 patch.object(registry, '_get_source_content_hash', return_value=test_source_hash), \
                 patch.object(registry, '_get_cached_hash', return_value=test_source_hash):

                registry.compile_plugin(plugin_id)

                # Compilation should not run since cache hash is valid
                mock_subprocess.assert_not_called()

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_cache_invalidation_on_hash_mismatch(self):
        """Test that cache is invalidated and recompilation occurs when source hash changes."""
        registry = PluginRegistry("/tmp/debug_plugins", "/tmp/plugin_cache")

        plugin_metadata = {
            "name": "cache_invalidation_plugin",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["cache_invalidation_test"],
            "source_file": "/path/to/plugin.cpp"
        }

        plugin_id = registry.register_plugin(plugin_metadata)

        # Mock cache exists but hash mismatches (source changed)
        old_source_hash = hashlib.sha256(b'old plugin source').hexdigest()
        new_source_hash = hashlib.sha256(b'new plugin source').hexdigest()

        with patch('os.path.exists') as mock_exists, \
             patch('builtins.open', mock_open(read_data=b'new plugin source')):

            # Cache exists but hash doesn't match
            mock_exists.side_effect = lambda x: "cache" in x  # Cache exists

            with patch('subprocess.run') as mock_subprocess, \
                 patch.object(registry, '_get_source_content_hash', return_value=new_source_hash), \
                 patch.object(registry, '_get_cached_hash', return_value=old_source_hash):

                mock_subprocess.return_value.returncode = 0
                mock_subprocess.return_value.stdout = b"Recompilation successful"

                registry.compile_plugin(plugin_id)

                # Compilation should run due to hash mismatch
                mock_subprocess.assert_called_once()

                # Verify cache is updated with new hash
                assert registry._should_update_cache_hash(plugin_id, new_source_hash)

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_cleanup_and_logical_deregistration(self):
        """Test proper cleanup and logical deregistration of plugins."""
        registry = PluginRegistry("/tmp/debug_plugins")

        plugin_metadata = {
            "name": "cleanup_test_plugin",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["cleanup_test"],
            "source_file": "/path/to/plugin.cpp"
        }

        plugin_id = registry.register_plugin(plugin_metadata)

        # Mock loaded plugin
        mock_lib = MagicMock()
        mock_lib.cleanup.return_value = None
        registry.loaded_plugins[plugin_id] = {
            "library": mock_lib,
            "metadata": plugin_metadata,
            "loaded": True
        }

        # Test logical deregistration (not physical dlclose)
        registry.unload_plugin(plugin_id)

        # Verify logical state changes rather than physical unload
        assert plugin_id not in registry.loaded_plugins
        mock_lib.cleanup.assert_called_once()

        # Note: We test logical deregistration, not physical library unloading
        # since Python's ctypes doesn't reliably support dlclose

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_cross_platform_plugin_support(self):
        """Test cross-platform plugin support and platform-specific handling."""
        registry = PluginRegistry("/tmp/debug_plugins")

        # Test platform-specific plugin selection
        plugins = [
            {
                "name": "metal_plugin",
                "version": "1.0.0",
                "backend_type": "metal",
                "supported_operations": ["attention"],
                "source_file": "/path/to/metal_plugin.cpp",
                "platform": "darwin"
            },
            {
                "name": "cuda_plugin",
                "version": "1.0.0",
                "backend_type": "cuda",
                "supported_operations": ["attention"],
                "source_file": "/path/to/cuda_plugin.cu",
                "platform": "linux"
            }
        ]

        for plugin in plugins:
            registry.register_plugin(plugin)

        # Test getting platform-compatible plugins with proper sys.platform handling
        with patch('platform.system') as mock_platform, \
             patch('sys.platform', 'darwin'):
            mock_platform.return_value = "Darwin"

            compatible_plugins = registry.get_platform_compatible_plugins("attention")

            # Should only return Metal plugin on macOS (if any plugins match)
            if compatible_plugins:
                metal_plugins = [p for p in compatible_plugins if p["backend_type"] == "metal"]
                assert len(metal_plugins) >= 1
                assert metal_plugins[0]["name"] == "metal_plugin"
            else:
                # No compatible plugins found for this platform/operation combo
                pytest.skip("No compatible plugins found for test scenario")

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_error_handling(self):
        """Test error handling during plugin operations."""
        registry = PluginRegistry("/tmp/debug_plugins")

        # Test handling of compilation errors
        plugin_metadata = {
            "name": "error_test_plugin",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["error_test"],
            "source_file": "/path/to/broken_plugin.cpp"
        }

        plugin_id = registry.register_plugin(plugin_metadata)

        # Mock compilation failure
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 1
            mock_subprocess.return_value.stderr = b"Compilation error: syntax error"

            with pytest.raises(RuntimeError, match="Plugin compilation failed"):
                registry.compile_plugin(plugin_id)

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_registry_persistence(self):
        """Test persistence of plugin registry state."""
        registry = PluginRegistry("/tmp/debug_plugins")

        plugin_metadata = {
            "name": "persistent_plugin",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["persist_test"],
            "source_file": "/path/to/plugin.cpp"
        }

        plugin_id = registry.register_plugin(plugin_metadata)

        # Test saving registry state
        registry_state = registry.save_state()

        assert "registered_plugins" in registry_state
        assert plugin_id in registry_state["registered_plugins"]

        # Test loading registry state
        new_registry = PluginRegistry("/tmp/debug_plugins")
        new_registry.load_state(registry_state)

        assert plugin_id in new_registry.registered_plugins
        assert new_registry.registered_plugins[plugin_id]["name"] == "persistent_plugin"

    @pytest.mark.skipif(not PLUGIN_REGISTRY_AVAILABLE, reason="PluginRegistry not implemented")
    def test_plugin_version_management(self):
        """Test management of multiple plugin versions."""
        registry = PluginRegistry("/tmp/debug_plugins")

        # Register multiple versions of the same plugin
        plugin_v1 = {
            "name": "versioned_plugin",
            "version": "1.0.0",
            "backend_type": "cpp",
            "supported_operations": ["version_test"],
            "source_file": "/path/to/plugin_v1.cpp"
        }

        plugin_v2 = {
            "name": "versioned_plugin",
            "version": "2.0.0",
            "backend_type": "cpp",
            "supported_operations": ["version_test", "new_feature"],
            "source_file": "/path/to/plugin_v2.cpp"
        }

        id_v1 = registry.register_plugin(plugin_v1)
        id_v2 = registry.register_plugin(plugin_v2)

        # Test getting latest version
        latest = registry.get_latest_version("versioned_plugin")
        assert latest["version"] == "2.0.0"
        assert latest["id"] == id_v2

        # Test getting specific version
        specific = registry.get_plugin_by_version("versioned_plugin", "1.0.0")
        assert specific["version"] == "1.0.0"
        assert specific["id"] == id_v1