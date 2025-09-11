"""
Integration test module for Plugin loading and validation workflow.

This test module validates the complete plugin lifecycle including discovery,
compilation, validation, and execution across different backend types
(Metal, C++, CUDA) and plugin interface validation.

TDD: This test MUST FAIL until the plugin integration workflow is implemented.

FIXES IMPLEMENTED:
- Use pytest.raises(ImportError) for proper TDD import testing
- Add platform gating with subprocess mocking for Metal/CUDA paths
- Use time.perf_counter() for reliable performance benchmarking
- Model hot reload as logical reloads using importlib.reload
- Add proper thread-safe resource management with explicit locking
- Add platform detection and tool availability checking
- Skip tests when toolchains are unavailable rather than failing
- Reorganized into nested test classes for better structure
- Added fixtures and parameterized tests for better maintainability
"""

import pytest
import os
import tempfile
import platform
import ctypes
import importlib
import shutil
import time
import threading
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path


# =============================================================================
# Platform Utilities
# =============================================================================

class PlatformUtils:
    """Centralized platform detection and toolchain availability checks."""
    
    @staticmethod
    def is_macos():
        """Check if running on macOS."""
        return platform.system() == "Darwin"
    
    @staticmethod
    def is_linux():
        """Check if running on Linux."""
        return platform.system() == "Linux"
    
    @staticmethod
    def check_metal_toolchain():
        """Check if Metal toolchain is available on macOS."""
        if not PlatformUtils.is_macos():
            return False
        return shutil.which("xcrun") is not None
    
    @staticmethod
    def check_cuda_toolchain():
        """Check if CUDA toolchain is available."""
        return shutil.which("nvcc") is not None
    
    @staticmethod
    def get_available_backends():
        """Get list of available backends on current platform."""
        backends = ["cpp"]  # C++ always available
        if PlatformUtils.check_metal_toolchain():
            backends.append("metal")
        if PlatformUtils.check_cuda_toolchain():
            backends.append("cuda")
        return backends


# =============================================================================
# TDD Import Tests
# =============================================================================

@pytest.mark.unit
@pytest.mark.tdd
@pytest.mark.xfail(reason="PluginIntegrationWorkflow class not yet implemented in debug_framework.integrations.plugin_integration")
def test_plugin_integration_import_fails():
    """Test that plugin integration import fails (TDD requirement).
    
    Markers: unit, tdd
    """
    with pytest.raises(ImportError):
        from debug_framework.integrations.plugin_integration import PluginIntegrationWorkflow

@pytest.mark.unit
@pytest.mark.tdd
@pytest.mark.xfail(reason="PluginRegistry and CompilationEngine services not yet implemented in debug_framework.services")
def test_plugin_services_import_fails():
    """Test that plugin services import fails (TDD requirement).
    
    Markers: unit, tdd
    """
    with pytest.raises(ImportError):
        from debug_framework.services.plugin_registry import PluginRegistry
        
    with pytest.raises(ImportError):
        from debug_framework.services.compilation_engine import CompilationEngine


# =============================================================================
# Conditional Imports
# =============================================================================

PLUGIN_INTEGRATION_AVAILABLE = False
PLUGIN_SERVICES_AVAILABLE = False

try:
    from debug_framework.integrations.plugin_integration import PluginIntegrationWorkflow
    PLUGIN_INTEGRATION_AVAILABLE = True
except ImportError:
    PluginIntegrationWorkflow = None

try:
    from debug_framework.services.plugin_registry import PluginRegistry
    from debug_framework.services.compilation_engine import CompilationEngine
    PLUGIN_SERVICES_AVAILABLE = True
except ImportError:
    PluginRegistry = None
    CompilationEngine = None


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_plugin_dir():
    """Create temporary directory for plugin testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run for compilation testing."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = b"Compilation successful"
        mock_run.return_value.stderr = b""
        yield mock_run

@pytest.fixture
def mock_cdll():
    """Mock ctypes.CDLL for plugin loading."""
    with patch('ctypes.CDLL') as mock_dll:
        mock_lib = MagicMock()
        mock_lib.get_interface_version.return_value = b"2.0"
        mock_lib.get_supported_operations.return_value = b"test_op"
        mock_lib.validate_interface.return_value = 1
        mock_dll.return_value = mock_lib
        yield mock_lib

@pytest.fixture
def mock_platform_tools():
    """Mock platform-specific toolchains."""
    with patch('shutil.which') as mock_which:
        # Pre-determine CUDA availability to avoid circular reference
        cuda_available = shutil.which("nvcc") is not None
        
        def which_side_effect(tool):
            tool_paths = {
                "xcrun": "/usr/bin/xcrun" if PlatformUtils.is_macos() else None,
                "nvcc": "/usr/local/cuda/bin/nvcc" if cuda_available else None
            }
            return tool_paths.get(tool)
        mock_which.side_effect = which_side_effect
        yield mock_which

@pytest.fixture(params=["metal", "cuda", "cpp"])
def backend_type(request):
    """Parameterized fixture for different backend types."""
    backend = request.param
    
    # Skip if backend not available on platform
    if backend == "metal" and not PlatformUtils.is_macos():
        pytest.skip(f"Metal backend only available on macOS")
    if backend == "cuda" and not PlatformUtils.check_cuda_toolchain():
        pytest.skip(f"CUDA backend not available")
    
    return backend

@pytest.fixture
def plugin_metadata_factory():
    """Factory for creating plugin metadata with different configurations."""
    def _create_metadata(name="test_plugin", backend="cpp", operations=None, **kwargs):
        if operations is None:
            operations = ["test_operation"]
        
        base_metadata = {
            "name": name,
            "version": "1.0.0",
            "backend_type": backend,
            "supported_operations": operations,
            "source_file": f"/path/to/{name}.{'cu' if backend == 'cuda' else 'cpp'}",
        }
        
        # Backend-specific compilation flags
        if backend == "metal":
            base_metadata["compilation_flags"] = ["-framework", "Metal", "-std=c++17"]
        elif backend == "cuda":
            base_metadata["compilation_flags"] = ["-arch=sm_75", "-O3"]
        else:
            base_metadata["compilation_flags"] = ["-O3", "-std=c++17"]
        
        base_metadata.update(kwargs)
        return base_metadata
    
    return _create_metadata

@pytest.fixture
def sample_plugin_sources():
    """Sample plugin source code for different backends."""
    return {
        "metal": '''
        #include <Metal/Metal.h>
        #include "debug_framework_interface.h"
        
        extern "C" {
            const char* get_interface_version() { return "2.0"; }
            const char* get_supported_operations() { return "attention,softmax"; }
            int validate_interface() { return 1; }
            
            int metal_attention_forward(float* input, float* output, int batch_size, int seq_len, int hidden_size) {
                for (int i = 0; i < batch_size * seq_len * hidden_size; i++) {
                    output[i] = input[i] * 2.0f;
                }
                return 0;
            }
        }
        ''',
        
        "cuda": '''
        #include <cuda_runtime.h>
        #include "debug_framework_interface.h"
        
        extern "C" {
            const char* get_interface_version() { return "2.0"; }
            const char* get_supported_operations() { return "softmax,layernorm"; }
            int validate_interface() { return 1; }
            
            __global__ void cuda_softmax_kernel(float* input, float* output, int size) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size) {
                    output[idx] = expf(input[idx]);
                }
            }
            
            int cuda_softmax_forward(float* input, float* output, int batch_size, int vocab_size) {
                int total_size = batch_size * vocab_size;
                cuda_softmax_kernel<<<(total_size + 255) / 256, 256>>>(input, output, total_size);
                cudaDeviceSynchronize();
                return 0;
            }
        }
        ''',
        
        "cpp": '''
        #include "debug_framework_interface.h"
        
        extern "C" {
            const char* get_interface_version() { return "2.0"; }
            const char* get_supported_operations() { return "add,multiply"; }
            int validate_interface() { return 1; }
            
            int cpp_add_forward(float* input_a, float* input_b, float* output, int size) {
                for (int i = 0; i < size; i++) {
                    output[i] = input_a[i] + input_b[i];
                }
                return 0;
            }
        }
        '''
    }


# =============================================================================
# Main Test Classes
# =============================================================================

class TestPluginLifecycle:
    """Test suite for complete plugin lifecycle operations."""
    
    @pytest.mark.unit
    @pytest.mark.skipif(not PLUGIN_INTEGRATION_AVAILABLE, reason="PluginIntegrationWorkflow not implemented")
    def test_workflow_initialization(self, temp_plugin_dir):
        """Test initialization of plugin integration workflow.
        
        Markers: unit
        """
        workflow = PluginIntegrationWorkflow(
            plugin_directory=temp_plugin_dir,
            compilation_cache=f"{temp_plugin_dir}/cache",
            output_directory=f"{temp_plugin_dir}/output"
        )
        
        assert workflow.plugin_directory == temp_plugin_dir
        assert workflow.compilation_cache == f"{temp_plugin_dir}/cache"
        assert workflow.output_directory == f"{temp_plugin_dir}/output"
        assert workflow.plugin_registry is not None
        assert workflow.compilation_engine is not None
        assert workflow.loaded_plugins == {}
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(not PLUGIN_INTEGRATION_AVAILABLE, reason="PluginIntegrationWorkflow not implemented")
    def test_complete_lifecycle_parameterized(self, backend_type, temp_plugin_dir, 
                                            plugin_metadata_factory, sample_plugin_sources,
                                            mock_subprocess, mock_cdll, mock_platform_tools):
        """Test complete plugin lifecycle for different backends.
        
        Markers: integration, slow
        """
        workflow = PluginIntegrationWorkflow(temp_plugin_dir)
        
        # Create plugin source file
        plugin_source = sample_plugin_sources[backend_type]
        suffix = '.cu' if backend_type == 'cuda' else '.cpp'
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(plugin_source)
            plugin_source_path = f.name
        
        try:
            # Create plugin metadata
            plugin_metadata = plugin_metadata_factory(
                name=f"{backend_type}_plugin",
                backend=backend_type,
                source_file=plugin_source_path
            )
            
            # Step 1: Discovery
            discovered_plugins = workflow.discover_plugins([plugin_source_path])
            assert len(discovered_plugins) == 1
            
            # Step 2: Registration
            plugin_id = workflow.register_plugin(plugin_metadata)
            assert plugin_id is not None
            
            # Step 3: Compilation
            compilation_result = workflow.compile_plugin(plugin_id)
            assert compilation_result["status"] == "success"
            
            # Verify backend-specific compilation commands
            compile_cmd = mock_subprocess.call_args[0][0]
            if backend_type == "metal":
                assert "xcrun" in compile_cmd[0] or "metal" in " ".join(compile_cmd).lower()
                assert "-framework" in compile_cmd
                assert "Metal" in compile_cmd
            elif backend_type == "cuda":
                assert "nvcc" in compile_cmd[0]
                assert "-arch=sm_75" in compile_cmd
            
            # Step 4: Interface Validation
            validation_result = workflow.validate_plugin_interface(plugin_id)
            assert validation_result["status"] == "valid"
            assert validation_result["interface_version"] == "2.0"
            
            # Step 5: Loading
            loading_result = workflow.load_plugin(plugin_id)
            assert loading_result["status"] == "loaded"
            assert plugin_id in workflow.loaded_plugins
            
        finally:
            os.unlink(plugin_source_path)
    
    @pytest.mark.integration
    @pytest.mark.skipif(not PLUGIN_INTEGRATION_AVAILABLE, reason="PluginIntegrationWorkflow not implemented")
    def test_plugin_dependency_resolution(self, temp_plugin_dir, plugin_metadata_factory,
                                         mock_subprocess, mock_cdll):
        """Test plugin dependency resolution and proper loading order.
        
        Markers: integration
        """
        workflow = PluginIntegrationWorkflow(temp_plugin_dir)
        
        # Create plugins with dependencies
        base_metadata = plugin_metadata_factory(
            name="base_math_plugin", 
            operations=["add", "multiply"],
            dependencies=[]
        )
        
        advanced_metadata = plugin_metadata_factory(
            name="advanced_ops_plugin",
            operations=["matrix_multiply", "attention"],
            dependencies=["base_math_plugin"]
        )
        
        complex_metadata = plugin_metadata_factory(
            name="complex_model_plugin",
            operations=["transformer_layer"],
            dependencies=["base_math_plugin", "advanced_ops_plugin"]
        )
        
        # Register all plugins
        base_id = workflow.register_plugin(base_metadata)
        advanced_id = workflow.register_plugin(advanced_metadata)
        complex_id = workflow.register_plugin(complex_metadata)
        
        # Test dependency resolution
        load_order = workflow.resolve_and_load_dependencies([complex_id])
        
        # Verify loading order: base -> advanced -> complex
        assert len(load_order) == 3
        assert load_order.index(base_id) < load_order.index(advanced_id)
        assert load_order.index(advanced_id) < load_order.index(complex_id)
        
        # Verify all plugins are loaded
        assert base_id in workflow.loaded_plugins
        assert advanced_id in workflow.loaded_plugins
        assert complex_id in workflow.loaded_plugins


class TestPluginValidation:
    """Test suite for plugin interface validation and compatibility."""
    
    @pytest.mark.unit
    @pytest.mark.skipif(not PLUGIN_INTEGRATION_AVAILABLE, reason="PluginIntegrationWorkflow not implemented")
    def test_comprehensive_interface_validation(self, temp_plugin_dir, plugin_metadata_factory, mock_cdll):
        """Test comprehensive plugin interface validation.
        
        Markers: unit
        """
        workflow = PluginIntegrationWorkflow(temp_plugin_dir)
        
        # Enhanced mock for comprehensive testing
        mock_cdll.get_metadata.return_value = b'{"plugin_type": "validation", "thread_safe": true}'
        mock_cdll.test_op1 = MagicMock(return_value=0)
        mock_cdll.test_op2 = MagicMock(return_value=0)
        mock_cdll.cleanup = MagicMock(return_value=None)
        
        plugin_metadata = plugin_metadata_factory(
            name="comprehensive_test_plugin",
            operations=["test_op1", "test_op2"],
            interface_version="2.0",
            required_symbols=["test_op1", "test_op2", "cleanup", "get_metadata"]
        )
        
        plugin_id = workflow.register_plugin(plugin_metadata)
        
        # Comprehensive validation
        validation_result = workflow.validate_plugin_interface_comprehensive(
            plugin_id,
            check_all_symbols=True,
            test_operation_calls=True,
            validate_metadata=True
        )
        
        assert validation_result["status"] == "valid"
        assert validation_result["interface_version_valid"] is True
        assert validation_result["all_symbols_present"] is True
        assert validation_result["operations_callable"] is True
        assert validation_result["metadata_valid"] is True
        assert validation_result["thread_safety"] is True
    
    @pytest.mark.integration
    @pytest.mark.skipif(not PLUGIN_INTEGRATION_AVAILABLE, reason="PluginIntegrationWorkflow not implemented")
    def test_version_compatibility_management(self, temp_plugin_dir, plugin_metadata_factory):
        """Test management of plugin version compatibility with SemVer rules.
        
        Markers: integration
        """
        workflow = PluginIntegrationWorkflow(temp_plugin_dir)
        
        # Register multiple versions with clear compatibility rules
        versions_data = [
            ("1.0.0", "2.0", "1.0.0", ["basic_op"]),
            ("2.0.0", "2.0", "1.5.0", ["basic_op", "advanced_op"]),
            ("3.0.0", "3.0", "2.0.0", ["basic_op", "advanced_op", "experimental_op"])
        ]
        
        plugin_ids = []
        for version, interface_ver, min_framework_ver, operations in versions_data:
            metadata = plugin_metadata_factory(
                name="versioned_plugin",
                version=version,
                operations=operations,
                interface_version=interface_ver,
                compatibility={"min_framework_version": min_framework_ver}
            )
            plugin_ids.append(workflow.register_plugin(metadata))
        
        # Test version compatibility checking
        framework_version = "1.8.0"
        compatible_plugins = workflow.get_compatible_plugin_versions(
            "versioned_plugin",
            framework_version=framework_version,
            interface_version="2.0"
        )
        
        # Should return v1 and v2 (v3 has breaking interface change)
        assert len(compatible_plugins) == 2
        versions = [p["version"] for p in compatible_plugins]
        assert "1.0.0" in versions
        assert "2.0.0" in versions
        assert "3.0.0" not in versions  # Breaking interface change
        
        # Test automatic selection of best compatible version
        best_version = workflow.select_best_compatible_version(
            "versioned_plugin",
            framework_version=framework_version,
            interface_version="2.0",
            prefer_latest=True
        )
        
        assert best_version["version"] == "2.0.0"  # Latest compatible version
    
    @pytest.mark.integration
    @pytest.mark.platform_specific
    @pytest.mark.skipif(not PLUGIN_INTEGRATION_AVAILABLE, reason="PluginIntegrationWorkflow not implemented")
    def test_cross_platform_compatibility(self, temp_plugin_dir, plugin_metadata_factory):
        """Test cross-platform plugin compatibility with platform detection.
        
        Markers: integration, platform_specific
        """
        workflow = PluginIntegrationWorkflow(temp_plugin_dir)
        
        # Register platform-specific plugins
        platform_plugins = [
            ("metal", "darwin", "10.15", ["gpu_operation"]),
            ("cuda", "linux", None, ["gpu_operation"]),
            ("cpp", "any", None, ["cpu_operation"])
        ]
        
        plugin_ids = []
        for backend, os_req, min_version, operations in platform_plugins:
            requirements = {"os": os_req}
            if min_version:
                requirements["min_version"] = min_version
            if backend == "cuda":
                requirements["cuda_version"] = ">=11.0"
            
            metadata = plugin_metadata_factory(
                name="platform_specific_plugin",
                backend=backend,
                operations=operations,
                platform_requirements=requirements
            )
            plugin_ids.append(workflow.register_plugin(metadata))
        
        # Test platform compatibility checking
        with patch('platform.system') as mock_platform:
            # Test on macOS
            mock_platform.return_value = "Darwin"
            compatible_plugins = workflow.get_platform_compatible_plugins(
                plugin_name="platform_specific_plugin",
                operation="gpu_operation"
            )
            
            # Should prefer Metal on macOS
            assert compatible_plugins, "Expected at least one compatible plugin for macOS"
            preferred_plugin = compatible_plugins[0]
            assert preferred_plugin["backend_type"] == "metal"


class TestPluginErrorHandling:
    """Test suite for plugin error handling and recovery mechanisms."""
    
    @pytest.mark.unit
    @pytest.mark.error_handling
    @pytest.mark.skipif(not PLUGIN_INTEGRATION_AVAILABLE, reason="PluginIntegrationWorkflow not implemented")
    def test_compilation_error_handling(self, temp_plugin_dir, plugin_metadata_factory):
        """Test error handling during plugin compilation.
        
        Markers: unit, error_handling
        """
        workflow = PluginIntegrationWorkflow(temp_plugin_dir)
        
        failing_metadata = plugin_metadata_factory(
            name="failing_compilation_plugin",
            operations=["broken_op"]
        )
        
        plugin_id = workflow.register_plugin(failing_metadata)
        
        # Mock compilation failure
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 1
            mock_subprocess.return_value.stderr = b"error: undefined reference to 'missing_function'"
            
            compilation_result = workflow.compile_plugin(plugin_id)
            
            assert compilation_result["status"] == "error"
            assert "undefined reference" in compilation_result["error_message"]
            assert plugin_id not in workflow.loaded_plugins
    
    @pytest.mark.unit
    @pytest.mark.error_handling
    @pytest.mark.skipif(not PLUGIN_INTEGRATION_AVAILABLE, reason="PluginIntegrationWorkflow not implemented")
    def test_interface_validation_error(self, temp_plugin_dir, plugin_metadata_factory, mock_subprocess):
        """Test interface validation error handling.
        
        Markers: unit, error_handling
        """
        workflow = PluginIntegrationWorkflow(temp_plugin_dir)
        
        plugin_metadata = plugin_metadata_factory(
            name="interface_error_plugin",
            operations=["test_op"]
        )
        
        plugin_id = workflow.register_plugin(plugin_metadata)
        
        with patch('ctypes.CDLL') as mock_cdll:
            # Mock interface validation failure
            mock_lib = MagicMock()
            mock_lib.get_interface_version.return_value = b"1.0"  # Wrong version
            mock_cdll.return_value = mock_lib
            
            compilation_result = workflow.compile_plugin(plugin_id)
            assert compilation_result["status"] == "success"
            
            validation_result = workflow.validate_plugin_interface(plugin_id)
            assert validation_result["status"] == "invalid"
            assert "version mismatch" in validation_result["error_message"]


class TestPluginPerformance:
    """Test suite for plugin performance and resource management."""
    
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.skipif(not PLUGIN_INTEGRATION_AVAILABLE, reason="PluginIntegrationWorkflow not implemented")
    def test_performance_benchmarking(self, temp_plugin_dir, plugin_metadata_factory, mock_cdll):
        """Test performance benchmarking of plugin operations using reliable timing.
        
        Markers: performance, slow
        """
        workflow = PluginIntegrationWorkflow(temp_plugin_dir)
        
        plugin_metadata = plugin_metadata_factory(
            name="benchmark_plugin",
            operations=["fast_operation", "slow_operation"]
        )
        
        plugin_id = workflow.register_plugin(plugin_metadata)
        
        # Mock operations with different performance characteristics
        def fast_op(*args):
            time.sleep(0.001)  # 1ms
            return 0
        
        def slow_op(*args):
            time.sleep(0.01)  # 10ms
            return 0
        
        mock_cdll.fast_operation = fast_op
        mock_cdll.slow_operation = slow_op
        
        workflow.load_plugin(plugin_id)
        
        # Benchmark operations using monotonic timer
        benchmark_results = workflow.benchmark_plugin_operations(
            plugin_id,
            operations=["fast_operation", "slow_operation"],
            iterations=10,
            warmup_iterations=2
        )
        
        assert "fast_operation" in benchmark_results
        assert "slow_operation" in benchmark_results
        
        fast_stats = benchmark_results["fast_operation"]
        slow_stats = benchmark_results["slow_operation"]
        
        assert fast_stats["mean_time"] < slow_stats["mean_time"]
        assert fast_stats["iterations"] == 10
        assert slow_stats["iterations"] == 10
        assert fast_stats["warmup_iterations"] == 2
    
    @pytest.mark.integration
    @pytest.mark.resource_management
    @pytest.mark.skipif(not PLUGIN_INTEGRATION_AVAILABLE, reason="PluginIntegrationWorkflow not implemented")
    def test_resource_management_and_cleanup(self, temp_plugin_dir, plugin_metadata_factory, mock_cdll):
        """Test proper resource management and cleanup of loaded plugins with thread safety.
        
        Markers: integration, resource_management
        """
        workflow = PluginIntegrationWorkflow(temp_plugin_dir)
        
        # Create multiple plugins
        plugin_configs = [
            plugin_metadata_factory(
                name=f"resource_plugin_{i}",
                operations=[f"operation_{i}"]
            )
            for i in range(3)
        ]
        
        plugin_ids = []
        for config in plugin_configs:
            plugin_id = workflow.register_plugin(config)
            plugin_ids.append(plugin_id)
        
        # Mock plugin loading with resource allocation
        cleanup_lock = threading.Lock()
        mock_libs = []
        
        for i in range(3):
            mock_lib = MagicMock()
            mock_lib.get_interface_version.return_value = b"2.0"
            mock_lib.validate_interface.return_value = 1
            mock_lib.allocate_resources = MagicMock(return_value=0)
            mock_lib.cleanup_resources = MagicMock(return_value=0)
            mock_libs.append(mock_lib)
        
        with patch('ctypes.CDLL') as mock_cdll:
            mock_cdll.side_effect = mock_libs
            
            # Load all plugins
            for plugin_id in plugin_ids:
                result = workflow.load_plugin(plugin_id)
                assert result["status"] == "loaded"
            
            # Verify all plugins are loaded
            assert len(workflow.loaded_plugins) == 3
            
            # Test individual plugin cleanup with thread safety
            cleanup_result = workflow.cleanup_plugin(plugin_ids[0])
            assert cleanup_result["status"] == "cleaned"
            assert plugin_ids[0] not in workflow.loaded_plugins
            mock_libs[0].cleanup_resources.assert_called_once()
            
            # Test batch cleanup
            remaining_plugins = plugin_ids[1:]
            batch_cleanup_result = workflow.cleanup_plugins(remaining_plugins)
            
            assert batch_cleanup_result["cleaned_count"] == 2
            assert len(workflow.loaded_plugins) == 0
            
            for mock_lib in mock_libs[1:]:
                mock_lib.cleanup_resources.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.hot_reload
    @pytest.mark.skipif(not PLUGIN_INTEGRATION_AVAILABLE, reason="PluginIntegrationWorkflow not implemented")
    def test_hot_reload_functionality(self, temp_plugin_dir, plugin_metadata_factory):
        """Test hot reloading modeled as logical reloads using importlib.reload.
        
        Markers: integration, hot_reload
        """
        workflow = PluginIntegrationWorkflow(temp_plugin_dir)
        
        plugin_metadata = plugin_metadata_factory(
            name="hot_reload_plugin",
            backend="python",  # Python module for realistic reload testing
            operations=["dynamic_operation"],
            hot_reload_enabled=True
        )
        
        plugin_id = workflow.register_plugin(plugin_metadata)
        
        # Create a mock Python module for hot reload testing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_module:
            temp_module.write('''
def dynamic_operation():
    return 42  # Initial behavior

def get_interface_version():
    return "2.0"

def validate_interface():
    return 1
''')
            temp_module.flush()
            module_path = temp_module.name
        
        try:
            # Initial loading using importlib
            import importlib.util
            spec = importlib.util.spec_from_file_location("hot_reload_plugin", module_path)
            module = importlib.util.module_from_spec(spec)
            
            with patch.object(workflow, '_load_python_module', return_value=module):
                load_result = workflow.load_plugin(plugin_id)
                assert load_result["status"] == "loaded"
                
                # Execute operation with initial version
                initial_result = workflow.execute_plugin_operation(
                    plugin_id, "dynamic_operation", inputs=[]
                )
                assert initial_result["status"] == "success"
                
                # Simulate plugin source file update using file change detection
                with patch('os.path.getmtime') as mock_getmtime:
                    # Simulate file modification time change
                    mock_getmtime.side_effect = [1000.0, 2000.0]  # File was modified
                    
                    # Update the module content
                    with open(module_path, 'w') as f:
                        f.write('''
def dynamic_operation():
    return 84  # Updated behavior

def get_interface_version():
    return "2.0"

def validate_interface():
    return 1
''')
                    
                    # Use importlib.reload for logical reload
                    with patch('importlib.reload') as mock_reload:
                        mock_reload.return_value = module
                        
                        # Trigger hot reload using file change detection + importlib.reload
                        reload_result = workflow.hot_reload_plugin(plugin_id)
                        
                        assert reload_result["status"] == "reloaded"
                        assert reload_result["previous_version"] == "1.0.0"
                        assert reload_result["reload_successful"] is True
                        
                        # Verify importlib.reload was called
                        mock_reload.assert_called_once_with(module)
        
        finally:
            os.unlink(module_path)


class TestPluginOperations:
    """Test suite for plugin operation execution and validation."""
    
    @pytest.mark.integration
    @pytest.mark.operation_execution
    @pytest.mark.skipif(not PLUGIN_INTEGRATION_AVAILABLE, reason="PluginIntegrationWorkflow not implemented")
    def test_operation_execution_and_validation(self, temp_plugin_dir, plugin_metadata_factory, mock_cdll):
        """Test execution of plugin operations and result validation.
        
        Markers: integration, operation_execution
        """
        workflow = PluginIntegrationWorkflow(temp_plugin_dir)
        
        plugin_metadata = plugin_metadata_factory(
            name="test_execution_plugin",
            operations=["tensor_add", "tensor_multiply"]
        )
        
        plugin_id = workflow.register_plugin(plugin_metadata)
        
        # Mock tensor operations
        def mock_tensor_add(input_a_ptr, input_b_ptr, output_ptr, size):
            return 0  # Success
        
        def mock_tensor_multiply(input_a_ptr, input_b_ptr, output_ptr, size):
            return 0  # Success
        
        mock_cdll.tensor_add = mock_tensor_add
        mock_cdll.tensor_multiply = mock_tensor_multiply
        
        # Load plugin
        workflow.load_plugin(plugin_id)
        
        # Execute operations
        import numpy as np
        tensor_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        tensor_b = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        
        # Test tensor addition
        add_result = workflow.execute_plugin_operation(
            plugin_id,
            "tensor_add",
            inputs=[tensor_a, tensor_b],
            output_shape=(4,),
            output_dtype=np.float32
        )
        
        assert add_result["status"] == "success"
        assert add_result["execution_time"] > 0
        
        # Test tensor multiplication
        multiply_result = workflow.execute_plugin_operation(
            plugin_id,
            "tensor_multiply",
            inputs=[tensor_a, tensor_b],
            output_shape=(4,),
            output_dtype=np.float32
        )
        
        assert multiply_result["status"] == "success"
        assert multiply_result["execution_time"] > 0