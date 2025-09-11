"""
Test module for PluginDefinition model.

This test module validates the PluginDefinition data model which manages
configuration and metadata for alternative backend implementations that can be
plugged into the validation framework.

TDD: This test MUST FAIL until the PluginDefinition model is implemented.
"""

import pytest
import json
import os
import tempfile
from datetime import datetime
from unittest.mock import patch, MagicMock

# Proper TDD pattern - use try/except for conditional loading
try:
    from debug_framework.models.plugin_definition import PluginDefinition
    PLUGINDEFINITION_AVAILABLE = True
except ImportError:
    PluginDefinition = None
    PLUGINDEFINITION_AVAILABLE = False


class TestPluginDefinition:
    """Test suite for PluginDefinition model functionality."""

    def test_plugin_definition_import_fails(self):
        """Test that PluginDefinition import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.models.plugin_definition import PluginDefinition

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_plugin_definition_creation(self):
        """Test basic PluginDefinition object creation."""
        source_files = ["src/metal_attention.metal", "src/metal_binding.cpp"]
        compile_config = {
            "target_platform": "metal",
            "compiler_flags": ["-std=c++17", "-O2"],
            "dependencies": ["Metal.framework"],
            "output_path": "lib/metal_attention.dylib"
        }
        interface_config = {
            "checkpoint_mapping": {
                "post_attention": "forward_attention",
                "post_mlp": "forward_mlp"
            },
            "validation": {
                "extract_signatures": True,
                "runtime_type_check": True
            }
        }
        
        plugin = PluginDefinition(
            name="metal_attention",
            version="1.0.0",
            target_platform="metal",
            source_files=source_files,
            compile_config=compile_config,
            interface_config=interface_config
        )
        
        assert plugin.name == "metal_attention"
        assert plugin.version == "1.0.0"
        assert plugin.target_platform == "metal"
        assert plugin.source_files == source_files
        assert plugin.compile_config == compile_config
        assert plugin.interface_config == interface_config
        assert plugin.is_compiled is False
        assert plugin.created_at is not None

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_name_validation(self):
        """Test plugin name validation rules."""
        valid_names = ["metal_attention", "cpp_mlp", "cuda_softmax", "attention_v2"]
        invalid_names = ["metal-attention", "metal attention", "Metal@Attention", "123invalid"]
        
        source_files = ["src/plugin.cpp"]
        compile_config = {"target_platform": "cpp"}
        interface_config = {"checkpoint_mapping": {}}
        
        # Test valid names
        for name in valid_names:
            plugin = PluginDefinition(
                name=name,
                version="1.0.0",
                target_platform="cpp",
                source_files=source_files,
                compile_config=compile_config,
                interface_config=interface_config
            )
            assert plugin.name == name

        # Test invalid names
        for name in invalid_names:
            with pytest.raises(ValueError, match="name must follow naming convention"):
                PluginDefinition(
                    name=name,
                    version="1.0.0",
                    target_platform="cpp",
                    source_files=source_files,
                    compile_config=compile_config,
                    interface_config=interface_config
                )

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_version_validation(self):
        """Test semantic versioning validation."""
        valid_versions = ["1.0.0", "0.1.0", "10.5.2", "1.0.0-alpha", "2.1.0-beta.1"]
        invalid_versions = ["1.0", "v1.0.0", "1.0.0.0", "1.x.0", "invalid"]
        
        source_files = ["src/plugin.cpp"]
        compile_config = {"target_platform": "cpp"}
        interface_config = {"checkpoint_mapping": {}}
        
        # Test valid versions
        for version in valid_versions:
            plugin = PluginDefinition(
                name="test_plugin",
                version=version,
                target_platform="cpp",
                source_files=source_files,
                compile_config=compile_config,
                interface_config=interface_config
            )
            assert plugin.version == version

        # Test invalid versions
        for version in invalid_versions:
            with pytest.raises(ValueError, match="version must follow semantic versioning"):
                PluginDefinition(
                    name="test_plugin",
                    version=version,
                    target_platform="cpp",
                    source_files=source_files,
                    compile_config=compile_config,
                    interface_config=interface_config
                )

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_target_platform_validation(self):
        """Test target platform validation."""
        valid_platforms = ["metal", "cpp", "objc", "cuda"]
        
        source_files = ["src/plugin.cpp"]
        compile_config = {"target_platform": "cpp"}
        interface_config = {"checkpoint_mapping": {}}
        
        # Test valid platforms
        for platform in valid_platforms:
            plugin = PluginDefinition(
                name="test_plugin",
                version="1.0.0",
                target_platform=platform,
                source_files=source_files,
                compile_config=compile_config,
                interface_config=interface_config
            )
            assert plugin.target_platform == platform

        # Test invalid platform
        with pytest.raises(ValueError, match="Invalid target platform"):
            PluginDefinition(
                name="test_plugin",
                version="1.0.0",
                target_platform="invalid_platform",
                source_files=source_files,
                compile_config=compile_config,
                interface_config=interface_config
            )

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_source_files_validation(self):
        """Test source files validation."""
        compile_config = {"target_platform": "cpp"}
        interface_config = {"checkpoint_mapping": {}}
        
        # Test valid source files
        with tempfile.TemporaryDirectory() as temp_dir:
            source_file = os.path.join(temp_dir, "plugin.cpp")
            with open(source_file, 'w') as f:
                f.write("// Test plugin source")
            
            with patch('os.path.exists', return_value=True):
                plugin = PluginDefinition(
                    name="test_plugin",
                    version="1.0.0",
                    target_platform="cpp",
                    source_files=[source_file],
                    compile_config=compile_config,
                    interface_config=interface_config
                )
                assert plugin.source_files == [source_file]

        # Test empty source files
        with pytest.raises(ValueError, match="source_files must contain at least one valid file path"):
            PluginDefinition(
                name="test_plugin",
                version="1.0.0",
                target_platform="cpp",
                source_files=[],
                compile_config=compile_config,
                interface_config=interface_config
            )

        # Test non-existent source files
        with pytest.raises(ValueError, match="source file does not exist"):
            PluginDefinition(
                name="test_plugin",
                version="1.0.0",
                target_platform="cpp",
                source_files=["/nonexistent/file.cpp"],
                compile_config=compile_config,
                interface_config=interface_config
            )

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_compile_config_validation(self):
        """Test compile configuration validation."""
        source_files = ["src/plugin.cpp"]
        interface_config = {"checkpoint_mapping": {}}
        
        # Test valid compile config
        valid_config = {
            "target_platform": "cpp",
            "compiler_flags": ["-std=c++17", "-O2"],
            "dependencies": ["pthread"],
            "output_path": "lib/plugin.so"
        }
        
        plugin = PluginDefinition(
            name="test_plugin",
            version="1.0.0",
            target_platform="cpp",
            source_files=source_files,
            compile_config=valid_config,
            interface_config=interface_config
        )
        assert plugin.compile_config == valid_config

        # Test missing target platform in compile config
        invalid_config = {
            "compiler_flags": ["-std=c++17"],
            "output_path": "lib/plugin.so"
        }
        
        with pytest.raises(ValueError, match="compile_config must specify target platform requirements"):
            PluginDefinition(
                name="test_plugin",
                version="1.0.0",
                target_platform="cpp",
                source_files=source_files,
                compile_config=invalid_config,
                interface_config=interface_config
            )

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_interface_config_validation(self):
        """Test interface configuration validation."""
        source_files = ["src/plugin.cpp"]
        compile_config = {"target_platform": "cpp"}
        
        # Test valid interface config
        valid_config = {
            "checkpoint_mapping": {
                "post_attention": "forward_attention",
                "post_mlp": "forward_mlp"
            },
            "validation": {
                "extract_signatures": True,
                "runtime_type_check": True
            }
        }
        
        plugin = PluginDefinition(
            name="test_plugin",
            version="1.0.0",
            target_platform="cpp",
            source_files=source_files,
            compile_config=compile_config,
            interface_config=valid_config
        )
        assert plugin.interface_config == valid_config

        # Test invalid checkpoint names in mapping
        invalid_config = {
            "checkpoint_mapping": {
                "invalid_checkpoint": "some_function"
            }
        }
        
        with pytest.raises(ValueError, match="interface_config must map to valid checkpoint names"):
            PluginDefinition(
                name="test_plugin",
                version="1.0.0",
                target_platform="cpp",
                source_files=source_files,
                compile_config=compile_config,
                interface_config=invalid_config
            )

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_compilation_status_management(self):
        """Test compilation status and error handling."""
        source_files = ["src/plugin.cpp"]
        compile_config = {"target_platform": "cpp"}
        interface_config = {"checkpoint_mapping": {}}
        
        plugin = PluginDefinition(
            name="test_plugin",
            version="1.0.0",
            target_platform="cpp",
            source_files=source_files,
            compile_config=compile_config,
            interface_config=interface_config
        )
        
        # Initial state
        assert plugin.is_compiled is False
        assert plugin.binary_path is None
        assert plugin.compilation_errors is None
        assert plugin.last_compiled_at is None
        
        # Test successful compilation
        binary_path = "/path/to/compiled/plugin.so"
        plugin.mark_compiled(binary_path)
        
        assert plugin.is_compiled is True
        assert plugin.binary_path == binary_path
        assert plugin.compilation_errors is None
        assert plugin.last_compiled_at is not None
        
        # Test compilation failure
        error_message = "Compilation failed: undefined symbol"
        plugin.mark_compilation_failed(error_message)
        
        assert plugin.is_compiled is False
        assert plugin.compilation_errors == error_message

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_platform_specific_configurations(self):
        """Test platform-specific compile configurations."""
        source_files = ["src/plugin.metal", "src/binding.cpp"]
        interface_config = {"checkpoint_mapping": {"post_attention": "forward_attention"}}
        
        # Metal platform configuration
        metal_config = {
            "target_platform": "metal",
            "compiler_flags": ["-std=metal2.0"],
            "dependencies": ["Metal.framework", "MetalKit.framework"],
            "output_path": "lib/metal_plugin.metallib"
        }
        
        metal_plugin = PluginDefinition(
            name="metal_plugin",
            version="1.0.0",
            target_platform="metal",
            source_files=source_files,
            compile_config=metal_config,
            interface_config=interface_config
        )
        assert metal_plugin.target_platform == "metal"
        assert "Metal.framework" in metal_plugin.compile_config["dependencies"]
        
        # CUDA platform configuration
        cuda_config = {
            "target_platform": "cuda",
            "compiler_flags": ["-std=c++17", "--gpu-architecture=sm_70"],
            "dependencies": ["cuda", "cublas"],
            "output_path": "lib/cuda_plugin.so"
        }
        
        cuda_plugin = PluginDefinition(
            name="cuda_plugin",
            version="1.0.0",
            target_platform="cuda",
            source_files=["src/plugin.cu"],
            compile_config=cuda_config,
            interface_config=interface_config
        )
        assert cuda_plugin.target_platform == "cuda"
        assert "--gpu-architecture=sm_70" in cuda_plugin.compile_config["compiler_flags"]

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_interface_validator_relationship(self):
        """Test relationship with InterfaceValidator."""
        source_files = ["src/plugin.cpp"]
        compile_config = {"target_platform": "cpp"}
        interface_config = {"checkpoint_mapping": {"post_attention": "forward_attention"}}
        
        plugin = PluginDefinition(
            name="test_plugin",
            version="1.0.0",
            target_platform="cpp",
            source_files=source_files,
            compile_config=compile_config,
            interface_config=interface_config
        )
        
        # Mock InterfaceValidator creation
        with patch('debug_framework.models.interface_validator.InterfaceValidator') as mock_validator:
            mock_validator_instance = MagicMock()
            mock_validator.return_value = mock_validator_instance
            
            # Test validator creation
            validator = plugin.create_interface_validator()
            mock_validator.assert_called_once_with(plugin_id=plugin.id)
            
            # Test validator access
            plugin.interface_validator = mock_validator_instance
            assert plugin.get_interface_validator() == mock_validator_instance

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_database_integration(self):
        """Test database persistence operations."""
        source_files = ["src/plugin.cpp"]
        compile_config = {"target_platform": "cpp"}
        interface_config = {"checkpoint_mapping": {"post_attention": "forward_attention"}}
        
        plugin = PluginDefinition(
            name="test_plugin",
            version="1.0.0",
            target_platform="cpp",
            source_files=source_files,
            compile_config=compile_config,
            interface_config=interface_config
        )
        
        # Mock database operations
        with patch('debug_framework.services.database_manager.DatabaseManager') as mock_db:
            mock_db_instance = MagicMock()
            mock_db.return_value = mock_db_instance
            
            # Test save operation
            plugin.save()
            mock_db_instance.insert_plugin_definition.assert_called_once()
            
            # Test load operation
            mock_db_instance.get_plugin_definition.return_value = {
                'id': 1,
                'name': 'test_plugin',
                'version': '1.0.0',
                'target_platform': 'cpp',
                'source_files': json.dumps(source_files),
                'compile_config': json.dumps(compile_config),
                'interface_config': json.dumps(interface_config),
                'is_compiled': False,
                'binary_path': None,
                'compilation_errors': None,
                'created_at': datetime.now().isoformat()
            }
            
            loaded_plugin = PluginDefinition.load(1)
            assert loaded_plugin.name == "test_plugin"
            assert loaded_plugin.version == "1.0.0"

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_plugin_uniqueness(self):
        """Test plugin name uniqueness constraints."""
        source_files = ["src/plugin.cpp"]
        compile_config = {"target_platform": "cpp"}
        interface_config = {"checkpoint_mapping": {}}
        
        # Mock database to simulate existing plugin
        with patch('debug_framework.services.database_manager.DatabaseManager') as mock_db:
            mock_db_instance = MagicMock()
            mock_db.return_value = mock_db_instance
            
            # Simulate duplicate name error
            mock_db_instance.insert_plugin_definition.side_effect = Exception("UNIQUE constraint failed: name")
            
            plugin = PluginDefinition(
                name="existing_plugin",
                version="1.0.0",
                target_platform="cpp",
                source_files=source_files,
                compile_config=compile_config,
                interface_config=interface_config
            )
            
            with pytest.raises(ValueError, match="Plugin name 'existing_plugin' already exists"):
                plugin.save()

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_json_serialization(self):
        """Test JSON serialization/deserialization."""
        source_files = ["src/plugin.cpp", "src/helper.cpp"]
        compile_config = {
            "target_platform": "cpp",
            "compiler_flags": ["-std=c++17", "-O2"],
            "dependencies": ["pthread"]
        }
        interface_config = {
            "checkpoint_mapping": {"post_attention": "forward_attention"},
            "validation": {"extract_signatures": True}
        }
        
        plugin = PluginDefinition(
            name="test_plugin",
            version="1.0.0",
            target_platform="cpp",
            source_files=source_files,
            compile_config=compile_config,
            interface_config=interface_config
        )
        
        # Test serialization
        plugin_dict = plugin.to_dict()
        assert plugin_dict["name"] == "test_plugin"
        assert plugin_dict["version"] == "1.0.0"
        assert plugin_dict["source_files"] == source_files
        assert plugin_dict["compile_config"] == compile_config
        assert plugin_dict["interface_config"] == interface_config
        
        # Test deserialization
        restored_plugin = PluginDefinition.from_dict(plugin_dict)
        assert restored_plugin.name == plugin.name
        assert restored_plugin.version == plugin.version
        assert restored_plugin.source_files == plugin.source_files

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_compilation_pipeline_integration(self):
        """Test integration with compilation pipeline."""
        source_files = ["src/plugin.cpp"]
        compile_config = {"target_platform": "cpp", "output_path": "lib/plugin.so"}
        interface_config = {"checkpoint_mapping": {"post_attention": "forward_attention"}}
        
        plugin = PluginDefinition(
            name="test_plugin",
            version="1.0.0",
            target_platform="cpp",
            source_files=source_files,
            compile_config=compile_config,
            interface_config=interface_config
        )
        
        # Mock compilation engine
        with patch('debug_framework.services.compilation_engine.CompilationEngine') as mock_engine:
            mock_engine_instance = MagicMock()
            mock_engine.return_value = mock_engine_instance
            mock_engine_instance.compile_plugin.return_value = {
                "success": True,
                "binary_path": "/path/to/compiled/plugin.so",
                "errors": None
            }
            
            # Test compilation trigger
            result = plugin.compile()
            
            assert result["success"] is True
            assert plugin.is_compiled is True
            assert plugin.binary_path == "/path/to/compiled/plugin.so"

    @pytest.mark.skipif(not PLUGINDEFINITION_AVAILABLE, reason="PluginDefinition not implemented")
    def test_checkpoint_mapping_validation(self):
        """Test checkpoint mapping validation."""
        source_files = ["src/plugin.cpp"]
        compile_config = {"target_platform": "cpp"}
        
        valid_checkpoint_names = [
            "post_embedding", "post_rope", "post_attention", 
            "pre_mlp", "post_mlp", "final_output"
        ]
        
        # Test valid checkpoint mappings
        for checkpoint in valid_checkpoint_names:
            interface_config = {
                "checkpoint_mapping": {checkpoint: f"forward_{checkpoint}"}
            }
            
            plugin = PluginDefinition(
                name="test_plugin",
                version="1.0.0",
                target_platform="cpp",
                source_files=source_files,
                compile_config=compile_config,
                interface_config=interface_config
            )
            assert checkpoint in plugin.interface_config["checkpoint_mapping"]

        # Test mapping to multiple checkpoints
        interface_config = {
            "checkpoint_mapping": {
                "post_attention": "forward_attention",
                "post_mlp": "forward_mlp",
                "final_output": "forward_output"
            }
        }
        
        plugin = PluginDefinition(
            name="multi_checkpoint_plugin",
            version="1.0.0",
            target_platform="cpp",
            source_files=source_files,
            compile_config=compile_config,
            interface_config=interface_config
        )
        
        mapped_checkpoints = list(plugin.interface_config["checkpoint_mapping"].keys())
        assert len(mapped_checkpoints) == 3
        assert "post_attention" in mapped_checkpoints
        assert "post_mlp" in mapped_checkpoints
        assert "final_output" in mapped_checkpoints