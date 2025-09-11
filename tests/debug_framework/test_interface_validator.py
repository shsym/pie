"""
Test module for InterfaceValidator model.

This test module validates the InterfaceValidator data model which is responsible
for extracting function signatures from compiled plugin binaries and validating
tensor compatibility.

TDD: This test MUST FAIL until the InterfaceValidator model is implemented.
"""

import pytest
import json
import ctypes
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open

# Proper TDD pattern - use try/except for conditional loading
try:
    from debug_framework.models.interface_validator import InterfaceValidator
    INTERFACEVALIDATOR_AVAILABLE = True
except ImportError:
    InterfaceValidator = None
    INTERFACEVALIDATOR_AVAILABLE = False


class TestInterfaceValidator:
    """Test suite for InterfaceValidator model functionality."""

    def test_interface_validator_import_fails(self):
        """Test that InterfaceValidator import fails (TDD requirement)."""
        with pytest.raises(ImportError):
            from debug_framework.models.interface_validator import InterfaceValidator

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_interface_validator_creation(self):
        """Test basic InterfaceValidator object creation."""
        extracted_signatures = {
            "forward_attention": {
                "args": ["float*", "float*", "int", "int"],
                "return_type": "void",
                "attributes": {"device": "gpu"}
            }
        }
        tensor_requirements = {
            "input_tensor": {
                "dtype": "float32",
                "shape": [-1, 128, 64],
                "layout": "contiguous"
            }
        }
        validation_rules = {
            "strict_shapes": True,
            "allow_dtype_conversion": False,
            "memory_layout_check": True
        }
        
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures=extracted_signatures,
            tensor_requirements=tensor_requirements,
            validation_rules=validation_rules
        )
        
        assert validator.plugin_id == 1
        assert validator.extracted_signatures == extracted_signatures
        assert validator.tensor_requirements == tensor_requirements
        assert validator.validation_rules == validation_rules
        assert validator.validation_status == "pending"

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_signature_extraction_validation(self):
        """Test that extracted_signatures must contain valid function metadata."""
        # Valid signature format
        valid_signatures = {
            "forward_attention": {
                "args": ["float*", "float*", "int", "int"],
                "return_type": "void",
                "attributes": {"device": "gpu", "precision": "fp32"}
            },
            "forward_mlp": {
                "args": ["float*", "int", "int"],
                "return_type": "float*",
                "attributes": {"device": "cpu"}
            }
        }
        
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures=valid_signatures,
            tensor_requirements={},
            validation_rules={}
        )
        assert validator.extracted_signatures == valid_signatures

        # Invalid signature format - missing required fields
        invalid_signatures = {
            "forward_attention": {
                "args": ["float*", "float*"],
                # Missing return_type
            }
        }
        
        with pytest.raises(ValueError, match="extracted_signatures must contain valid function metadata"):
            InterfaceValidator(
                plugin_id=1,
                extracted_signatures=invalid_signatures,
                tensor_requirements={},
                validation_rules={}
            )

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_tensor_requirements_validation(self):
        """Test tensor requirements specification validation."""
        # Valid tensor requirements
        valid_requirements = {
            "input_tensor": {
                "dtype": "float32",
                "shape": [-1, 128, 64],
                "layout": "contiguous"
            },
            "weight_tensor": {
                "dtype": "float16", 
                "shape": [64, 64],
                "layout": "strided"
            }
        }
        
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures={},
            tensor_requirements=valid_requirements,
            validation_rules={}
        )
        assert validator.tensor_requirements == valid_requirements

        # Invalid tensor requirements - missing required fields
        invalid_requirements = {
            "input_tensor": {
                "dtype": "float32",
                # Missing shape and layout
            }
        }
        
        with pytest.raises(ValueError, match="tensor_requirements must specify dtype, shape, and memory layout"):
            InterfaceValidator(
                plugin_id=1,
                extracted_signatures={},
                tensor_requirements=invalid_requirements,
                validation_rules={}
            )

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_validation_status_management(self):
        """Test validation status transitions and requirements."""
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures={},
            tensor_requirements={},
            validation_rules={}
        )
        
        # Initial status
        assert validator.validation_status == "pending"
        
        # Test valid status transitions
        valid_statuses = ["valid", "invalid", "warning"]
        for status in valid_statuses:
            validator.validation_status = status
            assert validator.validation_status == status

        # Test invalid status
        with pytest.raises(ValueError, match="Invalid validation status"):
            validator.validation_status = "unknown_status"

        # Test that plugin cannot be used unless status is 'valid'
        validator.validation_status = "invalid"
        assert validator.can_be_used() is False
        
        validator.validation_status = "valid"
        assert validator.can_be_used() is True

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_validation_errors_handling(self):
        """Test validation error storage and retrieval."""
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures={},
            tensor_requirements={},
            validation_rules={}
        )
        
        # Test adding validation errors
        errors = [
            {
                "type": "signature_mismatch",
                "function": "forward_attention", 
                "expected": "float*",
                "actual": "int*",
                "message": "Parameter type mismatch"
            },
            {
                "type": "tensor_incompatibility",
                "tensor": "input_tensor",
                "issue": "shape_mismatch",
                "message": "Expected shape [32, 128] but got [32, 64]"
            }
        ]
        
        validator.add_validation_errors(errors)
        stored_errors = validator.get_validation_errors()
        
        assert len(stored_errors) == 2
        assert stored_errors[0]["type"] == "signature_mismatch"
        assert stored_errors[1]["type"] == "tensor_incompatibility"
        
        # Test that errors provide specific mismatch details
        signature_error = stored_errors[0]
        assert "expected" in signature_error
        assert "actual" in signature_error
        assert "message" in signature_error

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_binary_signature_extraction(self):
        """Test extraction of function signatures from compiled binaries."""
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures={},
            tensor_requirements={},
            validation_rules={}
        )
        
        # Mock binary analysis
        with patch('debug_framework.bindings.signature_extractor.extract_signatures') as mock_extract:
            mock_extract.return_value = {
                "forward_attention": {
                    "args": ["float*", "float*", "int", "int"],
                    "return_type": "void",
                    "attributes": {"calling_convention": "cdecl"}
                }
            }
            
            # Test signature extraction from binary
            binary_path = "/path/to/plugin.so"
            signatures = validator.extract_signatures_from_binary(binary_path)
            
            mock_extract.assert_called_once_with(binary_path)
            assert "forward_attention" in signatures
            assert signatures["forward_attention"]["return_type"] == "void"

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_tensor_compatibility_validation(self):
        """Test tensor compatibility validation between requirements and runtime."""
        tensor_requirements = {
            "input_tensor": {
                "dtype": "float32",
                "shape": [-1, 128, 64],
                "layout": "contiguous"
            }
        }
        
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures={},
            tensor_requirements=tensor_requirements,
            validation_rules={"strict_shapes": True}
        )
        
        # Test compatible tensor
        runtime_tensor_spec = {
            "dtype": "float32",
            "shape": [32, 128, 64],
            "layout": "contiguous"
        }
        
        is_compatible = validator.validate_tensor_compatibility("input_tensor", runtime_tensor_spec)
        assert is_compatible is True
        
        # Test incompatible dtype
        incompatible_spec = {
            "dtype": "float16",  # Different dtype
            "shape": [32, 128, 64],
            "layout": "contiguous"
        }
        
        is_compatible = validator.validate_tensor_compatibility("input_tensor", incompatible_spec)
        assert is_compatible is False
        
        # Test incompatible shape (with strict_shapes enabled)
        incompatible_shape = {
            "dtype": "float32",
            "shape": [32, 64, 64],  # Different shape
            "layout": "contiguous"
        }
        
        is_compatible = validator.validate_tensor_compatibility("input_tensor", incompatible_shape)
        assert is_compatible is False

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_runtime_validation_configuration(self):
        """Test runtime validation configuration options."""
        validation_rules = {
            "strict_shapes": False,  # Allow shape flexibility
            "allow_dtype_conversion": True,  # Allow dtype conversion
            "memory_layout_check": True,  # Enforce layout requirements
            "auto_adapt_tensors": True  # Auto-adapt tensor formats
        }
        
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures={},
            tensor_requirements={},
            validation_rules=validation_rules
        )
        
        # Test that relaxed rules allow more flexibility
        assert validator.validation_rules["strict_shapes"] is False
        assert validator.validation_rules["allow_dtype_conversion"] is True
        
        # Test rule application
        assert validator.allows_dtype_conversion() is True
        assert validator.requires_strict_shapes() is False
        assert validator.checks_memory_layout() is True

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_cross_platform_signature_handling(self):
        """Test handling of platform-specific function signatures."""
        # Metal platform signatures
        metal_signatures = {
            "forward_attention": {
                "args": ["device float*", "device float*", "uint", "uint"],
                "return_type": "void",
                "attributes": {
                    "kernel": True,
                    "platform": "metal",
                    "thread_group_size": [32, 1, 1]
                }
            }
        }
        
        metal_validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures=metal_signatures,
            tensor_requirements={},
            validation_rules={}
        )
        
        assert metal_validator.get_platform() == "metal"
        assert metal_validator.is_kernel_function("forward_attention") is True
        
        # C++ platform signatures
        cpp_signatures = {
            "forward_attention": {
                "args": ["float*", "float*", "int", "int"],
                "return_type": "void",
                "attributes": {
                    "platform": "cpp",
                    "calling_convention": "cdecl"
                }
            }
        }
        
        cpp_validator = InterfaceValidator(
            plugin_id=2,
            extracted_signatures=cpp_signatures,
            tensor_requirements={},
            validation_rules={}
        )
        
        assert cpp_validator.get_platform() == "cpp"
        assert cpp_validator.get_calling_convention("forward_attention") == "cdecl"

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_interface_validation_execution(self):
        """Test complete interface validation process."""
        signatures = {
            "forward_attention": {
                "args": ["float*", "float*", "int", "int"],
                "return_type": "void",
                "attributes": {"platform": "cpp"}
            }
        }
        
        requirements = {
            "input_tensor": {
                "dtype": "float32",
                "shape": [-1, 128, 64],
                "layout": "contiguous"
            }
        }
        
        validation_rules = {
            "strict_shapes": True,
            "allow_dtype_conversion": False
        }
        
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures=signatures,
            tensor_requirements=requirements,
            validation_rules=validation_rules
        )
        
        # Execute validation
        result = validator.execute_validation()
        
        assert result["status"] in ["valid", "invalid", "warning"]
        assert "errors" in result
        assert "warnings" in result
        assert validator.last_validation_at is not None

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_plugin_relationship(self):
        """Test relationship with PluginDefinition."""
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures={},
            tensor_requirements={},
            validation_rules={}
        )
        
        # Mock plugin relationship
        with patch('debug_framework.models.plugin_definition.PluginDefinition') as mock_plugin:
            mock_plugin_instance = MagicMock()
            mock_plugin_instance.id = 1
            mock_plugin_instance.name = "test_plugin"
            mock_plugin.load.return_value = mock_plugin_instance
            
            # Test accessing associated plugin
            plugin = validator.get_plugin()
            mock_plugin.load.assert_called_once_with(1)
            assert plugin.id == 1
            assert plugin.name == "test_plugin"

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_database_integration(self):
        """Test database persistence operations."""
        signatures = {"forward_attention": {"args": ["float*"], "return_type": "void"}}
        requirements = {"input": {"dtype": "float32", "shape": [-1], "layout": "contiguous"}}
        rules = {"strict_shapes": True}
        
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures=signatures,
            tensor_requirements=requirements,
            validation_rules=rules
        )
        
        # Mock database operations
        with patch('debug_framework.services.database_manager.DatabaseManager') as mock_db:
            mock_db_instance = MagicMock()
            mock_db.return_value = mock_db_instance
            
            # Test save operation
            validator.save()
            mock_db_instance.insert_interface_validator.assert_called_once()
            
            # Test load operation
            mock_db_instance.get_interface_validator.return_value = {
                'id': 1,
                'plugin_id': 1,
                'extracted_signatures': json.dumps(signatures),
                'tensor_requirements': json.dumps(requirements),
                'validation_rules': json.dumps(rules),
                'validation_status': 'valid',
                'validation_errors': json.dumps([]),
                'last_validation_at': datetime.now().isoformat()
            }
            
            loaded_validator = InterfaceValidator.load(1)
            assert loaded_validator.plugin_id == 1
            assert loaded_validator.extracted_signatures == signatures

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_json_serialization(self):
        """Test JSON serialization/deserialization."""
        signatures = {
            "forward_attention": {
                "args": ["float*", "int"],
                "return_type": "void",
                "attributes": {"platform": "cpp"}
            }
        }
        requirements = {
            "input": {"dtype": "float32", "shape": [-1, 64], "layout": "contiguous"}
        }
        rules = {"strict_shapes": True, "allow_dtype_conversion": False}
        
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures=signatures,
            tensor_requirements=requirements,
            validation_rules=rules
        )
        
        # Test serialization
        validator_dict = validator.to_dict()
        assert validator_dict["plugin_id"] == 1
        assert validator_dict["extracted_signatures"] == signatures
        assert validator_dict["tensor_requirements"] == requirements
        assert validator_dict["validation_rules"] == rules
        
        # Test deserialization
        restored_validator = InterfaceValidator.from_dict(validator_dict)
        assert restored_validator.plugin_id == validator.plugin_id
        assert restored_validator.extracted_signatures == validator.extracted_signatures
        assert restored_validator.tensor_requirements == validator.tensor_requirements

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_auto_tensor_adaptation(self):
        """Test automatic tensor format adaptation."""
        requirements = {
            "input_tensor": {
                "dtype": "float32",
                "shape": [-1, 128, 64],
                "layout": "contiguous"
            }
        }
        
        validation_rules = {
            "auto_adapt_tensors": True,
            "allow_dtype_conversion": True
        }
        
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures={},
            tensor_requirements=requirements,
            validation_rules=validation_rules
        )
        
        # Test tensor adaptation
        runtime_spec = {
            "dtype": "float16",  # Different dtype
            "shape": [32, 128, 64],
            "layout": "strided"  # Different layout
        }
        
        adapted_spec = validator.adapt_tensor("input_tensor", runtime_spec)
        
        # Should adapt to meet requirements
        assert adapted_spec["dtype"] == "float32"  # Converted
        assert adapted_spec["shape"] == [32, 128, 64]  # Shape preserved
        assert adapted_spec["layout"] == "contiguous"  # Layout adapted

    @pytest.mark.skipif(not INTERFACEVALIDATOR_AVAILABLE, reason="InterfaceValidator not implemented")
    def test_validation_warning_handling(self):
        """Test handling of validation warnings vs errors."""
        validator = InterfaceValidator(
            plugin_id=1,
            extracted_signatures={},
            tensor_requirements={},
            validation_rules={}
        )
        
        # Test adding warnings (non-critical issues)
        warnings = [
            {
                "type": "performance_warning",
                "message": "Non-contiguous memory layout may impact performance",
                "severity": "low"
            }
        ]
        
        validator.add_validation_warnings(warnings)
        
        # Warnings should not prevent plugin usage
        validator.validation_status = "warning"
        assert validator.can_be_used() is True
        
        # But errors should prevent usage
        errors = [
            {
                "type": "signature_mismatch",
                "message": "Critical function signature incompatibility",
                "severity": "high"
            }
        ]
        
        validator.add_validation_errors(errors)
        validator.validation_status = "invalid"
        assert validator.can_be_used() is False