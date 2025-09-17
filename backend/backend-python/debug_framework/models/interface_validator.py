"""
InterfaceValidator model for the debug framework.

This module defines the InterfaceValidator data model which is responsible
for extracting function signatures from compiled plugin binaries and validating
tensor compatibility.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from debug_framework.services import database_manager


class InterfaceValidator:
    """
    Validates plugin interfaces and tensor compatibility.

    Extracts function signatures from compiled plugin binaries, validates tensor
    requirements, and ensures compatibility between plugin interfaces and the
    validation framework's expectations.
    """

    # Valid validation statuses
    VALID_STATUSES = {"pending", "valid", "invalid", "warning"}

    # Required fields in function signatures
    REQUIRED_SIGNATURE_FIELDS = {"args", "return_type"}

    # Required fields in tensor requirements
    REQUIRED_TENSOR_FIELDS = {"dtype", "shape", "layout"}

    def __init__(
        self,
        plugin_id: int,
        extracted_signatures: Dict[str, Any],
        tensor_requirements: Dict[str, Any],
        validation_rules: Dict[str, Any],
        id: Optional[int] = None,
        validation_status: str = "pending",
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        validation_warnings: Optional[List[Dict[str, Any]]] = None,
        last_validation_at: Optional[datetime] = None
    ):
        """
        Initialize an InterfaceValidator instance.

        Args:
            plugin_id: ID of the associated PluginDefinition
            extracted_signatures: Function signatures extracted from binary
            tensor_requirements: Tensor specifications and requirements
            validation_rules: Configuration for validation behavior
            id: Unique validator identifier (auto-assigned if None)
            validation_status: Current validation status (pending, valid, invalid, warning)
            validation_errors: List of validation errors encountered
            validation_warnings: List of validation warnings encountered
            last_validation_at: Timestamp of last validation execution

        Raises:
            ValueError: If validation rules are violated
        """
        # Validate plugin ID
        if not isinstance(plugin_id, int) or plugin_id <= 0:
            raise ValueError("plugin_id must be a positive integer")

        # Validate extracted signatures format
        self._validate_extracted_signatures(extracted_signatures)

        # Validate tensor requirements format
        self._validate_tensor_requirements(tensor_requirements)

        # Validate validation status
        self._validate_status(validation_status)

        # Store attributes
        self.id = id  # None until saved to database
        self.plugin_id = plugin_id
        self.extracted_signatures = extracted_signatures.copy() if extracted_signatures else {}
        self.tensor_requirements = tensor_requirements.copy() if tensor_requirements else {}
        self.validation_rules = validation_rules.copy() if validation_rules else {}
        self._validation_status = validation_status
        self.validation_errors = validation_errors or []
        self.validation_warnings = validation_warnings or []
        self.last_validation_at = last_validation_at

    def _validate_extracted_signatures(self, signatures: Dict[str, Any]) -> None:
        """Validate that extracted signatures contain required metadata."""
        if not isinstance(signatures, dict):
            raise ValueError("extracted_signatures must be a dictionary")

        for function_name, signature in signatures.items():
            if not isinstance(signature, dict):
                raise ValueError("extracted_signatures must contain valid function metadata")

            # Check for required fields
            missing_fields = self.REQUIRED_SIGNATURE_FIELDS - set(signature.keys())
            if missing_fields:
                raise ValueError("extracted_signatures must contain valid function metadata")

            # Validate args is a list
            if not isinstance(signature.get("args"), list):
                raise ValueError("extracted_signatures must contain valid function metadata")

    def _validate_tensor_requirements(self, requirements: Dict[str, Any]) -> None:
        """Validate that tensor requirements contain necessary specifications."""
        if not isinstance(requirements, dict):
            raise ValueError("tensor_requirements must be a dictionary")

        for tensor_name, requirement in requirements.items():
            if not isinstance(requirement, dict):
                raise ValueError("tensor_requirements must specify dtype, shape, and memory layout")

            # Check for required fields
            missing_fields = self.REQUIRED_TENSOR_FIELDS - set(requirement.keys())
            if missing_fields:
                raise ValueError("tensor_requirements must specify dtype, shape, and memory layout")

    def _validate_status(self, status: str) -> None:
        """Validate that status is one of the allowed values."""
        if status not in self.VALID_STATUSES:
            valid_statuses = ", ".join(sorted(self.VALID_STATUSES))
            raise ValueError(f"Invalid validation status '{status}'. Must be one of: {valid_statuses}")

    @property
    def validation_status(self) -> str:
        """Get current validation status."""
        return self._validation_status

    @validation_status.setter
    def validation_status(self, status: str) -> None:
        """Set validation status with validation."""
        self._validate_status(status)
        self._validation_status = status

    def can_be_used(self) -> bool:
        """
        Check if plugin can be used based on validation status.

        Returns:
            True if plugin is valid or has warnings only, False for invalid status
        """
        return self._validation_status in {"valid", "warning"}

    def add_validation_errors(self, errors: List[Dict[str, Any]]) -> None:
        """
        Add validation errors to the current list.

        Args:
            errors: List of error dictionaries with type, message, and other details
        """
        if not isinstance(errors, list):
            raise ValueError("errors must be a list")

        self.validation_errors.extend(errors)

    def add_validation_warnings(self, warnings: List[Dict[str, Any]]) -> None:
        """
        Add validation warnings to the current list.

        Args:
            warnings: List of warning dictionaries with type, message, and other details
        """
        if not isinstance(warnings, list):
            raise ValueError("warnings must be a list")

        self.validation_warnings.extend(warnings)

    def get_validation_errors(self) -> List[Dict[str, Any]]:
        """Get current validation errors."""
        return self.validation_errors.copy()

    def get_validation_warnings(self) -> List[Dict[str, Any]]:
        """Get current validation warnings."""
        return self.validation_warnings.copy()

    def extract_signatures_from_binary(self, binary_path: str) -> Dict[str, Any]:
        """
        Extract function signatures from compiled binary.

        Args:
            binary_path: Path to compiled plugin binary

        Returns:
            Dictionary of function signatures extracted from binary

        Note:
            This method uses dynamic imports to avoid circular dependencies.
        """
        from debug_framework.bindings.signature_extractor import extract_signatures

        signatures = extract_signatures(binary_path)
        self.extracted_signatures.update(signatures)
        return signatures

    def validate_tensor_compatibility(self, tensor_name: str, runtime_spec: Dict[str, Any]) -> bool:
        """
        Validate compatibility between tensor requirements and runtime specification.

        Args:
            tensor_name: Name of tensor to validate
            runtime_spec: Runtime tensor specification (dtype, shape, layout)

        Returns:
            True if tensor is compatible, False otherwise
        """
        if tensor_name not in self.tensor_requirements:
            return True  # No requirements means compatible

        required_spec = self.tensor_requirements[tensor_name]

        # Check dtype compatibility
        if not self._check_dtype_compatibility(required_spec["dtype"], runtime_spec.get("dtype")):
            return False

        # Check shape compatibility
        if not self._check_shape_compatibility(required_spec["shape"], runtime_spec.get("shape")):
            return False

        # Check layout compatibility
        if not self._check_layout_compatibility(required_spec["layout"], runtime_spec.get("layout")):
            return False

        return True

    def _check_dtype_compatibility(self, required_dtype: str, runtime_dtype: str) -> bool:
        """Check if data types are compatible."""
        if required_dtype == runtime_dtype:
            return True

        # Check if dtype conversion is allowed
        if self.validation_rules.get("allow_dtype_conversion", False):
            # Allow conversion between compatible types
            float_types = {"float16", "float32", "float64"}
            if required_dtype in float_types and runtime_dtype in float_types:
                return True

        return False

    def _check_shape_compatibility(self, required_shape: List[int], runtime_shape: List[int]) -> bool:
        """Check if shapes are compatible."""
        if not isinstance(required_shape, list) or not isinstance(runtime_shape, list):
            return False

        if len(required_shape) != len(runtime_shape):
            return False

        # If strict shapes is disabled, allow flexible dimensions
        strict_shapes = self.validation_rules.get("strict_shapes", True)

        for req_dim, runtime_dim in zip(required_shape, runtime_shape):
            if req_dim == -1:  # -1 means any size allowed
                continue
            if strict_shapes and req_dim != runtime_dim:
                return False

        return True

    def _check_layout_compatibility(self, required_layout: str, runtime_layout: str) -> bool:
        """Check if memory layouts are compatible."""
        if required_layout == runtime_layout:
            return True

        # Check if layout conversion is supported
        if not self.validation_rules.get("memory_layout_check", True):
            return True  # Layout checking disabled

        return False

    def allows_dtype_conversion(self) -> bool:
        """Check if validation rules allow dtype conversion."""
        return self.validation_rules.get("allow_dtype_conversion", False)

    def requires_strict_shapes(self) -> bool:
        """Check if validation rules require strict shape matching."""
        return self.validation_rules.get("strict_shapes", True)

    def checks_memory_layout(self) -> bool:
        """Check if validation rules enforce memory layout requirements."""
        return self.validation_rules.get("memory_layout_check", True)

    def get_platform(self) -> str:
        """
        Get platform from extracted signatures.

        Returns:
            Platform name (metal, cpp, cuda, etc.) or "unknown"
        """
        if not self.extracted_signatures:
            return "unknown"

        # Look for platform in any signature's attributes
        for signature in self.extracted_signatures.values():
            if "attributes" in signature:
                platform = signature["attributes"].get("platform")
                if platform:
                    return platform

        return "unknown"

    def is_kernel_function(self, function_name: str) -> bool:
        """
        Check if a function is a kernel function (for GPU platforms).

        Args:
            function_name: Name of function to check

        Returns:
            True if function is marked as a kernel, False otherwise
        """
        if function_name not in self.extracted_signatures:
            return False

        signature = self.extracted_signatures[function_name]
        if "attributes" in signature:
            return signature["attributes"].get("kernel", False)

        return False

    def get_calling_convention(self, function_name: str) -> str:
        """
        Get calling convention for a function.

        Args:
            function_name: Name of function

        Returns:
            Calling convention string or "unknown"
        """
        if function_name not in self.extracted_signatures:
            return "unknown"

        signature = self.extracted_signatures[function_name]
        if "attributes" in signature:
            return signature["attributes"].get("calling_convention", "unknown")

        return "unknown"

    def execute_validation(self) -> Dict[str, Any]:
        """
        Execute complete interface validation process.

        Returns:
            Dictionary with validation results including status, errors, warnings
        """
        self.validation_errors.clear()
        self.validation_warnings.clear()

        # Perform various validations
        self._validate_signature_consistency()
        self._validate_tensor_compatibility_all()
        self._check_platform_specific_requirements()

        # Determine final status
        if self.validation_errors:
            self._validation_status = "invalid"
        elif self.validation_warnings:
            self._validation_status = "warning"
        else:
            self._validation_status = "valid"

        self.last_validation_at = datetime.now()

        return {
            "status": self._validation_status,
            "errors": self.validation_errors.copy(),
            "warnings": self.validation_warnings.copy()
        }

    def _validate_signature_consistency(self) -> None:
        """Validate internal consistency of extracted signatures."""
        for function_name, signature in self.extracted_signatures.items():
            # Check for basic consistency
            if not signature.get("args") or not signature.get("return_type"):
                self.validation_errors.append({
                    "type": "signature_incomplete",
                    "function": function_name,
                    "message": f"Function {function_name} has incomplete signature"
                })

    def _validate_tensor_compatibility_all(self) -> None:
        """Validate all tensor requirements against current configuration."""
        for tensor_name, requirements in self.tensor_requirements.items():
            # This is a placeholder - in practice, this would validate against actual runtime tensors
            # For now, we just check the requirements are well-formed
            if not all(key in requirements for key in self.REQUIRED_TENSOR_FIELDS):
                self.validation_errors.append({
                    "type": "tensor_requirements_incomplete",
                    "tensor": tensor_name,
                    "message": f"Tensor {tensor_name} has incomplete requirements"
                })

    def _check_platform_specific_requirements(self) -> None:
        """Check platform-specific validation requirements."""
        platform = self.get_platform()

        if platform == "metal":
            # Check Metal-specific requirements
            for function_name, signature in self.extracted_signatures.items():
                if self.is_kernel_function(function_name):
                    if "thread_group_size" not in signature.get("attributes", {}):
                        self.validation_warnings.append({
                            "type": "performance_warning",
                            "function": function_name,
                            "message": f"Metal kernel {function_name} missing thread group size optimization"
                        })

    def adapt_tensor(self, tensor_name: str, runtime_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically adapt tensor format to meet requirements.

        Args:
            tensor_name: Name of tensor to adapt
            runtime_spec: Current runtime tensor specification

        Returns:
            Adapted tensor specification
        """
        if tensor_name not in self.tensor_requirements:
            return runtime_spec.copy()

        required_spec = self.tensor_requirements[tensor_name]
        adapted_spec = runtime_spec.copy()

        # Adapt dtype if allowed
        if self.allows_dtype_conversion():
            adapted_spec["dtype"] = required_spec["dtype"]

        # Adapt layout if auto-adaptation is enabled
        if self.validation_rules.get("auto_adapt_tensors", False):
            adapted_spec["layout"] = required_spec["layout"]

        return adapted_spec

    def get_plugin(self):
        """
        Get the associated PluginDefinition.

        Returns:
            PluginDefinition instance

        Note:
            This method imports PluginDefinition dynamically to avoid circular imports.
        """
        from debug_framework.models.plugin_definition import PluginDefinition
        return PluginDefinition.load(self.plugin_id)

    def save(self) -> int:
        """
        Save interface validator to database.

        Returns:
            Database ID of the saved validator
        """
        db_manager = database_manager.DatabaseManager()

        # Prepare data for database insertion
        validator_data = {
            "plugin_id": self.plugin_id,
            "extracted_signatures": json.dumps(self.extracted_signatures),
            "tensor_requirements": json.dumps(self.tensor_requirements),
            "validation_rules": json.dumps(self.validation_rules),
            "validation_status": self._validation_status,
            "validation_errors": json.dumps(self.validation_errors),
            "validation_warnings": json.dumps(self.validation_warnings),
            "last_validation_at": self.last_validation_at.isoformat() if self.last_validation_at else None
        }

        if self.id is None:
            # Insert new validator
            self.id = db_manager.insert_interface_validator(validator_data)
        else:
            # Update existing validator
            db_manager.update_interface_validator(self.id, validator_data)

        return self.id

    @classmethod
    def load(cls, validator_id: int) -> 'InterfaceValidator':
        """
        Load interface validator from database by ID.

        Args:
            validator_id: Database ID of the validator

        Returns:
            InterfaceValidator instance loaded from database

        Raises:
            ValueError: If validator not found in database
        """
        db_manager = database_manager.DatabaseManager()
        validator_data = db_manager.get_interface_validator(validator_id)

        if not validator_data:
            raise ValueError(f"Interface validator with ID {validator_id} not found")

        return cls.from_dict(validator_data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert InterfaceValidator to dictionary representation."""
        return {
            "id": self.id,
            "plugin_id": self.plugin_id,
            "extracted_signatures": self.extracted_signatures,
            "tensor_requirements": self.tensor_requirements,
            "validation_rules": self.validation_rules,
            "validation_status": self._validation_status,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "last_validation_at": self.last_validation_at.isoformat() if self.last_validation_at else None
        }

    def to_json(self) -> str:
        """Convert InterfaceValidator to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterfaceValidator':
        """Create InterfaceValidator from dictionary representation."""
        # Parse JSON fields
        extracted_signatures = json.loads(data.get("extracted_signatures", "{}")) if isinstance(data.get("extracted_signatures"), str) else data.get("extracted_signatures", {})
        tensor_requirements = json.loads(data.get("tensor_requirements", "{}")) if isinstance(data.get("tensor_requirements"), str) else data.get("tensor_requirements", {})
        validation_rules = json.loads(data.get("validation_rules", "{}")) if isinstance(data.get("validation_rules"), str) else data.get("validation_rules", {})
        validation_errors = json.loads(data.get("validation_errors", "[]")) if isinstance(data.get("validation_errors"), str) else data.get("validation_errors", [])
        validation_warnings = json.loads(data.get("validation_warnings", "[]")) if isinstance(data.get("validation_warnings"), str) else data.get("validation_warnings", [])

        # Parse timestamp field
        last_validation_at = None
        if data.get("last_validation_at"):
            if isinstance(data["last_validation_at"], str):
                try:
                    last_validation_at = datetime.fromisoformat(data["last_validation_at"].replace('Z', '+00:00'))
                except ValueError:
                    pass  # Keep as None if parsing fails
            elif isinstance(data["last_validation_at"], datetime):
                last_validation_at = data["last_validation_at"]

        return cls(
            id=data.get("id"),
            plugin_id=data["plugin_id"],
            extracted_signatures=extracted_signatures,
            tensor_requirements=tensor_requirements,
            validation_rules=validation_rules,
            validation_status=data.get("validation_status", "pending"),
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
            last_validation_at=last_validation_at
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'InterfaceValidator':
        """Create InterfaceValidator from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of InterfaceValidator."""
        return f"InterfaceValidator(id={self.id}, plugin_id={self.plugin_id}, status='{self._validation_status}')"

    def __repr__(self) -> str:
        """Detailed representation of InterfaceValidator."""
        return (f"InterfaceValidator(id={self.id}, plugin_id={self.plugin_id}, "
                f"validation_status='{self._validation_status}', "
                f"signatures={len(self.extracted_signatures)}, "
                f"requirements={len(self.tensor_requirements)})")