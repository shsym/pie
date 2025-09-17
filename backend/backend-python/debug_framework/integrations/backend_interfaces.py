"""
Backend Interface Hierarchy for L4MA Debug Integration

Provides a clean separation of concerns with different backend implementations
for tensor computation comparison and validation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union
import time
import warnings
import numpy as np
from enum import Enum

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from model.l4ma import L4maModel, L4maForCausalLM, L4maAttention, L4maMlp, L4maDecoderLayer
    from config.l4ma import L4maArch
    L4MA_MODEL_AVAILABLE = True
except ImportError:
    L4MA_MODEL_AVAILABLE = False


class BackendType(Enum):
    """Enumeration of supported backend types."""
    L4MA_PYTHON = "l4ma_python"
    METAL = "metal"
    CUDA = "cuda"
    MOCK = "mock"


class TensorComputationResult:
    """Encapsulates results from backend tensor computations."""

    def __init__(
        self,
        output: np.ndarray,
        computation_time: float,
        backend_type: BackendType,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.output = output
        self.computation_time = computation_time
        self.backend_type = backend_type
        self.metadata = metadata or {}
        self.timestamp = time.perf_counter()


class BackendInterface(ABC):
    """
    Abstract base class for all backend implementations.

    Defines the common interface that all backends must implement
    for tensor operations and validation.
    """

    def __init__(self, backend_type: BackendType):
        self.backend_type = backend_type
        self.is_available = False
        self.initialization_time = None
        self.performance_metrics: Dict[str, List[float]] = {
            'attention_times': [],
            'mlp_times': [],
            'embedding_times': [],
            'normalization_times': []
        }
        self._error_count = 0

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the backend and check availability.

        Returns:
            bool: True if backend is successfully initialized
        """
        pass

    @abstractmethod
    def run_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        **kwargs
    ) -> TensorComputationResult:
        """
        Run attention computation.

        Args:
            query: Query tensor [batch_size, seq_len, head_dim]
            key: Key tensor [batch_size, seq_len, head_dim]
            value: Value tensor [batch_size, seq_len, head_dim]
            **kwargs: Additional backend-specific parameters

        Returns:
            TensorComputationResult with attention output
        """
        pass

    @abstractmethod
    def run_mlp(
        self,
        hidden_states: np.ndarray,
        **kwargs
    ) -> TensorComputationResult:
        """
        Run MLP computation.

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            **kwargs: Additional backend-specific parameters

        Returns:
            TensorComputationResult with MLP output
        """
        pass

    @abstractmethod
    def run_embedding(
        self,
        input_ids: np.ndarray,
        **kwargs
    ) -> TensorComputationResult:
        """
        Run embedding lookup.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            **kwargs: Additional backend-specific parameters

        Returns:
            TensorComputationResult with embedding output
        """
        pass

    def run_normalization(
        self,
        hidden_states: np.ndarray,
        **kwargs
    ) -> TensorComputationResult:
        """
        Run normalization computation (optional, can be overridden).

        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            **kwargs: Additional backend-specific parameters

        Returns:
            TensorComputationResult with normalized output
        """
        # Default implementation - just return input (identity)
        start_time = time.perf_counter()
        computation_time = time.perf_counter() - start_time

        return TensorComputationResult(
            output=hidden_states.copy(),
            computation_time=computation_time,
            backend_type=self.backend_type,
            metadata={'operation': 'identity_normalization'}
        )

    def validate_inputs(
        self,
        operation: str,
        *tensors: np.ndarray,
        **kwargs
    ) -> Tuple[bool, str]:
        """
        Validate input tensors for given operation.

        Args:
            operation: Name of the operation ('attention', 'mlp', 'embedding')
            tensors: Input tensors to validate
            **kwargs: Additional validation parameters

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not tensors:
                return False, "No input tensors provided"

            for i, tensor in enumerate(tensors):
                if not isinstance(tensor, np.ndarray):
                    return False, f"Tensor {i} is not a numpy array"

                if tensor.size == 0:
                    return False, f"Tensor {i} is empty"

                if not np.isfinite(tensor).all():
                    return False, f"Tensor {i} contains non-finite values"

            # Operation-specific validation
            if operation == "attention":
                if len(tensors) < 3:
                    return False, "Attention requires at least 3 tensors (Q, K, V)"

                q, k, v = tensors[:3]
                if q.shape != k.shape or k.shape != v.shape:
                    return False, f"QKV shape mismatch: Q{q.shape}, K{k.shape}, V{v.shape}"

            elif operation == "mlp":
                if len(tensors) < 1:
                    return False, "MLP requires at least 1 input tensor"

            elif operation == "embedding":
                if len(tensors) < 1:
                    return False, "Embedding requires at least 1 input tensor"

                ids = tensors[0]
                if not np.issubdtype(ids.dtype, np.integer):
                    return False, "Embedding input must be integer tensor"

                if ids.min() < 0:
                    return False, "Embedding input contains negative indices"

            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get backend capabilities and status.

        Returns:
            Dictionary with backend information
        """
        return {
            'backend_type': self.backend_type.value,
            'is_available': self.is_available,
            'initialization_time': self.initialization_time,
            'error_count': self._error_count,
            'supported_operations': ['attention', 'mlp', 'embedding', 'normalization'],
            'performance_metrics': {
                op: {
                    'count': len(times),
                    'avg_time': np.mean(times) if times else 0.0,
                    'total_time': sum(times)
                }
                for op, times in self.performance_metrics.items()
            }
        }

    def record_performance(self, operation: str, computation_time: float):
        """Record performance metrics for an operation."""
        metric_key = f"{operation}_times"
        if metric_key in self.performance_metrics:
            self.performance_metrics[metric_key].append(computation_time)

    def increment_error_count(self):
        """Increment error counter."""
        self._error_count += 1

    def cleanup(self):
        """Cleanup backend resources (can be overridden)."""
        pass


class MockBackend(BackendInterface):
    """
    Mock backend implementation for testing purposes.

    Returns predictable random data for validation testing.
    """

    def __init__(self, seed: int = 42):
        super().__init__(BackendType.MOCK)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def initialize(self) -> bool:
        """Initialize mock backend."""
        start_time = time.perf_counter()

        # Mock initialization delay
        time.sleep(0.001)

        self.initialization_time = time.perf_counter() - start_time
        self.is_available = True
        return True

    def run_attention(
        self,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        **kwargs
    ) -> TensorComputationResult:
        """Mock attention computation."""
        start_time = time.perf_counter()

        # Validate inputs
        is_valid, error_msg = self.validate_inputs("attention", query, key, value)
        if not is_valid:
            raise ValueError(f"Attention input validation failed: {error_msg}")

        # Mock computation with deterministic random output
        output = self.rng.rand(*query.shape).astype(np.float32)

        computation_time = time.perf_counter() - start_time
        self.record_performance("attention", computation_time)

        return TensorComputationResult(
            output=output,
            computation_time=computation_time,
            backend_type=self.backend_type,
            metadata={'operation': 'mock_attention', 'seed': self.seed}
        )

    def run_mlp(
        self,
        hidden_states: np.ndarray,
        **kwargs
    ) -> TensorComputationResult:
        """Mock MLP computation."""
        start_time = time.perf_counter()

        # Validate inputs
        is_valid, error_msg = self.validate_inputs("mlp", hidden_states)
        if not is_valid:
            raise ValueError(f"MLP input validation failed: {error_msg}")

        # Mock computation maintaining input shape
        output = self.rng.rand(*hidden_states.shape).astype(np.float32)

        computation_time = time.perf_counter() - start_time
        self.record_performance("mlp", computation_time)

        return TensorComputationResult(
            output=output,
            computation_time=computation_time,
            backend_type=self.backend_type,
            metadata={'operation': 'mock_mlp', 'seed': self.seed}
        )

    def run_embedding(
        self,
        input_ids: np.ndarray,
        **kwargs
    ) -> TensorComputationResult:
        """Mock embedding computation."""
        start_time = time.perf_counter()

        # Validate inputs
        is_valid, error_msg = self.validate_inputs("embedding", input_ids)
        if not is_valid:
            raise ValueError(f"Embedding input validation failed: {error_msg}")

        # Mock embedding output
        hidden_size = kwargs.get('hidden_size', 4096)
        output_shape = input_ids.shape + (hidden_size,)
        output = self.rng.rand(*output_shape).astype(np.float32)

        computation_time = time.perf_counter() - start_time
        self.record_performance("embedding", computation_time)

        return TensorComputationResult(
            output=output,
            computation_time=computation_time,
            backend_type=self.backend_type,
            metadata={'operation': 'mock_embedding', 'hidden_size': hidden_size, 'seed': self.seed}
        )


def get_recommended_backend() -> BackendType:
    """
    Get the recommended backend type based on the current platform and availability.

    Returns:
        BackendType: The recommended backend for the current system
    """
    import sys
    import platform

    # Check if running on Apple Silicon or Intel Mac
    if sys.platform == 'darwin':
        # Check for Apple Silicon
        if platform.machine() == 'arm64':
            return BackendType.METAL
        else:
            # Intel Mac - Metal is still preferred over CPU
            return BackendType.METAL

    # For CUDA systems (Linux/Windows with NVIDIA GPU)
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            return BackendType.CUDA
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    # Default fallback to L4MA Python for CPU
    return BackendType.L4MA_PYTHON


def create_backend(backend_type: BackendType, **kwargs) -> BackendInterface:
    """
    Factory function to create backend instances.

    Args:
        backend_type: Type of backend to create
        **kwargs: Backend-specific initialization parameters

    Returns:
        Initialized backend instance

    Raises:
        ValueError: If backend type is not supported
        RuntimeError: If backend initialization fails
    """
    if backend_type == BackendType.MOCK:
        backend = MockBackend(seed=kwargs.get('seed', 42))
    elif backend_type == BackendType.L4MA_PYTHON:
        from .l4ma_python_backend import L4MAPythonBackend
        backend = L4MAPythonBackend(
            l4ma_model_reference=kwargs.get('l4ma_model_reference'),
            device=kwargs.get('device', 'cpu')
        )
    elif backend_type == BackendType.METAL:
        from .metal_backend import MetalBackend
        backend = MetalBackend(
            metal_backend_path=kwargs.get('metal_backend_path')
        )
    elif backend_type == BackendType.CUDA:
        from .cuda_backend import CUDABackend
        backend = CUDABackend(
            cuda_backend_path=kwargs.get('cuda_backend_path')
        )
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")

    # Initialize the backend
    if not backend.initialize():
        raise RuntimeError(f"Failed to initialize {backend_type.value} backend")

    return backend


def create_auto_backend(**kwargs) -> BackendInterface:
    """
    Create a backend automatically based on the current system.

    Args:
        **kwargs: Backend-specific initialization parameters

    Returns:
        Initialized backend instance

    Raises:
        RuntimeError: If no suitable backend can be initialized
    """
    recommended_type = get_recommended_backend()

    # Try the recommended backend first
    try:
        return create_backend(recommended_type, **kwargs)
    except (ValueError, RuntimeError) as e:
        warnings.warn(f"Failed to initialize recommended backend {recommended_type.value}: {e}")

    # Try fallback backends in order of preference
    fallback_order = [BackendType.L4MA_PYTHON, BackendType.MOCK]

    for fallback_type in fallback_order:
        if fallback_type != recommended_type:
            try:
                return create_backend(fallback_type, **kwargs)
            except (ValueError, RuntimeError) as e:
                warnings.warn(f"Failed to initialize fallback backend {fallback_type.value}: {e}")
                continue

    # If all else fails, raise an error
    raise RuntimeError("Could not initialize any backend. Please check your system configuration.")