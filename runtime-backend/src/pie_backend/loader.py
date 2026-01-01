"""
Model loading for PIE backend.

This module handles the messy file I/O, ztensor vs safetensors,
TOML parsing, and weight schema loading.

WeightSchema provides a declarative API for defining how model weights
are loaded, fused, sharded, and transformed.

Example:
    schema = (
        Schema("llama3")
        .define("token_embeds",
            Source("model.embed_tokens.weight")
            .shard("row"))
        .define("layers.*.attn.qkv",
            Source.fuse([
                "model.layers.*.self_attn.q_proj.weight",
                "model.layers.*.self_attn.k_proj.weight",
                "model.layers.*.self_attn.v_proj.weight",
            ], dim=0)
            .shard("column")
            .quantize())
    )

    weights = schema.load(reader, config, num_layers=32)
"""

from __future__ import annotations

import tomllib
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import torch
from tqdm import tqdm
import ztensor
import safetensors

from .quantization import quantize
from .config import RuntimeConfig


# =============================================================================
# WEIGHT SCHEMA CLASSES
# =============================================================================

ReaderFn = Callable[[str], torch.Tensor]


@dataclass
class Source:
    """
    Defines a source tensor or fusion of multiple source tensors.
    
    Use method chaining to apply transforms:
        Source("weight.name").shard("row").quantize()
        Source.fuse([...], dim=0).transform(my_fn, arg1=val1)
        Source("sinks").dtype(torch.float32)
    """
    
    _patterns: list[str]
    _fuse_dim: int | None = None
    _sharding: str | None = None  # None, 'column', or 'row'
    _should_quantize: bool = False
    _expected_shapes: list[tuple[int, ...] | None] | None = None
    _transform_fn: Callable[[list[torch.Tensor], dict[str, Any]], torch.Tensor | dict] | None = None
    _transform_kwargs: dict[str, Any] | None = None
    _dtype: torch.dtype | None = None
    _gather_only: bool = False  # For transforms that need multiple tensors without concatenation
    
    def __init__(self, pattern: str):
        """Create a source from a single tensor pattern."""
        self._patterns = [pattern]
        self._fuse_dim = None
        self._sharding = None
        self._should_quantize = False
        self._expected_shapes = None
        self._transform_fn = None
        self._transform_kwargs = None
        self._dtype = None
        self._gather_only = False
    
    @classmethod
    def fuse(cls, patterns: list[str], dim: int = 0) -> Source:
        """
        Create a fused source from multiple tensor patterns.
        
        Args:
            patterns: List of tensor name patterns to fuse
            dim: Dimension along which to concatenate (default: 0)
                 Use dim=None with gather_only=True to gather without concat.
        
        Returns:
            A new Source configured for fusion
        """
        source = cls.__new__(cls)
        source._patterns = patterns
        source._fuse_dim = dim
        source._sharding = None
        source._should_quantize = False
        source._expected_shapes = None
        source._transform_fn = None
        source._transform_kwargs = None
        source._dtype = None
        source._gather_only = False
        return source
    
    @classmethod
    def gather(cls, patterns: list[str]) -> Source:
        """
        Gather multiple tensors without fusing. Used with .transform() for
        operations that need multiple input tensors.
        
        Args:
            patterns: List of tensor name patterns to gather
        
        Returns:
            A new Source configured for gathering (no concatenation)
        """
        source = cls.__new__(cls)
        source._patterns = patterns
        source._fuse_dim = None
        source._sharding = None
        source._should_quantize = False
        source._expected_shapes = None
        source._transform_fn = None
        source._transform_kwargs = None
        source._dtype = None
        source._gather_only = True
        return source
    
    def shard(self, strategy: str) -> Source:
        """
        Set sharding strategy: 'column', 'row', or 'interleaved_column'.
        
        'interleaved_column': Shards input tensors individually along dim 0 BEFORE fusion.
                              Useful for fused weights like QKV where head alignment matters.
        """
        if strategy not in ('column', 'row', 'interleaved_column'):
            raise ValueError(f"Invalid sharding strategy: {strategy}. Use 'column', 'row', or 'interleaved_column'.")
        self._sharding = strategy
        return self
    
    def quantize(self) -> Source:
        """Enable quantization for this weight."""
        self._should_quantize = True
        return self
    
    def transform(
        self,
        fn: Callable[[list[torch.Tensor], dict[str, Any]], torch.Tensor | dict],
        **kwargs: Any,
    ) -> Source:
        """
        Apply a custom transformation function to loaded tensors.
        
        The function receives:
            - tensors: List of loaded tensors (from patterns)
            - kwargs: Additional keyword arguments passed here + 'device'
        
        It should return:
            - A single tensor, OR
            - A dict with 'output_type' key selecting which tensor to return
        
        Example:
            Source.gather(["blocks", "scales", "bias"])
                .transform(prepare_moe_weights, output_type="weights")
        """
        self._transform_fn = fn  
        self._transform_kwargs = kwargs
        return self
    
    def dtype(self, dt: torch.dtype) -> Source:
        """Convert tensor to specified dtype."""
        self._dtype = dt
        return self
    
    def expect_shapes(self, shapes: list[tuple[int, ...] | None]) -> Source:
        """Set expected shapes for validation (optional)."""
        self._expected_shapes = shapes
        return self
    
    @property
    def is_fused(self) -> bool:
        return self._fuse_dim is not None and not self._gather_only
    
    @property
    def is_gathered(self) -> bool:
        return self._gather_only
    
    @property
    def has_transform(self) -> bool:
        return self._transform_fn is not None
    
    @property
    def patterns(self) -> list[str]:
        return self._patterns
    
    @property
    def fuse_dim(self) -> int | None:
        return self._fuse_dim
    
    @property
    def sharding(self) -> str | None:
        return self._sharding
    
    @property
    def should_quantize(self) -> bool:
        return self._should_quantize


@dataclass
class Definition:
    """A named weight definition: logical name -> Source with transforms."""
    
    name: str  # Logical name, may contain '*' for layer patterns
    source: Source
    
    def has_layer_pattern(self) -> bool:
        """Check if this definition uses layer patterns (*)."""
        return "*" in self.name
    
    def expand_for_layer(self, layer_idx: int) -> str:
        """Expand the logical name for a specific layer."""
        return self.name.replace("*", str(layer_idx))
    
    def expand_source_for_layer(self, layer_idx: int) -> list[str]:
        """Expand source patterns for a specific layer."""
        return [p.replace("*", str(layer_idx)) for p in self.source.patterns]


class WeightStore:
    """Container for loaded weights, accessible by logical name."""
    
    def __init__(self):
        self._weights: dict[str, torch.Tensor] = {}
    
    def put(self, name: str, tensor: torch.Tensor) -> None:
        """Store a tensor by logical name."""
        self._weights[name] = tensor
    
    def get(self, name: str) -> torch.Tensor:
        """Retrieve a tensor by logical name."""
        if name not in self._weights:
            raise KeyError(f"Weight '{name}' not found in store")
        return self._weights[name]
    
    def get_list(self, pattern: str, count: int) -> list[torch.Tensor]:
        """
        Retrieve a list of tensors matching a pattern.
        
        Args:
            pattern: Pattern with '*' placeholder (e.g., "layers.*.proj_qkv")
            count: Number of items to retrieve
        
        Returns:
            List of tensors for indices 0..count-1
        """
        return [self.get(pattern.replace("*", str(i))) for i in range(count)]
    
    def keys(self) -> list[str]:
        """List all stored weight names."""
        return list(self._weights.keys())
    
    def __contains__(self, name: str) -> bool:
        return name in self._weights
    
    def __len__(self) -> int:
        return len(self._weights)


class Schema:
    """
    Schema definition for weight loading.
    
    Use method chaining to define weights:
        schema = Schema("model_name").define("name", Source(...))
    """
    
    def __init__(self, name: str):
        self.name = name
        self._definitions: list[Definition] = []
    
    def define(self, name: str, source: Source) -> Schema:
        """
        Define a logical weight with its source.
        
        Args:
            name: Logical name (may contain '*' for per-layer weights)
            source: Source configuration
        
        Returns:
            self for method chaining
        """
        self._definitions.append(Definition(name=name, source=source))
        return self
    
    
    def load(
        self,
        reader: ReaderFn,
        config: "RuntimeConfig",
        num_layers: int = 0,
    ) -> WeightStore:
        """
        Load all weights according to the schema.
        
        Args:
            reader: Function to read tensors by name
            config: Runtime configuration (device, sharding, quantization)
            num_layers: Number of layers (for expanding '*' patterns)
        
        Returns:
            WeightStore with all loaded weights
        """
        store = WeightStore()
        
        for defn in self._definitions:
            if defn.has_layer_pattern():
                # Expand for each layer
                for layer_idx in range(num_layers):
                    logical_name = defn.expand_for_layer(layer_idx)
                    physical_names = defn.expand_source_for_layer(layer_idx)
                    tensor = self._load_single(
                        reader, config, defn.source, physical_names
                    )
                    store.put(logical_name, tensor)
            else:
                # Single tensor (not per-layer)
                tensor = self._load_single(
                    reader, config, defn.source, defn.source.patterns
                )
                store.put(defn.name, tensor)
        #print("Loaded", len(store), flush=True)
        return store
    
    
    def _load_single(
        self,
        reader: ReaderFn,
        config: "RuntimeConfig",
        source: Source,
        physical_names: list[str],
    ) -> torch.Tensor:
        #print("Loading", physical_names)
        """Load a single (possibly fused/gathered) tensor with transforms applied."""
        # Read tensors
        tensors = [reader(name) for name in physical_names]
        
        # Apply custom transform if specified
        if source.has_transform:
            # Build kwargs for transform function
            transform_kwargs = dict(source._transform_kwargs or {})
            transform_kwargs["device"] = str(config.device)
            
            # Call the transform function
            result = source._transform_fn(tensors, transform_kwargs)  # type: ignore[misc]
            
            # Handle dict result (select output by type)
            if isinstance(result, dict):
                output_type = transform_kwargs.get("output_type")
                if output_type is None:
                    raise ValueError(
                        "Transform returned dict but no 'output_type' specified"
                    )
                tensor = result[output_type]
            else:
                tensor = result
        # Apply interleaved sharding BEFORE fusion (if requested)
        # This is CRITICAL for fused QKV weights where naive chunking breaks head alignment
        if config.world_size > 1 and source.sharding == 'interleaved_column':
            # Shard each source source tensor individually along dim=0 (Column Parallel)
            sharded_tensors = []
            # print(f"DEBUG: Interleaved sharding for {physical_names}", flush=True)
            for i, t in enumerate(tensors):
                if t.shape[0] % config.world_size != 0:
                     raise ValueError(
                        f"Cannot interleaved-shard tensor {t.shape}: dim 0 not divisible by world_size={config.world_size}"
                    )
                # Ensure we have a clean copy in memory (avoid ztensor/mmap issues with views)
                t_clone = t.clone()
                chunk = torch.chunk(t_clone, config.world_size, dim=0)[config.rank]
                # Check chunk validity
                # print(f"  Shard {i}: {t.shape} -> {chunk.shape}", flush=True)
                sharded_tensors.append(chunk)
            tensors = sharded_tensors

        # Fuse if needed (concatenation)
        if source.is_fused:
            tensor = torch.cat(tensors, dim=source.fuse_dim)
        # Gathered but no transform - just take first tensor (unusual case)
        elif source.is_gathered:
            raise ValueError(
                "Source.gather() should be used with .transform(). "
                "For single tensor, use Source(pattern) instead."
            )
        else:
            tensor = tensors[0]
        
        # Apply dtype conversion
        if source._dtype is not None:
            tensor = tensor.to(source._dtype)
        
        # Apply sharding (only if world_size > 1)
        # interleaved_column is handled above, so we skip it here
        if config.world_size > 1 and source.sharding is not None and source.sharding != 'interleaved_column':
            dim = 1 if source.sharding == 'row' else 0
            
            # Validate dimension is divisible by world_size
            if tensor.shape[dim] % config.world_size != 0:
                raise ValueError(
                    f"Cannot shard tensor of shape {tuple(tensor.shape)} along dim={dim}: "
                    f"dimension size {tensor.shape[dim]} is not divisible by world_size={config.world_size}"
                )
            
            tensor = torch.chunk(tensor.contiguous(), config.world_size, dim=dim)[config.rank]
        
        # Apply quantization (lazy import to avoid dependency issues)
        if source.should_quantize and config.quantization is not None:
            
            tensor = quantize(tensor, config.quantization)
        
        # Move to device
        tensor = tensor.to(config.device)
        
        return tensor


# =============================================================================
# MODEL LOADER
# =============================================================================


class ModelLoader:
    """
    Handles model loading, TOML parsing, and weight I/O.
    
    This separates the loading concerns from the runtime orchestration.
    The loader returns a WeightStore and model config - the runtime
    is responsible for creating the ForwardPass and KV cache.
    """

    def __init__(self, config: "RuntimeConfig"):
        """
        Initialize the model loader.
        
        Args:
            config: Runtime configuration
        """
        self.config = config
        self.info: dict = {}

    def load(self) -> tuple[WeightStore, dict, dict]:
        """
        Load the model weights and return components.
        
        Returns:
            Tuple of (weights, normalized_arch, model_info)
        """
        # Load model info from TOML
        self.info = self._load_toml()
        arch_type = self.info["architecture"]["type"]

        # Normalize architecture fields
        normalized_arch = self._normalize_arch_fields(self.info["architecture"])

        # Get schema for architecture type
        match arch_type:
            case "llama3" | "l4ma":
                from .model import llama3
                schema = llama3.LLAMA3_SCHEMA
                num_layers = int(normalized_arch["num_layers"])

            case "qwen2":
                from .model import qwen2
                schema = qwen2.QWEN2_SCHEMA
                num_layers = int(normalized_arch["num_layers"])

            case "qwen3":
                from .model import qwen3
                schema = qwen3.QWEN3_SCHEMA
                num_layers = int(normalized_arch["num_layers"])
            case "gpt_oss" | "gptoss":
                from .model import gpt_oss
                # GPT-OSS uses a factory function because MoE transforms need dimensions
                model_config = gpt_oss.ModelConfig.from_dict(normalized_arch)
                schema = gpt_oss.create_gpt_oss_schema(model_config)
                num_layers = model_config.num_layers
            case _:
                raise ValueError(f"Unsupported architecture type: {arch_type}")

        # Load weights using schema
        weights = self.load_weights(schema, num_layers)

        return weights, normalized_arch, self.info

    def load_weights(self, schema: Schema, num_layers: int) -> WeightStore:
        """
        Load weights using the provided schema.
        
        Args:
            schema: Weight schema defining the tensor mapping
            num_layers: Number of layers for expanding '*' patterns
            
        Returns:
            WeightStore with all loaded weights
        """
        # Determine path to model weight files
        model_dir = Path(self.config.cache_dir) / "models" / self.config.model


        # Load weights
        with ExitStack() as stack:
            readers: dict[str, object] = {}

            # Build tensor name -> reader mapping
            param_files = self.info.get("parameters", [])
            for param_file in tqdm(
                param_files, desc="Scanning tensor files", unit="files"
            ):
                param_path = model_dir / param_file

                if param_path.suffix == ".zt":
                    f = stack.enter_context(ztensor.Reader(str(param_path)))
                    names = f.get_tensor_names()
                elif param_path.suffix == ".safetensors":
                    f = stack.enter_context(
                        safetensors.safe_open(
                            str(param_path), framework="pt", device="cpu"
                        )
                    )
                    names = list(f.keys())
                else:
                    continue

                for n in names:
                    readers[n] = f

            def reader(
                name: str, *, expected_shape: tuple[int, ...] | None = None
            ) -> torch.Tensor:
                f = readers.get(name)
                if f is None:
                    raise KeyError(f"Tensor '{name}' not found")

                # ztensor vs safetensors
                t = (
                    f.read_tensor(name, to="torch")  # ztensor
                    if hasattr(f, "read_tensor")
                    else f.get_tensor(name)  # safetensors
                )

                if expected_shape is not None and tuple(t.shape) != tuple(
                    expected_shape
                ):
                    raise ValueError(
                        f"{name} has shape {tuple(t.shape)}, expected {tuple(expected_shape)}"
                    )
                return t

            # Load weights using schema
            weights = schema.load(
                reader=reader,
                config=self.config,
                num_layers=num_layers,
            )

        return weights

    def _load_toml(self) -> dict:
        """
        Load model metadata from TOML file.
        
        Returns:
            Parsed TOML dictionary
        """
        # Try model subdirectory first
        model_info_path = (
            Path(self.config.cache_dir) / "models" / f"{self.config.model}.toml"
        )

        if not model_info_path.exists():
            raise ValueError(
                f'Metadata file for model "{self.config.model}" not found. '
                f"Expected: {self.config.cache_dir}/models/{self.config.model}.toml"
            )

        with open(model_info_path, "rb") as f:
            return tomllib.load(f)

    def _normalize_arch_fields(self, arch: dict) -> dict:
        """
        Normalize YAML/TOML field names to match ModelConfig.from_dict expectations.
        
        Args:
            arch: Raw architecture dictionary from TOML
            
        Returns:
            Normalized architecture dictionary
        """
        normalized = dict(arch)

        # Map YAML field names -> expected names
        field_map = {
            "head_dim": "head_size",
            "num_heads": "num_query_heads",
            "num_heads_kv": "num_key_value_heads",
            "high_freq_factor": "high_frequency_factor",
            "low_freq_factor": "low_frequency_factor",
        }

        # Normalize top-level fields
        for old, new in field_map.items():
            if old in normalized and new not in normalized:
                normalized[new] = normalized.pop(old)

        # Normalize rope subfields
        if "rope" in normalized:
            rope = dict(normalized["rope"])
            for old, new in field_map.items():
                if old in rope and new not in rope:
                    rope[new] = rope.pop(old)
            # Add rope.factor default if missing
            if "factor" not in rope:
                rope["factor"] = 1.0
            normalized["rope"] = rope

        # Add missing fields with defaults
        if "rms_norm_eps" not in normalized:
            normalized["rms_norm_eps"] = 1e-5

        # Get vocab_size from tokenizer section if not in architecture
        if "vocab_size" not in normalized and "tokenizer" in self.info:
            tokenizer = self.info["tokenizer"]
            if "vocab_size" in tokenizer:
                normalized["vocab_size"] = tokenizer["vocab_size"]

        return normalized
