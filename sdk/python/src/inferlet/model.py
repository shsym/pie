"""
Model and Queue wrapper classes for inferlet-core bindings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from wit_world.imports import runtime as _runtime
from wit_world.imports import inferlet_core_common as _common
from wit_world.imports import forward as _forward
from .async_utils import await_future

if TYPE_CHECKING:
    from .tokenizer import Tokenizer
    from .forward import ForwardPass
    from .messaging import Blob
    from .context import Context


class Queue:
    """
    Command queue for a specific model instance.
    Manages execution of commands and resource allocation.
    """

    def __init__(self, inner: _common.Queue, service_id: int) -> None:
        self._inner = inner
        self._service_id = service_id

    @property
    def service_id(self) -> int:
        """The service ID for the queue."""
        return self._service_id

    def synchronize(self) -> bool:
        """
        Begins a synchronization process for the queue.
        Returns True if synchronization was successful.
        """
        result = self._inner.synchronize()
        return await_future(result, "synchronize result was None")

    def set_priority(self, priority: _common.Priority) -> None:
        """Change the queue's priority."""
        self._inner.set_priority(priority)

    def debug_query(self, query: str) -> str:
        """Executes a debug command on the queue and returns the result."""
        result = self._inner.debug_query(query)
        return await_future(result, "debug_query result was None")

    # Resource management
    def allocate_resources(self, resource_type: int, count: int) -> list[int]:
        """Allocate resources of the specified type."""
        return list(_common.allocate_resources(self._inner, resource_type, count))

    def deallocate_resources(self, resource_type: int, ptrs: list[int]) -> None:
        """Deallocate resources of the specified type."""
        _common.deallocate_resources(self._inner, resource_type, ptrs)

    def export_resources(self, resource_type: int, ptrs: list[int], name: str) -> None:
        """Export resources with a name for later import."""
        _common.export_resources(self._inner, resource_type, ptrs, name)

    def import_resources(self, resource_type: int, name: str) -> list[int]:
        """Import resources by name."""
        return list(_common.import_resources(self._inner, resource_type, name))

    def get_all_exported_resources(self, resource_type: int) -> list[tuple[str, int]]:
        """Get all exported resources of a type."""
        return list(_common.get_all_exported_resources(self._inner, resource_type))

    def release_exported_resources(self, resource_type: int, name: str) -> None:
        """Release exported resources by name."""
        _common.release_exported_resources(self._inner, resource_type, name)

    # KV Page convenience methods (resource_type=0 is KvPage)
    KV_PAGE_TYPE = 0

    def allocate_kv_page(self) -> int:
        """Allocate a single KV page pointer."""
        ptrs = self.allocate_resources(self.KV_PAGE_TYPE, 1)
        return ptrs[0]

    def allocate_kv_pages(self, count: int) -> list[int]:
        """Allocate multiple KV page pointers."""
        return self.allocate_resources(self.KV_PAGE_TYPE, count)

    def deallocate_kv_page(self, ptr: int) -> None:
        """Deallocate a single KV page pointer."""
        self.deallocate_resources(self.KV_PAGE_TYPE, [ptr])

    def deallocate_kv_pages(self, ptrs: list[int]) -> None:
        """Deallocate multiple KV page pointers."""
        self.deallocate_resources(self.KV_PAGE_TYPE, ptrs)

    # KV Page export/import
    def export_kv_pages(self, ptrs: list[int], name: str) -> None:
        """Export KV pages with a name for later import."""
        self.export_resources(self.KV_PAGE_TYPE, ptrs, name)

    def import_kv_pages(self, name: str) -> list[int]:
        """Import KV pages by name."""
        return self.import_resources(self.KV_PAGE_TYPE, name)

    def get_all_exported_kv_pages(self) -> list[tuple[str, int]]:
        """Get all exported KV pages."""
        return self.get_all_exported_resources(self.KV_PAGE_TYPE)

    def release_exported_kv_pages(self, name: str) -> None:
        """Release exported KV pages by name."""
        self.release_exported_resources(self.KV_PAGE_TYPE, name)

    # Embedding pointer convenience methods (resource_type=1 is Embed)
    EMBED_TYPE = 1

    def allocate_embed_ptr(self) -> int:
        """Allocate a single embedding pointer."""
        ptrs = self.allocate_resources(self.EMBED_TYPE, 1)
        return ptrs[0]

    def allocate_embed_ptrs(self, count: int) -> list[int]:
        """Allocate multiple embedding pointers."""
        return self.allocate_resources(self.EMBED_TYPE, count)

    def deallocate_embed_ptr(self, ptr: int) -> None:
        """Deallocate a single embedding pointer."""
        self.deallocate_resources(self.EMBED_TYPE, [ptr])

    def deallocate_embed_ptrs(self, ptrs: list[int]) -> None:
        """Deallocate multiple embedding pointers."""
        self.deallocate_resources(self.EMBED_TYPE, ptrs)

    # Embedding pointer export/import
    def export_embed_ptrs(self, ptrs: list[int], name: str) -> None:
        """Export embedding pointers with a name for later import."""
        self.export_resources(self.EMBED_TYPE, ptrs, name)

    def import_embed_ptrs(self, name: str) -> list[int]:
        """Import embedding pointers by name."""
        return self.import_resources(self.EMBED_TYPE, name)

    def get_all_exported_embeds(self) -> list[tuple[str, int]]:
        """Get all exported embeddings."""
        return self.get_all_exported_resources(self.EMBED_TYPE)

    def release_exported_embeds(self, name: str) -> None:
        """Release exported embeddings by name."""
        self.release_exported_resources(self.EMBED_TYPE, name)

    # Adapter management (resource_type=2 is Adapter)
    ADAPTER_TYPE = 2

    def allocate_adapter(self) -> int:
        """Allocate a single adapter pointer."""
        ptrs = self.allocate_resources(self.ADAPTER_TYPE, 1)
        return ptrs[0]

    def deallocate_adapter(self, ptr: int) -> None:
        """Deallocate an adapter pointer."""
        self.deallocate_resources(self.ADAPTER_TYPE, [ptr])

    def export_adapter(self, ptr: int, name: str) -> None:
        """Export an adapter for later import."""
        self.export_resources(self.ADAPTER_TYPE, [ptr], name)

    def import_adapter(self, name: str) -> int:
        """Import an adapter by name."""
        ptrs = self.import_resources(self.ADAPTER_TYPE, name)
        return ptrs[0]

    def get_all_exported_adapters(self) -> list[str]:
        """Get all exported adapter names."""
        return [name for name, _ in self.get_all_exported_resources(self.ADAPTER_TYPE)]

    def release_exported_adapter(self, name: str) -> None:
        """Release an exported adapter by name."""
        self.release_exported_resources(self.ADAPTER_TYPE, name)

    def upload_adapter(self, adapter_ptr: int, name: str, blob: "Blob") -> None:
        """
        Upload adapter weights.

        Args:
            adapter_ptr: Pointer to the adapter resource
            name: Name of the adapter layer (e.g., 'lora_A', 'lora_B')
            blob: Binary blob containing the adapter weights
        """
        from wit_world.imports import inferlet_adapter_common as _adapter

        _adapter.upload_adapter(self._inner, adapter_ptr, name, blob._inner)

    def download_adapter(self, adapter_ptr: int, name: str) -> "Blob":
        """
        Download adapter weights.

        Args:
            adapter_ptr: Pointer to the adapter resource
            name: Name of the adapter layer (e.g., 'lora_A', 'lora_B')

        Returns:
            Binary blob containing the adapter weights
        """
        from wit_world.imports import inferlet_adapter_common as _adapter
        from .messaging import Blob

        future = _adapter.download_adapter(self._inner, adapter_ptr, name)
        blob_inner = await_future(future, "download_adapter result was None")
        return Blob._from_inner(blob_inner)

    # Zero-Order Evolution methods

    def initialize_adapter(
        self,
        adapter_ptr: int,
        rank: int,
        alpha: float,
        population_size: int,
        mu_fraction: float,
        initial_sigma: float,
    ) -> None:
        """
        Initialize an adapter for zero-order evolution.

        Args:
            adapter_ptr: Pointer to the adapter resource
            rank: LoRA rank
            alpha: LoRA alpha scaling factor
            population_size: Number of candidates per generation
            mu_fraction: Fraction of top performers to keep
            initial_sigma: Initial perturbation standard deviation
        """
        from wit_world.imports import inferlet_zo_evolve as _zo

        _zo.initialize_adapter(
            self._inner,
            adapter_ptr,
            rank,
            alpha,
            population_size,
            mu_fraction,
            initial_sigma,
        )

    def update_adapter(
        self,
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ) -> None:
        """
        Update adapter weights based on evolution scores.

        Args:
            adapter_ptr: Pointer to the adapter resource
            scores: Fitness scores for each candidate
            seeds: Random seeds used for each candidate's perturbation
            max_sigma: Maximum allowed sigma value
        """
        from wit_world.imports import inferlet_zo_evolve as _zo

        _zo.update_adapter(self._inner, adapter_ptr, scores, seeds, max_sigma)

    # ForwardPass creation
    def create_forward_pass(self) -> "ForwardPass":
        """Create a new forward pass for executing model inference."""
        from .forward import ForwardPass

        inner = _forward.create_forward_pass(self._inner)
        return ForwardPass(inner, self)

    def __enter__(self) -> "Queue":
        return self

    def __exit__(self, *args: object) -> None:
        pass  # Resources managed by WASM runtime


class Model:
    """
    Represents a specific model instance.
    Provides access to model metadata and queue creation.
    """

    def __init__(self, inner: _common.Model) -> None:
        self._inner = inner
        self._tokenizer: Tokenizer | None = None
        self._eos_tokens_cache: list[list[int]] | None = None

    @property
    def name(self) -> str:
        """The model's name (e.g. 'llama-3.1-8b-instruct')."""
        return self._inner.get_name()

    @property
    def traits(self) -> list[str]:
        """The full set of model traits."""
        return list(self._inner.get_traits())

    def has_traits(self, required_traits: list[str]) -> bool:
        """Checks if the model has all the specified traits."""
        available = set(self.traits)
        return all(trait in available for trait in required_traits)

    @property
    def description(self) -> str:
        """Human-readable description of the model."""
        return self._inner.get_description()

    @property
    def prompt_template(self) -> str:
        """The prompt formatting template in Jinja format."""
        return self._inner.get_prompt_template()

    @property
    def stop_tokens(self) -> list[str]:
        """The stop tokens for the model."""
        return list(self._inner.get_stop_tokens())

    @property
    def eos_tokens(self) -> list[list[int]]:
        """
        The EOS (end-of-sequence) tokens as tokenized arrays (lazy cached).

        Returns:
            List of tokenized stop sequences, where each sequence is a list
            of token IDs representing one stop token string.
        """
        if self._eos_tokens_cache is None:
            self._eos_tokens_cache = [
                self.tokenizer.encode(stop_token) for stop_token in self.stop_tokens
            ]
        return self._eos_tokens_cache

    @property
    def service_id(self) -> int:
        """The service ID for the model."""
        return self._inner.get_service_id()

    @property
    def kv_page_size(self) -> int:
        """The size of a KV page for this model."""
        return self._inner.get_kv_page_size()

    @property
    def tokenizer(self) -> "Tokenizer":
        """The tokenizer for this model (lazy cached)."""
        if self._tokenizer is None:
            from .tokenizer import Tokenizer
            from wit_world.imports import tokenize as _tokenize

            inner = _tokenize.get_tokenizer(self._inner)
            self._tokenizer = Tokenizer(inner)
        return self._tokenizer

    def create_queue(self) -> Queue:
        """Create a new command queue for this model."""
        queue_resource = self._inner.create_queue()
        return Queue(queue_resource, self.service_id)

    def create_context(self) -> "Context":
        """
        Create a new context for this model.

        This is a convenience method equivalent to `Context(model)`.

        Returns:
            A new Context instance for this model.
        """
        from .context import Context

        return Context(self)


def get_model(name: str) -> Model | None:
    """
    Retrieve a model by its name.
    Returns None if no model with the specified name is found.
    """
    model_resource = _runtime.get_model(name)
    if model_resource is None:
        return None
    return Model(model_resource)


def get_all_models() -> list[str]:
    """Get a list of all available model names."""
    return list(_runtime.get_all_models())


def get_auto_model() -> Model:
    """
    Get the first available model automatically.
    Raises ValueError if no models are available.
    """
    models = get_all_models()
    if not models:
        raise ValueError("No models available")
    model = get_model(models[0])
    if model is None:
        raise ValueError(f"Model {models[0]} not found")
    return model


def get_all_models_with_traits(traits: list[str]) -> list[str]:
    """Get names of models that have all specified traits."""
    return list(_runtime.get_all_models_with_traits(traits))
