"""Tests for RpcEndpoint class."""

import pytest
from pydantic import BaseModel

from pycrust import RpcEndpoint, ValidationError


class AddArgs(BaseModel):
    a: int
    b: int


class UserRequest(BaseModel):
    name: str
    age: int


def test_endpoint_creation():
    """Test creating an RpcEndpoint."""
    endpoint = RpcEndpoint("test_service")
    assert endpoint.service_name == "test_service"
    assert len(endpoint._methods) == 0


def test_register_simple_method():
    """Test registering a simple method without validation."""
    endpoint = RpcEndpoint("test_service")

    @endpoint.register()
    def ping() -> str:
        return "pong"

    assert "ping" in endpoint._methods
    assert endpoint._schemas["ping"] is None

    # Test dispatch
    result = endpoint._dispatch("ping", {})
    assert result == "pong"


def test_register_method_with_args():
    """Test registering a method with arguments."""
    endpoint = RpcEndpoint("test_service")

    @endpoint.register()
    def add(a: int, b: int) -> int:
        return a + b

    result = endpoint._dispatch("add", {"a": 5, "b": 3})
    assert result == 8


def test_register_with_custom_name():
    """Test registering a method with a custom name."""
    endpoint = RpcEndpoint("test_service")

    @endpoint.register(name="custom_add")
    def add(a: int, b: int) -> int:
        return a + b

    assert "custom_add" in endpoint._methods
    assert "add" not in endpoint._methods

    result = endpoint._dispatch("custom_add", {"a": 10, "b": 20})
    assert result == 30


def test_register_with_pydantic_validation():
    """Test registering a method with Pydantic validation."""
    endpoint = RpcEndpoint("test_service")

    @endpoint.register(request_model=AddArgs)
    def add(a: int, b: int) -> int:
        return a + b

    assert endpoint._schemas["add"] == AddArgs

    # Valid input
    result = endpoint._dispatch("add", {"a": 100, "b": 200})
    assert result == 300


def test_pydantic_validation_error():
    """Test that Pydantic validation errors are raised properly."""
    endpoint = RpcEndpoint("test_service")

    @endpoint.register(request_model=AddArgs)
    def add(a: int, b: int) -> int:
        return a + b

    # Invalid input (string instead of int)
    with pytest.raises(ValueError, match="Validation error"):
        endpoint._dispatch("add", {"a": "not_an_int", "b": 5})


def test_method_not_found():
    """Test that calling a non-existent method raises an error."""
    endpoint = RpcEndpoint("test_service")

    with pytest.raises(ValueError, match="Method not found"):
        endpoint._dispatch("nonexistent", {})


def test_register_method_programmatically():
    """Test registering a method without the decorator."""
    endpoint = RpcEndpoint("test_service")

    def multiply(x: float, y: float) -> float:
        return x * y

    endpoint.register("mul", multiply)

    assert "mul" in endpoint._methods
    result = endpoint._dispatch("mul", {"x": 2.5, "y": 4.0})
    assert result == 10.0


def test_dispatch_with_list_args():
    """Test dispatch with list arguments."""
    endpoint = RpcEndpoint("test_service")

    @endpoint.register()
    def sum_list(*args: int) -> int:
        return sum(args)

    result = endpoint._dispatch("sum_list", [1, 2, 3, 4, 5])
    assert result == 15


def test_dispatch_with_complex_return():
    """Test dispatch returning complex types."""
    endpoint = RpcEndpoint("test_service")

    @endpoint.register()
    def get_user_info(name: str, age: int) -> dict:
        return {
            "name": name,
            "age": age,
            "is_adult": age >= 18,
            "tags": ["user", "active"],
        }

    result = endpoint._dispatch("get_user_info", {"name": "Alice", "age": 25})
    assert result == {
        "name": "Alice",
        "age": 25,
        "is_adult": True,
        "tags": ["user", "active"],
    }


def test_multiple_methods():
    """Test registering multiple methods."""
    endpoint = RpcEndpoint("test_service")

    @endpoint.register()
    def add(a: int, b: int) -> int:
        return a + b

    @endpoint.register()
    def sub(a: int, b: int) -> int:
        return a - b

    @endpoint.register()
    def mul(a: int, b: int) -> int:
        return a * b

    assert len(endpoint._methods) == 3
    assert endpoint._dispatch("add", {"a": 10, "b": 5}) == 15
    assert endpoint._dispatch("sub", {"a": 10, "b": 5}) == 5
    assert endpoint._dispatch("mul", {"a": 10, "b": 5}) == 50


def test_method_exception_handling():
    """Test that exceptions in methods are propagated."""
    endpoint = RpcEndpoint("test_service")

    @endpoint.register()
    def divide(a: int, b: int) -> float:
        return a / b

    # This should raise ZeroDivisionError
    with pytest.raises(ZeroDivisionError):
        endpoint._dispatch("divide", {"a": 10, "b": 0})


# ==============================================================================
# Real-world complex data structure tests
# ==============================================================================


class HandshakeRequest(BaseModel):
    """Request message for handshake with version information."""

    version: str


class HandshakeResponse(BaseModel):
    """Response message containing model and tokenizer information."""

    version: str
    model_name: str
    model_traits: list[str]
    model_description: str
    prompt_template: str
    prompt_template_type: str
    prompt_stop_tokens: list[str]
    kv_page_size: int
    max_batch_tokens: int
    resources: dict[int, int]
    tokenizer_num_vocab: int
    tokenizer_merge_table: dict[int, bytes]
    tokenizer_special_tokens: dict[str, int]
    tokenizer_split_regex: str
    tokenizer_escape_non_printable: bool


class ForwardPassRequest(BaseModel):
    """Request message for forward pass inference."""

    input_tokens: list[int]
    input_token_positions: list[int]
    input_embed_ptrs: list[int]
    input_embed_positions: list[int]
    adapter: int | None
    adapter_seed: int | None
    mask: list[list[int]]
    kv_page_ptrs: list[int] = []
    kv_page_last_len: int = 0
    output_token_indices: list[int] = []
    output_token_samplers: list[dict] = []
    output_embed_ptrs: list[int] = []
    output_embed_indices: list[int] = []


class ForwardPassResponse(BaseModel):
    """Response message containing inference results."""

    tokens: list[int]
    dists: list[tuple[list[int], list[float]]]


class UpdateAdapterRequest(BaseModel):
    """Request message for adapter updates."""

    adapter_ptr: int
    scores: list[float]
    seeds: list[int]
    max_sigma: float


class UploadAdapterRequest(BaseModel):
    """Request message for adapter upload."""

    adapter_ptr: int
    name: str
    adapter_data: bytes | list[int]


def test_handshake_complex_response():
    """Test handling complex handshake response with nested structures."""
    endpoint = RpcEndpoint("model_service")

    def handshake(version: str) -> dict:
        return {
            "version": "1.0.0",
            "model_name": "llama-7b",
            "model_traits": ["text-generation", "chat", "instruct"],
            "model_description": "A large language model for text generation",
            "prompt_template": "<|user|>{prompt}<|assistant|>",
            "prompt_template_type": "chat",
            "prompt_stop_tokens": ["<|end|>", "<|user|>"],
            "kv_page_size": 256,
            "max_batch_tokens": 4096,
            "resources": {0: 1024, 1: 2048, 2: 512},
            "tokenizer_num_vocab": 32000,
            "tokenizer_merge_table": {0: b"hello", 1: b"world", 2: b"test"},
            "tokenizer_special_tokens": {"<pad>": 0, "<eos>": 1, "<bos>": 2},
            "tokenizer_split_regex": r"\s+",
            "tokenizer_escape_non_printable": True,
        }

    endpoint.register("handshake", handshake, request_model=HandshakeRequest)

    result = endpoint._dispatch("handshake", {"version": "1.0.0"})
    assert result["version"] == "1.0.0"
    assert result["model_name"] == "llama-7b"
    assert len(result["model_traits"]) == 3
    assert result["kv_page_size"] == 256
    assert result["resources"][0] == 1024
    assert result["tokenizer_special_tokens"]["<eos>"] == 1


def test_forward_pass_complex_request():
    """Test handling complex forward pass request with nested lists."""
    endpoint = RpcEndpoint("model_service")

    def forward_pass(
        input_tokens: list[int],
        input_token_positions: list[int],
        input_embed_ptrs: list[int],
        input_embed_positions: list[int],
        adapter: int | None,
        adapter_seed: int | None,
        mask: list[list[int]],
        kv_page_ptrs: list[int],
        kv_page_last_len: int,
        output_token_indices: list[int],
        output_token_samplers: list[dict],
        output_embed_ptrs: list[int],
        output_embed_indices: list[int],
    ) -> dict:
        # Simulate forward pass processing
        num_tokens = len(input_tokens)
        return {
            "tokens": [tok + 1 for tok in input_tokens[:3]],  # Simple transform
            "dists": [
                ([100, 200, 300], [0.5, 0.3, 0.2]),
                ([101, 201, 301], [0.6, 0.25, 0.15]),
            ],
        }

    endpoint.register("forward_pass", forward_pass, request_model=ForwardPassRequest)

    request_data = {
        "input_tokens": [1, 2, 3, 4, 5, 6, 7, 8],
        "input_token_positions": [0, 1, 2, 3, 4, 5, 6, 7],
        "input_embed_ptrs": [0x1000, 0x2000],
        "input_embed_positions": [0, 4],
        "adapter": 42,
        "adapter_seed": 12345,
        "mask": [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
        "kv_page_ptrs": [0x3000, 0x4000, 0x5000],
        "kv_page_last_len": 128,
        "output_token_indices": [5, 6, 7],
        "output_token_samplers": [{"temperature": 0.7}, {"temperature": 0.8}],
        "output_embed_ptrs": [0x6000],
        "output_embed_indices": [7],
    }

    result = endpoint._dispatch("forward_pass", request_data)
    assert result["tokens"] == [2, 3, 4]
    assert len(result["dists"]) == 2
    assert result["dists"][0][0] == [100, 200, 300]
    assert result["dists"][0][1] == [0.5, 0.3, 0.2]


def test_adapter_update_with_floats():
    """Test handling adapter update with many float values."""
    endpoint = RpcEndpoint("model_service")

    def update_adapter(
        adapter_ptr: int,
        scores: list[float],
        seeds: list[int],
        max_sigma: float,
    ) -> dict:
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "adapter_ptr": adapter_ptr,
            "avg_score": avg_score,
            "num_updates": len(scores),
            "clamped_sigma": min(max_sigma, 1.0),
        }

    endpoint.register(
        "update_adapter", update_adapter, request_model=UpdateAdapterRequest
    )

    # Large batch of scores
    scores = [0.1 * i for i in range(100)]
    seeds = list(range(100))

    result = endpoint._dispatch(
        "update_adapter",
        {
            "adapter_ptr": 0x7000,
            "scores": scores,
            "seeds": seeds,
            "max_sigma": 0.5,
        },
    )

    assert result["adapter_ptr"] == 0x7000
    assert result["num_updates"] == 100
    assert 4.9 < result["avg_score"] < 5.0  # avg of 0..9.9
    assert result["clamped_sigma"] == 0.5


def test_large_batch_processing():
    """Test processing large batches of data."""
    endpoint = RpcEndpoint("batch_service")

    def process_batch(items: list[dict]) -> dict:
        total = sum(item.get("value", 0) for item in items)
        return {
            "count": len(items),
            "total": total,
            "average": total / len(items) if items else 0,
        }

    endpoint.register("process_batch", process_batch)

    # Create a large batch
    batch = [{"id": i, "value": i * 2, "name": f"item_{i}"} for i in range(1000)]

    result = endpoint._dispatch("process_batch", {"items": batch})
    assert result["count"] == 1000
    assert result["total"] == sum(i * 2 for i in range(1000))


def test_nested_dict_structures():
    """Test deeply nested dictionary structures."""
    endpoint = RpcEndpoint("config_service")

    def update_config(config: dict) -> dict:
        # Simulate merging config
        return {
            "status": "updated",
            "merged_keys": list(config.keys()),
            "nested_depth": _get_depth(config),
        }

    def _get_depth(d: dict, level: int = 0) -> int:
        if not isinstance(d, dict) or not d:
            return level
        return max(_get_depth(v, level + 1) for v in d.values())

    endpoint.register("update_config", update_config)

    nested_config = {
        "level1": {
            "level2": {
                "level3": {
                    "level4": {
                        "value": 42,
                        "settings": {"a": 1, "b": 2},
                    }
                }
            }
        },
        "options": {
            "feature_flags": {
                "experimental": True,
                "beta": {"enabled": True, "users": [1, 2, 3]},
            }
        },
    }

    result = endpoint._dispatch("update_config", {"config": nested_config})
    assert result["status"] == "updated"
    assert "level1" in result["merged_keys"]
    assert "options" in result["merged_keys"]
    assert result["nested_depth"] >= 4


def test_mixed_type_list_handling():
    """Test handling lists with mixed types in responses."""
    endpoint = RpcEndpoint("data_service")

    def get_mixed_data() -> dict:
        return {
            "results": [
                {"type": "int", "value": 42},
                {"type": "float", "value": 3.14159},
                {"type": "str", "value": "hello"},
                {"type": "list", "value": [1, 2, 3]},
                {"type": "dict", "value": {"nested": True}},
                {"type": "none", "value": None},
                {"type": "bool", "value": True},
            ]
        }

    endpoint.register("get_mixed_data", get_mixed_data)

    result = endpoint._dispatch("get_mixed_data", {})
    assert len(result["results"]) == 7
    assert result["results"][0]["value"] == 42
    assert result["results"][1]["value"] == 3.14159
    assert result["results"][4]["value"]["nested"] is True
    assert result["results"][5]["value"] is None


def test_programmatic_registration_with_validation():
    """Test programmatic registration with Pydantic validation."""
    endpoint = RpcEndpoint("validated_service")

    def create_user(name: str, age: int) -> dict:
        return {"id": 1, "name": name, "age": age, "created": True}

    endpoint.register("create_user", create_user, request_model=UserRequest)

    result = endpoint._dispatch("create_user", {"name": "Alice", "age": 30})
    assert result["name"] == "Alice"
    assert result["age"] == 30
    assert result["created"] is True

    # Test validation failure
    with pytest.raises(ValueError, match="Validation error"):
        endpoint._dispatch("create_user", {"name": "Bob", "age": "not_an_int"})


def test_high_throughput_scenario():
    """Test simulating high-throughput RPC scenario."""
    endpoint = RpcEndpoint("high_throughput")

    call_count = 0

    def increment(value: int) -> dict:
        nonlocal call_count
        call_count += 1
        return {"result": value + 1, "call_id": call_count}

    endpoint.register("increment", increment)

    # Simulate many rapid calls
    results = []
    for i in range(10000):
        result = endpoint._dispatch("increment", {"value": i})
        results.append(result)

    assert len(results) == 10000
    assert results[0]["result"] == 1
    assert results[9999]["result"] == 10000
    assert call_count == 10000


def test_binary_data_in_response():
    """Test handling binary data (as list of ints, simulating bytes)."""
    endpoint = RpcEndpoint("binary_service")

    def get_binary_data(size: int) -> dict:
        # Simulate binary data as list of integers (0-255)
        data = list(range(min(size, 256)))
        return {
            "data": data,
            "size": len(data),
            "checksum": sum(data) % 256,
        }

    endpoint.register("get_binary_data", get_binary_data)

    result = endpoint._dispatch("get_binary_data", {"size": 100})
    assert result["size"] == 100
    assert result["data"][0] == 0
    assert result["data"][99] == 99
    assert result["checksum"] == sum(range(100)) % 256
