"""
inferlet-py - Python SDK for writing Pie inferlets

This mirrors the inferlet-js library API with Pythonic idioms.
"""

__version__ = "0.1.0"

# Runtime functions
from .runtime import (
    get_version,
    get_instance_id,
    get_arguments,
    set_return,
    was_return_set,
    debug_query,
)

# KVS functions
from .kvs import (
    store_get,
    store_set,
    store_delete,
    store_exists,
    store_list_keys,
)

# Messaging functions
from .messaging import (
    send,
    receive,
    send_blob,
    receive_blob,
    broadcast,
    subscribe,
    Blob,
    Subscription,
)

# Model and Queue
from .model import (
    Model,
    Queue,
    get_model,
    get_all_models,
    get_auto_model,
    get_all_models_with_traits,
)

# Tokenizer
from .tokenizer import Tokenizer

# Sampler
from .sampler import Sampler

# ForwardPass
from .forward import ForwardPass, ForwardPassResult

# KV Page management
from .kv_page import KvPage, KvPageManager

# Chat formatting
from .chat import ChatFormatter, format_messages, ToolCall, Message

# BRLE (attention mask encoding)
from .brle import Brle, causal_mask, causal_mask_raw

# Context (main high-level API)
from .context import Context, GenerateResult

# Drafter (speculative decoding)
from .drafter import Drafter, EmptyDrafter

__all__ = [
    # Version
    "__version__",
    # Runtime
    "get_version",
    "get_instance_id",
    "get_arguments",
    "set_return",
    "was_return_set",
    "debug_query",
    # KVS
    "store_get",
    "store_set",
    "store_delete",
    "store_exists",
    "store_list_keys",
    # Messaging
    "send",
    "receive",
    "send_blob",
    "receive_blob",
    "broadcast",
    "subscribe",
    "Blob",
    "Subscription",
    # Model
    "Model",
    "Queue",
    "get_model",
    "get_all_models",
    "get_auto_model",
    "get_all_models_with_traits",
    # Tokenizer
    "Tokenizer",
    # Sampler
    "Sampler",
    # ForwardPass
    "ForwardPass",
    "ForwardPassResult",
    # KV Page
    "KvPage",
    "KvPageManager",
    # Chat
    "ChatFormatter",
    "format_messages",
    "ToolCall",
    "Message",
    # BRLE
    "Brle",
    "causal_mask",
    "causal_mask_raw",
    # Context
    "Context",
    "GenerateResult",
    # Drafter
    "Drafter",
    "EmptyDrafter",
]
