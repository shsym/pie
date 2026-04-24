"""Pie configuration: typed schema, loading, and defaults.

Configuration dataclasses, TOML config loading, and platform-aware defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import toml

from pie import path as pie_path


# -- Schema -------------------------------------------------------------------


@dataclass
class AuthConfig:
    """Authentication configuration."""

    enabled: bool = True


@dataclass
class TelemetryConfig:
    """OpenTelemetry tracing configuration."""

    enabled: bool = False
    endpoint: str = "http://localhost:4317"
    service_name: str = "pie"


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    hf_repo: str = ""
    device: list[str] = field(default_factory=lambda: ["cuda:0"])
    tensor_parallel_size: int | None = None
    activation_dtype: str = "bfloat16"
    weight_dtype: str = "bfloat16"
    kv_page_size: int = 16
    max_batch_tokens: int | None = None
    max_dist_size: int = 32
    max_num_embeds: int = 128
    max_num_adapters: int = 32
    max_adapter_rank: int = 8
    gpu_mem_utilization: float = 0.8
    cpu_mem_budget_in_gb: int = 0
    use_cuda_graphs: bool = False
    random_seed: int = 42
    dummy_mode: bool = False
    # Default compute-wallet cap. `None` (the default) means *unlimited*:
    # the process is not capped by the runtime. Clients can still opt into
    # a cap on a per-launch basis via `client.launch_process(..., token_budget=N)`.
    default_token_budget: int | None = None
    # Default market endowment (in KV pages) assigned to a process launched
    # without an explicit token_budget. One unit = one page of long-run
    # entitled GPU residency under contention.
    default_endowment_pages: int = 64
    # Admission oversubscription factor: `Σ endowment ≤ capacity × factor`.
    oversubscription_factor: float = 4.0
    max_batch_size: int = 512

    # Name derived from hf_repo if not explicitly set
    name: str = ""

    def __post_init__(self):
        if not self.name:
            self.name = self.hf_repo
        if self.default_token_budget is not None and self.default_token_budget <= 0:
            raise ValueError(
                f"Model {self.name or self.hf_repo!r}: default_token_budget, when set, "
                f"must be > 0 (got {self.default_token_budget!r})"
            )
        if self.default_endowment_pages <= 0:
            raise ValueError(
                f"Model {self.name or self.hf_repo!r}: default_endowment_pages "
                f"must be > 0 (got {self.default_endowment_pages!r})"
            )
        if self.oversubscription_factor <= 0.0:
            raise ValueError(
                f"Model {self.name or self.hf_repo!r}: oversubscription_factor "
                f"must be > 0 (got {self.oversubscription_factor!r})"
            )


@dataclass
class Config:
    """Top-level engine configuration.

    This is the single source of truth passed through the system,
    replacing the untyped dict pairs (engine_config, model_configs).
    """

    host: str = "127.0.0.1"
    port: int = 8080
    verbose: bool = False
    registry: str = "https://registry.pie-project.org/"

    auth: AuthConfig = field(default_factory=AuthConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    models: list[ModelConfig] = field(default_factory=list)
    allow_filesystem: bool = False
    max_concurrent_processes: int | None = None
    python_snapshot: bool = True

    @property
    def primary_model(self) -> ModelConfig:
        """Get the first (primary) model config."""
        if not self.models:
            raise ValueError("No model configuration found")
        return self.models[0]


# -- Defaults -----------------------------------------------------------------

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


def get_default_device() -> str:
    """Get the default device based on the platform."""
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_default_config_content() -> str:
    """Create the default configuration file content."""
    device = get_default_device()
    formatted_device = f'"{device}"'

    return f"""\
host = "127.0.0.1"
port = 8080
verbose = false
registry = "https://registry.pie-project.org/"

# Max concurrent processes (comment out or remove for no limit)
# max_concurrent_processes = 64

# Apply the host-side snapshot optimization to Python inferlets.
# When true, CPython initialization is captured once at install time
# and subsequent launches start from the post-init state.
python_snapshot = true

[auth]
enabled = false

[telemetry]
enabled = false
endpoint = "http://localhost:4317"
service_name = "pie"

# Model configuration (can have multiple [[model]] sections)
[[model]]
hf_repo = "{DEFAULT_MODEL}"

# Default compute cap per process. Unset (commented out) = unlimited — the
# recommended default. Clients that want a cap pass `token_budget=N` to
# `launch_process()` instead. Set a value here only to apply a cap to every
# process that doesn't declare one (e.g. for shared hosting).
# default_token_budget = 1024

# Default market endowment, in KV pages, for processes launched without an
# explicit token_budget. Controls this process's long-run share of GPU
# residency under contention.
default_endowment_pages = 64

# Admission oversubscription factor. Σ endowment ≤ capacity × factor.
# 1.0 = strict (no contention ever). >1.0 = overbook.
oversubscription_factor = 4.0

# Device assignment (single GPU or list for tensor parallel)
device = [{formatted_device}]

# Tensor Parallelism degree (splits model across GPUs within a group)
# Set to 1 for Data Parallelism only (each GPU runs full model independently)
# Set to len(device) for Tensor Parallelism only (all GPUs share one model)
# Example: 4 GPUs with tensor_parallel_size=2 → 2 DP groups of 2 TP GPUs
tensor_parallel_size = 1

# Precision settings
activation_dtype = "bfloat16"
weight_dtype = "bfloat16"

# KV cache configuration
kv_page_size = 16

# Distribution/sampling
max_dist_size = 32

# Embedding limits
max_num_embeds = 128

# Adapter (LoRA) settings
max_num_adapters = 32
max_adapter_rank = 8

# Memory management
gpu_mem_utilization = 0.8

# CPU memory budget for KV cache swap (GB), 0 = disabled
cpu_mem_budget_in_gb = 0

# CUDA graphs (experimental)
use_cuda_graphs = false

# Random seed
random_seed = 42

"""


# -- Loader -------------------------------------------------------------------


def load_config(
    config_path: Path | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    no_auth: bool = False,
    verbose: bool = False,
    registry: str | None = None,
    dummy_mode: bool = False,
) -> Config:
    """Load configuration from TOML file and merge CLI overrides.

    Args:
        config_path: Path to TOML config file (defaults to ~/.pie/config.toml).
        host: Override host address.
        port: Override port.
        no_auth: Shorthand to disable auth.
        verbose: Enable verbose logging.
        registry: Override registry URL.
        dummy_mode: Enable dummy mode on all models.

    Returns:
        Fully resolved Config.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If no model configuration is found.
    """
    file_path = config_path or pie_path.get_default_config_path()
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration not found at {file_path}")

    raw = toml.loads(file_path.read_text())

    auth_section = raw.get("auth", {})
    telemetry_section = raw.get("telemetry", {})

    # Auth
    auth_enabled = not no_auth and auth_section.get("enabled", True)

    # Models
    model_configs_raw = raw.get("model", [])
    if isinstance(model_configs_raw, dict):
        model_configs_raw = [model_configs_raw]
    if not model_configs_raw:
        raise ValueError("No model configuration found")

    models = []
    for mc in model_configs_raw:
        # Normalize device to list
        device = mc.get("device", ["cuda:0"])
        if isinstance(device, str):
            device = [device]

        m = ModelConfig(
            hf_repo=mc.get("hf_repo", ""),
            device=device,
            tensor_parallel_size=mc.get("tensor_parallel_size"),
            activation_dtype=mc.get("activation_dtype", "bfloat16"),
            weight_dtype=mc.get("weight_dtype", "bfloat16"),
            kv_page_size=mc.get("kv_page_size", 16),
            max_batch_tokens=mc.get("max_batch_tokens"),
            max_dist_size=mc.get("max_dist_size", 32),
            max_num_embeds=mc.get("max_num_embeds", 128),
            max_num_adapters=mc.get("max_num_adapters", 32),
            max_adapter_rank=mc.get("max_adapter_rank", 8),
            gpu_mem_utilization=mc.get("gpu_mem_utilization", 0.8),
            cpu_mem_budget_in_gb=mc.get("cpu_mem_budget_in_gb", 0),
            use_cuda_graphs=mc.get("use_cuda_graphs", False),
            random_seed=mc.get("random_seed", 42),
            dummy_mode=dummy_mode or mc.get("dummy_mode", False),
            # None (omitted) = unlimited; set explicitly to cap.
            default_token_budget=mc.get("default_token_budget"),
            default_endowment_pages=mc.get("default_endowment_pages", 64),
            oversubscription_factor=mc.get("oversubscription_factor", 4.0),
            name=mc.get("name", ""),
        )
        models.append(m)

    return Config(
        host=host or raw.get("host", "127.0.0.1"),
        port=port or raw.get("port", 8080),
        verbose=verbose or raw.get("verbose", False),
        registry=registry or raw.get("registry", "https://registry.pie-project.org/"),
        auth=AuthConfig(enabled=auth_enabled),
        telemetry=TelemetryConfig(
            enabled=telemetry_section.get("enabled", False),
            endpoint=telemetry_section.get("endpoint", "http://localhost:4317"),
            service_name=telemetry_section.get("service_name", "pie"),
        ),
        models=models,
        allow_filesystem=raw.get("allow_filesystem", False),
        max_concurrent_processes=raw.get("max_concurrent_processes"),
        python_snapshot=raw.get("python_snapshot", True),
    )
