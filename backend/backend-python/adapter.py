import torch
import torch.nn as nn
import math


## Abstract Base Classes and Utilities
### ---------------------------------------------------------------------------------

class Adapter:
    """
    Abstract base class defining the interface for a LoRA-style adapter.
    """

    @property
    def num_layers(self) -> int:
        """Returns the number of layers the adapter manages."""
        raise NotImplementedError("Subclasses should implement this method.")

    @property
    def rank(self) -> int:
        """Returns the rank of the adapter."""
        raise NotImplementedError("Subclasses should implement this method.")

    @property
    def alpha(self) -> float:
        """Returns the alpha scaling factor of the adapter."""
        raise NotImplementedError("Subclasses should implement this method.")

    @property
    def in_features(self) -> int:
        """Returns the input feature dimension of the adapter."""
        raise NotImplementedError("Subclasses should implement this method.")

    @property
    def out_features(self) -> int:
        """Returns the output feature dimension of the adapter."""
        raise NotImplementedError("Subclasses should implement this method.")

    def down_proj(self, layer_idx: int) -> torch.Tensor:
        """Returns the down-projection weight (A) for a specific layer."""
        raise NotImplementedError("Subclasses should implement this method.")

    def up_proj(self, layer_idx: int) -> torch.Tensor:
        """Returns the up-projection weight (B) for a specific layer."""
        raise NotImplementedError("Subclasses should implement this method.")


class AdapterBuffer:
    """
    Utility class to manage and apply a buffer of adapters in a batched context.
    (This class was provided as part of the original context.)
    """
    adapters: list[Adapter]
    adapter_indices: torch.Tensor
    x_indptr: torch.Tensor
    segment_size: int

    down_proj_buffer: torch.Tensor
    up_proj_buffer: torch.Tensor
    down_buffer: torch.Tensor
    up_buffer: torch.Tensor

    def __init__(self,
                 adapters: list[Adapter],
                 adapter_indices: torch.Tensor,
                 x_indptr: torch.Tensor,
                 segment_gemm_wrapper
                 ):
        self.adapters = adapters
        self.adapter_indices = adapter_indices
        self.x_indptr = x_indptr
        self.segment_size = x_indptr[-1].item()
        self.segment_gemm_wrapper = segment_gemm_wrapper

        # buffers for storing weights
        self.down_proj_buffer = torch.empty((len(adapters), adapters[0].in_features, adapters[0].rank), device=x_indptr.device)
        self.up_proj_buffer = torch.empty((len(adapters), adapters[0].rank, adapters[0].out_features), device=x_indptr.device)

        # buffers for activations
        self.down_buffer = torch.empty((self.segment_size, adapters[0].rank), device=x_indptr.device)
        self.up_buffer = torch.empty((self.segment_size, adapters[0].out_features), device=x_indptr.device)

    def compute_lora_delta(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        # first update the buffers with the current adapter weights
        for i in range(len(self.adapters)):
            adapter = self.adapters[i]
            self.down_proj_buffer[i].copy_(adapter.down_proj(layer_idx))
            self.up_proj_buffer[i].copy_(adapter.up_proj(layer_idx))

        # do segmented GEMM
        self.segment_gemm_wrapper.run(
            x=x,
            weights=self.down_proj_buffer,
            batch_size=self.segment_size,
            weight_column_major=False,
            out=self.down_buffer,
            seg_indptr=self.x_indptr,
            weight_indices=self.adapter_indices
        )

        self.segment_gemm_wrapper.run(
            x=self.down_buffer,
            weights=self.up_proj_buffer,
            batch_size=self.segment_size,
            weight_column_major=False,
            out=self.up_buffer,
            seg_indptr=self.x_indptr,
            weight_indices=self.adapter_indices
        )

        scale = self.adapters[0].alpha / self.adapters[0].rank
        return scale * self.up_buffer


class BaseAdapter(Adapter):
    """A basic adapter implementation holding weight tensors for multiple layers."""
    base_num_layers: int
    base_rank: int
    base_alpha: float
    base_in_features: int
    base_out_features: int
    base_weight_down: torch.Tensor
    base_weight_up: torch.Tensor

    def __init__(self, num_layers: int, in_features: int | list[int], out_features: int | list[int], rank: int, alpha: float):
        super().__init__()
        self.base_num_layers = num_layers
        self.base_rank = rank
        self.base_alpha = alpha

        in_features_total = sum(in_features) if isinstance(in_features, list) else in_features
        out_features_total = sum(out_features) if isinstance(out_features, list) else out_features

        self.base_in_features = in_features_total
        self.base_out_features = out_features_total
        self.base_weight_down = torch.empty((num_layers, in_features_total, rank))
        self.base_weight_up = torch.empty((num_layers, rank, out_features_total))

    @property
    def num_layers(self) -> int:
        return self.base_num_layers

    @property
    def rank(self) -> int:
        return self.base_rank

    @property
    def alpha(self) -> float:
        return self.base_alpha

    @property
    def in_features(self) -> int:
        return self.base_in_features

    @property
    def out_features(self) -> int:
        return self.base_out_features

    def down_proj(self, layer_idx: int) -> torch.Tensor:
        return self.base_weight_down[layer_idx]

    def up_proj(self, layer_idx: int) -> torch.Tensor:
        return self.base_weight_up[layer_idx]


## CMA-ES Implementation
### ---------------------------------------------------------------------------------

class MutableAdapter(BaseAdapter):
    """
    The central controller for the evolutionary strategy. 🧬

    This class manages a separate search distribution (mean, sigma, covariance)
    for each layer's adapter weights.
    """

    def __init__(self,
                 num_layers: int,
                 in_features: int | list[int],
                 out_features: int | list[int],
                 rank: int,
                 alpha: float,
                 population_size: int,
                 mu_fraction: float = 0.5,
                 initial_sigma: float = 0.02,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device = None):

        super().__init__(num_layers, in_features, out_features, rank, alpha)

        final_device = device if device is not None else self.base_weight_down.device

        self.base_weight_down = self.base_weight_down.to(device=final_device, dtype=dtype)
        self.base_weight_up = self.base_weight_up.to(device=final_device, dtype=dtype)

        for i in range(num_layers):
            nn.init.kaiming_uniform_(self.base_weight_down[i], a=math.sqrt(5))
            nn.init.zeros_(self.base_weight_up[i])

        self.population_size = population_size
        self.mu = int(population_size * mu_fraction)
        if self.mu == 0:
            raise ValueError(f"Number of parents (mu) must be > 0.")

        d_per_layer = self.base_weight_down[0].numel() + self.base_weight_up[0].numel()
        self.d = d_per_layer

        self.sigma = torch.full((num_layers, 1, 1), initial_sigma, dtype=dtype, device=final_device)
        self.ps_down = torch.zeros_like(self.base_weight_down)
        self.ps_up = torch.zeros_like(self.base_weight_up)
        self.pc_down = torch.zeros_like(self.base_weight_down)
        self.pc_up = torch.zeros_like(self.base_weight_up)
        self.diag_C_down = torch.ones_like(self.base_weight_down)
        self.diag_C_up = torch.ones_like(self.base_weight_up)

        log_weights = torch.log(torch.tensor(self.mu + 0.5, device=final_device)) - torch.log(torch.arange(1, self.mu + 1, dtype=dtype, device=final_device))
        self.weights = log_weights / log_weights.sum()
        self.weights_reshaped = self.weights.view(-1, 1, 1)

        self.mueff = self.weights.sum() ** 2 / (self.weights ** 2).sum()
        self.cc = (4 + self.mueff / self.d) / (self.d + 4 + 2 * self.mueff / self.d)
        self.cs = (self.mueff + 2) / (self.d + self.mueff + 5)
        self.c1 = 2 / ((self.d + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.d + 2) ** 2 + self.mueff))
        damps_term = torch.sqrt((self.mueff - 1) / (self.d + 1)) - 1
        self.damps = 1 + 2 * torch.clamp(damps_term, min=0) + self.cs
        self.chiN = self.d ** 0.5 * (1 - 1 / (4 * self.d) + 1 / (21 * self.d ** 2))

    @torch.no_grad()
    def update(self, scores: list[float], seeds: list[int]) -> None:
        if len(scores) != self.population_size or len(seeds) != self.population_size:
            raise ValueError(f"Expected scores and seeds for {self.population_size} individuals.")

        device = self.base_weight_down.device
        dtype = self.base_weight_down.dtype

        scores_tensor = torch.tensor(scores, device=device, dtype=torch.float32)
        seeds_tensor = torch.tensor(seeds, device=device, dtype=torch.long)
        _, best_indices = torch.sort(scores_tensor, descending=True)
        top_seeds = seeds_tensor[best_indices[:self.mu]].tolist()

        # --- OPTIMIZED: Step 1 - Generate all noise in a batch ---
        z_down_shape = self.base_weight_down.shape[1:]
        z_up_shape = self.base_weight_up.shape[1:]
        all_z_down = torch.empty((self.mu, self.num_layers, *z_down_shape), device=device, dtype=dtype)
        all_z_up = torch.empty((self.mu, self.num_layers, *z_up_shape), device=device, dtype=dtype)
        generator = torch.Generator(device=device)

        for i, seed in enumerate(top_seeds):
            generator.manual_seed(seed)
            # Generate noise for all layers for the i-th individual at once
            all_z_down[i] = torch.randn(self.num_layers, *z_down_shape, generator=generator)
            all_z_up[i] = torch.randn(self.num_layers, *z_up_shape, generator=generator)

        # --- Step 2 - Loop through layers to apply updates ---
        for layer_idx in range(self.num_layers):
            # Get noise for the current layer by slicing the pre-generated tensor
            selected_z_down = all_z_down[:, layer_idx]
            selected_z_up = all_z_up[:, layer_idx]

            # --- Adaptation logic for the current layer (unchanged) ---
            old_mean_down = self.base_weight_down[layer_idx].clone()
            old_mean_up = self.base_weight_up[layer_idx].clone()
            sigma_l = self.sigma[layer_idx]
            diag_C_down_l = self.diag_C_down[layer_idx]
            diag_C_up_l = self.diag_C_up[layer_idx]
            std_dev_down = torch.sqrt(diag_C_down_l)
            std_dev_up = torch.sqrt(diag_C_up_l)

            selected_parents_down = old_mean_down + sigma_l * std_dev_down * selected_z_down
            selected_parents_up = old_mean_up + sigma_l * std_dev_up * selected_z_up

            new_mean_down = torch.sum(selected_parents_down * self.weights_reshaped, dim=0)
            new_mean_up = torch.sum(selected_parents_up * self.weights_reshaped, dim=0)
            self.base_weight_down[layer_idx].copy_(new_mean_down)
            self.base_weight_up[layer_idx].copy_(new_mean_up)

            step_down = (self.base_weight_down[layer_idx] - old_mean_down) / sigma_l
            step_up = (self.base_weight_up[layer_idx] - old_mean_up) / sigma_l

            C_inv_sqrt_step_down = step_down / std_dev_down
            C_inv_sqrt_step_up = step_up / std_dev_up

            ps_down_l = self.ps_down[layer_idx]
            ps_up_l = self.ps_up[layer_idx]
            self.ps_down[layer_idx] = (1 - self.cs) * ps_down_l + torch.sqrt(self.cs * (2 - self.cs) * self.mueff) * C_inv_sqrt_step_down
            self.ps_up[layer_idx] = (1 - self.cs) * ps_up_l + torch.sqrt(self.cs * (2 - self.cs) * self.mueff) * C_inv_sqrt_step_up

            norm_ps = torch.sqrt(torch.sum(self.ps_down[layer_idx] ** 2) + torch.sum(self.ps_up[layer_idx] ** 2))
            self.sigma[layer_idx] *= torch.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1))

            h_sigma_cond = (norm_ps / torch.sqrt(1 - (1 - self.cs) ** (2 * (self.population_size + 1)))) < (1.4 + 2 / (self.d + 1)) * self.chiN
            h_sigma = 1.0 if h_sigma_cond else 0.0

            pc_down_l = self.pc_down[layer_idx]
            pc_up_l = self.pc_up[layer_idx]
            self.pc_down[layer_idx] = (1 - self.cc) * pc_down_l + h_sigma * torch.sqrt(self.cc * (2 - self.cc) * self.mueff) * step_down
            self.pc_up[layer_idx] = (1 - self.cc) * pc_up_l + h_sigma * torch.sqrt(self.cc * (2 - self.cc) * self.mueff) * step_up

            y_down_parents = std_dev_down * selected_z_down
            y_up_parents = std_dev_up * selected_z_up
            rank_one_update_down = self.c1 * (self.pc_down[layer_idx] ** 2)
            rank_one_update_up = self.c1 * (self.pc_up[layer_idx] ** 2)
            rank_mu_update_down = self.cmu * torch.sum((y_down_parents ** 2) * self.weights_reshaped, dim=0)
            rank_mu_update_up = self.cmu * torch.sum((y_up_parents ** 2) * self.weights_reshaped, dim=0)

            self.diag_C_down[layer_idx] = (1 - self.c1 - self.cmu) * diag_C_down_l + rank_one_update_down + rank_mu_update_down
            self.diag_C_up[layer_idx] = (1 - self.c1 - self.cmu) * diag_C_up_l + rank_one_update_up + rank_mu_update_up


class AdapterWithMutation(Adapter):
    """
    Represents a single individual in the population. 🍃

    It samples a set of weights from the distribution defined by a MutableAdapter.
    Noise is pre-computed for all layers upon instantiation to improve performance.
    """
    adapter: MutableAdapter
    seed: int
    # MODIFIED: Attributes are now tensors, not dictionaries.
    z_down_noise: torch.Tensor
    z_up_noise: torch.Tensor

    def __init__(self, adapter: MutableAdapter, seed: int):
        self.adapter = adapter
        self.seed = seed

        num_layers = self.adapter.num_layers
        device = self.adapter.base_weight_down.device
        dtype = self.adapter.base_weight_down.dtype
        generator = torch.Generator(device=device).manual_seed(self.seed)

        # --- SIMPLIFIED: Generate and store noise tensors directly ---
        # The shape from the adapter is already (num_layers, in_features, rank).
        down_noise_shape = self.adapter.base_weight_down.shape
        up_noise_shape = self.adapter.base_weight_up.shape

        self.z_down_noise = torch.randn(down_noise_shape, generator=generator, device=device, dtype=dtype)
        self.z_up_noise = torch.randn(up_noise_shape, generator=generator, device=device, dtype=dtype)

    def _sample(self, mean_weight: torch.Tensor, diag_C: torch.Tensor, sigma: torch.Tensor, z_noise: torch.Tensor) -> torch.Tensor:
        std_dev = torch.sqrt(diag_C)
        return mean_weight + sigma * std_dev * z_noise

    def down_proj(self, layer_idx: int) -> torch.Tensor:
        """Retrieves pre-computed noise and samples the down-projection weight."""
        # MODIFIED: Index the tensor directly instead of dictionary lookup.
        z_down = self.z_down_noise[layer_idx]
        mean_weight = self.adapter.down_proj(layer_idx)
        diag_C_down = self.adapter.diag_C_down[layer_idx]
        sigma = self.adapter.sigma[layer_idx]
        return self._sample(mean_weight, diag_C_down, sigma, z_down)

    def up_proj(self, layer_idx: int) -> torch.Tensor:
        """Retrieves pre-computed noise and samples the up-projection weight."""
        # MODIFIED: Index the tensor directly instead of dictionary lookup.
        z_up = self.z_up_noise[layer_idx]
        mean_weight = self.adapter.up_proj(layer_idx)
        diag_C_up = self.adapter.diag_C_up[layer_idx]
        sigma = self.adapter.sigma[layer_idx]
        return self._sample(mean_weight, diag_C_up, sigma, z_up)

    @property
    def num_layers(self) -> int:
        return self.adapter.num_layers

    @property
    def rank(self) -> int:
        return self.adapter.rank

    @property
    def alpha(self) -> float:
        return self.adapter.alpha

    @property
    def in_features(self) -> int:
        return self.adapter.in_features

    @property
    def out_features(self) -> int:
        return self.adapter.out_features