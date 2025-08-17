import time

import torch
import torch.nn as nn
import math


class Adapter:
    """
    Abstract base class defining the interface for a LoRA-style adapter.
    """

    adapter_id: int
    rank: int
    alpha: float
    out_features: list[int]

    def __init__(self, adapter_id: int, rank: int, alpha: float, out_features: list[int]):
        self.adapter_id = adapter_id
        self.rank = rank
        self.alpha = alpha
        self.out_features = out_features


class AdapterBuffer:
    """
    Utility class to manage and apply a buffer of adapters in a batched context.
    (This class was provided as part of the original context.)
    """
    adapter_indices: torch.Tensor
    adapter_at_layer: list[tuple[torch.Tensor, torch.Tensor]]
    x_indptr: torch.Tensor
    num_segments: int
    num_adapter_group: int
    out_features: list[int]
    nnz: int

    def __init__(self,
                 rank: int,
                 alpha: float,
                 adapter_indices: torch.Tensor,
                 adapter_at_layer: list[tuple[torch.Tensor, torch.Tensor]],
                 x_indptr: torch.Tensor,
                 segment_gemm_wrapper,
                 out_features: list[int],
                 ):
        self.rank = rank
        self.alpha = alpha
        self.adapter_indices = adapter_indices
        self.adapter_at_layer = adapter_at_layer
        self.x_indptr = x_indptr
        self.segment_gemm_wrapper = segment_gemm_wrapper
        self.out_features = out_features

        self.num_segments = x_indptr.shape[0]
        self.nnz = x_indptr[-1].item()

        num_group = len(out_features)
        self.num_adapter_group = num_group

        self.out_features = out_features

    @torch.inference_mode()
    def compute_lora_delta(self, layer_idx: int, x: torch.Tensor) -> list[torch.Tensor]:
        # start_time = time.time()
        down_projs, up_projs = self.adapter_at_layer[layer_idx]

        # do segmented GEMM
        down = self.segment_gemm_wrapper.run(
            x=x,
            weights=down_projs,
            batch_size=self.num_segments,
            weight_column_major=False,
            seg_indptr=self.x_indptr,
            weight_indices=self.adapter_indices
        )

        down_buffer_by_grp = torch.split(down, self.rank, dim=-1)
        up_proj_buffer_by_grp = torch.split(up_projs, self.out_features, dim=-1)

        # do segmented gemm by group (nnz, rank) * (seg, rank, out_features[i])
        out_list = []
        scale = self.alpha / self.rank
        for i in range(self.num_adapter_group):
            out = self.segment_gemm_wrapper.run(
                x=down_buffer_by_grp[i].contiguous(),
                weights=up_proj_buffer_by_grp[i].contiguous(),
                batch_size=self.num_segments,
                weight_column_major=False,
                seg_indptr=self.x_indptr,
                weight_indices=self.adapter_indices
            )
            out_list.append(scale * out)

        return out_list


## CMA-ES Implementation
### ---------------------------------------------------------------------------------
class MutableAdapter(Adapter):
    """
    CMA-ES controller over LoRA adapter weights (per-layer, diagonal covariance).
    Maintains the mean directly in `adapters_by_layer[...][adapter_id]`.
    """

    @torch.inference_mode()
    def __init__(self,
                 rank: int,
                 alpha: float,
                 adapter_id: int,
                 adapters_by_layer: list[tuple[torch.Tensor, torch.Tensor]],
                 out_features: list[int],
                 population_size: int,
                 mu_fraction: float,
                 initial_sigma: float,
                 min_sigma: float = 1e-5,
                 min_var: float = 1e-8,
                 max_var: float = 1e4):

        super().__init__(adapter_id, rank, alpha, out_features)

        self.rank = rank
        self.alpha = alpha
        self.adapter_id = adapter_id

        # Shapes and dtypes from model params
        down0, up0 = adapters_by_layer[0]
        device = down0.device
        w_dtype = down0.dtype
        num_layers = len(adapters_by_layer)

        # Dimension per layer = size of (single adapter) DOWN + UP
        d_per_layer = down0[adapter_id].numel() + up0[adapter_id].numel()

        # Strategy state in float32 for stability
        f32 = torch.float32
        self.device = device
        self.w_dtype = w_dtype
        self.num_layers = num_layers
        self.d = float(d_per_layer)  # float for formulas

        self.down_shape = (num_layers, *down0.shape[1:])  # like a single adapter's slice per layer
        self.up_shape = (num_layers, *up0.shape[1:])

        # Initialize mean in model weights (you already do this)
        for layer_idx in range(num_layers):
            dW, uW = adapters_by_layer[layer_idx]
            nn.init.kaiming_uniform_(dW[adapter_id], a=math.sqrt(5))
            nn.init.zeros_(uW[adapter_id])

        # Recombination weights (rank-based, log)
        self.population_size = int(population_size)
        self.mu = max(1, int(population_size * mu_fraction))
        ranks = torch.arange(1, self.mu + 1, dtype=f32, device=device)
        logw = torch.log(torch.tensor(self.mu + 0.5, dtype=f32, device=device)) - torch.log(ranks)
        self.weights = (logw / logw.sum()).to(f32)  # (mu,)
        self.weights_reshaped = self.weights.view(-1, 1, 1)  # for broadcasting
        self.mueff = (self.weights.sum() ** 2) / (self.weights.pow(2).sum())

        # Strategy params (diag CMA-ES)
        self.cc = (4.0 + self.mueff / self.d) / (self.d + 4.0 + 2.0 * self.mueff / self.d)
        self.cs = (self.mueff + 2.0) / (self.d + self.mueff + 5.0)
        self.c1 = 2.0 / ((self.d + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((self.d + 2.0) ** 2 + self.mueff))
        damps_term = torch.sqrt((self.mueff - 1.0) / (self.d + 1.0)) - 1.0
        self.damps = 1.0 + 2.0 * torch.clamp(damps_term, min=0.0) + self.cs
        # E||N(0, I_d)|| (approx)
        self.chiN = (self.d ** 0.5) * (1.0 - 1.0 / (4.0 * self.d) + 1.0 / (21.0 * self.d ** 2))

        # Per-layer step-size and paths (float32)
        self.sigma = torch.full((num_layers, 1, 1), float(initial_sigma), dtype=f32, device=device)
        self.ps_down = torch.zeros(self.down_shape, dtype=f32, device=device)
        self.ps_up = torch.zeros(self.up_shape, dtype=f32, device=device)
        self.pc_down = torch.zeros(self.down_shape, dtype=f32, device=device)
        self.pc_up = torch.zeros(self.up_shape, dtype=f32, device=device)

        # Diagonal covariance starts as identity (ones), not zeros
        self.diag_C_down = torch.ones(self.down_shape, dtype=f32, device=device)
        self.diag_C_up = torch.ones(self.up_shape, dtype=f32, device=device)

        # Bounds
        self.min_sigma = float(min_sigma)
        self.min_var = float(min_var)
        self.max_var = float(max_var)

    @torch.inference_mode()
    def update(self,
               adapters_by_layer: list[tuple[torch.Tensor, torch.Tensor]],
               scores: list[float],
               seeds: list[int],
               max_sigma: float
               ) -> None:
        if len(scores) != self.population_size or len(seeds) != self.population_size:
            raise ValueError(f"Expected scores and seeds for {self.population_size} individuals.")

        device = self.device
        f32 = torch.float32
        eps = 1e-12

        # Select top-µ by score (descending)
        scores_t = torch.tensor(scores, device=device, dtype=f32)
        seeds_t = torch.tensor(seeds, device=device, dtype=torch.long)
        _, best_idx = torch.sort(scores_t, descending=True)
        top = seeds_t[best_idx[:self.mu]].tolist()

        # Recreate parents' noise (must match generation order/shapes)
        all_z_down = torch.empty((self.mu, *self.down_shape), device=device, dtype=f32)
        all_z_up = torch.empty((self.mu, *self.up_shape), device=device, dtype=f32)
        gen = torch.Generator(device=device)
        for i, s in enumerate(top):
            gen.manual_seed(int(s))
            all_z_down[i].copy_(torch.randn(self.down_shape, generator=gen, device=device, dtype=f32))
            all_z_up[i].copy_(torch.randn(self.up_shape, generator=gen, device=device, dtype=f32))

        W = self.weights_reshaped.to(f32)  # (µ,1,1)

        for layer_idx in range(self.num_layers):
            z_down = all_z_down[:, layer_idx]  # (µ, …)
            z_up = all_z_up[:, layer_idx]  # (µ, …)

            dW, uW = adapters_by_layer[layer_idx]
            mean_down_old = dW[self.adapter_id].to(f32)
            mean_up_old = uW[self.adapter_id].to(f32)

            sigma_l = self.sigma[layer_idx].to(f32)  # (1,1)
            std_down = torch.sqrt(self.diag_C_down[layer_idx].to(f32)).clamp_min(self.min_var ** 0.5)
            std_up = torch.sqrt(self.diag_C_up[layer_idx].to(f32)).clamp_min(self.min_var ** 0.5)

            # Parents in parameter space
            parents_down = mean_down_old + sigma_l * std_down * z_down
            parents_up = mean_up_old + sigma_l * std_up * z_up

            # Recombine (log-weights)
            mean_down_new = torch.sum(parents_down * W, dim=0)
            mean_up_new = torch.sum(parents_up * W, dim=0)

            # Write new mean back into model weights (cast to weight dtype)
            dW[self.adapter_id].copy_(mean_down_new.to(dW.dtype))
            uW[self.adapter_id].copy_(mean_up_new.to(uW.dtype))

            # Whitened step for step-size path
            step_down = (mean_down_new - mean_down_old) / (sigma_l + eps)
            step_up = (mean_up_new - mean_up_old) / (sigma_l + eps)
            Cinv_sqrt_step_down = step_down / (std_down + eps)
            Cinv_sqrt_step_up = step_up / (std_up + eps)

            # Update ps (step-size path)
            c = torch.sqrt(self.cs * (2.0 - self.cs) * self.mueff).to(f32)
            ps_down_new = (1.0 - self.cs) * self.ps_down[layer_idx].to(f32) + c * Cinv_sqrt_step_down
            ps_up_new = (1.0 - self.cs) * self.ps_up[layer_idx].to(f32) + c * Cinv_sqrt_step_up

            # Layerwise norm of the combined path
            norm_ps = torch.sqrt(ps_down_new.pow(2).sum() + ps_up_new.pow(2).sum())

            # Step-size update (steady-state denom ≈ 1, so no iter needed)
            factor = torch.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1.0))
            sigma_new = sigma_l * factor
            if max_sigma and max_sigma > 0:
                sigma_new = torch.clamp(sigma_new, min=self.min_sigma, max=float(max_sigma))
            else:
                sigma_new = torch.clamp(sigma_new, min=self.min_sigma)

            # h_sigma without iter: steady-state approximation (denominator ~ 1)
            h_sigma = (norm_ps < (1.4 + 2.0 / (self.d + 1.0)) * self.chiN).to(f32)

            # Covariance path (rank-1)
            c_pc = torch.sqrt(self.cc * (2.0 - self.cc) * self.mueff).to(f32)
            pc_down_new = (1.0 - self.cc) * self.pc_down[layer_idx].to(f32) + h_sigma * c_pc * step_down
            pc_up_new = (1.0 - self.cc) * self.pc_up[layer_idx].to(f32) + h_sigma * c_pc * step_up

            # Rank-µ terms (diag)
            y_down_parents = std_down * z_down
            y_up_parents = std_up * z_up
            rank1_down = self.c1 * pc_down_new.pow(2)
            rank1_up = self.c1 * pc_up_new.pow(2)
            rankmu_down = self.cmu * torch.sum((y_down_parents.pow(2)) * W, dim=0)
            rankmu_up = self.cmu * torch.sum((y_up_parents.pow(2)) * W, dim=0)

            diag_C_down_new = ((1.0 - self.c1 - self.cmu) * self.diag_C_down[layer_idx].to(f32)
                               + rank1_down + rankmu_down).clamp(self.min_var, self.max_var)
            diag_C_up_new = ((1.0 - self.c1 - self.cmu) * self.diag_C_up[layer_idx].to(f32)
                             + rank1_up + rankmu_up).clamp(self.min_var, self.max_var)

            # Write back to state (cast to original state dtypes)
            self.ps_down[layer_idx].copy_(ps_down_new.to(self.ps_down.dtype))
            self.ps_up[layer_idx].copy_(ps_up_new.to(self.ps_up.dtype))
            self.pc_down[layer_idx].copy_(pc_down_new.to(self.pc_down.dtype))
            self.pc_up[layer_idx].copy_(pc_up_new.to(self.pc_up.dtype))
            self.sigma[layer_idx].copy_(sigma_new.to(self.sigma.dtype))
            self.diag_C_down[layer_idx].copy_(diag_C_down_new.to(self.diag_C_down.dtype))
            self.diag_C_up[layer_idx].copy_(diag_C_up_new.to(self.diag_C_up.dtype))

    @torch.inference_mode()
    def apply_mutation(self,
                       adapters_by_layer: list[tuple[torch.Tensor, torch.Tensor]],
                       adapter_id: int,
                       seed: int):
        """Write a single mutated copy (mean + sigma * C^{1/2} z) into adapter_id across all layers."""
        f32 = torch.float32
        gen = torch.Generator(device=self.device).manual_seed(int(seed))

        # Sample z in f32, then cast result to weight dtype when writing
        z_down = torch.randn(self.down_shape, generator=gen, device=self.device, dtype=f32)
        z_up = torch.randn(self.up_shape, generator=gen, device=self.device, dtype=f32)

        noise_down = self.sigma * torch.sqrt(self.diag_C_down).clamp_min(self.min_var ** 0.5) * z_down
        noise_up = self.sigma * torch.sqrt(self.diag_C_up).clamp_min(self.min_var ** 0.5) * z_up

        for layer_idx in range(self.num_layers):
            dW, uW = adapters_by_layer[layer_idx]
            base_down = dW[self.adapter_id].to(f32)
            base_up = uW[self.adapter_id].to(f32)
            dW[adapter_id].copy_((base_down + noise_down[layer_idx]).to(self.w_dtype))
            uW[adapter_id].copy_((base_up + noise_up[layer_idx]).to(self.w_dtype))
