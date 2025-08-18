import time

import torch
import torch.nn as nn
import math
import rand_mv


class Adapter:
    """
    Abstract base class defining the interface for a LoRA-style adapter.
    """

    adapter_id: int
    rank: int
    alpha: float
    out_features: list[int]
    out_features_indptr: list[int]

    def __init__(self, adapter_id: int, rank: int, alpha: float, out_features: list[int]):
        self.adapter_id = adapter_id
        self.rank = rank
        self.alpha = alpha
        self.out_features = out_features

        self.out_features_indptr = [0]
        for i in range(len(out_features)):
            self.out_features_indptr.append(self.out_features_indptr[-1] + out_features[i])


class CmaesAdapter(Adapter):
    num_layers: int

    population_size: int
    mu_fraction: float
    initial_sigma: float
    min_sigma: float
    min_var: float
    max_var: float

    qkv_down_weight: list[torch.Tensor]
    qkv_up_weight: list[torch.Tensor]
    qkv_down_sigma: list[torch.Tensor]
    qkv_up_sigma: list[torch.Tensor]

    @torch.inference_mode()
    def __init__(self,
                 rank: int,
                 alpha: float,
                 in_features: int,
                 out_features: list[int],
                 num_layers: int,
                 population_size: int,
                 mu_fraction: float,
                 initial_sigma: float,
                 min_sigma: float,
                 min_var: float,
                 max_var: float,
                 device: torch.device,
                 dtype: torch.dtype,
                 ):
        super().__init__(0, rank, alpha, out_features)
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.in_features = in_features
        self.sum_out = int(sum(out_features))

        # CMA-ES default knobs (can be overridden from outside if desired)
        self.population_size = population_size
        self.mu_fraction = mu_fraction
        self.initial_sigma = initial_sigma
        self.min_sigma = min_sigma
        self.min_var = min_var
        self.max_var = max_var

        # Parameter tensors (means) + effective per-weight stdevs used by kernels
        self.qkv_down_weight = []
        self.qkv_up_weight = []
        self.qkv_down_sigma = []
        self.qkv_up_sigma = []

        for _ in range(num_layers):
            # DOWN: (in_features, rank * 3)
            down_cols = rank * len(out_features)
            qkv_down_weight = torch.empty((in_features, down_cols), dtype=dtype, device=device)
            # UP: (rank, d_q + d_k + d_v)
            qkv_up_weight = torch.empty((rank, self.sum_out), dtype=dtype, device=device)

            # Effective elementwise stddev S used by kernels; start at initial_sigma
            qkv_down_sigma = torch.full_like(qkv_down_weight, self.initial_sigma, dtype=dtype, device=device)
            qkv_up_sigma = torch.full_like(qkv_up_weight, self.initial_sigma, dtype=dtype, device=device)

            nn.init.kaiming_uniform_(qkv_down_weight, a=math.sqrt(5))
            nn.init.zeros_(qkv_up_weight)

            self.qkv_down_weight.append(qkv_down_weight)
            self.qkv_up_weight.append(qkv_up_weight)
            self.qkv_down_sigma.append(qkv_down_sigma)
            self.qkv_up_sigma.append(qkv_up_sigma)

        # ===== Diagonal CMA-ES state (float32 for stability) =====
        f32 = torch.float32
        self.d_per_layer = float(in_features * rank * len(out_features) + rank * self.sum_out)

        # Recombination weights
        self.mu = max(1, int(self.population_size * self.mu_fraction))
        ranks = torch.arange(1, self.mu + 1, dtype=f32, device=device)
        logw = torch.log(torch.tensor(self.mu + 0.5, dtype=f32, device=device)) - torch.log(ranks)
        self.weights = (logw / logw.sum()).to(f32)  # (mu,)
        self.weights_reshaped = self.weights.view(-1, 1, 1)
        self.mueff = (self.weights.sum() ** 2) / (self.weights.pow(2).sum())

        # Strategy parameters
        d = self.d_per_layer
        self.cc = (4.0 + self.mueff / d) / (d + 4.0 + 2.0 * self.mueff / d)
        self.cs = (self.mueff + 2.0) / (d + self.mueff + 5.0)
        self.c1 = 2.0 / ((d + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1, 2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((d + 2.0) ** 2 + self.mueff))
        damps_term = torch.sqrt((self.mueff - 1.0) / (d + 1.0)) - 1.0
        self.damps = 1.0 + 2.0 * torch.clamp(damps_term, min=0.0) + self.cs
        self.chiN = (d ** 0.5) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d ** 2))

        # Per-layer scalar step-size (sigma) and paths / diag covariance (var=1 initially)
        self.layer_sigma = [torch.tensor(self.initial_sigma, dtype=f32, device=device) for _ in range(num_layers)]

        self.ps_down = [torch.zeros((in_features, rank * len(out_features)), dtype=f32, device=device) for _ in
                        range(num_layers)]
        self.ps_up = [torch.zeros((rank, self.sum_out), dtype=f32, device=device) for _ in range(num_layers)]
        self.pc_down = [torch.zeros((in_features, rank * len(out_features)), dtype=f32, device=device) for _ in
                        range(num_layers)]
        self.pc_up = [torch.zeros((rank, self.sum_out), dtype=f32, device=device) for _ in range(num_layers)]
        self.diag_C_down = [torch.ones((in_features, rank * len(out_features)), dtype=f32, device=device) for _ in
                            range(num_layers)]
        self.diag_C_up = [torch.ones((rank, self.sum_out), dtype=f32, device=device) for _ in range(num_layers)]

    @torch.inference_mode()
    def update(self,
               scores: list[float],
               seeds: list[int],
               max_sigma: float
               ) -> None:

        if len(scores) != self.population_size or len(seeds) != self.population_size:
            raise ValueError(f"Expected {self.population_size} scores and seeds.")

        device = self.device
        f32 = torch.float32
        eps = 1e-12

        # Select top-µ by score (maximize)
        scores_t = torch.tensor(scores, device=device, dtype=f32)
        seeds_t = torch.tensor(seeds, device=device, dtype=torch.long)
        _, best_idx = torch.sort(scores_t, descending=True)
        top_idx = best_idx[:self.mu]
        top_seeds = seeds_t[top_idx]  # (µ,)

        W = self.weights_reshaped  # (µ,1,1) f32 on device

        # Convenience
        d_q, d_k, d_v = self.out_features
        rank = self.rank

        for layer_idx in range(self.num_layers):
            # ===== Recreate parents' noise matrices (bit-identical to runtime sampling) =====
            # DOWN: split columns into [rank, rank, rank]
            S_down = self.qkv_down_sigma[layer_idx].to(f32)  # (I, 3*rank)
            S_down_q, S_down_k, S_down_v = torch.split(S_down, [rank, rank, rank], dim=-1)

            # Seeds for Q/K/V (match AdapterBuffer)
            Wd_q = rand_mv.batched_randn_generate(
                seeds=top_seeds + layer_idx,
                S=S_down_q, device=device, dtype=torch.float32)
            Wd_k = rand_mv.batched_randn_generate(
                seeds=top_seeds + (layer_idx + 100),
                S=S_down_k, device=device, dtype=torch.float32)
            Wd_v = rand_mv.batched_randn_generate(
                seeds=top_seeds + (layer_idx + 200),
                S=S_down_v, device=device, dtype=torch.float32)
            # (µ, I, rank) each -> concat on O
            Wd = torch.cat([Wd_q, Wd_k, Wd_v], dim=-1)  # (µ, I, 3*rank)

            # UP: split columns into [d_q, d_k, d_v]
            S_up = self.qkv_up_sigma[layer_idx].to(f32)  # (rank, d_q+d_k+d_v)
            S_up_q, S_up_k, S_up_v = torch.split(S_up, [d_q, d_k, d_v], dim=-1)

            Wu_q = rand_mv.batched_randn_generate(
                seeds=top_seeds - layer_idx,
                S=S_up_q, device=device, dtype=torch.float32)
            Wu_k = rand_mv.batched_randn_generate(
                seeds=top_seeds - (layer_idx + 100),
                S=S_up_k, device=device, dtype=torch.float32)
            Wu_v = rand_mv.batched_randn_generate(
                seeds=top_seeds - (layer_idx + 200),
                S=S_up_v, device=device, dtype=torch.float32)
            Wu = torch.cat([Wu_q, Wu_k, Wu_v], dim=-1)  # (µ, rank, d_sum)

            # ===== Recombine means (parents = mean + noise) =====
            mean_down_old = self.qkv_down_weight[layer_idx].to(f32)  # (I, 3*rank)
            mean_up_old = self.qkv_up_weight[layer_idx].to(f32)  # (rank, d_sum)

            parents_down = mean_down_old.unsqueeze(0) + Wd  # (µ, I, 3*rank)
            parents_up = mean_up_old.unsqueeze(0) + Wu  # (µ, rank, d_sum)

            mean_down_new = torch.sum(parents_down * W, dim=0)  # (I, 3*rank)
            mean_up_new = torch.sum(parents_up * W, dim=0)  # (rank, d_sum)

            # Write back (cast to model dtype)
            self.qkv_down_weight[layer_idx].copy_(mean_down_new.to(self.dtype))
            self.qkv_up_weight[layer_idx].copy_(mean_up_new.to(self.dtype))

            # ===== Step-size path (ps) and covariance path (pc) updates =====
            sigma_l = self.layer_sigma[layer_idx].to(f32)  # scalar
            std_down = torch.sqrt(self.diag_C_down[layer_idx]).clamp_min(self.min_var ** 0.5)  # (I, 3*rank)
            std_up = torch.sqrt(self.diag_C_up[layer_idx]).clamp_min(self.min_var ** 0.5)  # (rank, d_sum)

            # Normalized steps
            step_down = (mean_down_new - mean_down_old) / (sigma_l + eps)  # (I, 3*rank)
            step_up = (mean_up_new - mean_up_old) / (sigma_l + eps)  # (rank, d_sum)

            Cinv_sqrt_step_down = step_down / (std_down + eps)
            Cinv_sqrt_step_up = step_up / (std_up + eps)

            c_ps = torch.sqrt(self.cs * (2.0 - self.cs) * self.mueff).to(f32)
            ps_down_new = (1.0 - self.cs) * self.ps_down[layer_idx] + c_ps * Cinv_sqrt_step_down
            ps_up_new = (1.0 - self.cs) * self.ps_up[layer_idx] + c_ps * Cinv_sqrt_step_up

            # Combined norm for step-size adaptation
            norm_ps = torch.sqrt(ps_down_new.pow(2).sum() + ps_up_new.pow(2).sum())
            sigma_new = sigma_l * torch.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1.0))
            if max_sigma and max_sigma > 0:
                sigma_new = torch.clamp(sigma_new, min=self.min_sigma, max=float(max_sigma))
            else:
                sigma_new = torch.clamp(sigma_new, min=self.min_sigma)

            # Heaviside for covariance path (no iteration; steady-state denom ~ 1)
            h_sigma = (norm_ps < (1.4 + 2.0 / (self.d_per_layer + 1.0)) * self.chiN).to(f32)

            c_pc = torch.sqrt(self.cc * (2.0 - self.cc) * self.mueff).to(f32)
            pc_down_new = (1.0 - self.cc) * self.pc_down[layer_idx] + h_sigma * c_pc * step_down
            pc_up_new = (1.0 - self.cc) * self.pc_up[layer_idx] + h_sigma * c_pc * step_up

            # ===== Diagonal covariance update (rank-1 + rank-µ) =====
            # y_parent = (parent - mean_old) / sigma_l  (shape: like params)
            y_down_parents = (parents_down - mean_down_old.unsqueeze(0)) / (sigma_l + eps)
            y_up_parents = (parents_up - mean_up_old.unsqueeze(0)) / (sigma_l + eps)

            rank1_down = self.c1 * pc_down_new.pow(2)
            rank1_up = self.c1 * pc_up_new.pow(2)
            rankmu_down = self.cmu * torch.sum((y_down_parents.pow(2)) * W, dim=0)
            rankmu_up = self.cmu * torch.sum((y_up_parents.pow(2)) * W, dim=0)

            diag_C_down_new = ((1.0 - self.c1 - self.cmu) * self.diag_C_down[layer_idx]
                               + rank1_down + rankmu_down).clamp(self.min_var, self.max_var)
            diag_C_up_new = ((1.0 - self.c1 - self.cmu) * self.diag_C_up[layer_idx]
                             + rank1_up + rankmu_up).clamp(self.min_var, self.max_var)

            # ===== Persist state and refresh runtime S tensors =====
            self.ps_down[layer_idx].copy_(ps_down_new)
            self.ps_up[layer_idx].copy_(ps_up_new)
            self.pc_down[layer_idx].copy_(pc_down_new)
            self.pc_up[layer_idx].copy_(pc_up_new)
            self.diag_C_down[layer_idx].copy_(diag_C_down_new)
            self.diag_C_up[layer_idx].copy_(diag_C_up_new)
            self.layer_sigma[layer_idx] = sigma_new

            # Effective stdevs used by kernels: S = sigma * sqrt(diag_C)
            Sd_new = (sigma_new * torch.sqrt(diag_C_down_new)).to(self.dtype)
            Su_new = (sigma_new * torch.sqrt(diag_C_up_new)).to(self.dtype)
            self.qkv_down_sigma[layer_idx].copy_(Sd_new)
            self.qkv_up_sigma[layer_idx].copy_(Su_new)


class AdapterBuffer:
    """
    Utility class to manage and apply a buffer of adapters in a batched context.
    (This class was provided as part of the original context.)
    """
    adapter: CmaesAdapter
    seeds: torch.Tensor

    def __init__(self,
                 adapter: CmaesAdapter,
                 seeds: torch.Tensor,
                 ):
        self.adapter = adapter
        self.seeds = seeds

    @torch.inference_mode()
    def compute_lora_delta(self, layer_idx: int, x: torch.Tensor) -> list[torch.Tensor]:
        assert x.shape[0] == self.seeds.shape[0], "Batch size must match seeds."
        B = x.shape[0]
        rank = self.adapter.rank

        # Short-hands
        out_indptr = self.adapter.out_features_indptr  # built from [d_q, d_k, d_v]
        d_q, d_k, d_v = self.adapter.out_features

        # ===== 1) Noise paths =====
        # DOWN noise uses 3 equal chunks of size `rank` each (Q/K/V).
        Sd = self.adapter.qkv_down_sigma[layer_idx]  # (in_features, 3*rank)
        Sd_q, Sd_k, Sd_v = torch.split(Sd, [rank, rank, rank], dim=-1)

        q_noise_down = rand_mv.batched_randn_matmul(
            x,
            seeds=self.seeds + layer_idx,
            S=Sd_q,
            out_dtype=x.dtype,
        )
        k_noise_down = rand_mv.batched_randn_matmul(
            x,
            seeds=self.seeds + (layer_idx + 100),
            S=Sd_k,
            out_dtype=x.dtype,
        )
        v_noise_down = rand_mv.batched_randn_matmul(
            x,
            seeds=self.seeds + (layer_idx + 200),
            S=Sd_v,
            out_dtype=x.dtype,
        )

        # UP noise uses column slices [d_q, d_k, d_v].
        Su = self.adapter.qkv_up_sigma[layer_idx]  # (rank, d_q+d_k+d_v)
        Su_q = Su[:, out_indptr[0]:out_indptr[1]].contiguous()
        Su_k = Su[:, out_indptr[1]:out_indptr[2]].contiguous()
        Su_v = Su[:, out_indptr[2]:out_indptr[3]].contiguous()

        q_noise_up = rand_mv.batched_randn_matmul(
            q_noise_down,
            seeds=self.seeds - layer_idx,
            S=Su_q,
            out_dtype=x.dtype,
        )
        k_noise_up = rand_mv.batched_randn_matmul(
            k_noise_down,
            seeds=self.seeds - (layer_idx + 100),
            S=Su_k,
            out_dtype=x.dtype,
        )
        v_noise_up = rand_mv.batched_randn_matmul(
            v_noise_down,
            seeds=self.seeds - (layer_idx + 200),
            S=Su_v,
            out_dtype=x.dtype,
        )

        # ===== 2) Mean paths (deterministic LoRA delta) =====
        # DOWN mean: (B, in_features) @ (in_features, 3*rank) -> split into 3 x (B, rank)
        qkv_down = x @ self.adapter.qkv_down_weight[layer_idx]
        q_down, k_down, v_down = torch.split(qkv_down, [rank, rank, rank], dim=-1)

        # UP mean: take column slices for Q/K/V
        W_up = self.adapter.qkv_up_weight[layer_idx]  # (rank, d_q+d_k+d_v)
        Wq = W_up[:, out_indptr[0]:out_indptr[1]]  # (rank, d_q)
        Wk = W_up[:, out_indptr[1]:out_indptr[2]]  # (rank, d_k)
        Wv = W_up[:, out_indptr[2]:out_indptr[3]]  # (rank, d_v)

        q_mean = q_down @ Wq
        k_mean = k_down @ Wk
        v_mean = v_down @ Wv

        # ===== 3) Combine mean + noise =====
        q_final = q_mean + q_noise_up
        k_final = k_mean + k_noise_up
        v_final = v_mean + v_noise_up

        return [q_final, k_final, v_final]
