import torch
import torch.nn.functional as F
import time
import pandas as pd

# -----------------------------------------------------------------------------
# 1. The Three Implementations
# -----------------------------------------------------------------------------


def method_a_original(logits, temperatures):
    """
    Original Approach: Masking + Indexing + CPU Syncs (simulated logic)
    """
    _SAMPLING_EPS = 1e-5
    probs = torch.empty_like(logits)

    # Simulate the check for greedy/non-greedy (often involves item() calls in worst case)
    # We keep it minimal here to be charitable to the original implementation
    greedy_mask = temperatures.squeeze() < _SAMPLING_EPS

    # Non-greedy path
    if (~greedy_mask).any():
        non_greedy_indices = (~greedy_mask).nonzero(as_tuple=True)[0]
        non_greedy_temps = temperatures[non_greedy_indices]
        non_greedy_logits = logits[non_greedy_indices]
        scaled_logits = non_greedy_logits / non_greedy_temps
        probs[non_greedy_indices] = torch.softmax(scaled_logits, dim=-1)

    # Greedy path
    if greedy_mask.any():
        greedy_indices = greedy_mask.nonzero(as_tuple=True)[0]
        greedy_logits = logits[greedy_indices]
        greedy_argmax = greedy_logits.argmax(dim=-1)
        greedy_probs = torch.zeros_like(greedy_logits)
        greedy_probs.scatter_(1, greedy_argmax.unsqueeze(1), 1.0)
        probs[greedy_indices] = greedy_probs

    return probs


def method_b_unsafe(logits, temperatures):
    """
    Unsafe Approach: Clamping + Single Div (Code B)
    """
    # Clamp and divide blindly
    scaled_logits = logits / torch.clamp(temperatures, min=1e-6)
    probs = torch.softmax(scaled_logits, dim=-1)
    return probs


def method_c_optimized(logits, temperatures, greedy_threshold=1e-5):
    """
    Optimized Approach: Branchless safe_scaled_softmax
    """
    greedy_mask = temperatures < greedy_threshold

    # Branchless logic
    safe_temps = torch.where(greedy_mask, 1.0, temperatures)
    scaled_logits = logits / safe_temps
    probs_sampling = torch.softmax(scaled_logits, dim=-1)

    greedy_indices = logits.argmax(dim=-1)
    probs_greedy = F.one_hot(greedy_indices, num_classes=logits.shape[-1])
    probs_greedy = probs_greedy.to(dtype=logits.dtype)

    return torch.where(greedy_mask, probs_greedy, probs_sampling)


def method_d_hybrid(logits, temperatures):
    """
    Hybrid Approach: Best of both worlds.
    - Small Batch (<64): Use Branchless (Avoids CPU syncs/launch overhead)
    - Large Batch (>=64): Use Sync-Free Masking (Saves Memory Bandwidth)
    """
    if logits.shape[0] < 64:
        return method_c_optimized(logits, temperatures)

    # Sync-Free Masking for Large Batches
    # Avoids 'if .any()' check which syncs CPU
    _SAMPLING_EPS = 1e-5
    probs = torch.empty_like(logits)

    greedy_mask = temperatures.squeeze() < _SAMPLING_EPS

    # Blindly compute indices (Launch overhead is negligible for large batch)
    non_greedy_indices = (~greedy_mask).nonzero(as_tuple=True)[0]
    greedy_indices = greedy_mask.nonzero(as_tuple=True)[0]

    # Path 1: Sampling
    ng_logits = logits[non_greedy_indices]
    ng_temps = temperatures[non_greedy_indices]
    # We rely on PyTorch to handle empty tensors efficiently
    probs[non_greedy_indices] = torch.softmax(ng_logits / ng_temps, dim=-1)

    # Path 2: Greedy
    g_logits = logits[greedy_indices]
    g_argmax = g_logits.argmax(dim=-1)
    g_probs = torch.zeros_like(g_logits)
    g_probs.scatter_(1, g_argmax.unsqueeze(1), 1.0)
    probs[greedy_indices] = g_probs

    return probs


# -----------------------------------------------------------------------------
# 2. Benchmark Harness
# -----------------------------------------------------------------------------


def benchmark(batch_sizes, vocab_size=32000, num_runs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on: {torch.cuda.get_device_name(0)}")
    print(f"Vocab Size: {vocab_size} | Runs per batch: {num_runs}")
    print("-" * 65)

    results = []

    for bs in batch_sizes:
        # Prepare Data
        logits = torch.randn(bs, vocab_size, device=device, dtype=torch.float16)

        # Mix of greedy (0.0) and sampling (0.7-1.0) temperatures
        temps = torch.rand(bs, 1, device=device, dtype=torch.float16)
        # Force 50% to be effectively greedy
        temps[: bs // 2] = 0.0
        # Make the rest normal sampling temps
        temps[bs // 2 :] += 0.1

        # Define runners
        methods = {
            "Original (Masking)": method_a_original,
            "Unsafe (Clamping)": method_b_unsafe,
            "Optimized (Branchless)": method_c_optimized,
            "Hybrid (Best)": method_d_hybrid,
        }

        row = {"Batch Size": bs}

        for name, func in methods.items():
            # Warmup
            for _ in range(10):
                _ = func(logits, temps)
            torch.cuda.synchronize()

            # Timing
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            for _ in range(num_runs):
                _ = func(logits, temps)
            end_event.record()
            torch.cuda.synchronize()

            # Calculate avg latency in microseconds
            latency_ms = start_event.elapsed_time(end_event) / num_runs
            row[name] = latency_ms

        results.append(row)

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# 3. Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print(
            "CUDA not detected. Benchmarking on CPU is not recommended for this test."
        )
    else:
        # Batch sizes: single user -> heavy batched inference
        batches = [1, 8, 32, 128, 512, 1024]
        df = benchmark(batches, vocab_size=128000)  # 128k vocab (e.g. Llama-3 / GPT-4)

        print("\nLatency (ms) - Lower is better:")
        print(df.round(3).to_string(index=False))

        # Calculate Speedup of Optimized vs Original
        # Calculate Speedup of Optimized vs Original
        print("\nSpeedup (Optimized vs Original):")
        speedups = df["Original (Masking)"] / df["Hybrid (Best)"]
        for bs, speedup in zip(df["Batch Size"], speedups):
            print(f"Batch {bs}: {speedup:.2f}x faster")
