import torch
import flashinfer

# Warm up CUDA context
torch.cuda.init()
_ = torch.zeros(1, device="cuda")

def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()

input = torch.randn([16, 48, 64], device="cuda", dtype=torch.bfloat16)
input_fp8, input_inv_s = to_float8(input, dtype=torch.float8_e4m3fn)

# column major weight
weight = torch.randn([16, 80, 64], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
weight_fp8, weight_inv_s = to_float8(weight, dtype=torch.float8_e4m3fn)

out = flashinfer.bmm_fp8(input_fp8, weight_fp8, input_inv_s, weight_inv_s, torch.bfloat16)

print(f"out.shape: {out.shape}")
print(f"out.dtype: {out.dtype}")
print(f"Expected shape: torch.Size([16, 48, 80])")
print(f"Expected dtype: torch.bfloat16")
print(f"Shape matches: {out.shape == torch.Size([16, 48, 80])}")
print(f"Dtype matches: {out.dtype == torch.bfloat16}")
print(f"Contains NaN: {torch.isnan(out).any().item()}")
print(f"Contains Inf: {torch.isinf(out).any().item()}")
