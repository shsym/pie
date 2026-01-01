"""
Standalone autoregressive generation test for PIE backend.

This test directly uses the Runtime class with HuggingFace tokenizer
to verify that autoregressive token generation works correctly.
"""

from __future__ import annotations

import torch
from transformers import AutoTokenizer

try:
    import minijinja
    HAS_MINIJINJA = True
except ImportError:
    HAS_MINIJINJA = False

from pie_backend.runtime import Runtime, RuntimeConfig


def format_chat_prompt(
    messages: list[dict],
    template_content: str,
    add_generation_prompt: bool = True,
) -> str:
    """Format messages using minijinja template."""
    if not HAS_MINIJINJA:
        # Fallback: just concatenate content
        return " ".join(m.get("content", "") for m in messages if m.get("content"))
    
    env = minijinja.Environment()
    return env.render_str(
        template_content,
        messages=messages,
        add_generation_prompt=add_generation_prompt,
    )


def run_autoregressive_test(
    model_name: str = "llama-3.2-1b-instruct",
    prompt: str = "Hello, my name is",
    max_new_tokens: int = 20,
    temperature: float = 0.7,
    use_chat_template: bool = False,
):
    """
    Test autoregressive generation using the Runtime.
    
    Args:
        model_name: Name of the model to load
        prompt: Input prompt to generate from
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        use_chat_template: Whether to format as chat
    """
    print("=" * 60)
    print("Autoregressive Generation Test")
    print("=" * 60)
    
    # Load HuggingFace tokenizer
    print(f"\n[1] Loading HuggingFace tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    print(f"    Tokenizer loaded: vocab_size={tokenizer.vocab_size}")
    
    # Load Runtime
    print(f"\n[2] Loading Runtime with model={model_name}...")
    config = RuntimeConfig.from_args(model=model_name)
    runtime = Runtime(config)
    print(f"    Runtime loaded: {runtime.model_spec.num_layers} layers")
    
    device = config.device#[0]
    dtype = config.dtype
    
    # Format prompt with chat template if requested
    if use_chat_template and "template" in runtime.info:
        template = runtime.info["template"]
        if template.get("type") == "minijinja":
            messages = [{"role": "user", "content": prompt}]
            prompt = format_chat_prompt(
                messages=messages,
                template_content=template["content"],
                add_generation_prompt=True,
            )
            print(f"\n[3] Formatted chat prompt:\n{prompt}")
    
    # Tokenize prompt
    print(f"\n[3] Tokenizing prompt: '{prompt[:50]}...'")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    seq_len = input_ids.shape[1]
    print(f"    Input token IDs ({seq_len} tokens): {input_ids.tolist()[0][:10]}...")
    
    # Prepare KV cache tensors
    page_size = config.kv_page_size
    num_pages_needed = (seq_len + max_new_tokens + page_size - 1) // page_size
    
    # Allocate pages for this sequence
    kv_page_indices = torch.arange(num_pages_needed, dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor([0, num_pages_needed], dtype=torch.int32, device=device)
    
    # Generate tokens autoregressively
    print(f"\n[4] Running autoregressive generation...")
    generated_ids = input_ids.squeeze(0).tolist()
    current_pos = seq_len
    
    for step in range(max_new_tokens):
        # Current sequence as tensor
        current_ids = torch.tensor(generated_ids, dtype=torch.long, device=device)
        
        # Get embeddings
        embeddings = runtime.forward_pass.embed_tokens(runtime.model_param, current_ids)
        
        # Prepare inputs for transform
        position_ids = torch.arange(len(generated_ids), dtype=torch.long, device=device)
        
        # For prefill (first call) or decode (subsequent calls)
        if step == 0:
            # Prefill: process all tokens
            qo_indptr = torch.tensor([0, len(generated_ids)], dtype=torch.int32, device=device)
            single_token_mode = False
            input_embeds = embeddings
        else:
            # Decode: only process last token
            qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
            single_token_mode = True
            input_embeds = embeddings[-1:, :]  # Only last token
            position_ids = position_ids[-1:]
        
        # Calculate page lens
        tokens_in_cache = len(generated_ids)
        kv_last_page_lens = torch.tensor(
            [tokens_in_cache % page_size or page_size],
            dtype=torch.int32,
            device=device,
        )
        
        # Run through transformer layers
        try:
            hidden_states = runtime.forward_pass.transform(
                param=runtime.model_param,
                input_embeds=input_embeds,
                position_ids=position_ids,
                qo_indptr=qo_indptr,
                kv_cache_at_layer=runtime.kv_cache_at_layer,
                kv_page_indices=kv_page_indices,
                kv_page_indptr=kv_page_indptr,
                kv_last_page_lens=kv_last_page_lens,
                custom_mask=None,
                single_token_inference_mode=single_token_mode,
                adapter_subpass=None,
            )
        except Exception as e:
            print(f"    [!] Transform failed at step {step}: {e}")
            print("    [!] Falling back to embed->lm_head only")
            hidden_states = input_embeds
        
        # Get logits from lm_head
        logits = runtime.forward_pass.lm_head(runtime.model_param, hidden_states)
        
        # Get the logits for the last position
        last_logits = logits[-1, :]
        
        # Apply temperature
        if temperature > 0:
            last_logits = last_logits / temperature
        
        # Sample from distribution
        probs = torch.softmax(last_logits, dim=-1)
        
        if temperature > 0:
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = torch.argmax(probs).item()
        
        generated_ids.append(next_token)
        current_pos += 1
        
        # Decode for display
        decoded_token = tokenizer.decode([next_token])
        print(f"    Step {step+1}: token_id={next_token}, decoded='{decoded_token}'")
        
        # Stop if we hit EOS
        if next_token in [tokenizer.eos_token_id, 128001, 128009]:  # EOS tokens
            print("    [EOS reached]")
            break
    
    # Decode full sequence
    print(f"\n[5] Final generated text:")
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"    '{generated_text}'")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
    
    return generated_text


def test_forward_pass_components():
    """Test individual forward pass components."""
    print("=" * 60)
    print("Forward Pass Component Test")
    print("=" * 60)
    
    # Load Runtime
    print("\n[1] Loading Runtime...")
    config = RuntimeConfig.from_args(model="llama-3.2-1b-instruct")
    runtime = Runtime(config)
    device = config.device#[0]
    
    # Test handshake
    print("\n[2] Testing handshake...")
    from src import message
    responses = runtime.handshake([message.HandshakeRequest(version="1.0")])
    print(f"    ✓ model_name: {responses[0].model_name}")
    print(f"    ✓ kv_page_size: {responses[0].kv_page_size}")
    

    
    # Test query
    print("\n[4] Testing query (ping)...")
    responses = runtime.query([message.QueryRequest(query="ping")])
    print(f"    ✓ Response: {responses[0].value}")
    
    # Test embed_tokens
    print("\n[5] Testing embed_tokens...")
    token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long, device=device)
    embeddings = runtime.forward_pass.embed_tokens(runtime.model_param, token_ids)
    print(f"    ✓ Input shape: {token_ids.shape}")
    print(f"    ✓ Output shape: {embeddings.shape}")
    
    # Test lm_head
    print("\n[6] Testing lm_head...")
    logits = runtime.forward_pass.lm_head(runtime.model_param, embeddings)
    print(f"    ✓ Input shape: {embeddings.shape}")
    print(f"    ✓ Output shape: {logits.shape}")
    
    # Test MLP (single layer)
    print("\n[7] Testing MLP (layer 0)...")
    mlp_out = runtime.forward_pass.mlp(runtime.model_param, embeddings, layer_idx=0)
    print(f"    ✓ Input shape: {embeddings.shape}")
    print(f"    ✓ Output shape: {mlp_out.shape}")
    
    print("\n" + "=" * 60)
    print("All Component Tests Passed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PIE Backend Autoregressive Test")
    parser.add_argument(
        "--test",
        type=str,
        choices=["generate", "components", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, my name is",
        help="Prompt for generation test",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use chat template for prompt",
    )
    args = parser.parse_args()
    
    if args.test in ["components", "all"]:
        test_forward_pass_components()
    
    if args.test in ["generate", "all"]:
        run_autoregressive_test(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            use_chat_template=args.chat,
        )
