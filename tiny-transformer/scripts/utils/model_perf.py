import torch
import time
from model.transformer import TinyTransformer


def measure_baseline_decode(iters, model, tokens , config):
    """
    Naive decode baseline:
    For each new token, run a full forward over the entire growing prefix.
    This simulates autoregressive generation WITHOUT KV cache.
    """
    B, T0 = tokens.shape
    device = tokens.device

    # We must ensure we don’t run past max_seq_len
    assert T0 + iters <= config.max_seq_len, (
        f"prefix ({T0}) + decode_steps ({iters}) exceeds max_seq_len "
        f"{config.max_seq_len}"
    )

    # Warmup
    cur = tokens
    with torch.no_grad():
        _ = model(cur)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        for i in range(iters):
            # Run full forward on the prefix
            _ = model(cur)

            # Fake generating a new token (value doesn't matter for timing)
            new_tok = cur[:, -1:].clone()
            cur = torch.cat([cur, new_tok], dim=1)  # grows prefix size

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_ms = (end - start) * 1000.0
    ms_per_token = total_ms / iters
    tokens_per_sec = 1.0 / ((end - start) / iters)

    print(f"[baseline naive decode] {ms_per_token:.4f} ms/token, "
          f"{tokens_per_sec:.1f} tokens/sec")

    return ms_per_token, tokens_per_sec

def measure_kv_cache(iters, model, tokens, config):
    # ---- 1. Model ----
    print("Model created!\n")
    device = tokens.device
    print("Input tokens:", tokens.shape)

    # Optional: reset caches if your model supports it
    if hasattr(model, "reset_cache"):
        model.reset_cache()

    # ---- 3. PREFILL ----
    print("\nRunning KV prefill...")
    B,T = tokens.shape
    with torch.no_grad():
        for t in range(T):
            tok_t = tokens[:, t:t+1]                     # [B,1]
            embed_t = model.token_emb(tok_t)             # [B,1,d_model]

            pos_t = model.pos_emb(
                torch.tensor([t], device=device)
            ).unsqueeze(1)                               # [1,1,d_model]

            x_t = embed_t + pos_t                        # [B,1,d_model]
            model.forward_cache(x_t)                     # Populates caches

    torch.cuda.synchronize()

    # ---- 4. DECODE BENCHMARK ----
    print("Benchmarking KV-cache decode...")

    # use the last token as starting point
    last_token = tokens[:, -1:]     # [B,1]

    start = time.perf_counter()
    with torch.no_grad():
        for i in range(iters):
            pos_idx = min(T - 1 + i, config.max_seq_len - 1)
            embed = model.token_emb(last_token)

            pos = model.pos_emb(
                torch.tensor([pos_idx], device=device)
            ).unsqueeze(1)

            x = embed + pos

            out = model.forward_cache(x)
    torch.cuda.synchronize()
    end = time.perf_counter()

    ms_per_token = (end - start) / iters * 1000
    tokens_per_sec = 1.0 / ((end - start) / iters)

    print(f"KV-cache decode: {ms_per_token:.4f} ms/token")
    print(f"KV-cache decode throughput: {tokens_per_sec:.1f} tokens/sec")

    return ms_per_token, tokens_per_sec