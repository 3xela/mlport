import torch
from configs import TransformerConfig
from model.transformer import TinyTransformer


# ---------------------------
# 1. Create model + dummy input
# ---------------------------

config = TransformerConfig(
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    vocab_size=1000,
    max_seq_len=32,
    use_bias=True,
    dropout=0.0,
    causal=True,
    backend=None
)

model = TinyTransformer(config)
model.eval()

B = 2
T = 10
tokens = torch.randint(0, config.vocab_size, (B, T))

print("Tokens:", tokens)
print("Tokens shape:", tokens.shape)

# ---------------------------
# 2. PREFILL using forward_cache
# ---------------------------

# Reset KV caches (if you have a method for this — otherwise clear manually)
if hasattr(model, "reset_cache"):
    model.reset_cache()

print("\n=== PREFILL PHASE ===")

for t in range(T - 1):  # all tokens except the last
    tok_t = tokens[:, t:t+1]                     # shape [B,1]
    embed_t = model.token_emb(tok_t)             # [B,1,d_model]

    pos_t = model.pos_emb(torch.tensor([t]))     # [1, d_model]
    pos_t = pos_t.unsqueeze(1)                   # [1,1,d_model] broadcast over batch

    x_t = embed_t + pos_t                        # [B,1,d_model]

    out_t = model.forward_cache(x_t)             # populates KV caches
    print(f"Prefill step {t}: x_t -> out_t shape {out_t.shape}")


# ---------------------------
# 3. DECODE: one new token with KV cache
# ---------------------------

print("\n=== DECODE PHASE ===")

last_token = tokens[:, -1:]                      # last prefix token
embed_last = model.token_emb(last_token)         # [B,1,d_model]

pos_last = model.pos_emb(torch.tensor([T - 1]))  # position of last token in prefix
pos_last = pos_last.unsqueeze(1)                 # [1,1,d_model]

x_last = embed_last + pos_last                   # [B,1,d_model]

out = model.forward_cache(x_last)
print("Decode output shape:", out.shape)
