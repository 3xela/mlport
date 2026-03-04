import torch
import torch.nn as nn
from transformer_configs.params_cfg import TransformerConfig
from .layers import TransformerBlock

class TinyTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_layers = config.n_layers
        self.d_ff = config.d_ff
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.d_head = config.d_head
        self.causal = config.causal
        self.use_KV_cache = config.use_KV_cache

        self.token_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, self.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.n_heads, self.d_head, self.d_ff, self.causal, self.max_seq_len, self.use_KV_cache)
            for _ in range(self.n_layers)
        ])
        self.final_ln = nn.LayerNorm(self.d_model)

    def forward(self , tokens):
        B , T = tokens.shape
        positions = torch.arange(T).to(tokens.device)
        embed = self.token_emb(tokens)
        pos = self.pos_emb(positions)[None, : , :]
        x = embed + pos

        for block in self.blocks:
            x = block(x)

        return self.final_ln(x)
    
    def forward_cache(self, x):
        for block in self.blocks:
            x = block.forward_cache(x)
        x = self.final_ln(x)
        return x
    