import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .mlp import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, d_model , n_heads, d_head, d_ff, causal, max_seq_len, use_KV_cache):
        super().__init__()
        self.mha1 = MultiHeadAttention(d_model, n_heads, d_head, causal, max_seq_len, use_KV_cache)
        self.ln1= nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp  = FeedForward(d_model, d_ff)
        
    def forward(self, x):
        y = self.ln1(x)
        attn = self.mha1(y)
        x = x + attn
        y = self.ln2(x)
        mlp = self.mlp(y)
        x = x+mlp
        return x
    
    def forward_cache(self, x):        
        y = self.ln1(x)
        attn = self.mha1.forward_cache(y)
        x = x + attn
        y = self.ln2(x)
        mlp_out = self.mlp(y)
        x = x + mlp_out
        return x