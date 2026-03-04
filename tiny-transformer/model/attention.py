import torch.nn as nn
import torch
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, causal, max_seq_len, use_KV_cache):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.causal = causal
        self.max_seq_len = max_seq_len
        self.use_KV_cache = use_KV_cache
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim = -1)
        self.k_cache = None
        self.v_cache = None
        if self.causal:
            self.register_buffer(
                "causal_mask",
                torch.tril(
                    torch.ones(1, 1, self.max_seq_len, self.max_seq_len, dtype=torch.bool)
                )
            )
        else:
            self.register_buffer("causal_mask", None)

    def forward_cache(self, x):
        B = x.shape[0]
        Q_raw  = self.W_q(x[:, -1, :]).unsqueeze(1)
        K_raw  = self.W_k(x[:, -1, :]).unsqueeze(1)
        V_raw  = self.W_v(x[:, -1, :]).unsqueeze(1)

        Q = Q_raw.view(B, 1, self.n_heads, self.d_head).transpose(1,2)
        K = K_raw.view(B, 1, self.n_heads, self.d_head).transpose(1,2)
        V = V_raw.view(B, 1, self.n_heads, self.d_head).transpose(1,2)
         
        if self.k_cache is None:
            self.k_cache = K
            self.v_cache = V
        else:
            self.k_cache = torch.cat([self.k_cache, K], dim = 2)
            self.v_cache = torch.cat([self.v_cache, V], dim = 2)

        scores = Q @ self.k_cache.transpose(-1,-2)

        attn = self.softmax(scores)
        out = attn @ self.v_cache
        out = out.transpose(1,2).contiguous().reshape(B,1,self.d_model)
        out = self.W_o(out)
        return out
    
    def forward(self, x):
        B, T , _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, T , self.n_heads, self.d_head).transpose(1,2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1,2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1,2)

        scores = Q @ K.transpose(-1,-2)
        scores = scores / math.sqrt(self.d_head)

        if self.causal:
            mask = self.causal_mask[:, :, :T, :T]
            scores = scores.masked_fill(~mask, float('-inf'))


        attn = self.softmax(scores)
        out = attn @ V
        out = out.transpose(1,2).contiguous().reshape(B,T,self.d_model)
        out = self.W_o(out)
        return out
    
    def reset_cache(self):
        self.k_cache = None
        self.v_cache = None