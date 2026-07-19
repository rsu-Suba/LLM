import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        x_fp32 = x.float()
        var = x_fp32.pow(2).mean(-1, keepdim=True)
        x = x_fp32 * torch.rsqrt(var + self.eps)
        return x.type_as(self.weight) * self.weight

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x):
        seq = x.shape[1]
        return (self.cos_cached[:seq], self.sin_cached[:seq])

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(hidden, dim, bias=False)
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads=None, causal=False, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = (num_heads if num_kv_heads is None else num_kv_heads)
        self.head_dim = dim // num_heads
        self.causal = causal
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)
    def forward(self, query, key=None, value=None, mask=None):
        is_cross = key is not None
        if key is None:
            key = query
        if value is None:
            value = key
        B = query.shape[0]
        Q = query.shape[1]
        K = key.shape[1]
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(B, Q, self.num_heads, self.head_dim)
        k = k.view(B, K, self.num_kv_heads, self.head_dim)
        v = v.view(B, K, self.num_kv_heads, self.head_dim)
        if not is_cross:
            cos, sin = self.rope(query)
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        repeat = self.num_heads // self.num_kv_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=self.dropout.p if self.training else 0.0)
        else:
            score = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.causal:
                causal_mask = torch.triu(torch.ones(Q, K, device=q.device), diagonal=1, ).bool()
                score.masked_fill_(causal_mask[None, None], -1e9)
            
            score = score + mask
            attn = F.softmax(score.float(), dim=-1).type_as(score)
            attn = self.dropout(attn)
            out = attn @ v
            
        out = (out.transpose(1, 2).reshape(B, Q, self.dim))
        return self.out_proj(out)

