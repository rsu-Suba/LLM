import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import RMSNorm, SwiGLU, Attention

class LinearEncoderBlock(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.pwconv = nn.Conv1d(dim, dim, kernel_size=1)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_dim)
    def forward(self, x):
        residual = x
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.dwconv(y)
        y = self.pwconv(y)
        y = y.transpose(1, 2)
        x = residual + F.silu(y)
        x = x + self.ffn(self.norm2(x))
        return x

class DynamicLatentCompressor(nn.Module):
    def __init__(self, dim, ff_dim, heads, compression_ratio):
        super().__init__()
        self.ratio = compression_ratio
        self.pool = nn.Conv1d(dim, dim, kernel_size=compression_ratio, stride=compression_ratio)
        self.query_norm = RMSNorm(dim)
        self.memory_norm = RMSNorm(dim)
        self.cross = Attention(dim, heads, causal=False)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_dim)
    def forward(self, token_memory):
        B, S, D = token_memory.shape
        pad = (self.ratio - (S % self.ratio)) % self.ratio
        if pad:
            token_memory = F.pad(token_memory, (0, 0, 0, pad))
        pooled = self.pool(token_memory.transpose(1, 2))
        latent = pooled.transpose(1, 2)
        latent = latent + self.cross(self.query_norm(latent), self.memory_norm(token_memory))
        latent = latent + self.ffn(self.ffn_norm(latent))
        return latent

class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_norm = RMSNorm(cfg.dim)
        self.self_attn = Attention(cfg.dim, cfg.num_heads, cfg.num_kv_heads, causal=True)
        self.cross_norm = RMSNorm(cfg.dim)
        self.cross_attn = Attention(cfg.dim, cfg.num_heads, causal=False)
        self.ffn_norm = RMSNorm(cfg.dim)
        self.ffn = SwiGLU(cfg.dim, cfg.ff_dim)
    def forward(self, x, latent):
        x = x + self.self_attn(self.self_norm(x))
        x = x + self.cross_attn(self.cross_norm(x), latent, latent)
        x = x + self.ffn(self.ffn_norm(x))
        return x

