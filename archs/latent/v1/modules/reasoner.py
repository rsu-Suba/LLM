import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import RMSNorm, SwiGLU, Attention
from controller import EntropyFusedController, PointerTracker, WindowPredictor

class SequenceMixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
    def forward(self, latent):
        residual = latent
        x = self.norm(latent)
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        gate = torch.sigmoid(self.gate(residual))
        x = self.out(F.silu(x))
        return residual + gate * x

class LocalWindowReader(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.memory_norm = RMSNorm(dim)
        self.cross = Attention(dim, heads, causal=False)
        self.gate = nn.Linear(dim, dim, bias=False)
    def build_mask(self, pointer, window, seq_len):
        B, L, _ = pointer.shape
        pos = torch.arange(seq_len, device=pointer.device)
        pos = pos.view(1, 1, seq_len)
        dist = torch.abs(pos - pointer)
        valid = dist <= window
        mask = torch.zeros((B, 1, L, seq_len), device=pointer.device)
        mask.masked_fill_(~valid.unsqueeze(1), -1e9)
        return mask
    def forward(self, latent, prompt, pointer, window, read_probability):
        mask = self.build_mask(pointer, window, prompt.shape[1])
        update = self.cross(self.query_norm(latent), self.memory_norm(prompt), self.memory_norm(prompt), mask)
        gate = torch.sigmoid(self.gate(latent))
        update = (update * gate * read_probability)
        return latent + update

class LatentReasoningBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        self.mixer = SequenceMixer(cfg.dim)
        self.ffn = SwiGLU(cfg.dim, cfg.ff_dim)
        self.controller = EntropyFusedController(cfg.dim)
        self.pointer = PointerTracker(cfg.dim, )
        self.window = WindowPredictor(cfg.dim, )
        self.reader = LocalWindowReader(cfg.dim, cfg.num_heads)
        self.entropy_estimator = nn.Linear(cfg.dim, 1, bias=False)
        
    def forward(self, latent, previous, prompt, pointer):
        latent = self.mixer(latent)
        latent = latent + self.ffn(self.norm2(latent))
        
        entropy = self.entropy_estimator(self.norm1(latent))
            
        read_prob, skip_prob = self.controller(latent, previous, entropy)
        pointer = self.pointer(latent, pointer, prompt.shape[1])
        window = self.window(latent, )
        
        effective_read_prob = read_prob
            
        latent = self.reader(latent, prompt, pointer, window, effective_read_prob)
        return latent, pointer, read_prob, skip_prob, window

class AdaptiveReasoning(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.threshold = cfg.act_threshold
        self.max_steps = cfg.max_reason_steps
        self.blocks = nn.ModuleList([LatentReasoningBlock(cfg) for _ in range(cfg.reason_layers)])
    def forward(self, latent, prompt):
        B, L, D = latent.shape
        pointer = torch.arange(L, device=latent.device, dtype=torch.float32)
        pointer = pointer.view(1, L, 1)
        pointer = pointer.repeat(B, 1, 1)
        previous = latent
        history = []
        read_cost = 0.0
        skip_cost = 0.0
        for step in range(self.max_steps):
            block = self.blocks[step % len(self.blocks)]
            
            if self.training and latent.requires_grad:
                import torch.utils.checkpoint as cp
                next_latent, pointer, read_prob, skip_prob, window = cp.checkpoint(
                    block, latent, previous, prompt, pointer,
                    use_reentrant=False
                )
            else:
                next_latent, pointer, read_prob, skip_prob, window = block(latent, previous, prompt, pointer)
                
            aux = {
                "read_probability": read_prob,
                "skip_probability": skip_prob,
                "window": window,
            }
            
            previous = latent
            latent = next_latent
            
            history.append(aux)
            read_cost = (read_cost + aux["read_probability"].mean())
            skip_cost = (skip_cost + aux["skip_probability"].mean())

        return latent, {
            "reason_steps": self.max_steps,
            "read_cost": read_cost,
            "skip_cost": skip_cost,
            "history": history,
        }

