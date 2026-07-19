import torch
import torch.nn as nn
from dataclasses import dataclass
from modules.attention import RMSNorm
from modules.encoder_decoder import LinearEncoderBlock, DynamicLatentCompressor, DecoderBlock
from modules.reasoner import AdaptiveReasoning

@dataclass
class LatentBrainConfig:
    vocab_size: int = 32000
    dim: int = 768
    ff_dim: int = 2048
    num_heads: int = 12
    num_kv_heads: int = 4
    encode_layers: int = 2
    reason_layers: int = 8
    decode_layers: int = 4
    compression_ratio: int = 16
    dropout: float = 0.0
    max_position: int = 8192
    act_threshold: float = 0.95
    max_reason_steps: int = 16

class AuxiliaryLoss(nn.Module):
    def __init__(self, read_weight=0.0, step_weight=0.001, window_weight=0.001):
        super().__init__()
        self.read_weight = read_weight
        self.step_weight = step_weight
        self.window_weight = window_weight
    def forward(self, info):
        loss = 0
        loss += (info["read_cost"] * self.read_weight)
        for h in info["history"]:
            loss += (h["window"].mean() * self.window_weight)
        return loss

class LatentBrainLLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.encoder = nn.ModuleList([LinearEncoderBlock(cfg.dim, cfg.ff_dim, ) for _ in range(cfg.encode_layers)])
        self.compressor = DynamicLatentCompressor(cfg.dim, cfg.ff_dim, cfg.num_heads, cfg.compression_ratio)
        self.reasoner = AdaptiveReasoning(cfg)
        self.decoder = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.decode_layers)])
        self.decoder_norm = RMSNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.aux_loss = AuxiliaryLoss()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def encode(self, prompt_ids):
        x = self.embed(prompt_ids)
        for block in self.encoder:
            x = block(x)
        latent = self.compressor(x)
        return latent, x

    def decode(self, latent, target_ids):
        x = self.embed(target_ids)
        for block in self.decoder:
            x = block(x, latent)
        x = self.decoder_norm(x)
        logits = self.lm_head(x)
        return logits

    def forward(self, prompt_ids, target_ids, return_aux=False):
        latent, prompt_memory = self.encode(prompt_ids)
        latent, info = self.reasoner(latent, prompt_memory)
        logits = self.decode(latent, target_ids)
        if not return_aux:
            return logits
        aux = self.aux_loss(info, )
        return logits, {
            "reason_steps": info["reason_steps"],
            "read_cost": info["read_cost"],
            "skip_cost": info["skip_cost"],
            "aux_loss": aux,
            "history": info["history"],
        }

    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens=128, temperature=1.0, top_k=None):
        latent, prompt_memory = self.encode(prompt_ids)
        latent, _ = self.reasoner(latent, prompt_memory)
        output = torch.full((prompt_ids.size(0), 1), 1, dtype=torch.long, device=prompt_ids.device)
        for _ in range(max_new_tokens):
            logits = self.decode(latent, output)
            next_logits = logits[:, -1]
            next_logits /= temperature
            if top_k is not None:
                values, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < values[:, [-1]]] = -1e9
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            output = torch.cat([output, next_token], dim=1)
        return output

def build_latent_brain_from_params(params: dict) -> LatentBrainLLM:
    cfg = LatentBrainConfig(
        vocab_size=params.get("VOCAB_SIZE", 28000),
        dim=params.get("EMBED_DIM", 512),
        ff_dim=params.get("FF_DIM", 1440),
        num_heads=params.get("NUM_HEADS", 8),
        num_kv_heads=params.get("NUM_KV_HEADS", 2),
        encode_layers=params.get("ENCODE_LAYERS", max(1, params.get("NUM_TRANSFORMER_BLOCKS", 8) // 4)),
        reason_layers=params.get("REASON_LAYERS", max(1, params.get("NUM_TRANSFORMER_BLOCKS", 8) // 2)),
        decode_layers=params.get("DECODE_LAYERS", max(1, params.get("NUM_TRANSFORMER_BLOCKS", 8) // 4)),
        compression_ratio=params.get("COMPRESSION_RATIO", 64),
        dropout=params.get("DROPOUT", 0.0)
    )
    return LatentBrainLLM(cfg)
