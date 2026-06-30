# Reasoning for PyTorch 1B model, Apple Silicon
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import argparse
import sys
import time

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_f32 = x.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        return (x_f32 * torch.rsqrt(variance + self.eps)).type_as(x) * self.weight

def apply_rotary_emb(x, head_dim):
    seq_len = x.shape[2]
    pos = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim))
    sincos = torch.einsum("i,j->ij", pos, inv_freq)
    sin, cos = torch.sin(sincos)[None, None, :, :], torch.cos(sincos)[None, None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    sin, cos = sin.to(x.dtype), cos.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_emb(q, self.head_dim), apply_rotary_emb(k, self.head_dim)
        num_repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(num_repeat, dim=1)
        v = v.repeat_interleave(num_repeat, dim=1)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(attn_out.transpose(1, 2).contiguous().view(B, S, D))

class SwiGLU(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(dim, ff_dim, bias=False)
        self.w3 = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, ff_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Diamond1B(nn.Module):
    def __init__(self, vocab_size=28000, dim=2048, heads=16, kv_heads=4, ff_dim=5504, layers=20):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, kv_heads, ff_dim) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, x, use_checkpointing=False):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)

def sample(logits, temp, top_k, top_p):
    if temp == 0.0:
        return torch.argmax(logits, dim=-1).item()
    logits = logits / temp
    if top_k > 0:
        top_k_val, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < top_k_val[..., -1, None]] = -float('Inf')
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[0, indices_to_remove] = -float('Inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default="宇宙の果てには、")
    parser.add_argument('--tokens', type=int, default=40)
    parser.add_argument('--temp', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=45)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--penalty', type=float, default=1.1)
    args = parser.parse_args()

    MODEL_SAVE_PATH = "model/diamond_1B.pt"
    TOKENIZER_PATH = "data/tokenizer/tokenizer.model"
    MAX_LEN = 512

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading Diamond 1B (10-Hour Checkpoint) to {device}...")

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    model = Diamond1B()

    raw_state_dict = torch.load(MODEL_SAVE_PATH, map_location="cpu", weights_only=False)
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
    model.load_state_dict(clean_state_dict)

    model.to(dtype=torch.bfloat16, device=device)
    model.eval()

    print("\n---Prompt---")
    print(f"Input: {args.prompt}")
    print("\n--- Generating ---")
    sys.stdout.write(f"Result: {args.prompt}")
    sys.stdout.flush()

    generated_ids = sp.encode(args.prompt)
    initial_len = len(generated_ids)
    eos_id = sp.eos_id()

    last_printed_len = len(args.prompt)
    start_time = time.time()

    with torch.no_grad():
        for step in range(args.tokens):
            curr_input = generated_ids[-MAX_LEN:]
            input_tensor = torch.tensor([curr_input], dtype=torch.long, device=device)
            
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :].float().unsqueeze(0)
            
            for gid in set(generated_ids):
                if next_token_logits[0, gid] > 0:
                    next_token_logits[0, gid] /= args.penalty
                else:
                    next_token_logits[0, gid] *= args.penalty
            
            next_token_logits[0, eos_id] = -float('Inf')
            next_id = sample(next_token_logits, args.temp, top_k=args.top_k, top_p=args.top_p)
            generated_ids.append(next_id)
            
            full_text = sp.decode(generated_ids)
            new_text = full_text[last_printed_len:]
            sys.stdout.write(new_text)
            sys.stdout.flush()
            last_printed_len = len(full_text)

    elapsed = time.time() - start_time
    total_generated = len(generated_ids) - initial_len

    print("\n\n--- Finished ---")
    print(f"  Generated tokens: {total_generated}")
    print(f"  Time taken:       {elapsed:.2f} sec")
    print(f"  Speed:            {total_generated / elapsed:.2f} tokens/s")
    print("-" * 25)