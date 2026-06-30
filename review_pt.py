# Reasoning for PyTorch model (.pt)
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import argparse
import os
import numpy as np
import sentencepiece as spm
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

def apply_rope(x, cos_buffer, sin_buffer):
    B, S, H, D = x.shape
    cos = cos_buffer[:, :S, :, :].to(x.dtype)
    sin = sin_buffer[:, :S, :, :].to(x.dtype)
    
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    
    res = torch.stack([rx1, rx2], dim=-1)
    return res.view_as(x)

class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

    def forward(self, x, cos, sin):
        B, S, E = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_queries_per_kv > 1:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out)

class SwiGLU(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, ff_dim, bias=False)
        self.w2 = nn.Linear(dim, ff_dim, bias=False)

    def forward(self, x):
        return F.silu(self.w1(x)) * self.w2(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads=None, ff_dim=None):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.att = RoPEMultiHeadAttention(embed_dim, num_heads, num_kv_heads)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ff_dim)
        self.wo = nn.Linear(ff_dim, embed_dim, bias=False)

    def forward(self, x, cos, sin):
        x = x + self.att(self.norm1(x), cos, sin)
        x = x + self.wo(self.ffn(self.norm2(x)))
        return x

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.norm = RMSNorm(embed_dim)

    def forward(self, x):
        return self.norm(self.token_emb(x))

class TiedOutput(nn.Module):
    def __init__(self, vocab_size, embedding_weight):
        super().__init__()
        self.embedding_weight = embedding_weight
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, x):
        return F.linear(x, self.embedding_weight, self.bias)

class PyTorchModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_blocks, num_heads, num_kv_heads=None, ff_dim=None, max_len=768):
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, num_kv_heads, ff_dim)
            for _ in range(num_blocks)
        ])
        self.norm = RMSNorm(embed_dim)
        self.output_layer = TiedOutput(vocab_size, self.embedding.token_emb.weight)
        
        head_dim = embed_dim // num_heads
        pos = torch.arange(max_len, dtype=torch.float32)
        indices = torch.arange(0, head_dim, 2, dtype=torch.float32)
        freqs = 1.0 / (10000.0 ** (indices / head_dim))
        angles = pos[:, None] * freqs[None, :]
        cos = torch.cos(angles)[None, :, None, :]
        sin = torch.sin(angles)[None, :, None, :]
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        cos = self.rope_cos
        sin = self.rope_sin
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        return self.output_layer(x)

def load_or_convert_model(model_name, params):
    model_save_path = params['MODEL_SAVE_PATH']
    converted_pt_path = model_save_path.replace('.weights.h5', '_converted.pt')
    
    vocab_size = params['VOCAB_SIZE']
    max_len = params['MAX_LEN']
    embed_dim = params['EMBED_DIM']
    num_blocks = params['NUM_TRANSFORMER_BLOCKS']
    num_heads = params['NUM_HEADS']
    num_kv = params.get('NUM_KV_HEADS', num_heads)
    ff_dim = params.get('FF_DIM', int(embed_dim * 8 / 3))
    
    pt_model = PyTorchModel(vocab_size, embed_dim, num_blocks, num_heads, num_kv_heads=num_kv, ff_dim=ff_dim, max_len=max_len)
    
    if os.path.exists(converted_pt_path):
        pt_model.load_state_dict(torch.load(converted_pt_path, map_location='cpu'))
        return pt_model
        
    print(f"Converting Keras weights from {model_save_path} to PyTorch...")
    import tensorflow as tf
    from model import build_model as build_tf_model
    
    tf_model = build_tf_model(vocab_size, max_len, embed_dim, num_blocks, num_heads, num_kv_heads=num_kv, ff_dim=ff_dim)
    tf_model.load_weights(model_save_path)
    
    tf_weights = tf_model.weights
    w_idx = 0

    pt_model.embedding.token_emb.weight.data = torch.from_numpy(tf_weights[w_idx].numpy())
    w_idx += 1
    pt_model.embedding.norm.weight.data = torch.from_numpy(tf_weights[w_idx].numpy())
    w_idx += 1

    for i in range(num_blocks):
        block = pt_model.blocks[i]
        block.att.q_proj.weight.data = torch.from_numpy(tf_weights[w_idx].numpy().T)
        w_idx += 1
        block.att.k_proj.weight.data = torch.from_numpy(tf_weights[w_idx].numpy().T)
        w_idx += 1
        block.att.v_proj.weight.data = torch.from_numpy(tf_weights[w_idx].numpy().T)
        w_idx += 1
        block.att.o_proj.weight.data = torch.from_numpy(tf_weights[w_idx].numpy().T)
        w_idx += 1
        block.ffn.w1.weight.data = torch.from_numpy(tf_weights[w_idx].numpy().T)
        w_idx += 1
        block.ffn.w2.weight.data = torch.from_numpy(tf_weights[w_idx].numpy().T)
        w_idx += 1
        block.wo.weight.data = torch.from_numpy(tf_weights[w_idx].numpy().T)
        w_idx += 1
        block.norm1.weight.data = torch.from_numpy(tf_weights[w_idx].numpy())
        w_idx += 1
        block.norm2.weight.data = torch.from_numpy(tf_weights[w_idx].numpy())
        w_idx += 1

    pt_model.norm.weight.data = torch.from_numpy(tf_weights[w_idx].numpy())
    w_idx += 1

    pt_model.output_layer.bias.data = torch.from_numpy(tf_weights[w_idx].numpy())
    w_idx += 1

    assert w_idx == len(tf_weights), f"Used {w_idx} weights, but TF model has {len(tf_weights)} weights"
    
    os.makedirs(os.path.dirname(converted_pt_path), exist_ok=True)
    torch.save(pt_model.state_dict(), converted_pt_path)
    print(f"Saved converted PyTorch weights to {converted_pt_path}.")
    
    return pt_model

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
    parser.add_argument('--model', type=str, default='default')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--tokens', type=int, default=None)
    parser.add_argument('--temp', type=float, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--penalty', type=float, default=None)
    args = parser.parse_args()

    with open("model_param.yaml", 'r') as f:
        config_file = yaml.safe_load(f)

    model_name = config_file['default_model'] if args.model == 'default' else args.model
    params = config_file[model_name]
    gen_params = params.get('generation', {})

    MAX_LEN = params['MAX_LEN']
    TOKENIZER_PATH = "data/tokenizer/tokenizer.model"

    PROMPT = args.prompt if args.prompt else gen_params.get('prompt', "こんにちは")
    TARGET_TOKENS = args.tokens if args.tokens is not None else gen_params.get('max_new_tokens', 100)
    ABS_MAX_TOKENS = 1000
    TEMPERATURE = args.temp if args.temp is not None else gen_params.get('temperature', 0.8)
    TOP_K = args.top_k if args.top_k is not None else gen_params.get('top_k', 40)
    TOP_P = args.top_p if args.top_p is not None else gen_params.get('top_p', 0.9)
    REPETITION_PENALTY = args.penalty if args.penalty is not None else gen_params.get('repetition_penalty', 1.2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading model '{model_name}' to {device}...")

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    model = load_or_convert_model(model_name, params)
    
    if device.type == "mps":
        model.to(dtype=torch.bfloat16, device=device)
    else:
        model.to(device=device)
    model.eval()

    print("\n---Prompt---")
    print(f"Input: {PROMPT}")

    print("\n--- Generating ---")
    sys.stdout.write(f"Result: {PROMPT}")
    sys.stdout.flush()

    generated_ids = sp.encode(PROMPT)
    initial_len = len(generated_ids)
    eos_id = sp.eos_id()

    last_printed_len = len(PROMPT)
    start_time = time.time()

    with torch.no_grad():
        for step in range(ABS_MAX_TOKENS):
            curr_input = generated_ids[-MAX_LEN:]
            input_tensor = torch.tensor([curr_input], dtype=torch.long, device=device)
            
            logits = model(input_tensor)
            next_token_logits = logits[0, -1, :].float().unsqueeze(0)
            
            for gid in set(generated_ids):
                if next_token_logits[0, gid] > 0:
                    next_token_logits[0, gid] /= REPETITION_PENALTY
                else:
                    next_token_logits[0, gid] *= REPETITION_PENALTY
            
            new_tokens_count = len(generated_ids) - initial_len
            if new_tokens_count < TARGET_TOKENS:
                next_token_logits[0, eos_id] = -float('Inf')
            
            next_id = sample(next_token_logits, TEMPERATURE, TOP_K, TOP_P)
            
            if next_id == eos_id:
                break
                
            generated_ids.append(next_id)
            
            full_text = sp.decode(generated_ids)
            new_text = full_text[last_printed_len:]
            sys.stdout.write(new_text)
            sys.stdout.flush()
            last_printed_len = len(full_text)
            
            new_tokens_count = len(generated_ids) - initial_len
            if new_tokens_count >= TARGET_TOKENS:
                if any(mark in new_text for mark in ["。", "！", "？", "!", "?"]):
                    break

    end_time = time.time()
    elapsed = end_time - start_time
    total_generated = len(generated_ids) - initial_len

    print("\n\n--- Finished ---")
    print(f"  Generated tokens: {total_generated}")
    print(f"  Time taken:       {elapsed:.2f} sec")
    print(f"  Tokens per sec:   {total_generated / elapsed:.2f} tokens/s")
    print(f"  Parameters:       Temp={TEMPERATURE}, Top-K={TOP_K}, Top-P={TOP_P}, Penalty={REPETITION_PENALTY}")
    print("-" * 25)
