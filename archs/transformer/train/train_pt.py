import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.optimization import Adafactor
import time
import numpy as np
import math
import os
import yaml
import argparse

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

class ModelPT(nn.Module):
    def __init__(self, vocab_size, dim, heads, kv_heads, ff_dim, layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, kv_heads, ff_dim) for _ in range(layers)])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, use_checkpointing=True):
        x = self.embed(x)
        for block in self.blocks:
            if use_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)
        return self.lm_head(x)

def create_batch_generator(bin_path, batch_size, max_len, device):
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    offsets = np.arange(max_len + 1)
    
    while True:
        starts = np.random.randint(0, total_tokens - (max_len + 1), size=batch_size)
        idx_matrix = starts[:, None] + offsets[None, :]
        batch = data[idx_matrix].astype(np.int32)
        x = torch.tensor(batch[:, :-1], dtype=torch.long, device=device)
        y = torch.tensor(batch[:, 1:], dtype=torch.long, device=device)
        yield x, y

def get_lr(step, warmup_steps, total_steps, peak_lr):
    if step < warmup_steps:
        return peak_lr * (step / warmup_steps)
    if step > total_steps:
        return peak_lr / 10.0
    decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return (peak_lr / 10.0) + coeff * (peak_lr - (peak_lr / 10.0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='diamond_1B')
    parser.add_argument('--resume_step', type=int, default=0)
    args = parser.parse_args()

    import os
    with open(os.path.join(os.path.dirname(__file__), "../param.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    if args.model not in config:
        print(f"Error: {args.model} not found in param.yaml")
        exit(1)
        
    p = config[args.model]
    
    BIN_PATH = "data/corpus/token/train_10B.bin"
    VAL_BIN_PATH = "data/corpus/token/val_5M.bin"
    MODEL_SAVE_PATH = p['MODEL_SAVE_PATH'].replace(".weights.h5", ".pt")
    MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", MODEL_SAVE_PATH)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    BATCH_SIZE = p['BATCH_SIZE']
    GRAD_ACCUM_STEPS = p.get('GRAD_ACCUM_STEPS', 1)
    MAX_LEN = p['MAX_LEN']
    PEAK_LR = p['PEAK_LEARNING_RATE']
    TOTAL_STEPS = 50863
    
    WARMUP_STEPS = 1000
    SAVE_EVERY_STEPS = 250
    VAL_EVERY_STEPS = 500
    VAL_STEPS = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"--- Starting/Resuming PyTorch Training: {args.model} ---")
    
    model = ModelPT(
        vocab_size=p['VOCAB_SIZE'],
        dim=p['EMBED_DIM'],
        heads=p['NUM_HEADS'],
        kv_heads=p.get('NUM_KV_HEADS', p['NUM_HEADS']),
        ff_dim=p.get('FF_DIM', p['EMBED_DIM'] * 4),
        layers=p['NUM_TRANSFORMER_BLOCKS']
    ).to(dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32, device=device)
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading pre-trained weights from {MODEL_SAVE_PATH}...")
        try:
            state_dict = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            print("Successfully loaded Pre-trained Base Model!")
        except Exception as e:
            print(f"Error loading state dict: {e}")
            exit()
    else:
        print(f"Starting fresh. Weights will be saved to {MODEL_SAVE_PATH}")

    optimizer = Adafactor(
        model.parameters(),
        lr=0.0,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=1e-2
    )

    if not os.path.exists(BIN_PATH):
        print(f"Training data {BIN_PATH} not found. Skipping dataset load for dry-run.")
        exit(0)

    train_gen = create_batch_generator(BIN_PATH, BATCH_SIZE, MAX_LEN, device)
    val_gen = create_batch_generator(VAL_BIN_PATH, BATCH_SIZE, MAX_LEN, device)

    model.train()
    
    progress_file = MODEL_SAVE_PATH.replace(".pt", "_progress.txt")
    global_update_step = args.resume_step
    
    if global_update_step == 0 and os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as pf:
                content = pf.read().strip()
                if content:
                    parts = content.split(",")
                    if len(parts) >= 1:
                        global_update_step = int(parts[0].strip())
                        print(f"Found progress file! Resuming automatically from step {global_update_step}...")
        except Exception as e:
            print(f"Failed to read progress file: {e}")

    RESTART_WARMUP_STEPS = 50

    print(f"\nTARGET: {TOTAL_STEPS} Updates (Starting from Step {global_update_step})")
    
    try:
        while global_update_step < TOTAL_STEPS:
            start_time = time.time()
            accumulated_loss = 0.0
            
            target_lr = get_lr(global_update_step, WARMUP_STEPS, TOTAL_STEPS, PEAK_LR)
            
            steps_since_restart = global_update_step - (args.resume_step if args.resume_step > 0 else (global_update_step if 'steps_since_restart' not in locals() else 0))
            if 'restart_start_step' not in locals():
                restart_start_step = global_update_step
                
            steps_since_restart = global_update_step - restart_start_step
            if steps_since_restart < RESTART_WARMUP_STEPS:
                lr = target_lr * ((steps_since_restart + 1) / RESTART_WARMUP_STEPS)
            else:
                lr = target_lr

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            
            optimizer.zero_grad()
            
            for micro_step in range(GRAD_ACCUM_STEPS):
                inputs, targets = next(train_gen)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if torch.cuda.is_available() else torch.autocast(device_type='cpu'):
                    logits = model(inputs, use_checkpointing=True)
                    loss = F.cross_entropy(logits.view(-1, p['VOCAB_SIZE']), targets.view(-1), ignore_index=0) / GRAD_ACCUM_STEPS
                
                loss.backward()
                accumulated_loss += loss.item()
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache() 
            
            global_update_step += 1
            step_time = time.time() - start_time
            
            progress_pct = (global_update_step / TOTAL_STEPS) * 100
            
            print(f"Update {global_update_step}/{TOTAL_STEPS} ({progress_pct:.2f}%) | Loss: {accumulated_loss:.4f} | LR: {lr:.2e} | Time: {step_time:.2f}s")
            
            if global_update_step % SAVE_EVERY_STEPS == 0:
                print(f"\n[AUTO SAVE] Saving at {progress_pct:.2f}% progress...")
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                with open(progress_file, "w") as pf:
                    pf.write(f"{global_update_step}, {TOTAL_STEPS}, {progress_pct:.2f}%\n")
                print("Weights and progress secure.\n")
                
            if global_update_step % VAL_EVERY_STEPS == 0:
                print(f"\n[VALIDATION]")
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for _ in range(VAL_STEPS):
                        inputs, targets = next(val_gen)
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16) if torch.cuda.is_available() else torch.autocast(device_type='cpu'):
                            logits = model(inputs, use_checkpointing=False)
                            loss = F.cross_entropy(logits.view(-1, p['VOCAB_SIZE']), targets.view(-1), ignore_index=0)
                        val_loss += loss.item()
                val_loss /= VAL_STEPS
                print(f"Val Loss: {val_loss:.4f}\n")
                model.train()

        print("\n--- CORPUS FULLY COMPLETED! ---")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        if os.path.exists(progress_file):
            os.remove(progress_file)

    except KeyboardInterrupt:
        print("\n\n[PAUSED] Ctrl+C detected. Saving progress...")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        progress_pct = (global_update_step / TOTAL_STEPS) * 100
        
        with open(progress_file, "w") as pf:
            pf.write(f"{global_update_step}, {TOTAL_STEPS}, {progress_pct:.2f}%\n")
            
        print(f"Saved at {progress_pct:.2f}%. Progress written to {progress_file}. Run the script again to resume automatically!")