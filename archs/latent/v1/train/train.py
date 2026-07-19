import argparse
import math
import os
import sys
import time
import threading
from torch.utils.tensorboard import SummaryWriter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from transformers.optimization import Adafactor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from model import LatentBrainLLM, LatentBrainConfig


class PrefetchBatchGenerator:
    def __init__(self, bin_path, batch_size, prompt_len, target_len, device):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)
        self.batch_size = batch_size
        self.prompt_len = prompt_len
        self.target_len = target_len
        self.chunk_len = prompt_len + target_len + 1
        self.offsets = np.arange(self.chunk_len)
        self.device = device
        self.use_cuda = device.type == "cuda"
        
        segment_size = self.total_tokens // self.batch_size
        self.pointers = np.array([i * segment_size + np.random.randint(0, 100000) for i in range(self.batch_size)])

        self._prefetch_batch = None
        self._prefetch_ready = threading.Event()
        self._prefetch_request = threading.Event()
        self._stop = False
        self._thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self._thread.start()
        
        self._prefetch_request.set()

        print(f"Streaming sequentially from {bin_path} ({self.total_tokens:,} tokens) [prefetch=on]")

    def _generate_one(self):
        for i in range(self.batch_size):
            if self.pointers[i] + self.chunk_len > self.total_tokens:
                self.pointers[i] = np.random.randint(0, self.total_tokens // 2)

        idx_matrix = self.pointers[:, None] + self.offsets[None, :]
        batch = torch.from_numpy(self.data[idx_matrix].astype(np.int32))
        
        self.pointers += self.chunk_len

        if self.use_cuda:
            prompt = batch[:, : self.prompt_len].pin_memory()
            target_in = batch[:, self.prompt_len : self.prompt_len + self.target_len].pin_memory()
            labels = batch[:, self.prompt_len + 1 : self.prompt_len + self.target_len + 1].pin_memory()
        else:
            prompt = batch[:, : self.prompt_len]
            target_in = batch[:, self.prompt_len : self.prompt_len + self.target_len]
            labels = batch[:, self.prompt_len + 1 : self.prompt_len + self.target_len + 1]

        return prompt, target_in, labels

    def _prefetch_loop(self):
        while not self._stop:
            self._prefetch_request.wait()
            self._prefetch_request.clear()
            if self._stop:
                break
            self._prefetch_batch = self._generate_one()
            self._prefetch_ready.set()

    def __next__(self):
        self._prefetch_ready.wait()
        self._prefetch_ready.clear()
        prompt, target_in, labels = self._prefetch_batch
        self._prefetch_request.set()
        if self.use_cuda:
            prompt = prompt.to(self.device, non_blocking=True).long()
            target_in = target_in.to(self.device, non_blocking=True).long()
            labels = labels.to(self.device, non_blocking=True).long()
        else:
            prompt = prompt.long()
            target_in = target_in.long()
            labels = labels.long()
        return prompt, target_in, labels

    def __iter__(self):
        return self

    def stop(self):
        self._stop = True
        self._prefetch_request.set()


def get_lr(step, warmup_steps, total_steps, peak_lr):
    if step < warmup_steps:
        return peak_lr * float(step + 1) / max(1, warmup_steps)
    progress = min(1.0, float(step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * 0.1 + cosine * (peak_lr - peak_lr * 0.1)


def evaluate(model, val_gen, val_steps, vocab_size, device_type):
    model.eval()
    total_loss = 0.0
    total_gate = 0.0
    total_p_skip = 0.0
    total_p_read = 0.0
    with torch.no_grad():
        for _ in range(val_steps):
            prompt, target_in, labels = next(val_gen)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                logits, aux = model(prompt, target_in, return_aux=True)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), labels.reshape(-1), ignore_index=0)
            total_loss += loss.item()
            
            s_cost = aux.get("skip_cost", 0.0)
            if hasattr(s_cost, "item"): s_cost = s_cost.item()
            r_cost = aux.get("read_cost", 0.0)
            if hasattr(r_cost, "item"): r_cost = r_cost.item()
            
            total_p_skip += s_cost / max(1, aux.get("reason_steps", 10))
            total_p_read += r_cost / max(1, aux.get("reason_steps", 10))
    model.train()
    return (
        total_loss / val_steps,
        total_p_skip / val_steps,
        total_p_read / val_steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="iron")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "../param.yaml"), help="設定ファイルのパス")
    parser.add_argument("--train_bin", type=str, default="data/corpus/token/train_wiki.bin")
    parser.add_argument("--val_bin", type=str, default="data/corpus/token/val_5M.bin")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--stop_step", type=int, default=None, help="指定したステップに到達したら保存して安全に終了する")
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--val_every", type=int, default=1000)
    parser.add_argument("--val_steps", type=int, default=20)
    parser.add_argument("--start_step", type=int, default=0, help="再開時の初期ステップ")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no_compile", action="store_true", help="torch.compileを無効化")
    parser.add_argument("--batch_size", type=int, default=0, help="バッチサイズをオーバーライド(0でyaml値を使用)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.model not in config:
        raise ValueError(f"{args.model} not found in {args.config}")
    params = config[args.model]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    batch_size = args.batch_size if args.batch_size > 0 else params["BATCH_SIZE"]
    prompt_len = params.get("PROMPT_LEN", params["MAX_LEN"] // 2)
    target_len = params.get("TARGET_LEN", params["MAX_LEN"] // 2)
    grad_accum_steps = params.get("GRAD_ACCUM_STEPS", 1)
    peak_lr = params["PEAK_LEARNING_RATE"]
    save_path = params["MODEL_SAVE_PATH"].replace(".weights.h5", ".pt")
    save_path = os.path.join(os.path.dirname(args.config), save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"--- Train brain model: {args.model} ---")
    print(f"Device: {device} | prompt_len={params.get('PROMPT_LEN', 128)} | target_len={params.get('TARGET_LEN', 64)}")
    
    cfg = LatentBrainConfig(
        vocab_size=params.get("VOCAB_SIZE", 28000),
        dim=params.get("EMBED_DIM", 768),
        ff_dim=params.get("FF_DIM", 2048),
        num_heads=params.get("NUM_HEADS", 12),
        num_kv_heads=params.get("NUM_KV_HEADS", 4),
        encode_layers=params.get("ENCODE_LAYERS", 4),
        reason_layers=params.get("REASON_LAYERS", 8),
        decode_layers=params.get("DECODE_LAYERS", 4),
        compression_ratio=params.get("COMPRESSION_RATIO", 16),
        dropout=params.get("DROPOUT", 0.0),
        act_threshold=params.get("ACT_THRESHOLD", 0.95),
        max_reason_steps=params.get("MAX_REASON_STEPS", 16)
    )
    model = LatentBrainLLM(cfg).to(device=device, dtype=dtype)
    
    if args.resume and os.path.exists(save_path):
        print(f"Loading {save_path} with strict=False (resuming from pre-trained weights)...")
        state_dict = torch.load(save_path, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys (these will be randomly initialized): {missing}")
        if unexpected:
            print(f"Unexpected keys in state dict: {unexpected}")

    if device_type == "cuda" and not args.no_compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Compilation will happen on first forward pass.")
    else:
        print("torch.compile disabled.")

    optimizer = Adafactor(
        model.parameters(),
        lr=peak_lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.1
    )

    train_gen = PrefetchBatchGenerator(args.train_bin, batch_size, prompt_len, target_len, device)
    val_gen = PrefetchBatchGenerator(args.val_bin, batch_size, prompt_len, target_len, device)

    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print("TF32 enabled | cuDNN benchmark enabled")

    writer = SummaryWriter(log_dir=f"runs/{args.model}")
    
    model.train()
    try:
        for step in range(args.start_step, args.steps):
            if args.stop_step is not None and step >= args.stop_step:
                print(f"Reached stop_step {args.stop_step}. Saving and exiting gracefully...")
                break
                
            start = time.time()
            lr = get_lr(step, args.warmup_steps, args.steps, peak_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for micro in range(grad_accum_steps):
                prompt, target_in, labels = next(train_gen)
                with torch.autocast(device_type=device_type, dtype=dtype):
                    logits, aux = model(prompt, target_in, return_aux=True)
                    ce_loss = F.cross_entropy(
                        logits.reshape(-1, params["VOCAB_SIZE"]),
                        labels.reshape(-1),
                        ignore_index=0,
                    )
                    loss = (ce_loss + aux["aux_loss"]) / grad_accum_steps
                
                    read_cost = aux.get("read_cost", torch.tensor(0.0, device=device_type))
                    avg_read_prob = read_cost / max(1, aux.get("reason_steps", 10))
                
                    penalty_low = F.relu(0.05 - avg_read_prob) ** 2
                    penalty_high = F.relu(avg_read_prob - 0.40) ** 2
                    read_penalty = (10.0 * (penalty_low + penalty_high)) / grad_accum_steps
                    
                    loss = loss + read_penalty

                loss.backward()
                total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            elapsed = time.time() - start
            if (step + 1) % 1 == 0:
                r_cost = aux.get("read_cost", 0.0)
                if hasattr(r_cost, "item"): r_cost = r_cost.item()
                s_cost = aux.get("skip_cost", 0.0)
                if hasattr(s_cost, "item"): s_cost = s_cost.item()
            
                mean_p_read = r_cost / max(1, aux.get("reason_steps", 10))
                mean_p_skip = s_cost / max(1, aux.get("reason_steps", 10))
            
                print(
                    f"Step {step}/{args.steps} | "
                    f"loss={total_loss:.4f} | "
                    f"p_read={mean_p_read:.3f} | "
                    f"p_skip={mean_p_skip:.3f} | "
                    f"lr={lr:.2e} | "
                    f"{elapsed:.2f}s"
                )
                writer.add_scalar("Train/Loss", total_loss, step)
                writer.add_scalar("Train/p_read", mean_p_read, step)
                writer.add_scalar("Train/p_skip", mean_p_skip, step)
                writer.add_scalar("Train/LR", lr, step)      
            if (step + 1) % 50 == 0 and device_type == "cuda":
                torch.cuda.empty_cache()

            if (step + 1) % args.val_every == 0:
                val_loss, val_p_skip, val_p_read = evaluate(
                    model,
                    val_gen,
                    args.val_steps,
                    params["VOCAB_SIZE"],
                    device_type,
                )
                print(f"\n--- VAL Step {step+1} | loss={val_loss:.4f} | p_read={val_p_read:.3f} | p_skip={val_p_skip:.3f}\n")
                writer.add_scalar("Val/Loss", val_loss, step)
                writer.add_scalar("Val/p_read", val_p_read, step)
                writer.add_scalar("Val/p_skip", val_p_skip, step)

            if (step + 1) % args.save_every == 0:
                state_dict = model.state_dict()
                clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                torch.save(clean_state, save_path)
                print(f"Saved {save_path}")
    except KeyboardInterrupt:
        print("\n[Ctrl+C] Training interrupted! Saving the current model state...")
        writer.close()

    state_dict = model.state_dict()
    clean_state = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    torch.save(clean_state, save_path)
    print(f"Done. Saved {save_path}")

    train_gen.stop()
    val_gen.stop()
    writer.close()


if __name__ == "__main__":
    main()
