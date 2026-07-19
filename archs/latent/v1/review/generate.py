import torch
import torch.nn.functional as F
import yaml
import os
import sys
import sentencepiece as spm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from model import LatentBrainLLM, LatentBrainConfig, build_latent_brain_from_params

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def print_reasoning_log(history, step=0):
    print(f"\n\033[1m[🧠 Reasoning Phase (Step {step})]\033[0m")
    for layer_idx, aux in enumerate(history):
        p_read = aux["read_probability"].mean().item()
        p_skip = aux["skip_probability"].mean().item()
        c_read = f"\033[96m{p_read:.3f}\033[0m"
        c_skip = f"\033[93m{p_skip:.3f}\033[0m"
        print(f"  Layer {layer_idx + 1:2d} | Read: {c_read} | Skip: {c_skip}")
    print(f"\033[92m  => Finished thinking!\033[0m\n")


def generate(model, prompt_ids, max_new_tokens=64, temperature=0.6, top_k=35, top_p=0.7, repetition_penalty=1.2, chunk_size=8):
    model.eval()
    device = prompt_ids.device
    
    with torch.no_grad():
        latent, prompt_memory = model.encode(prompt_ids)
        latent, info = model.reasoner(latent, prompt_memory)
        print_reasoning_log(info["history"], step=0)
        
        target_ids = torch.tensor([[1]], device=device)
        chunk_token_count = 0
        
        for step in range(max_new_tokens):
            context_target_ids = target_ids[:, -64:]
            logits = model.decode(latent, context_target_ids)
            next_token_logits = logits[:, -1, :]
            for token_id in set(target_ids[0].tolist()):
                if next_token_logits[0, token_id] < 0:
                    next_token_logits[0, token_id] *= repetition_penalty
                else:
                    next_token_logits[0, token_id] /= repetition_penalty
            
            next_token_logits = next_token_logits / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            target_ids = torch.cat([target_ids, next_token], dim=1)
            chunk_token_count += 1
            
            if next_token.item() == 2:
                break
                
            if chunk_size > 0 and chunk_token_count == chunk_size:
                chunk_target_ids = target_ids[:, -chunk_size:]
                chunk_emb = model.embed(chunk_target_ids)
                for block in model.encoder:
                    chunk_emb = block(chunk_emb)
                    
                prompt_memory = torch.cat([prompt_memory, chunk_emb], dim=1)
                latent, info = model.reasoner(latent, prompt_memory)
                print_reasoning_log(info["history"], step=step+1)
                
                chunk_token_count = 0
                
        return target_ids

if __name__ == "__main__":
    import os
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="iron")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "../param.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if args.model not in config:
        raise ValueError(f"Model {args.model} not found in config")
        
    params = config[args.model]
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = LatentBrainConfig(
        vocab_size=params.get("VOCAB_SIZE", 28000),
        dim=params.get("EMBED_DIM", 768),
        ff_dim=params.get("FF_DIM", 2048),
        num_heads=params.get("NUM_HEADS", 12),
        num_kv_heads=params.get("NUM_KV_HEADS", 4),
        encode_layers=params.get("ENCODE_LAYERS", 4),
        reason_layers=params.get("REASON_LAYERS", 10),
        decode_layers=params.get("DECODE_LAYERS", 6),
        compression_ratio=params.get("COMPRESSION_RATIO", 32),
        dropout=params.get("DROPOUT", 0.0),
        act_threshold=params.get("ACT_THRESHOLD", 0.95),
        max_reason_steps=params.get("MAX_REASON_STEPS", 10)
    )
    model = LatentBrainLLM(cfg).to(device)
    
    model_save_path = os.path.join(os.path.dirname(config_path), params["MODEL_SAVE_PATH"])
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    
    sp = spm.SentencePieceProcessor(model_file='data/tokenizer/tokenizer.model')
    
    input_text = "おはようございます。今日の天気は"
    input_ids = sp.encode(input_text, out_type=int)
    
    prompt_ids = torch.tensor([input_ids], device=device)
    
    print(f"Generating continuation for: '{input_text}'")
    out_ids = generate(model, prompt_ids, max_new_tokens=64)
    
    out_ids_np = out_ids[0].tolist()
    if out_ids_np[0] == 1:
        out_ids_np = out_ids_np[1:]
        
    generated_text = sp.decode(out_ids_np)
    print("\n\033[92m[Generated Text]\033[0m")
    print(input_text + generated_text)
    print("\nGenerated sequence length:", len(out_ids_np))
