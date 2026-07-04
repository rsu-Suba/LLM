import torch
import torch.nn.functional as F
import yaml
import argparse
import os
import sentencepiece as spm
import sys
import time

from train_1B import Diamond1B

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
    args = parser.parse_args()

    with open("model_param.yaml", 'r') as f:
        config_file = yaml.safe_load(f)

    model_name = config_file['default_model'] if args.model == 'default' else args.model
    params = config_file[model_name]
    gen_params = params.get('generation', {})

    MODEL_SAVE_PATH = "model/diamond_1B.pt"
    MAX_LEN = params['MAX_LEN']
    TOKENIZER_PATH = "data/tokenizer/tokenizer.model"

    PROMPT = args.prompt if args.prompt else gen_params.get('prompt', "こんにちは")
    TARGET_TOKENS = gen_params.get('max_new_tokens', 100)
    ABS_MAX_TOKENS = 1000
    TEMPERATURE = gen_params.get('temperature', 0.8)
    TOP_K = gen_params.get('top_k', 40)
    TOP_P = gen_params.get('top_p', 0.9)
    REPETITION_PENALTY = gen_params.get('repetition_penalty', 1.2)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    
    print(f"Loading 1B Model to {device}...")
    model = Diamond1B(
        vocab_size=params['VOCAB_SIZE'],
        dim=params['EMBED_DIM'],
        heads=params['NUM_HEADS'],
        kv_heads=params.get('NUM_KV_HEADS', 4),
        ff_dim=params.get('FF_DIM', 5504),
        layers=params['NUM_TRANSFORMER_BLOCKS']
    )
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu", weights_only=False))
    model.to(device)
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
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type != 'mps' else torch.float16):
                logits = model(input_tensor, use_checkpointing=False)
            
            next_token_logits = logits[0, -1, :].float().unsqueeze(0)
            
            for gid in set(generated_ids):
                if next_token_logits[0, gid] > 0:
                    next_token_logits[0, gid] /= REPETITION_PENALTY
                else:
                    next_token_logits[0, gid] *= REPETITION_PENALTY
            
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