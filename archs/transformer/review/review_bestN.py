# Best-of-N sampling for TensorFlow Keras model
import tensorflow as tf
import os
import numpy as np
import sentencepiece as spm
import yaml
import argparse
import sys
import time
import random
from datetime import datetime
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(1, os.path.join(_HERE, '../../..'))
from model import build_model, TokenEmbedding, TransformerBlock, RMSNorm, WarmupCosineDecay, TiedOutput
from generation_utils import top_k_top_p_logits

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

tf.keras.mixed_precision.set_global_policy('mixed_float16')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--prompt', type=str, default=None)
parser.add_argument('--n', type=int, default=3)
args = parser.parse_args()

LOG_DIR = "./chat_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_next_log_path():
    existing = os.listdir(LOG_DIR)
    indices = [int(f.split('.')[0]) for f in existing if f.split('.')[0].isdigit()]
    next_idx = max(indices) + 1 if indices else 1
    return os.path.join(LOG_DIR, f"{next_idx:05d}.log")

import os
with open(os.path.join(os.path.dirname(__file__), "../param.yaml"), 'r') as f:
    config_file = yaml.safe_load(f)

model_name = config_file['default_model'] if args.model == 'default' else args.model
params = config_file[model_name]
gen_params = params.get('generation', {})

MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", params['MODEL_SAVE_PATH'])
MAX_LEN = params['MAX_LEN']
TOKENIZER_PATH = "data/tokenizer/tokenizer.model"
VOCAB_SIZE = params['VOCAB_SIZE']
PROMPT = args.prompt if args.prompt else gen_params.get('prompt', "こんにちは")
TARGET_TOKENS = gen_params.get('max_new_tokens', 100)
ABS_MAX_TOKENS = 1000
N_CANDIDATES = args.n
TEMPERATURE = gen_params.get('temperature', 0.8)
TOP_K = gen_params.get('top_k', 40)
TOP_P = gen_params.get('top_p', 0.9)
REPETITION_PENALTY = gen_params.get('repetition_penalty', 1.2)
CHUNK_SIZE = 35

CONTINUITY_PHRASES = ["さらに、", "また、", "具体的には、", "そのため、", "その上で、", "これによって、"]

sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
num_kv = params.get('NUM_KV_HEADS', None)
model = build_model(VOCAB_SIZE, MAX_LEN, params['EMBED_DIM'], params['NUM_TRANSFORMER_BLOCKS'], params['NUM_HEADS'], num_kv_heads=num_kv)
model.load_weights(MODEL_SAVE_PATH)
period_id = sp.piece_to_id("。")
eos_id = sp.eos_id()
pad_id = sp.pad_id()

@tf.function
def predict_next(input_tokens):
    return model(input_tokens, training=False)[:, -1, :]

def get_chunk_candidate(base_ids):
    current_ids = list(base_ids)
    log_probs = []
    initial_text_len = len(sp.decode(base_ids))

    for i in range(CHUNK_SIZE):
        curr_input = current_ids[-MAX_LEN:]
        padded_input = tf.keras.preprocessing.sequence.pad_sequences([curr_input], maxlen=MAX_LEN, padding='post', value=pad_id)

        logits = model(padded_input, training=False)[0, len(curr_input)-1, :]
        logits = tf.cast(logits, tf.float32)
        probs = tf.nn.softmax(logits)

        logits_np = logits.numpy()
        for gid in set(current_ids):
            if logits_np[gid] > 0: logits_np[gid] /= REPETITION_PENALTY
            else: logits_np[gid] *= REPETITION_PENALTY

        logits_sampled = tf.convert_to_tensor([logits_np])
        logits_sampled = logits_sampled / TEMPERATURE
        logits_sampled = top_k_top_p_logits(logits_sampled, k=TOP_K, p=TOP_P)
        next_id = int(tf.random.categorical(logits_sampled, 1)[0, 0].numpy())

        token_log_prob = tf.math.log(probs[next_id] + 1e-10).numpy()
        log_probs.append(token_log_prob)
        current_ids.append(next_id)

        if next_id == eos_id:
            break

        if i > 0:
            full_text = sp.decode(current_ids)
            new_text = full_text[initial_text_len:]
            if any(mark in new_text for mark in ["。", "！", "？", "!", "?", "\n"]):
                break

    new_part = current_ids[len(base_ids):]
    score = np.mean(log_probs) if log_probs else -100
    return new_part, score, sp.decode(new_part)

print(f"\n--- Generating ---")
print(f"Prompt: {PROMPT}")
sys.stdout.write("Result: " + PROMPT)
sys.stdout.flush()

history_ids = sp.encode(PROMPT)
last_printed_len = len(sp.decode(history_ids))
initial_len = len(history_ids)
step_logs = []

while True:
    candidates = []
    for _ in range(N_CANDIDATES):
        cand_ids, score, text = get_chunk_candidate(history_ids)
        candidates.append({"ids": cand_ids, "score": score, "text": text})

    best_cand = max(candidates, key=lambda x: x["score"])
    history_ids.extend(best_cand["ids"])

    scores = np.array([c["score"] for c in candidates])
    exp_scores = np.exp(scores - np.max(scores))
    probs = exp_scores / np.sum(exp_scores)
    for i, c in enumerate(candidates): c["prob"] = probs[i]
    step_logs.append({"candidates": candidates, "selected": best_cand["text"]})

    full_text = sp.decode(history_ids)
    new_text = full_text[last_printed_len:]
    sys.stdout.write(new_text)
    sys.stdout.flush()
    last_printed_len = len(full_text)

    if len(history_ids) - initial_len >= TARGET_TOKENS or history_ids[-1] == eos_id:
        break


log_path = get_next_log_path()
with open(log_path, "w", encoding="utf-8") as f:
    f.write(f"--- Chat Log: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    f.write(f"Full Result:\n{sp.decode(history_ids)}\n")
    f.write("\n--- DECISION PROCESS ---\n")
    for i, step in enumerate(step_logs, 1):
        f.write(f"\n[Step {i}] {step.get('selected','')}\n")
        if "candidates" in step and step["candidates"]:
            sorted_cands = sorted(step["candidates"], key=lambda x: x["score"], reverse=True)
            for j, cand in enumerate(sorted_cands, 1):
                f.write(f"  Candidate {j} {'★' if cand['text'] == step['selected'] else '  '} (Prob: {cand['prob']:.1%}):\n    \"{cand['text'].strip()}\"\n")

print(f"\n\n--- Finished ---")
print(f"Log: {log_path}")
