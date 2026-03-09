import tensorflow as tf
import tensorflow_text as tf_text
from model import build_model, TokenAndPositionEmbedding, TransformerBlock, RMSNorm, WarmupCosineDecay, TiedOutput
import yaml
import argparse
import os
import numpy as np
import sentencepiece as spm
import sys
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--prompt', type=str, default=None)
args = parser.parse_args()

with open("model_param.yaml", 'r') as f:
    config_file = yaml.safe_load(f)

model_name = config_file['default_model'] if args.config == 'default' else args.config
params = config_file[model_name]
gen_params = params.get('generation', {})

MODEL_SAVE_PATH = params['MODEL_SAVE_PATH']
MAX_LEN = params['MAX_LEN']
TOKENIZER_PATH = "data/tokenizer/tokenizer.model"

PROMPT = args.prompt if args.prompt else gen_params.get('prompt', "こんにちは")
TARGET_TOKENS = gen_params.get('max_new_tokens', 100)
ABS_MAX_TOKENS = 1000
TEMPERATURE = gen_params.get('temperature', 0.8)
TOP_K = gen_params.get('top_k', 40)
TOP_P = gen_params.get('top_p', 0.9)
REPETITION_PENALTY = gen_params.get('repetition_penalty', 1.2)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
model = build_model(params['VOCAB_SIZE'], params['MAX_LEN'], params['EMBED_DIM'], params['NUM_TRANSFORMER_BLOCKS'], params['NUM_HEADS'])
model.load_weights(MODEL_SAVE_PATH)

from generation_utils import top_k_top_p_logits

def sample(logits):
    logits = logits / TEMPERATURE
    logits = top_k_top_p_logits(logits, k=TOP_K, p=TOP_P)
    return tf.random.categorical(logits, 1)[0, 0].numpy()

print("\n---Prompting---")
print(f"Input: {PROMPT}")

print("\n--- Generating ---")
sys.stdout.write(f"Result: {PROMPT}")
sys.stdout.flush()

generated_ids = sp.encode(PROMPT)
initial_len = len(generated_ids)
eos_id = sp.eos_id()
pad_id = sp.pad_id()

last_printed_len = len(PROMPT)
start_time = time.time()

for step in range(ABS_MAX_TOKENS):
    curr_input = generated_ids[-MAX_LEN:]
    padded_input = tf.keras.preprocessing.sequence.pad_sequences([curr_input], maxlen=MAX_LEN, padding='post', value=pad_id)
    
    logits = model(padded_input, training=False)[0, len(curr_input)-1, :]
    logits = tf.cast(logits, tf.float32)
    
    logits_np = logits.numpy()
    for gid in set(generated_ids):
        if logits_np[gid] > 0:
            logits_np[gid] /= REPETITION_PENALTY
        else:
            logits_np[gid] *= REPETITION_PENALTY
    
    next_id = int(sample(tf.convert_to_tensor([logits_np])))
    
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
