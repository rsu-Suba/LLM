import tensorflow as tf
import os
import tensorflow_text as tf_text
from model import build_model, TokenEmbedding, TransformerBlock, RMSNorm, WarmupCosineDecay, TiedOutput
import yaml
import argparse

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='default')
args = parser.parse_args()

with open("model_param.yaml", 'r') as f:
    config = yaml.safe_load(f)

model_name = config['default_model'] if args.model == 'default' else args.model
params = config[model_name]
print(f"--- Evaluating Model: {model_name} ---")

MODEL_SAVE_PATH = params['MODEL_SAVE_PATH']
BATCH_SIZE = params['BATCH_SIZE']
VOCAB_SIZE = params['VOCAB_SIZE']
MAX_LEN = params['MAX_LEN']
EMBED_DIM = params['EMBED_DIM']
NUM_TRANSFORMER_BLOCKS = params['NUM_TRANSFORMER_BLOCKS']
NUM_HEADS = params['NUM_HEADS']

VAL_BIN_PATH = "data/corpus/token/val.bin"
VAL_SAMPLES = 3000

tf.keras.mixed_precision.set_global_policy('mixed_float16')

import numpy as np

def load_val(batch_size, max_len):
    raw = np.fromfile(VAL_BIN_PATH, dtype=np.uint16).astype(np.int32)
    total = len(raw) // (max_len + 1)
    data = raw[:total * (max_len + 1)].reshape((total, max_len + 1))
    
    np.random.seed(42)
    idx = np.random.choice(total, min(VAL_SAMPLES, total), replace=False)
    sampled = data[idx]
    
    ds = tf.data.Dataset.from_tensor_slices(sampled)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(lambda b: (b[:, :-1], b[:, 1:]), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

val_dataset = load_val(BATCH_SIZE, MAX_LEN)

model = build_model(VOCAB_SIZE, MAX_LEN, EMBED_DIM, NUM_TRANSFORMER_BLOCKS, NUM_HEADS, num_kv_heads=params.get('NUM_KV_HEADS'))
model.load_weights(MODEL_SAVE_PATH)
print(f"Loaded weights <-'{MODEL_SAVE_PATH}'")

print("\n--- Starting Perplexity Evaluation ---")
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
results = model.evaluate(val_dataset)
validation_loss = results
perplexity = tf.exp(validation_loss)

print("\n--- Evaluation Results ---")
print(f"Validation Loss: {validation_loss:.4f}")
print(f"Perplexity:      {perplexity:.4f}")
