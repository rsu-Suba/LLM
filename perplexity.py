import tensorflow as tf
import os
import tensorflow_text as tf_text
from model import build_model, TokenAndPositionEmbedding, TransformerBlock, RMSNorm, WarmupCosineDecay, TiedOutput
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

parser = argparse.ArgumentParser(description="Trained LLM evaluation script.")
parser.add_argument('--model', type=str, default='default', help='Name of the model config to use')
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

VAL_CORPUS_PATH = "data/corpus/val.txt"
TOKENIZER_PATH = "data/tokenizer/tokenizer.model"
VAL_DATASET_SIZE = 10000

tf.keras.mixed_precision.set_global_policy('mixed_float16')

import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
print("Loaded <- Tokenizer")

def encode_and_shape(text_tensor):
    with open(TOKENIZER_PATH, 'rb') as f:
        m = f.read()
    tokenizer_tf = tf_text.SentencepieceTokenizer(model=m, add_bos=True, add_eos=True)
    
    encoded_ragged = tokenizer_tf.tokenize(text_tensor)
    encoded_tensor = encoded_ragged.to_tensor(default_value=0, shape=[None, MAX_LEN + 1])
    x = encoded_tensor[:, :-1]
    y = encoded_tensor[:, 1:]
    return x, y

val_dataset = (
    tf.data.TextLineDataset(VAL_CORPUS_PATH)
    .take(VAL_DATASET_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .map(encode_and_shape, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

model = build_model(VOCAB_SIZE, MAX_LEN, EMBED_DIM, NUM_TRANSFORMER_BLOCKS, NUM_HEADS)
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
print("-" * 25)
