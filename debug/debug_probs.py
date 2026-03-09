import tensorflow as tf
import sentencepiece as spm
from model import build_model
import yaml
import os
import numpy as np

with open("model_param.yaml", 'r') as f:
    config = yaml.safe_load(f)
p = config["coal"]

sp = spm.SentencePieceProcessor(model_file="data/tokenizer/tokenizer.model")

model = build_model(p['VOCAB_SIZE'], p['MAX_LEN'], p['EMBED_DIM'], p['NUM_TRANSFORMER_BLOCKS'], p['NUM_HEADS'])
model.load_weights(p['MODEL_SAVE_PATH'])

prompt = "「こんにちは」"
ids = sp.encode(prompt)
padded_ids = tf.keras.preprocessing.sequence.pad_sequences([ids], maxlen=p['MAX_LEN'], padding='post')

logits = model(padded_ids, training=False)[0, len(ids)-1, :]
probs = tf.nn.softmax(logits).numpy()

top_indices = np.argsort(probs)[-10:][::-1]

print(f"--- Top 10 Predictions for next token after '{prompt}' ---")
for i in top_indices:
    piece = sp.id_to_piece(int(i))
    prob = probs[i]
    print(f"ID {i:5}: {piece:<15} (Prob: {prob:.4f})")
