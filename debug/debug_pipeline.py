import tensorflow as tf
import tensorflow_text as tf_text
from model import build_model, TokenAndPositionEmbedding, TransformerBlock, RMSNorm, WarmupCosineDecay, TiedOutput
import os

MAX_LEN = 128
BATCH_SIZE = 64
TRAIN_CORPUS_PATH = "data/corpus/train.txt"
TOKENIZER_PATH = "data/tokenizer/tokenizer.model"

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer_model = f.read()

tokenizer = tf_text.SentencepieceTokenizer(
    model=tokenizer_model,
    add_bos=True,
    add_eos=True
)

def split_x_y(tokens):
    x = tokens[:, :-1]
    y = tokens[:, 1:]
    sample_weight = tf.cast(y != 0, dtype=tf.float32)
    return x, y, sample_weight

dataset = (
    tf.data.TextLineDataset(TRAIN_CORPUS_PATH)
    .take(100)
    .map(lambda x: tokenizer.tokenize(x))
    .unbatch() 
    .batch(MAX_LEN + 1, drop_remainder=True)
    .batch(BATCH_SIZE, drop_remainder=True)
    .map(split_x_y)
)

print("--- Data Pipeline Inspection ---")
for x, y, sw in dataset.take(1):
    print(f"X shape: {x.shape}")
    print(f"Y shape: {y.shape}")
    
    first_x = x[0].numpy()
    first_y = y[0].numpy()
    first_sw = sw[0].numpy()
    
    print("\nFirst sample X (IDs):")
    print(first_x)
    print("\nFirst sample Y (IDs):")
    print(first_y)
    
    print(f"\nNon-zero tokens in this sample: {tf.reduce_sum(first_sw).numpy()}")
    
    decoded_text = tokenizer.detokenize(first_x).numpy().decode('utf-8')
    print("\nDecoded X:")
    print(decoded_text)

print("\n--- End of Inspection ---")
