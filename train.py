import tensorflow as tf
import os
import numpy as np
from model import build_model, TokenAndPositionEmbedding, TransformerBlock, RMSNorm, WarmupCosineDecay, TiedOutput
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import yaml
import argparse


steps_per_epoch = 5000
TOTAL_EPOCHS = 100
WARMUP_STEPS = 1000


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='default')
args = parser.parse_args()

with open("model_param.yaml", 'r') as f:
    config = yaml.safe_load(f)

model_name = config['default_model'] if args.config == 'default' else args.config
params = config[model_name]
print(f"--- Train model : {model_name} ---")

BATCH_SIZE = params['BATCH_SIZE']
MAX_LEN = params['MAX_LEN']
VOCAB_SIZE = params['VOCAB_SIZE']
EMBED_DIM = params['EMBED_DIM']
NUM_TRANSFORMER_BLOCKS = params['NUM_TRANSFORMER_BLOCKS']
NUM_HEADS = params['NUM_HEADS']
PEAK_LEARNING_RATE = params['PEAK_LEARNING_RATE']
MODEL_SAVE_PATH = params['MODEL_SAVE_PATH']

tf.keras.mixed_precision.set_global_policy('mixed_float16')

def get_hybrid_dataset(bin_path, batch_size, max_len):
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    total_tokens = len(data)
    print(f"Streaming from {bin_path} ({total_tokens:,} tokens)")

    def generator():
        offsets = np.arange(max_len + 1)
        while True:
            starts = np.random.randint(0, total_tokens - (max_len + 1), size=batch_size)
            idx_matrix = starts[:, None] + offsets[None, :]
            batch = data[idx_matrix].astype(np.int32)
            yield batch[:, :-1], batch[:, 1:]

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, max_len), dtype=tf.int32),
            tf.TensorSpec(shape=(batch_size, max_len), dtype=tf.int32)
        )
    )
    return ds.prefetch(2)

print("Preparing datasets...")
train_dataset = get_hybrid_dataset("data/corpus/train.bin", BATCH_SIZE, MAX_LEN)
val_dataset = get_hybrid_dataset("data/corpus/val.bin", BATCH_SIZE, MAX_LEN)

model = build_model(VOCAB_SIZE, MAX_LEN, EMBED_DIM, NUM_TRANSFORMER_BLOCKS, NUM_HEADS)
if os.path.exists(MODEL_SAVE_PATH):
    try:
        model.load_weights(MODEL_SAVE_PATH)
        print(f"Loaded existing weights <- '{MODEL_SAVE_PATH}'")
    except:
        print("Starting from scratch.")

lr_schedule = WarmupCosineDecay(
    peak_learning_rate=PEAK_LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    total_steps=steps_per_epoch * TOTAL_EPOCHS,
    end_learning_rate=1e-5
)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, clipnorm=1.0, epsilon=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    jit_compile=True 
)

checkpoint = ModelCheckpoint(filepath=MODEL_SAVE_PATH, save_weights_only=True, monitor='loss', save_best_only=False, verbose=1)
early_stop = EarlyStopping(monitor='loss', patience=10, verbose=1)

print(f"\n Starting training...")
try:
    model.fit(
        train_dataset,
        epochs=TOTAL_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=10,
        callbacks=[checkpoint, early_stop]
    )
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving weights...")
    model.save_weights(MODEL_SAVE_PATH)
    print(f"Weights saved to '{MODEL_SAVE_PATH}'")

print("\nDone.")
