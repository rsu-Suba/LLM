import tensorflow as tf
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from model import build_model, WarmupCosineDecay
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import yaml
import argparse

steps_per_epoch = 4000
TOTAL_EPOCHS = 10
WARMUP_STEPS = 1000

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='default')
parser.add_argument('--base_weights', type=str, default=None)
parser.add_argument('--output_weights', type=str, default=None)
args = parser.parse_args()

import os
with open(os.path.join(os.path.dirname(__file__), "../param.yaml"), 'r') as f:
    config = yaml.safe_load(f)

model_name = config['default_model'] if args.model == 'default' else args.model
params = config[model_name]
print(f"--- Instruct Tuning : {model_name} (Loss Masked) ---")

BATCH_SIZE = params['BATCH_SIZE']
MAX_LEN = params['MAX_LEN']
VOCAB_SIZE = params['VOCAB_SIZE']
EMBED_DIM = params['EMBED_DIM']
NUM_TRANSFORMER_BLOCKS = params['NUM_TRANSFORMER_BLOCKS']
NUM_HEADS = params['NUM_HEADS']
PEAK_LEARNING_RATE = params['PEAK_LEARNING_RATE'] / 5.0
GRAD_ACCUM_STEPS = params['GRAD_ACCUM_STEPS']

_base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", params['MODEL_SAVE_PATH'])
BASE_WEIGHTS = args.base_weights if args.base_weights else _base_path
INSTRUCT_WEIGHTS = args.output_weights if args.output_weights else BASE_WEIGHTS.replace('.weights.h5', '_instruct.weights.h5')
os.makedirs(os.path.dirname(BASE_WEIGHTS), exist_ok=True)

tf.keras.mixed_precision.set_global_policy('mixed_float16')

def get_masked_dataset(npz_path, batch_size):
    print(f"Loading {npz_path} ...")
    data = np.load(npz_path)
    x = data['x']
    y = data['y']
    w = data['w']
    total_sequences = len(x)
    print(f"Found {total_sequences:,} sequences.")

    ds = tf.data.Dataset.from_tensor_slices((x, y, w))
    ds = ds.shuffle(buffer_size=min(total_sequences, 10000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)

print("Preparing Masked Instruct datasets...")
train_dataset = get_masked_dataset("data/corpus/token/instruct_train_masked.npz", BATCH_SIZE)
val_dataset   = get_masked_dataset("data/corpus/token/instruct_val_masked.npz",   BATCH_SIZE)

model = build_model(VOCAB_SIZE, MAX_LEN, EMBED_DIM, NUM_TRANSFORMER_BLOCKS, NUM_HEADS, num_kv_heads=params.get('NUM_KV_HEADS'))

if os.path.exists(INSTRUCT_WEIGHTS):
    model.load_weights(INSTRUCT_WEIGHTS)
    print(f"Loaded existing Instruct weights <- '{INSTRUCT_WEIGHTS}'")
elif os.path.exists(BASE_WEIGHTS):
    model.load_weights(BASE_WEIGHTS)
    print(f"Loaded Base weights for fine-tuning <- '{BASE_WEIGHTS}'")
else:
    print("Warning: No base weights found! Starting from scratch.")

INSTRUCT_LEARNING_RATE = 2.0e-5

lr_schedule = WarmupCosineDecay(
    peak_learning_rate=INSTRUCT_LEARNING_RATE,
    warmup_steps=WARMUP_STEPS // GRAD_ACCUM_STEPS,
    total_steps=(steps_per_epoch // GRAD_ACCUM_STEPS) * TOTAL_EPOCHS,
    end_learning_rate=INSTRUCT_LEARNING_RATE / 10.0
)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        clipnorm=1.0,
        epsilon=1e-4,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    jit_compile=True
)

checkpoint = ModelCheckpoint(filepath=INSTRUCT_WEIGHTS, save_weights_only=True, monitor='val_loss', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

print(f"\n Starting Masked Instruction Tuning...")
print(f" Saving best model to: {INSTRUCT_WEIGHTS}")
try:
    model.fit(
        train_dataset,
        epochs=TOTAL_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=50,
        callbacks=[checkpoint, early_stop]
    )
except KeyboardInterrupt:
    print("\nTraining interrupted by user. Saving weights...")
    model.save_weights(INSTRUCT_WEIGHTS)
    print(f"Weights saved to '{INSTRUCT_WEIGHTS}'")

print("\nDone.")
