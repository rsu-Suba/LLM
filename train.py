import tensorflow as tf
import os
import tensorflow_text as tf_text
from model import build_model, TokenAndPositionEmbedding, TransformerBlock, TiedOutputDense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import datetime
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
parser.add_argument('--config', type=str, default='default', help='Name of the model config to use (e.g., model_150M)')
args = parser.parse_args()

with open("model_param.yaml", 'r') as f:
    config = yaml.safe_load(f)

if args.config == 'default':
    model_name = config['default_model']
else:
    model_name = args.config
params = config[model_name]
print(f"--- Using model config: {model_name} ---")


INITIAL_EPOCH = 0
TOTAL_EPOCHS = 10
END_LEARNING_RATE = 0.00001
WARMUP_PER = 0.05
DATASET = 5_000_000
VAL_DATASET = 250_000
DATASET_SHUFFLE = 200_000



MODEL_SAVE_PATH = params['MODEL_SAVE_PATH']
PEAK_LEARNING_RATE = params['PEAK_LEARNING_RATE']
BATCH_SIZE = params['BATCH_SIZE']
VOCAB_SIZE = params['VOCAB_SIZE']
MAX_LEN = params['MAX_LEN']
EMBED_DIM = params['EMBED_DIM']
NUM_TRANSFORMER_BLOCKS = params['NUM_TRANSFORMER_BLOCKS']
NUM_HEADS = params['NUM_HEADS']
TRAIN_CORPUS_PATH = "data/corpus/train.txt"
VAL_CORPUS_PATH = "data/corpus/val.txt"
TOKENIZER_PATH = "data/tokenizer/tokenizer.model"

tf.keras.mixed_precision.set_global_policy('mixed_float16')


with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer_model = f.read()

tokenizer = tf_text.SentencepieceTokenizer(
    model=tokenizer_model,
    add_bos=True,
    add_eos=True
)
print("Loaded <- TensorFlow Text Tokenizer")

def encode_and_shape(text_tensor):
    encoded_ragged = tokenizer.tokenize(text_tensor)
    encoded_tensor = encoded_ragged.to_tensor(
        default_value=0,
        shape=[None, MAX_LEN + 1]
    )
    x = encoded_tensor[:, :-1]
    y = encoded_tensor[:, 1:]
    return x, y

train_dataset = (
    tf.data.TextLineDataset(TRAIN_CORPUS_PATH)
    .take(DATASET)
    .shuffle(DATASET_SHUFFLE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .map(encode_and_shape, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

print("Created -> Train Pipeline")

val_dataset = (
    tf.data.TextLineDataset(VAL_CORPUS_PATH)
    .take(VAL_DATASET)
    .batch(BATCH_SIZE, drop_remainder=True)
    .map(encode_and_shape, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
print("Created -> Val Pipeline")

if os.path.exists(MODEL_SAVE_PATH):
    model = build_model(
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        embed_dim=EMBED_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        num_heads=NUM_HEADS
    )
    model.load_weights(MODEL_SAVE_PATH)
    print(f"Loaded model <-'{MODEL_SAVE_PATH}'")
else:
    print("404 <- keras model")
    model = build_model(
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        embed_dim=EMBED_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        num_heads=NUM_HEADS
    )

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
)

model_checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1
)

TOTAL_STEPS = DATASET // BATCH_SIZE * TOTAL_EPOCHS

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=PEAK_LEARNING_RATE,
    decay_steps=TOTAL_STEPS * (1 - WARMUP_PER),
    alpha=END_LEARNING_RATE / PEAK_LEARNING_RATE,
    warmup_target=PEAK_LEARNING_RATE,
    warmup_steps=int(TOTAL_STEPS * WARMUP_PER)
)

class LearningRateLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        current_lr = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        print(f"\nEpoch {epoch+1}: Learning rate is {current_lr:.6f}.")

    def on_batch_end(self, batch, logs=None):
        logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.learning_rate)

lr_logger_callback = LearningRateLogger()

model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule, clipnorm=1.0, weight_decay=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    jit_compile=True
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    update_freq=50
)

model.summary()
print("h5 model compiled")
print(f"\nEpoch {INITIAL_EPOCH + 1} -> {TOTAL_EPOCHS}\n")

model.fit(
    train_dataset,
    epochs=TOTAL_EPOCHS,
    initial_epoch=INITIAL_EPOCH,
    validation_data=val_dataset,
    callbacks=[early_stopping_callback,
               model_checkpoint_callback,
               lr_logger_callback,
               tensorboard_callback]
)

print("\nModel train finish\n")