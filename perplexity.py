import tensorflow as tf
import os
import tensorflow_text as tf_text
from model import build_model, TokenAndPositionEmbedding, TransformerBlock, TiedOutputDense
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

parser = argparse.ArgumentParser(description="Trained LLM evaluation script (calculates perplexity and generates sample text).")
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

MODEL_SAVE_PATH = params['MODEL_SAVE_PATH']
BATCH_SIZE = params['BATCH_SIZE']
VOCAB_SIZE = params['VOCAB_SIZE']
MAX_LEN = params['MAX_LEN']
EMBED_DIM = params['EMBED_DIM']
NUM_TRANSFORMER_BLOCKS = params['NUM_TRANSFORMER_BLOCKS']
NUM_HEADS = params['NUM_HEADS']

VAL_CORPUS_PATH = "data/corpus/val.txt"
TOKENIZER_PATH = "data/tokenizer/tokenizer.model"
VAL_DATASET_SIZE = 150_000

NUM_TOKENS_TO_GENERATE = 100
TEMPERATURE = 1.3
TOP_K = 70
TOP_P = 0.98
REPETITION_PENALTY = 1.7
MAX_REPEAT = 20
NGRAM_BLOCK = 3
prompt = "今日の天気は"

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

val_dataset = (
    tf.data.TextLineDataset(VAL_CORPUS_PATH)
    .take(VAL_DATASET_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .map(encode_and_shape, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)
print("Created -> Val Pipeline")

model = build_model(
    vocab_size=VOCAB_SIZE,
    max_len=MAX_LEN,
    embed_dim=EMBED_DIM,
    num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
    num_heads=NUM_HEADS
)
model.load_weights(MODEL_SAVE_PATH)
print(f"Loaded model weights <-'{MODEL_SAVE_PATH}'")

print("\n--- Starting Perplexity Evaluation ---")
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)
results = model.evaluate(val_dataset)
validation_loss = results
perplexity = tf.exp(validation_loss)
print("--- Perplexity Evaluation Finished ---")

@tf.function
def predict_next(input_tokens):
    return model(input_tokens, training=False)[:, -1, :]

def top_k_top_p_logits(logits, k=0, p=1.0):
    if k > 0:
        values, _ = tf.math.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        logits = tf.where(logits < min_values, -1e9, logits)
    if p < 1.0:
        sorted_logits = tf.sort(logits, direction='DESCENDING')
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
        cutoff_index = tf.argmax(cumulative_probs > p, axis=-1)
        cutoff_value = tf.gather(sorted_logits, cutoff_index, batch_dims=1)
        logits = tf.where(logits < cutoff_value[:, tf.newaxis], -1e9, logits)
    return logits

def apply_ngram_block(generated_ids, logits, n=3):
    if len(generated_ids) < n:
        return logits
    recent_ngram = tuple(generated_ids[-n:])
    for i in range(len(generated_ids) - n):
        if tuple(generated_ids[i:i+n]) == recent_ngram:
            logits = tf.tensor_scatter_nd_update(
                logits, [[0, recent_ngram[-1]]], [-1e9]
            )
            break
    return logits

print("\n--- Starting Text Generation ---")
initial_tokens_list = tokenizer.tokenize([prompt]).to_list()[0]
generated_tokens_ids = initial_tokens_list[:]
eos_id = int(tokenizer.string_to_id("</s>").numpy())
pad_id = int(tokenizer.string_to_id("<pad>").numpy())

for step in range(NUM_TOKENS_TO_GENERATE):
    current_sequence = generated_tokens_ids[-MAX_LEN:]
    padded_input = tf.keras.preprocessing.sequence.pad_sequences(
        [current_sequence], maxlen=MAX_LEN, padding='post', value=pad_id
    )
    input_tensor = tf.convert_to_tensor(padded_input, dtype=tf.int32)

    logits = predict_next(input_tensor)
    logits = tf.cast(logits, tf.float32)

    if generated_tokens_ids:
        last_token = generated_tokens_ids[-1]
        penalty_value = logits[0, last_token]
        if penalty_value > 0:
            penalty_value /= REPETITION_PENALTY
        else:
            penalty_value *= REPETITION_PENALTY
        logits = tf.tensor_scatter_nd_update(
            logits, [[0, last_token]], [penalty_value]
        )

    logits = apply_ngram_block(generated_tokens_ids, logits, n=NGRAM_BLOCK)
    logits = tf.tensor_scatter_nd_update(logits, [[0, pad_id]], [-1e9])
    logits /= TEMPERATURE
    logits = top_k_top_p_logits(logits, k=TOP_K, p=TOP_P)

    next_token_id = tf.random.categorical(logits, 1)[0, 0].numpy()

    if next_token_id == eos_id:
        break
    if len(generated_tokens_ids) > MAX_REPEAT and all(
        t == generated_tokens_ids[-1] for t in generated_tokens_ids[-MAX_REPEAT:]
    ):
        print("Early stop: same token repeated too much.")
        break

    generated_tokens_ids.append(next_token_id)

output_ids = generated_tokens_ids[len(initial_tokens_list):]
generated_text = tokenizer.detokenize([output_ids]).numpy()[0].decode('utf-8')

print("\n--- Evaluation Metrics ---")
print(f"Validation Loss: {validation_loss:.4f}")
print(f"Perplexity:      {perplexity:.4f}")
print("--------------------------")
print("(Lower Perplexity is better.)")

print("\n--- Generated Text ---")
print(prompt + generated_text)
print("--------------------------")

