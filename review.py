import tensorflow as tf
import tensorflow_text as tf_text
from model import build_model, TokenAndPositionEmbedding, TransformerBlock, TiedOutputDense
import yaml
import argparse
import os

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

tf.keras.mixed_precision.set_global_policy('mixed_float16')
TOKENIZER_PATH = "data/tokenizer/tokenizer.model"
MODEL_SAVE_PATH = params['MODEL_SAVE_PATH']
MAX_LEN = params['MAX_LEN']

NUM_TOKENS_TO_GENERATE = 100
TEMPERATURE = 1.4
TOP_K = 50
TOP_P = 0.95
REPETITION_PENALTY = 2.0
FORBID_EOS_UNTIL_STEP = 10
prompt = "人工知能とは、"

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer_model = f.read()
tokenizer = tf_text.SentencepieceTokenizer(
    model=tokenizer_model,
    add_bos=True,
    add_eos=False
)
print("Loaded <- TensorFlow Text Tokenizer")

model = build_model(
    vocab_size=params['VOCAB_SIZE'],
    max_len=params['MAX_LEN'],
    embed_dim=params['EMBED_DIM'],
    num_transformer_blocks=params['NUM_TRANSFORMER_BLOCKS'],
    num_heads=params['NUM_HEADS']
)
model.load_weights(MODEL_SAVE_PATH)
print(f"Loaded model weights <-'{MODEL_SAVE_PATH}'")

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

@tf.function
def predict_next(input_tokens):
    return model(input_tokens, training=False)[:, -1, :]

print("\n--- Generating ---")
print(f"Input: {prompt}")

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
        unique_ids = tf.constant(list(set(generated_tokens_ids)), dtype=tf.int32)
        
        penalty_values = tf.gather(logits, unique_ids, axis=1)
        penalty_values = tf.where(penalty_values > 0, penalty_values / REPETITION_PENALTY, penalty_values * REPETITION_PENALTY)
        
        indices = tf.expand_dims(unique_ids, axis=1)
        batch_indices = tf.zeros_like(indices)
        indices = tf.concat([batch_indices, indices], axis=1)
        
        updates = tf.squeeze(penalty_values, axis=0)
        
        logits = tf.tensor_scatter_nd_update(logits, indices, updates)

    logits = tf.tensor_scatter_nd_update(logits, [[0, pad_id]], [-1e9])
    if step < FORBID_EOS_UNTIL_STEP:
        logits = tf.tensor_scatter_nd_update(logits, [[0, eos_id]], [-1e9])

    logits /= TEMPERATURE
    logits = top_k_top_p_logits(logits, k=TOP_K, p=TOP_P)
    
    next_token_id = tf.random.categorical(logits, 1)[0, 0]
    next_token_id = int(next_token_id.numpy())

    if next_token_id == eos_id:
        break
        
    generated_tokens_ids.append(next_token_id)

output_ids = generated_tokens_ids[len(initial_tokens_list):]
generated_text = tokenizer.detokenize([output_ids]).numpy()[0].decode('utf-8')

print("\n--- Result ---")
print(prompt + generated_text)
