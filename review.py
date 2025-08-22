import tensorflow as tf
import tensorflow_text as tf_text
import sentencepiece as spm
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



NUM_TOKENS_TO_GENERATE = 100
FORBID_EOS_UNTIL_STEP = 10
TEMPERATURE = 0.8
TOP_K = 30
TOP_P = 0.9
PUNCTUATION_PENALTY =  tf.constant(5.0, dtype=tf.float32)
prompt = "今日の天気は"



TOKENIZER_PATH = "data/tokenizer/tokenizer.model"
MODEL_SAVE_PATH = params['MODEL_SAVE_PATH']
PEAK_LEARNING_RATE = params['PEAK_LEARNING_RATE']
BATCH_SIZE = params['BATCH_SIZE']
VOCAB_SIZE = params['VOCAB_SIZE']
MAX_LEN = params['MAX_LEN']
EMBED_DIM = params['EMBED_DIM']
NUM_TRANSFORMER_BLOCKS = params['NUM_TRANSFORMER_BLOCKS']
NUM_HEADS = params['NUM_HEADS']

with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer_model = f.read()

tokenizer = tf_text.SentencepieceTokenizer(
    model=tokenizer_model,
    add_bos=True,
    add_eos=True
)
print("Loaded <- TensorFlow Text Tokenizer")

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
    raise FileNotFoundError(f"404 <- {MODEL_SAVE_PATH}")

def top_k_top_p_logits(logits, k=0, p=1.0):
    logits = tf.cast(logits, tf.float32)

    if k > 0:
        values, _ = tf.math.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        logits = tf.where(logits < min_values, tf.ones_like(logits) * -1e10, logits)

    if p < 1.0:
        sorted_logits = tf.sort(logits, direction='DESCENDING')
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)
        cutoff_index = tf.argmax(cumulative_probs > p, axis=-1)
        cutoff_value = tf.gather(sorted_logits, cutoff_index, batch_dims=1)
        logits = tf.where(logits < cutoff_value[:, tf.newaxis], -1e10, logits)

    return logits

@tf.function
def generate_step(input_tokens, step, eos_id, pad_id, penalty_ids, PUNCTUATION_PENALTY):
    logits = model(input_tokens, training=False)[:, -1, :]
    logits = tf.cast(logits, tf.float32)

    logits = tf.tensor_scatter_nd_update(logits, [[0, pad_id]], [-1e9])
    if tf.less(step, FORBID_EOS_UNTIL_STEP):
        logits = tf.tensor_scatter_nd_update(logits, [[0, eos_id]], [-1e9])

    unique_penalty_ids, _ = tf.unique(penalty_ids)
    updates = tf.gather(logits, unique_penalty_ids, axis=1) - PUNCTUATION_PENALTY
    indices = tf.expand_dims(unique_penalty_ids, axis=1)
    indices = tf.pad(indices, [[0, 0], [1, 0]])

    logits = tf.tensor_scatter_nd_update(logits, indices, updates[0])
    logits = top_k_top_p_logits(logits, k=TOP_K, p=TOP_P)
    logits /= TEMPERATURE
    
    next_token_id = tf.random.categorical(logits, 1)[0]
    return tf.cast(next_token_id, tf.int32)



input_tokens = tokenizer.tokenize([prompt])
input_tokens = input_tokens.to_tensor(default_value=0, shape=[1, MAX_LEN])

print("\n--- Generating ---")
print(f"Input: {prompt}")

generated_tokens_ids = []
eos_id = tokenizer.string_to_id("</s>")
pad_id = tokenizer.string_to_id("<pad>") 

for step in tf.range(NUM_TOKENS_TO_GENERATE):
    penalty_ids = input_tokens[0, -5:]
    next_id = generate_step(input_tokens, tf.cast(step, tf.int32), eos_id, pad_id, penalty_ids, PUNCTUATION_PENALTY)
    if next_id[0] == eos_id:
        break

    generated_tokens_ids.append(next_id[0])
    input_tokens = tf.concat([input_tokens[:, 1:], next_id[:, tf.newaxis]], axis=1)

generated_text = tokenizer.detokenize([generated_tokens_ids]).numpy()[0].decode('utf-8')
print("\n--- Result ---")
print(prompt + generated_text)