import tensorflow as tf
from model import build_model
from generation_utils import top_k_top_p_logits
import yaml
import argparse
import os
import sentencepiece as spm
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

tf.keras.mixed_precision.set_global_policy('mixed_float16')

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='default')
args = parser.parse_args()

with open("model_param.yaml", 'r') as f:
    config_file = yaml.safe_load(f)

model_name = config_file['default_model'] if args.model == 'default' else args.model
params = config_file[model_name]
gen_params = params.get('generation', {})

BASE_WEIGHTS = params['MODEL_SAVE_PATH']
INSTRUCT_WEIGHTS = BASE_WEIGHTS.replace('.weights.h5', '_instruct.weights.h5')

MODEL_SAVE_PATH = INSTRUCT_WEIGHTS if os.path.exists(INSTRUCT_WEIGHTS) else BASE_WEIGHTS

MAX_LEN = params['MAX_LEN']
TOKENIZER_PATH = "data/tokenizer/tokenizer.model"
TEMPERATURE = gen_params.get('temperature', 0.7)
TOP_K = gen_params.get('top_k', 40)
TOP_P = gen_params.get('top_p', 0.8)
REPETITION_PENALTY = gen_params.get('repetition_penalty', 1.1)

sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
model = build_model(params['VOCAB_SIZE'], params['MAX_LEN'], params['EMBED_DIM'], params['NUM_TRANSFORMER_BLOCKS'], params['NUM_HEADS'], num_kv_heads=params.get('NUM_KV_HEADS'))

print(f"Loading weights from {MODEL_SAVE_PATH}...")
model.load_weights(MODEL_SAVE_PATH)
print("Model ready!\n")

def sample(logits):
    if TEMPERATURE == 0.0:
        return tf.argmax(logits, axis=-1)[0].numpy()
    logits = logits / TEMPERATURE
    logits = top_k_top_p_logits(logits, k=TOP_K, p=TOP_P)
    return tf.random.categorical(logits, 1)[0, 0].numpy()

def chat():
    print(f"\nChat: 'exit' or 'quit' to stop")
    print(f"  モード: {'対話モデル (Instruct)' if '_instruct' in MODEL_SAVE_PATH else 'ベースモデル (Base)'}")
    print(f"  Parameters:       Temp={TEMPERATURE}, Top-K={TOP_K}, Top-P={TOP_P}, Penalty={REPETITION_PENALTY}")
    
    history = ""
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()

    while True:
        try:
            user_input = input("ユーザー: ")
        except (EOFError, KeyboardInterrupt):
            break
            
        if user_input.lower() in ['exit', 'quit']:
            break
        if not user_input.strip():
            continue
            
        prompt = f"ユーザー: {user_input}\nAI: "
        history += prompt
        
        sys.stdout.write("AI: ")
        sys.stdout.flush()
        
        generated_ids = sp.encode(history)
        last_printed_len = len(history)
        
        for step in range(MAX_LEN):
            curr_input = generated_ids[-MAX_LEN:]
            padded_input = tf.keras.preprocessing.sequence.pad_sequences([curr_input], maxlen=MAX_LEN, padding='post', value=pad_id)
            
            logits = model(padded_input, training=False)[0, len(curr_input)-1, :]
            logits = tf.cast(logits, tf.float32).numpy()
            
            for gid in set(generated_ids):
                if logits[gid] > 0:
                    logits[gid] /= REPETITION_PENALTY
                else:
                    logits[gid] *= REPETITION_PENALTY
                    
            next_id = int(sample(tf.convert_to_tensor([logits])))
            
            if next_id == eos_id:
                history += " <EOS>\n"
                break
                
            generated_ids.append(next_id)
            
            full_text = sp.decode(generated_ids)
            new_text = full_text[last_printed_len:]
            sys.stdout.write(new_text)
            sys.stdout.flush()
            
            last_printed_len = len(full_text)
            history = full_text
            
        print("\n")

if __name__ == '__main__':
    chat()
