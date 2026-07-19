import tensorflow as tf
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(1, os.path.join(_HERE, '../../..'))
from model import build_model
from generation_utils import top_k_top_p_logits
import sentencepiece as spm
import yaml
import sys
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except: pass

import os
with open(os.path.join(os.path.dirname(__file__), "../param.yaml"), "r") as f:
    config = yaml.safe_load(f)
params = config["ruby"]
sp = spm.SentencePieceProcessor(model_file="data/tokenizer/tokenizer.model")

print("Building model architecture...")
model = build_model(params["VOCAB_SIZE"], params["MAX_LEN"], params["EMBED_DIM"], params["NUM_TRANSFORMER_BLOCKS"], params["NUM_HEADS"], num_kv_heads=params.get("NUM_KV_HEADS"))

weight_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../model/ruby_10.weights.h5")
print(f"Loading {weight_path}...")
model.load_weights(weight_path)

TEMPERATURE = 0.85
TOP_K = 40
TOP_P = 0.85
REPETITION_PENALTY = 1.18
MAX_NEW_TOKENS = 150

def format_prompt(text):
    text = text.strip()

    question_suffixes = [
        "ってなに？", "って何？", "ってなに", "って何", "って？", "って",
        "はどうして？", "はどうして", "はなんで？", "はなんで", "なんで？", "なんで",
        "とは何ですか？", "とは何ですか", "とは何か？", "とは何か",
        "について教えて", "について", "とは？", "とは", "？", "?"
    ]
    for suffix in question_suffixes:
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break

    text = text.strip()
    if not text:
        return ""

    if text.endswith("、"):
        return text
    if text[-1] in ["は", "が", "で", "に", "を"]:
        return text + "、"

    return text + "とは、"

while True:
    try:
        user_input = input("調べたい単語は？ > ")
    except (EOFError, KeyboardInterrupt):
        break

    if user_input.lower() in ['exit', 'quit']:
        break
    if not user_input.strip():
        continue

    prompt = format_prompt(user_input)
    print(f"\n【生成プロンプト】: {prompt}")

    sys.stdout.write("【本文】: " + prompt)
    sys.stdout.flush()

    generated_ids = sp.encode(prompt)
    last_printed_len = len(prompt)

    for step in range(MAX_NEW_TOKENS):
        curr_input = generated_ids[-params["MAX_LEN"]:]
        padded_input = tf.keras.preprocessing.sequence.pad_sequences([curr_input], maxlen=params["MAX_LEN"], padding="post", value=sp.pad_id())

        logits = model(padded_input, training=False)[0, len(curr_input)-1, :]
        logits = tf.cast(logits, tf.float32).numpy()

        for gid in set(generated_ids):
            if logits[gid] > 0:
                logits[gid] /= REPETITION_PENALTY
            else:
                logits[gid] *= REPETITION_PENALTY

        logits = logits / TEMPERATURE
        logits_t = top_k_top_p_logits(tf.convert_to_tensor([logits]), k=TOP_K, p=TOP_P)
        next_id = int(tf.random.categorical(logits_t, 1)[0, 0].numpy())

        if next_id == sp.eos_id():
            break

        generated_ids.append(next_id)

        full_text = sp.decode(generated_ids)
        new_text = full_text[last_printed_len:]
        sys.stdout.write(new_text)
        sys.stdout.flush()

        last_printed_len = len(full_text)
