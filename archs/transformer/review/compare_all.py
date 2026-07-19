import tensorflow as tf, numpy as np, yaml, time
import sentencepiece as spm
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
sys.path.insert(1, os.path.join(_HERE, '../../..'))
from model import build_model, TokenEmbedding, TransformerBlock, RMSNorm, WarmupCosineDecay, TiedOutput
from generation_utils import top_k_top_p_logits

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try: tf.config.experimental.set_memory_growth(gpu, True)
        except: pass

tf.keras.mixed_precision.set_global_policy('mixed_float16')

with open(os.path.join(os.path.dirname(__file__), "../param.yaml")) as f:
    cfg = yaml.safe_load(f)

sp = spm.SentencePieceProcessor(model_file="data/tokenizer/tokenizer.model")
eos_id = sp.eos_id()
pad_id = sp.pad_id()

PROMPT = "おいしいオムライスを作るために大事なことは、"
N_SAMPLES = 3000
VAL_PATH = "data/corpus/token/val.bin"

TARGETS = [
    # {"label": "Diamond-1B", "config": "diamond-1b", "path": "model/diamond.weights - Copy.h5", "by_name": True},
]

def load_val(batch_size, max_len):
    raw = np.fromfile(VAL_PATH, dtype=np.uint16).astype(np.int32)
    total = len(raw) // (max_len + 1)
    data = raw[:total * (max_len + 1)].reshape((total, max_len + 1))
    idx = np.random.choice(total, min(N_SAMPLES, total), replace=False)
    sampled = data[idx]
    ds = tf.data.Dataset.from_tensor_slices(sampled)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(lambda b: (b[:, :-1], b[:, 1:]), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE), len(sampled)

def generate(model, max_len, temperature, top_k, top_p, rep_penalty, max_tokens=60):
    ids = sp.encode(PROMPT)
    for _ in range(max_tokens):
        curr = ids[-max_len:]
        pad = tf.keras.preprocessing.sequence.pad_sequences([curr], maxlen=max_len, padding='post', value=pad_id)
        logits = tf.cast(model(pad, training=False)[0, len(curr)-1, :], tf.float32).numpy()
        for gid in set(ids):
            if logits[gid] > 0:
                logits[gid] /= rep_penalty
            else:
                logits[gid] *= rep_penalty
        lt = tf.convert_to_tensor([logits]) / temperature
        lt = top_k_top_p_logits(lt, k=top_k, p=top_p)
        nid = int(tf.random.categorical(lt, 1)[0, 0].numpy())
        if nid == eos_id:
            break
        ids.append(nid)
        txt = sp.decode(ids[len(sp.encode(PROMPT)):])
        if len(ids) - len(sp.encode(PROMPT)) >= max_tokens:
            if any(m in txt[-4:] for m in ["。","！","？","!","?"]): break
    return sp.decode(ids)

results = []
for t in TARGETS:
    p = cfg[t["config"]]
    gen = p.get("generation", {})
    MAX_LEN = p["MAX_LEN"]
    TEMP = gen.get("temperature", 0.8)
    TOPK = gen.get("top_k", 40)
    TOPP = gen.get("top_p", 0.9)
    REP = gen.get("repetition_penalty", 1.2)
    BS = p.get("BATCH_SIZE", 16)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", t["path"])

    print(f"  {t['label']}  ({model_path})")
    print(f"  Config: EMBED={p['EMBED_DIM']} Blocks={p['NUM_TRANSFORMER_BLOCKS']} KV={p.get('NUM_KV_HEADS','full')}")

    if not os.path.exists(model_path):
        print("ファイルなし"); continue

    model = build_model(p['VOCAB_SIZE'], MAX_LEN, p['EMBED_DIM'], p['NUM_TRANSFORMER_BLOCKS'], p['NUM_HEADS'], num_kv_heads=p.get('NUM_KV_HEADS'))
    if t["by_name"]:
        model.load_weights(model_path, by_name=True)
    else:
        model.load_weights(model_path)
    print(f"ロード完了  params={model.count_params():,}")

    t0 = time.time()
    text = generate(model, MAX_LEN, TEMP, TOPK, TOPP, REP)
    gen_t = time.time() - t0
    gen_tok = len(sp.encode(text)) - len(sp.encode(PROMPT))
    print(f"  生成: {gen_t:.1f}s  {gen_tok}tok  ({gen_tok/gen_t:.2f} tok/s)")

    ds, n = load_val(BS, MAX_LEN)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    loss = model.evaluate(ds, verbose=0)
    ppl = float(np.exp(loss))
    print(f"  PPL : {ppl:.1f}  (Loss={loss:.4f},  samples={n:,})")

    results.append({
        "label": t["label"], "path": model_path,
        "params": model.count_params(),
        "ppl": ppl, "loss": float(loss),
        "text": text, "gen_tok": gen_tok, "gen_t": gen_t,
        "embed": p['EMBED_DIM'], "blocks": p['NUM_TRANSFORMER_BLOCKS'],
    })

    del model; tf.keras.backend.clear_session()

ppl_sorted = sorted(results, key=lambda x: x["ppl"])
print(f"\n{'PPL順位':^6} {'モデル':<20} {'PPL':>8} {'Loss':>7} {'パラメータ':>12}")
for i, r in enumerate(ppl_sorted):
    print(f"{r['label']:<20} {r['ppl']:>8.1f} {r['loss']:>7.4f} {r['params']:>12,}")

print("\n── 生成テキスト ──")
for r in results:
    print(f"\n┌─ {r['label']}")
    for line in r["text"].splitlines():
        print(f"│  {line}")
    print(f"└  ({r['gen_tok']}tok / {r['gen_t']:.1f}s)")
print()
