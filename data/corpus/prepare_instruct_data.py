import numpy as np
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm
import os
import yaml

def main():
    print("Loading Databricks Dolly 15k Japanese dataset...")
    ds = load_dataset("kunishou/databricks-dolly-15k-ja")
    sp = spm.SentencePieceProcessor(model_file='data/tokenizer/tokenizer.model')
    eos_id = sp.eos_id()
    pad_id = sp.pad_id()
    
    with open("model_param.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    MAX_LEN = config["ruby"]["MAX_LEN"]
    
    x_seqs = []
    y_seqs = []
    w_seqs = []
    
    print("Formatting and padding instruction data with Loss Masking...")
    for item in tqdm(ds['train']):
        instruction = item['instruction'].strip()
        context = item['input'].strip()
        response = item['output'].strip()
        
        if context:
            continue
            
        user_text = f"ユーザー: {instruction}\n"
        ai_text = f"AI: {response}"
        
        user_ids = sp.encode(user_text)
        ai_ids = sp.encode(ai_text)
        
        full_ids = user_ids + ai_ids + [eos_id]
        weights = [0.0] * len(user_ids) + [1.0] * len(ai_ids) + [1.0]
        
        if len(full_ids) <= MAX_LEN + 1:
            pad_len = (MAX_LEN + 1) - len(full_ids)
            full_ids = full_ids + [pad_id] * pad_len
            weights = weights + [0.0] * pad_len
            
            x_seqs.append(full_ids[:-1])
            y_seqs.append(full_ids[1:])
            w_seqs.append(weights[1:])
        else:
            full_ids = full_ids[:MAX_LEN] + [eos_id]
            weights = weights[:MAX_LEN] + [1.0]
            x_seqs.append(full_ids[:-1])
            y_seqs.append(full_ids[1:])
            w_seqs.append(weights[1:])

    basic_chats = [
        ("こんにちは", "こんにちは！何かお手伝いできることはありますか？"),
        ("何ができる？", "私はAIアシスタントです。質問に答えたり、文章を作成したりできます。"),
        ("自己紹介して", "私はAIアシスタントです。日本語で会話することができます。"),
        ("日本の首都はどこですか？", "日本の首都は東京です。"),
        ("おいしいオムライスを作る手順を教えて", "オムライスの作り方ですね。まずチキンライスを作り、その上に薄焼き卵を乗せてケチャップをかけます。")
    ]
    for q, a in basic_chats:
        for _ in range(50):
            u_ids = sp.encode(f"ユーザー: {q}\n")
            a_ids = sp.encode(f"AI: {a}")
            f_ids = u_ids + a_ids + [eos_id]
            w_arr = [0.0] * len(u_ids) + [1.0] * len(a_ids) + [1.0]
            
            pad_len = (MAX_LEN + 1) - len(f_ids)
            f_ids = f_ids + [pad_id] * pad_len
            w_arr = w_arr + [0.0] * pad_len
            
            x_seqs.append(f_ids[:-1])
            y_seqs.append(f_ids[1:])
            w_seqs.append(w_arr[1:])
            
    print(f"\nTotal formatted sequences: {len(x_seqs):,}")
    
    os.makedirs('data/corpus/token', exist_ok=True)
    
    np.random.seed(42)
    indices = np.random.permutation(len(x_seqs))
    
    x_arr = np.array(x_seqs, dtype=np.int32)[indices]
    y_arr = np.array(y_seqs, dtype=np.int32)[indices]
    w_arr = np.array(w_seqs, dtype=np.float32)[indices]
    
    val_size = int(len(x_arr) * 0.05)
    
    np.savez("data/corpus/token/instruct_train_masked.npz", 
             x=x_arr[:-val_size], y=y_arr[:-val_size], w=w_arr[:-val_size])
    np.savez("data/corpus/token/instruct_val_masked.npz", 
             x=x_arr[-val_size:], y=y_arr[-val_size:], w=w_arr[-val_size:])
    
    print("Done! Masked instruction tuning dataset is ready.")

if __name__ == "__main__":
    main()
