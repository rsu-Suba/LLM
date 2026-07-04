# HuggingFace, 10B : python prepare_data.py --source hf --out_file data/corpus/token/train_10B.bin --target_tokens 10000000000
# Local :  python prepare_data.py --source local --input_file data/corpus/text/wiki.txt --out_file data/corpus/token/wiki_train.bin
import os
import argparse
import sentencepiece as spm
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

def process_huggingface(sp, out_file, target_tokens):
    print("Loading HuggingFace streaming dataset (izumi-lab/cc100-ja-filter-ja-normal)...")
    ds = load_dataset('izumi-lab/cc100-ja-filter-ja-normal', split='train', streaming=True)
    BATCH_SIZE = 5_000_000 # 5M tokens per flush
    
    if os.path.exists(out_file):
        os.remove(out_file)
        
    f = open(out_file, 'wb')
    buffer = []
    total_tokens = 0
    
    pbar_args = {"desc": "Tokenizing HF", "unit": "tok"}
    if target_tokens is not None:
        pbar_args["total"] = target_tokens
    pbar = tqdm(**pbar_args)
    
    try:
        for item in ds:
            text = item['text'].strip()
            if not text:
                continue
                
            encoded = sp.encode(text)
            encoded.append(sp.eos_id())
            buffer.extend(encoded)
            
            if len(buffer) >= BATCH_SIZE:
                arr = np.array(buffer, dtype=np.uint16)
                f.write(arr.tobytes())
                total_tokens += len(buffer)
                pbar.update(len(buffer))
                buffer = []
                
            if target_tokens is not None and total_tokens >= target_tokens:
                break
                
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            f.write(arr.tobytes())
            total_tokens += len(buffer)
            pbar.update(len(buffer))
            
    except KeyboardInterrupt:
        print("\nInterrupted. Saving processed tokens so far...")
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            f.write(arr.tobytes())
            total_tokens += len(buffer)
            
    finally:
        f.close()
        pbar.close()
        
    return total_tokens

def process_local(sp, input_file, out_file, target_tokens):
    print(f"Loading local text corpus from {input_file}...")
    BATCH_SIZE = 1_000_000 # 1M tokens per flush
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"{input_file} not found.")
        
    if os.path.exists(out_file):
        os.remove(out_file)
        
    f = open(out_file, 'wb')
    buffer = []
    total_tokens = 0
    
    pbar_args = {"desc": f"Tokenizing {os.path.basename(input_file)}", "unit": "tok"}
    if target_tokens is not None:
        pbar_args["total"] = target_tokens
    pbar = tqdm(**pbar_args)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as text_file:
            for line in text_file:
                text = line.strip()
                if not text:
                    continue
                    
                encoded = sp.encode(text)
                encoded.append(sp.eos_id())
                buffer.extend(encoded)
                
                if len(buffer) >= BATCH_SIZE:
                    arr = np.array(buffer, dtype=np.uint16)
                    f.write(arr.tobytes())
                    total_tokens += len(buffer)
                    pbar.update(len(buffer))
                    buffer = []
                    
                if target_tokens is not None and total_tokens >= target_tokens:
                    break
                    
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            f.write(arr.tobytes())
            total_tokens += len(buffer)
            pbar.update(len(buffer))
    except KeyboardInterrupt:
        print("\nInterrupted. Saving processed tokens so far...")
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            f.write(arr.tobytes())
            total_tokens += len(buffer)
            
    finally:
        f.close()
        pbar.close()
        
    return total_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, choices=['hf', 'local'], required=True)
    parser.add_argument('--input_file', type=str, default='data/corpus/text/wiki.txt')
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--target_tokens', type=int, default=None)
    
    args = parser.parse_args()
    
    sp = spm.SentencePieceProcessor(model_file='data/tokenizer/tokenizer.model')
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
    
    if args.source == 'hf':
        total = process_huggingface(sp, args.out_file, args.target_tokens)
    else:
        total = process_local(sp, args.input_file, args.out_file, args.target_tokens)
        
    print(f"\nSuccessfully created {args.out_file} with {total:,} tokens!")
    print(f"File size: {os.path.getsize(args.out_file) / (1024**3):.2f} GB")

if __name__ == '__main__':
    main()
