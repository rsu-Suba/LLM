import random
import os
import argparse
import re
from tqdm import tqdm

MC4_PATH = "data/corpus/text/mc4.txt"
AOZORA_PATH = "data/corpus/text/aozora.txt"
WIKI_PATH = "data/corpus/text/wiki.txt"

def get_paragraphs(file_path, min_len=150, min_periods=2):
    paragraphs = []
    print(f"Scanning {os.path.basename(file_path)} for paragraphs...")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            text = line.strip()
            if len(text) >= min_len and text.count("。") >= min_periods:
                if len(text) < 2000: 
                    paragraphs.append(text)
            
            if len(paragraphs) >= 1000000: break
            
    return paragraphs

def build_corpus(mode="train"):
    if mode == "train":
        targets = {"mc4": 500000, "aozora": 350000, "wiki": 150000}
        output_path = "data/corpus/text/train_raw.txt"
    else:
        targets = {"mc4": 50000, "aozora": 35000, "wiki": 15000}
        output_path = "data/corpus/text/val_raw.txt"
    
    print(f"\n--- Building {mode.upper()} Paragraph Corpus ---")
    
    final_paragraphs = []

    mc4_src = get_paragraphs(MC4_PATH, min_len=100, min_periods=1)
    random.shuffle(mc4_src)
    split_idx = int(len(mc4_src) * 0.9)
    mc4_pool = mc4_src[:split_idx] if mode == "train" else mc4_src[split_idx:]
    final_paragraphs.extend((mc4_pool * (targets["mc4"] // len(mc4_pool) + 1))[:targets["mc4"]])
    print(f"  mC4 Paragraphs: {targets['mc4']:,}")

    aozora_src = get_paragraphs(AOZORA_PATH, min_len=150, min_periods=2)
    random.shuffle(aozora_src)
    split_idx = int(len(aozora_src) * 0.9)
    aozora_pool = aozora_src[:split_idx] if mode == "train" else aozora_src[split_idx:]
    final_paragraphs.extend((aozora_pool * (targets["aozora"] // len(aozora_pool) + 1))[:targets["aozora"]])
    print(f"  Aozora Paragraphs: {targets['aozora']:,}")

    wiki_src = get_paragraphs(WIKI_PATH, min_len=200, min_periods=3)
    random.shuffle(wiki_src)
    split_idx = int(len(wiki_src) * 0.9)
    wiki_pool = wiki_src[:split_idx] if mode == "train" else wiki_src[split_idx:]
    final_paragraphs.extend(wiki_pool[:targets["wiki"]])
    print(f"  Wiki Paragraphs: {targets['wiki']:,}")

    print("Finalizing shuffle...")
    random.shuffle(final_paragraphs)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in final_paragraphs:
            f.write(p + "\n")
            
    print(f"Success! {mode.upper()} paragraph corpus created at {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'val'])
    args = parser.parse_args()
    build_corpus(args.mode)
