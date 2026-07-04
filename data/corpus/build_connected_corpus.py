import random
import os
from tqdm import tqdm

from cleaner.ng_words import ng_words as NG_KEYWORDS

MC4_PATH = "data/corpus/text/mc4.txt"
AOZORA_PATH = "data/corpus/text/aozora.txt"
WIKI_PATH = "data/corpus/text/wiki.txt"

def is_quality_block(block):
    return not any(kw in block for kw in NG_KEYWORDS)

def get_connected_blocks(file_path, label, lines_per_block=10, min_line_len=30, max_blocks=60000):
    blocks = []
    print(f"Reading {label} ({os.path.basename(file_path)})...")
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        current_block = []
        for line in f:
            line = line.strip()
            if not line or len(line) < min_line_len:
                continue
                
            current_block.append(line)
            
            if len(current_block) >= lines_per_block:
                block_text = "\n".join(current_block)
                if label != "MC4" or is_quality_block(block_text):
                    blocks.append(block_text)
                current_block = []
            
            if len(blocks) >= max_blocks: break
            
        if current_block and len(current_block) > 1:
            block_text = "\n".join(current_block)
            if label != "MC4" or is_quality_block(block_text):
                blocks.append(block_text)
            
    print(f"  -> Collected {len(blocks):,} quality blocks from {label}")
    return blocks

def build_corpus():
    print("\n--- High-Quality Corpus ---")
    
    mc4_blocks = get_connected_blocks(MC4_PATH, "MC4", max_blocks=60000)
    aozora_blocks = get_connected_blocks(AOZORA_PATH, "AOZORA", max_blocks=60000)
    wiki_blocks = get_connected_blocks(WIKI_PATH, "WIKI", max_blocks=60000)
    
    all_blocks = mc4_blocks + aozora_blocks + wiki_blocks
    print(f"Total blocks combined: {len(all_blocks):,}")
    
    random.shuffle(all_blocks)
    
    target_train_lines = 1200000
    target_val_lines = 15000
    
    train_blocks = []
    val_blocks = []
    current_train_lines = 0
    current_val_lines = 0
    
    for b in all_blocks:
        b_lines = b.count("\n") + 1
        if current_train_lines < target_train_lines:
            train_blocks.append(b)
            current_train_lines += b_lines
        elif current_val_lines < target_val_lines:
            val_blocks.append(b)
            current_val_lines += b_lines
        else:
            break
            
    def save_blocks(blocks, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line_count = 0
        with open(path, "w", encoding="utf-8") as f:
            for b in blocks:
                f.write(b + "\n")
                line_count += b.count("\n") + 1
        print(f"Saved {line_count:,} lines to {path}")

    save_blocks(train_blocks, "data/corpus/text/train.txt")
    save_blocks(val_blocks, "data/corpus/text/val.txt")

if __name__ == "__main__":
    build_corpus()
