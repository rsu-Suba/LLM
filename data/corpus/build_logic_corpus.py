import random
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "cleaner"))
try:
    from ng_words import ng_words
except ImportError:
    ng_words = []

MC4_PATH = "data/corpus/text/mc4.txt"
AOZORA_PATH = "data/corpus/text/aozora.txt"
WIKI_PATH = "data/corpus/text/wiki.txt"

LOGIC_KEYWORDS = [
    "まず", "次に", "最後に", "第一に", "したがって", "ゆえに", "よって", "そのため",
    "なぜなら", "理由として", "背景には", "つまり", "具体的には", "結論として", "もし", "ならば"
]

def get_block_score(block, label):
    score = 0
    for kw in LOGIC_KEYWORDS:
        if kw in block: score += 2
    if "である。" in block or "となる。" in block: score += 1
    if label == "AOZORA": score += 1
    return score

def collect_blocks(file_path, label, target_count=150000, lines_per_block=10):
    print(f"Processing {label} ({os.path.basename(file_path)})...")
    if not os.path.exists(file_path): return []

    pool = []
    current_block = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or len(line) < 30: continue
            
            current_block.append(line)
            if len(current_block) >= lines_per_block:
                block_text = "\n".join(current_block)
                if not any(nw in block_text for nw in ng_words):
                    score = get_block_score(block_text, label)
                    if score > 0 or random.random() < 0.2:
                        pool.append((score, block_text))
                current_block = []
            
            if len(pool) >= 1000000: break 

    pool.sort(key=lambda x: x[0], reverse=True)
    selected = [b for score, b in pool[:target_count]]
    print(f"  -> {label}: Collected {len(selected):,} blocks")
    return selected

def build_corpus():
    print("\n--- Building Logic Corpus (3.5x Scale) ---")
    
    wiki_blocks = collect_blocks(WIKI_PATH, "WIKI", target_count=150000)
    aozora_blocks = collect_blocks(AOZORA_PATH, "AOZORA", target_count=100000)
    mc4_blocks = collect_blocks(MC4_PATH, "MC4", target_count=100000)
    
    all_blocks = wiki_blocks + aozora_blocks + mc4_blocks
    random.shuffle(all_blocks)
    
    split_idx = int(len(all_blocks) * 0.99)
    train_blocks = all_blocks[:split_idx]
    val_blocks = all_blocks[split_idx:]
    
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
