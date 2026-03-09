import os
import re
from datasets import load_dataset
from tqdm import tqdm
from cleaner.ng_words import ng_words

OUTPUT_PATH = "data/corpus/mc4.txt"
TARGET_COUNT = 1_000_000
MIN_LEN = 60
MAX_LEN = 400
JOSHI_REQUIRED = ["の", "は", "が", "を", "に", "と"]
MIN_JOSHI_TYPES = 4
STOP_CHARS = ["。", "！", "？"]

def is_clean(text):
    text = text.strip()
    if not (MIN_LEN <= len(text) <= MAX_LEN): return False
    if not any(text.endswith(sc) for sc in STOP_CHARS): return False
    text_lower = text.lower()

    if any(kw in text_lower for kw in ng_words): return False
    joshi_found = sum(1 for j in JOSHI_REQUIRED if j in text)

    if joshi_found < MIN_JOSHI_TYPES: return False
    hiragana_count = len(re.findall(r'[ぁ-ん]', text))

    if hiragana_count / len(text) < 0.35: return False
    if len(re.findall(r'\d{4}', text)) > 1: return False
    if re.search(r'(.)\1{3,}', text): return False
    if re.search(r'[一-龠]{10,}', text): return False

    return True

def fetch_large_mc4():
    print(f"Loading C4 Japanese (streaming)...")
    ds = load_dataset("allenai/c4", "ja", split="train", streaming=True)
    
    count = 0
    scanned = 0
    
    print(f"Fetching {TARGET_COUNT} sentences...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        pbar = tqdm(total=TARGET_COUNT)
        
        for entry in ds:
            scanned += 1
            raw_text = entry["text"].replace("\n", " ")
            sentences = re.split(r'(?<=[。！？])\s*', raw_text)
            
            for s in sentences:
                s = s.strip()
                if is_clean(s):
                    f.write(s + "\n")
                    count += 1
                    pbar.update(1)
                    if count >= TARGET_COUNT:
                        break
            
            if count >= TARGET_COUNT:
                break
                
    pbar.close()
    print(f"\n--- Refinement Complete ---")
    print(f"Saved: {count} sentences from {scanned} documents scanned.")

if __name__ == "__main__":
    fetch_large_mc4()
