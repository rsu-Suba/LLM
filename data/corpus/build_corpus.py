import random
import os

MC4_PATH = "data/corpus/mc4.txt"
AOZORA_PATH = "data/corpus/aozora.txt"
WIKI_PATH = "data/corpus/wiki.txt"

OUTPUT_PATH = "data/corpus/train.txt"

def build():
    print("Building 6:3:1 corpus...")
    
    with open(MC4_PATH, "r", encoding="utf-8") as f:
        mc4 = [l.strip() for l in f if l.strip()]
    random.shuffle(mc4)
    mc4 = mc4[:300000] 

    wiki_final = []
    with open(WIKI_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 1000000: break
            if "。" in line and len(line) > 50:
                wiki_final.append(line.strip())
    random.shuffle(wiki_final)
    wiki_final = wiki_final[:150000]

    with open(AOZORA_PATH, "r", encoding="utf-8") as f:
        aozora = [l.strip() for l in f if l.strip()]
    random.shuffle(aozora)
    aozora = aozora[:50000]


    all_lines = mc4 + aozora + wiki_final
    random.shuffle(all_lines)
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")
            
    print(f"Success! Golden ratio corpus saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    build()
