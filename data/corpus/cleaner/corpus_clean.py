import os
import re
import multiprocessing
from tqdm import tqdm
import sys

from ng_words import ng_words

def compile_ng_pattern(keywords):
    pattern = "|".join(re.escape(kw) for kw in keywords)
    return re.compile(pattern, re.IGNORECASE)

NG_PATTERN = compile_ng_pattern(ng_words)

def is_clean(line):
    line = line.strip()
    if not line: return None
    
    if NG_PATTERN.search(line):
        return None
    
    if line.count("。") < 1: return None
    if len(line) < 50: return None
    
    return line

def process_chunk(lines):
    results = []
    for line in lines:
        cleaned = is_clean(line)
        if cleaned:
            results.append(cleaned)
    return results

def fast_clean_file(input_path, output_path):
    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    total = len(all_lines)
    num_cores = multiprocessing.cpu_count()
    chunk_size = total // num_cores
    chunks = [all_lines[i : i + chunk_size] for i in range(0, total, chunk_size)]
    
    print(f"Cleaning with {num_cores} cores using optimized regex...")
    with multiprocessing.Pool(num_cores) as pool:
        chunk_results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))
    
    print("Writing results...")
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunk_results:
            for line in chunk:
                f.write(line + "\n")
                
    print(f"Finished. Output saved to {output_path}")

if __name__ == "__main__":
    input_file = "data/corpus/text/train_raw.txt"
    output_file = "data/corpus/text/train.txt"
    if os.path.exists(input_file):
        fast_clean_file(input_file, output_file)
    else:
        print(f"File not found: {input_file}")
    
    input_file = "data/corpus/text/val_raw.txt"
    output_file = "data/corpus/text/val.txt"
    if os.path.exists(input_file):
        fast_clean_file(input_file, output_file)
    else:
        print(f"File not found: {input_file}")
