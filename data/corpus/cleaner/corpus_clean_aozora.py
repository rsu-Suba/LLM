import json
import re
from tqdm import tqdm

INPUT_JSONL_PATH = "data/extracted_text/Aozora/aozorabunko.jsonl"
OUTPUT_CORPUS_PATH = "data/extracted_text/Aozora/aozora.txt"

print(f"Loaded < {INPUT_JSONL_PATH}")

with open(INPUT_JSONL_PATH, 'r', encoding='utf-8') as f_in, \
     open(OUTPUT_CORPUS_PATH, 'w', encoding='utf-8') as f_out:
    
    for line in tqdm(f_in, desc="Extracting"):
        try:
            data = json.loads(line)
            text = data.get("text", "")
            text = re.sub(r'\n{2,}', '<<PARAGRAPH_BREAK>>', text)
            text = text.replace('\n', '')
            text = text.replace('<<PARAGRAPH_BREAK>>', '\n')
            text = re.sub(r'[ \t　]+', ' ', text).strip()

            if text:
                f_out.write(text + '\n')
        except json.JSONDecodeError:
            print(f"Error: Skipped invalid line {line.strip()}")

print(f"Extract complete: '{OUTPUT_CORPUS_PATH}'")
