import numpy as np
import tensorflow_text as tf_text
import os
from tqdm import tqdm

TOKENIZER_PATH = "data/tokenizer/tokenizer.model"

def process_file(input_path, output_path):
    print(f"Loading tokenizer...")
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer_model = f.read()
    tokenizer = tf_text.SentencepieceTokenizer(model=tokenizer_model, add_bos=False, add_eos=False)

    print(f"Opening {input_path}...")
    all_token_ids = []
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        chunk_size = 10000
        lines = []
        for line in tqdm(f, desc=f"Tokenizing {os.path.basename(input_path)}"):
            lines.append(line.strip())
            if len(lines) >= chunk_size:
                tokens = tokenizer.tokenize(lines)
                flat_ids = tokens.flat_values.numpy().astype(np.uint16)
                all_token_ids.append(flat_ids)
                lines = []
        
        if lines:
            tokens = tokenizer.tokenize(lines)
            flat_ids = tokens.flat_values.numpy().astype(np.uint16)
            all_token_ids.append(flat_ids)

    print("Concatenating all IDs...")
    final_ids = np.concatenate(all_token_ids)
    
    print(f"Saving to {output_path}...")
    final_ids.tofile(output_path)
    print(f"Total tokens: {len(final_ids):,}\n")

if __name__ == "__main__":
    process_file("data/corpus/train.txt", "data/corpus/train.bin")
    process_file("data/corpus/val.txt", "data/corpus/val.bin")
