import numpy as np
import sentencepiece as spm
import os
from collections import Counter

TOKENIZER_PATH = "data/tokenizer/tokenizer.model"
BIN_PATH = "data/corpus/train.bin"
target_word = ""

def count_tokens(target_word=target_word, top_n=20):
    if not os.path.exists(BIN_PATH):
        print(f"Error: {BIN_PATH} not found. Please run pretokenize.py first.")
        return

    sp = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    target_id = sp.piece_to_id(target_word)
    
    print(f"Loading binary data from {BIN_PATH}...")
    data = np.memmap(BIN_PATH, dtype=np.uint16, mode='r')
    
    print(f"Counting occurrences of '{target_word}' (ID: {target_id})...")
    count = np.count_nonzero(data == target_id)
    
    print(f"\n--- Result ---")
    print(f"'{target_word}' appeared: {count:,} times")
    print(f"Percentage: {count / len(data):.4%}")

    if top_n > 0:
        print(f"\n--- Top {top_n} Frequent Tokens ---")
        sample_size = min(len(data), 10000000)
        sample_data = data[:sample_size]
        common = Counter(sample_data).most_common(top_n)
        
        for rank, (token_id, freq) in enumerate(common, 1):
            piece = sp.id_to_piece(int(token_id))
            print(f"{rank}. ID {token_id:5}: {piece:<15} ({freq:,} times in sample)")

if __name__ == "__main__":
    count_tokens(target_word)
