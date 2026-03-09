import numpy as np
import os

BIN_PATH = "data/corpus/train.bin"
VOCAB_SIZE = 28000

def check_data():
    if not os.path.exists(BIN_PATH):
        print(f"Error: {BIN_PATH} not found.")
        return

    print(f"Checking data: {BIN_PATH}")
    data = np.fromfile(BIN_PATH, dtype=np.uint16)
    
    max_id = np.max(data)
    min_id = np.min(data)
    
    print(f"Total tokens: {len(data):,}")
    print(f"Max ID found: {max_id}")
    print(f"Min ID found: {min_id}")
    
    if max_id >= VOCAB_SIZE:
        print(f"CRITICAL ERROR: Found IDs >= VOCAB_SIZE ({VOCAB_SIZE})")
        invalid_indices = np.where(data >= VOCAB_SIZE)[0]
        print(f"Number of invalid IDs: {len(invalid_indices)}")
        print(f"First 10 invalid indices: {invalid_indices[:10]}")
        print(f"Invalid values: {data[invalid_indices[:10]]}")
    else:
        print("Data range is OK.")

    if np.any(np.isnan(data.astype(float))):
        print("ERROR: Found NaNs in data")
    
    print("Check finished.")

if __name__ == "__main__":
    check_data()
