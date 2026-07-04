import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/corpus/token/train_wiki.bin')
    parser.add_argument('--output_val', type=str, default='data/corpus/token/val_wiki.bin')
    parser.add_argument('--val_tokens', type=int, default=300_000)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} does not exist.")
        return

    file_size_bytes = os.path.getsize(args.input)
    val_size_bytes = args.val_tokens * 2
    
    if file_size_bytes <= val_size_bytes:
        print("Error: Input file is too small to extract the requested validation tokens.")
        return

    print(f"Extracting {args.val_tokens:,} tokens ({val_size_bytes / (1024**2):.2f} MB) from the end of {args.input} ...")

    with open(args.input, 'rb') as f:
        f.seek(-val_size_bytes, os.SEEK_END)
        val_data = f.read(val_size_bytes)
        
    with open(args.output_val, 'wb') as f_val:
        f_val.write(val_data)

    print(f"Saved validation data to: {args.output_val}")

    new_size_bytes = file_size_bytes - val_size_bytes
    os.truncate(args.input, new_size_bytes)
    
    print(f"Truncated original training file to: {new_size_bytes / (1024**3):.2f} GB ({new_size_bytes // 2:,} tokens)")

if __name__ == '__main__':
    main()
