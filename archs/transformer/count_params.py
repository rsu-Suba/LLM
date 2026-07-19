import os
import sys
import yaml
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from train.train_pt import ModelPT

def format_params(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)

def main():
    parser = argparse.ArgumentParser(description="Count model parameters for Normal Transformer")
    parser.add_argument("--model", type=str, default="gold", help="対象モデル名 (例: gold)")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'param.yaml')
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if args.model not in config_data:
        if args.model == "default" and "default_model" in config_data:
            args.model = config_data["default_model"]
        else:
            print(f"Model '{args.model}' not found in {config_path}")
            print(f"Available models: {list(config_data.keys())}")
            return
            
    p = config_data[args.model]
    print(f"\n--- Model: {args.model} (Normal Transformer)")
    
    model = ModelPT(
        vocab_size=p['VOCAB_SIZE'],
        dim=p['EMBED_DIM'],
        heads=p['NUM_HEADS'],
        kv_heads=p.get('NUM_KV_HEADS', p['NUM_HEADS']),
        ff_dim=p.get('FF_DIM', p['EMBED_DIM'] * 4),
        layers=p['NUM_TRANSFORMER_BLOCKS']
    )
    
    total_params = sum(param.numel() for param in model.parameters())
    embed_params = sum(param.numel() for param in model.embed.parameters())
    blocks_params = sum(param.numel() for param in model.blocks.parameters())
    
    print(f"\nTotal Parameters:    {format_params(total_params):>10} ({total_params:,})")
    print(f"Embedding:           {format_params(embed_params):>10}")
    print(f"Transformer Blocks:  {format_params(blocks_params):>10}")
    print(f"LM Head (Tied):      {format_params(embed_params):>10}\n")

if __name__ == "__main__":
    main()
