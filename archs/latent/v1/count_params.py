import os
import sys
import yaml
import torch
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from model import build_latent_brain_from_params

def format_params(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)

def main():
    parser = argparse.ArgumentParser(description="Count model parameters")
    parser.add_argument("--model", type=str, default="brain_250M", help="対象モデル名 (例: brain_250M)")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), 'param.yaml')
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    if args.model not in config_data:
        print(f"Model '{args.model}' not found in {config_path}")
        print(f"Available models: {list(config_data.keys())}")
        return
        
    params = config_data[args.model]
    print(f"\n--- Model: {args.model}")
    
    model = build_latent_brain_from_params(params)
    
    # Module params
    total_params = sum(p.numel() for p in model.parameters())
    embed_params = sum(p.numel() for p in model.embed.parameters())
    
    # Encoder params
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    compressor_params = sum(p.numel() for p in model.compressor.parameters())
    
    # Reasoner params
    reasoner_params = sum(p.numel() for p in model.reasoner.parameters())
    
    # Decoder params
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
    
    print(f"\nTotal Parameters:    {format_params(total_params):>10} ({total_params:,})")
    print(f"Embedding:           {format_params(embed_params):>10}")
    print(f"Encoder (Linear):    {format_params(encoder_params):>10}")
    print(f"Compressor (Latent): {format_params(compressor_params):>10}")
    print(f"Reasoner (Blocks):   {format_params(reasoner_params):>10}")
    print(f"Decoder (Blocks):    {format_params(decoder_params):>10}")
    print(f"LM Head (Tied):      {format_params(lm_head_params):>10}\n")

if __name__ == "__main__":
    main()

