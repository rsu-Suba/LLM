import yaml
import argparse
import os

def calc_params(params):
    vocab_size = params['VOCAB_SIZE']
    max_len = params['MAX_LEN']
    embed_dim = params['EMBED_DIM']
    num_blocks = params['NUM_TRANSFORMER_BLOCKS']
    num_heads = params['NUM_HEADS']

    print(f"\n--- Model Spec: {embed_dim} dim, {num_blocks} layers, {num_heads} heads, {vocab_size} vocab ---")
    
    # 1. Embedding
    token_embed = vocab_size * embed_dim
    pos_embed = max_len * embed_dim
    pos_token_embed = token_embed + pos_embed
    print(f"  Embedding (Token+Pos): {pos_token_embed:,}")

    # 2. Transformer Blocks
    # Multi-Head Attention
    qkv = 3 * (embed_dim * embed_dim + embed_dim)
    wo = embed_dim * embed_dim + embed_dim
    att_param = qkv + wo
    
    # Feed Forward
    ff_dim = embed_dim * 4
    ffn1 = embed_dim * ff_dim + ff_dim
    ffn2 = ff_dim * embed_dim + embed_dim
    ffn_param = ffn1 + ffn2
    
    # RMSNorm
    rmsnorm_param = embed_dim * 2
    
    block_param = att_param + ffn_param + rmsnorm_param
    total_transformer = block_param * num_blocks
    print(f"  Transformer Blocks:    {total_transformer:,} ({num_blocks} x {block_param:,})")

    # 3. Final Layers
    final_norm = embed_dim
    output_bias = vocab_size
    final_layers = final_norm + output_bias
    print(f"  Final Layers (Norm+Bias): {final_layers:,}")

    # Total
    total = pos_token_embed + total_transformer + final_layers
    print("-" * 45)
    print(f"  TOTAL PARAMETERS:      {total:,}")
    print(f"  APPROXIMATE SIZE:      {total / 1e6:.2f} M")
    return total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='default', help='Config name from model_param.yaml')
    args = parser.parse_args()

    yaml_path = "model_param.yaml"
    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found.")
        exit(1)

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    if args.config == 'default':
        model_name = config['default_model']
    else:
        model_name = args.config

    if model_name not in config:
        print(f"Error: Config '{model_name}' not found in {yaml_path}")
        exit(1)

    print(f"Using config: {model_name}")
    calc_params(config[model_name])
