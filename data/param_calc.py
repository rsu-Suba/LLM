VOCAB_SIZE = 50000
MAX_LEN = 1024
EMBED_DIM = 768
NUM_TRANSFORMER_BLOCKS = 16
NUM_HEADS = 12

def calc():
    PosEmbed = (VOCAB_SIZE + MAX_LEN) * EMBED_DIM
    print("Position Embedding: " + f"{PosEmbed:,}")

    key_dim = EMBED_DIM // NUM_HEADS
    total_dim = key_dim * NUM_HEADS
    qkv = 3 * ((EMBED_DIM + 1) * total_dim)
    out = (total_dim + 1) * EMBED_DIM
    att_param = qkv + out
    print("Multi-Head: " + f"{att_param:,}")

    FF_DIM = EMBED_DIM * 4
    ffn1 = (EMBED_DIM + 1) * FF_DIM
    ffn2 = (FF_DIM + 1) * EMBED_DIM
    ffn_param = ffn1 + ffn2
    print("Feed Forward: " + f"{ffn_param:,}")

    lnorm = 2 * EMBED_DIM * 2
    print("LayerNorm: " + f"{lnorm:,}")

    transformer_param = (att_param + ffn_param + lnorm) * NUM_TRANSFORMER_BLOCKS
    print("Transformer: " + f"{transformer_param:,}")

    dense_param = VOCAB_SIZE
    print("Dense: " + f"{dense_param:,}")

    param = PosEmbed + transformer_param + dense_param
    return f"{param:,}"

if __name__ == '__main__':
    print("Total model param: " + calc())