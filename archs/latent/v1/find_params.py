from model import LatentBrainConfig, LatentBrainLLM

def get_params(dim, ff_dim, enc, reason, dec):
    cfg = LatentBrainConfig(
        vocab_size=28000, dim=dim, ff_dim=ff_dim, num_heads=dim//64, num_kv_heads=dim//256 if dim >= 256 else 1,
        encode_layers=enc, reason_layers=reason, decode_layers=dec, compression_ratio=32
    )
    model = LatentBrainLLM(cfg)
    return sum(p.numel() for p in model.parameters())

targets = [50_000_000, 150_000_000, 250_000_000, 500_000_000]

for target in targets:
    best_diff = float('inf')
    best_cfg = None
    best_count = 0
    for dim in [384, 512, 640, 768, 896, 1024, 1152, 1280]:
        ff_dim = dim * 3
        if ff_dim % 128 != 0:
            ff_dim = (ff_dim // 128) * 128
        for enc in [2, 4]:
            for reason in [6, 8, 10, 12, 16]:
                for dec in [2, 4, 6]:
                    count = get_params(dim, ff_dim, enc, reason, dec)
                    diff = abs(count - target)
                    if diff < best_diff:
                        best_diff = diff
                        best_cfg = (dim, ff_dim, enc, reason, dec)
                        best_count = count
    print(f"Target: {target/1e6}M -> Best: {best_count/1e6:.2f}M with cfg: dim={best_cfg[0]}, ff_dim={best_cfg[1]}, enc={best_cfg[2]}, reason={best_cfg[3]}, dec={best_cfg[4]}")
