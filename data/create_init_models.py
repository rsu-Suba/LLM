import tensorflow as tf
import yaml
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import build_model

def create_init_models():
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_param.yaml'))
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    model_sizes = ["coal", "iron", "gold", "diamond"]

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'model'))
    os.makedirs(models_dir, exist_ok=True)

    print("--- Initial Model Generation ---")
    for size in model_sizes:
        if size not in config:
            print(f"Skipping {size}: Not found in yaml.")
            continue
            
        params = config[size]
        print(f"\nBuilding {size.upper()} model...")
        
        try:
            model = build_model(
                vocab_size=params['VOCAB_SIZE'],
                max_len=params['MAX_LEN'],
                embed_dim=params['EMBED_DIM'],
                num_transformer_blocks=params['NUM_TRANSFORMER_BLOCKS'],
                num_heads=params['NUM_HEADS']
            )
            
            save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', params['MODEL_SAVE_PATH']))
            
            print(f"Saving initial weights to {save_path}...")
            model.save_weights(save_path)
            print(f"Successfully created {size} (Parameters: {model.count_params():,})")
            
            del model
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"Failed to create {size}: {e}")

    print("\n--- Done. All initial models are ready for mining! ---")

if __name__ == "__main__":
    tf.config.optimizer.set_jit(False)
    create_init_models()
