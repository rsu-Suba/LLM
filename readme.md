## LLM from scratch
#### Project to build a 135M LLM from scratch using Japanese Wikipedia corpus.

### Environment
- Python 3.11.9
- TensorFlow 2.19.0
- Keras 3.10.0
- SentencePiece 0.2.0
- PyYAML 6.0.2

### Models
| Tier | Params |
| :--- | :--- |
| Coal | 23M |
| Iron | 42M |
| Gold | 96M |
| Diamond | 135M |

### Usage
1. Make tokenizer from corpus
```bash
python data/tokenizer/trainTokenizer.py
```
2. Adjust model params
```yaml
# model_param.yaml
diamond:
  VOCAB_SIZE: 28000
  MAX_LEN: 256
  EMBED_DIM: 864
  NUM_TRANSFORMER_BLOCKS: 12
  NUM_HEADS: 12
  MODEL_SAVE_PATH: "data/model/diamond.weights.h5"
  BATCH_SIZE: 16
  PEAK_LEARNING_RATE: 0.0003
```
3. Start training
```bash
python train.py --model diamond
```

---
#### Model structure
- GPT-like Transformer (Decoder only)
- Causal Self-Attention (Masked)
- Pre-LN Architecture
- Token + Position embeddings
- Feed Forward layer with 4x hidden size
- Weight tying (shared input/output embeddings)
- Dropout + residual connections
- See also: [`model.py`](./model.py)
---
- Model parameters can be changed in `model_param.yaml`

## Dataset
- Tokenizer was trained on Japanese Wikipedia texts.
- Wikipedia texts are licensed under CC BY-SA 3.0 and GFDL.
- For more details, see [Wikipedia Terms of Use](https://foundation.wikimedia.org/wiki/Terms_of_Use).
