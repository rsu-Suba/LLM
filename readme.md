## LLM from scratch
#### Project to build a 1B LLM from scratch using CC100 & Japanese Wikipedia corpus.

### Environment
- Python 3.11.9
- TensorFlow 2.19.0
- Keras 3.10.0
- SentencePiece 0.2.0
- PyYAML 6.0.2

### Models
| Tier | Params |
| :--- | :--- |
| Gold | 96M |
| Diamond | 135M |
| Ruby | 220M |
| 1B | 1B |

### Usage
1. Make tokenizer from corpus
```bash
python data/tokenizer/trainTokenizer.py
```
2. Adjust model params
```yaml
# model_param.yaml
diamond_1B:
  VOCAB_SIZE: 28000
  MAX_LEN: 512
  EMBED_DIM: 2048
  FF_DIM: 5504
  NUM_TRANSFORMER_BLOCKS: 20
  NUM_HEADS: 16
  NUM_KV_HEADS: 4
  MODEL_SAVE_PATH: "model/diamond_1B.weights.h5"
  BATCH_SIZE: 1
  GRAD_ACCUM_STEPS: 128
  PEAK_LEARNING_RATE: 2.0e-4
```
3. Start training
```bash
python train/train.py --model diamond
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

## Generate sample
Hyper params:
| max_new_tokens | temperature | top_k | top_p | loop_penalty |
| :-- | :-- | :-- | :-- | :-- |
|40 | 0.75 | 45 | 0.7 | 1.1 |

#### Prompt : 「日本の四季にはそれぞれ特徴があり、」

**1B** (PyTorch / CC-100 10B):
> 日本の四季にはそれぞれ特徴があり、春夏秋冬と季節によって花柄や形も異なります。また、「秋の七草」は春の風物詩として親しまれています。 夏の果物は旬の果実を

#### Prompt : 「おいしいオムライスを作るために大事なことは、」

**1B** (PyTorch / CC-100 10B):
> おいしいオムライスを作るために大事なことは、まず自分が作りたい料理を自分好みに作るということです。また、その調理法やレシピが食卓にどのように役に立つのかを知ることです。「私はどんな味付けが好きですか?」と質問すると「私の好きなのはサラダでしょう」とおっしゃる方もいらっしゃいます。しかし、それはあくまで理想であって、「美味しいから食べる」というものではなく、「おいしく作れる」のです。

#### Prompt : 「人工知能が人間の知性を超える日は、」

**1B** (PyTorch / CC-100 10B):
> 人工知能が人間の知性を超える日は、人類の進歩と人間的成長は相反する。人間が自ら作り出した知的能力を生かして社会や経済を発展させようとする時、人間は他人を自分の力で動かせるようになるのである。そして人類に求められるのは「生きる」ことではなく、「自我の自己実現」、すなわち「創造的な生活」「人としての尊厳ある生活をすることである。」であり、「人生とは自分自身の選択によって成り
