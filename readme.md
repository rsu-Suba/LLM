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
| Coal | 23M |
| Iron | 48M |
| Gold | 109M |
| Diamond | 135M |
| Ruby | 154M |
| 1B | 943M |

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
python archs/transformer/train/train.py --model diamond
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
- See also: [`model.py`](./archs/transformer/model.py)
---
- Model parameters can be changed in `param.yaml`

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

---

## LatentBrain Architecture (`archs/latent/v1`)
### 設計

通常のTransformerはトークン列を直接処理するが、LatentBrainは以下の脳的な情報処理フローを模倣する：

```
入力トークン列
   ↓ Encoder（特徴抽出・局所パターン把握）
   ↓ Compressor（高圧縮：1/32 に要約 → 作業記憶）
   ↓ Reasoner（潜在空間上で反復推論）
   ↓ Decoder（Cross-Attention で出力トークンを生成）
出力トークン列
```

### モジュール構成

| モジュール | クラス | 役割 |
| :--- | :--- | :--- |
| **Encoder** | `LinearEncoderBlock` | Depthwise Conv + SwiGLU で局所特徴を抽出 |
| **Compressor** | `DynamicLatentCompressor` | Stride Conv + Cross-Attention で1/`compression_ratio`に圧縮（作業記憶の生成） |
| **Reasoner** | `AdaptiveReasoning` | 圧縮潜在を最大 `max_reason_steps` 回繰り返し更新 |
| └ 推論ブロック | `LatentReasoningBlock` | Mixer → FFN → コントローラー → LocalWindowReader |
| └ コントローラー | `EntropyFusedController` | エントロピーを入力に受け取り、**読む/スキップ** を動的に決定 |
| └ ポインター | `PointerTracker` | 現在の注目位置を追跡・移動（最大 ±32 トークン） |
| └ ウィンドウ | `WindowPredictor` | 参照するローカル窓幅を動的に予測（4〜128 トークン） |
| **Decoder** | `DecoderBlock` | Self-Attention + Cross-Attention（潜在参照）で出力生成 |

### 推論の仕組み (AdaptiveReasoning)

```
latent (圧縮済み作業記憶)
 ↓
[LatentReasoningBlock] × max_reason_steps (最大16回)
  ├── SequenceMixer    : Depthwise Conv + Gating で系列全体を混合
  ├── SwiGLU FFN       : 非線形変換
  ├── EntropyFusedController : エントロピー推定 → read/skip 確率を出力
  ├── PointerTracker   : 元のトークン列上の注目位置を更新
  ├── WindowPredictor  : 参照窓幅を動的決定
  └── LocalWindowReader: ポインター周辺のみを Cross-Attention で参照
 ↓
更新された latent
```

エントロピーが高い（不確かな）タイミングでは read が増加し、  
確信が高い場合は skip して計算を節約する。

### モデルサイズ

| Tier | Params | `EMBED_DIM` | Encode / Reason / Decode |
| :--- | :--- | :--- | :--- |
| coal | 50M | 384 | 2 / 8 / 4 |
| iron | 170M | 640 | 4 / 12 / 6 |
| gold | 270M | 896 | 4 / 10 / 4 |
| diamond | 580M | 1280 | 4 / 12 / 4 |

共通: `VOCAB_SIZE=28000`, `COMPRESSION_RATIO=32`, `MAX_LEN=1152 (PROMPT=1024 + TARGET=128)`

### ファイル構成

```
archs/latent/v1/
├── model.py                   # LatentBrainLLM 本体
├── param.yaml                 # モデルサイズ設定
├── modules/
│   ├── attention.py           # RMSNorm / SwiGLU / Attention (RoPE + GQA)
│   ├── encoder_decoder.py     # LinearEncoderBlock / DynamicLatentCompressor / DecoderBlock
│   ├── controller.py          # EntropyFusedController / PointerTracker / WindowPredictor
│   └── reasoner.py            # LatentReasoningBlock / AdaptiveReasoning
├── train/
└── review/
```
