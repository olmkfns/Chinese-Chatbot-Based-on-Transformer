# Chinese Chatbot

A Chinese dialogue system with hand-written Transformer / LLaMA / Qwen2 architectures, built from scratch in PyTorch.

[📖 中文文档](README.md)

---

## Features

- **Four model tiers** — three hand-written architectures + one API wrapper
- **Hand-written Qwen2** — self-implemented RMSNorm + RoPE + GQA + SwiGLU, directly loads HuggingFace Qwen2-1.5B pretrained weights, zero training required
- **LLaMA-style GPT** — hand-written Decoder-Only (RMSNorm / RoPE / SwiGLU / no bias)
- **Encoder-Decoder Transformer** — complete Pre-LN Transformer from scratch
- **Short-term memory** — multi-turn conversation history in GPT / Qwen modes
- **Weight tying** — shared Embedding and output projection weights
- **Noam scheduler** — learning rate warmup from the original paper
- **Label smoothing** — memory-efficient CE implementation
- **Mixed precision** — AMP automatic mixed precision training
- **KV-Cache** — incremental decoding, 5-10× speedup
- **Repetition penalty + N-gram blocking** — eliminates degenerate outputs
- **Multi-corpus** — single or joint corpus training with automatic organization
- **CLI-driven** — `--model` / `--corpora` / `--epoch` / `--batch` / `--fenci`
- **Dual tokenization** — `jieba` (word-level) / `space` (whitespace split)
- **Dual format** — `.json` (LCCC standard) / `.conv` (legacy)

---

## Project Structure

```
.
├── models/               # Model modules
│   ├── __init__.py           # Unified exports
│   ├── model.py              # Encoder-Decoder Transformer (Lite)
│   ├── model_gpt.py          # LLaMA-style GPT (Middle)
│   ├── model_qwen2_hand.py   # Hand-written Qwen2 + GQA + HF weights (Pro)
│   └── model_qwen.py         # HuggingFace Qwen wrapper (API comparison / fine-tune)
├── config.py             # Hyperparameters + corpus / model selection
├── data_loader.py        # Data preprocessing / vocabulary / DataLoader
├── train.py              # Training script
├── inference.py          # Inference & interactive chat
├── requirements.txt      # Dependencies
│
├── data/                 # Corpus directory
│   ├── xiaohuangji/
│   └── LCCC-base-split/
│
└── checkpoints/          # Model checkpoints (by corpus, by model type)
    └── <corpus>/
        ├── best_model_gpt.pt
        ├── best_model_transformer.pt
        └── history_gpt.json
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Inference (no training needed)

```bash
# Hand-written Qwen2-1.5B (auto-downloads weights)
python inference.py --model qwen2hand

# API Qwen (for comparison)
python inference.py --model qwen --pretrained
```

### 3. Training

```bash
# GPT (LLaMA-style) — LCCC corpus
python train.py --model gpt --corpora LCCC-base-split --fenci space --epoch 10 --batch 32

# GPT — xiaohuangji corpus
python train.py --model gpt --corpora xiaohuangji --fenci jieba

# Transformer
python train.py --model transformer --corpora xiaohuangji
```

---

## Model Comparison

| Tier | `--model` | Architecture | Params | Training | Quality |
|------|-----------|-------------|--------|----------|---------|
| Lite | `transformer` | Encoder-Decoder (hand-written) | ~15M | Required | Basic |
| Middle | `gpt` | LLaMA-style Decoder-Only (hand-written) | ~76M | Required | Moderate |
| **Pro** | **`qwen2hand`** | **Qwen2 hand-written + GQA + HF weights** | **1.5B** | **None** | **Best** |
| API | `qwen` | HuggingFace Qwen wrapper | 100M-1.5B | Optional | For comparison |

### Architecture Files

| File | Architecture | Custom Components |
|------|-------------|-------------------|
| `model.py` | Pre-LN Transformer | MHA / FFN / Sinusoidal PE |
| `model_gpt.py` | LLaMA-style | RMSNorm / RoPE / SwiGLU / KV-Cache |
| `model_qwen2_hand.py` | Qwen2 | RMSNorm / RoPE / **GQA** / SwiGLU / KV-Cache |
| `model_qwen.py` | API wrapper | None (delegates to HuggingFace) |

---

## Model Architecture

### Qwen2 (Pro) — Hand-written

```
Token IDs → Embedding
          → Qwen2Block × 28
             ├── RMSNorm → GQA Self-Attention + RoPE → Residual
             │     Q: 12 heads, KV: 2 heads (6 Q per KV group)
             └── RMSNorm → SwiGLU FFN (gate/up/down) → Residual
          → RMSNorm → Linear LM Head
```

**GQA**: KV heads = 2, Q heads = 12. Each KV head is shared by 6 Q heads. KV-Cache 6× smaller during inference.

### LLaMA-style GPT (Middle)

```
Token IDs → Embedding
          → DecoderLayer × N
             ├── RMSNorm → RoPE MHA (no bias) → Residual
             └── RMSNorm → SwiGLU FFN (gate+up combined) → Residual
          → RMSNorm → Linear LM Head (tied weights)
```

### Transformer Encoder-Decoder (Lite)

```
Input → Embedding + Positional Encoding
      → Encoder × N: Pre-LN → MHA → FFN → Residual
      → Decoder × N: Pre-LN → Masked MHA → Cross MHA → FFN → Residual
      → Linear → Output
```

---

## Inference Commands

```bash
# Hand-written Qwen2 (recommended)
python inference.py --model qwen2hand

# Your trained GPT
python inference.py --model gpt --corpora LCCC-base-split --fenci space

# Your trained Transformer
python inference.py --model transformer

# API Qwen (comparison)
python inference.py --model qwen --pretrained
```

### Chat Commands

| Command | Description |
|---------|-------------|
| `/clear` | Clear conversation memory |
| `/history` | View current memory |
| `/beam` | Switch to Beam Search |
| `/sample` | Switch to sampling |
| `quit` / `exit` | Exit |

---

## Configuration

### Corpus & Model

| Parameter | Default | Description |
|-----------|---------|-------------|
| `corpora` | `("xiaohuangji",)` | Corpus folder under `data/` |
| `model_type` | `"gpt"` | `transformer` \| `gpt` \| `qwen2hand` \| `qwen` |
| `fenci_mode` | `"jieba"` | `jieba` (word) \| `space` (whitespace) |
| `vocab_size` | 50000 | Vocabulary size cap |
| `min_freq` | 3 | Minimum token frequency |

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--model` | `transformer` \| `gpt` \| `qwen2hand` \| `qwen` |
| `--corpora` | Corpus names, comma-separated |
| `--fenci` | `jieba` \| `space` |
| `--epoch` | Training epochs |
| `--batch` | Batch size |
| `--device` | `cuda` / `cpu` |
| `--resume` | Resume from checkpoint |

### Architecture Parameters

| Parameter | GPT Default | Qwen2Hand | Description |
|-----------|------------|-----------|-------------|
| `d_model` | 512 | 1536 | Hidden dimension |
| `n_heads` | 8 | 12 | Query attention heads |
| `n_kv_heads` | — | 2 | KV heads (GQA) |
| `n_layers` | 6 | 28 | Number of layers |
| `d_ff` | 2048 | 8960 | FFN hidden dimension |
| `max_len` | 120 | 32768 | Max training sequence length |

---

## Datasets

### Built-in Corpora

| Corpus | ID | Size | Format | Tokenization |
|--------|-----|------|--------|-------------|
| XiaoHuangJi | `xiaohuangji` | ~500K pairs | `.conv` | `jieba` |
| LCCC-base | `LCCC-base-split` | ~8.9M pairs | `.json` | `space` |

### Format Specification

**.conv**:
```
M message 1
M message 2
E
```

**.json** (LCCC standard):
```json
[["hello", "hi there"], ["how are you", "fine thanks"]]
```

Messages are space-pretokenized. Adjacent messages form QA pairs. Formats auto-detected by extension.

---

## Dependencies

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0.0
- **NumPy** / **tqdm**
- **jieba** (for `--fenci jieba`)
- **transformers** + **safetensors** (for `qwen2hand` / `qwen`)
- **huggingface_hub** (for weight downloads)

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) — Yang et al., 2024
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023

---

## License

MIT License
