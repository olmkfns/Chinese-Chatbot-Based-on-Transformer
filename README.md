# 中文聊天机器人

基于 PyTorch 从零手写 Transformer / LLaMA / Qwen2 架构的中文对话机器人，参考论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 及 LLaMA / Qwen2 技术报告。

[📖 English Docs](README_ENG.md)

---

## 特色

- **四档模型** — 三档手写架构 + 一档 API 封装，命令行自由切换
- **手写 Qwen2** — 自写 RMSNorm + RoPE + GQA + SwiGLU，直接加载 HuggingFace Qwen2-1.5B 预训练权重，免训练即可对话
- **LLaMA-style GPT** — 手写 Decoder-Only 模型（RMSNorm / RoPE / SwiGLU / 无 bias）
- **Encoder-Decoder Transformer** — 完整手写 Pre-LN Transformer（MHA / FFN / 位置编码）
- **短期记忆** — GPT / Qwen 模式支持多轮对话上下文
- **权重共享** — Embedding 与输出投影共享权重
- **Noam 学习率调度** — 内置 Warmup 机制
- **标签平滑** — 内存高效的 Label Smoothing CE 实现
- **混合精度训练** — AMP 自动混合精度
- **KV-Cache 增量解码** — 推理时复用历史 K/V
- **重复惩罚 + N-gram 阻断** — 消除退化重复输出
- **多语料支持** — 单语料 / 多语料联合训练，自动分目录管理
- **命令行驱动** — `--model` / `--corpora` / `--epoch` / `--batch` / `--fenci` 覆盖配置
- **双分词引擎** — `jieba`（词级）/ `space`（空格切分）
- **双格式语料** — `.json`（LCCC 标准）/ `.conv`（传统格式）

---

## 项目结构

```
.
├── models/               # 模型模块
│   ├── __init__.py           # 统一导出
│   ├── model.py              # Encoder-Decoder Transformer 手写 (Lite)
│   ├── model_gpt.py          # LLaMA-style GPT 手写 (Middle)
│   ├── model_qwen2_hand.py   # Qwen2 手写 + GQA + HF 权重 (Pro)
│   └── model_qwen.py         # HuggingFace Qwen 封装 (API 对比 / 微调)
├── config.py             # 超参数 + 语料 / 模型选择
├── data_loader.py        # 数据预处理 / 词表 / DataLoader
├── train.py              # 训练脚本
├── inference.py          # 推理 & 交互式聊天
├── requirements.txt      # 依赖
│
├── data/                 # 语料目录
│   ├── xiaohuangji/
│   │   ├── xiaohuangji50w_fenciA.conv
│   │   ├── vocab_gpt.json
│   │   └── vocab_transformer.json
│   └── LCCC-base-split/
│       ├── LCCC-base_train.json
│       ├── LCCC-base_valid.json
│       └── LCCC-base_test.json
│
└── checkpoints/          # 模型检查点 (按语料分目录, 按模型类型分文件)
    └── <corpus>/
        ├── best_model_gpt.pt
        ├── best_model_transformer.pt
        ├── history_gpt.json
        └── history_transformer.json
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

- Python 3.10+ / PyTorch 2.0+（推荐 CUDA）

### 2. 推理（免训练）

```bash
# 手写 Qwen2-1.5B — 自动下载 HF 权重, 直接对话
python inference.py --model qwen2hand

# API Qwen — 用于对比测试
python inference.py --model qwen --pretrained
```

### 3. 训练

```bash
# GPT (LLaMA-style) — LCCC 大语料
python train.py --model gpt --corpora LCCC-base-split --fenci space --epoch 10 --batch 32

# GPT (LLaMA-style) — 小黄鸡小语料
python train.py --model gpt --corpora xiaohuangji --fenci jieba

# Transformer (Encoder-Decoder)
python train.py --model transformer --corpora xiaohuangji
```

---

## 模型对比

| 版本 | `--model` | 架构 | 参数量 | 训练 | 效果 |
|------|-----------|------|--------|------|------|
| Lite | `transformer` | Encoder-Decoder 手写 | ~15M | 需要 | 学习级 |
| Middle | `gpt` | LLaMA-style Decoder-Only 手写 | ~76M* | 需要 | 中等 |
| **Pro** | **`qwen2hand`** | **Qwen2 手写 + GQA + HF 权重** | **1.5B** | **免训练** | **强** |
| API | `qwen` | HuggingFace Qwen 封装 | 100M~1.5B | 可选微调 | 用于对比 |

> \*GPT 参数量与 `vocab_size` 联动，vocab_size=100000 时约 76M，50000 时约 51M。

### 架构清单

| 文件 | 架构 | 自写组件 |
|------|------|---------|
| `model.py` | Pre-LN Transformer | MHA / FFN / 正弦位置编码 |
| `model_gpt.py` | LLaMA-style | RMSNorm / RoPE / SwiGLU / KV-Cache |
| `model_qwen2_hand.py` | Qwen2 | RMSNorm / RoPE / **GQA** / SwiGLU / KV-Cache |
| `model_qwen.py` | API 封装 | 无（委托 HuggingFace） |

---

## 模型架构

### Qwen2 手写架构 (Pro)

```
Token IDs → Embedding
          → Qwen2Block × 28
             ├── RMSNorm → GQA Self-Attention + RoPE → Residual
             │     Q: 12 heads, KV: 2 heads (每组 KV 共享 6 个 Q)
             └── RMSNorm → SwiGLU FFN (gate/up/down) → Residual
          → RMSNorm → Linear LM Head
```

**GQA (Grouped Query Attention)**：K/V 只有 2 个头，Q 有 12 个头，每个 KV 头被 6 个 Q 头共享。推理时 KV-Cache 减小 6 倍。

### LLaMA-style GPT (Middle)

```
Token IDs → Embedding
          → DecoderLayer × N
             ├── RMSNorm → RoPE Multi-Head Self-Attention (无 bias) → Residual
             └── RMSNorm → SwiGLU FFN (gate/up combined) → Residual
          → RMSNorm → Linear LM Head (共享 Embedding 权重)
```

**特性**：RMSNorm / RoPE (θ=10000) / SwiGLU / 无 bias / Speaker Token / 全序列 Loss

### Transformer Encoder-Decoder (Lite)

```
输入 → Token Embedding + Positional Encoding
     → Encoder × N: Pre-LN → MHA → FFN → Residual
     → Decoder × N: Pre-LN → Masked MHA → Cross MHA → FFN → Residual
     → Linear → Softmax → 输出
```

---

## 推理命令

```bash
# 手写 Qwen2 (推荐)
python inference.py --model qwen2hand

# 自己训练的 GPT
python inference.py --model gpt --corpora LCCC-base-split --fenci space

# 自己训练的 Transformer
python inference.py --model transformer

# API Qwen (对比测试)
python inference.py --model qwen --pretrained
```

### 聊天命令

| 命令 | 说明 |
|------|------|
| `/clear` | 清空对话记忆 |
| `/history` | 查看当前记忆 |
| `/beam` | 切换 Beam Search |
| `/sample` | 切换采样解码 |
| `quit` / `exit` | 退出 |

---

## 配置

### 语料与模型

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `corpora` | `("xiaohuangji",)` | 语料名（`data/` 下文件夹） |
| `model_type` | `"gpt"` | `transformer` \| `gpt` \| `qwen2hand` \| `qwen` |
| `fenci_mode` | `"jieba"` | `jieba` (词级) \| `space` (空格) |
| `vocab_size` | 50000 | 词表上限 |
| `min_freq` | 3 | 最低词频阈值 |

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--model` | `transformer` \| `gpt` \| `qwen2hand` \| `qwen` |
| `--corpora` | 语料名, 多语料逗号分隔 |
| `--fenci` | `jieba` \| `space` |
| `--epoch` | 训练轮数 |
| `--batch` | 批次大小 |
| `--device` | `cuda` / `cpu` |
| `--resume` | 从 checkpoint 恢复训练 |

### 架构参数

| 参数 | GPT 默认 | Qwen2Hand | 说明 |
|------|---------|-----------|------|
| `d_model` | 512 | 1536 | 隐层维度 |
| `n_heads` | 8 | 12 | Q 头数 |
| `n_kv_heads` | — | 2 | KV 头数 (GQA) |
| `n_layers` | 6 | 28 | 层数 |
| `d_ff` | 2048 | 8960 | FFN 隐层维度 |
| `max_len` | 120 | 32768 | 训练截断长度 |
| `dropout` | 0.1 | 0.0 | Dropout 比例 |

---

## 数据集

### 内置语料

| 语料 | 标识符 | 规模 | 格式 | 分词 |
|------|--------|------|------|------|
| 小黄鸡 | `xiaohuangji` | ~50 万对 | `.conv` | `jieba` |
| LCCC-base | `LCCC-base-split` | ~890 万对 | `.json` | `space` |

### 格式规范

**.conv** — 传统格式：
```
M 你在干嘛
M 在跟你聊天呀
E
```

**.json** — LCCC 标准：
```json
[["你 好 呀", "你 好 你 好"], ["吃 了 吗", "还 没 呢"]]
```

消息以空格预分词，相邻配对。两种格式可混用，程序按后缀自动识别。

### 添加语料

```bash
mkdir -p data/mycorpus
cp data.json data/mycorpus/
python train.py --corpora mycorpus --fenci space
```

---

## 依赖

- **Python** ≥ 3.10
- **PyTorch** ≥ 2.0.0
- **NumPy** / **tqdm**
- **jieba**（`--fenci jieba` 时）
- **transformers** + **safetensors**（`qwen2hand` / `qwen` 时）
- **huggingface_hub**（下载权重时）

---

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) — Yang et al., 2024
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023

---

## 语料下载

| 语料 | 链接 |
|------|------|
| 小黄鸡 | [Dialog_Corpus](https://github.com/candlewill/Dialog_Corpus) |
| LCCC-base | [CDial-GPT](https://github.com/thu-coai/CDial-GPT) |

## License

MIT License
