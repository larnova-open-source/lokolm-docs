# lokoLM

A minimal, from-scratch **decoder-only Transformer** language model in PyTorch.

lokoLM is a compact GPT-style autoregressive language model. It stacks decoder blocks —
each a causal self-attention sublayer followed by a position-wise feed-forward network,
both with pre-norm residual connections — on top of token and positional embeddings,
ending in a weight-tied language-modeling head.

## Installation

Install PyTorch. On a GPU machine, pick the wheel matching your CUDA version from
[pytorch.org](https://pytorch.org/get-started/locally/). For CUDA 12.1:

```powershell
# GPU (CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch
```

> **Common gotcha:** a plain `pip install torch` on a GPU box can pull a CPU-only build.
> Always verify with `python -c "import torch; print(torch.cuda.is_available())"` — it must
> print `True` to train on the GPU.

## Quickstart

```powershell
cd model

# 1. Sanity-check the model wires together
python -m lokolm.model

# 2. Train (auto-detects CUDA, falls back to CPU)
python train.py
```

Drop a text file named `input.txt` beside the scripts to train on your own corpus;
otherwise a tiny built-in sample is used so it runs out of the box.

## Architecture

lokoLM follows the GPT-2 design: pre-normalization, GELU activations, and a
language-modeling head tied to the token embedding.

```
         input token ids  (B, T)
                 |
     +-----------+-----------+
   token emb (wte)    positional emb (wpe)
     +-----------+-----------+
                 |  add
                 v
        +-------------------+   x n_layers
        |  Transformer Block|
        |  +--------------+ |
        |  | LayerNorm    | |
        |  | Causal Attn  | |--+ residual
        |  +--------------+ |  |
        |       + <---------+--+
        |  +--------------+ |
        |  | LayerNorm    | |
        |  | MLP (FFN)    | |--+ residual
        |  +--------------+ |  |
        |       + <---------+--+
        +-------------------+
                 |
            LayerNorm (ln_f)
                 |
          lm_head  (tied to wte)
                 |
                 v
           logits  (B, T, vocab)
```

### Causal self-attention

Multi-head attention with a lower-triangular mask so each position attends only to itself
and earlier positions. Q, K, V are computed in a single fused linear projection (`c_attn`).

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        # reshape to (B, n_heads, T, head_dim), scaled dot-product
        att = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1)**0.5)
        att = att.masked_fill(mask == 0, float('-inf'))  # causal
        y = F.softmax(att, dim=-1) @ v
        return self.c_proj(y)
```

### Feed-forward MLP

The position-wise feed-forward network applied after attention in each block. It expands
the hidden dimension by `mlp_ratio` (4x by default), applies a GELU nonlinearity, then
projects back down.

```python
class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio=4):
        super().__init__()
        self.c_fc   = nn.Linear(d_model, mlp_ratio * d_model)
        self.c_proj = nn.Linear(mlp_ratio * d_model, d_model)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))
```

### Transformer block

Pre-norm residual structure: LayerNorm is applied *before* each sublayer, and the sublayer
output is added back to its input. This is what keeps deep stacks stable during training.

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))   # attention sublayer
    x = x + self.mlp(self.ln_2(x))    # feed-forward sublayer
    return x
```

### LokoLM model

The top-level module ties everything together and exposes `forward` (returns logits and
optional cross-entropy loss) and `generate` (autoregressive sampling with temperature and
optional top-k).

```python
from lokolm import LokoLM

model = LokoLM(vocab_size=256, block_size=128,
               d_model=384, n_heads=6, n_layers=6)
logits, loss = model(idx, targets)          # training forward
out = model.generate(idx, max_new_tokens=100, top_k=50)
```

## Default hyperparameters

| Name | Default | Meaning |
|------|---------|---------|
| `vocab_size` | 256 | Token vocabulary (byte-level demo) |
| `block_size` | 128 | Maximum context length |
| `d_model` | 384 | Embedding / hidden dimension |
| `n_heads` | 6 | Attention heads |
| `n_layers` | 6 | Stacked decoder blocks |
| `mlp_ratio` | 4 | FFN expansion factor |
| `learning_rate` | 3e-4 | Peak LR (warmup -> cosine decay) |

## Project layout

The repository separates the model code from the documentation site:

```
lokoLM/
├── model/                  # Python model + training code
│   ├── lokolm/             # the importable package
│   │   ├── causal_self_attention.py   # multi-head causal self-attention
│   │   ├── mlp.py                      # position-wise feed-forward network
│   │   ├── model.py                    # Block + LokoLM model
│   │   └── __init__.py                 # exports (from lokolm import LokoLM)
│   └── train.py            # training loop with CUDA support
└── docs/                   # documentation site (this content)
    ├── content/            # Markdown: overview / training / roadmap
    └── index.html          # live Markdown viewer
```

See **[training.md](training.md)** for the full GPU/CUDA training guide.

## Roadmap

lokoLM v1 is intentionally a minimal, dense, GPT-2-style baseline for teaching and research.
The planned v2 path is documented in **[roadmap.md](roadmap.md)**, with a detailed
explanation of what each step is and how it improves on the current design:

- **Architecture:** RMSNorm, SwiGLU, RoPE, GQA, and MoE — modernize the core model.
- **Tokenization, fine-tuning & agents:** a **BPE tokenizer**, **Hugging Face + LoRA**
  fine-tuning, and **function calling** — make the base model efficient, cheap to adapt, and
  usable as a foundation for AI agents.
