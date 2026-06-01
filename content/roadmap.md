# Architecture Roadmap — Next Steps

lokoLM v1 is deliberately a **minimal, dense, GPT-2-style** decoder-only Transformer:
learned absolute positions, `LayerNorm`, a `GELU` feed-forward network, and full
multi-head attention. That makes it ideal for learning the fundamentals.

This page documents the upgrades that turn that baseline into a **modern**
Llama/Gemini-*family*-style model, and then into a **usable, fine-tunable, agent-capable**
model. Each item is presented as: **what it is**, **why it improves the current design**,
and **what changes** in this codebase. None of these are implemented yet — they are the
planned v2 path.

The roadmap has two tracks.

**Track A — Core architecture** (each is self-contained; the early ones compound best):

1. [RMSNorm](#1-rmsnorm) — simpler, faster normalization
2. [SwiGLU](#2-swiglu) — a better feed-forward network
3. [RoPE](#3-rope-rotary-positional-embeddings) — relative positions, length generalization
4. [GQA](#4-gqa-grouped-query-attention) — cheaper attention at inference
5. [MoE](#5-moe-mixture-of-experts) — more capacity at fixed compute

**Track B — Tokenization, fine-tuning & agents** (what makes the base model *usable* and
*adaptable* by others):

6. [BPE tokenizer](#6-bpe-tokenizer) — efficient subword vocabulary
7. [Hugging Face + LoRA fine-tuning](#7-hugging-face--lora-fine-tuning) — cheap adaptation of the base model
8. [Function calling](#8-function-calling--tool-use) — turn lokoLM into an agent

---

## 1. RMSNorm

**What it is.** Root Mean Square Layer Normalization. Where `LayerNorm` re-centers a
vector to zero mean *and* rescales it to unit variance (then applies a learned gain and
bias), RMSNorm drops the mean-centering and the bias entirely. It just divides by the
root-mean-square of the activations and applies a learned per-channel gain:

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps) * g
```

**Current design.** lokoLM uses `nn.LayerNorm` in each block (`ln_1`, `ln_2`) and at the
final norm (`ln_f`) in `model/lokolm/model.py`.

**Why it improves things.**
- **Fewer operations.** No mean subtraction and no bias term — fewer reductions and fewer
  parameters. In practice it's a small but free speedup, applied at every layer.
- **Equal or better quality.** Empirically, models train just as well (often slightly
  better) without the mean-centering step; the re-centering turns out to be the part of
  LayerNorm that matters least.
- **Standard in modern LLMs.** Llama, Mistral, and the Gemini family all use RMSNorm.

**What changes in this codebase.** Add an `RMSNorm` module and replace the three
`nn.LayerNorm` uses in `model.py`. Conceptually:

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.gain
```

**Cost / trade-off.** Essentially none. This is the lowest-risk upgrade — pure win.

---

## 2. SwiGLU

**What it is.** A *gated* feed-forward network. The current MLP is
`Linear → GELU → Linear`. SwiGLU replaces the single activation with a **gate**: it uses
two parallel input projections, passes one through a Swish/SiLU activation, and multiplies
them elementwise before the output projection:

```
SwiGLU(x) = (Swish(x W_gate) ⊙ (x W_up)) W_down        # ⊙ = elementwise product
Swish(z)  = z * sigmoid(z)
```

**Current design.** lokoLM's `model/lokolm/mlp.py` is `c_fc → GELU → c_proj` with a 4× hidden
expansion.

**Why it improves things.**
- **Better quality per parameter.** The multiplicative gate lets the network *modulate*
  one projection with another, a more expressive transform than a fixed pointwise
  nonlinearity. Gated FFNs consistently outperform plain GELU/ReLU MLPs at equal scale.
- **Compute-matched by design.** Because SwiGLU adds a third weight matrix, implementations
  shrink the hidden dimension to about **⅔** of the usual `4 × d_model` so the total
  parameter and FLOP count stays roughly the same — you get the quality gain for free.

**What changes in this codebase.** Rewrite `MLP` with three projections (`w_gate`, `w_up`,
`w_down`) and a SiLU gate, sizing the hidden dim to ≈ `8/3 × d_model`:

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        hidden = int(2/3 * mult * d_model)        # keep params ~constant
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up   = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))
```

**Cost / trade-off.** Slightly more code and one extra matmul; offset by the smaller hidden
dimension. Widely considered worth it.

---

## 3. RoPE (Rotary Positional Embeddings)

**What it is.** A way to inject position *inside* the attention operation instead of adding
a position vector to the token embeddings. RoPE **rotates** each query and key vector by an
angle proportional to its position in the sequence. Because the dot product of two rotated
vectors depends only on the *difference* of their angles, attention scores end up encoding
**relative** position automatically.

**Current design.** lokoLM uses a learned **absolute** position embedding table (`wpe`) in
`model.py`, added to the token embeddings. It has one row per position up to `block_size`.

**Why it improves things.**
- **Relative position for free.** Attention naturally "knows" how far apart two tokens are,
  which is what actually matters for language — not their absolute index.
- **Length generalization.** The learned table has a hard ceiling at `block_size` (position
  129 simply has no embedding). RoPE is a function of position, so it extends to longer
  sequences far more gracefully — the foundation that lets modern models reach very long
  contexts.
- **No extra parameters.** RoPE is computed, not learned, so the `wpe` table disappears.

**What changes in this codebase.** Remove the `wpe` embedding in `model.py`, precompute the
rotation frequencies, and apply the rotation to `q` and `k` inside
`causal_self_attention.py` *after* the head reshape and *before*
`scaled_dot_product_attention`:

```python
# q, k: (B, n_heads, T, head_dim)
q = apply_rotary(q, cos, sin)
k = apply_rotary(k, cos, sin)
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**Cost / trade-off.** A bit of index/trig bookkeeping for the rotation tables, but no
runtime parameters and a large payoff in context handling.

---

## 4. GQA (Grouped-Query Attention)

**What it is.** A middle ground between full multi-head attention (MHA) and multi-query
attention (MQA). In MHA every query head has its **own** key and value head. In MQA *all*
query heads share **one** K/V head. GQA splits the query heads into a few **groups**, and
each group shares one K/V head — e.g. 8 query heads but only 2 K/V heads.

**Current design.** lokoLM uses full MHA: `c_attn` produces `n_heads` each of Q, K, and V
(`model/lokolm/causal_self_attention.py`).

**Why it improves things.**
- **Smaller KV cache → faster inference.** During autoregressive generation, the keys and
  values of all past tokens are cached in memory. That cache is the main memory bottleneck
  and bandwidth cost of decoding. Fewer K/V heads means a proportionally smaller cache and
  faster token generation — the primary reason production models (Llama 2/3, Gemini) use GQA.
- **Negligible quality loss.** Sharing K/V across a group costs very little accuracy while
  cutting K/V memory by the group factor.

**What changes in this codebase.** Give `c_attn` separate output sizes for Q
(`n_heads × head_dim`) versus K/V (`n_kv_heads × head_dim`), then repeat each K/V head to
match its query group before attention. `scaled_dot_product_attention` already supports
broadcasting K/V across query heads via `enable_gqa=True` on recent PyTorch.

**Cost / trade-off.** Mostly an *inference* win; at small scale (and at lokoLM's
`block_size=128`) the benefit is modest, which is exactly why it's deferred past v1.

---

## 5. MoE (Mixture of Experts)

**What it is.** Replace the single feed-forward network in each block with **many** parallel
FFNs ("experts") plus a small **router** (gating network). For each token, the router picks
the top-`k` experts (often `k=2`) and sends the token only to those. The block holds the
combined parameters of *all* experts, but each token is processed by only a couple of them —
this is **sparse, conditional computation**.

**Current design.** lokoLM has one dense MLP per block; every token flows through the same
FFN weights (`model/lokolm/mlp.py`).

**Why it improves things.**
- **More capacity at (almost) fixed compute.** Total parameters scale with the number of
  experts, but per-token FLOPs scale only with `k`. You can grow model *knowledge/capacity*
  dramatically without proportionally growing the cost of each forward pass.
- **Specialization.** Different experts learn to handle different kinds of tokens/patterns,
  which the router learns to dispatch.
- **It's the headline scaling lever of frontier models.** Gemini 1.5+ and Mixtral are
  sparse MoE; it's how parameter counts reach the hundreds of billions while keeping
  inference affordable.

**What changes in this codebase.** Swap the block's MLP for a `MoE` module: a `Linear`
router producing per-expert logits, a top-`k` selection, `n_experts` copies of the SwiGLU
FFN, and a weighted combination of the chosen experts' outputs. Training also needs a
**load-balancing loss** so the router doesn't collapse onto a few experts.

```python
class MoE(nn.Module):
    def __init__(self, d_model, n_experts=8, k=2):
        super().__init__()
        self.k = k
        self.router  = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d_model) for _ in range(n_experts))
    # forward: route each token to its top-k experts, combine weighted outputs
    # + auxiliary load-balancing loss during training
```

**Cost / trade-off.** By far the most complex upgrade: routing, top-`k` dispatch,
load-balancing, and trickier distributed training. High payoff at scale, but it adds real
implementation and training complexity — hence last on the roadmap and well beyond the v1
teaching goal.

---

## 6. BPE Tokenizer

**What it is.** Byte Pair Encoding — a *subword* tokenizer. It starts from raw bytes and
iteratively merges the most frequent adjacent pair into a new token, repeating until it
reaches a target vocabulary size. The result is a vocabulary of **subword units**: common
words become single tokens, rare words break into a few pieces, and any string is still
representable (no out-of-vocabulary problem, because it bottoms out at bytes).

**Current design.** lokoLM is **byte-level**: `vocab_size = 256`, so every byte is its own
token. With `block_size = 128` the model sees only ~128 bytes (~20–30 words) of context, and
sequences are as long as the raw text.

**Why it improves things.**
- **More text per token → effectively longer context.** A BPE token averages several
  characters, so the same `block_size` covers several times more actual text. The model
  "sees" more for the same sequence length.
- **Shorter sequences → cheaper training and inference.** Attention cost grows with sequence
  length; subword tokens cut the number of steps to cover a document.
- **Better learning of meaning.** The model operates on word/subword units instead of having
  to reassemble meaning byte-by-byte, which improves sample efficiency.
- **Ecosystem alignment.** A standard BPE vocabulary (ideally saved in Hugging Face
  `tokenizers` format) is the bridge to the rest of the tooling below — datasets, the
  `Trainer`, and adapters all assume a real tokenizer.

**What changes in this codebase.** Add `model/lokolm/tokenizer.py` with `train`, `encode`, and
`decode`, then encode the corpus in `train.py` and set `vocab_size` to the trained size.
You can implement BPE from scratch (great for teaching) and/or wrap the Hugging Face
`tokenizers` library for a production-grade, serializable tokenizer:

```python
# from scratch (teaching) — or load/save a Hugging Face tokenizer for compatibility
tok = BPETokenizer.train("input.txt", vocab_size=8000)
ids = tok.encode("hello world")
text = tok.decode(ids)
```

**Cost / trade-off.** Adds a preprocessing/training step for the tokenizer and a vocabulary
file to ship with the model. Low complexity, and it's the prerequisite for everything in
Track B.

---

## 7. Hugging Face + LoRA Fine-Tuning

**What it is.** Two complementary pieces:

- **Hugging Face compatibility** — wrapping lokoLM so it presents the standard interfaces
  (`PreTrainedModel` + a config, plus a saved tokenizer). Once it speaks that protocol, the
  whole ecosystem opens up: the `datasets` library for data, the `Trainer` for training
  loops, the Hub for sharing weights, and `transformers` utilities for generation.
- **LoRA (Low-Rank Adaptation)** — a *parameter-efficient fine-tuning* (PEFT) technique. The
  base model's weights are **frozen**, and small trainable low-rank matrices `A`/`B` are
  injected into the linear layers (typically the attention `c_attn`/`c_proj` projections).
  Only those tiny matrices are trained: `W_effective = W_frozen + B·A`, where `A` and `B`
  have a small rank `r` (e.g. 8 or 16).

**Current design.** lokoLM trains with a hand-written loop in `train.py` and saves a raw
`state_dict`. Adapting it to a new task today would mean full fine-tuning of all weights.

**Why it improves things.**
- **Cheap adaptation.** LoRA trains a fraction of a percent of the parameters, so
  fine-tuning fits on a single modest GPU and runs quickly — full fine-tuning of even a
  small model is far heavier.
- **Many tasks from one base.** Each LoRA adapter is a few MB. You can train and swap
  multiple adapters (domain A, domain B, a chat persona…) on top of the *same* frozen
  lokoLM, instead of storing a full copy per task.
- **Reuse the ecosystem instead of rebuilding it.** Being Hugging Face–compatible means
  researchers can fine-tune lokoLM with tools they already know (`peft`, `trl`, `Trainer`),
  pull datasets from the Hub, and publish results — a major win for a *teaching/research*
  model whose value is how easily others can build on it.

**What changes in this codebase.** Add a thin Hugging Face wrapper (a `LokoLMConfig` and a
`PreTrainedModel` subclass that delegates to the existing `LokoLM` module), save the BPE
tokenizer in HF format, then fine-tune with `peft`:

```python
from peft import LoraConfig, get_peft_model

base  = LokoLMForCausalLM.from_pretrained("lokolm-base")   # frozen weights
lcfg  = LoraConfig(r=16, target_modules=["c_attn", "c_proj"])
model = get_peft_model(base, lcfg)                          # only LoRA params train
# then train with the standard HF Trainer on your dataset
```

**Cost / trade-off.** Requires the BPE tokenizer (#6) and a wrapper to match the HF API.
Once that exists, fine-tuning becomes dramatically cheaper and more accessible — arguably the
highest-leverage item for adoption.

---

## 8. Function Calling & Tool Use

**What it is.** Teaching the model to produce **structured calls to external tools** — e.g.
emitting JSON like `{"name": "get_weather", "arguments": {"city": "Paris"}}` — and then to
consume the tool's result and continue. This is done by defining a **chat template** with
special tokens for roles (system / user / assistant / tool) and **fine-tuning on
function-calling datasets** so the model learns when and how to call tools.

**Current design.** lokoLM is a pure next-token predictor over plain text; it has no notion
of conversation roles, tools, or structured output.

**Why it improves things.**
- **Turns a text predictor into an agent.** A model that can reliably call functions can use
  calculators, search, databases, code execution, or any API — the core capability behind AI
  agents. The language model becomes the *reasoning/orchestration* layer, with tools doing
  the work it can't (or shouldn't) do internally.
- **Grounding and reliability.** Instead of hallucinating a fact or a calculation, the model
  fetches it — output becomes verifiable and up to date.
- **Composability.** Once lokoLM can be fine-tuned (#7) on tool-use formats, anyone can give
  it a new toolset for a new agent, on top of the same base model.

**What changes in this codebase.** Define a chat/tool template and the matching special
tokens in the tokenizer (#6), fine-tune (ideally via LoRA, #7) on a function-calling dataset,
and add an inference loop that parses the model's tool call, executes it, feeds the result
back, and lets the model continue:

```
user → model emits a tool call → run the tool → feed result back → model answers
```

**Cost / trade-off.** Depends on #6 and #7, plus instruction/tool-use fine-tuning data and a
small runtime to execute calls. It's a *capability* layer rather than an architecture change,
but it's what makes lokoLM genuinely useful for building agents.

---

## Summary

**Track A — architecture**

| Upgrade | Replaces | Main benefit | Adds params? | Complexity |
|---------|----------|--------------|--------------|------------|
| RMSNorm | LayerNorm | Faster, simpler norm | Fewer | Very low |
| SwiGLU | GELU MLP | Better quality per param | ~Same (resized) | Low |
| RoPE | Learned `wpe` | Relative pos + long context | Fewer | Medium |
| GQA | Full MHA | Smaller KV cache, faster decode | Fewer | Medium |
| MoE | Dense FFN | Huge capacity at fixed compute | Many more | High |

**Track B — tokenization, fine-tuning & agents**

| Upgrade | Builds on | Main benefit | Complexity |
|---------|-----------|--------------|------------|
| BPE tokenizer | — | Efficient subword vocab, more text per token | Low |
| HF + LoRA | BPE tokenizer | Cheap, accessible fine-tuning via the HF ecosystem | Medium |
| Function calling | BPE + LoRA | Tool use → build AI agents on lokoLM | Medium |

Together, **RMSNorm + SwiGLU + RoPE** are the highest-value, well-contained architecture
leap: they convert lokoLM's GPT-2-style block into a modern Llama/Gemini-family block while
keeping the codebase readable. **GQA** and **MoE** are the scaling-oriented steps that matter
most once the model and context grow large.

Track B is what makes lokoLM *useful to others*: a **BPE tokenizer** makes it efficient and
ecosystem-compatible, **Hugging Face + LoRA** make the base model cheap to fine-tune with
tools researchers already use, and **function calling** turns it into a foundation for
building AI agents. These build on each other in order — each step unlocks the next.
