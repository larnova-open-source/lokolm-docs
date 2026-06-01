# Inference — Generating Text

Training and inference are **separate steps**: [training](training.md) fits the model's
weights; inference *uses* those weights to produce text. Once you have a checkpoint
(`ckpt.pt`), [model/sample.py](https://github.com/Larnova-Open-Source/lokolm/blob/main/sample.py) generates from it.

```powershell
python sample.py --prompt "hello" --max-new-tokens 200 --temperature 0.8 --top-k 40
```

## How it works

The model already supports generation — [model.py](https://github.com/Larnova-Open-Source/lokolm/blob/main/lokolm/model.py) has a
`generate()` method. For each new token it:

1. **Crops** the running sequence to the last `block_size` tokens (the model can't attend
   beyond its context window).
2. Runs a **forward pass** and takes the logits at the final position.
3. Applies **temperature** and **top-k** filtering, softmaxes to a probability
   distribution, and **samples** the next token with `torch.multinomial`.
4. **Appends** the sampled token and repeats.

`sample.py` is the small driver around that loop. It:

- loads `ckpt.pt` and rebuilds the **exact trained model** from the `config` saved inside
  it (no need to mirror `train.py`'s settings),
- strips any `_orig_mod.` prefix left by `torch.compile`,
- encodes the prompt as raw **UTF-8 bytes** — lokoLM is byte-level (`vocab_size=256`), so
  there's no tokenizer step — generates, and decodes the bytes back to text.

## Sampling knobs

| Flag | Effect |
|------|--------|
| `--temperature` | Scales logits before sampling. `< 1.0` = more confident/repetitive, `> 1.0` = more random/creative. `1.0` = the model's raw distribution. |
| `--top-k` | Restricts sampling to the `k` most likely tokens each step, cutting off the unlikely tail. `0` disables it (sample from the full distribution). |
| `--max-new-tokens` | How many tokens (bytes) to generate. |
| `--prompt` | The text to continue. |
| `--seed` | Fixes the RNG so a run is reproducible. |

## A note on performance

This is **unoptimized** inference: every step recomputes attention over the whole
context from scratch. That's manageable at lokoLM's `block_size=512`, but it grows with
context length — and it's exactly what a **KV cache** would eliminate, caching past
keys/values so each step only computes attention for the new token. The [roadmap](roadmap.md)'s
[GQA](roadmap.md#4-gqa-grouped-query-attention) step shrinks that cache further, and
[function calling](roadmap.md#8-function-calling--tool-use) wraps this generation loop into
an agentic one (parse a tool call → run it → feed the result back → continue). Those are
inference *enhancements*; the basic generation here is already part of v1.
