# Train on Google Colab

Google Colab gives you a free GPU in the browser — the easiest way to train lokoLM if you
don't have a local NVIDIA card. The steps below clone the
[lokolm](https://github.com/larnova-open-source/lokolm) repository, train on a GPU, generate
text, and save your checkpoint so it survives the session.

Run each block in its own Colab **code cell**.

## 1. Turn on the GPU

In the Colab menu: **Runtime → Change runtime type → Hardware accelerator → GPU** (a free **T4**
is plenty for lokoLM), then **Save**. Do this *before* running anything.

## 2. Clone the repo

```python
!git clone https://github.com/larnova-open-source/lokolm.git
%cd lokolm
```

## 3. Check the GPU is visible

Colab ships with PyTorch + CUDA already installed, so there's usually **nothing to `pip
install`**. Confirm the GPU is there — this guards `get_device_name` behind the availability
check, so it prints a clear message instead of crashing when no GPU is attached:

```python
import torch
print("torch", torch.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("No GPU — Runtime → Change runtime type → GPU, then re-run from the top.")
```

You want it to print a GPU name (e.g. `Tesla T4`). If it says **No GPU** — or you hit
`AssertionError: Torch not compiled with CUDA enabled` — your runtime isn't on a GPU. Go to
**Runtime → Change runtime type → GPU → Save**; that **restarts and wipes the VM**, so re-run
your cells from the top (clone, data, etc.), not just this one. (On the free tier Colab
occasionally has no GPU available and only offers CPU — if so, wait and try again later.)

## 4. Add training data (recommended)

lokoLM trains on a single plain-text file named `input.txt`. Without one it falls back to a
tiny built-in string (fine for a smoke test, useless for real text). Two ways to provide a
corpus:

**Option A — download a public-domain corpus:**

```python
# ~1 MB of Shakespeare, a classic tiny-LM dataset
!wget -q -O input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
print(open("input.txt").read()[:200])
```

**Option B — upload your own `.txt`:**

```python
from google.colab import files
up = files.upload()                 # pick a .txt file from your computer
import os; os.rename(next(iter(up)), "input.txt")
```

> Use **unlabelled raw text** — lokoLM is a self-supervised next-token model, not an
> instruction/labelled model. See [Contributing](contributing.md) for what counts.

## 5. Train

```python
!python train.py
```

`train.py` auto-detects the GPU and prints the loss every 250 iterations. On a free T4 the
default ~85M-parameter config (`d_model=768`, `block_size=512`) fits in memory and runs; the
full 5000 iterations takes a while, so for a quick first run you can stop early once the loss
is dropping, or shrink the model first (see Tips).

When it finishes it saves `ckpt.pt`.

## 6. Generate text

```python
!python sample.py --prompt "ROMEO:" --max-new-tokens 300 --temperature 0.8 --top-k 40
```

See [Inference](inference.md) for what the sampling flags do.

## 7. Save your checkpoint (so you don't lose it)

Colab wipes the local disk when the session ends. Persist `ckpt.pt` either by **downloading**
it or by **copying it to Google Drive**:

```python
# Download to your computer
from google.colab import files
files.download("ckpt.pt")
```

```python
# …or save to Google Drive (survives across sessions)
from google.colab import drive
drive.mount("/content/drive")
!cp ckpt.pt /content/drive/MyDrive/lokolm-ckpt.pt
```

## Continue training in a later Colab session

`train.py` checkpoints every `eval_interval` steps and can resume from a checkpoint, so you
can train across several sessions. The trick is keeping `ckpt.pt` on **Google Drive** (Colab's
local disk is wiped each session).

**End of session 1 — save to Drive:**

```python
from google.colab import drive
drive.mount("/content/drive")
!cp ckpt.pt /content/drive/MyDrive/lokolm-ckpt.pt
```

**Start of session 2 — clone, remount Drive, copy the checkpoint back, and resume:**

```python
!git clone https://github.com/larnova-open-source/lokolm.git
%cd lokolm
from google.colab import drive
drive.mount("/content/drive")
!cp /content/drive/MyDrive/lokolm-ckpt.pt ckpt.pt          # bring the checkpoint back
!wget -q -O input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
!RESUME_FROM=ckpt.pt python train.py                       # picks up where it stopped
```

`RESUME_FROM` makes `train.py` reload the weights, optimizer state, and iteration number, so
momentum and the LR schedule continue seamlessly. If your previous run already reached
`max_iters`, **raise `max_iters` in `train.py`** before resuming, or there's nothing left to
train. (You can also point `RESUME_FROM` straight at the Drive path instead of copying:
`!RESUME_FROM=/content/drive/MyDrive/lokolm-ckpt.pt python train.py`.)

> For long runs, set `eval_interval` lower so checkpoints are written more often, and copy
> `ckpt.pt` to Drive periodically — a Colab disconnect then costs you at most a few intervals.

## Tips & gotchas

- **Save often for long runs.** Free Colab sessions disconnect after a while (and on idle).
  For real training, copy `ckpt.pt` to Drive periodically so a disconnect doesn't cost you
  the run.
- **Quicker experiments.** Edit the config at the top of
  [`train.py`](https://github.com/larnova-open-source/lokolm/blob/main/train.py) — e.g.
  `d_model=384`, `n_layers=6`, `block_size=256` — for a much faster, smaller model while you're
  getting the pipeline working.
- **T4 uses fp16, not bf16.** The T4 is a Turing GPU, so `train.py` automatically selects fp16
  mixed precision with a gradient scaler — no action needed.
- **`torch.compile` errors?** If the first iteration fails to compile on Colab, set
  `compile_model = False` in `train.py` and re-run; training works without it (just a bit
  slower).
- **Out of memory?** Lower `batch_size` (try 8), then `block_size`, in `train.py`.

## One-cell version

If you just want it running, paste this into a single GPU cell:

```python
!git clone https://github.com/larnova-open-source/lokolm.git
%cd lokolm
!wget -q -O input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
!python train.py
!python sample.py --prompt "ROMEO:" --max-new-tokens 300
```
