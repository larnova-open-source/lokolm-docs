# Training on GPU (CUDA)

This documents how to train the decoder-only Transformer in
[model/lokolm/model.py](https://github.com/larnova-open-source/lokolm/blob/main/lokolm/model.py) on an NVIDIA GPU. The training loop
lives in [model/train.py](https://github.com/larnova-open-source/lokolm/blob/main/train.py).

> **No local GPU?** The quickest way to train is a free GPU on Google Colab — see
> [Train on Colab](colab.md) for a copy-paste walkthrough.

## 1. Install PyTorch with CUDA

PyTorch ships separate wheels per CUDA version. **Do not** just `pip install torch`
on a GPU box — the default wheel may be CPU-only. Pick the build matching your driver
from <https://pytorch.org/get-started/locally/>. For CUDA 12.1:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify the GPU is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

This should print `True` and your GPU's name. If it prints `False`, you have a
CPU-only wheel or a driver mismatch — reinstall with the correct `--index-url`.

## 2. Run

```powershell
python train.py
```

Drop a text file named `input.txt` next to the script to train on your own corpus;
otherwise it uses a tiny built-in sample so the script runs out of the box.

## 3. What makes it use the GPU efficiently

The loop in [train.py](https://github.com/larnova-open-source/lokolm/blob/main/train.py) applies the standard set of CUDA training
techniques. Each maps to a specific line:

### Device placement
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LokoLM(...).to(device)
```
The model parameters and every input batch must live on the same device. `get_batch`
moves each batch to the GPU. The script degrades gracefully to CPU if no GPU is found.

### Mixed precision (autocast)
```python
with torch.autocast(device_type=device, dtype=amp_dtype):
    _, loss = model(x, y)
```
Runs matmuls in 16-bit instead of 32-bit. This roughly **halves memory** and gives a
large speedup on Tensor Cores, while keeping numerically sensitive ops in fp32 automatically.

- **bf16** (`torch.bfloat16`) — preferred on Ampere and newer (A100, RTX 30xx/40xx, H100).
  Same exponent range as fp32, so it needs **no loss scaling**.
- **fp16** (`torch.float16`) — for older GPUs (e.g. V100, GTX/RTX 20xx). Narrow range, so
  it requires a `GradScaler` to prevent gradient underflow.

The script auto-selects via `torch.cuda.is_bf16_supported()`.

### Gradient scaling (fp16 only)
```python
scaler = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))
scaler.scale(loss).backward()
scaler.unscale_(optimizer)        # unscale before clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
scaler.step(optimizer)
scaler.update()
```
The scaler multiplies the loss up before backward (so small fp16 gradients don't flush to
zero) and divides back out before the optimizer step. It's a no-op for bf16/fp32.

### torch.compile
```python
model = torch.compile(model)
```
PyTorch 2.x JIT-fuses the graph into optimized kernels — often a 1.3–2x speedup. The first
iteration is slow (compilation). Set `compile_model = False` if you hit a compile error.

### TF32 matmuls
```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```
Lets fp32 matmuls use Tensor Cores at reduced internal precision — free speedup on Ampere+
with negligible accuracy impact.

### Overlapping data transfer
```python
x = x.pin_memory().to(device, non_blocking=True)
```
Pinned host memory + `non_blocking=True` lets the CPU→GPU copy overlap with GPU compute
instead of stalling.

## 4. Tuning for your GPU

| Knob | Effect | If you hit out-of-memory |
|------|--------|--------------------------|
| `batch_size` | throughput | lower it first |
| `block_size` | context length; cost grows ~quadratically | lower it |
| `d_model`, `n_layers`, `n_heads` | model capacity | shrink the model |
| `amp_dtype` | memory + speed | ensure AMP is enabled (it is, on CUDA) |

**Monitor utilization** while training:
```powershell
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1
```
You want GPU utilization near 100%. If it's low, the GPU is starved — increase
`batch_size` or check that data loading isn't the bottleneck.

### Gradient accumulation (simulate a bigger batch)
If the batch you need won't fit, accumulate gradients over several micro-batches before
stepping. Conceptually:
```python
accum_steps = 4
for micro in range(accum_steps):
    x, y = get_batch("train")
    with torch.autocast(device_type=device, dtype=amp_dtype):
        _, loss = model(x, y)
        loss = loss / accum_steps
    scaler.scale(loss).backward()
# clip + step once, after the loop
```
This gives the effective batch size of `batch_size * accum_steps` at the memory cost of one.

## 5. Multi-GPU (brief)

For a single machine with multiple GPUs, use `torchrun` with PyTorch's
`DistributedDataParallel` (DDP):

```powershell
torchrun --standalone --nproc_per_node=NUM_GPUS train.py
```

DDP requires wrapping the model in `DistributedDataParallel` and sharding the data per
rank — out of scope for this minimal script, but [train.py](https://github.com/larnova-open-source/lokolm/blob/main/train.py) is structured so
that wrapping is straightforward when you need it.

## 6. Checkpoints

The script writes `ckpt.pt` **every `eval_interval` steps** (not just at the end), so a crash
or a disconnected Colab session doesn't cost you the whole run. Each checkpoint is a dict
holding the weights, the model config, the optimizer/scaler state, and the iteration number:

```python
ckpt = torch.load("ckpt.pt", map_location=device)
model = LokoLM(**ckpt["config"])
model.load_state_dict(ckpt["model"])
# also inside: ckpt["optimizer"], ckpt["scaler"], ckpt["iter"]
```

Storing the config alongside the weights means anything loading the checkpoint (such as
the inference script) rebuilds the exact model without hardcoding a config that has to
stay in sync with this file. The save is atomic (written to `ckpt.pt.tmp` then renamed), so a
checkpoint is never left half-written.

Note: if you trained with `torch.compile`, the state-dict keys are prefixed with
`_orig_mod.` — load into the compiled model, or strip the prefix when loading into a raw one
(both `sample.py` and the resume path below do this for you).

## 7. Resuming / continuing training in a later session

Because the checkpoint also stores the **optimizer state** and the **iteration count**, you
can stop and pick up exactly where you left off — momentum and the learning-rate schedule
continue seamlessly. Point `train.py` at a checkpoint via the `RESUME_FROM` environment
variable (or set `resume_from` in the file):

```powershell
# Windows PowerShell
$env:RESUME_FROM = "ckpt.pt"; python train.py
```
```bash
# Linux / macOS / Colab shell
RESUME_FROM=ckpt.pt python train.py
```

Training continues from the saved `iter` up to `max_iters`. **To train *further* than the
original run**, first raise `max_iters` in `train.py` (e.g. from 5000 to 8000) — otherwise the
loop has nothing left to do. Keep the model config (`d_model`, `n_layers`, …) unchanged when
resuming; the weights only fit the architecture they were trained with.

The learning-rate schedule is a pure function of the **absolute step**: it warms up, cosine-
decays from `learning_rate` to `min_lr` over `lr_decay_iters` steps, then holds at `min_lr`.
Crucially, `lr_decay_iters` is **independent of `max_iters`** and is **saved in the
checkpoint**, so a resumed run reproduces the exact same curve — no discontinuity. Extending a
run past `lr_decay_iters` simply trains the tail at the constant `min_lr` floor.

> Plan the horizon up front: if you intend to train for, say, 20k steps total across several
> sessions, set `lr_decay_iters = 20000` in the **first** session. Because resume restores the
> schedule from the checkpoint, editing `lr_decay_iters` on a later run is ignored (by design,
> to keep the curve exact); to deliberately re-shape it, start fresh or edit the checkpoint.

Once you have a checkpoint, see the [Inference](inference.md) guide for generating text, or
[Train on Colab](colab.md) for resuming across Colab sessions via Google Drive.
