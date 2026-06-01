# Training on GPU (CUDA)

This documents how to train the decoder-only Transformer in
[model/lokolm/model.py](https://github.com/Larnova-Open-Source/lokolm/blob/main/lokolm/model.py) on an NVIDIA GPU. The training loop
lives in [model/train.py](https://github.com/Larnova-Open-Source/lokolm/blob/main/train.py).

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

The loop in [train.py](https://github.com/Larnova-Open-Source/lokolm/blob/main/train.py) applies the standard set of CUDA training
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
rank — out of scope for this minimal script, but [train.py](https://github.com/Larnova-Open-Source/lokolm/blob/main/train.py) is structured so
that wrapping is straightforward when you need it.

## 6. Checkpoints

The script saves `ckpt.pt` at the end — a dict holding both the weights and the model
config:
```python
ckpt = torch.load("ckpt.pt", map_location=device)
model = LokoLM(**ckpt["config"])
model.load_state_dict(ckpt["model"])
```
Storing the config alongside the weights means anything loading the checkpoint (such as
the inference script) rebuilds the exact model without hardcoding a config that has to
stay in sync with this file.

Note: if you trained with `torch.compile`, the state-dict keys are prefixed with
`_orig_mod.` — load into the compiled model, or strip the prefix when loading into a raw one.

Once you have a checkpoint, see the [Inference](inference.md) guide for generating text.
