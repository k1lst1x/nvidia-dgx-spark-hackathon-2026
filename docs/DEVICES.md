# Running the API on CPU, NVIDIA CUDA and Apple MPS

The same code and the same checkpoints run on any machine. Each model picks
its device when the API starts, runs one test inference there, and falls back
to CPU if that fails. Nothing has to be edited per machine.

## How a device is chosen

`apps/api/device_manager.py` resolves one setting per model:

| Variable | Model | Older name, still read |
|---|---|---|
| `YOLO_DEVICE` | YOLO11 detector + BoT-SORT (`ULTRALYTICS_YOLO_MODEL`) | `ULTRALYTICS_YOLO_DEVICE` |
| `SEG_DEVICE` | Separate segmenter (only used when the tracker model is not `-seg`) | `ULTRALYTICS_YOLO_DEVICE` |
| `DEPTH_DEVICE` | Depth Anything V2 | `DEPTH_ANYTHING_DEVICE` |
| `SIGN_DEVICE` | Server sign models (Uni-Sign, YOLO sign detector) | `UNISIGN_DEVICE` |

Values: `auto` (default), `cpu`, `mps`, `cuda` (same as `cuda:0`), `cuda:1`, ...

- `auto` picks CUDA, then MPS, then CPU, **as reported by PyTorch**
  (`torch.cuda.is_available()`, `torch.backends.mps.is_available()`). An
  NVIDIA card with a CPU-only PyTorch build selects CPU.
- A requested device that does not exist (for example `cuda:1` on a one-GPU
  machine, or `mps` on Windows) selects CPU and logs why.
- **Warm-up:** each model is loaded once and runs one inference on its device
  at startup. If loading or that inference fails, a fresh copy is loaded on CPU,
  tested again, and used from then on.
- **Failure during tracking:** if the GPU fails while tracking (CUDA error,
  out of memory), the detector is rebuilt on CPU and the camera session starts
  over: new BoT-SORT state, new track IDs, and that session's motion history is
  cleared, so old IDs are never linked to new objects. It stays on CPU until
  the API restarts; it never switches back and forth.
- `GET /health` → `devices` lists PyTorch's view of the hardware, any problem
  found (such as a CPU-only build), and for each model what was requested, what
  it runs on, and whether it fell back.
- FP16 is **off by default**. `YOLO_HALF`, `SEG_HALF`, `DEPTH_HALF=true` turn it
  on, on CUDA only.

Models that do not fit on one GPU together can be split, for example
`YOLO_DEVICE=cuda:0` and `DEPTH_DEVICE=cpu`.

Check a machine before starting the API:

```powershell
cd apps/api
python device_manager.py
```

It prints the PyTorch version, whether it is a CUDA build, the GPUs PyTorch
can use, the `nvidia-smi` driver, what each model would select, and a plain
explanation if a GPU is present but unusable.

## Windows with an NVIDIA GPU

The default `pip install torch` on Windows is **CPU-only**. Install a CUDA
build into a separate environment (this keeps the system Python untouched):

```powershell
cd apps/api
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe device_manager.py   # expect "cuda_available": true
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Pick the CUDA index from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).
The driver must support that CUDA version: `nvidia-smi` shows the highest CUDA
version the driver supports in its header. Install PyTorch **before**
`requirements.txt`, or Ultralytics pulls the CPU build.

## Linux with an NVIDIA GPU, and DGX Spark

Directly on the host: as above, with `.venv/bin/python`. On Linux x86-64 the
default `pip install torch` is a CUDA build; check with `python device_manager.py`.

With Docker (needs the NVIDIA Container Toolkit):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
# Pin a CUDA build of PyTorch in the image if the default is CPU-only:
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130 docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Without `docker-compose.gpu.yml` the container sees no GPU and every model
runs on CPU.

DGX Spark is ARM64 with CUDA 13 and unified memory. Use a PyTorch build made
for ARM64 + CUDA (an NVIDIA PyTorch container, or a `cu130` wheel) and confirm
with `python device_manager.py`. `nvidia-smi` reports no separate VRAM figure
on unified memory; the dashboard shows that as unavailable.

## macOS (Apple Silicon)

```bash
cd apps/api
python3 -m venv .venv
.venv/bin/python -m pip install torch torchvision
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python device_manager.py   # expect "mps_available": true
```

The standard macOS wheels include MPS. With `auto`, every model selects `mps`.
If an operation is missing on MPS, the warm-up fails and that model runs on CPU
(see `/health`); set its variable to `cpu` to skip the attempt.

## CPU-only machines

Nothing to configure: `auto` selects CPU. Set `ULTRALYTICS_YOLO_CPU_MODEL` to a
small model (for example `yolo11n-seg.pt`) so CPU does not run the large GPU
model, and leave `DEPTH_ANYTHING_INPUT_SIZE=auto` (364 px on CPU).

## Tests

```powershell
cd apps/api
.\.venv\Scripts\python.exe -m pytest -q tests
```

- `tests/test_device_manager.py`: selection order, missing devices, CPU-only
  builds, per-model variables, FP16 opt-in, warm-up fallback (fake PyTorch).
- `tests/test_yolo_tracker_devices.py`: GPU failure mid-session restarts on
  CPU with a new session and stays there; sessions do not share tracks (fake
  YOLO).
- `tests/test_devices_hardware.py`: real YOLO11n + BoT-SORT on a real photo
  panned over 8 frames, on `cpu`, `cuda:0` and `mps`; each is skipped where it
  is not available. Checks that a track ID survives every frame.

## Measured: RTX 4050 Laptop GPU (6 GB), Windows

`benchmarks/benchmark_devices.py`, 60 frames, PyTorch 2.14.1 + CUDA 13.0,
Ultralytics 8.4.208. Per-frame time through the API's own `YoloTracker`
(JPEG decode + YOLO + BoT-SORT) and `DepthAnythingMetric`.

| Model | Device | Precision | Median ms | p95 ms | FPS | GPU memory peak |
|---|---|---|---|---|---|---|
| yolo11n | CPU | FP32 | 50.9 | 65.7 | 19.6 | |
| yolo11n | CUDA | FP32 | 18.5 | 21.2 | 54.1 | 75 MB |
| yolo11n | CUDA | FP16 | 20.2 | 27.3 | 49.6 | 54 MB |
| yolo11n-seg | CPU | FP32 | 73.8 | 88.3 | 13.5 | |
| yolo11n-seg | CUDA | FP32 | 25.9 | 40.0 | 38.7 | 104 MB |
| yolo11n-seg | CUDA | FP16 | 25.3 | 29.6 | 39.5 | 69 MB |
| yolo11m-seg | CPU | FP32 | 457.2 | 509.3 | 2.2 | |
| yolo11m-seg | CUDA | FP32 | 35.2 | 40.1 | 28.4 | 328 MB |
| yolo11m-seg | CUDA | FP16 | 27.1 | 31.7 | 36.9 | 195 MB |
| Depth Anything V2 Small (364 px) | CPU | FP32 | 256.7 | 285.7 | 3.9 | |
| Depth Anything V2 Small (518 px) | CUDA | FP32 | 55.6 | 58.3 | 18.0 | 313 MB |
| Depth Anything V2 Small (518 px) | CUDA | FP16 | 40.8 | 43.1 | 24.5 | 352 MB |

- **FP16 quality:** on every YOLO model, FP16 found the same boxes as FP32
  (100% matched at IoU ≥ 0.7, confidence change ≤ 0.003). Depth maps differed
  by a median 0.03%. FP16 helps the larger models (yolo11m-seg 23% faster, depth
  27% faster) and not yolo11n.
- **Whole `/v1/track` request** with this repo's `.env` (yolo11m-seg on GPU,
  metric depth, obstacle points): about 110–200 ms per frame on CUDA, against
  over 3.5 s for YOLO and depth alone on CPU before. Track IDs stayed the same
  over 8 consecutive frames.
- Process RAM is 0.8–2.1 GB in all cases; both models together use well under
  1 GB of the 6 GB GPU.

Rerun on another machine:

```powershell
.\.venv\Scripts\python.exe benchmarks/benchmark_devices.py --models models/yolo11n.pt models/yolo11m-seg.pt --depth
```

## Known issues and what was not tested

1. **CPU-only PyTorch was the cause on Windows.** The API's Python had
   `torch 2.15.0+cpu`, so CUDA was never available regardless of settings. The
   newest CUDA build is 2.14.1 (one release behind the CPU build).
   `device_manager.py` now names this problem instead of silently using CPU.
2. **FP16 used to be automatic on CUDA** for YOLO and depth. It is now opt-in,
   so a GPU deployment that relied on it is slower until `YOLO_HALF=true` /
   `DEPTH_HALF=true` are set. The measurements above support turning both on
   for yolo11m-seg and depth.
3. **First frame of a new image size** costs about 2.5 s on CUDA while kernels
   are chosen. The startup warm-up uses 640×480, the usual camera size;
   other sizes pay this once.
4. **Not tested here:** macOS / MPS (no Mac available), Linux, DGX Spark, the
   Docker GPU path (`docker-compose.gpu.yml`, `TORCH_INDEX_URL`), and more
   than one GPU. The hardware test and benchmark are written to run there
   unchanged.
5. **One BoT-SORT state for the whole API.** Two cameras streaming at once
   reset each other's tracks (unchanged behaviour); a reset now also clears that
   session's motion history.
6. **Uni-Sign** (off by default) uses `SIGN_DEVICE` but has no warm-up
   inference: it needs real pose input and a 1.2 GB checkpoint.
7. The browser's 250-word sign model runs in ONNX Runtime Web and is not
   affected by these settings.
