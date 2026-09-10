# Docker deployment

The default `docker/Dockerfile.flux-cu129` target installs FluxServe,
flux-scheduler, and flux-kernel into the image. No repository checkout or
`pip install` is needed after starting a deployment container.

## Build for the target GPUs

```bash
# H100 / H200
docker build -f docker/Dockerfile.flux-cu129 \
  --build-arg FLUX_KERNEL_CUDA_ARCH=90a -t fluxserve:h200 .

# Inspect the installed environment and verify native binaries without a GPU.
docker run --rm fluxserve:h200 env
docker run --rm fluxserve:h200 python -m flux_kernel.precompile --check
```

The default architecture list is `90a;100a;120`. A narrower build is faster;
use `100a` for B200, for example. CUDA, Torch, Triton, and the FlashInfer fork
revision are pinned in the Dockerfile. Build and runtime must use the same
native architecture configuration.

FlashInfer comes from `FLX-OSS/flashinfer-dllm`, branch `dllm/block-decode`, pinned
to `5ce4d077c33bbe167386cf5487a4a5bb4bcafbfd` (0.6.18). The older personal fork's
0.6.13 revision lacks the public paged block-extend API used by BF16 KV. Both
BF16 paged and ragged attention now use the public wrappers; FP8 KV retains its
FA2 path with explicit masks and per-layer scales.

The build verifies the pinned revision and the paged/ragged block-extend,
Graph offset-buffer, FP8 scale, and CUTLASS NVFP4 MoE interfaces without a GPU:

```bash
docker run --rm fluxserve:h200 python /opt/fluxserve/docker/check_flashinfer.py
```

The final manifest records these API checks separately from GPU validation.
FlashInfer installation preserves the installed Torch/CUDA stack using pip
constraints, with CUDA Python 12.9.7 and CUTLASS DSL 4.7.0. Its optional native
NIXL/NCCL EP build is disabled; FluxServe uses its own TP/EP communication path.

The CLI fixes the attention, KV dtype, and Graph mode for each serving process.
The full GPU test suite that mixes these configurations in one CUDA process
still triggers an illegal memory access in native BF16 padded Graph mode;
simpler mode-switch sequences pass, and the exact interaction is unresolved.
GPU tests isolate native padded mode in a subprocess and check real capture,
replay, and cleanup. The mixed-process failure is recorded separately in the
validation report; switching configurations within one process is not validated.

RMSNorm, RoPE, activation, and MoE CUDA libraries are compiled during the image
build without initializing a GPU. The build checks the installed libraries and
writes `/opt/fluxserve/build-info.json`, including architecture, dependency
versions, the FlashInfer revision, and a source digest. Host `.so`, object files,
build directories, and datasets are excluded from the Docker build context.

The pinned TVM FFI Torch DLPack bridges are also compiled for CPU and CUDA during
the build, and checked with an FP8 tensor. PyTorch 2.8 needs this bridge to pass
FP8 KV to FlashInfer. Their binary hashes are recorded in the build manifest;
the entrypoint seeds the writable TVM FFI cache from these binaries.

Deployment images set `FLUX_KERNEL_REQUIRE_PREBUILT=1`: missing or stale native
libraries produce an error before model loading instead of attempting to write
into read-only installed packages. Rebuild the image when changing its native
architecture, sources, or dependencies.

## Persistent caches and startup warmup

The entrypoint prepares writable caches under `/var/cache/fluxserve`. Mount a
named volume or a writable directory there to retain them between containers.
Compiler caches are separated by UID, build-manifest fingerprint, and installed
Python/compiler dependency versions. The model
download cache is shared across image revisions for the same UID.

The following explicit environment overrides are respected:

| Variable | Content |
|---|---|
| `FLUXSERVE_CACHE_DIR` | Root of the default cache directories |
| `TRITON_CACHE_DIR` | Triton compiled kernels |
| `TORCHINDUCTOR_CACHE_DIR` | Torch compiler cache |
| `TORCH_EXTENSIONS_DIR` | Torch extension builds |
| `FLASHINFER_WORKSPACE_BASE` | FlashInfer compiled kernels and workspace files |
| `TVM_FFI_CACHE_DIR` | FlashInfer's TVM FFI / Torch DLPack extension |
| `CUDA_CACHE_PATH` | CUDA driver compilation cache |
| `HF_HOME` | Model/config/tokenizer downloads |
| `HF_MODULES_CACHE` | Writable Transformers remote-code modules |
| `XDG_CACHE_HOME` | Other application caches |

An existing read-only `HF_HOME` is supported with `HF_HUB_OFFLINE=1`;
`HF_MODULES_CACHE` remains writable. Compiler caches must be writable. A bind
mount must permit the container's UID to create files. Keep the same UID when
reusing caches; `--user "$(id -u):$(id -g)"` is supported.

Before opening HTTP, `serve` completes representative eager prefill/diffusion
decode warmup and the existing online CUDA Graph preparation. It uses the loaded model's weight
format and actual KV dtype/scales, with batch sizes 1 and the configured mini
batch size (bounded by `max-num-seqs`). Temporary generation settings, RNG state,
and LLaDA2 KV data are restored after warmup. Every distributed rank finishes
startup before the HTTP application becomes available.

Ranks exchange startup outcomes before becoming ready. If any rank reports a
warmup failure, all ranks enter cleanup and report the failure; the CLI fallback
does not send a second shutdown command. If that exchange itself fails, the
launcher stops the remaining workers without another command collective.

This warms the exercised kernel variants; it does not promise that every future
batch/length combination will avoid JIT compilation. New shapes, GPU
architectures, and FP8 scales may require additional variants. The pinned
FlashInfer fork is warmed through its own wrappers; a stock upstream JIT-cache
wheel is not assumed compatible with this fork. CUDA Graphs are captured in each
new serving process and cannot be restored from the disk compiler cache.

## FP8 KV example

Prepare a model-matching scale JSON using the [KV calibration command](fp8-kv-cache.md)
and place it in `/path/to/calibration/scales.json`. Model weights and calibration
data are supplied separately from the image.

```bash
docker run --rm --gpus all --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  --read-only --tmpfs /tmp:rw,size=4g \
  -v fluxserve-cache:/var/cache/fluxserve \
  -v /path/to/calibration:/calibration:ro \
  -p 8000:8000 \
  fluxserve:h200 serve \
  --model thnkinbtfly/llada2.0-flash-fp8 \
  --quantization modelopt_fp8 \
  --kv-cache-dtype fp8_e4m3 --kv-cache-scales /calibration/scales.json \
  --tp-size 4 --ep-size 4 \
  --attention-backend flashinfer --scheduler-policy paged
```

For SDPA, use `--attention-backend sdpa --scheduler-policy default`.
To populate caches ahead of serving, use the same command and arguments with
`--warmup-only` appended. That command loads the model, completes warmup and any
requested Graph capture, then shuts down all workers without opening HTTP.
Reuse the cache volume for the subsequent serving container.

The Docker health check polls `http://127.0.0.1:8000/health`. It succeeds only
after executor startup. Warmup failures abort startup rather than reporting
healthy. If changing `--port`, also set
`FLUXSERVE_HEALTH_URL=http://127.0.0.1:<port>/health`.
The health check allows a 30-minute startup period; model loading/warmup logs
remain visible during that period.

`--skip-startup-warmup` is available for development; existing Graph preparation
still runs. For `bench_offline` and `calibrate_kv_cache`, the entrypoint prepares
the same caches and their existing benchmark warmup remains in use. Calibration
continues to exclude warmup from scale observations.

## Development environment

The dependency-only target retains the source-checkout workflow:

```bash
docker build -f docker/Dockerfile.flux-cu129 --target dependencies -t fluxserve:dev .
docker run --rm -it --gpus all --ipc=host \
  -v "$PWD":/workspace -w /workspace fluxserve:dev bash
# Inside the development container:
pip install -e flux-kernel/python --no-build-isolation
pip install -e flux-scheduler --no-build-isolation
pip install -e .
```

The deployment target also accepts `bash` or an explicit executable after the
image name. Its entrypoint uses `exec`, preserving signals and argument quoting.
HTTP shutdown closes the distributed workers and CUDA Graph resources before
the serving process exits. It first waits for any offloaded inference thread to
finish, even if its asyncio waiter was cancelled. Request release and executor
cleanup share the same execution lock; the CLI retains an idempotent cleanup
fallback.

Client cancellation immediately removes the request from the scheduler and
engine state and records its aborted result. Resource release runs in a tracked
task protected from the HTTP caller's cancellation, still using the execution
lock. Shutdown drains these pending releases before executor cleanup; release
failures are logged even when the client has already disconnected.

Current FlashInfer validation is recorded in [FlashInfer runtime experiments](../experiments/flashinfer-runtime.md).
Earlier image validation is recorded in [Docker deployment experiments](../experiments/docker-deployment.md).
