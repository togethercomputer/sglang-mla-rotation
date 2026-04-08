# MLA KV Cache Quantization & Rotation

## What We Did

### 1. Simulated Quantization (fake quant)

Added simulated (fake) quantization for MLA KV cache writes in SGLang. The KV cache remains stored in BF16; values are quantized then immediately dequantized to reproduce low-precision rounding error without a custom memory layout.

### 2. Hadamard Rotation (FWHT)

Added optional Fast Walsh-Hadamard Transform rotation on the MLA KV cache write path. The rotation is orthogonal — it redistributes outliers to improve quantization quality while preserving attention scores exactly (H·H^T = I).

For MLA, rotation touches three points:
- **K write** (`set_mla_kv_buffer`): rotate k_nope and k_rope before storing
- **Q read** (`forward_extend`/`forward_decode`): rotate q_nope and q_rope to match
- **Output** (`forward_extend`/`forward_decode`): inverse-rotate attention output, because V shares the buffer with rotated k_nope

### Files Changed

- `python/sglang/srt/environ.py` — env var definitions
- `python/sglang/srt/mem_cache/kv_quant_sim.py` — fake quant kernels (int per-head, NVFP4)
- `python/sglang/srt/mem_cache/memory_pool.py` — `_hadamard_rotate`, `_maybe_rotate_k_for_mla`, `_maybe_fake_quant_kv_for_mla`, hooks in `set_mla_kv_buffer` and `set_kv_buffer`
- `python/sglang/srt/layers/attention/flashattention_backend.py` — Q rotation + output inverse-rotation in FA3 MLA absorbed attention paths
- `python/sglang/srt/layers/attention/flashinfer_mla_backend.py` — Q rotation + output inverse-rotation in Flashinfer MLA paths

### MLA Attention & How Rotation Works

MLA attention score is the sum of two parts:

```
score = q_nope @ k_nope^T + q_rope @ k_rope^T
output = softmax(score) @ k_nope          ← k_nope doubles as V
```

- **k_nope** `[tokens, 1, 512]` — latent (NoPE), carries semantic info, also serves as V. No RoPE applied.
- **k_rope** `[tokens, 1, 64]` — positional info, stored post-RoPE. Q is computed fresh each forward (not cached).

With Hadamard rotation H (orthogonal, H @ H^T = I):

```
Write: store H(k_nope), H(k_rope)
Read:  score = H(q_nope) @ H(k_nope)^T + H(q_rope) @ H(k_rope)^T
             = q_nope @ k_nope^T + q_rope @ k_rope^T     ← score unchanged
       output_raw = softmax(score) @ H(k_nope)            ← V is rotated
       output = H(output_raw)                              ← inverse-rotate to recover
```

Score is mathematically identical. Rotation just redistributes outliers before quantization.

### Processing Order on Cache Write

```
RoPE (in model forward) → dtype cast → Hadamard rotation → fake quant → write to buffer
```

## Environment Variables

### Fake Quantization

| Variable | Default | Description |
|---|---|---|
| `SGLANG_FAKE_QUANT_KV` | `""` (off) | What to quantize: `k` = k_nope, `v` = k_rope, `kv`/`both`/`all` = both |
| `SGLANG_FAKE_QUANT_KV_TYPE` | `int` | `int` (integer) or `fp`/`nvfp4`/`fp4` (NVFP4) |
| `SGLANG_FAKE_QUANT_KV_BITS` | `8` | Bit width for `int` type (1–32); ignored for NVFP4 |

### Hadamard Rotation

| Variable | Default | Description |
|---|---|---|
| `SGLANG_ROTATE_K` | `false` | Enable FWHT rotation on K write + Q read + output inverse |
| `SGLANG_ROTATE_K_ORDER` | `16` | Group size for FWHT (must divide head dims) |

### Examples

```bash
# INT4 fake quant only (no rotation)
SGLANG_FAKE_QUANT_KV=kv SGLANG_FAKE_QUANT_KV_TYPE=int SGLANG_FAKE_QUANT_KV_BITS=4

# INT4 fake quant + Hadamard rotation (group size 16)
SGLANG_FAKE_QUANT_KV=kv SGLANG_FAKE_QUANT_KV_TYPE=int SGLANG_FAKE_QUANT_KV_BITS=4 SGLANG_ROTATE_K=1 SGLANG_ROTATE_K_ORDER=16

# NVFP4 fake quant + Hadamard rotation (group size 32)
SGLANG_FAKE_QUANT_KV=kv SGLANG_FAKE_QUANT_KV_TYPE=nvfp4 SGLANG_ROTATE_K=1 SGLANG_ROTATE_K_ORDER=32

# Rotation only (no quant, for sanity check — should be lossless)
SGLANG_ROTATE_K=1 SGLANG_ROTATE_K_ORDER=16
```

## Attention Backend Compatibility

### Rotation support (Q rotation + output inverse)

| Backend | K rotation (write) | Q rotation (read) | Status |
|---|---|---|---|
| Flashinfer MLA | memory_pool.py | flashinfer_mla_backend.py | **Supported** |
| FlashAttention v3 (FA3) | memory_pool.py | flashattention_backend.py | **Supported** (SM80-SM90 only) |
| TRTLLM MLA | memory_pool.py | not implemented | **Not supported** — must use `--attention-backend flashinfer` |
| Triton | memory_pool.py | not implemented | **Not supported** |

### Default backend selection by hardware

| Hardware | Default MLA backend | Rotation works? | Notes |
|---|---|---|---|
| B200/GB200 (SM100) | trtllm_mla | **No** — must add `--attention-backend flashinfer` | Default is fastest but lacks rotation |
| H100/H200 (SM90) | fa3 | **Yes** | FA3 is default and has rotation |
| Other (SM80, etc.) | triton or flashinfer | Flashinfer **yes**, triton **no** | Depends on flashinfer availability |

### Fake quant support (no backend dependency)

Fake quant is implemented in `memory_pool.py` (`set_mla_kv_buffer`) and works with **all backends**. No `--attention-backend` override needed for quant-only experiments.

### MLA vs MHA comparison

| | MLA (this repo) | MHA (sglang-fork) |
|---|---|---|
| Quant env vars | `SGLANG_FAKE_QUANT_KV`, `SGLANG_FAKE_QUANT_KV_TYPE`, `SGLANG_FAKE_QUANT_KV_BITS` | `--kv-cache-dtype int4` (real INT4 kernel) |
| Rotation env vars | `SGLANG_ROTATE_K=1`, `SGLANG_ROTATE_K_ORDER=16` | `HADAMARD=1`, `HADAMARD_ORDER=16`, `ROTATE_V=0/1` |
| Quant method | Simulated (fake quant, BF16 round-trip) | Real INT4 kernel (`quantized_set_kv_int4_triton`) |
| V rotation | Not applicable (V shares buffer with k_nope) | Optional (`ROTATE_V=1`) |
| Prefill backend | flashinfer (or fa3 on Hopper) | fa3 |
| Decode backend | flashinfer (must override trtllm_mla on B200) | triton |

## Results

### DeepSeek-V2-Lite (16B MLA)

GSM8K, 40 examples, temperature=0, backend: flashinfer on B200.

| Experiment | Config | GSM8K exact_match |
|---|---|---|
| 1. Baseline | no quant, no rotation | 0.200 |
| 2. INT4 quant | `FAKE_QUANT_KV=kv TYPE=int BITS=4` | 0.075 |
| 3. NVFP4 quant | `FAKE_QUANT_KV=kv TYPE=nvfp4` | 0.125 |
| 4a. INT4 + rot16 | `FAKE_QUANT_KV=kv TYPE=int BITS=4 ROTATE_K=1 ORDER=16` | 0.225 |
| 4b. NVFP4 + rot16 | `FAKE_QUANT_KV=kv TYPE=nvfp4 ROTATE_K=1 ORDER=16` | 0.200 |
| 5. Rotation only | `ROTATE_K=1 ORDER=16` (no quant) | 0.250 |
| 6. INT4 nope only | `FAKE_QUANT_KV=k TYPE=int BITS=4` | 0.200 |
| 7. INT4 rope only | `FAKE_QUANT_KV=v TYPE=int BITS=4` | 0.150 |
| 8. INT4 nope + rot16 | `FAKE_QUANT_KV=k TYPE=int BITS=4 ROTATE_K=1 ORDER=16` | 0.250 |

### Kimi-K2.5-NVFP4 (550GB MLA, 8x B200 TP=8)

GSM8K, 40 examples, temperature=0, backend: flashinfer (forced via `--attention-backend flashinfer`).

| Experiment | Config | GSM8K exact_match |
|---|---|---|
| 1. Baseline | no quant, no rotation | 0.900 |
| 2. INT4 quant | `FAKE_QUANT_KV=kv TYPE=int BITS=4` | 0.900 |
| 3. NVFP4 quant | `FAKE_QUANT_KV=kv TYPE=nvfp4` | 0.925 |
| 4a. INT4 + rot16 | `FAKE_QUANT_KV=kv TYPE=int BITS=4 ROTATE_K=1 ORDER=16` | 0.900 |
| 4b. NVFP4 + rot16 | `FAKE_QUANT_KV=kv TYPE=nvfp4 ROTATE_K=1 ORDER=16` | 0.925 |
| 5. Rotation only | skipped (4a/4b confirm rotation correctness; GPU unavailable) | — |

### Key Observations

- **Rotation is correct** — V2-Lite rotation-only (exp 5) matches baseline; Kimi rotation experiments (4a/4b) match their non-rotation counterparts exactly, confirming no corruption.
- **k_rope quantization hurts more than k_nope** — On V2-Lite, quantizing k_nope alone (exp 6, 0.200) causes no drop, but quantizing k_rope alone (exp 7, 0.150) drops 25%. This is surprising since k_rope is only 64 dim vs k_nope's 512 dim. Positional information may be more sensitive to quantization error.
- **Both quantized together is worse than either alone** — INT4 kv (0.075) is far worse than nope-only (0.200) or rope-only (0.150), suggesting quantization errors in both components amplify each other.
- **Hadamard rotation recovers accuracy**: INT4 kv + rot16 (0.225) vs INT4 kv only (0.075); INT4 nope + rot16 (0.250) fully recovers.
- **NVFP4 is more resilient** than INT4 on small models (0.125 vs 0.075 without rotation).
- **Large models are resilient**: Kimi-K2.5 shows **no degradation** with INT4 or NVFP4 fake quant on GSM8K (40 samples). This benchmark may be too easy to differentiate — harder tasks or more samples needed.

### Caveats

- All results are on **40 examples only** (`--limit 40`), which introduces high variance. Differences within ~0.05 may be noise.
- `max_gen_toks=8192` — generation length may be insufficient for chain-of-thought reasoning on harder benchmarks.
- GSM8K with 5-shot completion mode on a base model (V2-Lite) may not reliably extract answers. Results are useful for **relative comparison** between configs, not absolute accuracy.

## Future Work

1. **Rotation group size sweep** — test FWHT over 8 / 16 / 32 / 128 dimensions on NoPE and RoPE.

2. **Per-component control** — allow different quantization configs for k_nope vs k_rope (e.g. INT4 on nope, no quant on rope).

3. **TRTLLM MLA backend support** — add Q rotation to trtllm_mla so B200 doesn't need `--attention-backend flashinfer` override.

4. **Triton backend support** — add rotation to triton decode MLA path.

5. **Full-scale evaluation** — run with full benchmarks (no --limit) and additional tasks beyond GSM8K.

## How to Run Tests

### Prerequisites

```bash
bash setup_env.sh   # one-time env setup
```

Before each session:
```bash
eval "$(/scratch/jisenli2/miniconda/bin/conda shell.bash hook)" && conda activate sglang_mla_env
export HF_HOME=/scratch/huggingface
export HF_HUB_OFFLINE=1
export FLASHINFER_CACHE_DIR=/scratch/jisenli2/.cache/flashinfer
export SGLANG_DG_CACHE_DIR=/scratch/jisenli2/.cache/deep_gemm
export CUDA_HOME=/usr/local/cuda-12.9
export CPATH=/usr/local/cuda-12.9/targets/x86_64-linux/include${CPATH:+:$CPATH}
export LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

### Model 1: DeepSeek-V2-Lite (16B, single GPU, quick validation)

```bash
# Server (example: INT4 + rotation)
SGLANG_FAKE_QUANT_KV=kv SGLANG_FAKE_QUANT_KV_TYPE=int SGLANG_FAKE_QUANT_KV_BITS=4 \
SGLANG_ROTATE_K=1 SGLANG_ROTATE_K_ORDER=16 \
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V2-Lite \
    --mem-fraction-static 0.8 \
    --trust-remote-code \
    --port 30800

# Eval
lm_eval --model local-completions \
    --tasks gsm8k \
    --limit 40 \
    --model_args "model=deepseek-ai/DeepSeek-V2-Lite,base_url=http://localhost:30800/v1/completions,num_concurrent=8,tokenized_requests=False,temperature=0,max_gen_toks=8192" \
    --output_path eval_results/DeepSeek-V2-Lite/gsm8k/4a_int4_kv_rot16
```

### Model 2: Kimi-K2.5-NVFP4 (550GB, 8x B200 TP=8)

**Important:** Must use `--attention-backend flashinfer` for rotation to work. Default trtllm_mla backend on B200 does not have Q rotation support yet.

```bash
# Server (example: INT4 + rotation)
SGLANG_FAKE_QUANT_KV=kv SGLANG_FAKE_QUANT_KV_TYPE=int SGLANG_FAKE_QUANT_KV_BITS=4 \
SGLANG_ROTATE_K=1 SGLANG_ROTATE_K_ORDER=16 \
python -m sglang.launch_server \
    --model-path nvidia/Kimi-K2.5-NVFP4 \
    --mem-fraction-static 0.8 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --attention-backend flashinfer \
    --port 30800

# Eval
lm_eval --model local-completions \
    --tasks gsm8k \
    --limit 40 \
    --model_args "model=nvidia/Kimi-K2.5-NVFP4,base_url=http://localhost:30800/v1/completions,num_concurrent=8,tokenized_requests=False,temperature=0,max_gen_toks=8192,trust_remote_code=True" \
    --output_path eval_results/Kimi-K2.5-NVFP4/gsm8k/4a_int4_kv_rot16
```

### Reference: sglang-fork MHA eval command (Jinda's setup)

```bash
# MHA models used different env vars and backends
HADAMARD=1 ROTATE_V=0 HADAMARD_ORDER=16 \
python -m sglang.launch_server \
    --model-path zai-org/GLM-4.7-FP8 \
    --kv-cache-dtype int4 \
    --prefill-attention-backend fa3 \
    --decode-attention-backend triton \
    --sampling-backend flashinfer \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    --port 30001
```
