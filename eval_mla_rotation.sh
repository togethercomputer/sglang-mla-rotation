#!/bin/bash
# eval_mla_rotation.sh — MLA KV cache fake-quant + rotation evaluation
#
# Modes:
#   BASE  — no rotation, no quant (BF16 baseline)
#   QUANT — fake quant only (SGLANG_FAKE_QUANT_KV)
#   ROT   — rotation + fake quant (SGLANG_ROTATE_K + SGLANG_FAKE_QUANT_KV)
#
# Uses lm_eval for accuracy benchmarks.

set -eo pipefail

cleanup() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Interrupted — killing all child processes..."
    kill -- -$$ 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

# =============================================================================
# Model Configs
# =============================================================================
# Format: "mode|quant_kv|quant_type|quant_bits|rotate_k|rotate_order|model_name|eval_gpus|eval_tp|tasks"
#
#   mode         : BASE, QUANT, or ROT
#   quant_kv     : "" (off) or "kv"/"k"/"v" (what to fake-quant)
#   quant_type   : "int" or "nvfp4"
#   quant_bits   : 4, 8, etc. (ignored for nvfp4)
#   rotate_k     : 0 or 1
#   rotate_order : 16, 32, etc. (ignored if rotate_k=0)
#   model_name   : HuggingFace model ID
#   eval_gpus    : comma-separated GPU IDs
#   eval_tp      : tensor parallel size
#   tasks        : lm_eval task string (e.g. "gsm8k" or "mmlu_pro")

LM_EVAL_TASKS="gsm8k"
LM_EVAL_LIMIT=""  # empty = no limit; set a number for quick testing

MODEL_CONFIGS=(
    # --- DeepSeek-V2-Lite-16B-NVFP4 ---
    # Baseline (BF16, no quant, no rotation)
    "BASE ||int|4|0|0|Beambutbetter/Deepseek-V2-Lite-16B-NVFP4|0|1|${LM_EVAL_TASKS}"

    # INT4 fake quant only
    "QUANT|kv|int|4|0|0|Beambutbetter/Deepseek-V2-Lite-16B-NVFP4|0|1|${LM_EVAL_TASKS}"

    # NVFP4 fake quant only
    "QUANT|kv|nvfp4|4|0|0|Beambutbetter/Deepseek-V2-Lite-16B-NVFP4|0|1|${LM_EVAL_TASKS}"

    # INT4 + rotation order=16
    "ROT|kv|int|4|1|16|Beambutbetter/Deepseek-V2-Lite-16B-NVFP4|0|1|${LM_EVAL_TASKS}"

    # INT4 + rotation order=32
    "ROT|kv|int|4|1|32|Beambutbetter/Deepseek-V2-Lite-16B-NVFP4|0|1|${LM_EVAL_TASKS}"

    # INT4 fake quant on k_nope only (no k_rope quant)
    "QUANT|k|int|4|0|0|Beambutbetter/Deepseek-V2-Lite-16B-NVFP4|0|1|${LM_EVAL_TASKS}"
)

# =============================================================================
# Server & Eval Config
# =============================================================================
BASE_PORT=30100
NUM_CONCURRENT=8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/eval_results"
LOGS_DIR="$SCRIPT_DIR/eval_logs"

export HF_HOME=${HF_HOME:-/scratch/huggingface}
if [ -z "$HUGGING_FACE_HUB_TOKEN" ] && [ -f "$HOME/.cache/huggingface/token" ]; then
    export HUGGING_FACE_HUB_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"

CONDA_BASE="/scratch/$USER/miniconda"
CONDA_ENV_NAME="sglang_mla_env"
CONDA_ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV_NAME"
PYTHON="$CONDA_ENV_DIR/bin/python3"
LM_EVAL="$CONDA_ENV_DIR/bin/lm_eval"

export TRITON_CACHE_DIR="/dev/shm/triton_cache_$USER"

GPU_FREE_MEM_MB="${GPU_FREE_MEM_MB:-500}"
GPU_POLL_INTERVAL="${GPU_POLL_INTERVAL:-60}"

mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# =============================================================================
# Helpers
# =============================================================================

extract_model_short_name() { basename "$1"; }

unique_log_path() {
    local base="$1"
    if [ ! -e "$base" ]; then echo "$base"; return; fi
    local i=1
    while [ -e "${base}-${i}" ]; do i=$((i + 1)); done
    echo "${base}-${i}"
}

BATCH_LOG_FILE=""
log_message() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$BATCH_LOG_FILE"; }

wait_for_server() {
    local port="$1" pid="$2" label="$3"
    local max_wait=1800 elapsed=0
    log_message "Waiting for $label (port $port)..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "http://localhost:${port}/health" > /dev/null 2>&1; then
            log_message "Server ready (${elapsed}s)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            log_message "Server process died"
            return 1
        fi
        [ $((elapsed % 60)) -eq 0 ] && [ $elapsed -gt 0 ] && log_message "  Still waiting... ${elapsed}s"
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log_message "Server timeout after ${max_wait}s"
    return 1
}

stop_server() {
    local pid="$1" label="$2"
    log_message "Stopping $label (PID $pid)..."
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    log_message "Stopped $label"
}

gpus_are_free() {
    local gpu_list="$1"
    IFS=',' read -ra GPU_IDS <<< "$gpu_list"
    for gpu_id in "${GPU_IDS[@]}"; do
        local used_mb
        used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d '[:space:]')
        if [ -z "$used_mb" ]; then return 1; fi
        if [ "$used_mb" -ge "$GPU_FREE_MEM_MB" ]; then return 1; fi
    done
    return 0
}

wait_for_gpus_free() {
    local gpu_list="$1" label="$2"
    local waited=0
    while ! gpus_are_free "$gpu_list"; do
        if [ $((waited % 300)) -eq 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for GPU(s) [$gpu_list] to be free... ($((waited / 60))min) [$label]"
        fi
        sleep "$GPU_POLL_INTERVAL"
        waited=$((waited + GPU_POLL_INTERVAL))
    done
}

# =============================================================================
# eval_single_config
# =============================================================================

eval_single_config() {
    local mode="$1" quant_kv="$2" quant_type="$3" quant_bits="$4" \
          rotate_k="$5" rotate_order="$6" model_name="$7" \
          gpu_devices="$8" tp_size="$9" tasks="${10}" server_port="${11}"

    local model_short
    model_short="$(extract_model_short_name "$model_name")"

    # Build suffix for result dirs
    local suffix
    case "$mode" in
        BASE)  suffix="baseline" ;;
        QUANT) suffix="fq_${quant_kv}_${quant_type}${quant_bits}" ;;
        ROT)   suffix="fq_${quant_kv}_${quant_type}${quant_bits}_rot${rotate_order}" ;;
        *)     echo "ERROR: unknown mode '$mode'"; return 1 ;;
    esac

    mkdir -p "$LOGS_DIR/${model_short}"
    BATCH_LOG_FILE=$(unique_log_path "$LOGS_DIR/${model_short}/${suffix}.log")

    log_message "=========================================="
    log_message "Mode:     $mode"
    log_message "Model:    $model_name"
    log_message "TP:       $tp_size  GPUs: $gpu_devices"
    log_message "Quant:    kv=$quant_kv type=$quant_type bits=$quant_bits"
    log_message "Rotation: rotate_k=$rotate_k order=$rotate_order"
    log_message "Tasks:    $tasks"
    log_message "=========================================="

    # Build env vars for server
    local env_prefix=""
    if [[ "$mode" == "QUANT" || "$mode" == "ROT" ]]; then
        env_prefix="SGLANG_FAKE_QUANT_KV=$quant_kv SGLANG_FAKE_QUANT_KV_TYPE=$quant_type SGLANG_FAKE_QUANT_KV_BITS=$quant_bits"
    fi
    if [[ "$mode" == "ROT" ]]; then
        env_prefix="$env_prefix SGLANG_ROTATE_K=1 SGLANG_ROTATE_K_ORDER=$rotate_order"
    fi

    local server_log
    server_log=$(unique_log_path "$LOGS_DIR/${model_short}/${suffix}_server.log")

    log_message "Starting server (port $server_port)..."
    eval "$env_prefix \
    CUDA_VISIBLE_DEVICES=$gpu_devices \
    PATH=\"$(dirname "$PYTHON"):\$PATH\" \
    LIBRARY_PATH=\"/usr/local/cuda/targets/x86_64-linux/lib\${LIBRARY_PATH:+:\$LIBRARY_PATH}\" \
    LD_LIBRARY_PATH=\"/usr/local/cuda/targets/x86_64-linux/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}\" \
    \"$PYTHON\" -m sglang.launch_server \
        --model-path \"$model_name\" \
        --mem-fraction-static 0.8 \
        --chunked-prefill-size 4096 \
        --kv-cache-dtype auto \
        --tensor-parallel-size $tp_size \
        --host 0.0.0.0 \
        --port $server_port \
        --trust-remote-code \
        > \"$server_log\" 2>&1 &"
    local server_pid=$!
    log_message "Server PID: $server_pid"

    if ! wait_for_server "$server_port" "$server_pid" "SGLang server"; then
        tail -50 "$server_log" | tee -a "$BATCH_LOG_FILE"
        stop_server "$server_pid" "SGLang server"
        return 1
    fi

    # Run lm_eval
    local result_dir="$RESULTS_DIR/${model_short}/${suffix}"
    mkdir -p "$result_dir"

    local limit_flag=""
    if [ -n "$LM_EVAL_LIMIT" ]; then
        limit_flag="--limit $LM_EVAL_LIMIT"
    fi

    log_message "Running lm_eval: $tasks"
    set +e
    CUDA_VISIBLE_DEVICES="" \
    "$LM_EVAL" \
        --model local-completions \
        --tasks "$tasks" \
        $limit_flag \
        --model_args "model=${model_name},base_url=http://localhost:${server_port}/v1/completions,num_concurrent=${NUM_CONCURRENT},tokenized_requests=False" \
        --output_path "$result_dir" \
        2>&1 | tee -a "$BATCH_LOG_FILE"
    local eval_exit=${PIPESTATUS[0]}
    set -e

    if [ $eval_exit -ne 0 ]; then
        log_message "lm_eval failed (exit: $eval_exit)"
    else
        log_message "lm_eval completed"
    fi

    stop_server "$server_pid" "SGLang server"
    return $eval_exit
}

# =============================================================================
# Preflight
# =============================================================================

if [ ! -f "$PYTHON" ]; then
    echo "ERROR: conda env '$CONDA_ENV_NAME' not found (expected Python at $PYTHON)."
    echo "       Run: bash setup_env.sh"
    exit 1
fi

# =============================================================================
# Main — sequential execution
# =============================================================================

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] MLA KV Cache Rotation Evaluation"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configs: ${#MODEL_CONFIGS[@]} entry(s)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =========================================="
echo ""

OVERALL_EXIT=0
N=${#MODEL_CONFIGS[@]}

for i in "${!MODEL_CONFIGS[@]}"; do
    config="${MODEL_CONFIGS[$i]}"
    IFS='|' read -r mode quant_kv quant_type quant_bits rotate_k rotate_order \
                    model_name gpu_devices tp_size tasks <<< "$config"
    # Strip whitespace
    mode="${mode// /}"; quant_kv="${quant_kv// /}"; quant_type="${quant_type// /}"
    quant_bits="${quant_bits// /}"; rotate_k="${rotate_k// /}"; rotate_order="${rotate_order// /}"
    model_name="${model_name// /}"; gpu_devices="${gpu_devices// /}"; tp_size="${tp_size// /}"
    tasks="${tasks// /}"

    server_port=$((BASE_PORT + i))
    model_short="$(extract_model_short_name "$model_name")"
    label="[$((i+1))/${N}] ${model_short} mode=${mode} (gpu=$gpu_devices)"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label"
    wait_for_gpus_free "$gpu_devices" "$label"

    eval_single_config "$mode" "$quant_kv" "$quant_type" "$quant_bits" \
                       "$rotate_k" "$rotate_order" "$model_name" \
                       "$gpu_devices" "$tp_size" "$tasks" "$server_port"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: $label"
        OVERALL_EXIT=1
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] OK: $label"
    fi

    # Cooldown between configs
    if [ $((i + 1)) -lt "$N" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Cooldown 30s..."
        sleep 30
    fi
done

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done. Exit: $OVERALL_EXIT"
exit $OVERALL_EXIT
