#!/usr/bin/env bash
# Latency benchmark for GeoPrune image-token pruning, built on the official
# `vllm bench serve` CLI with the lmarena-ai/VisionArena-Chat dataset (the
# standard real-world VLM benchmark recommended by the vLLM docs).
#
# For each model the script:
#   1. starts a vLLM OpenAI-compatible server with --image-pruning-rate
#      either unset (baseline) or 0.5 (50% prune),
#   2. waits until /health is ready,
#   3. runs `vllm bench serve --backend openai-chat --dataset-name hf
#      --dataset-path lmarena-ai/VisionArena-Chat` against it,
#   4. dumps the JSON result and tears down the server.
#
# Output: per-model TTFT / TPOT / ITL / throughput, exactly matching the
# table format used in the upstream attention-score pruning PR.
#
# Usage:
#   bash benchmarks/run_geoprune_bench.sh
#
# Override anything via env vars:
#   MODELS_ONLY=qwen25 bash benchmarks/run_geoprune_bench.sh
#   NUM_PROMPTS=500 PRUNE_RATE=0.5 bash benchmarks/run_geoprune_bench.sh
#   DATASET_PATH=lmarena-ai/VisionArena-Chat HF_SPLIT=train ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

QWEN25="${QWEN25:-/loukang_data/models/Qwen/Qwen2.5-VL-7B-Instruct}"
QWEN3="${QWEN3:-/loukang_data/models/Qwen/Qwen3-VL-8B-Instruct}"
QWEN35="${QWEN35:-/loukang_data/models/Qwen/Qwen3.5-9B}"

# Real-world VLM benchmark. Swap for Lin-Chen/ShareGPT4V or yale-nlp/MMVU
# (multi-image) by overriding DATASET_PATH / HF_SPLIT.
DATASET_PATH="${DATASET_PATH:-lmarena-ai/VisionArena-Chat}"
HF_SPLIT="${HF_SPLIT:-train}"

NUM_PROMPTS="${NUM_PROMPTS:-200}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"   # 1 = sequential, clean per-request latency
REQUEST_RATE="${REQUEST_RATE:-inf}"
DTYPE="${DTYPE:-bfloat16}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-16384}"
LIMIT_IMG="${LIMIT_IMG:-4}"
PRUNE_RATE="${PRUNE_RATE:-0.5}"
PORT_BASE="${PORT_BASE:-9100}"
HOST="${HOST:-127.0.0.1}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/geoprune_bench}"
SERVER_LOG_DIR="${OUT_DIR}/server_logs"

mkdir -p "${OUT_DIR}" "${SERVER_LOG_DIR}"

MODELS_ONLY="${MODELS_ONLY:-qwen25,qwen3,qwen35}"

wait_for_ready() {
  local port="$1" timeout="${2:-600}"
  local start now
  start=$(date +%s)
  while true; do
    if curl -sSf "http://${HOST}:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    now=$(date +%s)
    if (( now - start > timeout )); then
      echo "  ERROR: server on port ${port} did not become ready in ${timeout}s" >&2
      return 1
    fi
    sleep 2
  done
}

run_one_pair() {
  local tag="$1" model="$2"
  local baseline_port=$((PORT_BASE))
  local prune_port=$((PORT_BASE + 1))

  for rate_name in baseline prune${PRUNE_RATE/.//}; do
    local port pruning_arg
    local rate_tag
    if [[ "${rate_name}" == "baseline" ]]; then
      port="${baseline_port}"
      pruning_arg=""
      rate_tag="baseline"
    else
      port="${prune_port}"
      pruning_arg="--image-pruning-rate ${PRUNE_RATE}"
      rate_tag="prune${PRUNE_RATE}"
    fi

    local server_log="${SERVER_LOG_DIR}/${tag}_${rate_tag}.server.log"
    local result_json="${OUT_DIR}/${tag}_${rate_tag}.client.json"

    echo
    echo "================================================================="
    echo "  ${tag} | ${rate_tag} | port=${port} | model=${model}"
    echo "================================================================="

    # Launch server in background.
    # shellcheck disable=SC2086
    nohup vllm serve "${model}" \
      --host "${HOST}" --port "${port}" \
      --dtype "${DTYPE}" \
      --tensor-parallel-size "${TP}" \
      --max-model-len "${MAX_LEN}" \
      --limit-mm-per-prompt "{\"image\": ${LIMIT_IMG}}" \
      --no-enable-prefix-caching \
      ${pruning_arg} \
      > "${server_log}" 2>&1 &
    local server_pid=$!
    echo "  server pid=${server_pid} log=${server_log}"

    # Make sure we always tear the server down.
    cleanup() {
      if kill -0 "${server_pid}" 2>/dev/null; then
        echo "  stopping server pid=${server_pid}"
        kill "${server_pid}" 2>/dev/null || true
        # give vLLM a chance to release CUDA cleanly
        for _ in 1 2 3 4 5 6 7 8 9 10; do
          kill -0 "${server_pid}" 2>/dev/null || break
          sleep 1
        done
        kill -9 "${server_pid}" 2>/dev/null || true
      fi
    }
    trap cleanup EXIT INT TERM

    wait_for_ready "${port}" 900

    # Run the official bench-serve client.
    vllm bench serve \
      --backend openai-chat \
      --host "${HOST}" --port "${port}" \
      --model "${model}" \
      --endpoint /v1/chat/completions \
      --dataset-name hf \
      --dataset-path "${DATASET_PATH}" \
      --hf-split "${HF_SPLIT}" \
      --num-prompts "${NUM_PROMPTS}" \
      --max-concurrency "${MAX_CONCURRENCY}" \
      --request-rate "${REQUEST_RATE}" \
      --save-result \
      --save-detailed \
      --result-dir "${OUT_DIR}" \
      --result-filename "$(basename "${result_json}")" \
      --ignore-eos

    cleanup
    trap - EXIT INT TERM
    sleep 5  # let CUDA settle before launching the next server
  done
}

if [[ ",${MODELS_ONLY}," == *",qwen25,"* ]]; then
  run_one_pair qwen25_vl_7b "${QWEN25}"
fi
if [[ ",${MODELS_ONLY}," == *",qwen3,"* ]]; then
  run_one_pair qwen3_vl_8b "${QWEN3}"
fi
if [[ ",${MODELS_ONLY}," == *",qwen35,"* ]]; then
  run_one_pair qwen35_vl_9b "${QWEN35}"
fi

echo
echo "Done."
echo "  Client JSONs : ${OUT_DIR}/*.client.json   (TTFT/TPOT/ITL/throughput)"
echo "  Server logs  : ${SERVER_LOG_DIR}/*.server.log"
echo
echo "Each client JSON contains the fields you need for the paper table:"
echo "  mean_ttft_ms, median_ttft_ms, p99_ttft_ms,"
echo "  mean_tpot_ms, median_tpot_ms, p99_tpot_ms,"
echo "  mean_itl_ms,  median_itl_ms,  p99_itl_ms,"
echo "  request_throughput, output_throughput, total_token_throughput."
