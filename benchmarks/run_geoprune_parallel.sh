#!/usr/bin/env bash
# Drive run_geoprune_bench.sh in parallel: each of the three Qwen VL models
# runs on its own GPU (default 4 / 5 / 6). Each per-model job sequentially
# benchmarks baseline -> 50% prune on the same card so memory peaks once.
#
# Usage:
#   bash benchmarks/run_geoprune_parallel.sh
#   GPUS=4,5,6 bash benchmarks/run_geoprune_parallel.sh
#   NUM_PROMPTS=100 bash benchmarks/run_geoprune_parallel.sh
#
# Streaming logs are written to ${OUT_DIR}/parallel_logs/<tag>.log; tail
# them with:
#   tail -F geoprune_bench/parallel_logs/qwen25_vl_7b.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Comma-separated GPU ids, one per model in order qwen25, qwen3, qwen35.
GPUS="${GPUS:-4,5,6}"
IFS=',' read -ra GPU_LIST <<<"${GPUS}"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/geoprune_bench}"
PARALLEL_LOG_DIR="${OUT_DIR}/parallel_logs"
mkdir -p "${PARALLEL_LOG_DIR}"

MODELS=(qwen25 qwen3 qwen35)
TAGS=(qwen25_vl_7b qwen3_vl_8b qwen35_vl_9b)

# Allocate non-overlapping ports per model to avoid collisions.
BASE_PORTS=(9100 9200 9300)

PIDS=()
for i in 0 1 2; do
  model="${MODELS[$i]}"
  tag="${TAGS[$i]}"
  gpu="${GPU_LIST[$i]}"
  port_base="${BASE_PORTS[$i]}"
  log="${PARALLEL_LOG_DIR}/${tag}.log"

  echo "[launch] ${tag} on GPU ${gpu}, port_base=${port_base}, log=${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    MODELS_ONLY="${model}" \
    PORT_BASE="${port_base}" \
    OUT_DIR="${OUT_DIR}" \
    nohup bash "${SCRIPT_DIR}/run_geoprune_bench.sh" \
      > "${log}" 2>&1 &
  PIDS+=("$!")
  sleep 1
done

echo
echo "Launched ${#PIDS[@]} parallel benchmark jobs: ${PIDS[*]}"
echo "Tail logs with:"
for i in 0 1 2; do
  echo "  tail -F ${PARALLEL_LOG_DIR}/${TAGS[$i]}.log"
done

# Wait for all jobs to finish, propagate non-zero exit codes.
failures=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[WARN] pid ${pid} exited non-zero"
    failures=$((failures + 1))
  fi
done

if (( failures > 0 )); then
  echo "[done] ${failures} job(s) failed; check the per-job logs."
  exit 1
fi
echo "[done] All jobs finished successfully. Results: ${OUT_DIR}/*.client.json"
