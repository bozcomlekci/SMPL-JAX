#!/usr/bin/env bash
set -euo pipefail

# Guarded, serial JAX benchmark runner to avoid host OOM killing VS Code.
# Defaults target GPU0 with a 12 GiB JAX memory cap on a 16 GiB board.

ENV_NAME="${ENV_NAME:-body}"
GPU_INDEX="${GPU_INDEX:-0}"
MEM_FRACTION="${MEM_FRACTION:-0.75}"
MAX_FRAMES="${MAX_FRAMES:-8192}"
SIZES="${SIZES:-1,8,32,128,512,1469,2048,4096,8192}"
REPEATS="${REPEATS:-1}"
WARMUP="${WARMUP:-0}"
SWEEP_REPEATS="${SWEEP_REPEATS:-1}"
SWEEP_WARMUP="${SWEEP_WARMUP:-0}"
OUTPUT_JSON="${OUTPUT_JSON:-benchmarks/results/rtx5080/jax_only_mem_under12gb.json}"
LOG_DIR="${LOG_DIR:-benchmarks/logs}"
SYSTEMD_MEMORY_MAX_GIB="${SYSTEMD_MEMORY_MAX_GIB:-22}"
SYSTEMD_SWAP_MAX_GIB="${SYSTEMD_SWAP_MAX_GIB:-8}"
SYSTEMD_TASKS_MAX="${SYSTEMD_TASKS_MAX:-256}"

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${OUTPUT_JSON}")"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${LOG_DIR}/jax_guarded_${TIMESTAMP}.log"
GPU_LOG="${LOG_DIR}/jax_guarded_${TIMESTAMP}_gpu.csv"
PROC_LOG="${LOG_DIR}/jax_guarded_${TIMESTAMP}_gpu_procs.csv"
VMSTAT_LOG="${LOG_DIR}/jax_guarded_${TIMESTAMP}_vmstat.log"
LOCK_FILE="${LOG_DIR}/jax_guarded.lock"

if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required but not available." >&2
  exit 2
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "ERROR: Another guarded benchmark run is active (lock: ${LOCK_FILE})." >&2
  exit 90
fi

# Mirror everything to console and file.
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "timestamp=${TIMESTAMP}"
echo "run_log=${RUN_LOG}"
echo "gpu_log=${GPU_LOG}"
echo "proc_log=${PROC_LOG}"
echo "vmstat_log=${VMSTAT_LOG}"
echo "out_json=${OUTPUT_JSON}"
echo "policy: serial run, gpu${GPU_INDEX} only, mem cap 12GB (fraction=${MEM_FRACTION})"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is required for this guarded runner." >&2
  exit 3
fi

# Refuse to start if another benchmark runtime is already active.
STUCK_PY="$(ps -eo pid=,args= | awk '/benchmarks\/benchmark_runtime.py/ && $0 !~ /awk/ {print}')"
if [[ -n "${STUCK_PY}" ]]; then
  echo "ERROR: Existing benchmark_runtime.py process(es) detected; refusing to run:" >&2
  echo "${STUCK_PY}" >&2
  exit 91
fi

nvidia-smi -i "${GPU_INDEX}" --query-gpu=name,memory.total,memory.used --format=csv,noheader

GPU_MON_PID=""
PROC_MON_PID=""
VMSTAT_PID=""

cleanup() {
  set +e
  if [[ -n "${GPU_MON_PID}" ]]; then
    kill "${GPU_MON_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${PROC_MON_PID}" ]]; then
    kill "${PROC_MON_PID}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${VMSTAT_PID}" ]]; then
    kill "${VMSTAT_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

nvidia-smi -i "${GPU_INDEX}" \
  --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv -lms 1000 >"${GPU_LOG}" 2>/dev/null &
GPU_MON_PID="$!"

nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory \
  --format=csv -lms 1000 >"${PROC_LOG}" 2>/dev/null &
PROC_MON_PID="$!"

vmstat 1 >"${VMSTAT_LOG}" 2>/dev/null &
VMSTAT_PID="$!"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export JAX_PLATFORMS="cuda"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}"
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

BENCH_CMD=(
  conda run -n "${ENV_NAME}" python benchmarks/benchmark_runtime.py
  --method-filter jax_smplx,jax_smpl
  --jax-platform cuda
  --warmup "${WARMUP}"
  --repeats "${REPEATS}"
  --batch-size-sweep
  --batch-size-sweep-sizes "${SIZES}"
  --batch-size-sweep-repeats "${SWEEP_REPEATS}"
  --batch-size-sweep-warmup "${SWEEP_WARMUP}"
  --max-frames "${MAX_FRAMES}"
  --tile-sequence-to-max-frames
  --no-include-smplxpp
  --no-include-torchure
  --json-out "${OUTPUT_JSON}"
)

echo "command=${BENCH_CMD[*]}"

USE_SYSTEMD=0
if command -v systemd-run >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  USE_SYSTEMD=1
fi

set +e
if [[ "${USE_SYSTEMD}" -eq 1 ]]; then
  echo "launch_mode=systemd-run scope (MemoryMax=${SYSTEMD_MEMORY_MAX_GIB}G, MemorySwapMax=${SYSTEMD_SWAP_MAX_GIB}G, TasksMax=${SYSTEMD_TASKS_MAX})"
  systemd-run --user --scope --quiet \
    -p "MemoryMax=${SYSTEMD_MEMORY_MAX_GIB}G" \
    -p "MemorySwapMax=${SYSTEMD_SWAP_MAX_GIB}G" \
    -p "TasksMax=${SYSTEMD_TASKS_MAX}" \
    "${BENCH_CMD[@]}"
  EXIT_CODE=$?
else
  echo "launch_mode=direct (systemd-run unavailable); continuing without cgroup caps"
  "${BENCH_CMD[@]}"
  EXIT_CODE=$?
fi
set -e

echo "exit_code=${EXIT_CODE}"

LEFTOVER="$(ps -eo pid=,args= | awk '/benchmarks\/benchmark_runtime.py/ && $0 !~ /awk/ {print}')"
if [[ -n "${LEFTOVER}" ]]; then
  echo "warning=leftover benchmark process detected after run"
  echo "${LEFTOVER}"
fi

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  echo "status=failed"
  exit "${EXIT_CODE}"
fi

if [[ ! -s "${OUTPUT_JSON}" ]]; then
  echo "ERROR: Expected output JSON missing or empty: ${OUTPUT_JSON}" >&2
  exit 92
fi

echo "status=ok"
