#!/bin/bash
# ============================================================================
# MBH batched, windowed likelihood -- GPU speed + batching benchmark (ONE GPU).
#
# Runs scripts/mbh/mbh_batched_gpu_benchmark.py on the production 6-month grid
# (dt 2.5 s, Nf 1440 x Nt 4320, EDGE_CROP_WAVELETS 60): the stock per-row path
# at response order {8, 30} x phentax T {30.4 d default, 90 d}, then
# MBHBatchedLikeMove.compute_like at B = 1,2,4,8,16,24,32 (s/row, stage split,
# JIT first call, CuPy/JAX/device memory, OOM, accuracy vs the stock
# cross-check generator). Markdown table in the log, everything in the JSON.
#
# PRECONDITION: the cluster checkout /shared/home/mlkatz1/lisa-analysis-tools
# must be on dev at or after the 2026-09-30 cd1l-merge merge (MBHBatchedLikeMove,
# MBHWindowedWDMSignalGen, WindowedGridAlignedMBHWaveform, utils/stagetimer.py,
# this script) with Eryn dev >= e448a17 (PR #50). LISAResponse.cu changed in
# that merge: rebuild the LAT GPU backend before submitting.
#
# The --output directory must exist before sbatch (slurm does not create it):
#   mkdir -p /shared/data/global_fit_output/mbh_benchmark
#   sbatch scripts/mbh/submit_mbh_batched_gpu_benchmark.sh
# Extra benchmark flags ride through BENCH_ARGS, e.g.
#   BENCH_ARGS="--source-id 17 --batch-sizes 16,24,32,48" sbatch ...
#   BENCH_ARGS="--orbits-file /shared/data/mojito_cache/.../MBHB/L1/<file>.h5" sbatch ...
# ============================================================================

#SBATCH --job-name=mbhbench
#SBATCH --partition=gpu-80-spot
#SBATCH --gres=gpu:1              # ONE GPU (benchmark is single-device)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/shared/data/global_fit_output/mbh_benchmark/mbhbench_%j.log

set -euo pipefail

# ---- environment (as submit_gf_6mo.sh) --------------------------------------
source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools

OUT_DIR=/shared/data/global_fit_output/mbh_benchmark
mkdir -p "${OUT_DIR}"
JOB_TAG=${SLURM_JOB_ID:-manual_$(date +%s)}

# ---- GPU telemetry (1 s nvidia-smi sampler; per-job CSV next to the JSON) ---
GPU_LOG=${OUT_DIR}/gpu_util_${JOB_TAG}.csv
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
  --format=csv,noheader,nounits -lms 1000 > "${GPU_LOG}" &
GPU_SMI_PID=$!
trap 'kill ${GPU_SMI_PID} 2>/dev/null || true' EXIT

# ---- threading policy (MPI-only, no OMP) ------------------------------------
export OMP_NUM_THREADS=1
# JAX allocates on demand (lisatools.detector sets the same); otherwise it
# preallocates 75% of the device and the memory columns are meaningless.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

GPU_BACKEND=${GPU_BACKEND:-cuda13x}

nvidia-smi
git log -1 --oneline || true
git status --short | head -40 || true

python scripts/mbh/mbh_batched_gpu_benchmark.py \
  --backend "${GPU_BACKEND}" \
  --out-dir "${OUT_DIR}" \
  --tag "${JOB_TAG}" \
  ${BENCH_ARGS:-}
