#!/bin/bash
#SBATCH --account=nbody_acc
#SBATCH --partition=batch_gpu
#SBATCH --qos=nbody_account_batch_gpu
#SBATCH --job-name=cd1l_post1d_tdet
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=64G
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --constraint=x86-64v4
#SBATCH --output=/data/nbody/majoburo/cd1l_pe/%x_%A_%a.log
#SBATCH --array=0-1
# t_det A/B: ids 2 and 14 exactly as the t_SSB campaign in runs/ (cold start,
# adaptive Fisher kernel, 32w x 5T x 5000) except CD1L_TDET=1. Compare
# matched-length tau against runs/roulet_spins. Uses the code-default move mix.
set -uo pipefail
REPO=${REPO:-/home/bustam1/lisastack_pr81/repos/LISAanalysistools}   # this checkout
source "$REPO/scripts/mbh/slurm/cd1l_env.sh"
IDS=(2 14)
export MBHB_ID=${IDS[$SLURM_ARRAY_TASK_ID]} CD1L_EPOCH_K=-1
export CD1L_NWALKERS=32 CD1L_NTEMPS=5 CD1L_NSTEPS=5000 CD1L_INIT_SCALE=1e-4
export CD1L_SAMPLER=mc CD1L_ROULET_SPINS=1 CD1L_TDET=1
export CD1L_OUT=$CD1L_ROOT/runs_tdet
if [ -f "$CD1L_OUT/roulet_spins/cd1l_mbh_id${MBHB_ID}_post1d_mc.h5" ]; then echo "id$MBHB_ID: chain exists, refusing to overwrite"; exit 1; fi
python scripts/mbh/cd1l_pe.py
