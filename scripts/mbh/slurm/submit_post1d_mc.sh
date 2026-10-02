#!/bin/bash
#SBATCH --account=nbody_acc
#SBATCH --partition=batch_gpu
#SBATCH --qos=nbody_account_batch_gpu
#SBATCH --job-name=cd1l_post1d_mc
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=10 --mem=64G
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --constraint=x86-64v4
#SBATCH --output=/data/nbody/majoburo/cd1l_pe/%x_%A_%a.log
#SBATCH --array=0-19
# Production campaign: all 20 CD1-L MBHBs, post-merger (epoch k from the task
# file), dt=10 s / order 8, 32 walkers x 5 temps x 5000 steps, roulet spins.
set -uo pipefail
REPO=${REPO:-/home/bustam1/lisastack_pr81/repos/LISAanalysistools}   # this checkout
source "$REPO/scripts/mbh/slurm/cd1l_env.sh"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CD1L_ROOT/runs/epoch_tasks_post1d.txt")
export MBHB_ID=$(echo $LINE | cut -d' ' -f1)
export CD1L_EPOCH_K=$(echo $LINE | cut -d' ' -f2)
# One temperature, 160 walkers: the fixed x10 ladder (T = 1..1e4) swapped into
# the cold chain on 0.2% of walker-steps (the posterior is unimodal post-merger),
# and 160 x 1 costs the same per step as 32 x 5, so this is 5x the cold walkers.
export CD1L_NWALKERS=${CD1L_NWALKERS:-160} CD1L_NTEMPS=${CD1L_NTEMPS:-1}
export CD1L_INIT_SCALE=${CD1L_INIT_SCALE:-1e-4}
export CD1L_SAMPLER=mc CD1L_ROULET_SPINS=1
# Move mix: Gaussian 0.5 / Stretch 0.5. At equilibrium they are comparable
# (one-step probe, t_det: tie on id2, Gaussian 1.9x on id14; 160x1 chain, t_SSB:
# Stretch 1.5x), and Stretch gains from a large, well-spread ensemble. Sky-partner
# Gibbs auto-drops when all partners are dead (every post-merger source).
export CD1L_STRETCH_WEIGHT=${CD1L_STRETCH_WEIGHT:-0.5}
# Response order 16: order 8 at dt=10 leaves a log L ripple locked to t_SSB
# mod 10 s (id2: first-harmonic amplitude 0.49 over 2048 posterior draws);
# order 16 removes it (0.055, cf. 0.04 at dt=2.5) for ~14% more per step.
export CD1L_ORDER=${CD1L_ORDER:-16}
# Sample t_det (arrival at the constellation centre) instead of t_SSB: an
# exact reparametrisation that removes the Doppler-delay ridge. Own output
# dir: t_SSB chains cannot be resumed under it (config hash differs).
export CD1L_TDET=${CD1L_TDET:-1}
if [ "$CD1L_TDET" = 1 ]; then export CD1L_OUT=${CD1L_OUT:-$CD1L_ROOT/runs_tdet_160x1}; fi
# Resume-aware budget: cd1l_pe.py runs CD1L_NSTEPS MORE steps, so a
# continuation after a walltime kill must ask only for what is left.
TARGET=${CD1L_TARGET_STEPS:-5000}
FP=${CD1L_OUT:-$CD1L_ROOT/runs}/roulet_spins/cd1l_mbh_id${MBHB_ID}_post1d_mc.h5
DONE=0
if [ -f "$FP" ]; then
  DONE=$(python -c "import h5py; print(int(h5py.File('$FP','r').attrs.get('cd1l_completed', 0)))")
fi
export CD1L_NSTEPS=$((TARGET - DONE))
if [ "$CD1L_NSTEPS" -le 0 ]; then echo "id$MBHB_ID: $DONE >= $TARGET steps, nothing to do"; exit 0; fi
echo "=== task ${SLURM_ARRAY_TASK_ID}: id${MBHB_ID} k=${CD1L_EPOCH_K} TDET=${CD1L_TDET} order=${CD1L_ORDER} steps=${DONE}+${CD1L_NSTEPS} stretch=${CD1L_STRETCH_WEIGHT} out=${CD1L_OUT:-$CD1L_ROOT/runs} ==="
python scripts/mbh/cd1l_pe.py
