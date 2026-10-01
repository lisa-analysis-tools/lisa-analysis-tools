#!/bin/bash
#SBATCH --account=nbody_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=debug_iacc
#SBATCH --job-name=cd1l_probe
#SBATCH --time=00:29:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64v4
#SBATCH --output=/data/nbody/majoburo/cd1l_pe/%x_%A_%a.log
#SBATCH --array=0-3
# One-step proposal probe: acceptance and ESJD of the best global Gaussian
# proposal applied to N exact posterior draws from a finished runs/ chain, in
# the t_SSB (TDET=0) or t_det (TDET=1) parametrisation. ~10 min per row; use
# N >= 2048 or the MC error dominates. IDS/TD override the default grid.
set -uo pipefail
REPO=${REPO:-/home/bustam1/lisastack_pr81/repos/LISAanalysistools}   # this checkout
source "$REPO/scripts/mbh/slurm/cd1l_env.sh"
IDS=(${IDS:-2 2 14 14}); TD=(${TD:-0 1 0 1})
export MBHB_ID=${IDS[$SLURM_ARRAY_TASK_ID]} CD1L_TDET=${TD[$SLURM_ARRAY_TASK_ID]}
export CD1L_PROBE_RECENTRE=0 CD1L_EPOCH_K=-1 CD1L_SAMPLER=mc CD1L_ROULET_SPINS=1
export CD1L_NWALKERS=32 CD1L_NTEMPS=5 CD1L_PROPOSAL_PROBE=1
export CD1L_PROBE_N=${CD1L_PROBE_N:-2048} CD1L_PROBE_F=${CD1L_PROBE_F:-0.1,0.3,1.0}
echo "###### id$MBHB_ID TDET=$CD1L_TDET N=$CD1L_PROBE_N ######"
CD1L_OUT=$CD1L_ROOT/probe_scratch python -u scripts/mbh/cd1l_pe.py
