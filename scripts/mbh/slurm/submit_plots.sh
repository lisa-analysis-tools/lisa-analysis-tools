#!/bin/bash
#SBATCH --account=nbody_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=debug_iacc
#SBATCH --job-name=cd1l_plots
#SBATCH --time=00:29:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=x86-64v4
#SBATCH --output=/data/nbody/majoburo/cd1l_pe/%x_%j.log
# Traces, corner and sky plots. R=<runs dir holding roulet_spins/>, P=<plot
# dir>, IDS="2 14"; CD1L_PLOT_STOP=<n> cuts chains to n steps (matched length).
# t_det chains are mapped back to t_SSB via the dvec_id*_mc.npy beside them.
set -uo pipefail
REPO=${REPO:-/home/bustam1/lisastack_pr81/repos/LISAanalysistools}   # this checkout
source "$REPO/scripts/mbh/slurm/cd1l_env.sh"
: "${R:?set R}" "${P:?set P}" "${IDS:?set IDS}"
for i in $IDS; do
  c=$R/roulet_spins/cd1l_mbh_id${i}_post1d_mc.h5
  [ -f "$c" ] || { echo "missing $c"; continue; }
  CD1L_PLOTS=$P CD1L_CHAIN=$c python scripts/mbh/cd1l_plots.py
done
