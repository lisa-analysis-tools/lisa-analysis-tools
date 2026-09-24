# SOURCE THIS AFTER EVERY salloc:   source scripts/diagnostics/gate_env.sh
#
# The GPU-routing gates (docs/gpu-routing-interactive-4gpu.md) need six
# environment variables, and EVERY ONE OF THEM DIES WITH THE SHELL. `salloc`
# hands you a NEW shell, so a fresh allocation starts with none of them --
# and the way that fails does not point at a missing export:
#
#   UCX ERROR no active messages transport ... different host id ...
#   Abort: MPIDI_OFI_mpi_init_hook: OFI get address vector map failed
#
# which reads as a broken MPI install. It happened three times on 2026-09-23
# and 09-24, twice after re-allocating onto different nodes; the tell is that
# the hostnames in the error differ from the ones in the last run that
# worked. Hence this file: paste one line, not a block.
#
# Sourcing is IDEMPOTENT -- re-source it whenever you are unsure.

# --- the launcher pins ------------------------------------------------------
# The proven set, identical to what the production submit scripts' multi-node
# branch exports. UCX finds no cross-node transport on this cluster (only
# self/sysv/posix/cma, which are shared-memory and correctly refused between
# hosts), so the fabric is pinned to libfabric's tcp provider. TCP is a
# CORRECTNESS choice; a faster provider is a separate measurement.
export I_MPI_HYDRA_BOOTSTRAP=slurm
export I_MPI_FABRICS=shm:ofi
export FI_PROVIDER=tcp

# `-ppn 1` does not bind on its own under the SLURM bootstrap: hydra honours
# the scheduler's per-node TASK COUNTS over it. SIZE-DEPENDENT -- at 3 tasks
# over 2 nodes block and cyclic agree, so it looks fine; at 5 tasks SLURM's
# block split is 3/2 and build_layout refuses ("3 compute ranks but the
# per-node GPU pool [0, 1] supports at most 2").
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0

# --- the run knobs ----------------------------------------------------------
# GPUS is the PER-NODE pool, not the total. Overridable: a value already in
# the environment wins, so `GPUS=0 source ...` still works for the R=1 arm.
export GPUS=${GPUS:-0,1}

# THE OPT-IN. The unified routing is off by default in the library, so without
# this the gates' new shapes RAISE -- and worse, G5 would silently take the
# legacy equal-band-count split and measure ~2x at R=4 instead of ~4x.
export GF_GPU_ROUTING=${GF_GPU_ROUTING:-1}

# --- report -----------------------------------------------------------------
if [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "[gate_env] WARNING: you EXECUTED this file, so the exports died with"
  echo "[gate_env]   it and your shell has nothing. Run: source $0"
fi

_gate_env_missing=""
for _v in I_MPI_HYDRA_BOOTSTRAP I_MPI_FABRICS FI_PROVIDER \
          I_MPI_JOB_RESPECT_PROCESS_PLACEMENT GPUS GF_GPU_ROUTING; do
  eval "_val=\${$_v:-}"
  [ -n "$_val" ] || _gate_env_missing="$_gate_env_missing $_v"
done
if [ -n "$_gate_env_missing" ]; then
  echo "[gate_env] MISSING:$_gate_env_missing"
else
  echo "[gate_env] ok: fabric=$I_MPI_FABRICS/$FI_PROVIDER" \
       "placement_respect=$I_MPI_JOB_RESPECT_PROCESS_PLACEMENT" \
       "GPUS=$GPUS GF_GPU_ROUTING=$GF_GPU_ROUTING" \
       "nodes=${SLURM_NODELIST:-<no allocation>}"
fi
unset _gate_env_missing _v _val

# GF_LAYOUT_DRY_RUN is deliberately NOT set here: it makes every run stop
# before fit.build(), which is right for G3 and WRONG for G4/G5. Set it per
# command instead -- `GF_LAYOUT_DRY_RUN=1 mpiexec ...`.
