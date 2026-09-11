#!/bin/bash
# Rewind a v8 10-walker store to the last iteration BEFORE the 2026-09-09
# relaunches (3mo: job 465 -> iteration 151, 1yr: job 466 -> iteration 29)
# and clear every sidecar that would otherwise undo the rewind. Run on the
# cluster with the job CANCELLED (scancel) and the store closed.
#
#   bash scripts/fstat_proposal/rewind_to_465.sh 3mo          # dry run (reports only)
#   bash scripts/fstat_proposal/rewind_to_465.sh 3mo --apply  # do it
#   bash scripts/fstat_proposal/rewind_to_465.sh 1yr --apply  # optional 1yr twin
#
# Then resubmit with the job-465 SCIENCE configuration:
#   sbatch --export=ALL,GB_SCIENCE_465=1 scripts/fstat_proposal/submit_gf_3mo_v8_10walkers.sh
# (the submit scripts carry an opt-in block that reverts every 467/469/471
# knob when GB_SCIENCE_465=1; see the block just above their mpiexec line).
#
# What this does (3mo; the 1yr numbers in brackets):
#   1. reset_recipe_stage.py STORE gb_search --iteration 151 [29] --apply
#      -- re-opens gb_search (3mo completed it at it 175 and moved to full_pe)
#      and moves global_fit.attrs["iteration"] back, so the next resume
#      starts from row 150 [28] = the state job 467 [468] resumed from (its
#      after-recipe-setup lnL matched that stored row to 0.1). Every
#      per-iteration dataset (chains, inds, band caps, band_temps, the
#      band-shutoff valve record) rewinds with the one counter.
#   2. gf_prod_*_testing_midit_checkpoint.pkl -> moved aside. It holds the
#      mid-iteration state of ~it 204 [41]; a checkpoint NEWER than the store
#      is what resume prefers, so left in place it would silently re-apply
#      the post-rewind state.
#   3. gf_prod_*_testing_running_backup_copy.h5 -> REWOUND IN PLACE (same
#      reset as the store) and left beside it, so a torn-store self-heal
#      promotes a copy that is already at the rewound iteration. It used to
#      be moved aside, which left the 2026-09-10 torn 3mo store with
#      nothing to self-heal from. A backup that is itself unreadable is
#      moved aside.
#   4. gb_fstat_fit/shared/epoch_0004, epoch_0005 -> moved aside (3mo only).
#      The move loads the LATEST epoch dir; those two were fitted at it 170
#      and 204 against the degraded residual. epoch_0003 is what the run was
#      using at it 150. (1yr has only epoch_0000; nothing to move.)
# Nothing is deleted: everything goes to STORE_DIR/rewind_<stamp>/.
set -euo pipefail
RUN="${1:?usage: rewind_to_465.sh 3mo|1yr [--apply]}"
APPLY="${2:-}"
case "$RUN" in
  3mo) DIR=/shared/data/global_fit_output/gf_prod_3mo_v8_10walkers; BASE=gf_prod_3mo_testing; IT=151; EPOCHS="epoch_0004 epoch_0005"; JOBNAME=gf3mo_v8 ;;
  1yr) DIR=/shared/data/global_fit_output/gf_prod_1yr_v8_10walkers; BASE=gf_prod_1yr_testing; IT=29;  EPOCHS="";                      JOBNAME=gf1yr_v8 ;;
  *) echo "unknown run $RUN"; exit 2 ;;
esac
STORE="$DIR/$BASE.h5"
STAMP=$(date +%Y%m%d_%H%M%S)
ASIDE="$DIR/rewind_$STAMP"
LAT="${LAT_ROOT:-$HOME/lisa-analysis-tools}"

echo "== rewind $RUN: $STORE -> iteration $IT (resume from row $((IT-1))) apply=${APPLY:-no}"
# ONE-WRITER-PER-STORE GUARD. ``lsof`` on the head node cannot see a
# compute node's open handle, so it passed while a job was still writing
# the store (2026-09-10: the rewind ran under a live job, the store was
# torn, and the moved-aside backup left nothing for the resume self-heal
# to promote). Ask Slurm instead: any RUNNING or PENDING job of this user
# with the run's job name blocks the rewind. Pending counts too -- it would
# start writing the moment the rewind finished.
if command -v squeue >/dev/null 2>&1; then
  LIVE=$(squeue -h -u "$USER" -n "$JOBNAME" -o "%i %T %M" 2>/dev/null || true)
  if [ -n "$LIVE" ]; then
    echo "JOB(S) NAMED $JOBNAME ARE RUNNING/PENDING -- scancel them first, then re-run:"; echo "$LIVE"; exit 3
  fi
fi
if lsof "$STORE" >/dev/null 2>&1; then echo "STORE IS OPEN locally -- scancel first"; exit 3; fi

# 1. counter + stage (dry run unless --apply)
python "$LAT/scripts/fstat_proposal/reset_recipe_stage.py" "$STORE" gb_search --iteration "$IT" ${APPLY:+--apply}

# 2. mid-iteration checkpoint -> aside (a checkpoint newer than the store
#    is what resume prefers; it would re-apply the post-rewind state).
f="$DIR/${BASE}_midit_checkpoint.pkl"
if [ -e "$f" ]; then
  echo "   sidecar: $f -> $ASIDE/"
  [ -n "$APPLY" ] && mkdir -p "$ASIDE" && mv "$f" "$ASIDE/"
fi
# 3. running backup -> REWOUND IN PLACE, never moved aside (2026-09-10
#    lesson: the 3mo store was torn and the resume self-heal had nothing to
#    promote because this script had moved the backup away). Rewinding the
#    backup to the same iteration makes it safe to keep beside the store:
#    if the primary is unreadable the self-heal promotes a copy that is
#    already at iteration $IT. A backup that cannot be rewound (itself
#    torn) is moved aside instead.
b="$DIR/${BASE}_running_backup_copy.h5"
if [ -e "$b" ]; then
  echo "   backup: rewinding $b in place"
  if ! python "$LAT/scripts/fstat_proposal/reset_recipe_stage.py" "$b" gb_search --iteration "$IT" ${APPLY:+--apply}; then
    echo "   backup unreadable -> $ASIDE/"
    [ -n "$APPLY" ] && mkdir -p "$ASIDE" && mv "$b" "$ASIDE/"
  fi
fi
for e in $EPOCHS; do
  d="$DIR/gb_fstat_fit/shared/$e"
  if [ -d "$d" ]; then
    echo "   epoch:   $d -> $ASIDE/"
    [ -n "$APPLY" ] && mkdir -p "$ASIDE" && mv "$d" "$ASIDE/"
  fi
done
echo "   latest F-stat epoch now: $(ls -d "$DIR"/gb_fstat_fit/shared/epoch_* 2>/dev/null | sort | tail -1)"

# 5. verify
python - "$STORE" "$IT" <<'EOF'
import sys, h5py, numpy as np
p, it = sys.argv[1], int(sys.argv[2])
with h5py.File(p, "r") as f:
    g = f["global_fit"]
    print(f"   store iteration attr = {int(g.attrs['iteration'])} (want {it})")
    ll = g["log_like"][it-1, 0, 0, :]
    print(f"   resume row {it-1} cold lnL median {np.median(ll):.1f}")
    inds = g["inds/gb"][it-1, 0, 0]
    print(f"   resume row gb leaves per walker: {inds.sum(axis=-1).tolist()}")
    for k in g["recipe"]:
        print(f"   recipe/{k}: status={bool(g['recipe'][k].attrs['status'])}")
EOF
[ -z "$APPLY" ] && echo "== DRY RUN: nothing written. Re-run with --apply."
