#!/bin/bash
# Downloaded snapshot tarball -> monitor HTML, in one call.
#
#   scripts/diagnostics/snapshot_to_html.sh <snapshot.tar.gz> [OUT.html] [WORKDIR]
#
# Example (the filename really does contain a space and parentheses):
#   scripts/diagnostics/snapshot_to_html.sh \
#       "gf_prod_6mo_v8_4gpu_snapshot.tar (13).gz" ~/monitor_6mo.html
#
# It extracts the tar, finds the run directory and the LIVE store inside it,
# builds the truth set if one is missing, and renders the page. Everything it
# does by hand below is a trap that produced a wrong analysis at least once:
#
#  * `--iteration` defaults to 78 in build_truth.py. On a young store that
#    reads an UNWRITTEN row and yields all-inf SNRs with no error at all, so
#    the iteration is computed here from the store's own `iteration` attr.
#  * `--kappa-out` defaults to the CWD, where it clobbers the LAT-root
#    3-month kappa cache. Both outputs go INTO the run directory.
#  * gf_monitor_gen.py reads AND REWRITES gf_arm_<tag>.npz in the working
#    directory, so the page is rendered from a scratch dir, never the repo
#    root.
#  * A run dir can hold more than one .h5 -- a tar refresh cannot DELETE, so
#    an older-format store survives beside the live one, plus there may be an
#    ~800-byte stub. The live one is picked by size and its iteration attr,
#    and the choice is printed.
#  * Concurrent interpreters have hard-crashed the 8 GB laptop. INTERLOCK=1
#    serialises the two heavy steps against any other python; it is OFF by
#    default for interactive use and should be set by UNATTENDED callers.
#
# Env overrides: ITERATION (row to build truth at), FLO/FHI (band, Hz),
# CATALOGUE (the GB catalogue hdf5, or the directory holding it -- the
# mojito brick cache lives somewhere different on every machine, so this is
# the knob to set when build_truth cannot find it; MOJITO_CAT and
# MOJITO_CACHE_DIR are honoured too and need no passthrough -- gf_monitor_gen
# reads the same env chain and derives its L1-brick base dir from it, so on
# a machine whose cache is not at the laptop default ~/.mojito_cache/... the
# monitor's residual-spectrum and data/template/residual panels ONLY render
# when one of these is set; without them, the DTR block silently degrades to
# three MISSING entries and the page loses those panels),
# PY (interpreter), SKIP_TRUTH=1 (reuse whatever truth npz is present),
# INTERLOCK=1 (wait for any other python to exit before each heavy step;
# default off -- see the note above).
set -euo pipefail

TAR=${1:?usage: snapshot_to_html.sh <snapshot.tar.gz> [OUT.html] [WORKDIR]}
OUT=${2:-}
WORK=${3:-$(mktemp -d "${TMPDIR:-/tmp}/gfmon.XXXXXX")}
PY=${PY:-python}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# INTERLOCK=1 serialises against any other python on the machine. DEFAULT
# OFF, because a person running this from a terminal already knows what
# else is running and should not be made to wait on their own editor's
# language server. It exists for UNATTENDED callers -- an agent or a batch
# script that cannot see the rest of the machine -- where two concurrent
# interpreters on an 8 GB laptop have hard-crashed the box.
#
# -x ONLY, never -f: `pgrep -f <interpreter path>` matches this script's
# own waiting shell (the path appears in its command line) and deadlocks
# forever. That has happened twice in production.
interlock() {
  [ "${INTERLOCK:-0}" = "1" ] || return 0
  while pgrep -x python >/dev/null || pgrep -x python3.12 >/dev/null; do
    echo "[snap2html] INTERLOCK=1 and another python is running, waiting..." >&2
    sleep 20
  done
}

[ -f "$TAR" ] || { echo "no such tarball: $TAR" >&2; exit 1; }
mkdir -p "$WORK/extract"
echo "[snap2html] tar mtime: $(date -r "$TAR" '+%Y-%m-%d %H:%M:%S')"
echo "[snap2html] extracting -> $WORK/extract"
tar -xzf "$TAR" -C "$WORK/extract"

# The archive lays out shared/data/global_fit_output/<run>/ ; take the
# deepest directory that actually holds a store rather than assuming a name.
RUN=$(find "$WORK/extract" -type d -path '*global_fit_output/*' -maxdepth 6 \
        -exec sh -c 'ls "$1"/*.h5 >/dev/null 2>&1' _ {} \; -print | head -1)
[ -n "$RUN" ] || { echo "no run directory with an .h5 inside $TAR" >&2; exit 1; }
echo "[snap2html] run dir: $RUN"

# Live store: biggest non-backup, non-stub .h5, tie-broken by mtime.
STORE=$(ls -S "$RUN"/*.h5 2>/dev/null | grep -v -e backup -e midit | head -1)
[ -n "$STORE" ] || { echo "no usable .h5 in $RUN" >&2; exit 1; }

interlock
read -r ITER_ATTR < <("$PY" - "$STORE" <<'PY'
import sys, h5py
# `iteration` lives on the global_fit GROUP, not the file root -- the root
# carries only extract bookkeeping (e.g. extract_torn_reads). Same shape as
# noise_model_identity, which is likewise a group rather than a root attr;
# assuming the root is how a reader silently gets nothing and falls back.
with h5py.File(sys.argv[1], "r") as f:
    it = f["global_fit"].attrs.get("iteration") if "global_fit" in f else None
    if it is None:
        it = f.attrs.get("iteration", -1)
    print(int(it))
PY
)
echo "[snap2html] store: $(basename "$STORE")  iteration attr: $ITER_ATTR"
[ "$ITER_ATTR" -gt 1 ] || { echo "store has no usable iteration attr" >&2; exit 1; }

# Rows are 0..iteration-1, so the LAST row is iteration-1 and we build one
# before it: sub-backends routinely lag the main backend by a row mid-flush.
IT=${ITERATION:-$((ITER_ATTR - 2))}
echo "[snap2html] building truth at iteration $IT (override with ITERATION=)"

TRUTH="$RUN/gb_truth_3to21.npz"
if [ "${SKIP_TRUTH:-0}" = "1" ] && [ -f "$TRUTH" ]; then
  echo "[snap2html] SKIP_TRUTH=1 and a truth set exists -- reusing it"
else
  interlock
  "$PY" "$HERE/build_truth.py" "$STORE" \
      --iteration "$IT" \
      --out       "$TRUTH" \
      --kappa-out "$RUN/kappa_grid.npz" \
      ${FLO:+--flo "$FLO"} ${FHI:+--fhi "$FHI"} \
      ${CATALOGUE:+--catalogue "$CATALOGUE"}
fi

# Rendered FROM the scratch dir: the generator rewrites gf_arm_<tag>.npz in
# whatever directory it is run from.
OUT=${OUT:-$WORK/$(basename "$RUN").html}
mkdir -p "$(dirname "$OUT")"
interlock
( cd "$WORK" && "$PY" "$HERE/gf_monitor_gen.py" "$RUN" "$OUT" )

echo
echo "[snap2html] PAGE: $OUT"
echo "[snap2html] run dir kept at: $RUN"
echo "[snap2html] arm caches written in: $WORK"
