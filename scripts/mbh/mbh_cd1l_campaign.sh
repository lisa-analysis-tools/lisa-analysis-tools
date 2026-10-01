#!/usr/bin/env bash
# Serial driver for scripts/mbh/mbh_cd1l_campaign.py: every CD1L MBHB (0-19) at 6mo, then centered.
# One python process per (source, window); they run one after another, never in parallel.
#
#   MOJITO_LIGHT_PATH=/path/to/mojito_light_v1_0_0 bash scripts/mbh/mbh_cd1l_campaign.sh
#   MOJITO_LIGHT_PATH=/path/to/mojito_light_v1_0_0 BACKEND=cuda12x OUT_DIR=mbh_cd1l_out \
#       bash scripts/mbh/mbh_cd1l_campaign.sh
#
# Env: SOURCES (default "0 1 ... 19"), WINDOWS (default "6mo centered"; also 24mo),
#      TEMPLATES (default prod,prod90,batched; tof is opt-in and untested), BACKEND (cpu,
#      cuda12x, cuda13x; default cpu), OUT_DIR (default mbh_cd1l_campaign_out), PYTHON
#      (default python), RSS_LIMIT_GB (peak-RSS kill, exit 42; default unset = off),
#      EXTRA_ARGS (appended to every call, e.g. "--window-days 120 --merger-at-days 100"),
#      plus the stock MBH_* knobs read by SourceMBHSettings (MBH_RESPONSE_ORDER, ...).
# A source without an L1 file is skipped with a message. A (src, window) outside the
# production admission rule is recorded (status skipped_outside_window) and marked done.
# Already-finished (src, window) pairs are skipped (their .done marker exists): rerun resumes.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SOURCES=${SOURCES:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"}
WINDOWS=${WINDOWS:-"6mo centered"}
TEMPLATES=${TEMPLATES:-prod,prod90,batched}
BACKEND=${BACKEND:-cpu}
OUT_DIR=${OUT_DIR:-mbh_cd1l_campaign_out}
PYTHON=${PYTHON:-python}
EXTRA_ARGS=${EXTRA_ARGS:-}
: "${MOJITO_LIGHT_PATH:?set MOJITO_LIGHT_PATH to the mojito light v1.0.0 directory}"
export MOJITO_LIGHT_PATH
[ -n "${RSS_LIMIT_GB:-}" ] && export RSS_LIMIT_GB
L1_DIR="$MOJITO_LIGHT_PATH/data/MBHB/L1"
mkdir -p "$OUT_DIR"
for win in $WINDOWS; do
  for src in $SOURCES; do
    tag="src${src}_${win}"
    [ -e "$OUT_DIR/$tag.done" ] && { echo "skip $tag (done)"; continue; }
    if ! ls "$L1_DIR"/MBHB_*source"${src}"_* >/dev/null 2>&1; then
      echo "skip $tag: no L1 file for source $src in $L1_DIR"
      continue
    fi
    echo "=== $tag $(date)"
    start=$(date +%s)
    # shellcheck disable=SC2086  # EXTRA_ARGS is word-split on purpose
    "$PYTHON" "$HERE/mbh_cd1l_campaign.py" --src "$src" --window "$win" --templates "$TEMPLATES" \
      --backend "$BACKEND" --out "$OUT_DIR/results.jsonl" $EXTRA_ARGS > "$OUT_DIR/$tag.log" 2>&1
    rc=$?
    echo "    exit $rc after $(( $(date +%s) - start )) s (log $OUT_DIR/$tag.log)"
    [ $rc -eq 0 ] && touch "$OUT_DIR/$tag.done"
  done
done
if [ -s "$OUT_DIR/results.jsonl" ]; then
  "$PYTHON" "$HERE/mbh_cd1l_campaign_summary.py" "$OUT_DIR/results.jsonl" | tee "$OUT_DIR/summary.md"
else
  echo "no results in $OUT_DIR/results.jsonl"
fi
