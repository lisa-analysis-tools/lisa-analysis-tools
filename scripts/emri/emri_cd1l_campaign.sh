#!/usr/bin/env bash
# Serial driver for scripts/emri/emri_cd1l_campaign.py: every CD1L EMRI (0-7) at 6mo, then 24mo.
# One python process per (source, duration): each holds one FEW generator (~6 GB transient).
#
#   MOJITO_LIGHT_PATH=/path/to/mojito_light_v1_0_0 \
#   EMRI_WDM_TABLE=/path/to/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5 \
#   bash scripts/emri/emri_cd1l_campaign.sh
#
# Env: SOURCES (default "0 1 2 3 4 5 6 7"), DURATIONS (default "6mo 24mo"),
#      TEMPLATES (default prod,tof,direct; direct is dropped when EMRI_WDM_TABLE is unset),
#      THRESH (default 1e-3), OUT_DIR (default emri_cd1l_campaign_out), PYTHON (default python),
#      plus the knobs in emri_cd1l_campaign.py (TOF_FINE_DT, MODE_BATCH, RSS_LIMIT_GB, ...).
# Already-finished (src, duration) pairs are skipped (their .done marker exists): rerun resumes.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SOURCES=${SOURCES:-"0 1 2 3 4 5 6 7"}
DURATIONS=${DURATIONS:-"6mo 24mo"}
TEMPLATES=${TEMPLATES:-prod,tof,direct}
THRESH=${THRESH:-1e-3}
OUT_DIR=${OUT_DIR:-emri_cd1l_campaign_out}
PYTHON=${PYTHON:-python}
: "${MOJITO_LIGHT_PATH:?set MOJITO_LIGHT_PATH to the mojito light v1.0.0 directory}"
export MOJITO_LIGHT_PATH
TABLE_ARGS=()
if [ -n "${EMRI_WDM_TABLE:-}" ]; then
  TABLE_ARGS=(--table "$EMRI_WDM_TABLE")
else
  TEMPLATES=$(echo "$TEMPLATES" | tr ',' '\n' | grep -v '^direct$' | paste -sd, -)
  echo "EMRI_WDM_TABLE unset: running templates $TEMPLATES"
fi
mkdir -p "$OUT_DIR"
for dur in $DURATIONS; do
  for src in $SOURCES; do
    tag="src${src}_${dur}"
    [ -e "$OUT_DIR/$tag.done" ] && { echo "skip $tag (done)"; continue; }
    echo "=== $tag $(date)"
    start=$(date +%s)
    "$PYTHON" "$HERE/emri_cd1l_campaign.py" --src "$src" --duration "$dur" --templates "$TEMPLATES" \
      --thresh "$THRESH" "${TABLE_ARGS[@]}" --out "$OUT_DIR/results.jsonl" > "$OUT_DIR/$tag.log" 2>&1
    rc=$?
    echo "    exit $rc after $(( $(date +%s) - start )) s (log $OUT_DIR/$tag.log)"
    [ $rc -eq 0 ] && touch "$OUT_DIR/$tag.done"
  done
done
"$PYTHON" "$HERE/emri_cd1l_campaign_summary.py" "$OUT_DIR/results.jsonl" | tee "$OUT_DIR/summary.md"
