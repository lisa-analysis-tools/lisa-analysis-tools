#!/usr/bin/env bash
# GB / VGB speed + accuracy over observation durations -- the twin of
# scripts/sobbh/sobbh_speed_durations.sh, scripts/emri/emri_speed_durations.sh
# and scripts/mbh/mbh_speed_durations.sh.
#
# Per duration (DAYS):
#   1. speed   -- every engine, warm wall time per scoring call vs rows per
#                 call on the run grid (Nf 1440 / dt 2.5; SPEED_ARGS=--laptop
#                 for Nf 180 / dt 20), sig-het setup cost per reference too.
#                 -> speed.jsonl
#   2. gate    -- accuracy vs a dense TD->WDM truth (CPU, Nf 180 / dt 20):
#                 template mm / norm ratio, delta-vs-delta tier pass for every
#                 engine. -> gate.jsonl
#   3. mojito  -- top-f VGBs (VGB brick) and galaxy GBs (GB brick) at their
#                 catalogue parameters vs the data window (in-slab neighbours
#                 subtracted): snr, snr_det, logL, mm vs data, dlogL vs
#                 production. -> mojito.jsonl
# Then: summary tables + scaling / accuracy plots (gb_speed_plots.py).
#
# Engines: chunked (production exact), sighet_reim (v5, GBGPU >= c81d8bb
# default), sighet_ampph (v5 pre-c81d8bb), lookup (reference-free direct WDM,
# Python prototype -- its speed is a Python number, not a kernel's).
#
# Cluster:  BACKEND=cuda12x TABLE=$PWD/wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5 \
#             MOJITO_LIGHT_PATH=/shared/data/mojito_cache bash scripts/gb/gb_speed_durations.sh
# Laptop:   DAYS=90 ROWS=4,16 SPEED_ARGS=--laptop GATE_ARGS="--gate-f0 2,16 \
#             --gate-cosi 0.02,0.8 --n-cand 1" VGB_TOP=2 GB_TOP=2 \
#             bash scripts/gb/gb_speed_durations.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}
BACKEND=${BACKEND:-cpu}
DAYS=${DAYS:-"180 360 720"}
ROWS=${ROWS:-8,64,512,4096}
MAX_SLOTS=${MAX_SLOTS:-256}
ENGINES=${ENGINES:-chunked,sighet_reim,sighet_ampph,lookup}
REPS=${REPS:-3}
EDGE=${EDGE:-60}
FOREGROUND=${FOREGROUND:-on}
SPEED_ARGS=${SPEED_ARGS:-}
GATE=${GATE:-1}
GATE_ARGS=${GATE_ARGS:-}
MOJITO=${MOJITO:-auto}          # auto: run if bricks are found; 0: skip; 1: require
VGB_TOP=${VGB_TOP:-6}
GB_TOP=${GB_TOP:-6}
MOJITO_ARGS=${MOJITO_ARGS:-}
# lookup table (any n_ref table at the 3600-s layer; the EMRI / SOBBH run tables serve):
TABLE=${TABLE:-${GB_LOOKUP_TABLE_PATH:-${SOBBH_LOOKUP_TABLE_PATH:-${EMRI_DIRECT_TABLE:-}}}}
OUT=${OUT:-gb_speed_durations_$(date +%Y%m%d_%H%M)}

mkdir -p "$OUT"
LAPTOP_TABLE=/Users/mkatz/Research/lisa_sprint_2026/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5
case ",$ENGINES," in
  *,lookup,*)
    if [ -z "$TABLE" ] && [ ! -f "$LAPTOP_TABLE" ]; then
      echo "[gb_speed_durations] ENGINES has lookup but no table: set TABLE= (e.g. the run's"
      echo "  wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5) or drop lookup from ENGINES"
      exit 2
    fi
    if [ -n "$TABLE" ] && [ ! -f "$TABLE" ]; then
      echo "[gb_speed_durations] TABLE=$TABLE does not exist"; exit 2
    fi ;;
esac
TABLE_ARG=()
if [ -n "$TABLE" ]; then TABLE_ARG=(--table "$TABLE"); fi
echo "[gb_speed_durations] OUT=$OUT BACKEND=$BACKEND DAYS='$DAYS' ROWS=$ROWS ENGINES=$ENGINES"
echo "[gb_speed_durations] GBGPU: $($PY -c 'import gbgpu, os, subprocess; d=os.path.dirname(gbgpu.__file__); print(subprocess.run(["git","-C",d,"log","-1","--format=%h %s"],capture_output=True,text=True).stdout.strip())' 2>/dev/null)"
FAILED=()

for D in $DAYS; do
  echo "=== ${D} d: speed"
  $PY "$HERE/gb_speed_accuracy.py" --step speed --days "$D" --backend "$BACKEND" \
      --engines "$ENGINES" --rows "$ROWS" --max-slots "$MAX_SLOTS" --reps "$REPS" \
      --edge "$EDGE" --foreground "$FOREGROUND" ${TABLE_ARG[@]+"${TABLE_ARG[@]}"} $SPEED_ARGS \
      --out "$OUT/speed.jsonl" > "$OUT/speed_${D}d.log" 2>&1 || FAILED+=("speed_${D}d")
  if [ "$GATE" = "1" ]; then
    echo "=== ${D} d: gate"
    $PY "$HERE/gb_speed_accuracy.py" --step gate --days "$D" --engines "$ENGINES" \
        --edge "$EDGE" --foreground "$FOREGROUND" ${TABLE_ARG[@]+"${TABLE_ARG[@]}"} $GATE_ARGS \
        --out "$OUT/gate.jsonl" > "$OUT/gate_${D}d.log" 2>&1 || FAILED+=("gate_${D}d")
  fi
  if [ "$MOJITO" != "0" ]; then
    echo "=== ${D} d: mojito"
    $PY "$HERE/gb_speed_accuracy.py" --step mojito --days "$D" --engines "$ENGINES" \
        --edge "$EDGE" --foreground "$FOREGROUND" --vgb-top "$VGB_TOP" --gb-top "$GB_TOP" \
        ${TABLE_ARG[@]+"${TABLE_ARG[@]}"} $SPEED_ARGS $MOJITO_ARGS \
        --out "$OUT/mojito.jsonl" > "$OUT/mojito_${D}d.log" 2>&1 || FAILED+=("mojito_${D}d")
    if [ "$MOJITO" = "1" ] && grep -q "no .* brick found" "$OUT/mojito_${D}d.log"; then
      FAILED+=("mojito_${D}d_no_brick")
    fi
  fi
done

echo "=== summary + plots"
$PY "$HERE/gb_speed_plots.py" "$OUT" 2>&1 | tee "$OUT/summary.txt" || FAILED+=("plots")
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "[gb_speed_durations] FAILED steps: ${FAILED[*]} (see $OUT/*.log)"
  exit 1
fi
echo "[gb_speed_durations] done: $OUT"
