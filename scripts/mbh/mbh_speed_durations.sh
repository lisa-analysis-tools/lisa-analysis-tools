#!/bin/bash
# MBH batched windowed likelihood vs the stock per-row path on one GPU, on the MERGER WINDOW --
# the MBH twin of scripts/emri/emri_speed_durations.sh and scripts/sobbh/sobbh_speed_durations.sh,
# WITHOUT their 180 / 360 / 720-d duration sweep: an MBH template is always the 90 d before / 10 d
# after its merger (the batched window), whatever the data length, so one merger-centred grid is
# the test. The grid: WINDOW_DAYS (120 d; Nt = 24 x days one-hour layers, Nf 1440, dt 2.5 s) with
# the merger MERGER_AT_DAYS (100 d) in -- the batched kept box (90 + 10 d + 1-d margins), its 4-d
# segment pads and the 60-layer edge crop fit inside, no clamping (mbh_cd1l_campaign.py's
# ``centered`` placement). With a brick a merger near either end of its file shifts the grid
# into the file (the batched window then clamps to the active box, as production clamps at the
# data edges). Every step applies the EMRI harness's check_window (the window end + 4e4 s inside
# the orbits' ltt table).
#
# Unit tests first (GPU == CPU checks skip without a GPU; the ...Phentax... classes hold the time):
#       python -m unittest tests.test_mbh_windowed_signal_gen tests.test_mbh_batched_move \
#           tests.test_mbh_harness_noise -v
#
# Steps (the active box = BAND x [EDGE, Nt - EDGE) layers):
#   1. speed -- mbh_batched_gpu_benchmark.py, source SRC: the stock per-row path
#      (MBH_LIKELIHOOD=full: response order 8 at phentax T = 30.44 d, the production default,
#      and T = 90 d) against MBHBatchedLikeMove.compute_like at batch_max_size B in CHUNKS:
#      s/row (warm), the device-synced stage split (pol / resp / xform / like|score), the
#      first-call JIT, CuPy / JAX / device memory, OOM, expose/fold s/walker, the accuracy
#      guard (max |dlogL| vs the stock cross-check generator), the source's optimal SNR. A
#      markdown table in the log, one JSON line per configuration (+ a summary line) in
#      ${OUT}/speed.jsonl.
#   2. accuracy -- mbh_batched_accuracy.py on the SAME grid (it imports the benchmark's
#      builders), per source in SRCS: truth + near-truth rows, the batched template against
#      the 90-d snapped stock template (the move's own cross-check reference; GATES) and the
#      production stock default (30.44 d; information): dlogL, mismatch, kept-layer max
#      relative error, the reference's power outside the kept box, ||delta||, SNRs,
#      mm_vs_production; one "[accuracy] src <id> ...: PASS|FAIL" line per source (|dlogL|
#      <= 0.5 and mismatch < 1e-6 vs the 90-d snapped stock on truth/near rows); JSON lines
#      in ${OUT}/accuracy.jsonl.
#   3. data (only with MOJITO_LIGHT_PATH) -- mbh_cd1l_campaign.py --window centered (the same
#      WINDOW_DAYS / MERGER_AT_DAYS) against the REAL mojito MBHB stream of every source whose
#      brick is present (default: all of them, from its --dry-run): prod, prod90 and batched
#      (DATA_TEMPLATES) scored against the data: logL, mismatch vs data, SNR ratio, per-band
#      residual SNR, the template-vs-template pairs; ${OUT}/data.jsonl and its summary tables
#      in ${OUT}/data_summary.md. A missing brick is skipped with a message, never a failure.
#   Speed + accuracy summary tables close the run.
#
#   salloc -p gpu-80-spot --gres=gpu:1 --cpus-per-task=4 --mem=64G -t 4:00:00
#   source /shared/home/mlkatz1/envs/gf_env/bin/activate; cd /shared/home/mlkatz1/lisa-analysis-tools
#       bash scripts/mbh/mbh_speed_durations.sh           # bricks via MOJITO_DATA_PATH, no data step
#       MOJITO_LIGHT_PATH=/shared/data/mojito_cache bash scripts/mbh/mbh_speed_durations.sh
#   /shared/data/mojito_cache is the flat mojito light v1.0.0 tree on the cluster (catalogues/,
#   data/MBHB/L1/; the 6-month launcher's MOJITO_DATA_PATH / MOJITO_INFO_PATH); the laptop's
#   nested ~/.mojito_cache/brickmarket/mojito_light_v1_0_0 layout is accepted too.
#   NEEDS origin/dev >= ddaad46f: the MBH scripts import RunBox / check_window from
#   scripts/emri/emri_batch_speed.py.
#
#   knobs (env): BACKEND (cuda13x), WINDOW_DAYS (120), MERGER_AT_DAYS (100), CHUNKS
#   ("1,2,4,8,16,32"), SRC (16; the speed source), SRCS ("16 17"; the accuracy sources -- ids
#   other than the hardcoded 16, 17 need CATALOGUE), CATALOGUE (mojito MBHB catalogue h5),
#   ORBITS (auto: the source's own mojito brick when found -- its L1 orbits and its time frame
#   -- else equal-arm; equal-arm; l1 = the brick or fail; or a brick path), L1_DIR (the only
#   place searched when set), MOJITO_DATA_PATH (/shared/data/mojito_cache: where ORBITS=auto
#   finds bricks), BAND ("2.5e-4,2.5e-2" Hz), EDGE (60 layers), FOREGROUND (on: XYZ2 scirdv1 +
#   the FittedHyperbolicTangentGalacticForeground at the grid's Tobs; off: the instrument
#   alone), REPS (3: timed stock rows per stock configuration, after one warm-up row),
#   MOJITO_LIGHT_PATH (unset: no data step), DATA_SRCS ("admitted" = every source with a brick,
#   or a list of ids), DATA_TEMPLATES ("prod,prod90,batched"), STEPS ("speed,accuracy,data"),
#   BENCH_ARGS (extra flags to the benchmark AND the accuracy script, e.g. "--window-pad-days
#   8"; the accuracy script ignores the speed-sweep ones), ACC_ARGS (accuracy only, e.g.
#   "--near 0"), DATA_ARGS (data step only), OUT (mbh_speed_durations_<date>).
#
# SAME as the EMRI / SOBBH harnesses: Nf, dt; the noise everywhere (the EMRI RunBox's
# XYZ2SensitivityMatrix(dom, model="scirdv1", stochastic_params=(Tobs,)) at the grid's Tobs,
# imported, FOREGROUND on/off); the box (BAND 0.25-25 mHz = the v9 6-month launcher's MIN_FREQ /
# MAX_FREQ, EDGE = EDGE_CROP_WAVELETS = 60); check_window; the brick search order (L1_DIR alone;
# else MOJITO_LIGHT_PATH/data/MBHB/L1, MOJITO_DATA_PATH, MOJITO_INFO_PATH, the catalogue's root,
# each directly then recursively); the brick's L1 orbits (frame icrs, windowed ltt read); the
# data accessor (MojitoL1File(fp).tdis.xyz_doppler[i0:i1]); the JSON names (data_snr, brick,
# tobs_s, foreground (bool), noise, snr_<template>, data_<template> = {mm, logL, snr_ratio, snr},
# mm_vs_production). DELIBERATELY DIFFERENT (every JSON line's differs_from_emri,
# scripts/mbh/mbh_harness.py): one merger-centred grid instead of the duration sweep; the
# batched template on its per-leaf sub-box window; two stock references (production 30.44 d and
# the 90-d snapped cross-check, which gates); band residual SNRs and kept-box extras; injected
# noiseless templates in speed / accuracy, the mojito stream in the separate data step.
#
# Memory (120-d grid, Nt 2880; N = 4.1 M samples): a stock row holds the full-grid TDI output
# (3 x N float64: 100 MB) plus the full-grid WDM transform's work arrays (a few times that) and
# the phentax strain (3.1 M samples at T = 90 d, 1.05 M at 30.44 d). A batched row's window
# lattice is ~3.8 M samples (2642-layer segment + 15000 s lead + 600 s tail: 91 MB of TDI per row
# before the phentax / response intermediates; job 677 measured the B sweep on an H100). The
# four walker containers hold 3 x 179 x 2760 residuals and a 3 x 3 x 179 x 2760 PSD each (+ its
# inverse): ~0.2 GB per container. The data step reads ~0.1 GB of stream and a ~0.2 GB ltt slice
# per source (the campaign measured 4.2 GB peak host RSS per source on the laptop CPU).
set -euo pipefail

BACKEND=${BACKEND:-cuda13x}
WINDOW_DAYS=${WINDOW_DAYS:-120}
MERGER_AT_DAYS=${MERGER_AT_DAYS:-100}
CHUNKS=${CHUNKS:-1,2,4,8,16,32}
SRC=${SRC:-16}
SRCS=${SRCS:-16 17}
CATALOGUE=${CATALOGUE:-}
ORBITS=${ORBITS:-auto}
L1_DIR=${L1_DIR:-}
export MOJITO_DATA_PATH=${MOJITO_DATA_PATH:-/shared/data/mojito_cache}
BAND=${BAND:-2.5e-4,2.5e-2}
EDGE=${EDGE:-60}
FOREGROUND=${FOREGROUND:-on}
REPS=${REPS:-3}
DATA_SRCS=${DATA_SRCS:-admitted}
DATA_TEMPLATES=${DATA_TEMPLATES:-prod,prod90,batched}
STEPS=${STEPS:-speed,accuracy,data}
BENCH_ARGS=${BENCH_ARGS:-}
ACC_ARGS=${ACC_ARGS:-}
DATA_ARGS=${DATA_ARGS:-}
OUT=${OUT:-mbh_speed_durations_$(date +%Y%m%d_%H%M)}
NF=1440
DT=2.5
mkdir -p "${OUT}"

# ---- knob checks: loud, before any GPU time is spent --------------------------------------
case ",${STEPS}," in *,speed,*|*,accuracy,*|*,data,*) ;; *) echo "[mbh] STEPS=${STEPS}: none of speed,accuracy,data"; exit 2 ;; esac
case "${FOREGROUND}" in on|off) ;; *) echo "[mbh] FOREGROUND=${FOREGROUND}: use on or off"; exit 2 ;; esac
[[ "${WINDOW_DAYS}" =~ ^[0-9]+$ ]] || { echo "[mbh] WINDOW_DAYS=${WINDOW_DAYS}: a whole number of days"; exit 2; }
grep -q "^def check_window" scripts/emri/emri_batch_speed.py && grep -q "^class RunBox" scripts/emri/emri_batch_speed.py || {
  echo "[mbh] scripts/emri/emri_batch_speed.py has no RunBox / check_window: this checkout predates"
  echo "      origin/dev ddaad46f (the EMRI harness helpers the MBH scripts import); merge origin/dev"
  exit 2
}
IFS=, read -r FMIN FMAX <<< "${BAND}"
[ -n "${FMIN}" ] && [ -n "${FMAX}" ] || { echo "[mbh] BAND=${BAND}: want 'f_lo,f_hi' in Hz"; exit 2; }
orb_args=()
case "${ORBITS}" in
  equal-arm|auto|l1) orb_args=(--orbits "${ORBITS}") ;;
  *) [ -f "${ORBITS}" ] || { echo "[mbh] ORBITS=${ORBITS}: not equal-arm/auto/l1 and no such file"; exit 2; }
     orb_args=(--orbits-file "${ORBITS}") ;;
esac
l1_args=()
[ -z "${L1_DIR}" ] || l1_args=(--l1-dir "${L1_DIR}")
cat_args=()
if [ -n "${CATALOGUE}" ]; then
  [ -f "${CATALOGUE}" ] || { echo "[mbh] CATALOGUE=${CATALOGUE} not found"; exit 2; }
  cat_args=(--catalogue "${CATALOGUE}")
fi
if [ -n "${MOJITO_LIGHT_PATH:-}" ]; then
  # the flat cluster tree, or the laptop's nested brickmarket layout (the monitor's rule)
  if [ ! -d "${MOJITO_LIGHT_PATH}/catalogues" ] && [ -d "${MOJITO_LIGHT_PATH}/brickmarket/mojito_light_v1_0_0/catalogues" ]; then
    MOJITO_LIGHT_PATH="${MOJITO_LIGHT_PATH}/brickmarket/mojito_light_v1_0_0"
  fi
  [ -d "${MOJITO_LIGHT_PATH}/catalogues" ] && [ -d "${MOJITO_LIGHT_PATH}/data" ] || {
    echo "[mbh] MOJITO_LIGHT_PATH=${MOJITO_LIGHT_PATH} holds no catalogues/ + data/ (mojito light v1.0.0 tree)"
    exit 2
  }
  export MOJITO_LIGHT_PATH
fi
quiet='lisaconstants\|warnings.warn\|hwloc\|FutureWarning\|DeprecationWarning\|cupy._util.experimental\|VACUUM_PERMEABILITY\|You may also open'
fails=0
fail() { echo "[mbh] FAILED: $1"; fails=$((fails + 1)); }
step() { case ",${STEPS}," in *,"$1",*) return 0 ;; *) return 1 ;; esac; }

nt=$((WINDOW_DAYS * 24))
act=$((nt - 2 * EDGE))
echo "[mbh] backend ${BACKEND}; merger-centred grid ${WINDOW_DAYS} d (Nt ${nt}, Tobs $((NF * nt * 5 / 2)) s), merger" \
     "${MERGER_AT_DAYS} d in; Nf ${NF} dt ${DT} s; band ${FMIN}-${FMAX} Hz, active t [${EDGE}, $((nt - EDGE))) = ${act} layers;" \
     "foreground ${FOREGROUND}; orbits ${ORBITS}; MOJITO_LIGHT_PATH ${MOJITO_LIGHT_PATH:-unset}; -> ${OUT}/"
# where ORBITS=auto and the data step look first (the campaign / benchmark then also search
# MOJITO_DATA_PATH, MOJITO_INFO_PATH recursively): say up front whether any brick is there
for d in ${L1_DIR:-"${MOJITO_LIGHT_PATH:-${MOJITO_DATA_PATH}}/data/MBHB/L1"}; do
  nb=$(ls "${d}"/MBHB_*_L1_source*.h5 2>/dev/null | wc -l | tr -d ' ')
  echo "[mbh] ${nb} MBHB L1 brick(s) in ${d}$([ "${nb}" = 0 ] && echo ' -- ORBITS=auto falls back to equal-arm unless one is found deeper; the data step needs them')"
done
grid=(--backend "${BACKEND}" --nf "${NF}" --dt "${DT}" --nt "${nt}" --merger-day "${MERGER_AT_DAYS}"
      --min-freq "${FMIN}" --max-freq "${FMAX}" --edge-crop "${EDGE}" --foreground "${FOREGROUND}"
      "${orb_args[@]}" ${l1_args[@]+"${l1_args[@]}"} ${cat_args[@]+"${cat_args[@]}"})

if step speed; then
  log="${OUT}/speed.log"
  echo "[mbh] speed, src ${SRC}, stock o8 T 30.44 d + 90 d vs batched B ${CHUNKS} -> ${log}"
  # shellcheck disable=SC2086  # BENCH_ARGS is word-split on purpose
  python scripts/mbh/mbh_batched_gpu_benchmark.py "${grid[@]}" --source-id "${SRC}" \
    --batch-sizes "${CHUNKS}" --stock-orders 8 --stock-T-days default,window --stock-rows "${REPS}" \
    --out-dir "${OUT}" --tag "src${SRC}" --jsonl "${OUT}/speed.jsonl" --strict ${BENCH_ARGS} \
    2>&1 | grep -v "${quiet}" > "${log}" || fail "${log}"
  grep "^|\|\[speed\]\|^\[bench\]\|NOT ADMITTED" "${log}" || true
fi

if step accuracy; then
  for src in ${SRCS//,/ }; do
    log="${OUT}/accuracy_src${src}.log"
    echo "[mbh] accuracy, src ${src} -> ${log}"
    # shellcheck disable=SC2086
    python scripts/mbh/mbh_batched_accuracy.py "${grid[@]}" --source-id "${src}" \
      --jsonl "${OUT}/accuracy.jsonl" --strict ${BENCH_ARGS} ${ACC_ARGS} \
      2>&1 | grep -v "${quiet}" > "${log}" || fail "${log}"
    grep "^\[accuracy\]" "${log}" | cut -c1-260 || true
  done
fi

if step data; then
  if [ -z "${MOJITO_LIGHT_PATH:-}" ]; then
    echo "[mbh] data step skipped (MOJITO_LIGHT_PATH unset)"
  else
    camp=(--window centered --window-days "${WINDOW_DAYS}" --merger-at-days "${MERGER_AT_DAYS}"
          --edge-crop "${EDGE}" --min-freq "${FMIN}" --max-freq "${FMAX}" ${l1_args[@]+"${l1_args[@]}"})
    if [ "${DATA_SRCS}" = admitted ]; then
      dry="${OUT}/data_dryrun.log"
      if python scripts/mbh/mbh_cd1l_campaign.py --src all --dry-run "${camp[@]}" > "${dry}" 2>&1; then
        ids=$(sed -n "s/^\[admitted\] centered l1=\([^ ]*\) no_l1=.*/\1/p" "${dry}")
        noids=$(sed -n "s/^\[admitted\] centered l1=[^ ]* no_l1=\(.*\)/\1/p" "${dry}")
        echo "[mbh] data, sources with a brick: ${ids:--}; without one (skipped): ${noids:--}"
        [ "${ids}" != "-" ] || ids=""
      else
        fail "${dry}"
        grep -v "${quiet}" "${dry}" | tail -5
        ids=""
      fi
    else
      ids="${DATA_SRCS}"
    fi
    for src in ${ids//,/ }; do
      log="${OUT}/data_src${src}.log"
      set +e
      # shellcheck disable=SC2086
      python scripts/mbh/mbh_cd1l_campaign.py --src "${src}" "${camp[@]}" --templates "${DATA_TEMPLATES}" \
        --backend "${BACKEND}" --foreground "${FOREGROUND}" --out "${OUT}/data.jsonl" ${DATA_ARGS} \
        2>&1 | grep -v "${quiet}" > "${log}"
      rc=${PIPESTATUS[0]}
      set -e
      if [ "${rc}" = 3 ]; then
        echo "[mbh] data, src ${src}: no MBHB L1 brick -- skipped"
      elif [ "${rc}" != 0 ]; then
        fail "${log}"
      fi
      grep "^\[campaign\]" "${log}" | grep -v "batched geometry" | cut -c1-220 || true
    done
  fi
fi

if [ -s "${OUT}/data.jsonl" ]; then
  python scripts/mbh/mbh_cd1l_campaign_summary.py "${OUT}/data.jsonl" > "${OUT}/data_summary.md" \
    || fail "data summary"
  echo "[mbh] data summary -> ${OUT}/data_summary.md"
  cat "${OUT}/data_summary.md"
fi

# ---- summary tables ------------------------------------------------------------------------
python - "${OUT}" <<'PY' || fail "summary"
import json
import os
import sys

out = sys.argv[1]


def lines(name):
    path = os.path.join(out, name)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


speed = [r for r in lines("speed.jsonl") if r.get("kind") == "summary"]
if speed:
    bs = sorted({int(b) for r in speed for b in r["batched_s_per_row"]})
    stock = sorted({k for r in speed for k in r["stock_s_per_row"]})
    print("\n[mbh] speed, s/row (warm; '-' = OOM / error / not run)")
    print("| grid days | src | SNR | orbits | " + " | ".join(stock) + " | " + " | ".join(f"B={b}" for b in bs) + " | best |")
    print("|" + "---|" * (5 + len(stock) + len(bs)))

    def f(v):
        return "-" if v is None else f"{v:.4g}"

    for r in sorted(speed, key=lambda r: (r["days"], r["source_id"])):
        best = r.get("best_batched") or {}
        print(f"| {r['days']:g} | {r['source_id']} | {r.get('snr_stock90') or 0:.1f} | {r['orbits'].split('(')[0]} | "
              + " | ".join(f(r["stock_s_per_row"].get(k)) for k in stock) + " | "
              + " | ".join(f(r["batched_s_per_row"].get(str(b))) for b in bs)
              + f" | {f(best.get('s_per_row'))} @ B={best.get('B', '-')} |")
acc = [r for r in lines("accuracy.jsonl") if r.get("kind") == "accuracy_source"]
if acc:
    print("\n[mbh] accuracy vs the 90-d snapped stock (gate: |dlogL| <= 0.5, mm < 1e-6; truth + near rows)")
    print("| grid days | src | SNR | verdict | max abs dlogL | max mm | mm_vs_production | orbits | noise |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(acc, key=lambda r: (r["days"], r["source_id"])):
        if r.get("status") != "ok":
            print(f"| {r['days']:g} | {r['source_id']} | - | SKIPPED | - | - | - | - | {r.get('reason', '-')[:80]} |")
            continue
        mvp = r.get("mm_vs_production")
        print(f"| {r['days']:g} | {r['source_id']} | {r['snr_stock90']:.1f} | {'PASS' if r['passed'] else 'FAIL'} | "
              f"{r['max_abs_dlogL_stock90']:.3e} | {r['max_mismatch_stock90']:.3e} | "
              f"{'-' if mvp is None else format(mvp, '.3e')} | {r['orbits'].split('(')[0]} | {r.get('noise', '-')} |")
PY

echo "[mbh] done: ${OUT}/ (${fails} failed step(s))"
[ "${fails}" -eq 0 ]
