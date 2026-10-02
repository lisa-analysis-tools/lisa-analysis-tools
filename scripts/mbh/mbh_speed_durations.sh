#!/bin/bash
# MBH batched windowed likelihood vs the stock per-row path at 6, 12 and 24 months on one GPU --
# the MBH twin of scripts/emri/emri_speed_durations.sh and scripts/sobbh/sobbh_speed_durations.sh.
# Production convention: 15552000 s per 6 months -> DAYS 180 / 360 / 720 (Nt = days * 24 one-hour
# layers, Nf 1440, dt 2.5 s, Tobs = Nf * Nt * dt). Every step applies the EMRI harness's
# check_window (scripts/emri/emri_batch_speed.py): the window end + the production wrapper's
# 4e4 s must lie inside the orbits' light-travel-time table -- the packaged equal-arm orbits end
# REF + 697.9 d, the 731-d mojito bricks' data and ltts REF + 730.5 d -- so 720 d needs the
# source's L1 brick (ORBITS=auto/l1) unless an equal-arm, merger-mid-grid window happens to fit
# (mojito MBHB 16 does; 17 does not). A window that does not fit is REFUSED, loudly.
#
# Unit tests first (GPU == CPU checks skip without a GPU; the ...Phentax... classes hold the time):
#       python -m unittest tests.test_mbh_windowed_signal_gen tests.test_mbh_batched_move \
#           tests.test_mbh_harness_noise -v
#
# Per duration (the active box = BAND x [EDGE, Nt - EDGE) layers):
#   1. speed -- mbh_batched_gpu_benchmark.py on that grid, source SRC: the stock per-row path
#      (MBH_LIKELIHOOD=full: response order 8 at phentax T = 30.44 d, the production default,
#      and T = 90 d) against MBHBatchedLikeMove.compute_like at batch_max_size B in CHUNKS:
#      s/row (warm), the device-synced stage split (pol / resp / xform / like|score), the
#      first-call JIT, CuPy / JAX / device memory, OOM, expose/fold s/walker, the accuracy
#      guard (max |dlogL| vs the stock cross-check generator), the source's optimal SNR. A
#      markdown table per duration in the log, one JSON line per configuration (+ a summary
#      line) in ${OUT}/speed.jsonl. The point across durations: the batched cost is window-
#      limited (~constant), the stock cost grows with the grid.
#   2. accuracy -- mbh_batched_accuracy.py on the SAME grid (it imports the benchmark's
#      builders), per source in SRCS: truth + near-truth rows, the batched template against
#      the 90-d snapped stock template (the move's own cross-check reference; GATES) and the
#      production stock default (30.44 d; information): dlogL, mismatch, kept-layer max
#      relative error, the reference's power outside the kept box, ||delta||, SNRs,
#      mm_vs_production; one "[accuracy] src <id> <days> d ...: PASS|FAIL|SKIPPED" line per
#      source against the acceptance used so far (|dlogL| <= 0.5 and mismatch < 1e-6 vs the
#      90-d snapped stock on truth/near rows); JSON lines in ${OUT}/accuracy.jsonl.
#   3. data (only with MOJITO_LIGHT_PATH) -- mbh_cd1l_campaign.py against the REAL mojito MBHB
#      stream of every source whose brick is present (default: every source the window
#      admits, from its --dry-run): prod, prod90 and batched (DATA_TEMPLATES) scored against
#      the data on the "<days>d" window (START_OFFSET_S after the brick start): logL,
#      mismatch vs data, SNR ratio, per-band residual SNR, the template-vs-template pairs;
#      ${OUT}/data.jsonl, its summary tables in ${OUT}/data_summary.md. A missing brick is
#      skipped with a message, never a failure.
#   A cross-duration speed + accuracy table closes the run.
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
#   knobs (env): BACKEND (cuda13x), DAYS ("180 360 720"), CHUNKS ("1,2,4,8,16,32"), SRC (16;
#   the speed source), SRCS ("16 17"; the accuracy sources -- ids other than the hardcoded 16,
#   17 need CATALOGUE), CATALOGUE (mojito MBHB catalogue h5), ORBITS (auto: the source's own
#   mojito brick when found -- its L1 orbits and its time frame -- else equal-arm; equal-arm;
#   l1 = the brick or fail; or a brick path), L1_DIR (the only place searched when set),
#   MOJITO_DATA_PATH (/shared/data/mojito_cache: where ORBITS=auto finds bricks),
#   START_OFFSET_S (5e4: window start after the brick start), BAND ("2.5e-4,2.5e-2" Hz),
#   EDGE (60 layers), FOREGROUND (on: XYZ2 scirdv1 + the FittedHyperbolicTangentGalacticForeground
#   at the window's Tobs; off: the instrument alone), REPS (3: timed stock rows per stock
#   configuration, after one warm-up row), MOJITO_LIGHT_PATH (unset: no data step),
#   DATA_SRCS ("admitted" or a list of ids), DATA_TEMPLATES ("prod,prod90,batched"), STEPS
#   ("speed,accuracy,data"), BENCH_ARGS (extra flags to the benchmark AND the accuracy script,
#   e.g. "--window-pad-days 8"; the accuracy script ignores the speed-sweep ones), ACC_ARGS
#   (accuracy only, e.g. "--near 0"), DATA_ARGS (data step only), OUT
#   (mbh_speed_durations_<date>).
#
# SAME as the EMRI / SOBBH harnesses: DAYS and the grid; the noise everywhere (the EMRI
# RunBox's XYZ2SensitivityMatrix(dom, model="scirdv1", stochastic_params=(Tobs,)) at the
# WINDOW's Tobs, imported, FOREGROUND on/off); the box (BAND 0.25-25 mHz = the v9 6-month
# launcher's MIN_FREQ / MAX_FREQ, EDGE = EDGE_CROP_WAVELETS = 60); check_window; the brick
# search order (L1_DIR alone; else MOJITO_LIGHT_PATH/data/MBHB/L1, MOJITO_DATA_PATH,
# MOJITO_INFO_PATH, the catalogue's root, each directly then recursively); the brick's L1
# orbits (frame icrs, windowed ltt read); the window start (START_OFFSET_S after the brick's
# tdis t0); the data accessor (MojitoL1File(fp).tdis.xyz_doppler[i0:i1]); the JSON names
# (data_snr, brick, tobs_s, foreground (bool), noise, snr_<template>, data_<template> =
# {mm, logL, snr_ratio, snr}, mm_vs_production). DELIBERATELY DIFFERENT (every JSON line's
# differs_from_emri, scripts/mbh/mbh_harness.py):
#   * the batched template lives on a merger-centred per-leaf WINDOW (90 d before / 10 d after
#     the merger, 1 d margins, clamped to the active box) computed on a segment 4 d (96 layers)
#     wider each side and cropped back -- a SUB-box of the grid, scored through
#     AnalysisContainer._slice_to_template (never RunBox.score on a full-grid array);
#   * two stock references: production (T 30.44 d, unsnapped epoch, order 8) and the 90-d
#     lattice-snapped stock (the move's cross-check, the one the acceptance gates on);
#   * extras: band residual SNRs, kept-box geometry, outside-box power, prod90 pairs;
#   * production MBH starts the data window AT the file start (offset 0); the harness's
#     START_OFFSET_S = 5e4 s (0.58 d) only changes which mergers are admitted (one in the first
#     0.58 d of the file is out; the admission end moves 0.58 d later);
#   * without a brick the merger is placed mid-grid (an MBH merger must be inside the window;
#     EMRI places at REF + START_OFFSET_S); the speed / accuracy steps score injected noiseless
#     stock truth templates (the brick only supplies orbits + time frame; a merger the window
#     does not admit is SKIPPED in the accuracy step and fails the speed step); the mojito
#     stream is scored in the separate data step (EMRI scores it inside its speed scripts);
#   * REPS counts timed stock rows (each batched B times max(2B, 32) rows).
#
# Memory per duration (6 / 12 / 24 months = Nt 4320 / 8640 / 17280; N = 6.2 / 12.4 / 24.9 M
# samples): a stock row holds the full-grid TDI output (3 x N float64: 149 / 299 / 597 MB)
# plus the full-grid WDM transform's work arrays (a few times that) and the phentax strain
# (3.1 M samples at T = 90 d, 1.05 M at 30.44 d). A batched row's window lattice is ~3.8 M
# samples at EVERY duration (2642-layer segment + 15000 s lead + 600 s tail: 91 MB of TDI per
# row before the phentax / response intermediates; job 677 measured the 6-month sweep on an
# H100). The four walker containers hold 3 x 179 x Nt_active residuals and a 3 x 3 x 179 x
# Nt_active PSD each (+ its inverse): ~0.3 / 0.6 / 1.2 GB per container at 6 / 12 / 24 months.
# With a brick, the host also holds the ltt slice (~0.3 / 0.6 / 1.2 GB; 24 months = nearly the
# whole table). The DATA step at 24 months is BIG: the 720-d stream read on the host (597 MB)
# and its full-grid transform, the ~1.2 GB ltt slice, three full-grid templates per source on
# the device, ~20 admitted sources one process each (tens of minutes); the campaign's CPU
# estimate is 12-20 GB host -- keep --mem 64G and set RSS_LIMIT_GB if the node is shared.
set -euo pipefail

BACKEND=${BACKEND:-cuda13x}
DAYS=${DAYS:-180 360 720}
CHUNKS=${CHUNKS:-1,2,4,8,16,32}
SRC=${SRC:-16}
SRCS=${SRCS:-16 17}
CATALOGUE=${CATALOGUE:-}
ORBITS=${ORBITS:-auto}
L1_DIR=${L1_DIR:-}
export MOJITO_DATA_PATH=${MOJITO_DATA_PATH:-/shared/data/mojito_cache}
START_OFFSET_S=${START_OFFSET_S:-5e4}
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
case ",${STEPS}," in *,speed,*|*,accuracy,*|*,data,*) ;; *) echo "[durations] STEPS=${STEPS}: none of speed,accuracy,data"; exit 2 ;; esac
case "${FOREGROUND}" in on|off) ;; *) echo "[durations] FOREGROUND=${FOREGROUND}: use on or off"; exit 2 ;; esac
grep -q "^def check_window" scripts/emri/emri_batch_speed.py && grep -q "^class RunBox" scripts/emri/emri_batch_speed.py || {
  echo "[durations] scripts/emri/emri_batch_speed.py has no RunBox / check_window: this checkout predates"
  echo "            origin/dev ddaad46f (the EMRI harness helpers the MBH scripts import); merge origin/dev"
  exit 2
}
IFS=, read -r FMIN FMAX <<< "${BAND}"
[ -n "${FMIN}" ] && [ -n "${FMAX}" ] || { echo "[durations] BAND=${BAND}: want 'f_lo,f_hi' in Hz"; exit 2; }
orb_args=()
case "${ORBITS}" in
  equal-arm|auto|l1) orb_args=(--orbits "${ORBITS}") ;;
  *) [ -f "${ORBITS}" ] || { echo "[durations] ORBITS=${ORBITS}: not equal-arm/auto/l1 and no such file"; exit 2; }
     orb_args=(--orbits-file "${ORBITS}") ;;
esac
l1_args=()
[ -z "${L1_DIR}" ] || l1_args=(--l1-dir "${L1_DIR}")
cat_args=()
if [ -n "${CATALOGUE}" ]; then
  [ -f "${CATALOGUE}" ] || { echo "[durations] CATALOGUE=${CATALOGUE} not found"; exit 2; }
  cat_args=(--catalogue "${CATALOGUE}")
fi
if [ -n "${MOJITO_LIGHT_PATH:-}" ]; then
  # the flat cluster tree, or the laptop's nested brickmarket layout (the monitor's rule)
  if [ ! -d "${MOJITO_LIGHT_PATH}/catalogues" ] && [ -d "${MOJITO_LIGHT_PATH}/brickmarket/mojito_light_v1_0_0/catalogues" ]; then
    MOJITO_LIGHT_PATH="${MOJITO_LIGHT_PATH}/brickmarket/mojito_light_v1_0_0"
  fi
  [ -d "${MOJITO_LIGHT_PATH}/catalogues" ] && [ -d "${MOJITO_LIGHT_PATH}/data" ] || {
    echo "[durations] MOJITO_LIGHT_PATH=${MOJITO_LIGHT_PATH} holds no catalogues/ + data/ (mojito light v1.0.0 tree)"
    exit 2
  }
  export MOJITO_LIGHT_PATH
fi
quiet='lisaconstants\|warnings.warn\|hwloc\|FutureWarning\|DeprecationWarning\|cupy._util.experimental\|VACUUM_PERMEABILITY\|You may also open'
fails=0
fail() { echo "[durations] FAILED: $1"; fails=$((fails + 1)); }
step() { case ",${STEPS}," in *,"$1",*) return 0 ;; *) return 1 ;; esac; }

echo "[durations] backend ${BACKEND}; days ${DAYS}; Nf ${NF} dt ${DT} s; band ${FMIN}-${FMAX} Hz, edge ${EDGE} layers;" \
     "foreground ${FOREGROUND}; orbits ${ORBITS}; start offset ${START_OFFSET_S} s; MOJITO_LIGHT_PATH ${MOJITO_LIGHT_PATH:-unset}; -> ${OUT}/"
for days in ${DAYS}; do
  [[ "${days}" =~ ^[0-9]+$ ]] || { echo "[durations] DAYS entry '${days}' is not a whole number of days"; exit 2; }
  nt=$((days * 24))
  if [ $((nt % 2)) -ne 0 ]; then   # WDMSettings needs an even layer count
    echo "[durations] ${days} d: Nt = ${nt} is odd; skipping"
    continue
  fi
  act=$((nt - 2 * EDGE))
  if [ "${act}" -le 0 ]; then
    echo "[durations] ${days} d: EDGE ${EDGE} leaves no active layers of Nt ${nt}; skipping"
    continue
  fi
  # the default window: 90 + 10 + 2 x 1 d = 102 d -> 2450 kept layers (+ 2 x 96 pad layers)
  fit="fits"
  [ "${act}" -ge 2450 ] || fit="is CLAMPED to the active box (the 3-month case)"
  echo "[durations] ${days} d: Nt ${nt} (Tobs $((NF * nt * 5 / 2)) s), active box [${EDGE}, $((nt - EDGE))) = ${act} layers; the default 2450-layer kept box ${fit}"
  grid=(--backend "${BACKEND}" --nf "${NF}" --dt "${DT}" --nt "${nt}" --min-freq "${FMIN}" --max-freq "${FMAX}"
        --edge-crop "${EDGE}" --foreground "${FOREGROUND}" --start-offset-s "${START_OFFSET_S}"
        "${orb_args[@]}" ${l1_args[@]+"${l1_args[@]}"} ${cat_args[@]+"${cat_args[@]}"})

  if step speed; then
    log="${OUT}/speed_${days}d.log"
    echo "[durations] ${days} d: speed, src ${SRC}, stock o8 T 30.44 d + 90 d vs batched B ${CHUNKS} -> ${log}"
    # shellcheck disable=SC2086  # BENCH_ARGS is word-split on purpose
    python scripts/mbh/mbh_batched_gpu_benchmark.py "${grid[@]}" --source-id "${SRC}" \
      --batch-sizes "${CHUNKS}" --stock-orders 8 --stock-T-days default,window --stock-rows "${REPS}" \
      --out-dir "${OUT}" --tag "${days}d_src${SRC}" --jsonl "${OUT}/speed.jsonl" --strict ${BENCH_ARGS} \
      2>&1 | grep -v "${quiet}" > "${log}" || fail "${log}"
    grep "^|\|\[speed\]\|^\[bench\]\|NOT ADMITTED" "${log}" || true
  fi

  if step accuracy; then
    for src in ${SRCS//,/ }; do
      log="${OUT}/accuracy_${days}d_src${src}.log"
      echo "[durations] ${days} d: accuracy, src ${src} -> ${log}"
      # shellcheck disable=SC2086
      python scripts/mbh/mbh_batched_accuracy.py "${grid[@]}" --source-id "${src}" \
        --jsonl "${OUT}/accuracy.jsonl" --strict ${BENCH_ARGS} ${ACC_ARGS} \
        2>&1 | grep -v "${quiet}" > "${log}" || fail "${log}"
      grep "^\[accuracy\]" "${log}" | cut -c1-260 || true
    done
  fi

  if step data; then
    if [ -z "${MOJITO_LIGHT_PATH:-}" ]; then
      echo "[durations] ${days} d: data step skipped (MOJITO_LIGHT_PATH unset)"
    else
      win="${days}d"
      camp=(--window "${win}" --edge-crop "${EDGE}" --min-freq "${FMIN}" --max-freq "${FMAX}"
            --start-offset-s "${START_OFFSET_S}" ${l1_args[@]+"${l1_args[@]}"})
      if [ "${DATA_SRCS}" = admitted ]; then
        dry="${OUT}/data_${days}d_dryrun.log"
        if python scripts/mbh/mbh_cd1l_campaign.py --src all --dry-run "${camp[@]}" > "${dry}" 2>&1; then
          ids=$(sed -n "s/^\[admitted\] ${win} l1=\([^ ]*\) no_l1=.*/\1/p" "${dry}")
          noids=$(sed -n "s/^\[admitted\] ${win} l1=[^ ]* no_l1=\(.*\)/\1/p" "${dry}")
          echo "[durations] ${days} d: data, admitted with a brick: ${ids:--}; admitted without one (skipped): ${noids:--}"
          [ "${ids}" != "-" ] || ids=""
        else
          fail "${dry}"
          ids=""
        fi
      else
        ids="${DATA_SRCS}"
      fi
      for src in ${ids//,/ }; do
        log="${OUT}/data_${days}d_src${src}.log"
        set +e
        # shellcheck disable=SC2086
        python scripts/mbh/mbh_cd1l_campaign.py --src "${src}" "${camp[@]}" --templates "${DATA_TEMPLATES}" \
          --backend "${BACKEND}" --foreground "${FOREGROUND}" --out "${OUT}/data.jsonl" ${DATA_ARGS} \
          2>&1 | grep -v "${quiet}" > "${log}"
        rc=${PIPESTATUS[0]}
        set -e
        if [ "${rc}" = 3 ]; then
          echo "[durations] ${days} d: data, src ${src}: no MBHB L1 brick -- skipped"
        elif [ "${rc}" != 0 ]; then
          fail "${log}"
        fi
        grep "^\[campaign\]" "${log}" | grep -v "batched geometry" | cut -c1-220 || true
      done
    fi
  fi
done

if [ -s "${OUT}/data.jsonl" ]; then
  python scripts/mbh/mbh_cd1l_campaign_summary.py "${OUT}/data.jsonl" > "${OUT}/data_summary.md" \
    || fail "data summary"
  echo "[durations] data summary -> ${OUT}/data_summary.md"
  cat "${OUT}/data_summary.md"
fi

# ---- across durations: the batched cost is window-limited, the stock cost grows with the grid
python - "${OUT}" <<'PY' || fail "cross-duration summary"
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
    print("\n[durations] speed, s/row (warm; '-' = OOM / error / not run)")
    print("| days | src | SNR | orbits | " + " | ".join(stock) + " | " + " | ".join(f"B={b}" for b in bs) + " | best |")
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
    print("\n[durations] accuracy vs the 90-d snapped stock (gate: |dlogL| <= 0.5, mm < 1e-6; truth + near rows)")
    print("| days | src | SNR | verdict | max abs dlogL | max mm | mm_vs_production | orbits | noise |")
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

echo "[durations] done: ${OUT}/ (${fails} failed step(s))"
[ "${fails}" -eq 0 ]
