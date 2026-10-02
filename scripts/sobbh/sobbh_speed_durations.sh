#!/bin/bash
# SOBBH lookup (direct-to-WDM) vs the chunked comp at 6, 12 and 24 months on one GPU -- the
# SOBBH twin of scripts/emri/emri_speed_durations.sh.
#
# Per duration (Nt = days * 24 one-hour layers):
#   1. sobbh_lookup_speed_gpu.py on the production grid (Nf 1440, dt 2.5 s), once per lookup
#      (LOOKUPS; kernel = the fused C++/CUDA lookup, sobbh_lookup_kernel.cu, the default; python =
#      tracer + table + inner products in row blocks), rows per call ROWS: warm get_ll / fill per
#      row count, the resp / pn / trac / look / inner split, the GPU pool, and against the chunked
#      comp (run alongside the FIRST lookup only) its time and ch/look, max|dll|; the response's
#      stage profile (MBHTDIONFLY_TIMING); one JSON line per row count in ${OUT}/speed.jsonl.
#   2. sobbh_lookup_gate.py: accuracy against the dense TD->WDM transform of the same response,
#      per catalogue-like source (mismatch, norm ratio, dlogL; the Python template, == the kernel
#      to 1e-10) and for the comp's scoring path (max |lnL lookup - exact| over GATE_ROWS rows),
#      once per lookup, on the laptop-preset grid (Nf 180, dt 20 s: the table depends on the layer
#      duration only, and the dense reference is a CPU transform); ${OUT}/gate.jsonl.
#
#   salloc ... (1 GPU), then from the repo root:
#       bash scripts/sobbh/sobbh_speed_durations.sh
#   knobs (env): BACKEND (cuda13x), DAYS ("180 360 720"; Nt must stay a multiple of the chunked
#   comp's Nt_sub = 32), TABLE (default SOBBH_LOOKUP_TABLE_PATH; any n_ref table with 3600-s
#   layers), ROWS ("8,32,96,288"), LOOKUPS ("kernel,python"), REPS (3), GATE_ROWS (4),
#   GATE ("1"; 0 skips the accuracy step), SPEED_ARGS (extra speed-script flags, e.g. "--laptop"
#   for a CPU smoke on the Nf 180 / dt 20 s grid), OUT (sobbh_speed_durations_<date>).
#
# A lookup the build does not have (kernel without a rebuilt module) FAILS its step loudly
# instead of falling back: rebuild with pip install -e . --no-build-isolation.
set -euo pipefail

BACKEND=${BACKEND:-cuda13x}
DAYS=${DAYS:-180 360 720}
TABLE=${TABLE:-${SOBBH_LOOKUP_TABLE_PATH:-}}
ROWS=${ROWS:-8,32,96,288}
LOOKUPS=${LOOKUPS:-kernel,python}
REPS=${REPS:-3}
GATE_ROWS=${GATE_ROWS:-4}
GATE=${GATE:-1}
SPEED_ARGS=${SPEED_ARGS:-}
OUT=${OUT:-sobbh_speed_durations_$(date +%Y%m%d_%H%M)}
mkdir -p "${OUT}"
[ -n "${TABLE}" ] && [ -f "${TABLE}" ] || {
  echo "[durations] lookup table '${TABLE}' not found (set TABLE=... or SOBBH_LOOKUP_TABLE_PATH)"
  exit 2
}
quiet='lisaconstants\|warnings.warn\|hwloc\|FutureWarning\|cupy._util.experimental\|VACUUM_PERMEABILITY\|You may also open'

for days in ${DAYS}; do
  nt=$((days * 24))
  if [ $((nt % 32)) -ne 0 ]; then
    echo "[durations] ${days} d: Nt = ${nt} is not a multiple of 32 (the chunked comp's Nt_sub); skipping"
    continue
  fi
  first=1
  for lk in ${LOOKUPS//,/ }; do
    extra=()
    [ "${first}" = 1 ] || extra=(--no-chunked)   # the chunked comp once per duration
    first=0
    log="${OUT}/speed_${days}d_${lk}.log"
    echo "[durations] ${days} d (Nt ${nt}): speed, lookup=${lk}, rows ${ROWS} -> ${log}"
    MBHTDIONFLY_TIMING=1 python scripts/sobbh/sobbh_lookup_speed_gpu.py --backend "${BACKEND}" \
      --nt "${nt}" --rows "${ROWS}" --repeats "${REPS}" --kernel "${lk}" --table "${TABLE}" \
      --out "${OUT}/speed.jsonl" ${extra[@]+"${extra[@]}"} ${SPEED_ARGS} 2>&1 | grep -v "${quiet}" > "${log}" \
      || echo "[durations] FAILED: ${log}"
    grep "^backend\|^chunked comp\|^ rows\|^ *[0-9][0-9]* " "${log}" || true
    sed -n '/stage timing/,/3b_tdi_config/p' "${log}" | head -6 || true
  done
  if [ "${GATE}" = 1 ]; then
    for lk in ${LOOKUPS//,/ }; do
      log="${OUT}/gate_${days}d_${lk}.log"
      echo "[durations] ${days} d (Nt ${nt}): accuracy vs the dense transform, lookup=${lk} -> ${log}"
      python scripts/sobbh/sobbh_lookup_gate.py --nt "${nt}" --rows "${GATE_ROWS}" --no-chunked \
        --kernel "${lk}" --table "${TABLE}" --out "${OUT}/gate.jsonl" 2>&1 | grep -v "${quiet}" \
        > "${log}" || echo "[durations] FAILED: ${log}"
      grep "^src\|^\[gate\]" "${log}" | cut -c1-160 || true
    done
  fi
done
if [ "${GATE}" = 1 ] && [ -f "${OUT}/gate.jsonl" ]; then
  # SNR of each catalogue-like source vs duration (sqrt(<h|h>) against scirdv1 instrument noise;
  # the speed tables carry the scored rows' SNR in their last column)
  python - "${OUT}/gate.jsonl" <<'PYEOF' || true
import json, sys
snr, f_low = {}, {}
for line in open(sys.argv[1]):
    d = json.loads(line)
    if "src" not in d:
        continue
    snr.setdefault(d["nt"], {})[d["src"]] = d["snr"]
    f_low[d["src"]] = d["f_low"]
nts = sorted(snr)
print("[durations] SNR (scirdv1, XYZ) per source vs duration:")
print("  src  f_low[Hz]  " + "  ".join(f"{nt * 3600 / 86400:6.0f} d" for nt in nts))
for i in sorted(f_low):
    print(f"  {i:3d}  {f_low[i]:9.4f}  " + "  ".join(f"{snr[nt].get(i, float('nan')):8.3g}" for nt in nts))
PYEOF
fi
echo "[durations] done: ${OUT}/"
