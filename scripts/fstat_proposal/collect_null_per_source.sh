#!/bin/bash
# ============================================================================
# PER-SOURCE TRUTH-INJECTION NULL TEST -- collector
# (companion to launch_null_per_source.sh)
#
# Reads each per-source store's LOGS for run.py's
#
#     initial log likelihood (after recipe setup): [...]
#
# and prints one row per source plus the sum. The sum is the cross-check: it
# should reconstruct the FILE-BACKED share of the combined null's -325.62.
#
# WHY LOGS AND NOT THE h5. With NULL_CHECK_ONLY=1 the run stops before the
# sampler exists, so the store has iteration 0 and there is no chain to read.
# The lnL line is a `logger.info`, so it lands in BOTH
#   ${STORE}/gf_prod_6mo_artifacts/globalfit_run.log   (always)
#   ${STORE}/slurm_stdout_JOBID.log                    (the 30 s mirror)
# and this script greps the store RECURSIVELY so either one satisfies it.
# The last match wins, so a resubmitted store reports its newest number.
#
# The printed value is the FIRST per-walker entry. run.py logs
# state.log_like[0], the whole cold-walker row; with *_START_FACTOR=0 every
# walker starts at the identical truth point, so they are all the same
# number (the canonical script's header says so explicitly).
#
# USAGE
#   bash scripts/fstat_proposal/collect_null_per_source.sh
#   STORE_ROOT=/somewhere/else bash scripts/fstat_proposal/collect_null_per_source.sh
#   COMBINED_REF=-325.62 bash scripts/fstat_proposal/collect_null_per_source.sh
# ============================================================================
set -uo pipefail

# ---- must MATCH launch_null_per_source.sh ----------------------------------
STORE_ROOT="${STORE_ROOT:-/shared/data/global_fit_output}"
STORE_PREFIX="${STORE_PREFIX:-gf_prod_6mo_v8_null_}"

# FILE-BACKED sources: these have a real mojito L1 brick, so their null
# MEASURES the mojito<->template mismatch. One run each.
MBH_IDS_LIST="2 5 16 18"
EMRI_IDS_LIST="1 2 3 4 5 6"
SOBBH_IDS_LIST="0 1 2"

# SYNTHESIZED sources: no L1 brick, rebuilt in-process from the catalogue
# with the same converters/epochs/orbits as the branch template, so the
# residual cancels to machine precision. No run is launched; the row is
# printed anyway so the table always shows the full 18-source census.
MBH_SKIP_LIST=""
EMRI_SKIP_LIST="0 7"
SOBBH_SKIP_LIST="3 4 5"

# The combined null this decomposes (submit_gf_6mo_v8_nogb_null.sh, 18
# sources at once).
COMBINED_REF="${COMBINED_REF:--325.62}"

MARKER="initial log likelihood (after recipe setup)"

# ---- pull one store's number ------------------------------------------------
# Echoes the lnL, or nothing at all if the store has not produced one yet.
_lnl() {
  local store="$1"
  [ -d "${store}" ] || return 0
  grep -rh -- "${MARKER}" "${store}" 2>/dev/null \
    | tail -1 \
    | sed 's/.*after recipe setup)://' \
    | tr -d '[],' \
    | awk '{print $1}'
}

_row() {  # branch id state
  local branch="$1" id="$2" state="$3"
  local tag="${branch}${id}"
  local store="${STORE_ROOT}/${STORE_PREFIX}${tag}/"

  if [ "${state}" = "skipped" ]; then
    printf '  %-6s %4s   %16s   %s\n' "${branch}" "${id}" "0 (exact)" \
      "synthesized from catalogue (skipped)"
    return 0
  fi

  local val
  val="$(_lnl "${store}")"
  if [ -z "${val}" ]; then
    if [ -d "${store}" ]; then
      printf '  %-6s %4s   %16s   %s\n' "${branch}" "${id}" "--" \
        "store exists, no lnL line yet (running or died early)"
    else
      printf '  %-6s %4s   %16s   %s\n' "${branch}" "${id}" "--" \
        "no store: ${store}"
    fi
    return 0
  fi

  printf '  %-6s %4s   %16s   %s\n' "${branch}" "${id}" "${val}" "file-backed brick"
  VALUES="${VALUES} ${val}"
  NHAVE=$((NHAVE + 1))
}

VALUES=""
NHAVE=0
NWANT=0

echo "PER-SOURCE TRUTH-INJECTION NULL -- initial log likelihood"
echo "stores: ${STORE_ROOT}/${STORE_PREFIX}BRANCHID/"
echo ""
printf '  %-6s %4s   %16s   %s\n' "branch" "id" "lnL" "provenance"
printf '  %-6s %4s   %16s   %s\n' "------" "----" "----------------" "----------"

for _id in ${MBH_IDS_LIST};   do NWANT=$((NWANT + 1)); _row mbh   "${_id}" have; done
for _id in ${MBH_SKIP_LIST};  do _row mbh   "${_id}" skipped; done
for _id in ${EMRI_IDS_LIST};  do NWANT=$((NWANT + 1)); _row emri  "${_id}" have; done
for _id in ${EMRI_SKIP_LIST}; do _row emri  "${_id}" skipped; done
for _id in ${SOBBH_IDS_LIST}; do NWANT=$((NWANT + 1)); _row sobbh "${_id}" have; done
for _id in ${SOBBH_SKIP_LIST};do _row sobbh "${_id}" skipped; done

# ---- eps-sweep variants (NULL_EMRI_EPS runs, 2026-09-16) --------------------
# The launcher tags sweep stores ${STORE_PREFIX}<branch><id>_eps<val>/ so they
# never collide with the baseline. Auto-discovered here: one row per variant
# store found, with the delta against its baseline row. delta ~ +|lnL_base|
# recovered means the deficit WAS mode truncation at the baseline eps.
_EPS_HEADER=0
for _d in "${STORE_ROOT}/${STORE_PREFIX}"*_eps*/; do
  [ -d "${_d}" ] || continue
  _tag="$(basename "${_d}")"; _tag="${_tag#"${STORE_PREFIX}"}"
  _base_tag="${_tag%%_eps*}"
  _eps="${_tag#*_eps}"
  _val="$(_lnl "${_d}")"
  _base_val="$(_lnl "${STORE_ROOT}/${STORE_PREFIX}${_base_tag}/")"
  if [ "${_EPS_HEADER}" = "0" ]; then
    echo ""
    echo "  EPS SWEEP VARIANTS (vs their baseline rows above)"
    printf '  %-10s %8s   %16s   %16s   %s\n' "source" "eps" "lnL" "baseline" "delta"
    printf '  %-10s %8s   %16s   %16s   %s\n' "------" "-----" "----------------" "----------------" "-----"
    _EPS_HEADER=1
  fi
  if [ -z "${_val}" ]; then
    printf '  %-10s %8s   %16s   %16s   %s\n' "${_base_tag}" "${_eps}" "--" \
      "${_base_val:---}" "(running or died early)"
  elif [ -n "${_base_val}" ]; then
    printf '  %-10s %8s   %16s   %16s   %s\n' "${_base_tag}" "${_eps}" \
      "${_val}" "${_base_val}" \
      "$(awk -v a="${_val}" -v b="${_base_val}" 'BEGIN{printf "%+.4g", a-b}')"
  else
    printf '  %-10s %8s   %16s   %16s   %s\n' "${_base_tag}" "${_eps}" \
      "${_val}" "--" "(no baseline run yet)"
  fi
done

SUM="$(printf '%s\n' ${VALUES} | awk '{s += $1} END {printf "%.6g", s+0}')"

echo ""
echo "  file-backed runs reporting: ${NHAVE} / ${NWANT}"
echo "  SUM of reported lnL       : ${SUM}"
echo "  combined null reference   : ${COMBINED_REF}"
if [ "${NHAVE}" -eq "${NWANT}" ] && [ "${NWANT}" -gt 0 ]; then
  printf '  residual (combined - sum) : %s\n' \
    "$(awk -v a="${COMBINED_REF}" -v b="${SUM}" 'BEGIN{printf "%.6g", a-b}')"
  echo ""
  echo "  The sum reconstructs the FILE-BACKED share of the combined null."
  echo "  The 5 synthesized ids contribute ~0 by construction, so a residual"
  echo "  far from 0 means the sources are NOT independent in the residual"
  echo "  (overlapping support) -- which is itself the finding."
else
  echo ""
  echo "  INCOMPLETE: the sum is only over the runs that have reported."
  echo "  Do not compare it to the combined reference yet."
fi
