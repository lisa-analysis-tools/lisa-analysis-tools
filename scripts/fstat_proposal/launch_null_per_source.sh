#!/bin/bash
# ============================================================================
# PER-SOURCE TRUTH-INJECTION NULL TEST -- launcher
# (user ask 2026-09-15: decompose the COMBINED null into one number per
# source: "a bash script that launches one null run PER SOURCE -- literally
# one source id at a time")
#
# ---- WHAT THIS IS ----------------------------------------------------------
# submit_gf_6mo_v8_nogb_null.sh measured ONE combined initial lnL of -325.62
# over 18 armed sources (sobbh 0-5, mbh 2/5/16/18, emri 0-7). That single
# number cannot say WHICH injection<->template mismatch it came from. This
# launcher runs the same null test once per source, so each job's
#
#     initial log likelihood (after recipe setup)
#
# IS that one source's systematic. Sum them back up and you should recover
# the file-backed share of -325.62 (collect_null_per_source.sh does this).
#
# ---- HOW IT ISOLATES ONE SOURCE --------------------------------------------
# run_combined_staged.py's id envs -- MBHB_IDS / EMRI_IDS / SOBHB_IDS, the
# CLASS-prefixed comma lists parsed at run_combined_staged.py:343-356 -- are
# the ONLY thing that arms a source branch. They feed ONE dict,
# general.mojito_source_ids (run_combined_staged.py:548-556), and that single
# dict drives BOTH ends of the null:
#
#   * the INJECTION -- all_sources.py:499-503 builds the loader's source_ids
#     from it, so only those bricks are summed into the data stream;
#   * the SAMPLED BRANCH -- the same loader's catalogue sets each branch's
#     injection table, nleaves and (with *_START_FACTOR=0) its exact-truth
#     start coords (source_runtime.py:721-729, 790-825, 852-885, 896-901;
#     run.py:844-905, 944-972).
#
# So one id in = exactly one source injected AND exactly one source sampled.
# An EMPTY id list leaves that branch unarmed, which drops the branch from
# the fit AND keeps its streams out of the data (the `branch in
# self._branch_names` gate at all_sources.py:499-503) -- which is why the
# other two branches are silenced here with empty id lists rather than with
# REMOVE_BRANCHES. REMOVE_BRANCHES only accepts gb/galfor/vgb/psd
# (run_combined_staged.py:453) and RAISES on a source branch name.
#
# ---- WHY GENERATED COPIES, NOT ENV OVERRIDES -------------------------------
# submit_gf_6mo_v8_nogb_null.sh sets almost everything with PLAIN `export`
# (not `${VAR:-default}`), so `MBHB_IDS=16 sbatch submit_...nogb_null.sh`
# would be silently overwritten by the script's own line. The canonical
# script is also OFF LIMITS to edit (another agent owns it). So each run gets
# a sed-derived COPY that differs in a handful of asserted lines. sbatch
# copies the script at submit time, so generating into a temp dir is safe --
# the copies can be deleted the moment every job is queued.
#
# ---- THE RUN ENDS AT THE READOUT -------------------------------------------
# Each generated copy exports NULL_CHECK_ONLY=1 (run.py's null_check_only):
# the run stops right after the initial-lnL print, with NO sampling at all.
# It is a CLEAN stop, not a sys.exit -- this is an `mpiexec -n 3` job (main +
# saver + spare) and the other two ranks sit in comm.recv, so the early exit
# still goes out through the ordinary "stop" / {"finish_run": True} sends and
# all three ranks end rc 0 in seconds. Set NULL_PER_SOURCE_ITERS=N to get a
# short SAMPLING run of N iterations instead (the knob is then not inserted).
#
# ---- USAGE -----------------------------------------------------------------
#   # see what would be submitted, generate the scripts, submit nothing:
#   DRY_RUN=1 bash scripts/fstat_proposal/launch_null_per_source.sh
#
#   # the real thing: 13 jobs, CHAINED one at a time (spot etiquette)
#   bash scripts/fstat_proposal/launch_null_per_source.sh
#
#   # submit all 13 independently (only on a quiet cluster)
#   PARALLEL=1 bash scripts/fstat_proposal/launch_null_per_source.sh
#
#   # 4 GPUs per job instead of the header's 2
#   NGPUS=4 bash scripts/fstat_proposal/launch_null_per_source.sh
#
# Then, once they have run:
#   bash scripts/fstat_proposal/collect_null_per_source.sh
# ============================================================================
set -euo pipefail

# ---- WHICH SOURCES GET A RUN (edit here) -----------------------------------
# FILE-BACKED sources only -- the ones with a real mojito L1 brick, verified
# from the combined null run's own [SOURCES] preflight scan
# (slurm_stdout_509.log). Those are the only ones whose null MEASURES
# anything: a brick-backed source leaves the mojito<->our-template waveform
# mismatch in the residual.
MBH_IDS_LIST="2 5 16 18"
EMRI_IDS_LIST="1 2 3 4 5 6"
SOBBH_IDS_LIST="0 1 2"

# ---- DELIBERATELY SKIPPED (re-enable by swapping these lines in) -----------
# These 5 ids have NO L1 brick. SYNTHESIZE_MISSING_BRICKS=1 rebuilds them
# in-process from their CATALOGUE parameters, using the same converters,
# epochs and orbits as the branch template -- so the data and the template
# are the SAME waveform and the residual cancels to machine precision. Their
# null is ZERO BY DEFINITION; a job for one of them measures nothing and only
# burns an allocation. The combined run's loader marks them
# "[HYBRID] ... no L1 brick found -- catalogue parameters kept".
#
#   EMRI_IDS_LIST="0 1 2 3 4 5 6 7"   # 0 and 7 are synthesized
#   SOBBH_IDS_LIST="0 1 2 3 4 5"      # 3, 4 and 5 are synthesized
#
# (collect_null_per_source.sh still prints a row for all 18 so the table
# always shows the full census, with these 5 marked "synthesized (skipped)".)

# ---- where things live -----------------------------------------------------
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANONICAL="${_HERE}/submit_gf_6mo_v8_nogb_null.sh"

# Store root + prefix MUST match what the canonical script uses, because the
# generated copies only swap the leaf name (canonical:
# STORE_DIR=/shared/data/global_fit_output/gf_prod_6mo_v8_nogb_null/).
STORE_ROOT="${STORE_ROOT:-/shared/data/global_fit_output}"
STORE_PREFIX="${STORE_PREFIX:-gf_prod_6mo_v8_null_}"

# sbatch copies the script at submit time, so a temp dir is fine.
GEN_DIR="${GEN_DIR:-${TMPDIR:-/tmp}/null_per_source_generated}"

DRY_RUN="${DRY_RUN:-0}"
PARALLEL="${PARALLEL:-0}"
NGPUS="${NGPUS:-}"
NULL_PER_SOURCE_ITERS="${NULL_PER_SOURCE_ITERS:-}"

if [ ! -f "${CANONICAL}" ]; then
  echo "[LAUNCH] FATAL: canonical null script not found: ${CANONICAL}" >&2
  exit 2
fi

# NGPUS -> partition, mirroring the canonical script's own self-dispatch
# (2 -> gpu-80-spot, 4 -> gpu-160-spot). Nothing else about resources is
# touched; with NGPUS unset the generated copy's own #SBATCH header wins.
_PART=""
if [ -n "${NGPUS}" ]; then
  case "${NGPUS}" in
    2) _PART=gpu-80-spot ;;
    4) _PART=gpu-160-spot ;;
    *) echo "[LAUNCH] FATAL: NGPUS=${NGPUS} unsupported (2 or 4)." >&2; exit 2 ;;
  esac
fi

mkdir -p "${GEN_DIR}"

# ---- sed with a MANDATORY before/after check -------------------------------
# A silent no-op sed is the classic failure mode here: the copy looks fine,
# the job runs the FULL 18-source null, and the "per-source" number is a lie.
# So every edit asserts three things: the anchor was there, the new text
# landed, and the old text is gone.
_sed_assert() {  # file  sed_expr  old_fixed  new_fixed  label
  local f="$1" expr="$2" old="$3" new="$4" label="$5"
  if ! grep -qF -- "${old}" "${f}"; then
    echo "[GEN] FATAL: ${label}: anchor NOT FOUND in ${f}" >&2
    echo "[GEN]        expected to find: ${old}" >&2
    echo "[GEN]        the canonical script changed shape -- fix this launcher." >&2
    exit 3
  fi
  sed -i.bak "${expr}" "${f}"
  rm -f "${f}.bak"
  if ! grep -qF -- "${new}" "${f}"; then
    echo "[GEN] FATAL: ${label}: substitution did NOT land in ${f}" >&2
    echo "[GEN]        expected to find: ${new}" >&2
    exit 3
  fi
  if grep -qF -- "${old}" "${f}"; then
    echo "[GEN] FATAL: ${label}: the ORIGINAL text is still present in ${f}" >&2
    echo "[GEN]        still there: ${old}" >&2
    exit 3
  fi
}

# ---- generate one per-source script ----------------------------------------
# $1 branch tag (mbh|emri|sobbh)   $2 source id
_generate() {
  local branch="$1" id="$2"
  local tag="${branch}${id}"
  local store="${STORE_ROOT}/${STORE_PREFIX}${tag}/"
  local out="${GEN_DIR}/submit_null_${tag}.sh"

  # branch -> the mojito CLASS whose *_IDS env arms it, and the one
  # SOURCE_TYPES stream it injects.
  local cls
  case "${branch}" in
    mbh)   cls=MBHB ;;
    emri)  cls=EMRI ;;
    sobbh) cls=SOBHB ;;
    *) echo "[GEN] FATAL: unknown branch ${branch}" >&2; exit 3 ;;
  esac

  cp "${CANONICAL}" "${out}"
  chmod +x "${out}"

  # (1) STORE DIR -- its own store, so 13 runs never share an h5 or a log.
  _sed_assert "${out}" \
    "s|^STORE_DIR=.*|STORE_DIR=${store}|" \
    "STORE_DIR=/shared/data/global_fit_output/gf_prod_6mo_v8_nogb_null/" \
    "STORE_DIR=${store}" \
    "store dir"

  # (2) JOB NAME -- scheduler label only (SLURM_JOB_NAME is read nowhere in
  #     the canonical script), so 13 chained jobs are distinguishable in
  #     squeue. The --output path and the SLURM_LOG mirror are deliberately
  #     LEFT ALONE: they agree with each other by name, and %j already makes
  #     each job's log unique. The mirror still copies it into this run's
  #     own STORE_DIR, which is what the collector reads.
  _sed_assert "${out}" \
    "s|^#SBATCH --job-name=.*|#SBATCH --job-name=gfnull_${tag}|" \
    "#SBATCH --job-name=gf6mo_null" \
    "#SBATCH --job-name=gfnull_${tag}" \
    "job name"

  # (3) SOURCE_TYPES -- this run injects ONE class of stream.
  _sed_assert "${out}" \
    "s|^export SOURCE_TYPES=.*|export SOURCE_TYPES=${cls}   # PER-SOURCE NULL: one class only|" \
    "export SOURCE_TYPES=SOBHB,MBHB,EMRI" \
    "export SOURCE_TYPES=${cls}" \
    "source types"

  # (4) THE ID LISTS -- the whole point. The chosen branch keeps exactly one
  #     id; the other two get an EMPTY list, which leaves them unarmed, so
  #     they are neither sampled nor injected.
  local mbh_ids="" emri_ids="" sobbh_ids=""
  case "${branch}" in
    mbh)   mbh_ids="${id}" ;;
    emri)  emri_ids="${id}" ;;
    sobbh) sobbh_ids="${id}" ;;
  esac
  _sed_assert "${out}" \
    "s|^export MBHB_IDS=.*|export MBHB_IDS=${mbh_ids}   # PER-SOURCE NULL|" \
    "export MBHB_IDS=2,5,16,18" \
    "export MBHB_IDS=${mbh_ids}   # PER-SOURCE NULL" \
    "MBHB ids"
  _sed_assert "${out}" \
    "s|^export EMRI_IDS=.*|export EMRI_IDS=${emri_ids}   # PER-SOURCE NULL|" \
    "export EMRI_IDS=0,1,2,3,4,5,6,7" \
    "export EMRI_IDS=${emri_ids}   # PER-SOURCE NULL" \
    "EMRI ids"
  _sed_assert "${out}" \
    "s|^export SOBHB_IDS=.*|export SOBHB_IDS=${sobbh_ids}   # PER-SOURCE NULL|" \
    "export SOBHB_IDS=0,1,2,3,4,5" \
    "export SOBHB_IDS=${sobbh_ids}   # PER-SOURCE NULL" \
    "SOBHB ids"

  # (5) WHERE THE RUN STOPS.
  if [ -n "${NULL_PER_SOURCE_ITERS}" ]; then
    # Escape hatch: a short SAMPLING run instead of the bare readout.
    _sed_assert "${out}" \
      "s|^export NUM_ITERATIONS=.*|export NUM_ITERATIONS=${NULL_PER_SOURCE_ITERS}   # PER-SOURCE NULL: short run|" \
      "export NUM_ITERATIONS=10" \
      "export NUM_ITERATIONS=${NULL_PER_SOURCE_ITERS}   # PER-SOURCE NULL: short run" \
      "num iterations"
  else
    # Default: stop at the readout. Inserted right after the id block so it
    # reads next to the settings it belongs with. awk (not `sed a\`) because
    # BSD and GNU sed disagree about appending lines.
    local ins="export NULL_CHECK_ONLY=1   # PER-SOURCE NULL: stop at the initial-lnL print, no sampling"
    if ! grep -q "^export SOBHB_IDS=" "${out}"; then
      echo "[GEN] FATAL: NULL_CHECK_ONLY anchor (export SOBHB_IDS=) missing in ${out}" >&2
      exit 3
    fi
    awk -v ins="${ins}" '{print} /^export SOBHB_IDS=/{print ins}' \
      "${out}" > "${out}.tmp" && mv "${out}.tmp" "${out}"
    chmod +x "${out}"
    if ! grep -qF -- "export NULL_CHECK_ONLY=1" "${out}"; then
      echo "[GEN] FATAL: NULL_CHECK_ONLY insert did NOT land in ${out}" >&2
      exit 3
    fi
  fi

  # Syntax-check every generated script -- a broken copy must fail HERE, not
  # 20 minutes into an allocation.
  if ! bash -n "${out}"; then
    echo "[GEN] FATAL: generated script fails bash -n: ${out}" >&2
    exit 3
  fi

  echo "${out}"
}

# ---- build the (branch, id) work list --------------------------------------
RUNS=""
for _id in ${MBH_IDS_LIST}; do RUNS="${RUNS} mbh:${_id}"; done
for _id in ${EMRI_IDS_LIST}; do RUNS="${RUNS} emri:${_id}"; done
for _id in ${SOBBH_IDS_LIST}; do RUNS="${RUNS} sobbh:${_id}"; done

_N=0
for _r in ${RUNS}; do _N=$((_N + 1)); done

echo "[LAUNCH] canonical : ${CANONICAL}"
echo "[LAUNCH] generating: ${GEN_DIR}"
echo "[LAUNCH] stores    : ${STORE_ROOT}/${STORE_PREFIX}BRANCHID/"
echo "[LAUNCH] runs      : ${_N}  (mbh:${MBH_IDS_LIST} | emri:${EMRI_IDS_LIST} | sobbh:${SOBBH_IDS_LIST})"
if [ -n "${NULL_PER_SOURCE_ITERS}" ]; then
  echo "[LAUNCH] mode      : SAMPLING, NUM_ITERATIONS=${NULL_PER_SOURCE_ITERS}"
else
  echo "[LAUNCH] mode      : NULL_CHECK_ONLY=1 (stop at the initial-lnL print)"
fi
if [ "${PARALLEL}" = "1" ]; then
  echo "[LAUNCH] chaining  : OFF (PARALLEL=1 -- all jobs submitted independently)"
else
  echo "[LAUNCH] chaining  : ON  (each job --dependency=afterany the previous)"
fi
if [ -n "${NGPUS}" ]; then
  echo "[LAUNCH] resources : NGPUS=${NGPUS} -> --partition=${_PART} --gres=gpu:${NGPUS}"
fi
if [ "${DRY_RUN}" = "1" ]; then
  echo "[LAUNCH] DRY_RUN=1 : scripts are generated, NOTHING is submitted."
  echo "[LAUNCH]             In the commands below PREVJOBID stands for the"
  echo "[LAUNCH]             job id sbatch returns for the line above it;"
  echo "[LAUNCH]             a real run substitutes it automatically."
fi
echo ""

PREV_JOB=""
for _r in ${RUNS}; do
  _branch="${_r%%:*}"
  _id="${_r##*:}"
  # A generation failure must stop EVERYTHING: `_generate` runs in a command
  # substitution, so its `exit 3` only ends that subshell -- check here too
  # rather than trusting set -e to notice.
  if ! _script="$(_generate "${_branch}" "${_id}")"; then
    echo "[LAUNCH] FATAL: could not generate the script for ${_branch}${_id}." >&2
    exit 3
  fi
  if [ -z "${_script}" ] || [ ! -f "${_script}" ]; then
    echo "[LAUNCH] FATAL: generation produced no script for ${_branch}${_id}." >&2
    exit 3
  fi

  # Build the sbatch command.
  _cmd="sbatch"
  if [ -n "${NGPUS}" ]; then
    _cmd="${_cmd} --partition=${_PART} --gres=gpu:${NGPUS}"
  fi
  if [ "${PARALLEL}" != "1" ] && [ -n "${PREV_JOB}" ]; then
    _cmd="${_cmd} --dependency=afterany:${PREV_JOB}"
  fi
  _cmd="${_cmd} ${_script}"

  if [ "${DRY_RUN}" = "1" ]; then
    printf '%s\n' "${_cmd}"
    # Chained dry run: the NEXT command depends on this one's (unknown) id.
    if [ "${PARALLEL}" != "1" ]; then
      PREV_JOB="PREVJOBID"
    fi
  else
    _out="$(${_cmd})"
    echo "[LAUNCH] ${_branch}${_id}: ${_out}"
    # "Submitted batch job 12345" -> 12345
    _jid="$(printf '%s\n' "${_out}" | awk '/Submitted batch job/{print $NF}')"
    if [ "${PARALLEL}" != "1" ]; then
      if [ -z "${_jid}" ]; then
        echo "[LAUNCH] FATAL: could not parse a job id out of: ${_out}" >&2
        echo "[LAUNCH]        refusing to chain blindly -- the remaining jobs" >&2
        echo "[LAUNCH]        would all start at once. Submit the rest by hand." >&2
        exit 4
      fi
      PREV_JOB="${_jid}"
    fi
  fi
done

echo ""
if [ "${DRY_RUN}" = "1" ]; then
  echo "[LAUNCH] ${_N} scripts written to ${GEN_DIR} (nothing submitted)."
  echo "[LAUNCH] Inspect one with:  diff ${CANONICAL} ${GEN_DIR}/submit_null_mbh16.sh"
else
  echo "[LAUNCH] ${_N} jobs submitted."
  echo "[LAUNCH] Read the numbers with:"
  echo "[LAUNCH]   bash scripts/fstat_proposal/collect_null_per_source.sh"
fi
