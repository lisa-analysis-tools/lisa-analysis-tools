"""Derive scripts/fstat_proposal/submit_gf_1yr_v9.sh from submit_gf_6mo_v9_4gpu.sh.

Run from the LAT root after ANY edit to the 6mo script, then run
tests/test_submit_scripts_layout.py::OneYearV9TwinTest (it holds the same table).

Every replacement is an exact string with an expected match count; anything
else in the file is byte-identical. Re-runnable."""
import os, sys
src = "scripts/fstat_proposal/submit_gf_6mo_v9_4gpu.sh"
dst = "scripts/fstat_proposal/submit_gf_1yr_v9.sh"
text = open(src).read()

REPL = [
    # ---- naming (store, file names, SLURM) ---------------------------------
    ("#SBATCH --job-name=gf6mo_v9_4gpu     # job name",
     "#SBATCH --job-name=gf1yr_v9          # job name", 1),
    ("/shared/data/global_fit_output/gf6mo_v9_4gpu_%j.log",
     "/shared/data/global_fit_output/gf1yr_v9_%j.log", 1),
    ("STORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/}",
     "STORE_DIR=${STORE_DIR:-/shared/data/global_fit_output/gf_prod_1yr_v9/}", 1),
    ("export BASE_FILE_NAME=gf_prod_6mo\n", "export BASE_FILE_NAME=gf_prod_1yr\n", 1),
    # ---- Tobs and the knobs that scale with it (the 6mo-v8 -> 1yr-v8 precedent) ----
    ("export TOBS_TARGET=15552000        # 180 d; grid resolves Nf 1440 x Nt 4320 x dt 2.5 (exact factor-2 of 3 mo in Nt)",
     "export TOBS_TARGET=31104000        # 360 d; grid resolves Nf 1440 x Nt 8640 x dt 2.5 (exact factor-2 of 6 mo in Nt, as 6 mo was of 3 mo)", 1),
    ("export SIGHET_NT_LAYER=120\n", "export SIGHET_NT_LAYER=240\n", 1),
    ("export GB_NLEAVES_MAX=15000        # 6 mo: deeper confusion resolved; 3-mo ran 10000",
     "export GB_NLEAVES_MAX=20000        # 1 yr: deeper confusion resolved again; 6-mo ran 15000, 3-mo 10000", 1),
    ("export GB_N_SUBBANDS=8192   # PER GPU; total = x n_gpus. Slab ~0.5 MB/slot",
     "export GB_N_SUBBANDS=4096   # PER GPU; total = x n_gpus. Slab ~1.0 MB/slot at 1 yr", 1),
    ("export GB_RJ_INMODEL_CHUNK=32768  # byte-parity with the 3mo twin's 65536 (6mo cells ~2x bytes); floored to ntemps multiples by the column-atomic staging",
     "export GB_RJ_INMODEL_CHUNK=16384  # byte-parity with the 6mo 32768 (1yr cells ~2x bytes); floored to ntemps multiples by the column-atomic staging", 1),
    # ---- the sources inside a 360 d window (MBH session, 2026-10-03) --------
    ("export MBHB_IDS=2,5,16,18          # t_c 173.3 / 104.7 / 111.4 / 92.0 d",
     "export MBHB_IDS=0,2,3,4,5,7,9,12,15,16,18   # t_c 300.4/173.3/336.8/286.4/104.7/263.8/285.9/243.8/318.0/111.4/92.0 d (MBH session 2026-10-03). src 10 (2.8e6 Msun, SNR~1963) merges at 369.5 d, 9.5 d past the window, so the 7-d MBH_MERGER_TIME_BUFFER leaves it out; to model its in-window inspiral add 10 here and export MBH_MERGER_TIME_BUFFER=1209600", 1),
    # ---- the three preflight resolvers name the run's grid -----------------
    ("make_factory(1440, 4320)", "make_factory(1440, 8640)", 2),
    ("Nf 1440 x Nt 4320 at dt 2.5 s", "Nf 1440 x Nt 8640 at dt 2.5 s", 3),
    # ---- the warm-start / noise-pin parent is the 6mo v9 run (as 6mo's was the 3mo) ----
    ("${STORE_DIR}/warmstart/gf_prod_3mo_v8_10w_refereed.npz}",
     "${STORE_DIR}/warmstart/gf_prod_6mo_v9_4gpu_refereed.npz}", 1),
    ("GF_SEED_STORE=${GF_SEED_STORE:-/shared/data/global_fit_output/gf_prod_3mo_v8_10walkers/gf_prod_3mo_testing.h5}",
     "GF_SEED_STORE=${GF_SEED_STORE:-/shared/data/global_fit_output/gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5}", 1),
    ("GB_WARM_START_SOURCE_TOBS=${GB_WARM_START_SOURCE_TOBS:-7776000}",
     "GB_WARM_START_SOURCE_TOBS=${GB_WARM_START_SOURCE_TOBS:-15552000}   # the 6-month parent", 1),
]
for old, new, n in REPL:
    c = text.count(old)
    if c != n:
        sys.exit(f"REFUSING: {old!r} matches {c} times, expected {n}")
    text = text.replace(old, new)

HEADER = """#!/bin/bash
# ============================================================================
# PRODUCTION global fit -- 1yr_v9 (Tobs = 360 d). DERIVED 2026-10-03 FROM
# submit_gf_6mo_v9_4gpu.sh (dev 93f22bce, the 6mo production relaunch) BY
# scripts/fstat_proposal/derive_1yr_v9_launcher.py (exact-string replacement,
# re-run it after every 6mo edit): everything not listed here is
# BYTE-IDENTICAL to the 6mo script, and tests/test_submit_scripts_layout.py::
# OneYearV9TwinTest refuses any other drift. User ruling 2026-10-03: "identical
# runs with the exception of changing the usual Tobs changes ... the most
# minimal amount of changes needed for the waveforms and other areas due to
# 12 mo instead of 6". No "_4gpu" in the name: NGPUS picks the layout.
#
#   1YR-1. NAMING: --job-name gf1yr_v9, --output gf1yr_v9_%j.log,
#          STORE_DIR gf_prod_1yr_v9/, BASE_FILE_NAME gf_prod_1yr.
#   1YR-2. TOBS_TARGET 15552000 -> 31104000 (Nf 1440 x Nt 8640 x dt 2.5) and
#          the knobs that scale with it, exactly as 6mo-v8 -> 1yr-v8 did:
#          SIGHET_NT_LAYER 120 -> 240, GB_NLEAVES_MAX 15000 -> 20000,
#          GB_N_SUBBANDS 8192 -> 4096 per GPU (slot bytes double),
#          GB_RJ_INMODEL_CHUNK 32768 -> 16384 (byte parity).
#   1YR-3. MBHB_IDS: the 11 catalogue MBHBs merging inside 360 d + 7 d
#          (MBH session 2026-10-03); src 10 merges 9.5 d past the window and
#          is left out (see the MBHB_IDS line). EMRI_IDS / SOBHB_IDS unchanged.
#   1YR-4. The MBH / EMRI / SOBBH preflight resolvers name the 1yr grid
#          (make_factory(1440, 8640); comments). The lookup table is the SAME
#          canonical file (it depends on Nf and dt only), built once in the
#          new STORE_DIR on the first launch.
#   1YR-5. Warm start + noise pin parent = the 6mo v9 run
#          (GF_SEED_STORE gf_prod_6mo_v9_4gpu/gf_prod_6mo_testing.h5,
#          GB_WARM_START_SOURCE_TOBS 15552000, refereed npz
#          gf_prod_6mo_v9_4gpu_refereed.npz -- auto-built from the parent store
#          on the first launch when missing, as the 6mo's was from the 3mo).
#
# WAVEFORM SETTINGS AT 1 YR (all three windows, 2026-10-03): NO export line
# changes. EMRI direct: same table; EMRIs 0 (347 d) and 3 (256 d) plunge INSIDE
# the window and take the plunge path (fdot handoff + dense plunge chunk +
# 120 s stop taper, built alone, ~100-175 ms/row vs ~21 ms batched), validated
# on the laptop only -- watch "[EMRI_DIRECT] F fallbacks" = 0 and the
# every-10th cross-check. SOBBH lookup: same table, EVAL_DT 43200 holds
# (phase-tracer error is local), 0.036 s per 8-row call at 360 d (chunked
# 3.40 s). MBH batched: window knobs unchanged, decimation 2 safe for all 11,
# ~45 s per mbh_pe per rank (11 leaves). The exact SOBBH cross-check costs ~2x
# the 6mo one: put SOBBH_CHECK_LL_EVERY=10 on the line only if the 6mo run's
# first check stayed under ~10 % of its SOBBH leg when doubled.
#
# NOT carried (deliberately, "identical runs"): the 2026-09-08 1yr F-stat grid
# reduction (FSTAT_COMB_NSKY_MAX 256, FSTAT_N_ALPHA/N_SINDELTA 4) -- the 1yr v8
# 4-GPU script did not carry it either; revisit if the first F-stat fit's wall
# or dev0 memory says so. GALFOR_START_PARAMS stays the offline 3mo estimate
# (6mo ruling 2026-09-24 "just for now").
#
# LAUNCH (from the LAT root; the 6mo production line with the 1yr script):
#   GALFOR_RATCHET=1 MIDIT_CHECKPOINT=0 EMRI_TRAJ_WORKERS=8 \\
#     MBH_WINDOW_DECIMATE=2 NGPUS=4 ./scripts/fstat_proposal/submit_gf_1yr_v9.sh
# (a FRESH store: no CLOCK_START, no re-rung -- MBH_NTEMPS / EMRI_NTEMPS build
#  the ladders; the 6mo store was re-rung to 8, so pass MBH_NTEMPS=8
#  EMRI_NTEMPS=8 to match it.)
#
# ---- the 6mo v9 header follows verbatim ------------------------------------
"""
assert text.startswith("#!/bin/bash\n")
text = HEADER + text[len("#!/bin/bash\n"):]
open(dst, "w").write(text)
os.chmod(dst, os.stat(src).st_mode)
print("wrote", dst, len(text.splitlines()), "lines")
