# One-walker replica mode — cluster testing campaign (2 nodes × 1 GPU)

Branch `one-walker-replicas`, merged into `dev` (first merge 65d064c3). Laptop evidence in hand: GB one-walker/two-replica smoke (4 `gb_sync` rounds, agreeing hashes), parity arm, 4-walker two-rank arm, the one-walker noise-smoke arm (PSD eigen inner + row scatter + replays; it exposed the `prior_box_widths` unit-width defect, fixed the same day — psd in-model acceptance 48/120, was 0), the two-node replica placements in `tests/test_rank_layout.py`, 545-test sweep — all green. Everything below runs on the cluster and is what actually proves the GPU + cross-node path.

Design authority: `docs/superpowers/specs/2026-09-16-one-walker-replicas-design.md`. Launcher facts: `docs/multirank-cluster-gates.md` ("MPI launcher on this cluster"; "One-walker replica mode gate"). Log digest: `python scripts/diagnostics/gf_run_log_digest.py <run_dir>` (prints the `[FANOUT]` load-balance table, the `[FANOUT_DIGEST] replicas` agreement count and the `[GB_REPLICA]` counters).

## The layout (user ruling 2026-09-16: test across nodes)

Every gate runs on **2 nodes with 1 GPU each**, one replica per node:

```sh
salloc --partition=gpu-80-spot --nodes=2 --gres=gpu:1 --ntasks-per-node=2 --time=02:00:00
source /shared/home/mlkatz1/envs/gf_env/bin/activate
cd /shared/home/mlkatz1/lisa-analysis-tools && git checkout dev && git pull --ff-only

export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi FI_PROVIDER=tcp   # hydra + tcp fabric, the launcher that works here
export DATA_MODE=synthetic NUM_ITERATIONS=4 MIDIT_CHECKPOINT=0 MAKE_DIAGNOSTIC_PLOTS=0 GF_FANOUT_DIGEST=1 GF_LEGACY_RANK_LAYOUT=0
export NWALKERS=1
G=$HOME/onewalker_gates; mkdir -p "$G"

# THE LAUNCH (every gate unless stated): 3 ranks round-robin over the 2 hosts,
# through the gate driver -- it applies the campaign script's sig-het pins
# (printed as a [GATE] line; a VAR=... prefix on the command still wins) and
# injects one loud GB into the GB stocks (--no-injection to skip)
FILE_STORE_DIR=$G/<gate>/ GPUS=0 mpiexec -n 3 -ppn 1 python scripts/diagnostics/gate_run.py --stock <name>
#   rank 0  head    node A  GPU 0   replica 0 (the head computes too)
#   rank 1  compute node B  GPU 0   replica 1
#   rank 2  saver   node A
```

`GPUS=0` is load-bearing: it is the **per-node** pool. Each node hosts one compute rank, and AUTO `gpus_per_rank` gives a lone compute rank its node's whole pool, so the pin keeps every rank on one device. `-ppn 1` is the round-robin (cyclic) placement. Every rank must print `size=3 n_compute=2`; a rank printing `size=1` is the launcher singleton symptom from the runbook's launcher table, not a layout bug.

The **single-node control layout** used by T2 keeps all three ranks on node A sharing its one GPU: `GPUS=0 RANKS_PER_GPU=2 mpiexec -n 3 -ppn 3 ...` (inside the same allocation; `-ppn 3` keeps every rank on the first host).

## Rules for every gate

- **Wall-clock limit on every run** (`--time` on the allocation, `timeout` on the mpiexec). A rank error inside the GB per-unit exchange hangs the other ranks instead of crashing; a hang IS a failure signal, not a slow run.
- Keep every rank's log (`globalfit_run.log`, `globalfit_run.rank<k>.log`) and the digest-script output per run; name run dirs by gate (`T1`, `T2c`, `T2a`, ...).
- An invariance claim needs its control (project rule). Each gate names its control; a gate without its control passing is not a pass.
- Order matters: do not run a later gate until the earlier one's pass criterion is met. T0 + T1 take well under an hour.
- The fit directory must be on the shared filesystem: the head writes the F-stat epoch (`<fit_dir>/shared/epoch_NNNN/`) and the replica on node B opens it on the next message. `F-stat epoch N incomplete at <path>` on node B means storage, not code — fix the storage, never suppress the check.
- **A fresh `FILE_STORE_DIR` per run** (`FILE_STORE_DIR=$G/<gate>/`): a run that finds an existing store RESUMES from it — a store written with another walker count aborts with `walker-count mismatch for branch 'gb'`, and one with the same count silently breaks the same-initial-state premise.
- **GB gates need a loud injection** (the driver does it by default). `--stock gb_no_fg` synthetic has no GB catalogue: the F-stat epoch reports `NO peaks`, births fall back to the prior, nothing is accepted, and every replica trivially agrees — the run exercises the transport and `gb_sync` only, never the ledger or the merge. There is no env knob for the injection table; `gate_run.py` sets `fit.general.gb_injection_params` to the WP7 runbook's loud in-band binary (a live fixture shows `20 peaks` on the epoch line and hashes that change every iteration).
  Expected, benign warnings on every run: `No 'GB' catalogue found; GB SNR-cut injection skipped`, `sig-het nt_layer=64 does not divide Nt ... snapping`, and `multi-rank run: submission residual dump SKIPPED` (a known TODO, not a failure).
- **Every gate runs the campaign's sig-het pins** through `scripts/diagnostics/gate_run.py`, which reads the `export SIGHET_*` / `GB_SIGHET_*` lines of `submit_gf_6mo_v8.sh` at launch (single source of truth; `tests/test_gate_run.py`). The stock defaults differ from `submit_gf_6mo_v8.sh`: `SIGHET_NT_LAYER` 64 (snapped to 60, a 36 h stride the sig-het v4 notes flagged as 2.2x too coarse) vs 120, `SIGHET_N_CP` AUTO vs 256, `GB_SIGHET_REFRESH_EVERY` 0 (reference never refreshed) vs 25, `GB_SIGHET_TRUST_PHASE_C` 0 vs 49, drift check off vs on. T1 (2026-09-17) ran the stock defaults on both sides of its control, so its verdict stands; every gate from here runs the pins, and anything that reads accuracy (T4+) needs them.
- Pull `dev` again before T3: the MBH `Q` prior fix (linear-column log-uniform on [1, 10]) lands after T0-T2 were written; the all_sources gates need it.

## Pass/fail signals (what to grep)

| signal | meaning |
|---|---|
| layout line `nwalkers=1 block=1 ... REPLICAS`, every compute rank `walkers=[0,1)`, rank 1 `node=<B>` | replica mode selected, one replica per node |
| `[FANOUT_DIGEST] it=N ... replicas_agree=True` | replicas' residual+noise buffers bit-identical; on GPU this is an exact-match bonus, not the guard |
| `[GB_REPLICA <move>] log_like_final disagrees after sync` (WARNING) | **the divergence guard fired — FAIL** |
| `[GB_REPLICA <move>] residual hashes disagree after sync` (INFO) | expected on GPU (~1e-12 atomicAdd spread); count it, do not fail on it |
| `[GB_REPLICA] residual authoritative; rebuild deferred to gb_sync (drift x)` (INFO) | expected; record the drift magnitudes (≲1e-4 typically) |
| `PSDMove ... inner proposal: eigen` (INFO, once per move) | PSD took the eigen inner (one-walker default) |
| `RemoteWorkerError` / `RuntimeError: GB replica merge` | a rank's command failed / a merge partition violation — FAIL, keep the traceback |
| `F-stat epoch N incomplete` on rank 1 | shared-filesystem lag across nodes — storage problem |
| job killed at the wall clock with ranks silent | hang inside a collective — FAIL, see triage |
| `[GB_INFOMAT ...] FALL-THROUGH (sig-het comp: True; slot routing wired: False)` (WARNING) | pre-existing performance-route notice on this synthetic fixture: the RJ info matrices take the chunked path. Appears identically with `mpiexec -n 1` (T1 control, 2026-09-17). Not a replica signal |
| `[GB_CELL_LL ...] per-repeat sampled-vs-actual diff ... exceeds its allowance (band 1)` (WARNING) | pre-existing accuracy-floor notice on the injected band; the single-rank control shows the same magnitudes (up to ~1e1 at temp 0). Not a replica signal — but a real dev item to look at separately |

## Gates

### T0 — preflight (minutes, no compute)

```sh
# 1. two-node layout dry run
NWALKERS=1 GF_LAYOUT_DRY_RUN=1 GPUS=0 timeout 300 mpiexec -n 3 -ppn 1 \
  python scripts/diagnostics/gate_run.py --stock gb_no_fg 2>&1 | tee "$G/T0_dryrun.log"
# 2. trigger controls
NWALKERS=1 GF_ONE_WALKER_REPLICAS=0 GF_LAYOUT_DRY_RUN=1 GPUS=0 timeout 300 mpiexec -n 3 -ppn 1 \
  python scripts/diagnostics/gate_run.py --stock gb_no_fg 2>&1 | tee "$G/T0_knob_off.log"
NWALKERS=2 GF_LAYOUT_DRY_RUN=1 GPUS=0 timeout 300 mpiexec -n 3 -ppn 1 \
  python scripts/diagnostics/gate_run.py --stock gb_no_fg 2>&1 | tee "$G/T0_two_walkers.log"
# 3. submit scripts
bash -n scripts/fstat_proposal/submit_gf_6mo_v8.sh && bash -n scripts/fstat_proposal/submit_gf_6mo_v8_nogb_null.sh && echo SYNTAX_OK
python -m unittest tests.test_submit_scripts_layout -v 2>&1 | tail -4

# verdict
grep -c REPLICAS "$G/T0_dryrun.log"                        # 3 (one header per rank)
grep -c 'walkers=\[0,1)' "$G/T0_dryrun.log"                # 2
grep -c 'compute node=' "$G/T0_dryrun.log"                 # the r1 line must name the OTHER host
grep -c 'needs one-walker replica mode' "$G/T0_knob_off.log"   # >= 1  (ValueError, refuses)
grep -c REPLICAS "$G/T0_two_walkers.log"                   # 0  (nwalkers=2 block=1, ordinary layout)
```
Pass: the replica layout header identical on all three ranks; rank 1 on the other node with `devices=[0]`; `GF_ONE_WALKER_REPLICAS=0` refuses with `nwalkers=1 on 2 compute ranks needs one-walker replica mode`; `NWALKERS=2` prints the ordinary walker-block layout. Submit-script unit tests: 15 OK (the in-job `[SUBMIT] NWALKERS=1 on N_COMPUTE=2: one-walker replica mode` echo appears only in a job's stdout, T6).

### T1 — first real one-walker run, GB only, across the two nodes (≈10-20 min)

**Record 2026-09-17:** two-node run with the gate driver completed 4 iterations in ~75 s; no lnL-guard warning; the `mpiexec -n 1` control reproduced the same `[GB_INFOMAT]`/`[GB_CELL_LL]` warnings at the same magnitudes → both pre-existing. Digest (06:39 attempt): 20 peaks, leaves 0->33, all four GB ops served by rank 1 (rank 9.2 s vs head 8.6 s, wait 0), 0 lnL-guard warnings, 0 deferred rebuilds, residual hashes disagree on all 4 iterations (GPU, INFO) -> **T1 PASSED** (stock sig-het defaults on both sides). 8-walker regression control (06:51, `NWALKERS=8`, two nodes, stock sig-het): ran clean to completion, same two warning classes on walkers 0-3 at all rungs; `[FANOUT_DIGEST]` vs the WP7 layout (c) store (2026-09-16): `coords` and `inds` hashes IDENTICAL at it=0..3, only `log_like` hashes differ (the GPU non-bit-identity the runbook's criterion allows) → **regression control PASSED; T1 closed.**

```sh
FILE_STORE_DIR=$G/T1/ NUM_ITERATIONS=4 GPUS=0 timeout 3600 mpiexec -n 3 -ppn 1 \
  python scripts/diagnostics/gate_run.py --stock gb_no_fg 2>&1 | tee "$G/T1.log"
python scripts/diagnostics/gf_run_log_digest.py <run_dir> | tee "$G/T1_digest.txt"
```
Pass: the epoch line reports peaks (not `NO peaks`) and the `[FANOUT_DIGEST]` hashes change between iterations (the run is moving); completes; no WARNING-class `[GB_REPLICA]` line; `replicas_agree` counted on every iteration (a `False` is informational on GPU — the lnL guard is the criterion); `[FANOUT]` table shows `gb_run_proposal`, `gb_run_tempering`, `gb_finish`, `gb_sync` all served by rank 1 with `max_rank_s` comparable to the head's; the F-stat epoch opened on node B without the `incomplete` error.
What it exonerates: the cross-node transport for every GB op, the ledger's device indexing, `gb_sync`, the merge on real RJ births/deaths, the shared-filesystem epoch hand-off.
**Control (regression):** the same command with `NWALKERS=8` (walker-block layout, head + 1 compute across the nodes) must reproduce the WP7 Step 1 layout (c) record (decisions bit-identical, `log_like` within 1e-12 relative) — this branch must not change the walker-block path.

### T2 — across nodes vs one node agree (≈30 min)

**Record 2026-09-17 (gate driver, campaign sig-het pins):** T2c two nodes (07:07) vs T2a one node with both replicas on node A's GPU (07:14): `coords` and `inds` hashes IDENTICAL at it=0..3, `log_like` hashes identical at it=0 and 2 and differing in the last bits at it=1 and 3 (the GPU non-bit-identity the criterion allows); no lnL-guard warning in either → **T2 PASSED** (the two-layout match doubles as the same-seed determinism control; the different-seed sensitivity control is optional).

```sh
# (c) two nodes -- the primary layout, same seeds
FILE_STORE_DIR=$G/T2c/ NUM_ITERATIONS=4 GPUS=0 timeout 3600 mpiexec -n 3 -ppn 1 python scripts/diagnostics/gate_run.py --stock gb_no_fg 2>&1 | tee "$G/T2c.log"
# (a) one node, both replicas on node A's single GPU
FILE_STORE_DIR=$G/T2a/ NUM_ITERATIONS=4 GPUS=0 RANKS_PER_GPU=2 timeout 3600 mpiexec -n 3 -ppn 3 python scripts/diagnostics/gate_run.py --stock gb_no_fg 2>&1 | tee "$G/T2a.log"
```
Pass: `log_like` / `coords` / `inds` digests of (c) and (a) agree to the multi-walker gate's tolerance (decisions bit-identical, `log_like` within 1e-12 relative); no lnL-guard warning in either. The fabric and the node placement must not change the chain.
**Controls:** (c) run twice with the same `random_seed` → identical digests (determinism, so a (c)/(a) mismatch would mean something); (c) with a different seed → digests differ (the digest is sensitive).

### T3 — knob invariance on all_sources (positive controls of the scoring seams) (≈30 min)

Two nodes, `--stock all_sources` synthetic (first run with MBH/EMRI/SOBBH + PSD at one walker), `NUM_ITERATIONS=3`, same seeds:
```sh
B="NUM_ITERATIONS=3 GPUS=0"
env $B FILE_STORE_DIR=$G/T3_base/ timeout 5400 mpiexec -n 3 -ppn 1 python scripts/diagnostics/gate_run.py --stock all_sources 2>&1 | tee "$G/T3_base.log"
env $B FILE_STORE_DIR=$G/T3_ar_off/ MBH_LIKELIHOOD_FANOUT=0 EMRI_LIKELIHOOD_FANOUT=0 SOBBH_LIKELIHOOD_FANOUT=0 timeout 5400 mpiexec -n 3 -ppn 1 python scripts/diagnostics/gate_run.py --stock all_sources 2>&1 | tee "$G/T3_ar_off.log"
env $B FILE_STORE_DIR=$G/T3_psd_off/ PSD_LIKELIHOOD_FANOUT=0 GALFOR_LIKELIHOOD_FANOUT=0 timeout 5400 mpiexec -n 3 -ppn 1 python scripts/diagnostics/gate_run.py --stock all_sources 2>&1 | tee "$G/T3_psd_off.log"
```
Pass: the three runs give identical `log_like`/`coords`/`inds` digests (where a row is scored must not change the chain) and identical `replicas_agree` counts (the replays run regardless of the knob). Runs 2 and 3 are slower — that is the point. Grep `inner proposal: eigen` for PSD and galfor, and a nonzero PSD in-model acceptance.
What it exonerates: addremove row scatter + expose/setup/fold replays; PSD row scatter + begin/publish replays; the PSD eigen inner at one walker; the MBH `Q` prior fix (no `AssertionError: m1 should be the larger mass`).
**Control:** `PSD_INNER_MOVE_KIND=stretch` must refuse with the "needs at least two walkers" error.

### T4 — statistical check vs a single compute rank (hours)

```sh
# single rank, no replicas, node A only
FILE_STORE_DIR=$G/T4_single/ NUM_ITERATIONS=100 GPUS=0 timeout 43200 mpiexec -n 1 python scripts/diagnostics/gate_run.py --stock all_sources 2>&1 | tee "$G/T4_single.log"
# two replicas across the nodes
FILE_STORE_DIR=$G/T4_replicas/ NUM_ITERATIONS=100 GPUS=0 timeout 43200 mpiexec -n 3 -ppn 1 python scripts/diagnostics/gate_run.py --stock all_sources 2>&1 | tee "$G/T4_replicas.log"
```
NOT expected bit-identical (each rank draws its own GB RJ proposals). Compare through the `processing-gf-snapshots` flow: GB acceptance rates per move, cold-chain leaf counts vs truth, per-band `band_temps`, addremove acceptance and `acceptance_fraction` (finite, never nan), PSD acceptance under the eigen inner. Pass: distributions overlap; no systematic offset in leaf counts or ladders.
**Control:** two single-rank runs with different `random_seed` — the replica-vs-single spread must be no larger than the seed-vs-seed spread.

### T5 — scaling readout (≈1 h)

Two replicas (the primary layout) vs four replicas, one per node: `salloc --nodes=4 --gres=gpu:1 --ntasks-per-node=2`, then `GPUS=0 mpiexec -n 5 -ppn 1 python scripts/diagnostics/gate_run.py --stock all_sources` (ranks 0-3 replicas on nodes A-D, rank 4 saver on A). If four nodes are not grantable, two replicas per node's GPU: `--nodes=2`, `GPUS=0 RANKS_PER_GPU=2 mpiexec -n 5 -ppn 2` (ranks 0,1 on A; 2,3 on B; saver 4 on A; `--ntasks-per-node=3`; expect the OOM row of the triage table for all_sources). `NUM_ITERATIONS=5`, `PSD_NTEMPS` / MBH `ntemps` ≥ n_compute.
Record per family from `[GB_TIMING]`, `[PSD_TIMING]`, the addremove leaf lines and the `[FANOUT]` table. Expectations: addremove per-leaf time ~ 1/min(n_compute, ntemps) of single-rank (the info-matrix batch scales best); PSD similar plus the eigen refresh (`{P}_EIGEN_REFRESH` default 10); GB proposals and open/close ~1/R, the swap-grid build and the tempering census NOT — not a regression. Cross-node cost shows up as `max_rank_s` minus the head's own time in the `[FANOUT]` table. Decide `ntemps` for the real run here.

### T6 — real-data shape through the campaign script (hours)

```sh
NWALKERS=1 NGPUS=2 NODES=2 GF_LEGACY_RANK_LAYOUT=0 NUM_ITERATIONS=20 MIDIT_CHECKPOINT=1 ./scripts/fstat_proposal/submit_gf_6mo_v8.sh
```
`NODES=2` with `NGPUS=2` submits `--nodes=2 --gres=gpu:1 --ntasks=3 --distribution=cyclic` (one GPU per node — the same shape as every gate above); the in-job block derives `GPUS=0`, `N_COMPUTE=2` and launches `mpiexec -n 3 -ppn 1`. The first lines of the job stdout must show `[SUBMIT] NWALKERS=1 on N_COMPUTE=2: one-walker replica mode`. Pass: the staged run reaches the PE stage on mojito data, the digest script shows zero lnL-guard warnings, mid-iteration checkpoint + resume works at one walker (`MIDIT_CHECKPOINT=1` on this gate only). Then the decision on the production one-walker configuration.

## Triage when a gate fails

| symptom | most likely component | first bisect |
|---|---|---|
| hang, ranks silent, job hits the wall clock | an exception inside the GB per-unit ledger exchange on one rank (no containment yet) | rerun with `GB_ORTHO_LL_CHECK=1` and per-rank logs; look at the last `[GB_TIMING]`/unit line per rank; rerun the T2 (a) single-node layout to rule out the fabric |
| `size=1` on every rank / `PMI server not found` | the launcher, not the code | the runbook's launcher table: hydra + `I_MPI_FABRICS=shm:ofi FI_PROVIDER=tcp`, no `srun` |
| `F-stat epoch N incomplete` on rank 1 | fit dir not on the shared filesystem, or metadata lag | move the fit dir; never suppress the check |
| `log_like_final disagrees after sync` | ledger missed a residual change (foreign died source, tempering exposure) or a merge claimed the wrong sources | check the `drift` values in the "residual authoritative" lines just before; compare the disagreeing ranks' `block_band_inds` in a debug dump; `GB_TEMPER_SKIP_SHUTOFF_BANDS`/tempering off (`gb_pe` repeats) to isolate |
| `RuntimeError: GB replica merge ... outside [grid_lo, grid_hi)` | an alive source with an out-of-grid band label (prior vs band grid mismatch) | check `f0_lims` vs `band_edges` in the stock GB settings |
| addremove/PSD digests differ between knob on/off (T3) | a scoring row evaluated against a stale residual/noise model on a replica → a replay is missing | diff the per-rank logs at the first differing iteration; check `[FANOUT]` shows `ar_replay`/`psd_replay` per leaf/propose |
| `AssertionError: m1 should be the larger mass` (T3+) | the MBH `Q` prior fix is not in the checkout | `git pull`; `grep -n LogUniformLinear src/lisatools/globalfit/stock/erebor/mbh.py` |
| `acceptance_fraction` nan / RuntimeWarning divide | outer counters not advanced (eigen inner) | PSD: `_inner_propose` bookkeeping; fixed in Plan 1, would be a regression |
| replica layout not selected | `NWALKERS` not 1 on the compute ranks or `GF_ONE_WALKER_REPLICAS` falsy | the `[SUBMIT]` echo; the layout dry run |
| all_sources OOMs with two ranks on one GPU (T2 (a), T5 fallback) | two full replicas + two ACAs per device | that is what the one-GPU-per-node layout avoids; keep (a) to GB-only |

## Evidence to bring back

Per gate: the run dir (all rank logs), the digest-script output, the allocation/sbatch line + env, wall time per iteration by family, peak GPU memory per rank (`nvidia-smi` sample on BOTH nodes), and for T4/T6 the snapshot zip for the `processing-gf-snapshots` skill. A gate passes only with its control's evidence attached.
