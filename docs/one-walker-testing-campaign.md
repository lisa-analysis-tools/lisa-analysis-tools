# One-walker replica mode — cluster testing campaign

Branch `one-walker-replicas` (worktree `LISAanalysistools-onewalker`, base `dev` `aec22834`).
Laptop evidence already in hand: GB one-walker/two-replica smoke (4 `gb_sync` rounds, agreeing hashes), parity arm, 4-walker two-rank arm, 545-test sweep — all green; a one-walker noise-smoke arm (PSD eigen inner + row scatter + replays) is the last laptop item. Everything below runs on the cluster and is what actually proves the GPU path.

Design authority: `docs/superpowers/specs/2026-09-16-one-walker-replicas-design.md`. Launcher facts and the per-step commands: `docs/multirank-cluster-gates.md` ("One-walker replica mode gate", from line 370). Logs digest: `python scripts/diagnostics/gf_run_log_digest.py <run_dir>` (prints the `[FANOUT]` load-balance table, the `[FANOUT_DIGEST] replicas` agreement count and the `[GB_REPLICA]` counters).

## Rules for every gate

- Every run exports the common bundle: `DATA_MODE=synthetic NUM_ITERATIONS=<N> MIDIT_CHECKPOINT=0 MAKE_DIAGNOSTIC_PLOTS=0 GF_FANOUT_DIGEST=1 GF_LEGACY_RANK_LAYOUT=0` plus `NWALKERS=1` for the one-walker arms, and the launcher bundle `I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi FI_PROVIDER=tcp` (multi-node only).
- **Wall-clock limit on every run** (`--time` on the sbatch, or `timeout` on the mpiexec). A rank error inside the GB per-unit exchange hangs the other ranks instead of crashing; a hang IS a failure signal, not a slow run.
- Keep every rank's log (`globalfit_run.log`, `globalfit_run.rank<k>.log`) and the digest-script output per run; name run dirs by gate and layout (`T1a`, `T2b`, ...).
- An invariance claim needs its control (project rule). Each gate below names its control; a gate without its control passing is not a pass.
- Order matters: do not run a later gate until the earlier one's pass criterion is met. T0 and T1 together take well under an hour.

## Pass/fail signals (what to grep)

| signal | meaning |
|---|---|
| layout line `nwalkers=1 block=1 ... REPLICAS`, every compute rank `walkers=[0,1)` | replica mode selected |
| `[FANOUT_DIGEST] it=N ... replicas_agree=True` on every iteration | replicas' residual+noise buffers bit-identical (CPU) or the run is on GPU and this is the exact-match bonus |
| `[GB_REPLICA <move>] log_like_final disagrees after sync` (WARNING) | **the divergence guard fired — FAIL** |
| `[GB_REPLICA <move>] residual hashes disagree after sync` (INFO) | expected on GPU (~1e-12 atomicAdd spread); count it, do not fail on it |
| `[GB_REPLICA] residual authoritative; rebuild deferred to gb_sync (drift x)` (INFO) | expected; record the drift magnitudes (should be ≲1e-4 typically) |
| `PSDMove ... inner proposal: eigen` (INFO, once per move) | PSD took the eigen inner (one-walker default) |
| `RemoteWorkerError` / `RuntimeError: GB replica merge` | a rank's command failed / a merge partition violation — FAIL, keep the traceback |
| job killed at the wall clock with ranks silent | hang inside a collective — FAIL, see triage |

## Gates

### T0 — preflight (minutes, no compute)
1. Layout dry run (Step A): `GF_LAYOUT_DRY_RUN=1 GPUS=0,1 mpiexec -n 3 -ppn 3 python scripts/run_global.py --stock gb_no_fg`. Pass: the replica layout line, identical `digest()` on every rank.
2. **Control (trigger):** the same with `GF_ONE_WALKER_REPLICAS=0` must refuse with `ValueError: nwalkers=1 on 2 compute ranks needs one-walker replica mode, which GF_ONE_WALKER_REPLICAS=0 disables`; with `NWALKERS=2` it must print the ordinary walker-block layout (`block=1`, no `REPLICAS`).
3. Submit-script check: `bash -n` on both campaign scripts, then `python -m unittest tests.test_submit_scripts_layout -v` in the cluster env (it runs the extracted in-job block under bash and the dispatch block against a stub `sbatch`). The `[SUBMIT] NWALKERS=1 on N_COMPUTE=...: one-walker replica mode` echo itself only appears in a job's stdout (T6). The scripts read `NWALKERS` from the submitting shell (`export NWALKERS=${NWALKERS:-10}`, carried by the dispatch's `--export=ALL`); a plain resubmit stays at 10.

### T1 — first real one-walker run, GB only, shared GPU (≈10-20 min)
Layout (a): `GPUS=0 RANKS_PER_GPU=2 mpiexec -n 3 -ppn 3 python scripts/run_global.py --stock gb_no_fg`, `NUM_ITERATIONS=4`.
Pass: completes; no WARNING-class `[GB_REPLICA]` line; `replicas_agree` counted for every iteration (True on every iteration is expected here too: both ranks share one GPU and see the same atomics order only by luck — treat `False` as informational at T1 and check the lnL guard instead); `[FANOUT]` table shows `gb_run_proposal`, `gb_run_tempering`, `gb_finish`, `gb_sync` all served by both ranks with `max_rank_s` comparable.
What it exonerates: the transport, the ledger's device indexing, `gb_sync`, the merge on real RJ births/deaths.
**Control (regression):** the multi-walker Step 1 (a) command with `NWALKERS=8` from the existing runbook, same seeds, must reproduce the digests recorded at WP7 (this branch must not change the walker-block path).

### T2 — the three layouts agree (≈30 min)
Layouts (a), (b) `GPUS=0,1 mpiexec -n 3 -ppn 3`, (c) two nodes `GPUS=0 mpiexec -n 3 -ppn 1`, same seeds, `NUM_ITERATIONS=4`.
Pass: `log_like` / `coords` / `inds` digests agree across (a),(b),(c) to the multi-walker gate's tolerance; no lnL-guard warning anywhere.
**Control:** (c) with the two compute ranks on the same node (`-ppn 3`) vs across nodes — identical digests prove the fabric does not change results.

### T3 — knob invariance (positive controls of the scoring seams) (≈30 min)
Layout (b), `--stock all_sources` synthetic (first run with MBH/EMRI/SOBBH + PSD at one walker), `NUM_ITERATIONS=3`:
1. baseline (all knobs default);
2. `MBH_LIKELIHOOD_FANOUT=0 EMRI_LIKELIHOOD_FANOUT=0 SOBBH_LIKELIHOOD_FANOUT=0` — head scores every addremove row itself;
3. `PSD_LIKELIHOOD_FANOUT=0` (and `GALFOR_` if sampled).
Pass: 1, 2 and 3 give identical `log_like`/`coords`/`inds` digests (where a row is scored must not change the chain) and identical `replicas_agree` counts (the replays run regardless of the knob — decision 5, fixed in the Plan 1 review). Runtime of 2 and 3 is slower — that is the point.
What it exonerates: addremove row scatter + expose/setup/fold replays; PSD row scatter + begin/publish replays; the PSD eigen inner at one walker (grep the inner-kind line).
**Control:** `PSD_INNER_MOVE_KIND=stretch` must refuse with the "needs at least two walkers" error.

### T4 — statistical check vs a single compute rank (hours)
`GPUS=0 mpiexec -n 1` (no replicas) vs layout (b), `--stock all_sources` synthetic, `NUM_ITERATIONS=50-100`.
NOT expected bit-identical (each rank draws its own GB RJ proposals). Compare through the `processing-gf-snapshots` flow: GB acceptance rates per move, cold-chain leaf counts vs truth, per-band `band_temps`, addremove acceptance and `acceptance_fraction` (must be finite, never nan), PSD acceptance under the eigen inner. Pass: distributions overlap; no systematic offset in leaf counts or ladders.
**Control:** two single-rank runs with different `random_seed` — the replica-vs-single spread must be no larger than the seed-vs-seed spread.

### T5 — scaling readout (≈1 h)
Layout (b) vs 4 compute ranks on 2 nodes (`mpiexec -n 5 -ppn ...`, see the runbook's NGPUS=4 dispatch), `--stock all_sources`, `NUM_ITERATIONS=5`, `PSD_NTEMPS` / MBH `ntemps` ≥ n_compute.
Record per family from `[GB_TIMING]`, `[PSD_TIMING]`, the addremove leaf lines and the `[FANOUT]` table. Expectations (decision 13 + the Plan 2 caveats): addremove per-leaf time ~ 1/min(n_compute, ntemps) of the single-rank time (the info-matrix batch scales best); PSD similar plus the eigen refresh cost (`[PSD_TIMING]` `sens_refresh`/`sample` buckets; `{P}_EIGEN_REFRESH` default 10); GB: proposals and open/close ~1/R, the swap-grid build and the tempering census NOT — do not read that as a regression. Decide `ntemps` for the real run here.

### T6 — real-data shape (the campaign script) (hours)
`NWALKERS=1 GF_LEGACY_RANK_LAYOUT=0 ./submit_gf_6mo_v8.sh` (the exemption is in place and `NWALKERS` is env-overridable; `N_COMPUTE` from `NGPUS`; the first lines of the job stdout must show the `[SUBMIT] NWALKERS=1 on N_COMPUTE=...: one-walker replica mode` echo), short `NUM_ITERATIONS`, mojito data. Pass: the staged run reaches the PE stage, the digest script shows zero lnL-guard warnings, mid-iteration checkpoint + resume works at one walker (`MIDIT_CHECKPOINT=1` on this gate only). Then the decision on the production one-walker configuration.

## Triage when a gate fails

| symptom | most likely component | first bisect |
|---|---|---|
| hang, ranks silent, job hits the wall clock | an exception inside the GB per-unit ledger exchange on one rank (no containment yet) | rerun with `GB_ORTHO_LL_CHECK=1` and per-rank logs; look at the last `[GB_TIMING]`/unit line per rank; try `RANKS_PER_GPU=1` layout (b) to rule out the shared-GPU context |
| `log_like_final disagrees after sync` | ledger missed a residual change (foreign died source, tempering exposure) or a merge claimed the wrong sources | check the `drift` values in the "residual authoritative" lines just before; compare the disagreeing ranks' `block_band_inds` in a debug dump; `GB_TEMPER_SKIP_SHUTOFF_BANDS`/tempering off (`gb_pe` repeats) to isolate |
| `RuntimeError: GB replica merge ... outside [grid_lo, grid_hi)` | an alive source with an out-of-grid band label (prior vs band grid mismatch) | check `f0_lims` vs `band_edges` in the stock GB settings |
| addremove/PSD digests differ between knob on/off (T3) | a scoring row evaluated against a stale residual/noise model on a replica → a replay is missing | diff the per-rank logs at the first differing iteration; check `[FANOUT]` shows `ar_replay`/`psd_replay` per leaf/propose |
| `acceptance_fraction` nan / RuntimeWarning divide | outer counters not advanced (eigen inner) | PSD: `_inner_propose` bookkeeping; fixed in Plan 1, would be a regression |
| replica layout not selected | `NWALKERS` not 1 on the compute ranks (submit-script rounding) or `GF_ONE_WALKER_REPLICAS` falsy | the `[SUBMIT]` echo; the layout dry run |
| all-sources run OOMs on one GPU with `RANKS_PER_GPU=2` | two full replicas + two ACAs per device | use layout (b)/(c): one rank per GPU |

## Evidence to bring back

Per gate: the run dir (all rank logs), the digest-script output, the sbatch script + env, wall time per iteration by family, peak GPU memory per rank (`nvidia-smi` sample), and for T4/T6 the snapshot zip for the `processing-gf-snapshots` skill. A gate passes only with its control's evidence attached.
