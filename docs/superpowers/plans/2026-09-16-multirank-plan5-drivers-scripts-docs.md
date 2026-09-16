# Multi-rank Plan 5: settings knobs, drivers, submit scripts, diagnostics, docs (WP6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the walker-block layout launchable and observable: the two rank↔GPU settings become real knobs (with a single-compute-rank AUTO rule that keeps today's `-n 1` multi-GPU runs bit-identical), the drivers gain a layout dry run, the campaign submit scripts size `--ntasks`/`mpiexec -n` from the GPU count (2-node × 2-GPU allocations on `gpu-80-spot` for NGPUS=4) while keeping today's campaign resubmits on the legacy layout until the cluster gates pass, the diagnostics read every rank's log, and the docs describe the new roles. Plus the two cluster-gate tools the spec names (`GF_FANOUT_DIGEST`, `gf_state_digest.py`) and the WP7 runbook.

**Architecture:** Settings (`EreborGeneralSettings.gpus_per_rank: int | None = None` (AUTO), `ranks_per_gpu: int = 1`, env `GPUS_PER_RANK` / `RANKS_PER_GPU`) flow through `prepare_rank` into `build_layout`, where `gpus_per_rank=None` resolves per node to "the whole pool when exactly one compute rank lives on the node, else 1" — so `GPUS=0,1` at `-n 1` keeps driving both devices in-process exactly as today. Drivers print `layout.describe()` and stop before the build under `GF_LAYOUT_DRY_RUN=1`. The submit scripts compute `N_COMPUTE = NGPUS * RANKS_PER_GPU / GPUS_PER_RANK` (with the AUTO rule that is `NGPUS` at the defaults), pass `--ntasks=$((N_COMPUTE+1))` through the existing `exec sbatch` re-invocation, use `--nodes=2 --distribution=cyclic` + `srun` for NGPUS=4, check `NWALKERS % N_COMPUTE`, and export `GF_LEGACY_RANK_LAYOUT` (default 1 at NGPUS=2 until WP7 flips it; forced 0 at NGPUS=4). The diagnostics discover `globalfit_run*.log` (all ranks) instead of the head's file only. A head-side `[FANOUT_DIGEST]` line and `scripts/diagnostics/gf_state_digest.py` give WP7 its bit-identity comparators.

**Tech Stack:** Python 3.12 (dataclasses, `env_default`), bash (sbatch/srun), numpy hashing, unittest.

**Spec:** `docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md` — WP6, Verification (items 4-7), "Semantics that change only when n_compute > 1", Risks. Site map with anchors: `plan5-wp6-sitemap.md` (in this plan's SDD workspace; anchors at HEAD f8ce6cc7).

## Global Constraints

- Single-process behaviour BIT-IDENTICAL, INCLUDING `-n 1` with several GPUs: the AUTO rule gives the lone compute rank the whole per-node pool (today's in-process sharding). An explicit `GPUS_PER_RANK` overrides it.
- The running 6-month campaign's resubmits (`NGPUS=2`, `mpiexec -n 3`) must keep TODAY's roles after the eventual merge: the two campaign scripts export `GF_LEGACY_RANK_LAYOUT=${GF_LEGACY_RANK_LAYOUT:-1}` with a loud `[SUBMIT]` line; flipping the default to 0 is a WP7 decision after the cluster gates. `NGPUS=4` forces 0 (legacy cannot span nodes).
- `_HEADLINE_KNOBS` never gains rank/fanout/layout fields (ledger ruling); the new fields are plain `fit.general.*` attributes following the env-default rule (`stock/base.py` `env_default`; names = capitalized attribute names).
- Diagnostics edits are minimal (the main checkout carries UNCOMMITTED edits to `scripts/diagnostics/*` from another session — keep the diff to the log-discovery change plus a new parser function, no reformatting, to ease the eventual merge).
- Do not edit `LISAanalysistools/multinode_gpu_handoff.md` (untracked, lives in the main checkout, outside this branch): the closing note goes into this branch's docs and the user is told where the old doc is.
- Line length <= 100 on added Python lines; bash scripts pass `bash -n`.
- Tests: CPU, tiny fixtures, one python process at a time; never run tests/test_gbspecial_flow.py; batch regression lists <= 6 modules per process.
- Commit per task with the given message and the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`; never commit `.superpowers/`, `.wtenv/`; no `git push`.

---

### Task 1: The two settings knobs + the single-compute-rank AUTO pool rule

**Files:**
- Modify: `src/lisatools/globalfit/stock/erebor/fit.py` (`EreborGeneralSettings`, next to `nwalkers` ~58-61)
- Modify: `src/lisatools/globalfit/communication/ranks.py` (`build_layout` ~185-300; `prepare_rank` ~459-467)
- Test: `tests/test_rank_layout.py` (append), `tests/test_stock_globalfit.py` (append)

**Interfaces:**
- Produces: `EreborGeneralSettings.gpus_per_rank: typing.Optional[int]` (default `None` = AUTO; env `GPUS_PER_RANK`, cast `int`), `EreborGeneralSettings.ranks_per_gpu: int` (default 1; env `RANKS_PER_GPU`); `build_layout(..., gpus_per_rank=None, ...)` accepting `None`; `prepare_rank` passing the fields through unchanged (`None` stays `None`).
- AUTO semantics (`gpus_per_rank is None`): per node, `k_node = len(pool)` when exactly ONE compute rank lives on that node (and `ranks_per_gpu == 1` and the pool is non-empty), else `k_node = 1`. The capacity check and the blocked device assignment use `k_node`. `WalkerBlockLayout.gpus_per_rank` records the resolved value per layout (`None` → the resolved per-node value if uniform, else `None`; document). `describe()` prints the AUTO resolution in its header line.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rank_layout.py`:
```python
class AutoGpusPerRankTest(unittest.TestCase):
    """gpus_per_rank=None: a lone compute rank on a node owns the whole pool (today's -n 1)."""

    def test_single_rank_owns_the_whole_pool(self):
        lay = FakeWorld(1).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(lay.local_gpus(0), [0, 1])
        self.assertEqual(lay.n_compute, 1)

    def test_single_rank_explicit_one_narrows_to_the_first_device(self):
        lay = FakeWorld(1).run(
            lambda r, c: build_layout(c, 4, [0, 1], legacy=False, gpus_per_rank=1)
        )[0]
        self.assertEqual(lay.local_gpus(0), [0])

    def test_two_compute_ranks_on_a_two_gpu_node_get_one_device_each(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual((lay.local_gpus(0), lay.local_gpus(1)), ([0], [1]))

    def test_size2_on_one_gpu_still_demotes_rank1_and_head_owns_the_pool(self):
        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            lay = FakeWorld(2).run(lambda r, c: build_layout(c, 4, [0], legacy=False))[0]
        self.assertEqual(lay.role_of(1), RankRole.SAVER)
        self.assertEqual(lay.local_gpus(0), [0])

    def test_head_plus_saver_on_two_gpus_head_owns_both(self):
        # -n 2 on a 2-GPU pool: rank 1 is a COMPUTE rank (two compute ranks), not a saver
        lay = FakeWorld(2).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(lay.n_compute, 2)
        self.assertEqual((lay.local_gpus(0), lay.local_gpus(1)), ([0], [1]))

    def test_prepare_rank_passes_none_through(self):
        # the fixture's general block leaves gpus_per_rank unset -> AUTO
        fit = _Fit(_General(nwalkers=4, gpus=[0, 1]))
        fit.general.gpus_per_rank = None
        lay = FakeWorld(1).run(lambda r, c: prepare_rank(fit, c, device_count_fn=lambda: 2))[0]
        self.assertEqual(lay.local_gpus(0), [0, 1])
        self.assertEqual(fit.general.gpus, [0, 1])
```
(Adapt `_Fit`/`_General` to the fixture names at tests/test_rank_layout.py ~240-253; `prepare_rank` may pin via `CUDA_VISIBLE_DEVICES` in `visible` mode — pass the `environ={}`/`device_count_fn` arguments the existing `PrepareRankTest` uses so no real CUDA call happens, and assert `fit.general.gpus` is the rank-local list the mode yields: in `visible` mode with two devices it is `[0, 1]`.)

Append to `tests/test_stock_globalfit.py` (next to `test_compute_backend_env_knobs`):
```python
    def test_rank_layout_env_knobs(self):
        with _EnvGuard(GPUS_PER_RANK="2", RANKS_PER_GPU="1"):
            fit = erebor.get_stock("gb_no_fg")
            self.assertEqual(fit.general.gpus_per_rank, 2)
            self.assertEqual(fit.general.ranks_per_gpu, 1)
        with _EnvGuard(GPUS_PER_RANK=None, RANKS_PER_GPU=None):
            fit = erebor.get_stock("gb_no_fg")
            self.assertIsNone(fit.general.gpus_per_rank)  # AUTO
            self.assertEqual(fit.general.ranks_per_gpu, 1)
        self.assertNotIn("gpus_per_rank", erebor.get_stock("gb_no_fg").describe())
```

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p5t1.log python -m unittest tests.test_rank_layout tests.test_stock_globalfit.KnobTest -v; tail -n 25 .wtenv/p5t1.log`
Expected: the new tests FAIL (`local_gpus(0) == [0]`; no `gpus_per_rank` field).

- [ ] **Step 3: Implement**

`fit.py`, after `nwalkers`:
```python
    #: multi-rank layout (design spec 2026-09-15, Decision 2). ``gpus_per_rank``
    #: None = AUTO: a lone compute rank on a node drives the node's whole
    #: ``gpus`` pool (today's in-process multi-GPU run at ``-n 1``); several
    #: compute ranks on a node get one device each. An int pins it. At most one
    #: of the two may exceed 1. Plain attributes, never headline knobs.
    gpus_per_rank: typing.Optional[int] = dataclasses.field(
        default_factory=env_default("GPUS_PER_RANK", None, int)
    )
    ranks_per_gpu: int = dataclasses.field(default_factory=env_default("RANKS_PER_GPU", 1, int))
```
(`env_default(var, None, int)` must return `None` when unset — read `env_default`/`env_resolve` in stock/base.py; if the unset path returns the default untouched it already works.)

`ranks.py` `build_layout`: signature `gpus_per_rank=None`; replace the `k, m = int(gpus_per_rank), int(ranks_per_gpu)` block with
```python
    m = int(ranks_per_gpu)
    k_explicit = None if gpus_per_rank is None else int(gpus_per_rank)
    if m < 1 or (k_explicit is not None and k_explicit < 1):
        raise ValueError("gpus_per_rank and ranks_per_gpu must both be >= 1")
    if k_explicit is not None and k_explicit > 1 and m > 1:
        raise ValueError("at most one of gpus_per_rank / ranks_per_gpu may exceed 1")
```
and resolve `k` per node inside the placement loop:
```python
        n_comp_here = len(comp_here)
        if k_explicit is not None:
            k = k_explicit
        elif pool and n_comp_here == 1 and m == 1:
            k = len(pool)  # AUTO: a lone compute rank drives the whole pool (today's -n 1)
        else:
            k = 1
```
The size-2 fallback check (`len(pool) * m // k < 2`) runs BEFORE the node loop — evaluate it with `k = k_explicit if k_explicit is not None else 1` (AUTO never applies to a 2-compute-rank layout). Record the resolved value: `resolved_k = k` when uniform across nodes, else `None`; pass `gpus_per_rank=resolved_k` into `WalkerBlockLayout` and add `"gpus_per_rank=AUTO->{k}"` to the `describe()` header when AUTO resolved. Keep `k > 1` device assignment `pool[i*k:(i+1)*k]` (at n_comp_here == 1 that is the whole pool).

`prepare_rank`: `gpus_per_rank=getattr(general, "gpus_per_rank", None)` (no `or 1`), `ranks_per_gpu=int(getattr(general, "ranks_per_gpu", 1) or 1)`; update the docstring.

`select_rank_device` already handles a multi-device list (`CUDA_VISIBLE_DEVICES=<comma list>`; `general.gpus = [0..k-1]`) — confirm by reading it (~330-400) and its tests; `prepare_rank` with two devices in `visible` mode must give `general.gpus == [0, 1]`.

- [ ] **Step 4: Run to verify pass + regression**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p5t1b.log python -m unittest tests.test_rank_layout tests.test_stock_globalfit tests.test_run_multirank_helpers tests.test_driver_scripts_layout -v; tail -n 25 .wtenv/p5t1b.log`
Expected: all OK. Then `RUN_GF_SMOKE=1 ... tests.test_multirank_blank_smoke tests.test_multirank_noise_smoke` → OK.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/stock/erebor/fit.py src/lisatools/globalfit/communication/ranks.py tests/test_rank_layout.py tests/test_stock_globalfit.py
git commit -m "feat(layout): GPUS_PER_RANK / RANKS_PER_GPU knobs; a lone compute rank owns its node's whole GPU pool (today's -n 1)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `GF_LAYOUT_DRY_RUN=1` in both drivers + the `fanout.stop()` guard

**Files:**
- Modify: `src/lisatools/globalfit/communication/ranks.py` (new helper), `scripts/run_global.py` (stock path after `prepare_rank`), `scripts/fstat_proposal/run_combined_staged.py` (after `layout = prepare_rank(...)`), `src/lisatools/globalfit/run.py` (the HEAD `finally` in `run_global_fit`)
- Test: `tests/test_rank_layout.py` (append), `tests/test_driver_scripts_layout.py` (append)

**Interfaces:**
- Produces: `ranks.layout_dry_run(layout, comm, *, environ=None, out=print) -> bool`: when `environ.get("GF_LAYOUT_DRY_RUN", "0") == "1"` prints `layout.describe()` (every rank prints its own line-prefixed copy; the head prints the full table) and returns True; else returns False. The drivers `sys.exit(0)` when it returns True (before `fit.build()`). Layout errors already raise inside `prepare_rank` (non-zero exit through the abort hook).
- run.py: the `finally` calls `self.fanout.stop()` inside its own `try/except Exception: logger.exception("fanout.stop() failed during shutdown")` so a failing stop never supplants the original exception.

- [ ] **Step 1: Tests**

`tests/test_rank_layout.py`:
```python
class LayoutDryRunTest(unittest.TestCase):
    def test_prints_and_returns_true_only_when_armed(self):
        from lisatools.globalfit.communication.ranks import layout_dry_run

        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        lines = []
        self.assertTrue(layout_dry_run(lay, None, environ={"GF_LAYOUT_DRY_RUN": "1"}, out=lines.append))
        self.assertTrue(any("head" in ln for ln in lines))
        self.assertFalse(layout_dry_run(lay, None, environ={}, out=lines.append))
```
`tests/test_driver_scripts_layout.py`: extend the text tripwire to assert both scripts contain `layout_dry_run(` and `GF_LAYOUT_DRY_RUN` (in a comment or the help epilog).

- [ ] **Step 2: Run to verify failure**, **Step 3: Implement** (helper in ranks.py; drivers: `if layout_dry_run(layout, MPI.COMM_WORLD): sys.exit(0)` right after `prepare_rank`; `run_global.py`'s epilog gains one line "GF_LAYOUT_DRY_RUN=1: print the rank layout from every rank and exit before the build"; run.py guard), **Step 4: Run** `tests.test_rank_layout tests.test_driver_scripts_layout tests.test_run_multirank_helpers` + `RUN_GF_SMOKE=1 tests.test_multirank_blank_smoke` → OK.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/communication/ranks.py scripts/run_global.py scripts/fstat_proposal/run_combined_staged.py src/lisatools/globalfit/run.py tests/test_rank_layout.py tests/test_driver_scripts_layout.py
git commit -m "feat(drivers): GF_LAYOUT_DRY_RUN preflight; guarded fanout.stop() on shutdown

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Submit scripts (6mo main + null): rank count from the GPU count, 2-node NGPUS=4, legacy default

**Files:**
- Modify: `scripts/fstat_proposal/submit_gf_6mo_v8.sh` (NGPUS dispatch 397-416; `_NGPUS_EFF`/`GPUS` 527-532; `NWALKERS` 578; `mpiexec -n 3` 2640; the `#SBATCH --ntasks=3` comment line 374; the export block ~500-522), `scripts/fstat_proposal/submit_gf_6mo_v8_nogb_null.sh` (same sites at ~+60 lines)
- Test: `tests/test_submit_scripts_layout.py` (create)

Edits (both scripts identically; use the site map §3 anchors):
1. In the pre-submit dispatch block: `GPUS_PER_RANK=${GPUS_PER_RANK:-}` (empty = AUTO), `RANKS_PER_GPU=${RANKS_PER_GPU:-1}`; `_k=${GPUS_PER_RANK:-1}` for the count; `N_COMPUTE=$(( NGPUS * RANKS_PER_GPU / _k ))`; `NTASKS=$(( N_COMPUTE + 1 ))`.
2. `case "${NGPUS}"`: `2)` → `_NGPU_PART=gpu-80-spot; _NODES=1; _GRES=gpu:2`; `4)` → `_NGPU_PART=gpu-80-spot; _NODES=2; _GRES=gpu:2` (two nodes × two GPUs — the cluster's real 4-GPU allocation, design spec "Context"); delete the wrong "CANNOT span nodes / two 2-GPU jobs" warning block; new `[SUBMIT]` lines: `NGPUS=… -> sbatch --partition=… --gres=… --nodes=… --ntasks=… (N_COMPUTE=… compute ranks + 1 saver)`.
3. `exec sbatch --partition="${_NGPU_PART}" --gres="${_GRES}" --nodes="${_NODES}" --ntasks="${NTASKS}" --export=ALL,NGPUS=...,GPUS_PER_RANK=...,RANKS_PER_GPU=...,GF_LEGACY_RANK_LAYOUT=... "$0" "$@"`; `--distribution=cyclic` when `_NODES=2`.
4. Legacy default: before the dispatch, `if [ "${NGPUS}" = "4" ]; then GF_LEGACY_RANK_LAYOUT=0; else GF_LEGACY_RANK_LAYOUT=${GF_LEGACY_RANK_LAYOUT:-1}; fi; export GF_LEGACY_RANK_LAYOUT` with a loud `[SUBMIT]` line: legacy=1 → "TODAY's roles (one sampling rank drives all local GPUs; rank 1 stopped spare; rank 2 saver) — set GF_LEGACY_RANK_LAYOUT=0 for the walker-block layout after the WP7 gates"; legacy=0 → "walker-block layout: N_COMPUTE compute ranks + saver". Also export it in-job (the `--export=ALL` carries it; add `export GF_LEGACY_RANK_LAYOUT` near `OMP_NUM_THREADS`).
5. In-job: `_NGPUS_EFF` from `SLURM_GPUS_ON_NODE` stays the per-node pool; compute `N_COMPUTE_EFF=$(( ${SLURM_NNODES:-1} * _NGPUS_EFF * RANKS_PER_GPU / _k ))` under legacy=0 and `N_COMPUTE_EFF=1` under legacy=1 (legacy = one compute rank); the launch line becomes `if [ "${SLURM_NNODES:-1}" -gt 1 ]; then srun --ntasks="${SLURM_NTASKS}" --distribution=cyclic python …; else mpiexec -n "${SLURM_NTASKS:-3}" python …; fi` — under legacy the header's ntasks is still 3 (`N_COMPUTE=1` + spare + saver: keep `NTASKS=3` when legacy=1 so `mpiexec -n 3` is unchanged byte-for-byte for today's campaign).
6. `NWALKERS` divisibility: after `export NWALKERS=10`: `if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && [ $(( NWALKERS % N_COMPUTE_EFF )) -ne 0 ]; then echo "[SUBMIT] NWALKERS=${NWALKERS} is not a multiple of N_COMPUTE=${N_COMPUTE_EFF}; using NWALKERS=$(( (NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF )) (user decision at the first 4-GPU launch)"; export NWALKERS=$(( ... )); fi` (10 → 12 at N_COMPUTE=4).
7. Null script: same edits; keep the `#SBATCH` header at `--gres=gpu:2 --ntasks=3` (the manual `sbatch --gres=gpu:1` override still works: `-n 3` on a 1-GPU pool under legacy=1 is today's behaviour; under legacy=0 it errors at size >= 3 with insufficient GPUs — document in the `[SUBMIT]` line).

Test `tests/test_submit_scripts_layout.py`: `bash -n` both scripts; run each script's dispatch block in a sandbox: `SLURM_JOB_ID` unset, `sbatch` shadowed by a stub on `PATH` that prints its argv and exits 0, `NGPUS=2` → argv contains `--ntasks=3 --nodes=1 --partition=gpu-80-spot` and `GF_LEGACY_RANK_LAYOUT=1`; `NGPUS=4` → `--ntasks=5 --nodes=2 --gres=gpu:2 --partition=gpu-80-spot --distribution=cyclic` and `GF_LEGACY_RANK_LAYOUT=0`; `NGPUS=2 GF_LEGACY_RANK_LAYOUT=0` → `--ntasks=3` with legacy 0. (Use `subprocess.run(["bash", script], env=..., capture_output=True)` with `PATH` prepended by the stub dir; the scripts `exec sbatch` before any heavy work, so this is fast and needs no python env beyond the test.)

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p5t3.log python -m unittest tests.test_submit_scripts_layout -v` → OK. Commit:
```bash
git add scripts/fstat_proposal/submit_gf_6mo_v8.sh scripts/fstat_proposal/submit_gf_6mo_v8_nogb_null.sh tests/test_submit_scripts_layout.py
git commit -m "feat(submit): rank count from the GPU count; NGPUS=4 = gpu-80-spot 2 nodes x 2 GPUs; legacy layout stays the campaign default until WP7

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Diagnostics read every rank; `[FANOUT_DIGEST]` + `gf_state_digest.py` for the cluster gates

**Files:**
- Modify: `scripts/diagnostics/gf_monitor_gen.py` (log discovery ~292-297: exact `globalfit_run.log` → glob `globalfit_run*.log`, concatenated in rank order for the regex scans; `RJ_SPLIT_RE` untouched), `scripts/diagnostics/gf_run_log_digest.py` (`LOG` path template → glob the same way; add a `[FANOUT]` summary: per op, count / mean `head_s` / max `max_rank_s` / mean `wait_s`)
- Modify: `src/lisatools/globalfit/recipe.py` (`note_iteration` hook site) or `run.py` (the head's per-iteration hook): when `GF_FANOUT_DIGEST=1`, log `[FANOUT_DIGEST] it=<n> log_like=<sha1 of state.log_like bytes> coords=<sha1 of concatenated branch coords> inds=<sha1>` once per iteration on the head (after the iteration's moves; the eryn stopping-fn call is the existing per-iteration hook)
- Create: `scripts/diagnostics/gf_state_digest.py`: `python gf_state_digest.py <store.h5>` prints one line per array of `backend.get_last_sample()` (coords per branch, inds, log_like, log_prior, betas, every sub-state array) as `name shape sha1`, sorted, so two runs diff line by line
- Test: `tests/test_diagnostics_multirank.py` (create): a temp dir with `globalfit_run.log` + `globalfit_run.rank1.log` each holding one `[GB_ACCEPT rj-split gb] births ...` line → the monitor's discovery returns both files and the digest's rj-split count sums them; a synthetic `[FANOUT] op=... head_s=... max_rank_s=... wait_s=...` line parsed by the new summary; `gf_state_digest` on a tiny HDF store built by `tests/test_gf_substate_roundtrip.make_state` + `GFHDFBackend` (read tests/test_multirank_blank_smoke.py for how a store is written) prints deterministic sha1s.

Keep every diagnostics edit surgical (the main checkout has uncommitted edits in these files from another session).

Commit:
```bash
git add scripts/diagnostics/gf_monitor_gen.py scripts/diagnostics/gf_run_log_digest.py scripts/diagnostics/gf_state_digest.py src/lisatools/globalfit/recipe.py src/lisatools/globalfit/run.py tests/test_diagnostics_multirank.py
git commit -m "feat(diagnostics): read every rank's log; [FANOUT] summary; GF_FANOUT_DIGEST line and gf_state_digest.py for the cluster gates

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```
(stage only the files you actually touched)

---

### Task 5: Docs

**Files:**
- Rewrite: `docs/global-fit-launch.md` — roles (HEAD computes block 0 + sequences; COMPUTE one per GPU by default; SAVER = highest rank at np >= 3; SPARE only under `GF_LEGACY_RANK_LAYOUT=1`), the np table (np=1: one rank, whole local pool (AUTO); np=2 on 1 GPU: head + saver with a warning; np=2 on 2 GPUs: 2 compute ranks; np=3 on 2 GPUs: head + compute + saver; np=5 on 2×2: 4 compute + saver via `srun --distribution=cyclic`), the knobs (`GPUS`, `GPUS_PER_RANK` (AUTO), `RANKS_PER_GPU`, `NWALKERS % n_compute`, `GF_LEGACY_RANK_LAYOUT`, `GF_LAYOUT_DRY_RUN`, `GF_FANOUT_DIGEST`), the "Semantics that change only when n_compute > 1" list copied from the spec + the Plan 3/4 additions (per-block PSD plateau; pooled swap population; GB friend/infomat/F-stat tables rank-local; `[GB_TEMPER_CHECK]` per rank; no submission dump under several ranks), and the per-rank log files (`globalfit_run.rank<k>.log`, stdout prefix `[r<k>/c<k>]`).
- Modify: `src/lisatools/globalfit/run.py` module docstring (roles paragraph).
- Modify: `docs/multigpu-cluster-validation.md` — addendum: Gates 1-4 as written still exercise the in-process router because a lone rank owns the whole pool (AUTO); under several compute ranks the router runs single-shard per rank; pointer to the new runbook.
- Create: `docs/multirank-cluster-gates.md` — the WP7 runbook (Verification items 4-7 of the spec made concrete): dry runs (`GF_LAYOUT_DRY_RUN=1` for `-n 3` on 1 node and across 2 nodes), the three transport-parity layouts with exact env/CLI (`GPUS=0 RANKS_PER_GPU=2 mpiexec -n 3 …`; `GPUS=0,1 mpiexec -n 3 …`; `srun -N 2 --ntasks=3 --distribution=cyclic …`), the env bundle (`DATA_MODE=synthetic NWALKERS=8 NUM_ITERATIONS=N MIDIT_CHECKPOINT=0 MAKE_DIAGNOSTIC_PLOTS=0 GF_FANOUT_DIGEST=1 GF_LEGACY_RANK_LAYOUT=0`, same `random_seed`), what to diff (`[FANOUT_DIGEST]` lines; `gf_state_digest.py` outputs), the shared-GPU parity run, the statistical gate vs `GF_LEGACY_RANK_LAYOUT=1` on the 3mo recipe (snapshot tooling), the `[FANOUT]` load-balance readout, the `slice_state` payload-size measurement (`main_branches=` lever), and the two decisions WP7 closes (flip the campaign default to `GF_LEGACY_RANK_LAYOUT=0`; delete `_propose_legacy`).
- Modify: `docs/codebase-map.md` (one pointer line to the runbook and the communication package); a closing note in `docs/multirank-cluster-gates.md` stating that the old `LISAanalysistools/multinode_gpu_handoff.md` (untracked, main checkout) is SUPERSEDED by the design spec.

Commit: `docs: multi-rank launch guide, cluster-gates runbook, validation addendum, run.py roles docstring`.

---

### Task 6: Whole-plan verification

Run (batched): `tests.test_rank_layout tests.test_stock_globalfit tests.test_driver_scripts_layout tests.test_submit_scripts_layout tests.test_diagnostics_multirank tests.test_run_multirank_helpers` → OK; `RUN_GF_SMOKE=1 tests.test_multirank_blank_smoke tests.test_multirank_noise_smoke tests.test_globalfit_sample` → OK; `bash -n` both submit scripts; line gate over `git diff <plan base>..HEAD -- src tests scripts/diagnostics`; `git log --oneline <plan base>..HEAD`.

---

## Self-review notes

- Spec coverage WP6: settings knobs (T1, with the AUTO rule as a controller ruling to keep `-n 1` multi-GPU bit-identical — the spec's "1 for the current pipeline" is preserved for several compute ranks); drivers (`prepare_rank` already; dry run T2); `StockGlobalFit.run` already prepares; submit scripts (T3: knobs, `N_COMPUTE`, `--ntasks`, NGPUS=4 = gpu-80-spot 2×2 cyclic srun, NWALKERS check → 12, `[SUBMIT]` lines; PLUS the legacy default for the campaign = risk-section ruling); diagnostics negative filter → replaced by log discovery per the site map's finding (T4) + `[FANOUT]` summary; docs (T5); Verification tools `GF_FANOUT_DIGEST` and `gf_state_digest.py` (T4) and the WP7 runbook (T5).
- Decisions surfaced to the user in the final report: (1) AUTO pool rule; (2) campaign scripts default to the legacy layout until WP7; (3) submission dump stays skipped under several ranks (warning) — routing it through the fan-out is a follow-up; (4) `multinode_gpu_handoff.md` is untracked in the main checkout and is not edited here.
