# Multi-rank cluster gates (WP7 runbook)

## Purpose

The multi-rank walker-block port (one MPI rank per GPU, fan-out inside the
moves) has a laptop-side gate — FakeComm unit tests, CPU, one python
process, tiny synthetic fixtures — that is already green (see the design
spec's Verification items 1-3). This runbook is the **cluster half**:
concrete commands to run on real GPUs/nodes, and what to diff between them.
It makes concrete Verification items 4-7 of the design spec:
[`docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md`](superpowers/specs/2026-09-15-multirank-walker-blocks-design.md).
General rank/GPU knobs and the `np` matrix are in
[`docs/global-fit-launch.md`](global-fit-launch.md); the in-process
multi-GPU router's own gates are in
[`docs/multigpu-cluster-validation.md`](multigpu-cluster-validation.md).

## Prerequisites

- Work from branch `multirank-walker-blocks` (or wherever it has landed on
  `dev` after the finishing step below).
- `pip install -e '.[dev,testing]'` in the environment that will run the
  cluster job (matches the rest of the sprint's build recipe).
- Export `GF_LEGACY_RANK_LAYOUT=0` explicitly for every command in this
  runbook (the layout-level default is already `0`; since the 2026-09-16
  ruling the campaign submit scripts default to `0` at every `NGPUS` as
  well, with `GF_LEGACY_RANK_LAYOUT=1` as the rollback knob).
- gpu-80-spot throughout: `--nodes=1 --gres=gpu:2` for the 1-node runs,
  `--nodes=2 --gres=gpu:2` for the 2-node runs (design spec Verification
  section). Adjust partition/gres flags to whatever the cluster actually
  grants at the time — they are not load-bearing for the gates themselves,
  only the resulting rank/node/device placement is.

## MPI launcher on this cluster (learned at Step 0, 2026-09-16)

The cluster's MPI is **Intel MPI**; SLURM offers `pmix_v4` (`srun --mpi=list`).
What was tried, in order, inside a 2-node `salloc`:

| launch | result |
|---|---|
| `srun -N 2 --ntasks=3 --distribution=cyclic python ...` (no PMI) | `MPI startup(): PMI server not found` — every rank a **size-1 world**, each printing `r0 head ... walkers=[0,8)`: the placement was right (2 processes on node A, 1 on B) but nothing was shared |
| `I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so`, `--mpi=pmi2` | the library does not exist here; Intel MPI fell back to its own client and garbled the pmi2 wire protocol (`mpi/pmi2: request not begin with 'cmd='`) |
| `I_MPI_PMI=pmix I_MPI_PMI_LIBRARY=/opt/pmix/4.2.9/lib/libpmix.so`, `--mpi=pmix` | bootstraps, then Intel MPI's OFI layer aborts: `MPIDU_bc_table_create: Missing hostname or invalid host/port description in business card` |
| `I_MPI_HYDRA_BOOTSTRAP=slurm mpiexec -n 3 -ppn 1 python ...` | ranks launched on both nodes; UCX then finds **no cross-node transport** (`no active messages transport ... self/memory, sysv/memory, posix/memory, cma/memory`) |
| the same plus `I_MPI_FABRICS=shm:ofi FI_PROVIDER=tcp` | **works** — `size=3 n_compute=2`, `r1 compute` on the second node |

So every multi-node command in this runbook uses Intel MPI's own launcher
(hydra) bootstrapped from the SLURM allocation, with the fabric pinned to
libfabric's tcp provider:

```sh
export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi FI_PROVIDER=tcp
mpiexec -n <ntasks> -ppn 1 python ...        # -ppn 1 = round-robin over the hosts
```

`-ppn 1` places consecutive ranks A, B, A, B, ... — the "cyclic" placement the
walker-block layout wants (head + saver on node A, the compute ranks spread
over both nodes). For a 1-node layout inside the same 2-node allocation use
`-ppn <ntasks>` so all ranks stay on the first host. TCP is a correctness
choice: whether a faster provider (`fi_info -l`) matters is Step 4's
`[FANOUT]` measurement. The campaign submit scripts' multi-node branch
carries the same three exports and launch line.

Interactive use: `salloc --partition=gpu-80-spot --nodes=2 --gres=gpu:2
--ntasks-per-node=2 ...` lands you in a shell on the first allocated node;
`mpiexec` (hydra) started there reads the allocation, so no `srun` wrapper is
needed. Do **not** add `--gpu-bind`/`--gpus-per-task`: they renumber
`CUDA_VISIBLE_DEVICES` per task and fight the layout's own pinning from the
per-node `GPUS` pool. Your shell's exports propagate to the ranks.

## Step 0 — layout dry runs

Confirm the layout resolves the way you expect, on 1 node and across 2
nodes, before spending GPU time on anything else:

```sh
export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi FI_PROVIDER=tcp   # see above

# 1 node, 3 ranks (head + 1 compute + saver on a 2-GPU node)
GF_LAYOUT_DRY_RUN=1 GPUS=0,1 mpiexec -n 3 -ppn 3 \
  python scripts/run_global.py --stock <name>

# 2 nodes, 3 ranks (head + 1 compute on the other node + saver)
GF_LAYOUT_DRY_RUN=1 GPUS=0 mpiexec -n 3 -ppn 1 \
  python scripts/run_global.py --stock <name>
```

(Both passed on 2026-09-16 with `--stock gb_no_fg`, `NWALKERS=8`.) A run
that prints `size=1` and `walkers=[0,8)` on every rank is the singleton
symptom from the launcher table above, not a layout bug.

`GPUS=0` is load-bearing on the 2-node command, not decoration: `GPUS` is
the **per-node** pool, each node here hosts exactly ONE compute rank, and
AUTO `gpus_per_rank` gives a lone-per-node compute rank the node's WHOLE
pool. Left unpinned on a 2-GPU node that rank would own both devices and
run the in-process cross-device router (the configuration
`docs/multigpu-cluster-validation.md` flags as unvalidated under several
ranks) — a different configuration from the single-device layouts (a)/(b)
below, so the parity gate would be comparing two things that were never
supposed to match. A one-device pool keeps AUTO at 1 everywhere.

`GF_LAYOUT_DRY_RUN=1` prints `layout.describe()` from every rank and exits
before `fit.build()` allocates anything (`communication/ranks.py::
layout_dry_run`, wired into the drivers by Plan 5 Task 2). Expected shape of
the printed table (schematic — `communication/ranks.py::
WalkerBlockLayout.describe()`, illustrated here for the `np=3` layout above,
not a captured run):

```
walker-block layout: size=3 n_compute=2 nwalkers=<W> block=<W/2> gpus_per_rank=AUTO->1 ranks_per_gpu=1
  r0   head    node=<nodeA> local=0 devices=[0] slot=0 walkers=[0,<W/2>)
  r1   compute node=<nodeA> local=1 devices=[1] slot=0 walkers=[<W/2>,<W>)
  r2   saver   node=<nodeA> local=2 devices=[0] slot=0 walkers=[0,0)
```

(The schematic is the **1-node** `GPUS=0,1` command above.
`gpus_per_rank=AUTO->1` there because two compute ranks share the node, so
AUTO resolves each to one device — `WalkerBlockLayout.describe()`'s `gpk`
formatting for the `gpus_per_rank_auto` case. The 2-node command reaches the
same `AUTO->1` by the other route: one compute rank per node, but a pool of
one device because of the `GPUS=0` above. `local=` is the rank's index
within its node (`node_comm.Get_rank()`), so on the 1-node launch the saver
prints `local=2`, and the saver's `devices=` is `pool[local % len(pool)]` —
it builds on a device like everyone else, then releases it.)

Both dry runs should print an identical `size`/`n_compute`/`block` header
and the same per-rank role assignment shape; `node=`/`local=` differ between
the 1-node and 2-node launches, and so does `devices=` — the 1-node launch's
rank 1 gets `devices=[1]` (two compute ranks split a 2-device node pool),
while the 2-node launch's rank 1 gets `devices=[0]` (the `GPUS=0` pin makes
each node's pool deliberately one device, so every rank on it draws from a
pool of one). A layout error (bad divisibility,
over-subscribed pool) raises inside `build_layout`/`prepare_rank` before any
`describe()` call, so the dry run's job is mainly to *stop before `build()`*
on the happy path — a bad layout fails exactly the same way with or without
`GF_LAYOUT_DRY_RUN=1`.

Also check on the first real (non-dry) launch:

- **F-stat epoch completeness.** The head's GB `setup()` writes
  `fstat_grid_peaks_stacked.npz` / `fstat_centers.npz` / `DONE.json` into
  `<fit_dir>/shared/epoch_NNNN/` and the ranks open those paths on the very
  next message, so the epoch directory has to be visible and complete on
  every node. The head `fsync`s them (`[FSTAT_EPOCH ...] head flushed epoch
  N for the ranks: ...` on the head's log) and a rank that still sees an
  incomplete directory raises `F-stat epoch N incomplete at <path>` instead
  of silently falling back to the prior for births. If that fires on a
  2-node launch, the fit directory is not on a shared filesystem (or its
  metadata lag exceeds the message latency) — fix the storage, do not
  suppress the check.

## Step 1 — three transport-parity layouts

Three launches, all resolving to `n_compute=2`, that should be **bit-
identical** to each other given the same seed. Exact CLI, run from inside
one 2-node `salloc` (the launcher recipe is the "MPI launcher on this
cluster" section above):

```sh
export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi FI_PROVIDER=tcp   # see "MPI launcher"

# (a) 2 compute ranks sharing 1 GPU -- cheapest, run this first
GPUS=0 RANKS_PER_GPU=2 mpiexec -n 3 -ppn 3 \
  python scripts/run_global.py --stock <name>

# (b) 2 compute ranks on 2 GPUs of 1 node
GPUS=0,1 mpiexec -n 3 -ppn 3 \
  python scripts/run_global.py --stock <name>

# (c) 2 compute ranks across 2 nodes (round-robin: head + saver on A, compute on B)
GPUS=0 mpiexec -n 3 -ppn 1 \
  python scripts/run_global.py --stock <name>
```

(`-ppn 3` on (a)/(b) keeps all three ranks on the first host of a 2-node
allocation; on a 1-node allocation plain `mpiexec -n 3` is the same thing.)

Each launch is `-n 3` (2 compute ranks + 1 saver, `resolve_roles(3)` gives
`compute=(0, 1)`, `saver=2`). `GPUS=0` on (c) for the same reason as the
2-node dry run above: one compute rank per node means AUTO `gpus_per_rank`
would hand it the node's whole pool, so without the pin (c) would be a
multi-device run being diffed against two single-device ones.

Common env bundle for all three (design spec Verification item 5):

```sh
export DATA_MODE=synthetic
export NWALKERS=8
export NUM_ITERATIONS=<N>          # pick a small N for the first pass
export MIDIT_CHECKPOINT=0
export MAKE_DIAGNOSTIC_PLOTS=0
export GF_FANOUT_DIGEST=1
export GF_LEGACY_RANK_LAYOUT=0
```

Use the *same* `random_seed` across all three runs. The `erebor` stock
variants default `EreborGeneralSettings.random_seed` to a fixed int
(`103209`, no env knob) rather than drawing fresh entropy, so this is
already automatic unless a driver overrides `fit.general.random_seed`
explicitly — if it does, pin it to the same value for all three launches.

- **What the seed now covers.** With `general.random_seed` set, the GB
  build-time prior objects *and* the RJ birth container are seed-determined
  **per rank and per F-stat epoch**: the rank's build seed
  (`communication.ranks.rank_build_seed`, domain-tagged so it is never the
  integer the global `np.random`/`cupy.random` streams run on) reaches the
  move as `fstat_fit_kwargs["build_seed"]`, and `_birth_seed(k)` derives one
  birth stream per epoch for `build_gb_birth_distribution(seed=...)`. A
  resume that re-installs the same epoch on the same rank therefore rebuilds
  the identical birth container, and two ranks never share a birth stream.
  Those births are reproducible for the **first fit built in a process** —
  which is every production rank, one `fit.build()` per process (round 2
  measured run-1 legacy == run-2 legacy across processes exactly) — while a
  *second* fit built in the same process drifts, because per-process caches
  change how many draws the module-level `np.random` stream has served by
  prior-draw time and the four columns eryn's `UniformDistribution` fills
  (`phi0`, `cos_iota`, `psi`, `fdot_astro_ratio`) move with it; that is why
  the laptop parity harness (`tests/test_multirank_gb_smoke.py`) runs one fit
  per process, and not something a cluster rank ever hits.
  With `random_seed` unset nothing is seeded — every birth generator stays on
  OS entropy, exactly as before — and a bit-identity diff across layouts is
  then meaningless for any run whose model has alive GB leaves.

**What to diff:**
- `[FANOUT_DIGEST]` lines — a per-iteration state hash (`log_like` + coords
  + inds) the head emits when `GF_FANOUT_DIGEST=1`. Across the three
  `n_compute=2` layouts above the `coords` and `inds` hashes must be
  identical at every iteration; the `log_like` hash is NOT expected to match
  on GPU (see "Step 1 as actually run" below — few-ulp launch-to-launch
  noise), compare it numerically instead. The line is emitted from the recipe's post-iteration hook
  whether or not a fan-out exists, so a single-rank (`-n 1`) baseline run
  prints it too, but single mode reseeds nothing (`run.py::
  _resolve_seed_base`/`_seed_rank_streams` return `None` when
  `layout.is_single()`; eryn seeds the process-global numpy stream from
  `random_seed`, the device stream stays unseeded), whereas every rank of an
  `n_compute=2` layout derives its stream via `derive_rank_seed`. The `-n 1`
  line is useful for observability and as an `it=0` cross-check (identical
  initial state) — not for a per-iteration diff against the three layouts,
  and this doc makes no claim about `-n 1` run-to-run reproducibility either
  way.
  **Dead slots (fix round 4, 2026-09-16):** the `coords` hash — here and in
  `gf_state_digest.py` — now covers each rank's rejected-birth FILL in dead
  leaf slots, because `gb_finish` ships the block's whole branch and the head
  writes it. Nothing reads those values (`inds` is `False` there), and they
  are identical across the three prescribed layouts because
  `derive_rank_seed` depends only on `(base_seed, n_compute, fanout_rank)`,
  which those three share. So a hash mismatch confined to dead slots is not a
  port bug: compare `coords[inds]` first when a hash differs, and only treat
  it as a defect if the ALIVE leaves disagree.
- `python scripts/diagnostics/gf_state_digest.py <store.h5>` — a digest
  over `backend.get_last_sample()` covering the full saved `GFState`
  (coords, inds, log_like, betas, and every sub-state array; design spec
  Verification item 5). Run it against each layout's final store and diff
  the outputs; they must match exactly. **Sub-state caveat:** an HDF5 file
  does not record which Python classes wrote its `sub_backend/` groups, so
  the script guesses from a small branch-name registry
  (`gb`/`vgb`/`mbh`/`emri`/`sobbh`, else the generic `ModuleSubBackend`). A
  branch outside that registry — or one whose data does not fit the guessed
  class — is reported on stderr and its `substate/` rows are absent from
  the digest (in the worst case the whole reconstruction falls back to a
  bare `GFHDFBackend` and only main-state arrays are printed). Read the
  stderr warnings before concluding "identical": a diff over a digest whose
  sub-state rows were skipped is a weaker gate than a full one.

A mismatch anywhere in this step means the transport or the merge logic
broke bit-identity for some fan-out op — bisect by op (`WalkerFanoutMixin`
addremove/PSD vs GB's three-command protocol) using the per-op
`[FANOUT_DIGEST]` cadence rather than only the end-of-run state digest.

### Step 1 as actually run and PASSED (2026-09-16, gb_no_fg)

Two practical findings that the commands above do not show:

1. **The stock synthetic injections are too faint to move the fit.** With
   `DATA_MODE=synthetic` and no explicit injection table, the F-stat epoch
   fit reports **no peaks**, births fall back to the prior, nothing is
   accepted in five iterations and every `[FANOUT_DIGEST]` line repeats the
   same three hashes — three layouts that all do nothing always agree, so
   the comparison is vacuous. Use one loud in-band injection (the laptop
   parity gate's) through a six-line driver launched exactly like
   `run_global.py`; there is no env knob for the injection table:

   ```python
   # gate_gb.py -- run from the repo root
   from lisatools.globalfit.stock import erebor

   fit = erebor.get_stock("gb_no_fg")
   # amplitude, f0 [Hz], fdot, fddot, phi0, iota, psi, lambda, beta (GBGPU order)
   fit.general.gb_injection_params = [[1e-21, 7.5e-3, 1e-16, 0.0, 1.2, 0.9, 1.0, 4.0, -0.6]]
   fit.run()
   ```

   Give every layout its OWN `FILE_STORE_DIR` (`./gate_a/`, `./gate_b/`,
   `./gate_c/`): a run that finds an existing store RESUMES from it, which
   silently breaks the same-initial-state premise. A live fixture shows
   `20 peaks` on the epoch line and hashes that CHANGE every iteration.
2. **`log_like` is not bit-identical on GPU, and cannot be; the decisions
   are.** Measured over five moving iterations (2026-09-16, 8 walkers,
   layouts (a)/(b)/(c) all `n_compute=2`): `coords` and `inds` hashes
   identical at every iteration and in the final stores across all three
   layouts; `log_like` differs by at most 1.3e-9 on values near 7e5
   (relative 1.8e-15, i.e. a few ulps), and `substate/gb/d_h` differs the
   same way. Walker 0 — the head's block, computed on GPU 0 in EVERY
   layout — differs between (a) and (b), so this is launch-to-launch
   reduction-order noise of the likelihood kernel on one device, not a
   device-to-device difference and not transport. **Pass criterion,
   restated:** `coords` and `inds` bit-identical across the layouts at every
   iteration and in `gf_state_digest.py`; `log_like` (and `d_h`) within
   1e-12 relative, compared numerically:

   ```sh
   python - <<'EOF'
   import numpy as np
   from lisatools.globalfit.hdfbackend import GFHDFBackend
   ll = {r: np.asarray(GFHDFBackend(f"gate_{r}/gb_no_fg_test_2_testing.h5")
                       .get_last_sample().log_like)[0] for r in "abc"}
   for r in "bc":
       d = ll[r] - ll["a"]
       print(r, d, "max rel:", np.max(np.abs(d) / np.abs(ll["a"])))
   EOF
   ```

   A difference above ~1e-9 relative, or any `coords`/`inds` mismatch, is
   the real failure this gate exists to catch.

## Step 2 — shared-GPU parity run

Layout (a) above — `RANKS_PER_GPU=2` on a single GPU — is deliberately the
cheapest of the three to iterate on: no multi-node scheduling, no second
device required, and it exercises the rank-sharing path (`device_slot`
distinguishing the two ranks on one device) that the other two layouts
never touch. Run it first and confirm its `[FANOUT_DIGEST]` / state-digest
output already matches layout (b)'s (2 GPUs, 1 node) before spending
allocation time on the 2-node launch (c) — if (a) and (b) already disagree,
the mismatch is unrelated to node placement and there is no reason to burn
a 2-node allocation chasing it. Once (a) ≡ (b), run (c) to confirm node
placement itself introduces no divergence.

## Step 3 — statistical gate vs the legacy layout

Compare the new default (`GF_LEGACY_RANK_LAYOUT=0`) against today's
in-process 2-GPU run (`GF_LEGACY_RANK_LAYOUT=1`) on the 3mo recipe — not
bit-identity (the two layouts draw different RNG streams per rank), but
statistical equivalence:

```sh
# today's layout (one compute rank drives both GPUs)
GF_LEGACY_RANK_LAYOUT=1 GPUS=0,1 mpiexec -n 3 \
  python scripts/fstat_proposal/run_combined_staged.py

# the new walker-block layout (2 compute ranks + saver)
GF_LEGACY_RANK_LAYOUT=0 GPUS=0,1 mpiexec -n 3 \
  python scripts/fstat_proposal/run_combined_staged.py
```

Compare, via the existing snapshot tooling (the `processing-gf-snapshots`
skill flow already used for every prior gf_prod snapshot readout):
- overall and per-move acceptance rates,
- cold-chain leaf counts per branch,
- per-band tempering acceptance (GB `band_temps`, `[GB_TEMPER_CHECK]`).

Expect these statistically indistinguishable between the two layouts over
enough iterations, not identical iteration-by-iteration — the semantics
list in `docs/global-fit-launch.md` ("Semantics that change only when
several compute ranks exist") describes exactly which cadences differ
(pooled-once-per-propose ladder adaptation, rank-local eigen/infomat/friend
tables, ...), so a like-for-like statistical read has to account for those,
not treat any difference as a regression.

## Step 4 — load balance and payload-size measurement

**Measured 2026-09-16 (gb_no_fg, 8 walkers, layouts (b) and (c)):** `wait_s`
0.000 on every op in both layouts, and the wall time between consecutive ops
equals `max_rank_s` + ~10 ms — so the whole command/reply round trip,
including the two-node TCP fabric and the enlarged `gb_finish` payload,
costs ~10 ms against 9-15 s `gb_run_proposal` calls; `max_rank_s` tracks
`head_s` to within 0.14 s (balanced blocks); `gb_run_tempering` 0.3-0.5 s,
`gb_finish` 50-90 ms (the ceiling on the round-4 reply-size concern: no
narrowing needed). A faster fabric than tcp buys nothing at this scale.
Before 2026-09-16 the line went to `global_fit.log` (the `GlobalFit` logger,
propagation off) at DEBUG; it is now INFO on the module logger, i.e. in
`globalfit_run.log` next to `[FANOUT_DIGEST]`, where `summarize_fanout`
looks for it.

**`[FANOUT]` load-balance line.** Every non-single fan-out op logs one INFO
line on the head (`communication/fanout.py::WalkerFanout.run`):

```
[FANOUT] op=%s move=%s head_s=%.3f max_rank_s=%.3f wait_s=%.3f
```

`head_s` = wall time of the head's own local block; `max_rank_s` = the
slowest worker's reported wall time; `wait_s` = the MPI rendezvous time the
head spent waiting on `isend`/`recv` before running its own block. A
healthy load-balanced layout has `max_rank_s` close to `head_s` and
`wait_s` small relative to both; a persistently large `wait_s` or a
`max_rank_s` well above `head_s` points at an unbalanced walker block or a
slow node. Cross-check against `route_dispatch` in `[GB_TIMING]`, which
should drop to ~0 per rank once fan-out is live (the router no longer has
cross-device work to route within a rank). Also weigh the `gb_finish` reply:
since fix round 4 it carries the block's whole GB branch,
`ntemps x B x nleaves_max x ndim` float64 plus an `ntemps x B x nleaves_max`
bool, per rank per propose (search-mode `nleaves_max` is what makes this
grow). If that shows in the head/route budget, the narrowing is flagged as a
`TODO(multi-rank, WP7)` at the reply construction in `gbspecialstretch.py`.

**`slice_state` payload size.** `communication/walkerslice.py::slice_state`
ships every main branch's `coords`/`inds` unconditionally in every payload,
regardless of which sub-states the receiving move actually reads
(`sub_states` only gates the heavier per-branch tempered sub-state slice).
Concretely: an addremove or PSD `propose` fan-out today also ships GB's full
`coords`/`inds` arrays even though GB never participates in that mixin's
protocol. There is no `main_branches=` filter yet to cut this down — it is
a flagged, unbuilt lever (Plan 3's final review: "measure at WP7"), not a
bug. Measure it here: instrument or profile a `slice_state` call's pickled
payload size for a representative multi-branch recipe (e.g. `all_sources`)
and compare it against the size of just the branches the fanned-out move
actually reads (`fanout_branches`). If the gap is a meaningful fraction of
the per-iteration `[FANOUT]` `wait_s`/`head_s` budget, that is the
justification to build `main_branches=`; if it is negligible at production
`Tobs`/branch counts, leave it as documented, deferred overhead.

## Step 5. Parallel F-stat epoch fit (WP7 addendum, 2026-09-16)

The first epoch that REFITS under the walker-block layout
(`GB_FSTAT_REFIT_EVERY=50`, so epoch 1 of the 6mo continuation) is the
cluster half of the parallel F-stat gate. Measured baseline: epoch 0 took
1 h 45 min, all of it stage B, on one of two GPUs.

**BOTH sweeps are split now (2026-09-18).** Epoch 1 of
`gf_prod_6mo_v8_4gpu` measured 5883 s total: stage A's comb 2804 s
(46.7 min) on ONE GPU with three idle, and stage B 3068 s (51.1 min) over
four ranks at 3.34x. Stage A was 48% of the epoch and the ceiling on the
whole fit, so it is now split by contiguous NODE range per sky level — the
comb is 222.9M evals at 6mo, 83% of them in the single nsky=512 level, and
nodes are independent (the per-node reduction reads only that node's own
sky block). Expect stage A ~12-14 min at four ranks and the epoch ~65 min,
with stage B the ceiling again.

`GB_OPS` now has all eight ops: the four per-propose session commands
(`gb_run_proposal`, `gb_run_tempering`, `gb_finish`, `gb_sync` — the last is
one-walker replica mode's) plus four non-session F-stat ops the head issues
from `setup()`, before any session exists, listed in the order the fit
issues them: `gb_fstat_ref_row` (replicate the global reference walker's
residual + inverse-PSD row to every rank), `gb_fstat_comb` (stage A: one
command per sky LEVEL, contiguous node ranges), `gb_fstat_stage_b` (stage B:
one command per Mc group, contiguous box ranges), and `gb_fstat_release`
(drop the replicated row, the cached sig-het scorer AND the scorer's
reference blocks on the GB comp — `gb_wdm_comp._fstat`, the ~GB half — on
every worker once the epoch's artifacts are written, then free the memory
pool).

What to collect from the head's `globalfit_run.log`:

1. `F-stat grid fit epoch 1 starting (walker_ref=<w> GLOBAL on rank <r> row
   <l>, ..., n_compute=<n>, ...)` — the reference is the GLOBAL argmax and
   names its owner. Cross-check `walker_ref` against `DONE.json`, which now
   also records `n_compute`; both feed the epoch's cache fingerprint as
   `|wref=<walker_ref>`, so a restart whose global argmax moved restarts the
   stage-B sweep and any in-flight comb scan cleanly instead of stitching
   two residuals' rows into one grid. **The COMPLETED comb cache is covered
   too now (2026-09-18).** It used to be reloaded on file existence alone —
   `fingerprint_extra` salts `ckpt_fingerprint`, i.e. the in-flight progress
   files, and a finished `*_comb.npz` is not one — so a refit whose argmax
   moved re-selected its peak BOXES from the old walker's scan while stage B
   scored inside them at the new reference. That was ruled an acceptable
   limitation on 2026-09-17 (proposal quality, not correctness: births are
   MH-corrected) because a rescan cost the full 47-minute serial comb; with
   stage A split it costs ~12 min, so `run_comb_scan` now stamps
   `fingerprint_extra` into the npz and a mismatch rescans, logging
   `comb cache ... was scored against a different reference`. Epoch 1 hit
   exactly this case (the argmax moved from walker 0 to walker 2).
   **Budget ONE extra sweep on the first relaunch after this branch lands.**
   Appending `|wref=<w>` changed the salt, so every checkpoint a pre-branch
   process left IN FLIGHT is invalid once — even when the reference walker
   has not moved. Per epoch: a **complete** `*_peaks_stacked.npz`
   short-circuits before anything else and pays nothing; a **complete comb
   with an in-flight stage B** keeps the comb (a pre-branch comb carries no
   stamp, so it is reused with a `carries no reference-walker stamp`
   WARNING rather than rescanned — deliberately, so this relaunch does not
   also pay for stage A) and re-sweeps stage B from zero, up to the full
   1 h 45 min measured at 6mo; an **in-flight comb** re-scans the comb too,
   now split over the ranks. It happens on the first relaunch and never
   again. If that is unacceptable for the relaunch window, land the branch
   at an iteration boundary where no epoch is mid-fit.
2. `F-stat reference row replicated to <n> rank(s) ... <X> MB residual +
   <Y> MB invC per rank` — at 6mo expect roughly 18 MB and 54 MB.
   A wildly different size means `Nf_active` is not what the design assumed;
   record the real number. A status word in the broadcast header makes an
   OWNER-side snapshot failure surface as a named `RuntimeError` instead of
   a hang; a NON-owner rank failing before that header broadcast still
   hangs the run — a documented residual, not a bug to chase here.
3. One `[FSTAT_COMB] level l/L over <n> rank(s): nodes [...] | wall min ...
   max ... (imbalance ...%) | assembled ...s total` line per SKY LEVEL,
   plus one `[FSTAT_COMB] l<li> r<r> nodes [a, b) (n) in <N>s` per rank per
   level. At 6mo expect six levels (nsky 16..512) with the nsky=512 level
   carrying 83% of the work; the node counts printed should sum to that
   level's node count and the ranges are contiguous. Per-node cost is
   uniform within a level, so an imbalance here is the sig-het
   reference-block build over the range's f0 span — the same residue stage B
   has, and the same remedy (weight by f0 span rather than node count).
   `[stageA] comb + peak selection: <k> peaks in <T> (split by node range
   over the compute ranks)` is the whole-stage line; `(serial, this
   process)` there means the fan-out was not installed.
4. One `[FSTAT_STAGEB] group g/G over <n> rank(s): boxes [...] | wall min
   ... max ... (imbalance ...%)` line per Mc group. Per-box cost is uniform
   within a group, so the residue is the sig-het reference-block build,
   which scales with f0 span. An imbalance above ~20% is the signal to
   weight the split by f0 span instead of box count (spec, Risks).
5. `[FANOUT] op=gb_fstat_comb ...` per level and
   `[FANOUT] op=gb_fstat_stage_b ... head_s=... max_rank_s=... wait_s=...`
   per group — the transport view of the same balance.
6. `F-stat grid fit epoch 1 done in <W>s` — compare `W` against epoch 0's
   `wall_seconds` in `epoch_0000/DONE.json` and against epoch 1's own 5883 s
   pre-split measurement. Expect roughly `1/n_compute` of BOTH the stage-A
   and stage-B shares, plus the unchanged centre table, the row-broadcast
   overhead (~72 MB total per rank, residual + invC) and the per-rank
   partial-`.npy` I/O through the shared epoch directory. At four ranks on
   the 6mo fit that is ~65 min against 98 min measured, with stage B back as
   the dominant half (~78%).
7. `F-stat reference row released on <n> of <n> worker rank(s); the head
   keeps its own until the centre table is built.` — issued right after the
   epoch's `.npz` + `DONE.json` are written and flushed. What each worker
   actually gives back: the replicated row (~72 MB), the scorer closure, the
   sig-het REFERENCE BLOCKS on its GB comp (`clear_fstat_references`; the
   expensive half, and the one a dropped closure does not own), then
   `free_all_blocks()`. **Watch a worker's `nvidia-smi` across the
   ~50-iteration window after epoch 1 and record the step down** — the
   block's "~GB" is an unmeasured estimate and that number belongs here. A
   release failure (`gb_fstat_release failed after epoch <k> was written`)
   is logged as a WARNING and is never fatal — the epoch is already on disk,
   and a worker that kept its row loses it at the next `gb_fstat_ref_row`
   regardless.

At `n_compute == 1` (single rank, no fan-out) BOTH sweeps run the serial
path exactly as they did before the parallel-fit change, golden-gated to a
byte-identical npz — `_fstat_comb_runner` and `_fstat_stage_b_runner` both
refuse to be built there, which is what keeps that gate closed rather than a
docstring. None of the above fires; there is nothing to collect. Byte-identical
is not cost-identical, though: the reference-row body still runs in process,
so expect **+72 MB resident** (the two rows are hosted as copies and put
back on the device) and one host round trip per fit, and on a multi-GPU
single-rank box the sweep now runs on `acs.gpus[0]` (the holder's device)
rather than on the shard that owns `walker_ref`. No disk round trip and no
collective, as designed.

If a one-walker-replica run (`NWALKERS=1` on several compute ranks) hits
this refit, the same four ops fire: every rank's block is `(0, 1)`, the
reference walker's owner is `compute_ranks[0]` — the HEAD under the default
`main_rank=0`; with a non-zero `main_rank` `resolve_roles` orders the
compute ranks by world rank, so `compute_ranks[0]` is a worker and the
`Bcast` root is that worker (correct either way: every replica holds the
same walker at local row 0) — and both sweeps split over the replicas
exactly like walker blocks — read the log the same way. Run this step's
epoch-1 verification in the SAME allocation as the one-walker campaign's
`docs/one-walker-testing-campaign.md` T5 scaling gate: both need a
multi-node GPU allocation, and there is no reason to request two.

Correctness, if a serial refit of the same state is affordable: run one with
`GF_LEGACY_RANK_LAYOUT=1` into a scratch fit root and diff the two
`fstat_grid_peaks_stacked.npz` files key by key (the `assert_npz_identical`
helper in `tests/test_fstat_parallel_fit.py` is the reference comparison).
They must be byte-identical.

Failure modes to watch for:

- `gb_fstat_comb`/`gb_fstat_stage_b` `arrived before gb_fstat_ref_row` — a
  rank missed the replication command; the run aborts through
  `RemoteWorkerError`.
- `comb partial lN rM changed under us` / `stage-B partial gN rM changed
  under us` — shared-filesystem fault between the rank's reply and the
  head's read.
- `comb level N: assembled k node(s) from n partials, expected m` /
  `stage-B group N: assembled (...) from k partials, expected (...)` — a
  rank/range map mismatch; check that every rank sees the same
  `n_compute`.
- `comb level N partial <path> could not be read on the head ... although
  every rank reported writing one` (and its stage-B twin) — the epoch cache
  directory is NOT on a filesystem shared by every compute rank.
- A rank dying mid-sweep aborts the run as for any op; the per-rank
  checkpoints make the retry cheap AS LONG AS the relaunch uses the same
  `n_compute`. Stage A's are
  `<epoch>/fstat_grid_parts/comb_nsky*_r*.progress.npz` (the serial path
  keeps the legacy `comb_nsky{lv}` name, and the head's
  `ckpt_clear(_parts, "comb_")` catches both by prefix); stage B's are
  `stageb_g*_r*.progress.npz`, or `stageb_r*.progress.npz` with a SINGLE Mc
  group (`FSTAT_MC_GROUPING=0` or a band set narrow enough to make one
  group, where the legacy `stageb` prefix is preserved deliberately; the
  production 6mo fit has six groups). A changed rank count restarts each
  level/group cleanly — correct, but it pays the whole sweep again.
- `gb_fstat_release failed after epoch <k> was written` — logged, not
  fatal; see item 7 above.

## Decisions this runbook closes

Once Steps 0-4 are green:

1. **Flip the campaign default.** DONE 2026-09-16 (user ruling, after Steps
   0-2 and 4 passed; Step 3 skipped — the live 6mo campaign run at 8 walkers
   is the statistical read): `submit_gf_6mo_v8.sh` and its null sibling
   default `GF_LEGACY_RANK_LAYOUT=0` at every `NGPUS`; `NWALKERS` defaults to
   8 so the run can continue from 2 to 4 GPUs.
2. **Delete the legacy parity path.** `GBSpecialBase`'s `_propose_legacy`
   (the ~750-line byte-identical copy of the pre-port `propose` body kept
   only as the single-rank parity reference) and the
   `GB_PROPOSE_ORCHESTRATE` env var that opts a single-rank run into the new
   orchestrator early are both deletable once the cluster gates above are
   green (Plan 4 ledger ruling: `"_propose_legacy is deleted only after the
   WP7 cluster gates"`; dispatch at `gbspecialstretch.py:17888-17911`).

## One-walker replica mode gate

The ordered cluster campaign for this mode (gates T0-T6 with pass criteria, paired controls, triage and the evidence to collect) is `docs/one-walker-testing-campaign.md`; the steps below are the per-step commands it references.

`nwalkers=1` spread over several compute ranks, where every rank holds a full
replica of the residual instead of a disjoint walker block: GB/VGB dispersal
is by static per-rank band range with a per-unit cold-chain delta ledger and
a `gb_sync` fan-out command that rebuilds every replica's residual from the
head-merged branch; addremove/PSD dispersal is by head-controlled likelihood
row scatter. Knobs: `GF_ONE_WALKER_REPLICAS`, `{BRANCH}_LIKELIHOOD_FANOUT`,
`{P}_INNER_MOVE_KIND`, `{P}_EIGEN_REFRESH`, `{P}_EIGEN_EPS_REL`. Laptop gate
passed: `RUN_GF_GB_SMOKE=1 python -m unittest tests.test_multirank_gb_smoke`
— the one-walker/two-replica arm ran 4 `gb_sync` rounds with agreeing
residual hashes across replicas. Four caveats to carry into the cluster
gate: (a) the residual hash covers `linear_data_arr` + `linear_psd_arr`,
which can be zero-length under the `psd_storage="none"` invC storage mode;
(b) `run_tempering`'s cold-chain open/close is narrowed to the replica's own
band range (`_tempering_open_close_mask`), so a replica opens roughly
`1/n_replicas` of the grid per unit — but the tempering *swap* work it still
schedules is its own rows only, so the 1-walker arm is not expected to beat
the multi-walker arm on wall time; (c) `run_tempering` has no in-stage
residual-divergence detector in replica mode — the cross-replica guard is the
`gb_sync` per-rank `log_like_final` agreement (`rtol=1e-10`, `atol=1e-8`),
and the residual hash is an exact-match bonus that agrees on CPU but is
expected to differ on GPU at the ~1e-12 `atomicAdd` level (logged at INFO,
not WARNING); (d) `cap_stats` (the leaf-cap gate heuristic) is computed on
the head before the `gb_sync` rebuild, from its own residual and its partly
stale `d_h`/`h_h`; heuristic only.

The steps below mirror Step 1's three-layout parity setup (same MPI-launcher
exports, same common env bundle) with `NWALKERS=1` in place of `NWALKERS=8`,
since a one-walker run has no walker block left to divide.

### Step A — layout dry run

```sh
export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi FI_PROVIDER=tcp   # see "MPI launcher"

export NWALKERS=1 DATA_MODE=synthetic NUM_ITERATIONS=4 MIDIT_CHECKPOINT=0 \
       MAKE_DIAGNOSTIC_PLOTS=0 GF_FANOUT_DIGEST=1 GF_LEGACY_RANK_LAYOUT=0

# 1 node: head + 1 compute + saver on a 2-GPU node
GF_LAYOUT_DRY_RUN=1 GPUS=0,1 mpiexec -n 3 -ppn 3 \
  python scripts/run_global.py --stock <name>
# expect: "walker-block layout: ... nwalkers=1 block=1 ... REPLICAS" and every
# compute rank printing "walkers=[0,1)"

# 2 nodes (round-robin: head + saver on node A, the second replica on node B,
# GPUS=0 pins each node's pool to one device so AUTO gpus_per_rank stays 1)
GF_LAYOUT_DRY_RUN=1 GPUS=0 mpiexec -n 3 -ppn 1 \
  python scripts/run_global.py --stock <name>
# expect: the same header; rank 1 on the other node with devices=[0]; a rank
# printing size=1 is the launcher singleton symptom (see "MPI launcher")
```
(`tests/test_rank_layout.py::ReplicaModeTest` pins both placements — the
3-rank round-robin and the NGPUS=4 five-rank cyclic one — on a fake world;
the transport across nodes is only proven by Step B (c).)

### Step B — replica parity (three layouts, same seeds)

Common env bundle (design spec Verification item 5, same as Step 1's, with
`NWALKERS=1`):

```sh
export DATA_MODE=synthetic
export NWALKERS=1
export NUM_ITERATIONS=<N>          # pick a small N for the first pass
export MIDIT_CHECKPOINT=0
export MAKE_DIAGNOSTIC_PLOTS=0
export GF_FANOUT_DIGEST=1
export GF_LEGACY_RANK_LAYOUT=0
```

```sh
# (a) 2 compute ranks sharing 1 GPU -- cheapest, run this first
GPUS=0 RANKS_PER_GPU=2 mpiexec -n 3 -ppn 3 \
  python scripts/run_global.py --stock <name>

# (b) 2 compute ranks on 2 GPUs of 1 node
GPUS=0,1 mpiexec -n 3 -ppn 3 \
  python scripts/run_global.py --stock <name>

# (c) 2 compute ranks across 2 nodes (round-robin: head + saver on A, compute on B)
GPUS=0 mpiexec -n 3 -ppn 1 \
  python scripts/run_global.py --stock <name>
```

What to diff:
- every `[FANOUT_DIGEST]` line ends with `replicas_agree=True`;
- NO `"[GB_REPLICA ...] log_like_final disagrees after sync"` warning — that
  is the divergence guard. `"[GB_REPLICA ...] residual hashes disagree after
  sync"` is an INFO line and is EXPECTED on GPU (~1e-12 `atomicAdd` spread);
  `"[GB_REPLICA] residual authoritative"` info lines are expected too (drift
  logged, not repaired);
- `log_like` / `coords` / `inds` digests agree across (a), (b), (c) to the
  same tolerance the multi-walker gate (Step 1) uses.

### Step C — statistical check vs one compute rank

```sh
# single rank, no replicas
GPUS=0 mpiexec -n 1 \
  python scripts/run_global.py --stock <name>
```

vs layout (b) above. NOT expected bit-identical (rank RNG streams differ for
GB dead-slot draws); compare acceptance rates, cold-chain leaf counts, and
per-band tempering through the `processing-gf-snapshots` flow, as Step 3
does for the multi-walker port.

### Step D — timing readout

- Per scoring call the parallelism is `min(n_compute, ntemps)`: set
  `PSD_NTEMPS` / MBH `ntemps` `>= n_compute`.
- Measure the PSD eigen refresh cost (`{P}_EIGEN_REFRESH` default `10`) from
  `[PSD_TIMING]`.
- GB: `run_tempering`'s cold-chain open/close is narrowed to the owned band
  range, but the swap-grid build, the census and the chunk loop are not, so
  do not expect a ~n_replicas wall-time win from tempering alone.

**Parser note.** The `[FANOUT_DIGEST]` line is append-only;
`scripts/diagnostics/gf_run_log_digest.py` summarizes `replicas_agree` and
`[GB_REPLICA]` via `summarize_replica_digest` and `summarize_gb_replica`
(Task 3).

## Closing note

`LISAanalysistools/multinode_gpu_handoff.md` — untracked, in the **main**
checkout (`/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools/`, not
this worktree) — is **SUPERSEDED** by the design spec at
`docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md`. That
handoff doc recommended band-range sharding off a single-process
multi-GPU audit; the user instead ruled for walker-block sharding with one
rank per GPU (this port), which this runbook and
`docs/global-fit-launch.md` now document as the current architecture. The
handoff file itself lives outside this branch/worktree and is not edited
here — treat it as historical background only.
