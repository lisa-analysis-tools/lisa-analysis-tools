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
  runbook. The campaign submit script currently keeps the legacy layout as
  its own default at `NGPUS=2` (`docs/global-fit-launch.md`, "Campaign
  submit scripts") specifically so today's production launches are
  unaffected until this runbook's gates pass — so a gate run must opt in to
  the new layout, not rely on a script default that is deliberately still
  legacy.
- gpu-80-spot throughout: `--nodes=1 --gres=gpu:2` for the 1-node runs,
  `--nodes=2 --gres=gpu:2` for the 2-node runs (design spec Verification
  section). Adjust partition/gres flags to whatever the cluster actually
  grants at the time — they are not load-bearing for the gates themselves,
  only the resulting rank/node/device placement is.

## Step 0 — layout dry runs

Confirm the layout resolves the way you expect, on 1 node and across 2
nodes, before spending GPU time on anything else:

```sh
# 1 node, 3 ranks (head + 1 compute + saver on a 2-GPU node)
GF_LAYOUT_DRY_RUN=1 GPUS=0,1 mpiexec -n 3 \
  python scripts/run_global.py --stock <name>

# 2 nodes, 3 ranks (head + 1 compute on the other node + saver)
GF_LAYOUT_DRY_RUN=1 GPUS=0 srun -N 2 --ntasks=3 --distribution=cyclic \
  python scripts/run_global.py --stock <name>
```

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
identical** to each other given the same seed. Exact CLI (adapt only the
srun partition/gres flags to the cluster; the flags below are the shape,
not a guarantee they match every allocation):

```sh
# (a) 2 compute ranks sharing 1 GPU -- cheapest, run this first
GPUS=0 RANKS_PER_GPU=2 mpiexec -n 3 \
  python scripts/run_global.py --stock <name>

# (b) 2 compute ranks on 2 GPUs of 1 node
GPUS=0,1 mpiexec -n 3 \
  python scripts/run_global.py --stock <name>

# (c) 2 compute ranks across 2 nodes (adapt the srun flags to the cluster)
GPUS=0 srun -N 2 --ntasks=3 --distribution=cyclic \
  python scripts/run_global.py --stock <name>
```

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
  + inds) the head emits when `GF_FANOUT_DIGEST=1`. Only the three
  `n_compute=2` layouts above are expected to print the identical hash at
  every iteration. The line is emitted from the recipe's post-iteration hook
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

**`[FANOUT]` load-balance line.** Every non-single fan-out op logs one DEBUG
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
cross-device work to route within a rank).

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

## Decisions this runbook closes

Once Steps 0-4 are green:

1. **Flip the campaign default.** `submit_gf_6mo_v8.sh` (and siblings) stop
   defaulting `GF_LEGACY_RANK_LAYOUT` to `1` at `NGPUS=2` — the walker-block
   layout becomes the default at every `NGPUS`, matching what `NGPUS=4`
   already forces.
2. **Delete the legacy parity path.** `GBSpecialBase`'s `_propose_legacy`
   (the ~750-line byte-identical copy of the pre-port `propose` body kept
   only as the single-rank parity reference) and the
   `GB_PROPOSE_ORCHESTRATE` env var that opts a single-rank run into the new
   orchestrator early are both deletable once the cluster gates above are
   green (Plan 4 ledger ruling: `"_propose_legacy is deleted only after the
   WP7 cluster gates"`; dispatch at `gbspecialstretch.py:17888-17911`).

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
