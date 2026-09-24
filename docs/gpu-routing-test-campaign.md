# GPU-routing test campaign — costed

Gates for the unified rank layout (`n_compute = n_blocks × R`, i.e. GPUs >
walkers at any walker count). Ordered **cheapest first**, and every gate says
what it costs and what would make it fail. Nothing below needs a GPU until
G3, and nothing needs a *production* allocation until G6.

**Standing rule for every gate: stop at the first red.** Each one exists to
make the next one's failure interpretable; running G6 on a broken G2 wastes
an allocation and tells you nothing about the layout.

> **★ The routing is OPT-IN: `GF_GPU_ROUTING=1`.** That is why this work can
> live on `dev` with G3–G6 still outstanding — with the knob unset the
> library applies the pre-2026-09-23 rule exactly, and the two rules agree on
> every shape the legacy one accepted (G0's `test_gpu_routing_optin` sweeps
> 1–32 × 1–32 and asserts it). **Every gate from G3 on must export it**;
> `submit_gf_6mo_v8_gpurouting.sh` does it for you, the interactive runbook
> exports it in its env block, and a bare `mpiexec` does not. A layout line
> reading `gpu_routing=OFF(legacy)` means the gate measured the old rule.

---

## Cost summary

| Gate | What | Hardware | Cost | Blocking? |
|---|---|---|---|---|
| **G0** | Unit suites | laptop, CPU, 1 core | **~1 min** | yes |
| **G1** | Real-MPI layout preflight | laptop, CPU, 5 procs | **~10 s** | yes |
| **G2** | FakeWorld smokes at both axes | laptop, CPU | **~3 min** | yes |
| **G3** | Cluster dry runs at 4/8/16/32 | login node, no alloc | **~1 min** | yes |
| **G4** | Three-factorization parity | 1 node × 2 GPUs, interactive | **~30 min** | yes |
| **G5** | **T5 scaling readout** (R = 1/2/4) | 2 nodes × 2 GPUs | **~1 h** | **the decision gate** |
| **G6** | Short production shape | target allocation | **2-4 h** | before a long run |

Total to a go/no-go on the replica axis: **~2 GPU-hours**, essentially all of
it in G5. G0-G3 are free.

---

## G0 — unit suites (laptop, ~1 min)

```bash
.wtenv/wt_run.sh <src> .wtenv/g0.log python -m unittest \
  tests.test_gpu_routing_optin \
  tests.test_rank_layout tests.test_fanout_fakecomm tests.test_rowfanout \
  tests.test_walkerfanout_mixin tests.test_gb_replica_bands \
  tests.test_gb_replica_ledger tests.test_gb_replica_merge \
  tests.test_addremove_rows tests.test_psd_rows \
  tests.test_fstat_multiwalker_plan tests.test_submit_gpurouting_layout
```

`test_gpu_routing_optin` is listed first because it is the one that says
whether the rest of this campaign is even needed on `dev`: it pins that
`GF_GPU_ROUTING` unset reproduces the legacy rule on every shape the legacy
rule accepted (a 1–32 × 1–32 sweep), which is what makes this work safe to
carry on `dev` while the gates below are still outstanding.

**Pass:** all green. **Known pre-existing reds elsewhere on `dev`** (NOT
caused by this work, do not chase): the four
`test_fstat_gridfit.FitClockTest`/`FitDecisionTest` cases (they assert the
pre-2026-09-18 propose-census clock semantics), and
`test_submit_scripts_layout.ThreeMonthTwinTest` (the 3mo twin has drifted
from the 6mo script).

## G1 — real-MPI layout preflight (laptop, ~10 s) ✅ PASSED 2026-09-23

The first exercise of `build_layout` outside `FakeWorld`: real mpi4py, real
processes, real `allgather`. Catches anything that only works because
FakeWorld's `Split`/`allgather` are in-process.

```bash
export GF_LAYOUT_DRY_RUN=1 GPUS=""        # CPU: no GPU-pool capacity check
for NW in 4 2 1 6; do
  NWALKERS=$NW mpiexec --oversubscribe -n 5 python <preflight>
done
```

**Result:** 4→`n_blocks=4 R=1`, 2→`n_blocks=2 R=2`, 1→`n_blocks=1 R=4`,
6→`n_blocks=2 block=3 R=2`; **every rank agreed on `digest()` in all four**.
The GPU-pool capacity check also fired correctly when given 4 ranks against a
2-device pool.

**Pass:** every rank prints the same table; `n_blocks × R == n_compute` and
`n_blocks × block == nwalkers`. **Fail looks like:** ranks disagreeing on the
digest (the `allgather` in `build_layout` is not resolving identically), or a
rank reporting `size=1` (the launcher made singleton worlds — an MPI problem,
not a layout one).

## G2 — FakeWorld smokes at both axes (laptop, ~3 min)

**Both smokes are ENV-GATED and SKIP by default** — they do not run in an
ordinary `python -m unittest` sweep, and a green sweep therefore says
nothing about them. Arm them explicitly:

```bash
RUN_GF_SMOKE=1     python -m unittest tests.test_multirank_noise_smoke   # ~3.6 GB
RUN_GF_GB_SMOKE=1  python -m unittest tests.test_multirank_gb_smoke      # ~4.7 GB
```

Exercises a real propose through the fan-out with both axes live. **Pass:**
green, and no `[GB_REPLICA] log_like_final disagrees` line.

**KNOWN PRE-EXISTING RED (verified against clean `dev` @ `6d641643`,
2026-09-23):** `test_multirank_noise_smoke::test_one_walker_two_replicas`
fails with

```
galfor_pe/galfor: eigen sigma is 5.148e+00 prior widths along its own axis
-- draws cannot land inside the prior
```

byte-identical on `dev` and on this branch (`5.148136608105845` to the last
digit), so it is NOT a routing regression. It is the recorded galfor
conditioning problem -- 5 parameters over ~45 decades in a LINEAR basis, whose
fix is the already-wired `GALFOR_LOG_SAMPLING=1` on a fresh store, ruled out
of scope for this work. **Attribute before chasing:** the cheap check is
`git worktree add --detach /tmp/x <dev-sha>`, copy `.wtenv`, run the one test,
`git worktree remove`.

> **Memory on this laptop (8 GB total).** The GB smoke's ~4.7 GB is at the
> edge with ~4 GB typically free, and pushing it into swap is how the box
> gets into trouble. Run the two SEPARATELY, never in one sweep, check
> `memory_pressure` first, and prefer running the GB smoke when nothing else
> is on the machine. On a larger box this caution does not apply.
> Never run `tests/test_gbspecial_flow.py` here at all (8-26 GB balloon).

## G3 — cluster dry runs (login node, no allocation, ~1 min)

```bash
GF_LAYOUT_DRY_RUN=1 NWALKERS=4  NGPUS=16 ./submit_gf_6mo_v8_gpurouting.sh
GF_LAYOUT_DRY_RUN=1 NWALKERS=24 NGPUS=8  ./submit_gf_6mo_v8_gpurouting.sh
GF_LAYOUT_DRY_RUN=1 NWALKERS=24 NGPUS=32 ./submit_gf_6mo_v8_gpurouting.sh
```

Free, and it catches every placement error before an allocation is spent —
including the recorded `I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0` trap, where
SLURM's block split puts more compute ranks on a node than its GPU pool
holds. **Pass:** the expected `n_blocks`/`R`, and each block's R ranks on ONE
node (that is what `--distribution=block:block` is for).

## G4 — three-factorization parity (1 node × 2 GPUs, ~30 min)

The gate that proves the factorization is a **routing** choice, not a physics
one. Same store, same seeds, `GF_FANOUT_DIGEST=1`, 5 moving iterations:

```
NWALKERS=4 RANKS_PER_BLOCK=1     # (4 blocks, R=1) — today's layout
NWALKERS=2 RANKS_PER_BLOCK=2     # (2 blocks, R=2) — the new axis
NWALKERS=1 RANKS_PER_BLOCK=4     # (1 block,  R=4) — one-walker replicas
```

Use a **fresh `FILE_STORE_DIR` per arm** — a reused store RESUMES, and a
walker-count mismatch aborts.

**Pass:** each arm self-consistent; no `[GB_REPLICA] log_like_final
disagrees` warning; `[FANOUT] ... block path with N rank(s) per block` present
for psd/galfor at R > 1. **These arms sample different walker counts, so
their chains will differ** — that is not a failure.

## G5 — ★ T5 scaling readout (2 nodes × 2 GPUs, ~1 h) — THE DECISION GATE

**This is the only measurement of the replica axis, and it has never been
run.** Everything claimed about what dispersal buys is a model until this
exists. One walker, R = 1 vs 2 vs 4, `NUM_ITERATIONS=5`.

> **⚠ PREREQUISITE: the source-weighted band split must be in.** Before
> 2026-09-23 `GBSpecialBase._replica_band_weights` returned `None`, so the
> 1232 bands were split by COUNT. The galaxy is not uniform in frequency
> (most detectable sources are below ~5 mHz), so an equal-count split hands
> one rank most of the work and dispersal measures **~50 % efficient (≈2× at
> R = 4) instead of ~100 % (≈4×)**. A T5 run on the old split would measure
> the bad number and could wrongly condemn the replica axis. **Discard any
> T5 figures taken before that fix.**

Record from `[GB_TIMING]`:

* **`temper_swap_grid`** (new) + `sorter_build` + `unit_open_close` — the
  R-invariant work, i.e. `f_fixed`, whose reciprocal is the hard ceiling on
  what dispersal can ever buy. The 1-year log bounds the *instrumented* part
  at ~4 % (ceiling ~25×), but that was a LOWER bound because the swap-grid
  build had no span. It does now — this gate turns the ceiling into a number.
* `buffer_build` — the refill-round cost; prices whether more GPUs buy back
  single-pass buffer residency or ~5 % of the move.
* `[FANOUT] wait_s` and `max_rank_s − head_s` per op — load imbalance. The
  block axis measures ~99 % efficient today; this is the replica axis's
  equivalent number.

**Go/no-go:** R = 4 should approach ~4× on the dispersed part with the
weighted split. If it lands near 2×, check the weights are not `None` before
concluding anything about the design.

## G6 — short production shape (target allocation, 2-4 h)

Only after G5 says the axis pays. Run the real recipe at the intended shape
(e.g. 4 walkers / 16 GPUs) for ~20 iterations on a **fresh store** and
compare s/it and per-move times against the 4-GPU baseline (435 s/it median,
GB RJ moves 71-83 % of it).

**Watch:** peak GPU memory per card (6mo baseline is 52-55 GiB of 93.6);
`Alive sources per temp` NOT scaling with R; birth counts per propose
comparable to the R = 1 arm.

---

## What is NOT covered, and should not be assumed

* **The multi-walker F-stat union.** `GB_FSTAT_FIT_WALKERS` defaults to **1**
  (user ruling): the refit stays on the min-lnL cold walker. The union's
  pieces are unit-tested but the per-slot sweep loop is **not wired**, so
  setting the knob > 1 does nothing yet.
* **PSD/galfor under replicas is block-path, not flat.** Correct, but the
  R − 1 non-lead ranks idle during it (~1 % of the iteration).
* **`gpus_per_rank > 1` together with several compute ranks** remains
  layout-supported but unvalidated (see `docs/multigpu-cluster-validation.md`).
* **Long-run statistics.** None of these gates says the sampler converges;
  they say the routing is faithful.
