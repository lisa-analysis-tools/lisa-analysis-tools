# GPU-routing gates on an interactive 4-GPU allocation (2 nodes × 2 GPUs)

Copy-paste commands for gates **G3, G4 and G5** of
`docs/gpu-routing-test-campaign.md` on one interactive allocation. Total
~1.5 h of wall clock, of which G5 is the only part that answers the science
question.

The routing merged to `dev` on 2026-09-23, so `git pull` on the cluster is the
whole deployment — it is pure Python, with no native file touched and nothing
to recompile. It is OPT-IN (`GF_GPU_ROUTING=1`, exported in §0), so a checkout
carrying it behaves exactly as before until a run asks for it.

**Progress (2026-09-24, 2 nodes × 2 GPUs):** §0 sanity ✔ · G3 at `NWALKERS=4`
(`n_blocks=4 R=1`) ✔ · G3 at `NWALKERS=2` (`n_blocks=2 R=2 REPLICAS`) ✔ — the
first time GPUs > walkers has resolved on hardware. Round-robin placement
confirmed (A,B,A,B,A), each node's two compute ranks on distinct devices.

---

## 0. Allocation and environment (once)

```bash
salloc --nodes=2 --gres=gpu:2 --ntasks-per-node=3 --time=02:00:00 \
       --partition=gpu-80-ondemand
```

`--ntasks-per-node=3` so a 5-task launch fits (4 compute + 1 saver) with room
for the round-robin below.

> **EXPORT THIS BLOCK FIRST, IN THE SHELL YOU WILL RUN FROM.** `salloc` gives
> you a NEW shell, and these do not survive it — nor a second terminal, nor a
> reconnected session. Nothing below works without them, and the way it fails
> is not obvious: without the fabric pins Intel MPI dies inside
> `MPI_Init_thread` with `OFI get address vector map failed` and UCX
> complaining about `different host id`, which looks like a broken install
> rather than a missing export. If you are ever unsure whether you exported
> them, just paste the block again — it is idempotent.

```bash
# --- EDIT THESE TWO, then paste the rest verbatim -------------------------
export LAT=$HOME/lisa-analysis-tools        # your LAT checkout on the cluster
export G=$SCRATCH/gpurouting_gates          # somewhere with room for stores
# --------------------------------------------------------------------------

cd "$LAT"
git rev-parse --abbrev-ref HEAD             # expect: dev (merged 2026-09-23)
git log --oneline -1                        # should include the routing work
mkdir -p "$G"

# Intel MPI launcher pins. THESE THREE ARE THE PROVEN SET -- they are what
# the production submit scripts' multi-node branch exports verbatim, and the
# launcher table in docs/multirank-cluster-gates.md records how each
# alternative fails: bare `srun` (no PMI) gives SIZE-1 WORLDS, `--mpi=pmi2`
# finds no libpmi2, `--mpi=pmix` bootstraps but Intel MPI's OFI
# business-card exchange then aborts.
#
# WITHOUT THE LAST TWO you get this, and it is NOT a broken install:
#   UCX ERROR no active messages transport to <no debug data>:
#     self/memory ... sysv/memory - different host id ... cma/memory
#   Abort: MPIDI_OFI_mpi_init_hook: OFI get address vector map failed
# i.e. the only transports on offer are SHARED-MEMORY ones (self/sysv/posix/
# cma), correctly refused between two hosts -- UCX finds no cross-node
# transport on this cluster, so the fabric must be pinned to libfabric's tcp
# provider. TCP is a CORRECTNESS choice here; whether a faster provider
# (`fi_info -l`) is worth it is a separate measurement.
export I_MPI_HYDRA_BOOTSTRAP=slurm
export I_MPI_FABRICS=shm:ofi
export FI_PROVIDER=tcp

# ★ THE TRAP THAT COST A LAUNCH (2026-09-18, the first 4-GPU one-walker
# launch). SIZE-DEPENDENT, which is why Step 0 never saw it: under the SLURM
# bootstrap hydra honours the scheduler's PER-NODE TASK COUNTS over `-ppn`.
# At 3 tasks over 2 nodes block and cyclic happen to AGREE, so `-ppn 1` looks
# like it binds. At 5 tasks (4 compute + saver) SLURM's block split is 3/2,
# node A receives compute ranks 0,1,2 and `build_layout` refuses ("node ...-1:
# 3 compute ranks but the per-node GPU pool [0, 1] supports at most 2"). This
# restores the round robin A,B,A,B,A: node A takes compute 0, compute 2 and
# the saver; node B takes compute 1 and 3.
#
# ⚠ NOT IN dev's SUBMIT SCRIPTS. It is exported by the campaign scripts on the
# UNMERGED `one-walker-replicas` branch (with a test keeping it there), so a
# 5-rank shape launched from dev's scripts will hit the 3/2 refusal. Every
# gate in THIS runbook is a bare mpiexec, so exporting it here is what covers
# them; see the note under G4 for the submit-script path.
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0

# GPUS is the PER-NODE pool, not the total.
export GPUS=0,1

# ★ THE OPT-IN, AND EVERY GATE BELOW DEPENDS ON IT. The unified routing is
# OFF by default in the library (that is what made merging it to `dev` a
# no-op), so without this line `build_layout` applies the pre-2026-09-23
# rule: NWALKERS must divide the compute-rank count. G3's new shapes would
# then RAISE, and -- worse -- G5 would silently take the legacy equal-band-
# COUNT split instead of the source-weighted one and measure ~2x at R=4.
export GF_GPU_ROUTING=1
```

Every layout line printed below ends with `gpu_routing=on`. If you see
`gpu_routing=OFF(legacy)`, the export did not reach the ranks — stop, because
nothing after that point measures what it claims to.

**Sanity check before anything else** — this must print 5 distinct ranks and
two distinct hostnames:

```bash
mpiexec -n 5 -ppn 1 python -c \
 "from mpi4py import MPI;import socket;c=MPI.COMM_WORLD;\
print(c.Get_rank(), c.Get_size(), socket.gethostname())"
```

If `Get_size()` is 1 on every line, the launcher is making singleton worlds —
stop and fix that before reading anything else as a layout result.

> **`python -u` in every command below is load-bearing, not style.** These all
> pipe into `tee`/`grep`, and Python block-buffers stdout when it is a pipe
> rather than a tty. If the job then dies — an MPI abort `SIGKILL`s the
> siblings — the buffer dies with it and the command prints NOTHING, which
> reads as "no output, nothing happened" instead of "it died before
> flushing". Cost two rounds of confusion on 2026-09-23.


---

## G3 — layout dry runs (~2 min, no GPU work)

Resolves the layout on all 5 ranks and exits before `fit.build()`.

```bash
cd "$LAT"
for NW in 4 2 1 6 24; do
  echo "=== NWALKERS=$NW ==="
  GF_LAYOUT_DRY_RUN=1 NWALKERS=$NW \
    mpiexec -n 5 -ppn 1 python -u scripts/run_global.py --stock gb_no_fg \
    2>&1 | grep -E "walker-block layout|note:" | head -2
done

# explicit R, and a deliberately illegal one
GF_LAYOUT_DRY_RUN=1 NWALKERS=4 RANKS_PER_BLOCK=4 \
  mpiexec -n 5 -ppn 1 python -u scripts/run_global.py --stock gb_no_fg \
  2>&1 | grep "walker-block layout" | head -1
GF_LAYOUT_DRY_RUN=1 NWALKERS=4 RANKS_PER_BLOCK=3 \
  mpiexec -n 5 -ppn 1 python -u scripts/run_global.py --stock gb_no_fg \
  2>&1 | grep -oE "RANKS_PER_BLOCK=3 does not divide.*" | head -1
```

**Expect** (`n_compute=4` throughout):

| NWALKERS | n_blocks | block | ranks_per_block |
|---|---|---|---|
| 4 | 4 | 1 | AUTO→1 |
| 2 | 2 | 1 | AUTO→2 |
| 1 | 1 | 1 | AUTO→4 (`REPLICAS`) |
| 6 | 2 | 3 | AUTO→2 |
| 24 | 4 | 6 | AUTO→1 |
| 4 + `RANKS_PER_BLOCK=4` | 1 | 4 | 4 |

`RANKS_PER_BLOCK=3` must refuse and name the legal divisors. Every rank must
print the SAME table, and each rank's `devices=[...]` must be a device its own
node actually has.

**Stop here if any shape disagrees across ranks** — nothing below is
interpretable if the layout is not resolving identically.

---

## G4 — three-factorization parity (~30 min)

Same code, same injected source, three ways of spending the same 4 GPUs. This
is what shows the factorization is a *routing* choice.

```bash
cd "$LAT"
COMMON="NUM_ITERATIONS=4 GF_FANOUT_DIGEST=1"

# (a) 4 blocks x R=1 -- today's walker-block layout
env $COMMON FILE_STORE_DIR=$G/g4_a/ NWALKERS=4 RANKS_PER_BLOCK=1 \
  mpiexec -n 5 -ppn 1 python -u scripts/diagnostics/gate_run.py --stock gb_no_fg \
  2>&1 | tee $G/g4_a.log | tail -5

# (b) 2 blocks x R=2 -- THE NEW AXIS (previously impossible)
env $COMMON FILE_STORE_DIR=$G/g4_b/ NWALKERS=2 RANKS_PER_BLOCK=2 \
  mpiexec -n 5 -ppn 1 python -u scripts/diagnostics/gate_run.py --stock gb_no_fg \
  2>&1 | tee $G/g4_b.log | tail -5

# (c) 1 block x R=4 -- one-walker replicas (the pre-existing mode)
env $COMMON FILE_STORE_DIR=$G/g4_c/ NWALKERS=1 RANKS_PER_BLOCK=4 \
  mpiexec -n 5 -ppn 1 python -u scripts/diagnostics/gate_run.py --stock gb_no_fg \
  2>&1 | tee $G/g4_c.log | tail -5
```

**A FRESH `FILE_STORE_DIR` PER ARM IS MANDATORY.** A reused store RESUMES, and
a walker-count mismatch aborts with "walker-count mismatch" — which looks like
a layout failure and is not.

> **Expect every block to STRADDLE BOTH NODES in arms (b) and (c), and do not
> read the `[FANOUT]` times here as production numbers.** Blocks are
> contiguous in compute-rank order, so at R=2 block 0 is ranks (0,1) — and
> `-ppn 1` places those round-robin, i.e. on different hosts. The intra-block
> ledger allgather therefore crosses the TCP fabric. Confirmed on hardware
> 2026-09-24. It does not affect what G4 measures, which is whether the
> factorization changes the ANSWER, not how fast it gets there. Production
> co-locates a block instead: the submit script switches to
> `--distribution=block:block` whenever `RANKS_PER_BLOCK > 1`, for exactly
> this reason.

Read out:

```bash
for f in $G/g4_[abc].log; do
  echo "=== $f ==="
  grep -c "log_like_final disagrees" $f          # MUST be 0
  grep -oE "block path with [0-9]+ rank" $f | head -1   # psd/galfor at R>1
  grep -oE "Alive sources per temp[^;]*" $f | tail -1
done
```

**Pass:** no `log_like_final disagrees` in any arm; arms (b) and (c) print the
`block path with N rank(s) per block` line for psd/galfor; `Alive sources per
temp` does **not** scale with R (it was inflated by exactly R before the fix).

**These arms sample different walker counts, so their chains differ.** That is
not a failure — the gate is about mechanism, not agreement.

---

## G5 — ★ scaling readout (~1 h) — THE DECISION GATE

One walker, R = 1 → 2 → 4. The only measurement of the replica axis that
exists.

> **PREREQUISITE:** the source-weighted band split must be in this checkout.
> Verify before spending the hour, because on the old count-based split this
> measures ~2× at R=4 instead of ~4× and the honest reading would be
> "dispersal doesn't work":
>
> ```bash
> grep -A2 "def _replica_band_weights" src/lisatools/globalfit/moves/gbspecialstretch.py \
>   | grep -q "return None" && echo "STOP: old count split" || echo "OK: weighted"
> ```
>
> **AND `GF_GPU_ROUTING=1` must be exported** (section 0). The weighted split
> is gated on it — with the knob off the code is present but the equal-count
> split is what runs, which is exactly the ~2x-at-R=4 reading this
> prerequisite exists to prevent. Confirm from the run's own layout line:
>
> ```bash
> grep -m1 "walker-block layout" $G/g5_r4.log   # must contain gpu_routing=on
> ```

```bash
cd "$LAT"
# GB_PROP_TIMING_SYNC=all: drain EVERY run device at each span boundary, so a
# span carries its own kernel time. At the default (0) the spans are HOST wall,
# and any span containing a sync point absorbs the drain of everything queued
# before it -- the precedent is `fill_indmap_data`: 598 s measured, 45 s real.
# For a SCALING readout that misattribution is fatal, so pay the syncs here.
# (`[GB_TIMING]` itself is always emitted; there is no enable knob.)
S="NWALKERS=1 NUM_ITERATIONS=5 GB_PROP_TIMING_SYNC=all"

# R=1 -- baseline, ONE compute rank + saver, one GPU
env $S FILE_STORE_DIR=$G/g5_r1/ GPUS=0 \
  mpiexec -n 2 -ppn 1 python -u scripts/diagnostics/gate_run.py --stock gb_no_fg \
  2>&1 | tee $G/g5_r1.log | tail -3

# R=2 -- two compute ranks + saver
env $S FILE_STORE_DIR=$G/g5_r2/ GPUS=0,1 \
  mpiexec -n 3 -ppn 1 python -u scripts/diagnostics/gate_run.py --stock gb_no_fg \
  2>&1 | tee $G/g5_r2.log | tail -3

# R=4 -- four compute ranks + saver, both nodes
env $S FILE_STORE_DIR=$G/g5_r4/ GPUS=0,1 \
  mpiexec -n 5 -ppn 1 python -u scripts/diagnostics/gate_run.py --stock gb_no_fg \
  2>&1 | tee $G/g5_r4.log | tail -3
```

### The numbers to bring back

```bash
for R in 1 2 4; do
  echo "=== R=$R ==="
  grep "GB_TIMING" $G/g5_r$R.log | tail -1 | tr ' ' '\n' \
    | grep -E "^(total|run_proposal|run_tempering|temper_swap_grid|sorter_build|unit_open_close|buffer_build|temper_swap_score)=" 
  grep -oE "wait_s=[0-9.]+|max_rank_s=[0-9.]+|head_s=[0-9.]+" $G/g5_r$R.log | tail -6
done
```

1. **`f_fixed` — the ceiling.** `temper_swap_grid` + `sorter_build` +
   `unit_open_close` as a fraction of `total`. These do NOT divide by R, so
   `1/f_fixed` is the hard ceiling on what dispersal can ever buy. The 1-year
   log bounds the *instrumented* part at ~4 % (ceiling ~25×) — but that was a
   LOWER bound, because the swap-grid build had no span until now. **This run
   turns that ceiling from an estimate into a number.**
2. **Speedup.** `run_proposal` at R=2 and R=4 against R=1. With the weighted
   split, expect close to 2× and 4× on the dispersed part. **Landing near 2×
   at R=4 means checking the weights before concluding anything about the
   design.**
3. **`buffer_build`.** The refill-round cost — prices whether more GPUs buy
   back single-pass buffer residency or only ~5 % of the move.
4. **`wait_s` and `max_rank_s − head_s`.** Load imbalance on the replica axis.
   The block axis measures ~99 % efficient today; this is its counterpart.

### Decision

* R=4 approaching ~4× on the dispersed part, `f_fixed` small → **the replica
  axis pays; proceed to G6 at the target shape.**
* R=4 near 2× → check `_replica_band_weights` is not `None` **before**
  concluding anything.
* `f_fixed` large (say >15 %) → dispersal is capped below ~7× regardless of
  GPU count; spend the GPUs on walkers instead.

---

## Cleanup

```bash
du -sh $G           # stores are not small
# rm -rf $G         # once the logs are copied off
```

## What this allocation canNOT tell you

* **12/18/24-month behaviour.** These gates run the 6-month stock fit.
* **The multi-walker F-stat union.** `GB_FSTAT_FIT_WALKERS` defaults to 1 and
  its sweep loop is unwired; setting it >1 does nothing yet.
* **Production throughput.** G6 on the real recipe is a separate ask.
