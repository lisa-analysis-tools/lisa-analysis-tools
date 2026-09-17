# Parallel F-stat epoch fit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the GB F-stat epoch grid fit's stage B across every MPI compute rank (contiguous box ranges, per-rank checkpoints, `.npy` partials on the shared epoch dir) and fix the port defect that the fit's reference walker is chosen from the head's local block only.

**Architecture:** The head, inside `GBSpecialBase.setup()`, picks the reference walker by GLOBAL argmax over `WalkerFanout.gather_likelihood(acs)`. A new symmetric GB fan-out op `gb_fstat_ref_row` has the OWNING rank open the GB-free window on its LOCAL row, snapshot that walker's residual + inverse-PSD rows to host, restore its residual, and `Bcast` the pair to every compute rank; every rank wraps it in a public `FStatRefRowHolder` scored with `data_index=noise_index=0`. Stage A stays on the head, scored through that holder. Stage B is split per Mc group by contiguous box range via a second op `gb_fstat_stage_b`; each rank runs the EXISTING `run_stacked_peak_sweep` on its sliced inputs, saves a raw float64 `.npy` partial, and the head concatenates the partials in box order and writes the SAME `.npz` it writes today. Bit-identity of the assembled grid against the serial fit is the acceptance gate.

**Tech Stack:** Python 3.12, numpy, `unittest`; `lisatools.globalfit.communication` (`WalkerFanout`, `ComputeService`, `WalkerBlockLayout`, `FakeWorld`); `lisatools.sampling.fstat_gridfit`; `lisatools.globalfit.moves.gbspecialstretch` / `gbbands`. CPU-only on the laptop.

**Spec:** `docs/superpowers/plans/../specs/2026-09-16-parallel-fstat-fit-design.md` (in this worktree: `docs/superpowers/specs/2026-09-16-parallel-fstat-fit-design.md`). Evidence with file:line: the exploration notes quoted inline in each task below.

**Branch / worktree:** `fstat-parallel-fit` in `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-fstatfit`, based on `dev` at `cd930b87`.

## Deviation from the spec's architecture sketch (recorded, deliberate)

The spec's architecture block sketches the head as: factor the stage-B host
prep out into a plan, loop the groups on the head, fan each out, assemble,
then build the proposal and write the npz on the head.

This plan instead keeps ALL of that inside `run_stacked_stage_b` and injects
only the per-group kernel stream, through a `sweep_runner(spec, call_fstat,
xp=xp)` seam (Task 2). Reason: the spec's own decisions 5 and 7 require the
proposal build, the npz keys and the assembled grid to be *exactly* what the
serial fit produces, and the cheapest way to guarantee that is for the
parallel path to run the serial code — not a second copy of it. The writer is
still factored out (`write_stacked_npz`, decision 5) and is called by both
formats' branches. Everything the spec specifies as an outcome — contiguous
equal-count box ranges, per-rank checkpoint names, `.npy` partials, metadata-
only replies, `np.concatenate` in box order, identical npz keys, DONE.json
written last with the global `walker_ref` and `n_compute` — is unchanged.

## Global Constraints

These apply to EVERY task and every subagent.

- **Laptop rules (binding).** 8 GB RAM, CPU only. **ONE python process at a time on the whole machine.** Before every python run, check
  `ps -axo comm=,args= | awk '$1 ~ /python/ && /unittest/'`
  and if it prints anything, `sleep 30` and re-check (up to 30 minutes) before starting. Never more than 6 test modules per process. **NEVER import or run `tests/test_gbspecial_flow.py`** (10-26 GB). Tiny synthetic fixtures only.
- **How to run tests.** From the worktree root:
  ```sh
  source /Users/mkatz/miniconda3/etc/profile.d/conda.sh && conda activate deving
  .wtenv/wt_run.sh $PWD/src .wtenv/<name>.log python -m unittest tests.test_<name> -v
  ```
  `.wtenv/wt_run.sh` is the load-gated, `nice -10`, single-thread runner; it writes the output to the log file, so `cat .wtenv/<name>.log` afterwards. NEVER run bare `python -m unittest` (it would import the editable-installed main checkout, not this worktree).
- **Git.** Commit per task on branch `fstat-parallel-fit` only. **NEVER push. NEVER merge into dev. Never `git stash`. Never `git add -A`** — name the files. Never commit `.wtenv/` or `.superpowers/` (both are in `.git/info/exclude`).
- **Every commit message ends with EXACTLY this trailer, verbatim, on its own last line:**
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  ```
- **No backend strings as function kwargs** (repo rule): backend is chosen at construction; dispatch via `self.backend` / `self.xp`.
- **Deepcopy / pickle safety** (repo rule): never store an array module (`self.xp = cp`) as an instance attribute; guard `__getattr__` delegators against underscore/dunder probing.
- **Bit-identity is the gate.** The sweep has no RNG and no cross-row reduction. Any change that could alter the serial numbers is a defect, not a trade-off.
- **`n_compute == 1` is a direct call**: no MPI, no pickling, no extra copies beyond today's.
- **Style:** `black` line-length 100, `isort` black profile. Match the surrounding file's comment density — this codebase documents WHY in long comments; keep that.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `tests/test_fstat_parallel_fit.py` | NEW. The whole feature's test surface: golden serial stage-B npz, the split helpers, the 1-rank-vs-2-rank FakeWorld bit-identity gate, resume. | 1, 3, 10 |
| `tests/data/fstat_stage_b_golden_single.npz`, `tests/data/fstat_stage_b_golden_grouped.npz` | NEW. Goldens captured from the UNCHANGED serial code. | 1 |
| `src/lisatools/sampling/fstat_gridfit.py` | Library. Gains `StageBGroupSpec`, `run_stage_b_group`, `write_stacked_npz`, `split_box_range`, the `.npy` partial I/O + assembly helpers, and a `sweep_runner=` injection point on `run_stacked_stage_b`. No MPI ever enters this file. | 2, 3 |
| `src/lisatools/globalfit/moves/gbbands.py` | `FStatRefRowHolder` made public + `snapshot_ref_rows` factored out of `_sighet_fstat_multidevice`. | 4 |
| `src/lisatools/globalfit/communication/ranks.py` | `WalkerBlockLayout.owner_of(w)`. | 5 |
| `src/lisatools/globalfit/communication/fakecomm.py` | `FakeComm.Bcast` (buffer collective) so the FakeWorld gate can exercise the real code path. | 6 |
| `src/lisatools/globalfit/moves/gbspecialstretch.py` | The orchestration: global reference walker, the two new GB ops in `GB_OPS` + `gf_serve`, the head-side fan-out helpers, `_fstat_call(holder=...)`, `_run_fstat_fit` and `_install_ctr_table` rewiring. | 5, 6, 7, 8, 9 |
| `docs/multirank-cluster-gates.md`, `docs/global-fit-launch.md` | Runbook: the epoch-1 cluster verification steps and the `FSTAT_SIGHET_MULTIDEV` note. | 11 |

---

## Task 1: Golden serial stage-B npz

Capture what the CURRENT serial code produces, before anything changes, and pin it. Every later task must keep this test green — that is the bit-identity gate for the single-process path.

**Files:**
- Create: `tests/test_fstat_parallel_fit.py`
- Create: `tests/data/fstat_stage_b_golden_single.npz`, `tests/data/fstat_stage_b_golden_grouped.npz`

**Interfaces:**
- Consumes: `lisatools.sampling.fstat_gridfit.run_stacked_stage_b(call_fstat, peaks, *, xp, Tobs, band_edges_hz, mc_lims, ratio_max=None, cache_path=None, fingerprint_extra="", epoch=None)` (existing, unchanged in this task).
- Produces, for later tasks in this module:
  - `_fake_call_fstat(counter=None, raise_after=None) -> callable(params) -> (N (n,4), M_upper (n,10))`
  - `BAND_EDGES: np.ndarray` (7 edges, 6 sub-bands), `TOBS: float`
  - `make_peaks() -> np.ndarray (K,4)` — deterministic `(f0_mHz, F, node_idx, band_idx)` rows
  - `stage_b_env(**overrides) -> contextmanager` — pins the `FSTAT_*` knobs
  - `run_golden(tmpdir, *, grouped) -> str` — runs the serial stage B, returns the written `_peaks_stacked.npz` path
  - `write_goldens(out_dir="tests/data") -> None` — the capture entry point
  - `assert_npz_identical(testcase, path_a, path_b)` — every key, byte-for-byte

- [ ] **Step 1: Write the test module**

Create `tests/test_fstat_parallel_fit.py`:

```python
"""Parallel F-stat epoch fit: goldens, split helpers and the fan-out gate.

CPU-only, one python process, tiny synthetic fixtures. ``call_fstat`` is the
analytic fake from ``tests/test_fstat_gridfit.py`` (no kernel, no GPU, no
sampler), so every number here is a deterministic function of the node grid
-- which is exactly what makes bit-identity a meaningful assertion.

The goldens in ``tests/data`` were captured from the SERIAL stage B before
the parallel split existed. They are the single-process regression gate:
``n_compute == 1`` must stay byte-identical forever.
"""

import contextlib
import os
import shutil
import tempfile
import unittest

import numpy as np

from lisatools.sampling import fstat_gridfit as G

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
GOLDEN_SINGLE = os.path.join(DATA_DIR, "fstat_stage_b_golden_single.npz")
GOLDEN_GROUPED = os.path.join(DATA_DIR, "fstat_stage_b_golden_grouped.npz")

#: 6 sub-bands spanning a wide enough f0 range that the per-box Mc
#: requirement (~f0^(11/3)) crosses a ladder level -- that is what makes the
#: GROUPED golden actually carry more than one group.
BAND_EDGES = np.linspace(6.0e-3, 18.0e-3, 7)
TOBS = 7.776e6  # 90 d, the value the other fstat tests use


def _fake_call_fstat(counter=None, raise_after=None):
    """``params -> (N (n,4), M_upper (n,10))`` with an f0-dependent bump.

    ``M_upper`` is the upper triangle of the identity, so ``compute_fstat``
    reduces to ``0.5 * sum(N**2)`` and F is a clean analytic function of f0.
    Copied from tests/test_fstat_gridfit.py so this module never imports it
    (that module pins FSTAT_FDOT_AXIS process-wide in setUpModule).
    """
    state = {"rows": 0}

    def call(params):
        p = np.asarray(params.get() if hasattr(params, "get") else params)
        n = p.shape[0]
        if raise_after is not None and state["rows"] + n > raise_after:
            raise RuntimeError("simulated death mid-sweep")
        state["rows"] += n
        if counter is not None:
            counter["calls"] += 1
            counter["rows"] += n
        f0_mHz = p[:, 1] * 1e3
        amp = np.zeros(n)
        for c, a in ((6.5, 40.0), (9.0, 25.0), (12.0, 60.0), (16.0, 35.0)):
            amp += a * np.exp(-0.5 * ((f0_mHz - c) / 2e-3) ** 2)
        N = np.zeros((n, 4))
        N[:, 0] = np.sqrt(2.0 * np.maximum(amp, 0.0))
        M = np.zeros((n, 10))
        M[:, 0] = M[:, 4] = M[:, 7] = M[:, 9] = 1.0
        return N, M

    return call


def make_peaks():
    """A deterministic ``(K, 4)`` peak table spanning every interior band.

    Columns are ``select_comb_peaks``' contract: ``(f0_mHz, F, node_idx,
    band_idx)``. Built by hand rather than run through the comb so the box
    set -- and therefore the group structure -- is fixed by this file and
    cannot drift with a comb knob.
    """
    edges_mHz = BAND_EDGES * 1e3
    rows = []
    for bi in range(1, len(edges_mHz) - 2):  # interior bands only
        lo, hi = edges_mHz[bi], edges_mHz[bi + 1]
        for j, frac in enumerate((0.25, 0.5, 0.75)):
            f0 = lo + frac * (hi - lo)
            rows.append((f0, 10.0 + 3.0 * bi + j, 100 * bi + j, bi))
    return np.asarray(rows, dtype=float)


@contextlib.contextmanager
def stage_b_env(**overrides):
    """Pin every knob stage B reads, so a golden is reproducible."""
    env = {
        "FSTAT_BATCH": "512",
        "FSTAT_CKPT_SECS": "0",          # checkpoint every chunk
        "FSTAT_N_ALPHA": "2",
        "FSTAT_N_SINDELTA": "2",
        "FSTAT_PEAK_HALF_MHZ": "0.02",
        "FSTAT_FDOT_AXIS": "0",          # Mc basis: the documented escape
        "FSTAT_MC_GROUPING": "1",
        "FSTAT_PEAK_WEIGHTING": "fstat",
        "FSTAT_GRID_MEM_MB": "",
    }
    env.update({k: str(v) for k, v in overrides.items()})
    old = {k: os.environ.get(k) for k in env}
    try:
        for k, v in env.items():
            if v == "":
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def run_golden(tmpdir, *, grouped, sweep_runner=None):
    """Run the serial stage B into ``tmpdir``; return the stacked npz path.

    ``grouped=False`` pins ``FSTAT_N_MC`` so every box lands on one ladder
    level (one group, the legacy npz keys); ``grouped=True`` lets the auto
    criterion size each box, which the wide band grid splits into several.
    """
    cache_path = os.path.join(tmpdir, G.GRID_BASENAME)
    extra = {} if grouped else {"FSTAT_N_MC": "3"}
    kwargs = {} if sweep_runner is None else {"sweep_runner": sweep_runner}
    with stage_b_env(**extra):
        G.run_stacked_stage_b(
            _fake_call_fstat(), make_peaks(), xp=np, Tobs=TOBS,
            band_edges_hz=BAND_EDGES, mc_lims=[0.01, 1.0],
            cache_path=cache_path, fingerprint_extra="|epoch=0|gbfree=1",
            epoch=0, **kwargs)
    return cache_path.replace(".npz", "_peaks_stacked.npz")


def write_goldens(out_dir=DATA_DIR):
    """Capture both goldens from whatever stage B currently does."""
    os.makedirs(out_dir, exist_ok=True)
    for grouped, dest in ((False, GOLDEN_SINGLE), (True, GOLDEN_GROUPED)):
        d = tempfile.mkdtemp()
        try:
            shutil.copyfile(run_golden(d, grouped=grouped), dest)
            print(f"wrote {dest}")
        finally:
            shutil.rmtree(d, ignore_errors=True)


def assert_npz_identical(tc, path_a, path_b):
    """Every key present in both, byte-for-byte equal (dtype + shape too)."""
    a = np.load(path_a, allow_pickle=False)
    b = np.load(path_b, allow_pickle=False)
    tc.assertEqual(sorted(a.files), sorted(b.files),
                   f"key sets differ: {sorted(a.files)} vs {sorted(b.files)}")
    for key in sorted(a.files):
        xa, xb = np.asarray(a[key]), np.asarray(b[key])
        tc.assertEqual(xa.dtype, xb.dtype, f"{key}: dtype")
        tc.assertEqual(xa.shape, xb.shape, f"{key}: shape")
        tc.assertEqual(xa.tobytes(), xb.tobytes(), f"{key}: bytes differ")


class GoldenSerialStageBTest(unittest.TestCase):
    """The serial stage B must never move. This is the n_compute==1 gate."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_single_group_matches_the_golden(self):
        got = run_golden(self.d, grouped=False)
        assert_npz_identical(self, got, GOLDEN_SINGLE)

    def test_grouped_matches_the_golden(self):
        got = run_golden(self.d, grouped=True)
        assert_npz_identical(self, got, GOLDEN_GROUPED)

    def test_the_goldens_cover_both_npz_formats(self):
        single = np.load(GOLDEN_SINGLE, allow_pickle=False)
        grouped = np.load(GOLDEN_GROUPED, allow_pickle=False)
        self.assertIn("logp_grids", single.files,
                      "the single-group golden must use the LEGACY keys")
        self.assertNotIn("logp_grids", grouped.files)
        self.assertIn("group_sizes", grouped.files)
        self.assertGreaterEqual(
            len(np.asarray(grouped["group_sizes"])), 2,
            "the grouped golden must actually carry >1 Mc group; widen "
            "BAND_EDGES until the ladder splits")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Capture the goldens from the unchanged code**

```sh
source /Users/mkatz/miniconda3/etc/profile.d/conda.sh && conda activate deving
.wtenv/wt_run.sh $PWD/src .wtenv/golden_capture.log \
  python -c "from tests.test_fstat_parallel_fit import write_goldens; write_goldens()"
cat .wtenv/golden_capture.log
```
Expected: two `wrote .../tests/data/fstat_stage_b_golden_*.npz` lines and `[wt_run] exit=0`.

If `test_the_goldens_cover_both_npz_formats` later fails on `group_sizes >= 2`, widen `BAND_EDGES`' upper edge (e.g. `np.linspace(6.0e-3, 24.0e-3, 7)`), add a matching bump in `_fake_call_fstat`, re-capture, and re-run. Do NOT weaken the assertion.

- [ ] **Step 3: Check the golden file sizes are tiny**

```sh
ls -l tests/data/fstat_stage_b_golden_*.npz
```
Expected: each well under 1 MB. If either exceeds 1 MB, shrink the fixture (`FSTAT_PEAK_HALF_MHZ`, fewer peaks per band) and re-capture — these are committed files.

- [ ] **Step 4: Run the test module**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t1.log python -m unittest tests.test_fstat_parallel_fit -v
cat .wtenv/t1.log
```
Expected: 3 tests, all PASS.

- [ ] **Step 5: Commit**

```sh
git add tests/test_fstat_parallel_fit.py tests/data/fstat_stage_b_golden_single.npz tests/data/fstat_stage_b_golden_grouped.npz
git commit -m "$(cat <<'EOF'
test(fstat): golden serial stage-B npz (both formats) as the n_compute==1 bit-identity gate

Captured from the unchanged serial run_stacked_stage_b on a deterministic
analytic call_fstat: the single-group legacy-key npz and the multi-group
one. Every later step of the parallel-fit work must keep these byte-identical.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Factor the group sweep and the npz writer out of `run_stacked_stage_b`

Pure refactor. The serial path must stay byte-identical (Task 1's goldens prove it). This creates the seam the fan-out plugs into and the writer the head calls.

Evidence: `run_stacked_stage_b` at `src/lisatools/sampling/fstat_gridfit.py:970`; the per-group loop at `:1114-1152` (`node_shape = (b - a, n_f0, n_Mc, n_alpha, n_sd)` at `:1140`, checkpoint name `"stageb"` / `f"stageb_g{gi}"` at `:1151`); the single-group legacy writer at `:1170-1186`; the grouped writer at `:1217-1229`.

**Files:**
- Modify: `src/lisatools/sampling/fstat_gridfit.py`
- Test: `tests/test_fstat_parallel_fit.py` (Task 1's goldens; add one new test)

**Interfaces:**
- Consumes: Task 1's `run_golden(tmpdir, *, grouped, sweep_runner=None)`, `assert_npz_identical`.
- Produces:
  - `@dataclasses.dataclass(frozen=True) class StageBGroupSpec` with fields `gi: int, n_groups: int, a: int, b: int, f0_los: np.ndarray, f0_dxs: np.ndarray, mc_ax: np.ndarray, alpha_ax: np.ndarray, sd_ax: np.ndarray, node_shape: tuple, ckpt_name: str | None, parts_dir: str | None, fingerprint_extra: str, fdot_axis: bool, c_t: float`; properties `n_boxes -> int`; method `sub_range(a2, b2, *, ckpt_name=None) -> StageBGroupSpec`.
  - `run_stage_b_group(spec: StageBGroupSpec, call_fstat, *, xp) -> grid` — shape `spec.node_shape`.
  - `write_stacked_npz(stacked_path, *, grids_g, mc_ax_g, f0_los, f0_dxs, alpha_ax, sd_ax, grid_basis, grid_c_t, peaks, band_idx, band_edges_mHz, band_edges_hz, group_sizes=None) -> None`.
  - `run_stacked_stage_b(..., sweep_runner=None)` — `sweep_runner(spec, call_fstat, xp=xp) -> grid`; `None` = today's serial call.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fstat_parallel_fit.py` (before the `if __name__` block):

```python
class SweepRunnerInjectionTest(unittest.TestCase):
    """``sweep_runner`` sees one spec per group and may return any grid."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_default_runner_is_the_serial_path(self):
        seen = []

        def runner(spec, call_fstat, *, xp):
            seen.append(spec)
            return G.run_stage_b_group(spec, call_fstat, xp=xp)

        got = run_golden(self.d, grouped=True, sweep_runner=runner)
        assert_npz_identical(self, got, GOLDEN_GROUPED)
        self.assertGreaterEqual(len(seen), 2, "one spec per Mc group")
        for gi, spec in enumerate(seen):
            self.assertEqual(spec.gi, gi)
            self.assertEqual(spec.n_boxes, spec.b - spec.a)
            self.assertEqual(spec.node_shape[0], spec.n_boxes)
            self.assertEqual(len(spec.f0_los), spec.n_boxes)
            self.assertEqual(len(spec.f0_dxs), spec.n_boxes)

    def test_sub_range_slices_boxes_and_renames_the_checkpoint(self):
        seen = []

        def runner(spec, call_fstat, *, xp):
            seen.append(spec)
            return G.run_stage_b_group(spec, call_fstat, xp=xp)

        run_golden(self.d, grouped=True, sweep_runner=runner)
        spec = seen[0]
        if spec.n_boxes < 2:
            self.skipTest("first group has a single box")
        mid = spec.a + spec.n_boxes // 2
        left = spec.sub_range(spec.a, mid, ckpt_name="stageb_g0_r0")
        right = spec.sub_range(mid, spec.b, ckpt_name="stageb_g0_r1")
        self.assertEqual(left.n_boxes + right.n_boxes, spec.n_boxes)
        self.assertEqual(left.node_shape, (left.n_boxes,) + tuple(spec.node_shape[1:]))
        np.testing.assert_array_equal(
            np.concatenate([left.f0_los, right.f0_los]), spec.f0_los)
        np.testing.assert_array_equal(
            np.concatenate([left.f0_dxs, right.f0_dxs]), spec.f0_dxs)
        self.assertEqual(left.ckpt_name, "stageb_g0_r0")
        # the axes and the basis are group-wide, never sliced
        np.testing.assert_array_equal(left.mc_ax, spec.mc_ax)
        np.testing.assert_array_equal(right.alpha_ax, spec.alpha_ax)
        self.assertEqual(left.c_t, spec.c_t)
        self.assertEqual(right.fdot_axis, spec.fdot_axis)

    def test_split_sweeps_reassemble_bit_identically(self):
        """Two half-range sweeps concatenated == the whole-group sweep."""
        def runner(spec, call_fstat, *, xp):
            if spec.n_boxes < 2:
                return G.run_stage_b_group(spec, call_fstat, xp=xp)
            mid = spec.a + spec.n_boxes // 2
            parts = [
                G.run_stage_b_group(
                    spec.sub_range(lo, hi, ckpt_name=f"{spec.ckpt_name}_r{i}"),
                    call_fstat, xp=xp)
                for i, (lo, hi) in enumerate(
                    ((spec.a, mid), (mid, spec.b)))
            ]
            return np.concatenate(parts, axis=0)

        got = run_golden(self.d, grouped=True, sweep_runner=runner)
        assert_npz_identical(self, got, GOLDEN_GROUPED)
```

- [ ] **Step 2: Run it to see it fail**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t2a.log \
  python -m unittest tests.test_fstat_parallel_fit.SweepRunnerInjectionTest -v
cat .wtenv/t2a.log
```
Expected: FAIL — `TypeError: run_stacked_stage_b() got an unexpected keyword argument 'sweep_runner'` (and `AttributeError: module ... has no attribute 'run_stage_b_group'`).

- [ ] **Step 3: Add `StageBGroupSpec` and `run_stage_b_group`**

In `src/lisatools/sampling/fstat_gridfit.py`, add `import dataclasses` to the stdlib imports at the top, then insert this block immediately AFTER `run_stacked_peak_sweep` ends (just before `def mc_ladder_levels`, ~line 945):

```python
@dataclasses.dataclass(frozen=True)
class StageBGroupSpec:
    """Everything one stage-B Mc group's sweep needs, as host arrays.

    The unit the parallel fit ships and slices. ``a``/``b`` are ABSOLUTE box
    indices into the f0-sorted global box order, so a sub-range is addressed
    in the same coordinates the assembled grid is concatenated in (box is
    the SLOWEST axis of ``node_shape``, :func:`run_stacked_peak_sweep`).

    Everything except ``f0_los``/``f0_dxs``/``node_shape[0]`` is group-wide
    and is NEVER sliced: the Mc/alpha/sin-delta axes, the basis flag and the
    shear coefficient define what a row MEANS, and a rank that sliced them
    would score a different physical template while reporting the same box.
    """

    gi: int
    n_groups: int
    a: int
    b: int
    f0_los: np.ndarray
    f0_dxs: np.ndarray
    mc_ax: np.ndarray
    alpha_ax: np.ndarray
    sd_ax: np.ndarray
    node_shape: tuple
    ckpt_name: Optional[str]
    parts_dir: Optional[str]
    fingerprint_extra: str
    fdot_axis: bool
    c_t: float

    @property
    def n_boxes(self) -> int:
        return int(self.b) - int(self.a)

    def sub_range(self, a2, b2, *, ckpt_name=None) -> "StageBGroupSpec":
        """This group restricted to ABSOLUTE boxes ``[a2, b2)``.

        ``ckpt_name`` must differ per rank: the sweep's fingerprint hashes
        the SLICED inputs plus ``node_shape``, so two ranks' checkpoints are
        already mutually invalid -- but they must not collide on one path.
        """
        a2, b2 = int(a2), int(b2)
        if not (self.a <= a2 <= b2 <= self.b):
            raise ValueError(
                f"sub_range({a2}, {b2}) is outside group {self.gi}'s "
                f"box range [{self.a}, {self.b})")
        i0, i1 = a2 - int(self.a), b2 - int(self.a)
        return dataclasses.replace(
            self,
            a=a2,
            b=b2,
            f0_los=np.ascontiguousarray(self.f0_los[i0:i1]),
            f0_dxs=np.ascontiguousarray(self.f0_dxs[i0:i1]),
            node_shape=(i1 - i0,) + tuple(self.node_shape[1:]),
            ckpt_name=(self.ckpt_name if ckpt_name is None else ckpt_name),
        )


def run_stage_b_group(spec: StageBGroupSpec, call_fstat: Callable, *, xp):
    """Sweep ONE stage-B group (or one rank's box range of it).

    The single entry point both the serial fit and every compute rank use,
    so a split can never diverge from the serial code by construction. An
    EMPTY range (more ranks than boxes) short-circuits: the sweep's
    checkpoint layer is not defined at ``n_total == 0``, and there is
    nothing to score.
    """
    if int(spec.node_shape[0]) == 0:
        return xp.empty(tuple(spec.node_shape), dtype=xp.float64)
    ckpt = (os.path.join(spec.parts_dir, spec.ckpt_name)
            if (spec.parts_dir and spec.ckpt_name) else None)
    return run_stacked_peak_sweep(
        call_fstat, spec.f0_los, spec.f0_dxs, spec.mc_ax, spec.alpha_ax,
        spec.sd_ax, spec.node_shape, xp=xp, ckpt=ckpt,
        fingerprint_extra=spec.fingerprint_extra,
        fdot_axis=spec.fdot_axis, c_t=spec.c_t)
```

`Optional` and `Callable` are already imported at the top of the module (`from typing import Callable, Optional`); confirm before adding.

- [ ] **Step 4: Add `write_stacked_npz`**

Insert immediately after `run_stage_b_group`:

```python
def write_stacked_npz(stacked_path, *, grids_g, mc_ax_g, f0_los, f0_dxs,
                      alpha_ax, sd_ax, grid_basis, grid_c_t, peaks,
                      band_idx, band_edges_mHz, band_edges_hz,
                      group_sizes=None):
    """Write the stage-B cache. ``group_sizes=None`` = the LEGACY 1-group keys.

    Factored out of :func:`run_stacked_stage_b` so the parallel fit's head,
    which assembles each group from per-rank partials, writes through the
    SAME code -- the loader (``fstat_proposal.stacked_from_cache``)
    dispatches on ``"logp_grids" in keys``, so the two formats are not
    interchangeable and must never be produced by two separate writers.
    """
    os.makedirs(os.path.dirname(stacked_path), exist_ok=True)
    common = dict(
        f0_los=f0_los, f0_dxs=f0_dxs, alpha_ax=alpha_ax, sin_delta_ax=sd_ax,
        # THE BASIS IS PART OF THE CACHE. Axis 2's VALUES differ between the
        # two meanings, but a consumer that reads them as chirp masses when
        # they are Hz/s gets no error at all -- just births at absurd
        # parameters. Stamp it, and refuse a mismatch on load.
        grid_basis=grid_basis, grid_c_t=float(grid_c_t),
        peak_f0_mHz=peaks[:, 0], peak_F=peaks[:, 1], band_idx=band_idx,
        band_f0_lo=band_edges_mHz[band_idx],
        band_f0_hi=band_edges_mHz[band_idx + 1],
        band_edges=np.asarray(band_edges_hz, dtype=float),
    )
    if group_sizes is None:
        np.savez(stacked_path, logp_grids=_to_host(grids_g[0]),
                 mc_ax=mc_ax_g[0], **common)
        return
    group_arrays = {}
    for gi in range(len(grids_g)):
        group_arrays[f"logp_grids_g{gi}"] = _to_host(grids_g[gi])
        group_arrays[f"mc_ax_g{gi}"] = mc_ax_g[gi]
    np.savez(stacked_path, group_sizes=np.asarray(group_sizes, dtype=int),
             **common, **group_arrays)
```

- [ ] **Step 5: Rewire `run_stacked_stage_b` onto the new pieces**

Add `sweep_runner=None` to the signature (`fstat_gridfit.py:970-975`), keyword-only, after `epoch=None`. Document it in the docstring with one paragraph:

```
    ``sweep_runner`` replaces the per-group kernel stream with
    ``runner(spec, call_fstat, xp=xp) -> grid`` (``spec`` is a
    :class:`StageBGroupSpec`). ``None`` is the serial path,
    :func:`run_stage_b_group`, and is byte-identical to the historical
    code. The multi-rank fit passes a runner that splits the group by
    contiguous box range across the compute ranks and concatenates the
    partials -- box is the SLOWEST axis, so concatenation on axis 0 in box
    order reproduces the whole-group sweep exactly.
```

Replace the `grids_g.append(run_stacked_peak_sweep(...))` call (`:1149-1155`) with:

```python
        spec = StageBGroupSpec(
            gi=gi, n_groups=n_groups, a=a, b=b,
            f0_los=f0_los[a:b], f0_dxs=f0_dxs[a:b], mc_ax=mc_ax,
            alpha_ax=alpha_ax, sd_ax=sd_ax, node_shape=node_shape,
            ckpt_name=_ck, parts_dir=_parts,
            fingerprint_extra=fingerprint_extra,
            fdot_axis=_fdot_axis, c_t=_c_t,
        )
        runner = sweep_runner if sweep_runner is not None else run_stage_b_group
        grids_g.append(runner(spec, call_fstat, xp=xp))  # beta = 1: logp = F
```

Leave the `_ck = "stageb" if n_groups == 1 else f"stageb_g{gi}"` line above it exactly as is.

Replace the single-group `np.savez(...)` block (`:1170-1186`) with:

```python
        if cache_path:
            stacked_path = cache_path.replace(".npz", "_peaks_stacked.npz")
            write_stacked_npz(
                stacked_path, grids_g=grids_g, mc_ax_g=mc_ax_g,
                f0_los=f0_los, f0_dxs=f0_dxs, alpha_ax=alpha_ax, sd_ax=sd_ax,
                grid_basis=("fdot" if _fdot_axis else "Mc"),
                grid_c_t=(_c_t if _fdot_axis else 0.0), peaks=peaks,
                band_idx=band_idx, band_edges_mHz=band_edges_mHz,
                band_edges_hz=band_edges_hz, group_sizes=None)
            logger.info("[cache] wrote %s", stacked_path)
            ckpt_clear(_parts, "stageb")
        return stacked
```

Replace the grouped `np.savez(...)` block (`:1210-1233`) with:

```python
    if cache_path:
        stacked_path = cache_path.replace(".npz", "_peaks_stacked.npz")
        write_stacked_npz(
            stacked_path, grids_g=grids_g, mc_ax_g=mc_ax_g,
            f0_los=f0_los, f0_dxs=f0_dxs, alpha_ax=alpha_ax, sd_ax=sd_ax,
            grid_basis=("fdot" if _fdot_axis else "Mc"),
            grid_c_t=(_c_t if _fdot_axis else 0.0), peaks=peaks,
            band_idx=band_idx, band_edges_mHz=band_edges_mHz,
            band_edges_hz=band_edges_hz, group_sizes=_sizes)
        logger.info("[cache] wrote %s (%d Mc groups)", stacked_path, n_groups)
        ckpt_clear(_parts, "stageb")
    return stacked
```

Delete the now-unused `group_arrays` loop that preceded the grouped `np.savez`. Keep `_sizes = np.diff(g_edges).astype(int)` and the `[stageB] banded Mc stacks:` log line exactly where they are.

**Bit-identity note for the reviewer:** `np.savez` writes keys in call order, but `np.load` returns them by name and `assert_npz_identical` compares per key, so key ORDER is not part of the golden. The VALUES must be identical: `grid_c_t=float(_c_t if _fdot_axis else 0.0)` reproduces the old `float(_c_t if _fdot_axis else 0.0)` exactly.

- [ ] **Step 6: Run the tests**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t2b.log \
  python -m unittest tests.test_fstat_parallel_fit tests.test_fstat_gridfit tests.test_fstat_mc_groups tests.test_fstat_fdot_axis -v
cat .wtenv/t2b.log
```
Expected: all PASS, including both golden tests (this is the proof the refactor is byte-neutral).

- [ ] **Step 7: Commit**

```sh
git add src/lisatools/sampling/fstat_gridfit.py tests/test_fstat_parallel_fit.py
git commit -m "$(cat <<'EOF'
fstat_gridfit: StageBGroupSpec + run_stage_b_group + write_stacked_npz, with a sweep_runner seam

Pure refactor: the per-group kernel stream and the two npz writers move
behind named entry points, and run_stacked_stage_b gains an optional
sweep_runner so a caller can split a group by contiguous box range without
reimplementing the prep, the proposal build or the cache format. The serial
path is byte-identical -- both goldens stay green.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Contiguous box split + per-rank partial I/O

Pure functions, no MPI. These are what make the rank→range map reproducible across a resume and keep whole grids out of MPI replies.

Evidence for the design: the sig-het reference blocks are built on f0 boundary crossings with ONE block resident at a time (`gbbands.py` `_ensure_block`, exploration §"Sig-het reference blocks"), so an f0-contiguous range divides the block builds proportionally; interleaving would rebuild every block on every rank (the 4,600-rebuild pathology recorded at `fstat_gridfit.py:1033-1038`). Partials via the shared epoch dir because replies with whole group grids would be 163-653 MB against mpi4py's ~2 GiB pickle cap.

**Files:**
- Modify: `src/lisatools/sampling/fstat_gridfit.py`
- Test: `tests/test_fstat_parallel_fit.py`

**Interfaces:**
- Consumes: Task 2's `StageBGroupSpec`, `run_stage_b_group`.
- Produces:
  - `split_box_range(a, b, n_parts) -> list[tuple[int, int]]`
  - `stage_b_part_path(parts_dir, gi, rank) -> str`
  - `save_stage_b_part(parts_dir, gi, rank, grid) -> tuple[str, int, str]` — `(path, n_rows, sha1_16)`
  - `load_stage_b_part(parts_dir, gi, rank) -> np.ndarray`
  - `assemble_stage_b_group(parts_dir, gi, n_parts, node_shape, *, xp, sha1s=None) -> grid`
  - `clear_stage_b_parts(parts_dir, gi, n_parts) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_fstat_parallel_fit.py`:

```python
class SplitBoxRangeTest(unittest.TestCase):
    def test_contiguous_covering_and_ordered(self):
        for total, n in ((12, 3), (13, 3), (1, 1), (5, 5), (7, 4)):
            parts = G.split_box_range(10, 10 + total, n)
            self.assertEqual(len(parts), n)
            self.assertEqual(parts[0][0], 10)
            self.assertEqual(parts[-1][1], 10 + total)
            for (a0, b0), (a1, _b1) in zip(parts, parts[1:]):
                self.assertLessEqual(a0, b0)
                self.assertEqual(b0, a1, "ranges must be contiguous")
            self.assertEqual(sum(b - a for a, b in parts), total)

    def test_sizes_differ_by_at_most_one(self):
        parts = G.split_box_range(0, 13, 4)
        widths = sorted(b - a for a, b in parts)
        self.assertEqual(widths, [3, 3, 3, 4])
        self.assertEqual(parts, [(0, 4), (4, 7), (7, 10), (10, 13)])

    def test_more_ranks_than_boxes_gives_empty_tail_ranges(self):
        parts = G.split_box_range(0, 2, 4)
        self.assertEqual(parts, [(0, 1), (1, 2), (2, 2), (2, 2)])

    def test_is_a_pure_function_of_its_arguments(self):
        self.assertEqual(G.split_box_range(3, 29, 5), G.split_box_range(3, 29, 5))

    def test_zero_parts_raises(self):
        with self.assertRaises(ValueError):
            G.split_box_range(0, 10, 0)


class StageBPartIOTest(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_round_trip_with_checksum(self):
        rng = np.random.default_rng(7)
        grid = rng.normal(size=(3, 4, 2, 2, 2))
        path, n_rows, sha = G.save_stage_b_part(self.d, 1, 2, grid)
        self.assertTrue(os.path.exists(path))
        self.assertEqual(n_rows, 3)
        back = G.load_stage_b_part(self.d, 1, 2)
        self.assertEqual(back.dtype, np.float64)
        self.assertEqual(back.tobytes(), np.ascontiguousarray(grid).tobytes())
        _p2, _n2, sha2 = G.save_stage_b_part(self.d, 1, 2, grid)
        self.assertEqual(sha, sha2)

    def test_assemble_concatenates_in_rank_order_and_checks_shape(self):
        rng = np.random.default_rng(11)
        whole = rng.normal(size=(5, 4, 2, 2, 2))
        for r, (a, b) in enumerate(G.split_box_range(0, 5, 3)):
            G.save_stage_b_part(self.d, 0, r, whole[a:b])
        got = G.assemble_stage_b_group(self.d, 0, 3, whole.shape, xp=np)
        self.assertEqual(np.asarray(got).tobytes(),
                         np.ascontiguousarray(whole).tobytes())
        with self.assertRaises(RuntimeError):
            G.assemble_stage_b_group(self.d, 0, 3, (6, 4, 2, 2, 2), xp=np)

    def test_assemble_rejects_a_corrupt_partial(self):
        rng = np.random.default_rng(13)
        whole = rng.normal(size=(4, 2, 2, 2, 2))
        shas = {}
        for r, (a, b) in enumerate(G.split_box_range(0, 4, 2)):
            _p, _n, shas[r] = G.save_stage_b_part(self.d, 0, r, whole[a:b])
        G.save_stage_b_part(self.d, 0, 1, whole[2:4] + 1.0)  # tamper
        with self.assertRaises(RuntimeError):
            G.assemble_stage_b_group(self.d, 0, 2, whole.shape, xp=np, sha1s=shas)

    def test_clear_removes_only_this_group(self):
        g = np.zeros((1, 2, 2, 2, 2))
        G.save_stage_b_part(self.d, 0, 0, g)
        G.save_stage_b_part(self.d, 1, 0, g)
        G.clear_stage_b_parts(self.d, 0, 1)
        self.assertFalse(os.path.exists(G.stage_b_part_path(self.d, 0, 0)))
        self.assertTrue(os.path.exists(G.stage_b_part_path(self.d, 1, 0)))

    def test_part_names_are_cleared_by_the_existing_stageb_prefix(self):
        """ckpt_clear(parts, "stageb") must reach the per-rank checkpoints."""
        self.assertTrue(
            os.path.basename(G.stage_b_part_path(self.d, 0, 3)).startswith("stageb"))
```

- [ ] **Step 2: Run it to see it fail**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t3a.log \
  python -m unittest tests.test_fstat_parallel_fit.SplitBoxRangeTest tests.test_fstat_parallel_fit.StageBPartIOTest -v
cat .wtenv/t3a.log
```
Expected: FAIL — `AttributeError: module 'lisatools.sampling.fstat_gridfit' has no attribute 'split_box_range'`.

- [ ] **Step 3: Implement**

Add `import hashlib` to the stdlib imports if absent. Insert after `write_stacked_npz`:

```python
def split_box_range(a, b, n_parts):
    """Split boxes ``[a, b)`` into ``n_parts`` CONTIGUOUS, near-equal ranges.

    A pure function of ``(a, b, n_parts)`` -- that is load-bearing: a resume
    must reproduce the same rank -> range map, or a rank would find another
    rank's checkpoint under its own name (the fingerprint would reject it and
    the group would silently restart, which is safe but wastes the sweep).

    CONTIGUITY, not interleaving, is also load-bearing: the sig-het F-stat
    keeps ONE reference block resident and rebuilds it on f0 boundary
    crossings, so an f0-contiguous range divides the block builds
    proportionally, while an interleaved one would rebuild every block on
    every rank (measured 4,600 rebuilds ~ 350 s in the F-ordered-box
    incident, see run_stacked_stage_b's f0-sort comment).

    The first ``(b - a) % n_parts`` ranges get one extra box. With fewer
    boxes than parts the tail ranges come back EMPTY (zero width), which
    :func:`run_stage_b_group` short-circuits.
    """
    a, b, n = int(a), int(b), int(n_parts)
    if n <= 0:
        raise ValueError(f"n_parts must be positive, got {n_parts!r}")
    total = max(b - a, 0)
    base, rem = divmod(total, n)
    out, start = [], a
    for i in range(n):
        width = base + (1 if i < rem else 0)
        out.append((start, start + width))
        start += width
    return out


def stage_b_part_path(parts_dir, gi, rank) -> str:
    """``<parts>/stageb_g{gi}_r{rank}.npy`` -- one rank's slice of one group.

    The ``stageb`` prefix is deliberate: the existing
    ``ckpt_clear(_parts, "stageb")`` at the end of a successful stage B
    already removes the per-rank PROGRESS files by prefix, and these
    partials are cleared by :func:`clear_stage_b_parts` alongside them.
    """
    return os.path.join(parts_dir, f"stageb_g{int(gi)}_r{int(rank)}.npy")


def save_stage_b_part(parts_dir, gi, rank, grid):
    """Write one rank's finished slice as raw float64; return ``(path, n_rows, sha1)``.

    ``np.save`` through an open file object, NOT a path: given a path it
    appends ``.npy`` to whatever it is handed, which would turn the
    write-then-rename temp name into ``....npy.tmp.npy``. The rename is what
    makes the head's read of a partial atomic on a shared filesystem.
    """
    os.makedirs(parts_dir, exist_ok=True)
    arr = np.ascontiguousarray(_to_host(grid), dtype=np.float64)
    path = stage_b_part_path(parts_dir, gi, rank)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)
    os.replace(tmp, path)
    return path, int(arr.shape[0]), hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def load_stage_b_part(parts_dir, gi, rank):
    return np.load(stage_b_part_path(parts_dir, gi, rank), allow_pickle=False)


def assemble_stage_b_group(parts_dir, gi, n_parts, node_shape, *, xp,
                           sha1s=None):
    """Concatenate one group's per-rank partials in RANK (== box) order.

    Box is the slowest axis of ``node_shape``
    (:func:`run_stacked_peak_sweep`), and :func:`split_box_range` hands rank
    ``r`` a contiguous ascending range, so rank-ordered concatenation on
    axis 0 reproduces the whole-group sweep exactly. ``sha1s`` (rank ->
    digest, as the ranks reported them) is verified when given: a partial
    that changed between the reply and the read is a filesystem fault, and
    silently fitting on it would corrupt the epoch with no symptom.
    """
    parts = []
    for r in range(int(n_parts)):
        arr = np.ascontiguousarray(load_stage_b_part(parts_dir, gi, r))
        if sha1s is not None and sha1s.get(r) is not None:
            got = hashlib.sha1(arr.tobytes()).hexdigest()[:16]
            if got != sha1s[r]:
                raise RuntimeError(
                    f"stage-B partial g{gi} r{r} changed under us: reported "
                    f"sha1 {sha1s[r]}, read {got} "
                    f"({stage_b_part_path(parts_dir, gi, r)})")
        parts.append(arr)
    grid = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    if tuple(grid.shape) != tuple(node_shape):
        raise RuntimeError(
            f"stage-B group {gi}: assembled {tuple(grid.shape)} from "
            f"{n_parts} partials, expected {tuple(node_shape)}")
    return xp.asarray(grid)


def clear_stage_b_parts(parts_dir, gi, n_parts) -> None:
    """Remove one group's partials (and any leftover temp files)."""
    for r in range(int(n_parts)):
        for path in (stage_b_part_path(parts_dir, gi, r),
                     stage_b_part_path(parts_dir, gi, r) + ".tmp"):
            try:
                os.remove(path)
            except OSError:
                pass
```

- [ ] **Step 4: Run the tests**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t3b.log \
  python -m unittest tests.test_fstat_parallel_fit -v
cat .wtenv/t3b.log
```
Expected: all PASS (golden tests included).

- [ ] **Step 5: Commit**

```sh
git add src/lisatools/sampling/fstat_gridfit.py tests/test_fstat_parallel_fit.py
git commit -m "$(cat <<'EOF'
fstat_gridfit: contiguous box split + per-rank .npy partial I/O for stage B

split_box_range is a pure function of (a, b, n_parts) so a resume
reproduces the rank -> range map; contiguity keeps the sig-het reference
block builds proportional instead of rebuilding every block on every rank.
Partials go to the shared epoch _parts dir as raw float64 with a sha1, so no
grid ever travels through an MPI reply.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Public `FStatRefRowHolder` + `snapshot_ref_rows`

The transport shape for the reference row. `_FStatRefRowHolder` already exists and is already the right object (gbbands.py:926) — it is private to `_sighet_fstat_multidevice` and its row extraction is inlined there (gbbands.py:2350-2378). Make both reusable.

**Files:**
- Modify: `src/lisatools/globalfit/moves/gbbands.py`
- Test: `tests/test_fstat_ref_row_holder.py` (new)

**Interfaces:**
- Produces:
  - `class FStatRefRowHolder` (public name; `_FStatRefRowHolder = FStatRefRowHolder` alias retained for the existing internal references) — `FStatRefRowHolder(parent, device, data_row, psd_row)`, `.linear_data_arr == [data_row]`, `.linear_psd_arr == [psd_row]`, `.acs_total_entries == 1`, `len(holder) == 1`, `.gpus == [device] or None`, `.xp` delegating to the parent.
  - `snapshot_ref_rows(holder, view, intra_data, intra_noise, *, xp, device=None) -> (data_row_host, psd_row_host)` — contiguous host float arrays.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fstat_ref_row_holder.py`:

```python
"""The public F-stat reference row holder + the host row snapshot.

CPU-only, no ACA: a stand-in parent exposing exactly the attributes the
extraction reads (``linear_data_arr`` / ``linear_psd_arr`` flat buffers,
``acs_total_entries``, ``xp``, ``nchannels``, ``shape_sens``). The point is
the ROW ARITHMETIC -- one walker's residual row and inverse-PSD row pulled
out of the flat per-shard buffers -- which is what the fan-out ships.
"""

import unittest

import numpy as np

from lisatools.globalfit.moves import gbbands


class _FakeParent:
    """Minimal single-shard holder: B rows in one flat buffer per kind."""

    def __init__(self, n_rows, data_row_size, psd_row_size):
        self.acs_total_entries = int(n_rows)
        self.nchannels = 3
        self.shape_sens = (3, 3)
        self.device = None
        self.gpus = None
        rng = np.random.default_rng(3)
        self.linear_data_arr = [
            rng.normal(size=int(n_rows) * int(data_row_size))]
        self.linear_psd_arr = [
            rng.normal(size=int(n_rows) * int(psd_row_size))]
        self.psd_row_index = None

    @property
    def xp(self):
        return np


class FStatRefRowHolderTest(unittest.TestCase):
    def test_is_public(self):
        self.assertTrue(hasattr(gbbands, "FStatRefRowHolder"))
        self.assertIs(gbbands._FStatRefRowHolder, gbbands.FStatRefRowHolder)

    def test_single_slab_shape_and_delegation(self):
        parent = _FakeParent(4, 12, 36)
        holder = gbbands.FStatRefRowHolder(
            parent, None, np.zeros(12), np.zeros(36))
        self.assertEqual(len(holder), 1)
        self.assertEqual(holder.acs_total_entries, 1)
        self.assertEqual(len(holder.linear_data_arr), 1)
        self.assertIsNone(holder.gpus)
        self.assertIs(holder.xp, np)
        self.assertEqual(holder.nchannels, 3)  # delegated to the parent

    def test_device_sets_the_single_entry_gpu_list(self):
        parent = _FakeParent(2, 4, 4)
        holder = gbbands.FStatRefRowHolder(
            parent, 1, np.zeros(4), np.zeros(4))
        self.assertEqual(holder.gpus, [1])
        self.assertEqual(holder.device, 1)

    def test_underscore_attributes_are_not_delegated(self):
        parent = _FakeParent(2, 4, 4)
        holder = gbbands.FStatRefRowHolder(
            parent, None, np.zeros(4), np.zeros(4))
        with self.assertRaises(AttributeError):
            holder._not_a_real_attribute  # noqa: B018


class SnapshotRefRowsTest(unittest.TestCase):
    def test_pulls_exactly_one_walkers_rows(self):
        n_rows, drow, prow = 4, 12, 36
        parent = _FakeParent(n_rows, drow, prow)
        for row in range(n_rows):
            d, p = gbbands.snapshot_ref_rows(
                parent, parent, row, row, xp=np, device=None)
            np.testing.assert_array_equal(
                d, np.asarray(parent.linear_data_arr[0]).reshape(
                    n_rows, -1)[row])
            np.testing.assert_array_equal(
                p, np.asarray(parent.linear_psd_arr[0]).reshape(
                    n_rows, -1)[row])
            self.assertTrue(d.flags["C_CONTIGUOUS"])
            self.assertTrue(p.flags["C_CONTIGUOUS"])

    def test_the_snapshot_feeds_a_holder_that_scores_row_zero(self):
        parent = _FakeParent(3, 8, 8)
        d, p = gbbands.snapshot_ref_rows(parent, parent, 2, 2, xp=np)
        holder = gbbands.FStatRefRowHolder(parent, None, d, p)
        np.testing.assert_array_equal(
            np.asarray(holder.linear_data_arr[0]).reshape(1, -1)[0], d)
        np.testing.assert_array_equal(
            np.asarray(holder.linear_psd_arr[0]).reshape(1, -1)[0], p)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run it to see it fail**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t4a.log python -m unittest tests.test_fstat_ref_row_holder -v
cat .wtenv/t4a.log
```
Expected: FAIL — `AttributeError: module ... has no attribute 'FStatRefRowHolder'`.

- [ ] **Step 3: Rename the class and add the alias**

In `src/lisatools/globalfit/moves/gbbands.py:926`, rename `class _FStatRefRowHolder:` to `class FStatRefRowHolder:`. Update its docstring's first paragraph to drop "Private to ..." and say instead:

```
    Public because the multi-rank F-stat fit builds one on EVERY compute
    rank from the reference walker's broadcast row pair (design spec
    2026-09-16, decision 2) -- not only the in-process multi-device lanes.
```

Immediately after the class body add:

```python
#: Back-compat alias: the class was private until the multi-rank F-stat fit.
_FStatRefRowHolder = FStatRefRowHolder
```

Update the two construction sites inside `_sighet_fstat_multidevice` (~2385) and the other lane builder (~2563) to the public name, and the two docstring cross-references that name `_FStatRefRowHolder` (the `_parity_target` docstring at ~979 and ~998) to `FStatRefRowHolder`.

- [ ] **Step 4: Factor out `snapshot_ref_rows`**

Insert this module-level function just above `class _RoutedBandEngine` (or immediately after the `FStatRefRowHolder` alias, whichever keeps the file's existing ordering of module-level helpers):

```python
def snapshot_ref_rows(holder, view, intra_data, intra_noise, *, xp,
                      device=None):
    """Host COPIES of one walker's residual row and inverse-PSD row.

    Slice the walker's rows ON the owning device and host only the rows --
    ``asnumpy`` of the whole shard buffer would round-trip every walker.
    ``view`` is the shard (or the holder itself for a single-shard ACA) and
    ``intra_data`` / ``intra_noise`` are INTRA-shard row indices.

    Factored out of :meth:`_RoutedBandEngine._sighet_fstat_multidevice` so
    the multi-rank fit's owner rank ships exactly the rows the in-process
    multi-device lanes replicate -- one extraction, one set of layout
    assumptions.
    """
    n_slabs = int(view.acs_total_entries)
    dev = getattr(view, "device", None) if device is None else device
    with device_context(holder.xp, dev):
        data_row_host = np.ascontiguousarray(asnumpy(
            xp.asarray(view.linear_data_arr[0]).reshape(
                n_slabs, -1)[int(intra_data)]))
        _prow = getattr(view, "psd_row_index", None)
        if _prow is not None:
            # Shared-psd mirror: linear_psd_arr[0] is the parent's
            # per-walker FULL-BAND plane; the slot's row is the map entry
            # (only reachable for a full-band buffer in mirror mode -- the
            # production F-stat holders are the parent ACA). One psd row =
            # prod(shape_sens) x (Nf_active x Nt_active) = prod(shape_sens)
            # x (data row size / nchannels).
            _plane = xp.asarray(view.linear_psd_arr[0])
            _row = int(np.asarray(asnumpy(_prow))[int(intra_noise)])
            _per_row = (int(np.prod(holder.shape_sens))
                        * (int(data_row_host.size) // int(holder.nchannels)))
            _n_rows = int(_plane.size) // _per_row
            if _n_rows * _per_row != int(_plane.size) or _row >= _n_rows:
                raise RuntimeError(
                    "sig-het F-stat: mirror plane / row map mismatch "
                    f"(plane {int(_plane.size)} elements, per row {_per_row}, "
                    f"row {_row})")
            psd_row_host = np.ascontiguousarray(asnumpy(
                _plane.reshape(_n_rows, -1)[_row]))
        else:
            psd_row_host = np.ascontiguousarray(asnumpy(
                xp.asarray(view.linear_psd_arr[0]).reshape(
                    n_slabs, -1)[int(intra_noise)]))
    return data_row_host, psd_row_host
```

Then in `_sighet_fstat_multidevice` replace the whole inlined `n_slabs = ...` / `with device_context(...)` extraction block (gbbands.py ~2349-2378) with:

```python
        data_row_host, psd_row_host = snapshot_ref_rows(
            holder, view, intra_data, intra_noise, xp=xp)
```

Leave everything after it (the `lanes` loop) untouched.

- [ ] **Step 5: Run the tests**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t4b.log \
  python -m unittest tests.test_fstat_ref_row_holder tests.test_band_view_multi_shard tests.test_fstat_nm_lanes -v
cat .wtenv/t4b.log
```
Expected: all PASS. If `tests/test_band_view_multi_shard.py` does not exist under that name, run `ls tests | grep -i shard` and use the module(s) it prints instead (max 6 modules per process).

- [ ] **Step 6: Commit**

```sh
git add src/lisatools/globalfit/moves/gbbands.py tests/test_fstat_ref_row_holder.py
git commit -m "$(cat <<'EOF'
gbbands: FStatRefRowHolder made public + snapshot_ref_rows factored out

The multi-rank F-stat fit builds a row holder on EVERY compute rank from the
reference walker's broadcast row pair, so both the holder and the on-device
row extraction (which the in-process multi-device lanes already used) become
shared entry points instead of private inlined code. _FStatRefRowHolder
stays as an alias.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: The global reference walker

Fix the port defect: `_fstat_reference_walker` (gbspecialstretch.py:4192) takes `argmax` over the HEAD'S LOCAL BLOCK. Under the walker-block layout that is not the ensemble maximum, and the returned index is used as an ACA ROW index — a global index `>= B` would be out of range.

**Files:**
- Modify: `src/lisatools/globalfit/communication/ranks.py` (`WalkerBlockLayout.owner_of`)
- Modify: `src/lisatools/globalfit/moves/gbspecialstretch.py` (`_fstat_global_reference`)
- Test: `tests/test_fstat_parallel_fit.py`

**Interfaces:**
- Consumes: `WalkerFanout.gather_likelihood(acs) -> np.ndarray (nwalkers,)` in GLOBAL walker order (fanout.py:145; `acs.likelihood(complex=False)`, the same numbers today's `argmax` sees — `inner_product`'s `complex` default is False).
- Produces:
  - `WalkerBlockLayout.owner_of(w) -> tuple[int, int]` — `(owning compute rank, local row index)`.
  - `GBSpecialBase._fstat_global_reference(self, model) -> tuple[int, int, int, np.ndarray]` — `(w_global, owner_rank, local_index, lls)`; `lls` is the per-walker likelihood vector (or an all-NaN vector of length 1 on the fallback path).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fstat_parallel_fit.py`:

```python
class OwnerOfTest(unittest.TestCase):
    def _layout(self, nwalkers, n_compute):
        from lisatools.globalfit.communication import ranks as R
        from lisatools.globalfit.communication.fakecomm import FakeWorld

        world = FakeWorld(n_compute + 1)
        out = world.run(lambda r, comm: R.build_layout(
            comm, nwalkers, list(range(n_compute))))
        return out[0]

    def test_maps_every_global_walker_to_its_rank_and_local_row(self):
        layout = self._layout(8, 2)
        block = layout.block
        for w in range(8):
            rank, local = layout.owner_of(w)
            w0, w1 = layout.block_of(rank)
            self.assertTrue(w0 <= w < w1)
            self.assertEqual(local, w - w0)
            self.assertTrue(0 <= local < block)

    def test_single_compute_rank_is_the_identity(self):
        layout = self._layout(6, 1)
        for w in range(6):
            self.assertEqual(layout.owner_of(w), (layout.head_rank, w))

    def test_out_of_range_raises(self):
        layout = self._layout(4, 2)
        with self.assertRaises(ValueError):
            layout.owner_of(4)
        with self.assertRaises(ValueError):
            layout.owner_of(-1)
```

- [ ] **Step 2: Run it to see it fail**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t5a.log \
  python -m unittest tests.test_fstat_parallel_fit.OwnerOfTest -v
cat .wtenv/t5a.log
```
Expected: FAIL — `AttributeError: 'WalkerBlockLayout' object has no attribute 'owner_of'`.

If `build_layout`'s signature differs from `build_layout(comm, nwalkers, gpu_pool)`, read `src/lisatools/globalfit/communication/ranks.py:194` and adapt the helper — do NOT change `build_layout`.

- [ ] **Step 3: Add `owner_of` to the layout**

In `src/lisatools/globalfit/communication/ranks.py`, inside `class WalkerBlockLayout`, immediately after `block_of` (~:88):

```python
    def owner_of(self, w) -> tuple:
        """Global walker ``w`` -> ``(owning compute rank, local row index)``.

        The inverse of :meth:`block_of`. Needed wherever a head-side GLOBAL
        walker index has to become an ACA ROW index: rows are per-rank, so a
        global index is out of range on every rank but its owner (the
        multi-rank F-stat fit's reference walker is exactly that case).
        """
        w = int(w)
        if not (0 <= w < int(self.nwalkers)):
            raise ValueError(
                f"walker {w} is outside [0, {int(self.nwalkers)})")
        rank = self.compute_ranks[w // int(self.block)]
        w0, _w1 = self.block_of(rank)
        return int(rank), w - int(w0)
```

- [ ] **Step 4: Add `_fstat_global_reference` to the move**

In `src/lisatools/globalfit/moves/gbspecialstretch.py`, immediately after `_fstat_reference_walker` (it ends ~:4218), add:

```python
    def _fstat_global_reference(self, model):
        """``(w_global, owner_rank, local_index, lls)`` for the F-stat fit.

        The GLOBAL max-likelihood walker, not this rank's local one. Under
        the walker-block layout ``_fstat_reference_walker`` ranks only the
        HEAD'S OWN block, so the epoch would be fitted against the best of B
        walkers instead of the best of N -- and the index it returns is used
        downstream as an ACA ROW index, which a global index >= B would
        blow out. Every walker's likelihood comes back through the existing
        ``WalkerFanout.gather_likelihood`` (the ``LIKELIHOOD_OP`` builtin;
        ``acs.likelihood(complex=False)``, the same numbers the local argmax
        sees), and the layout converts the winner into the owning rank plus
        the row index ON that rank.

        With ONE compute rank this is a direct call that returns
        ``(w, head_rank, w, lls)`` -- the same walker
        ``_fstat_reference_walker`` picks today, with the same local index.
        """
        fanout = getattr(self, "fanout", None)
        if fanout is None:
            w = self._fstat_reference_walker(model)
            return int(w), 0, int(w), np.full(1, np.nan)
        try:
            lls = np.asarray(_to_numpy(fanout.gather_likelihood(
                model.analysis_container_arr)), dtype=float)
            w_global = int(np.argmax(lls))
        except Exception as exc:
            # Never silent: a broken ranking here quietly pins every F-stat
            # reference to walker 0 for the whole run.
            logger.warning(
                "%s: could not rank walkers for the F-stat reference (%r); "
                "falling back to walker 0.", self.name, exc)
            return 0, fanout.layout.head_rank, 0, np.full(1, np.nan)
        owner, local = fanout.layout.owner_of(w_global)
        return w_global, int(owner), int(local), lls
```

- [ ] **Step 5: Run the tests**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t5b.log \
  python -m unittest tests.test_fstat_parallel_fit tests.test_rank_layout tests.test_gb_rank_plumbing -v
cat .wtenv/t5b.log
```
Expected: all PASS. If `tests/test_rank_layout.py` is named differently, `ls tests | grep -i layout`.

- [ ] **Step 6: Commit**

```sh
git add src/lisatools/globalfit/communication/ranks.py src/lisatools/globalfit/moves/gbspecialstretch.py tests/test_fstat_parallel_fit.py
git commit -m "$(cat <<'EOF'
F-stat fit: the reference walker becomes the GLOBAL argmax, with its owner

WalkerBlockLayout.owner_of inverts block_of (global walker -> owning rank +
ACA row), and GBSpecialBase._fstat_global_reference ranks every walker
through WalkerFanout.gather_likelihood instead of the head's local block.
Under the walker-block port the local argmax fitted the epoch against the
best of B, and its index was then used as an ACA row index that a global
winner would have blown out.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `gb_fstat_ref_row` — the symmetric row broadcast

The reference row is replicated once, by a symmetric fan-out op. The OWNER rank opens the GB-free window on its LOCAL index, snapshots the rows, restores its residual, then every compute rank `Bcast`s the pair and wraps it in an `FStatRefRowHolder`.

Why a collective inside a command body is legal: the fan-out comm IS `ComputeService`'s comm, it includes the head, and every compute rank reaches the body exactly once per command (head at fanout.py:206, workers at fanout.py:330); precedent `allgather_walker_vector` (fanout.py:288).

**Files:**
- Modify: `src/lisatools/globalfit/communication/fakecomm.py` (`Bcast`)
- Modify: `src/lisatools/globalfit/moves/gbspecialstretch.py`
- Test: `tests/test_fstat_parallel_fit.py`

**Interfaces:**
- Consumes: Task 4's `gbbands.FStatRefRowHolder`, `gbbands.snapshot_ref_rows`; Task 5's `_fstat_global_reference`; `self._fanout_cmd(op, per_rank_payload, model)` (gbspecialstretch.py:18189) which returns `(replies, token)` and restores `nwalkers`/`ntemps`/`_prop_timer` in a `finally`.
- Produces:
  - `FakeComm.Bcast(buf, root=0) -> None` — fills `buf` in place on non-root ranks.
  - `GB_OPS` gains `"gb_fstat_ref_row"`.
  - `GBSpecialBase._bind_rank_acs(self, model)` — the three device-binding calls, factored out of `_enter_rank_block`.
  - `GBSpecialBase._fstat_ref_row_fanout(self, model, branches, w_global, owner_rank, local_index) -> None` — head-side; every rank ends holding `self._fstat_ref_holder`.
  - `GBSpecialBase._gb_serve_fstat_ref_row(self, payload, clock, model) -> dict` — reply `{"rank", "is_owner", "n_live", "data_bytes", "psd_bytes"}`.
  - Rank/head session attributes: `self._fstat_ref_holder` (an `FStatRefRowHolder` or `None`), `self._fstat_ref_walker` (the global index), `self._fstat_ref_call` (the cached `call_fstat` closure, set lazily in Task 8).

- [ ] **Step 1: Write the failing test for `FakeComm.Bcast`**

Append to `tests/test_fstat_parallel_fit.py`:

```python
class FakeCommBcastTest(unittest.TestCase):
    def test_buffer_broadcast_fills_every_rank_in_place(self):
        from lisatools.globalfit.communication.fakecomm import FakeWorld

        world = FakeWorld(3)

        def body(rank, comm):
            buf = np.zeros(4, dtype=np.float64)
            if rank == 1:
                buf[:] = [1.5, 2.5, 3.5, 4.5]
            comm.Bcast(buf, root=1)
            return buf.copy()

        out = world.run(body)
        for rank in range(3):
            np.testing.assert_array_equal(out[rank], [1.5, 2.5, 3.5, 4.5])

    def test_dtype_and_shape_are_preserved(self):
        from lisatools.globalfit.communication.fakecomm import FakeWorld

        world = FakeWorld(2)

        def body(rank, comm):
            buf = np.zeros((2, 3), dtype=np.float64)
            if rank == 0:
                buf[:] = np.arange(6, dtype=np.float64).reshape(2, 3)
            comm.Bcast(buf, root=0)
            return buf.copy()

        out = world.run(body)
        np.testing.assert_array_equal(
            out[1], np.arange(6, dtype=np.float64).reshape(2, 3))
```

- [ ] **Step 2: Run it to see it fail**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t6a.log \
  python -m unittest tests.test_fstat_parallel_fit.FakeCommBcastTest -v
cat .wtenv/t6a.log
```
Expected: FAIL — `AttributeError: 'FakeComm' object has no attribute 'Bcast'`.

- [ ] **Step 3: Implement `FakeComm.Bcast`**

In `src/lisatools/globalfit/communication/fakecomm.py`, add `import numpy as np` to the imports (after `import traceback`, keeping isort's stdlib/third-party split), and add to `FakeComm` immediately after `bcast` (~:137):

```python
    def Bcast(self, buf, root=0):
        """Buffer broadcast (uppercase MPI form): fills ``buf`` IN PLACE.

        mpi4py's ``Bcast`` moves contiguous buffers without pickling, which
        is what the F-stat reference row pair (tens of MB) uses. The fake
        moves the bytes through the existing slot exchange and writes them
        into each non-root rank's own array, so a caller that allocates its
        own receive buffer -- the real usage -- is exercised faithfully.
        """
        arr = buf[0] if isinstance(buf, (list, tuple)) else buf
        arr = np.asarray(arr)
        payload = self._exchange(
            arr.tobytes() if self._rank == int(root) else None)[int(root)]
        if self._rank != int(root):
            flat = arr.reshape(-1)
            got = np.frombuffer(payload, dtype=flat.dtype)
            if got.size != flat.size:
                raise ValueError(
                    f"Bcast buffer size mismatch: root sent {got.size} "
                    f"elements, this rank's buffer holds {flat.size}")
            flat[:] = got
        return None
```

`arr.reshape(-1)` on a C-contiguous array returns a VIEW, so the in-place write reaches the caller's buffer. Every array this broadcasts is allocated with `np.empty(..., dtype=np.float64)` on the receiving side, so contiguity holds.

- [ ] **Step 4: Write the failing test for the op**

Append to `tests/test_fstat_parallel_fit.py`:

```python
class RefRowOpTest(unittest.TestCase):
    """``gb_fstat_ref_row`` replicates the owner's rows to every rank."""

    def test_op_is_registered(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        self.assertIn("gb_fstat_ref_row", gbs.GB_OPS)
        self.assertIn("gb_fstat_stage_b", gbs.GB_OPS)

    def test_unknown_op_still_raises(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move.name = "gb_test"
        with self.assertRaises(ValueError):
            move.gf_serve("not_an_op", None, {}, None)
```

The end-to-end replication assertion lives in Task 10's FakeWorld gate, where a real fan-out exists.

- [ ] **Step 5: Register the ops and factor `_bind_rank_acs`**

In `gbspecialstretch.py`:

(a) at `:1931`, extend the tuple and document the two new members:

```python
#: The GB fan-out ops. The first three are the per-propose session (opened
#: by ``gb_run_proposal``); the last two are the F-stat epoch fit, issued
#: from the head's ``setup()`` BEFORE any session exists -- harmless,
#: because the session token is captured from the opening
#: ``gb_run_proposal`` and ``call_index`` is counted per ``(move, op)``.
GB_OPS = ("gb_run_proposal", "gb_run_tempering", "gb_finish",
          "gb_fstat_ref_row", "gb_fstat_stage_b")
```

(b) in `_enter_rank_block` (`:17280-17286`), replace the three binding calls with a call to a new factored method, and add the method just above `_enter_rank_block`:

```python
    def _bind_rank_acs(self, model):
        """Pin this rank's device and bind the ACA that arrived with the model.

        Run-time source of truth is the ACA that arrives with the model (one
        B-row ACA per rank): refresh the domain quantities and re-bind the
        parent engine if this ACA differs from the bound one. Shared by the
        per-propose session commands and the F-stat epoch-fit commands.
        """
        acs = model.analysis_container_arr
        pin_main_device(self.xp, acs.gpus)
        self._configure_domain(acs)
        self._bind_parent_acs(acs)
        return acs
```

and in `_enter_rank_block`:

```python
        acs = self._bind_rank_acs(model)
```

(c) extend `gf_serve` (`:17580`):

```python
    def gf_serve(self, op, payload, clock, model):
        """Serve one GB fan-out command on a compute rank (see :data:`GB_OPS`)."""
        if op == "gb_run_proposal":
            return self._gb_serve_run_proposal(payload, clock, model)
        if op == "gb_run_tempering":
            return self._gb_serve_run_tempering(payload, clock, model)
        if op == "gb_finish":
            return self._gb_serve_finish(payload, clock, model)
        if op == "gb_fstat_ref_row":
            return self._gb_serve_fstat_ref_row(payload, clock, model)
        if op == "gb_fstat_stage_b":
            return self._gb_serve_fstat_stage_b(payload, clock, model)
        raise ValueError(
            f"move {self.name!r} serves only {GB_OPS}, got {op!r}"
        )
```

(Task 7 adds `_gb_serve_fstat_stage_b`; add a placeholder in this task only if the module would not import otherwise — it will, because the name is resolved at call time.)

- [ ] **Step 6: Implement the op body and the head-side driver**

Add these methods to `GBSpecialBase`, immediately before `_run_fstat_fit` (`:20601`):

```python
    # ---- the F-stat epoch fit's two fan-out commands -----------------------

    def _fstat_ref_row_payload(self, w_global, owner_rank, local_index,
                               branches_present):
        """The command payload: the same dict for every rank (symmetric op)."""
        return {
            "walker_ref": int(w_global),
            "owner_rank": int(owner_rank),
            "local_index": int(local_index),
            "gb_free": bool(branches_present),
        }

    def _gb_serve_fstat_ref_row(self, payload, clock, model):
        """Replicate the reference walker's residual + inverse-PSD rows.

        SYMMETRIC: every compute rank -- the head included -- enters this
        body exactly once per command, so the ``Bcast`` inside it is legal
        on the fan-out communicator (same precedent as
        ``WalkerFanout.allgather_walker_vector``).

        The OWNER rank opens the GB-free window on its LOCAL row (today's
        ``_gb_free_residual`` semantics), snapshots the two rows to host and
        then RESTORES its residual -- so unlike the serial fit, no rank's
        live residual stays mutated for the duration of the sweep. Every
        rank then wraps the pair in an :class:`FStatRefRowHolder` scored
        with ``data_index = noise_index = 0``.

        The row sizes come from the owner first (a tiny pickled reply is not
        available here -- this is a collective), so they are broadcast as a
        2-element int64 header before the payload buffers.
        """
        from .gbbands import FStatRefRowHolder, snapshot_ref_rows

        acs = self._bind_rank_acs(model)
        payload = payload or {}
        w_global = int(payload["walker_ref"])
        owner_rank = int(payload["owner_rank"])
        local_index = int(payload["local_index"])
        # ``getattr``, not ``self.fanout``: ``_propose_legacy`` and the
        # single-process ``fit.sample()`` path build moves that never see a
        # fan-out at all, and this body is the one they take.
        fanout = getattr(self, "fanout", None)
        is_owner = (fanout is None) or (int(fanout.rank) == owner_rank)

        data_row = psd_row = None
        n_live = -1
        if is_owner:
            branches = getattr(self, "_fstat_ref_branches", None)
            with self._gb_free_residual(model, branches, local_index):
                n_live = int(getattr(self, "_gb_free_n_live", -1))
                data_row, psd_row = snapshot_ref_rows(
                    acs, acs, local_index, local_index, xp=self.xp,
                    device=(acs.gpus[0] if getattr(acs, "gpus", None) else None))

        if fanout is not None and not fanout.single:
            comm = fanout.comm
            root = fanout.layout.fanout_rank(owner_rank)
            header = np.zeros(2, dtype=np.int64)
            if is_owner:
                header[:] = (data_row.size, psd_row.size)
            comm.Bcast(header, root=root)
            if not is_owner:
                data_row = np.empty(int(header[0]), dtype=np.float64)
                psd_row = np.empty(int(header[1]), dtype=np.float64)
            else:
                data_row = np.ascontiguousarray(data_row, dtype=np.float64)
                psd_row = np.ascontiguousarray(psd_row, dtype=np.float64)
            comm.Bcast(data_row, root=root)
            comm.Bcast(psd_row, root=root)

        self._fstat_ref_walker = w_global
        self._fstat_ref_call = None       # rebuilt lazily against the holder
        self._fstat_ref_holder = FStatRefRowHolder(
            acs,
            (acs.gpus[0] if getattr(acs, "gpus", None) else None),
            self.xp.asarray(data_row),
            self.xp.asarray(psd_row),
        )
        return {
            "rank": int(getattr(fanout, "rank", 0)),
            "is_owner": bool(is_owner),
            "n_live": int(n_live),
            "data_bytes": int(np.asarray(data_row).nbytes),
            "psd_bytes": int(np.asarray(psd_row).nbytes),
        }

    def _fstat_ref_row_fanout(self, model, branches, w_global, owner_rank,
                              local_index):
        """HEAD: issue ``gb_fstat_ref_row``; every rank ends holding a holder.

        ``branches`` reaches the owner through ``_fstat_ref_branches`` rather
        than the payload: it is the live branch dict the GB-free window needs
        to build its ``BandSorter``, and it is not shippable -- the owner is
        the only rank that opens the window, and it already has its own
        slice. Cleared in the ``finally`` so a later command can never reuse
        a stale one.

        With NO fan-out at all (``_propose_legacy``, ``fit.sample()``) this
        is a direct call to the body -- no communicator is touched and the
        holder is built from this process's own rows.
        """
        self._fstat_ref_branches = branches
        payload = self._fstat_ref_row_payload(
            w_global, owner_rank, local_index, branches is not None)
        fanout = getattr(self, "fanout", None)
        if fanout is None:
            try:
                result = self._gb_serve_fstat_ref_row(payload, {}, model)
            finally:
                self._fstat_ref_branches = None
            logger.info(
                "%s: F-stat reference row built in-process from walker %d "
                "(no fan-out); %d cold GB signal(s) restored for the "
                "snapshot; %.1f MB residual + %.1f MB invC",
                self.name, int(w_global), int(result.get("n_live", -1)),
                result.get("data_bytes", 0) / 1e6,
                result.get("psd_bytes", 0) / 1e6)
            return {0: {"result": result}}
        try:
            replies, _token = self._fanout_cmd(
                "gb_fstat_ref_row", lambda rank, w0, w1: payload, model)
        finally:
            self._fstat_ref_branches = None
        owner_reply = next(
            (r["result"] for r in replies.values()
             if r["result"] and r["result"].get("is_owner")), None)
        n_live = -1 if owner_reply is None else int(owner_reply.get("n_live", -1))
        sizes = [(r["result"] or {}).get("data_bytes") for r in replies.values()]
        logger.info(
            "%s: F-stat reference row replicated to %d rank(s) from walker "
            "%d (rank %d, local row %d); %d cold GB signal(s) restored for "
            "the snapshot; %.1f MB residual + %.1f MB invC per rank",
            self.name, len(replies), int(w_global), int(owner_rank),
            int(local_index), n_live,
            (sizes[0] or 0) / 1e6,
            ((owner_reply or {}).get("psd_bytes", 0)) / 1e6)
        return replies

    def _fstat_release_ref_row(self):
        """Drop the replicated row holder (head + ranks) after the fit."""
        self._fstat_ref_holder = None
        self._fstat_ref_call = None
        self._fstat_ref_walker = None
```

`_gb_free_residual` must publish the live-source count so the log line above is honest. In `_gb_free_residual` (`:20529`), after `n_live = int(sorter.get_subset_bool(**sel).sum())`, add `self._gb_free_n_live = n_live`, and in the early-return branch (`yield; return`) add `self._gb_free_n_live = 0` before the `yield`.

Add the three session attributes to the class-level defaults next to the other `_fstat_*` attributes so a fresh move (and `test_gb_rank_session`'s skeleton) never `AttributeError`s:

```python
    #: The multi-rank F-stat fit's replicated reference row (spec 2026-09-16
    #: decision 2): set by ``gb_fstat_ref_row`` on EVERY compute rank,
    #: released at the end of the fit.
    _fstat_ref_holder = None
    _fstat_ref_call = None
    _fstat_ref_walker = None
    _fstat_ref_branches = None
    _gb_free_n_live = -1
```

- [ ] **Step 7: Run the tests**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t6b.log \
  python -m unittest tests.test_fstat_parallel_fit tests.test_fakecomm tests.test_fanout_fakecomm tests.test_gb_rank_session -v
cat .wtenv/t6b.log
```
Expected: all PASS.

- [ ] **Step 8: Commit**

```sh
git add src/lisatools/globalfit/communication/fakecomm.py src/lisatools/globalfit/moves/gbspecialstretch.py tests/test_fstat_parallel_fit.py
git commit -m "$(cat <<'EOF'
gb_fstat_ref_row: the reference row is replicated once by a symmetric fan-out op

The owner rank opens the GB-free window on its LOCAL row, snapshots the
walker's residual and inverse-PSD rows, restores its residual, and Bcasts
the pair on the fan-out comm; every rank wraps it in an FStatRefRowHolder
scored at data_index=0. No rank's live residual stays mutated for the
duration of the fit. FakeComm gains the uppercase buffer Bcast so the
laptop gate exercises the real path.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `gb_fstat_stage_b` — the per-group box-range fan-out

One command per Mc group. Each rank sweeps its contiguous box range with the EXISTING `run_stacked_peak_sweep` (through `run_stage_b_group`), checkpoints under its own name in the SHARED `_parts` dir, saves a `.npy` partial, and replies with metadata only. The head assembles.

**Files:**
- Modify: `src/lisatools/globalfit/moves/gbspecialstretch.py`
- Test: `tests/test_fstat_parallel_fit.py` (Task 10 carries the end-to-end gate; this task's test is the payload/slice contract)

**Interfaces:**
- Consumes: Task 2's `StageBGroupSpec`, `run_stage_b_group`; Task 3's `split_box_range`, `save_stage_b_part`, `assemble_stage_b_group`, `clear_stage_b_parts`; Task 6's `self._fstat_ref_holder`, `_fanout_cmd`.
- Produces:
  - `GBSpecialBase._fstat_stage_b_payload(self, spec, rank_index, a, b) -> dict`
  - `GBSpecialBase._gb_serve_fstat_stage_b(self, payload, clock, model) -> dict` — `{"gi", "a", "b", "n_rows", "sha1", "wall_s", "rank"}`
  - `GBSpecialBase._fstat_stage_b_runner(self, model) -> callable(spec, call_fstat, *, xp) -> grid`
  - `GBSpecialBase._fstat_holder_call(self, model) -> callable` — the cached holder-scored `call_fstat` (used by the head, the ranks and stage A)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fstat_parallel_fit.py`:

```python
class StageBPayloadTest(unittest.TestCase):
    """The shipped payload carries host arrays and reconstructs the spec."""

    def _spec(self):
        return G.StageBGroupSpec(
            gi=2, n_groups=3, a=10, b=18,
            f0_los=np.linspace(6.0, 6.7, 8),
            f0_dxs=np.full(8, 1e-3),
            mc_ax=np.linspace(0.01, 1.0, 3),
            alpha_ax=np.linspace(0.0, 2 * np.pi, 2),
            sd_ax=np.linspace(-1.0, 1.0, 2),
            node_shape=(8, 5, 3, 2, 2),
            ckpt_name="stageb_g2", parts_dir="/tmp/parts",
            fingerprint_extra="|epoch=3|gbfree=1",
            fdot_axis=False, c_t=0.0)

    def test_payload_round_trips_to_an_equivalent_sub_spec(self):
        import pickle

        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        spec = self._spec()
        payload = move._fstat_stage_b_payload(spec, 1, 13, 16)
        payload = pickle.loads(pickle.dumps(payload))   # the wire does this
        got = gbs.GBSpecialBase._fstat_stage_b_spec(payload)
        self.assertEqual(got.gi, 2)
        self.assertEqual((got.a, got.b), (13, 16))
        self.assertEqual(got.node_shape, (3, 5, 3, 2, 2))
        self.assertEqual(got.ckpt_name, "stageb_g2_r1")
        np.testing.assert_array_equal(got.f0_los, spec.f0_los[3:6])
        np.testing.assert_array_equal(got.f0_dxs, spec.f0_dxs[3:6])
        np.testing.assert_array_equal(got.mc_ax, spec.mc_ax)
        np.testing.assert_array_equal(got.alpha_ax, spec.alpha_ax)
        np.testing.assert_array_equal(got.sd_ax, spec.sd_ax)
        self.assertEqual(got.fingerprint_extra, spec.fingerprint_extra)
        self.assertEqual(got.parts_dir, spec.parts_dir)

    def test_payload_holds_no_device_arrays(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        payload = move._fstat_stage_b_payload(self._spec(), 0, 10, 13)
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                self.assertIs(type(value), np.ndarray, key)

    def test_empty_range_is_a_legal_payload(self):
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        payload = move._fstat_stage_b_payload(self._spec(), 3, 18, 18)
        got = gbs.GBSpecialBase._fstat_stage_b_spec(payload)
        self.assertEqual(got.n_boxes, 0)
        self.assertEqual(got.node_shape[0], 0)
```

- [ ] **Step 2: Run it to see it fail**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t7a.log \
  python -m unittest tests.test_fstat_parallel_fit.StageBPayloadTest -v
cat .wtenv/t7a.log
```
Expected: FAIL — `AttributeError: ... has no attribute '_fstat_stage_b_payload'`.

- [ ] **Step 3: Implement the payload, the spec reconstruction and the holder call**

Add to `GBSpecialBase`, immediately after `_fstat_release_ref_row` from Task 6:

```python
    def _fstat_holder_call(self, model):
        """The holder-scored ``call_fstat``, built ONCE per fit on this rank.

        Cached because the sig-het scorer is STATEFUL: its bucketed
        reference blocks are built lazily on f0 crossings and stashed on the
        comp, so rebuilding the closure per group would throw that cache
        away between groups -- exactly the rebuild pathology the f0-sorted
        box order exists to avoid.
        """
        if self._fstat_ref_call is None:
            self._fstat_ref_call = self._fstat_call(
                model, 0, holder=self._fstat_ref_holder)
        return self._fstat_ref_call

    @staticmethod
    def _fstat_stage_b_payload(spec, rank_index, a, b):
        """One rank's slice of one group, as host arrays (the wire pickles it).

        The AXES are shipped whole (they define what a row means, and they
        are kilobytes); only ``f0_los`` / ``f0_dxs`` / the leading axis of
        ``node_shape`` are sliced. ``ckpt_name`` is per rank so two ranks
        never share a progress file -- their fingerprints already differ,
        because the fingerprint hashes the SLICED inputs plus ``node_shape``.
        """
        sub = spec.sub_range(a, b, ckpt_name=f"{spec.ckpt_name}_r{int(rank_index)}")
        return {
            "gi": int(sub.gi),
            "n_groups": int(sub.n_groups),
            "a": int(sub.a),
            "b": int(sub.b),
            "rank_index": int(rank_index),
            "f0_los": np.ascontiguousarray(np.asarray(sub.f0_los, dtype=float)),
            "f0_dxs": np.ascontiguousarray(np.asarray(sub.f0_dxs, dtype=float)),
            "mc_ax": np.ascontiguousarray(np.asarray(sub.mc_ax, dtype=float)),
            "alpha_ax": np.ascontiguousarray(np.asarray(sub.alpha_ax, dtype=float)),
            "sd_ax": np.ascontiguousarray(np.asarray(sub.sd_ax, dtype=float)),
            "node_shape": tuple(int(v) for v in sub.node_shape),
            "ckpt_name": sub.ckpt_name,
            "parts_dir": sub.parts_dir,
            "fingerprint_extra": sub.fingerprint_extra,
            "fdot_axis": bool(sub.fdot_axis),
            "c_t": float(sub.c_t),
        }

    @staticmethod
    def _fstat_stage_b_spec(payload):
        """Rebuild the :class:`StageBGroupSpec` a payload describes."""
        from lisatools.sampling.fstat_gridfit import StageBGroupSpec

        return StageBGroupSpec(
            gi=int(payload["gi"]), n_groups=int(payload["n_groups"]),
            a=int(payload["a"]), b=int(payload["b"]),
            f0_los=np.asarray(payload["f0_los"], dtype=float),
            f0_dxs=np.asarray(payload["f0_dxs"], dtype=float),
            mc_ax=np.asarray(payload["mc_ax"], dtype=float),
            alpha_ax=np.asarray(payload["alpha_ax"], dtype=float),
            sd_ax=np.asarray(payload["sd_ax"], dtype=float),
            node_shape=tuple(int(v) for v in payload["node_shape"]),
            ckpt_name=payload["ckpt_name"], parts_dir=payload["parts_dir"],
            fingerprint_extra=payload["fingerprint_extra"],
            fdot_axis=bool(payload["fdot_axis"]), c_t=float(payload["c_t"]))

    def _gb_serve_fstat_stage_b(self, payload, clock, model):
        """Sweep this rank's contiguous box range of one stage-B group.

        Runs the SAME :func:`run_stage_b_group` the serial fit runs, on the
        sliced inputs, scored through the replicated reference row -- so a
        split can only differ from the serial sweep by ``FSTAT_BATCH``
        grouping, which is row-independent. The finished slice goes to the
        shared ``_parts`` directory as raw float64; the REPLY carries only
        metadata, because a whole group grid is 163-653 MB at production
        scale against mpi4py's ~2 GiB pickle cap.
        """
        from lisatools.sampling.fstat_gridfit import (
            run_stage_b_group,
            save_stage_b_part,
        )

        self._bind_rank_acs(model)
        if self._fstat_ref_holder is None:
            raise RuntimeError(
                f"{self.name}: gb_fstat_stage_b arrived before "
                "gb_fstat_ref_row -- no replicated reference row on this rank")
        spec = self._fstat_stage_b_spec(payload)
        rank_index = int(payload["rank_index"])
        t0 = time.perf_counter()
        grid = run_stage_b_group(spec, self._fstat_holder_call(model), xp=self.xp)
        _path, n_rows, sha = save_stage_b_part(
            spec.parts_dir, spec.gi, rank_index, grid)
        del grid
        self.mempool.free_all_blocks()
        wall = time.perf_counter() - t0
        logger.info(
            "%s: [FSTAT_STAGEB] g%d r%d boxes [%d, %d) (%d) in %.1fs",
            self.name, spec.gi, rank_index, spec.a, spec.b, n_rows, wall)
        return {"gi": int(spec.gi), "a": int(spec.a), "b": int(spec.b),
                "rank_index": rank_index, "n_rows": int(n_rows),
                "sha1": sha, "wall_s": float(wall)}
```

`time` is already imported at the top of `gbspecialstretch.py`; `self.mempool` is the move's existing cupy pool handle (used in `_propose_orchestrated`) — if `mempool.free_all_blocks()` is not available on the CPU backend, guard it the way the surrounding code does (`self.mempool.free_all_blocks()` is already called unguarded at the top of `_propose_orchestrated`, so it is safe).

- [ ] **Step 4: Implement the head-side runner**

Add immediately after `_gb_serve_fstat_stage_b`:

```python
    def _fstat_stage_b_runner(self, model):
        """HEAD: a ``sweep_runner`` that fans ONE group out over the ranks.

        ``run_stacked_stage_b`` does everything else -- the host prep, the
        f0 sort, the Mc grouping, the proposal build and the npz write --
        so the parallel path can differ from the serial one only in HOW each
        group's grid is produced. Ranges are contiguous and equal-count
        (:func:`split_box_range`), a pure function of ``(g_edges,
        n_compute)``, so a resume reproduces the same map; box is the
        slowest axis, so concatenating the partials in rank order
        reproduces the whole-group sweep exactly.
        """
        from lisatools.sampling.fstat_gridfit import (
            assemble_stage_b_group,
            clear_stage_b_parts,
            split_box_range,
        )

        layout = self.fanout.layout
        n_parts = int(layout.n_compute)

        def runner(spec, call_fstat, *, xp):
            ranges = split_box_range(spec.a, spec.b, n_parts)
            by_rank = {r: ranges[layout.fanout_rank(r)]
                       for r in layout.compute_ranks}
            t0 = time.perf_counter()
            replies, _token = self._fanout_cmd(
                "gb_fstat_stage_b",
                lambda rank, w0, w1: self._fstat_stage_b_payload(
                    spec, layout.fanout_rank(rank), *by_rank[rank]),
                model,
            )
            results = {}
            for rank, rep in replies.items():
                res = rep["result"] or {}
                results[int(res["rank_index"])] = res
            sha1s = {i: res.get("sha1") for i, res in results.items()}
            grid = assemble_stage_b_group(
                spec.parts_dir, spec.gi, n_parts, spec.node_shape, xp=xp,
                sha1s=sha1s)
            clear_stage_b_parts(spec.parts_dir, spec.gi, n_parts)
            walls = [float(res.get("wall_s", 0.0)) for res in results.values()]
            logger.info(
                "%s: [FSTAT_STAGEB] group %d/%d over %d rank(s): boxes %s | "
                "wall min %.1fs max %.1fs (imbalance %.0f%%) | assembled "
                "%.1fs total",
                self.name, spec.gi + 1, spec.n_groups, n_parts,
                [b - a for a, b in ranges], min(walls), max(walls),
                100.0 * (max(walls) - min(walls)) / max(max(walls), 1e-9),
                time.perf_counter() - t0)
            return grid

        return runner
```

- [ ] **Step 5: Run the tests**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t7b.log \
  python -m unittest tests.test_fstat_parallel_fit tests.test_gb_rank_session -v
cat .wtenv/t7b.log
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```sh
git add src/lisatools/globalfit/moves/gbspecialstretch.py tests/test_fstat_parallel_fit.py
git commit -m "$(cat <<'EOF'
gb_fstat_stage_b: one command per Mc group, split by contiguous box range

Each rank sweeps its equal-count contiguous range through the existing
run_stage_b_group, checkpoints under stageb_g{gi}_r{r} in the shared _parts
dir and writes a raw float64 .npy partial; the reply carries only
(gi, a, b, n_rows, sha1, wall_s). The head plugs the fan-out in as
run_stacked_stage_b's sweep_runner, so the prep, the grouping, the proposal
build and the npz format are literally the serial code.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Wire `_run_fstat_fit` onto the replicated row and the parallel stage B

**Files:**
- Modify: `src/lisatools/globalfit/moves/gbspecialstretch.py` (`_fstat_call` `:20476`, `_fstat_NM` `:6648`, `_run_fstat_fit` `:20601`, `run_fstat_grid_fit` call-through)
- Modify: `src/lisatools/sampling/fstat_gridfit.py` (`run_fstat_grid_fit` gains `sweep_runner=None` pass-through)
- Test: `tests/test_fstat_parallel_fit.py`

**Interfaces:**
- Consumes: Tasks 5, 6, 7.
- Produces:
  - `_fstat_call(self, model, walker_ref, *, holder=None)` — with `holder`, scores it at `data_index=noise_index=0`.
  - `_fstat_NM(self, model, params_phys, walker_ref, *, holder=None)` — same.
  - `run_fstat_grid_fit(..., sweep_runner=None)`.
  - `DONE.json` gains `n_compute`; `walker_ref` becomes the GLOBAL index.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fstat_parallel_fit.py`:

```python
class DoneManifestTest(unittest.TestCase):
    """DONE.json records the GLOBAL reference walker and the rank count."""

    def test_manifest_keys(self):
        import inspect

        from lisatools.globalfit.moves import gbspecialstretch as gbs

        src = inspect.getsource(gbs.GBSpecialBase._run_fstat_fit)
        for key in ("walker_ref", "n_compute", "n_peaks", "wall_seconds",
                    "num_proposals", "clock", "epoch"):
            self.assertIn(key, src, f"DONE.json must record {key!r}")

    def test_fstat_call_accepts_a_holder_override(self):
        import inspect

        from lisatools.globalfit.moves import gbspecialstretch as gbs

        sig = inspect.signature(gbs.GBSpecialBase._fstat_call)
        self.assertIn("holder", sig.parameters)
        self.assertEqual(sig.parameters["holder"].kind,
                         inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(sig.parameters["holder"].default)
        sig_nm = inspect.signature(gbs.GBSpecialBase._fstat_NM)
        self.assertIn("holder", sig_nm.parameters)

    def test_run_fstat_grid_fit_forwards_a_sweep_runner(self):
        import inspect

        sig = inspect.signature(G.run_fstat_grid_fit)
        self.assertIn("sweep_runner", sig.parameters)
        self.assertIsNone(sig.parameters["sweep_runner"].default)
```

- [ ] **Step 2: Run it to see it fail**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t8a.log \
  python -m unittest tests.test_fstat_parallel_fit.DoneManifestTest -v
cat .wtenv/t8a.log
```
Expected: FAIL on all three.

- [ ] **Step 3: `sweep_runner` pass-through in `run_fstat_grid_fit`**

In `src/lisatools/sampling/fstat_gridfit.py:1239`, add `sweep_runner=None` (keyword, after `ratio_max=None`), document it in one sentence, and forward it in the `run_stacked_stage_b(...)` call at `:1310`:

```python
    stacked = run_stacked_stage_b(
        call_fstat, peaks, xp=xp, Tobs=Tobs, band_edges_hz=band_edges_hz,
        mc_lims=mc_lims, ratio_max=ratio_max, cache_path=cache_path,
        fingerprint_extra=fingerprint_extra, epoch=epoch,
        sweep_runner=sweep_runner,
    )
```

Docstring sentence:

```
    ``sweep_runner`` is forwarded to :func:`run_stacked_stage_b`; ``None``
    is the serial per-group kernel stream. The stacked-cache short circuit
    above it is unchanged -- a complete epoch never re-enters the kernel,
    parallel or not.
```

- [ ] **Step 4: Holder override on `_fstat_call` / `_fstat_NM`**

`_fstat_call` (`:20476`) — change the signature to `def _fstat_call(self, model, walker_ref, *, holder=None):`, add this paragraph to the docstring, and route the holder:

```
        ``holder`` overrides the ACA the scorer reads. The multi-rank fit
        passes the replicated single-row :class:`FStatRefRowHolder`
        (``gb_fstat_ref_row``), which is scored at ``data_index =
        noise_index = 0`` -- the GLOBAL reference walker's index is NOT an
        ACA row index on any rank but its owner, and the holder removes the
        question entirely. Single-shard by construction
        (``len(linear_data_arr) == 1``), so the router passes it through.
```

In the sig-het branch, replace

```python
            sig_comp = self.gb_wdm_comp
            holder = model.analysis_container_arr
```
with
```python
            sig_comp = self.gb_wdm_comp
            if holder is None:
                holder = model.analysis_container_arr
                _di = _ni = int(walker_ref)
            else:
                _di = _ni = 0
```
and the `route_sighet_fstat(...)` call's `data_index=int(walker_ref), noise_index=int(walker_ref)` with `data_index=_di, noise_index=_ni`.

Change the fallback return to `return lambda params: self._fstat_NM(model, params, walker_ref, holder=holder)`.

`_fstat_NM` (`:6648`) — change the signature to `def _fstat_NM(self, model, params_phys, walker_ref, *, holder=None):`; add to the docstring:

```
        ``holder`` overrides the residual source (the multi-rank fit's
        replicated reference row, scored at row 0). The per-unit lane
        adapter is bypassed under an override: it is armed for a walker's
        ACA row, not for a shipped row pair.
```

and in the body:

```python
        _lanes = getattr(self, "_fstat_nm_lanes", None)
        if holder is None and _lanes is not None and _lanes[0] == int(walker_ref):
```
and
```python
            _row = 0 if holder is not None else int(walker_ref)
            di = xp.full(params_phys.shape[0], _row, dtype=xp.int32)
            _holder = model.analysis_container_arr if holder is None else holder
            comp, method_name = self._fstat_comp_method()
            return _RoutedBandEngine.route_fstat_ll(
                comp, method_name, _holder, params_phys,
                data_index=di, noise_index=di, convert_to_ra_dec=False)
```

- [ ] **Step 5: Rewrite `_run_fstat_fit`'s body**

Replace `_run_fstat_fit` (`:20601` through its `return stacked, n_peaks`) with:

```python
    def _run_fstat_fit(self, model, k: int, branches=None):
        """Fit epoch ``k``'s F-stat grid. HEAD-side; ranks serve two commands.

        Three things differ from the single-process fit, all of them forced
        by the walker-block layout (design spec 2026-09-16):

        1. The reference walker is the GLOBAL argmax
           (:meth:`_fstat_global_reference`), not this rank's local one.
        2. The whole sweep scores through the REPLICATED reference row
           (``gb_fstat_ref_row``), not the live residual -- so the GB-free
           window is opened once, on the OWNING rank, around the snapshot,
           and no rank's residual stays mutated for the hours the fit runs.
        3. Stage B is split per Mc group by contiguous box range over every
           compute rank (``gb_fstat_stage_b``); stage A stays here.

        At ONE compute rank all three collapse: the reference is the same
        walker, the holder is built from this process's own rows with no
        MPI, and ``sweep_runner`` stays ``None`` -- the serial path, whose
        grids are pinned byte-for-byte by
        ``tests/test_fstat_parallel_fit.py``'s goldens.
        """
        from lisatools.sampling.fstat_gridfit import run_fstat_grid_fit

        cache_dir = self._epoch_dir(k)
        os.makedirs(cache_dir, exist_ok=True)
        w_global, owner_rank, local_index, lls = self._fstat_global_reference(model)
        _fanout = getattr(self, "fanout", None)
        n_compute = 1 if _fanout is None else int(_fanout.layout.n_compute)
        # Auditability: the epoch line carries the reference walker's total
        # lnL (residual+PSD combination) alongside its GLOBAL index and the
        # rank that owns it, so a run log shows WHICH state each epoch's grid
        # was fitted against -- "same peaks after a refit" is only
        # diagnosable with this visible.
        try:
            _ll_ref = float(lls[w_global]) if lls.size > w_global else float("nan")
            _ll_spread = float(lls.max() - lls.min()) if lls.size > 1 else float("nan")
        except Exception:
            _ll_ref, _ll_spread = float("nan"), float("nan")
        band_edges = _to_numpy(self.band_edges)
        # f0_lims convention (gb.py): the interior span, band_edges[1:-1].
        f0_lims = (float(band_edges[1]), float(band_edges[-2]))
        mc_lims = self.fstat_fit_kwargs.get("mc_lims") or [0.001, 1.0]
        t0 = time.perf_counter()
        logger.info(
            "%s: F-stat grid fit epoch %d starting (walker_ref=%d GLOBAL on "
            "rank %d row %d, lnL=%.3f, cold-walker lnL spread=%.3f, "
            "n_compute=%d, cache %s)",
            self.name, k, w_global, owner_rank, local_index, _ll_ref,
            _ll_spread, n_compute, cache_dir)
        # THE FLAG IS PART OF THE FINGERPRINT. ``GB_FSTAT_GB_FREE`` changes
        # the RESIDUAL the sweep runs against but nothing else about the
        # sweep's inputs, so without this the two modes produce different
        # grids under the SAME cache key: flip the flag, refit at the same
        # epoch, and the checkpoint layer hands back the other mode's grid
        # with no error and no warning.
        _gb_free = os.environ.get("GB_FSTAT_GB_FREE", "1") == "1"
        self._fstat_ref_row_fanout(model, branches, w_global, owner_rank,
                                   local_index)
        try:
            stacked, n_peaks = run_fstat_grid_fit(
                self._fstat_holder_call(model),
                xp=self.xp,
                # 1.0/self.df, NOT basis_settings.Tobs: the latter is absent
                # on FDSettings, and under FSTAT_FDOT_AXIS this value is no
                # longer only a node-density input -- it sets the f_mid shear
                # coefficient, so a wrong one costs acceptance on every birth.
                Tobs=1.0 / float(self.df),
                band_edges_hz=band_edges,
                f0_lims_hz=f0_lims,
                mc_lims=mc_lims,
                ratio_max=_gb_fdot_astro_ratio_max(self),
                cache_dir=cache_dir,
                fingerprint_extra=f"|epoch={k}|gbfree={int(_gb_free)}",
                epoch=k,
                sweep_runner=(self._fstat_stage_b_runner(model)
                              if n_compute > 1 else None),
            )
        # NOTE: the holder is NOT released here. The centre table
        # (``_install_ctr_table``) must score against the SAME reference and
        # the SAME row, so the caller of both releases it.
        wall = time.perf_counter() - t0
        # Feed the propose timer: this runs outside every other top-level
        # span, so without it the refit lands in [GB_TIMING]'s untracked
        # remainder and reads as an unexplained stall.
        _tm = getattr(self, "_prop_timer", None)
        if _tm is not None:
            _tm.add("fstat_grid_fit", wall)
        logger.info("%s: F-stat grid fit epoch %d done in %.1fs (%d peaks)",
                    self.name, k, wall, n_peaks)
        try:
            with open(os.path.join(cache_dir, "DONE.json"), "w") as f:
                json.dump(dict(epoch=k, walker_ref=int(w_global),
                               n_peaks=int(n_peaks), wall_seconds=wall,
                               num_proposals=int(self.num_proposals),
                               # how many compute ranks split stage B -- the
                               # rank -> box-range map is a pure function of
                               # (g_edges, n_compute), so a resume under a
                               # DIFFERENT count restarts each group cleanly
                               # rather than resuming another rank's slice
                               n_compute=int(n_compute),
                               # the refit clock at fit time -- read back by
                               # _epoch_fit_clock so the cadence budget
                               # survives restarts (2026-08-24)
                               clock=int(self._fstat_clock())), f)
        except OSError as exc:  # manifest is bookkeeping, never fatal
            logger.warning("%s: could not write DONE.json (%r)", self.name, exc)
        return stacked, n_peaks
    ```

- [ ] **Step 6: Log stage A's wall separately (spec decision 3)**

Stage A's epoch-1 wall time is the trigger for a later split, so it must be readable on its own rather than inferred by subtraction. In `src/lisatools/sampling/fstat_gridfit.py::run_fstat_grid_fit`, wrap the comb branch (`if os.path.exists(comb_cache): ... else: run_comb_scan(...)`, `:1292-1309`) with a timer and log it:

```python
    _t_stage_a = time.time()
    if os.path.exists(comb_cache):
        ...                                    # unchanged body
    else:
        ...                                    # unchanged body
    logger.info("[stageA] comb + peak selection: %d peaks in %s "
                "(head-only; see the parallel-fit design spec for when this "
                "becomes worth splitting too)",
                int(len(peaks)), _fmt_secs(time.time() - _t_stage_a))
```

`time` and `_fmt_secs` are already imported in that module.

- [ ] **Step 7: Release the holder after the centre table**

The caller of both `_run_fstat_fit` and `_install_ctr_table` is `setup()`. Find the `setup()` body that calls them (grep `_run_fstat_fit(` and `_install_ctr_table(` in `gbspecialstretch.py`) and wrap the pair so the holder is released once both are done:

```python
        finally:
            # The replicated reference row is a per-FIT resource: the grids
            # and the centre table must both score against it (the
            # 2026-08-24 same-residual rule), and it is tens of MB per rank.
            self._fstat_release_ref_row()
```

Add the same release at the END of `_gb_serve_fstat_stage_b`'s LAST call? No — the ranks cannot know which group is last. Instead, release the rank-side holder at the start of the NEXT `gb_fstat_ref_row` (the assignment in `_gb_serve_fstat_ref_row` already overwrites it) and in `_exit_rank_block`'s teardown is NOT appropriate. Leave the rank-side holder alive until the next fit: it is one row pair and the next `gb_fstat_ref_row` replaces it. Record that choice in a comment on `_fstat_release_ref_row`:

```python
        """Drop the replicated row holder. HEAD-side: called once the fit AND
        the centre table are done. Compute ranks keep theirs until the next
        ``gb_fstat_ref_row`` overwrites it -- they have no way to know which
        stage-B command was the last, and it is one row pair.
        """
```

- [ ] **Step 8: Run the tests**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t8b.log \
  python -m unittest tests.test_fstat_parallel_fit tests.test_fstat_gridfit tests.test_fstat_ctr_epoch tests.test_fstat_birth_seed tests.test_fstat_grid_fit_timing -v
cat .wtenv/t8b.log
```
Expected: all PASS.

- [ ] **Step 9: Commit**

```sh
git add src/lisatools/sampling/fstat_gridfit.py src/lisatools/globalfit/moves/gbspecialstretch.py tests/test_fstat_parallel_fit.py
git commit -m "$(cat <<'EOF'
_run_fstat_fit: global reference, replicated row scoring, parallel stage B

The fit now takes the global argmax walker, replicates its GB-free residual
and inverse-PSD rows once through gb_fstat_ref_row, scores every sweep
through that holder (data_index=0), and splits stage B across the compute
ranks when n_compute > 1. DONE.json records the GLOBAL walker_ref and the
n_compute the split used. At one compute rank nothing changes but the
scoring route, which the goldens pin.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: The centre table uses the same global reference and the same row

`_install_ctr_table` (`:20683`) re-derives `walker_ref` at `:20738` and re-opens its own GB-free window at `:20739`. Under a global reference that silently scores the centres against a different residual — exactly the bug the 2026-08-24 comment there guards against.

**Files:**
- Modify: `src/lisatools/globalfit/moves/gbspecialstretch.py`
- Test: `tests/test_fstat_parallel_fit.py`

**Interfaces:**
- Consumes: Task 6's `self._fstat_ref_holder` / `_fstat_ref_walker`; Task 8's `_fstat_holder_call`.
- Produces: `_install_ctr_table` scores through the replicated holder when one is live; falls back to building its own (global reference + its own `gb_fstat_ref_row`) when it is not.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fstat_parallel_fit.py`:

```python
class CentreTableReferenceTest(unittest.TestCase):
    """The centre sweep must reuse the fit's replicated row, not re-derive one."""

    def test_no_local_reference_walker_derivation_left(self):
        import inspect

        from lisatools.globalfit.moves import gbspecialstretch as gbs

        src = inspect.getsource(gbs.GBSpecialBase._install_ctr_table)
        self.assertNotIn(
            "self._fstat_reference_walker(model)", src,
            "the centre table must not re-derive a LOCAL reference walker; "
            "it scores through the fit's replicated reference row")
        self.assertIn("_fstat_ref_holder", src)
        self.assertIn("_fstat_holder_call", src)

    def test_the_holder_call_is_cached_so_grids_and_centres_share_it(self):
        """Spec verification 1(d): SAME scorer object, not a rebuilt twin.

        The fit and the centre sweep must score against one reference row
        and one sig-het reference-block stash. ``_fstat_holder_call``
        caching it is what makes that literal -- a rebuilt closure would
        silently re-snapshot and re-bucket.
        """
        from lisatools.globalfit.moves import gbspecialstretch as gbs

        move = gbs.GBSpecialBase.__new__(gbs.GBSpecialBase)
        move._fstat_ref_holder = object()
        move._fstat_ref_call = None
        sentinel = object()
        calls = []

        def fake_fstat_call(model, walker_ref, *, holder=None):
            calls.append((walker_ref, holder))
            return sentinel

        move._fstat_call = fake_fstat_call
        first = move._fstat_holder_call(None)
        second = move._fstat_holder_call(None)
        self.assertIs(first, sentinel)
        self.assertIs(second, first, "the scorer must be built once per fit")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], 0, "the holder is scored at row 0")
        self.assertIs(calls[0][1], move._fstat_ref_holder)

        move._fstat_release_ref_row()
        self.assertIsNone(move._fstat_ref_call)
        self.assertIsNone(move._fstat_ref_holder)
```

- [ ] **Step 2: Run it to see it fail**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t9a.log \
  python -m unittest tests.test_fstat_parallel_fit.CentreTableReferenceTest -v
cat .wtenv/t9a.log
```
Expected: FAIL on the `assertNotIn`.

- [ ] **Step 3: Implement**

In `_install_ctr_table`, replace the `if need_sweep and model is not None:` branch body (`:20736-20742`, the `walker_ref = ...` / `with self._gb_free_residual(...)` / `call = ...` / `host = ...` lines) with:

```python
        if need_sweep and model is not None:
            # The center sweep MUST see the SAME residual the epoch's peak
            # grids were fitted against -- the GB-FREE one (2026-08-24 fix:
            # this sweep used to run AFTER the fit's GB-free window closed,
            # so at any real refit the amplitude/SNR centers for exactly the
            # loud already-recovered peaks would have been fitted against
            # noise). Under the walker-block layout that is now literal: the
            # fit's REPLICATED reference row is still live on every rank, so
            # the centre sweep scores through the same holder at the same
            # global reference walker, and nothing here re-derives either
            # (a LOCAL argmax would silently pick a different walker, and
            # its index is not even a valid ACA row off its owner).
            if self._fstat_ref_holder is None:
                # A load path that never ran a fit (an epoch dropped in
                # offline, or a centre table missing from a complete epoch):
                # replicate the row now, exactly as the fit does.
                w_global, owner_rank, local_index, _lls = (
                    self._fstat_global_reference(model))
                self._fstat_ref_row_fanout(model, branches, w_global,
                                           owner_rank, local_index)
                _release_after = True
            else:
                _release_after = False
            logger.info(
                "[FSTAT_CTR %s] epoch %d centre sweep scores through the "
                "fit's replicated reference row (walker %s)",
                self.name, k, self._fstat_ref_walker)
            try:
                host = build_fstat_center_table(
                    self._fstat_holder_call(model), **_table_kwargs)
            finally:
                if _release_after:
                    self._fstat_release_ref_row()
        else:
            host = build_fstat_center_table(None, **_table_kwargs)
```

- [ ] **Step 4: Run the tests**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t9b.log \
  python -m unittest tests.test_fstat_parallel_fit tests.test_fstat_ctr_epoch tests.test_fstat_ctr_audit tests.test_fstat_ctr_fallback tests.test_fstat_centers_timing_spans -v
cat .wtenv/t9b.log
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```sh
git add src/lisatools/globalfit/moves/gbspecialstretch.py tests/test_fstat_parallel_fit.py
git commit -m "$(cat <<'EOF'
_install_ctr_table: score the centres through the fit's replicated reference row

The centre sweep used to re-derive walker_ref and re-open its own GB-free
window. Under a GLOBAL reference that would score the centres against a
different walker's residual -- the exact defect the 2026-08-24 comment there
guards -- and the index would not be a valid ACA row off its owner. It now
reuses the live holder, and replicates one itself on the load-only path.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: The acceptance gate — 1-rank vs 2-rank bit-identity under FakeWorld

The gate the spec names: a 2-rank FakeWorld epoch fit's stacked `.npz` must be bit-identical to the 1-rank serial fit's, every key.

The test drives the fan-out at the LIBRARY level with a real `WalkerFanout` / `ComputeService` over a `FakeWorld`, using a stand-in move object that carries only the F-stat surface — a built GB move needs a GPU, a `BandSorter` and gigabytes, which the laptop rules forbid. The `sweep_runner` and the two `gf_serve` bodies under test are the real ones.

**Files:**
- Modify: `tests/test_fstat_parallel_fit.py`

**Interfaces:**
- Consumes: everything from Tasks 2-9.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fstat_parallel_fit.py`:

```python
class _StageBRankStub:
    """A stand-in move exposing exactly the F-stat stage-B rank surface.

    Real ``GBSpecialBase`` needs a GPU, a built ``BandSorter`` and gigabytes
    of buffers, which the laptop budget forbids -- but the code under test
    (``_fstat_stage_b_payload`` / ``_fstat_stage_b_spec`` /
    ``_gb_serve_fstat_stage_b`` / ``_fstat_stage_b_runner``) touches only the
    fan-out, the spec slicing and ``run_stage_b_group``. Those four methods
    are BORROWED from the real class, so the test exercises production code;
    everything else is a stub.
    """

    from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase as _B

    # ``_fstat_stage_b_payload`` / ``_fstat_stage_b_spec`` are staticmethods,
    # so reading them off the class yields plain functions -- re-wrap.
    _fstat_stage_b_payload = staticmethod(_B._fstat_stage_b_payload)
    _fstat_stage_b_spec = staticmethod(_B._fstat_stage_b_spec)
    _gb_serve_fstat_stage_b = _B._gb_serve_fstat_stage_b
    _fstat_stage_b_runner = _B._fstat_stage_b_runner
    gf_move_name = "gb_pe"
    name = "gb_stub"

    class _Pool:
        def free_all_blocks(self):
            return None

    def __init__(self, fanout, call_fstat):
        self.fanout = fanout
        self._call = call_fstat
        self._fstat_ref_holder = object()   # non-None: the guard's only use
        self.mempool = self._Pool()
        self.nwalkers = None
        self.ntemps = None
        self._prop_timer = None

    @property
    def xp(self):
        return np

    def _bind_rank_acs(self, model):
        return None

    def _fstat_holder_call(self, model):
        return self._call

    def _fanout_cmd(self, op, per_rank_payload, model):
        replies = self.fanout.run(
            op, move=self.gf_move_name, per_rank_payload=per_rank_payload,
            local_body=lambda p, _m: self.gf_serve(op, p, self.fanout.clock, model),
            merge=lambda r: r)
        return replies, None

    def gf_serve(self, op, payload, clock, model):
        if op == "gb_fstat_stage_b":
            return self._gb_serve_fstat_stage_b(payload, clock, model)
        raise ValueError(op)


class _RefRowRankStub:
    """A stand-in move for ``gb_fstat_ref_row``: fake ACA, fake GB-free window.

    Borrows the production ``_gb_serve_fstat_ref_row`` so the collective,
    the header exchange and the holder construction under test are the real
    ones. The GB-free window is faked as "+1.0 on this walker's residual
    row", which makes two things checkable at once: the SHIPPED row carries
    the window's effect, and the owner's LIVE residual is back to its
    original value afterwards.
    """

    from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase as _B

    _gb_serve_fstat_ref_row = _B._gb_serve_fstat_ref_row
    _fstat_ref_row_payload = staticmethod(_B._fstat_ref_row_payload)
    _fstat_ref_holder = None
    _fstat_ref_call = None
    _fstat_ref_walker = None
    _fstat_ref_branches = None
    name = "gb_stub"

    def __init__(self, fanout, acs):
        self.fanout = fanout
        self._acs = acs

    @property
    def xp(self):
        return np

    def _bind_rank_acs(self, model):
        return self._acs

    @contextlib.contextmanager
    def _gb_free_residual(self, model, branches, walker_ref):
        rows = np.asarray(self._acs.linear_data_arr[0]).reshape(
            self._acs.acs_total_entries, -1)
        rows[int(walker_ref)] += 1.0
        self._gb_free_n_live = 7
        try:
            yield
        finally:
            rows[int(walker_ref)] -= 1.0


class RefRowReplicationTest(unittest.TestCase):
    """Spec verification 1(a): every rank gets the OWNER's windowed row."""

    def _run(self, n_compute, owner_rank_index, local_index):
        from lisatools.globalfit.communication import ranks as R
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.fanout import (
            ComputeService,
            WalkerFanout,
        )
        from tests.test_fstat_ref_row_holder import _FakeParent

        world = FakeWorld(n_compute + 1)
        B = 2

        def body(rank, comm):
            layout = R.build_layout(comm, B * n_compute, list(range(n_compute)))
            if rank not in layout.compute_ranks:
                return None
            fcomm = layout.make_fanout_comm(comm)
            acs = _FakeParent(B, 6, 9)
            # make every rank's buffers DIFFERENT so a missing broadcast
            # cannot pass by coincidence
            acs.linear_data_arr[0] += 100.0 * layout.fanout_rank(rank)
            acs.linear_psd_arr[0] += 100.0 * layout.fanout_rank(rank)
            before = np.array(acs.linear_data_arr[0], copy=True)
            fanout = WalkerFanout(fcomm, layout, rank, model=None)
            stub = _RefRowRankStub(fanout, acs)
            owner_rank = layout.compute_ranks[owner_rank_index]
            w_global = layout.block_of(owner_rank)[0] + local_index
            if rank == layout.head_rank:
                replies = fanout.run(
                    "gb_fstat_ref_row", move="gb_pe",
                    per_rank_payload=lambda r, w0, w1:
                        stub._fstat_ref_row_payload(
                            w_global, owner_rank, local_index, True),
                    local_body=lambda p, _m: stub._gb_serve_fstat_ref_row(
                        p, fanout.clock, None),
                    merge=lambda r: r)
                fanout.stop()
            else:
                service = ComputeService(fcomm, layout, rank,
                                         registry={"gb_pe": stub}, model=None)
                service.serve()
            after = np.asarray(acs.linear_data_arr[0])
            return {
                "data_row": np.asarray(
                    stub._fstat_ref_holder.linear_data_arr[0]).copy(),
                "psd_row": np.asarray(
                    stub._fstat_ref_holder.linear_psd_arr[0]).copy(),
                "walker_ref": stub._fstat_ref_walker,
                "residual_restored": np.array_equal(before, after),
                "expected": (
                    np.asarray(before).reshape(B, -1)[local_index] + 1.0
                    if rank == owner_rank else None),
            }

        return world.run(body)

    def test_every_rank_holds_the_owners_windowed_row(self):
        for n_compute, owner_idx, local in ((2, 1, 0), (2, 0, 1), (3, 2, 1)):
            with self.subTest(n_compute=n_compute, owner=owner_idx):
                out = {r: v for r, v in self._run(n_compute, owner_idx, local).items()
                       if v is not None}
                expected = next(v["expected"] for v in out.values()
                                if v["expected"] is not None)
                for rank, v in out.items():
                    np.testing.assert_array_equal(
                        v["data_row"], expected,
                        f"rank {rank} did not receive the owner's row")
                    self.assertTrue(v["residual_restored"],
                                    f"rank {rank}'s live residual was left mutated")
                self.assertEqual(
                    len({v["walker_ref"] for v in out.values()}), 1,
                    "every rank must record the same GLOBAL reference walker")

    def test_single_compute_rank_needs_no_collective(self):
        out = {r: v for r, v in self._run(1, 0, 1).items() if v is not None}
        self.assertEqual(len(out), 1)
        v = next(iter(out.values()))
        np.testing.assert_array_equal(v["data_row"], v["expected"])
        self.assertTrue(v["residual_restored"])


class ParallelStageBGateTest(unittest.TestCase):
    """THE acceptance gate: 2 ranks == 1 rank, byte for byte."""

    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _fanout_run(self, n_compute, tmpdir, *, grouped=True, kill_rank=None):
        """Run the grouped stage B over ``n_compute`` FakeWorld ranks."""
        from lisatools.globalfit.communication import ranks as R
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.fanout import (
            ComputeService,
            WalkerFanout,
        )

        world = FakeWorld(n_compute + 1)
        out = {}

        def body(rank, comm):
            layout = R.build_layout(comm, 2 * n_compute, list(range(n_compute)))
            if rank not in layout.compute_ranks:
                return None
            fcomm = layout.make_fanout_comm(comm)
            if rank == layout.head_rank:
                fanout = WalkerFanout(fcomm, layout, rank, model=None)
                stub = _StageBRankStub(fanout, _fake_call_fstat())
                path = run_golden(tmpdir, grouped=grouped,
                                  sweep_runner=stub._fstat_stage_b_runner(None))
                fanout.stop()
                return path
            fanout = WalkerFanout(fcomm, layout, rank, model=None)
            stub = _StageBRankStub(fanout, _fake_call_fstat())
            service = ComputeService(fcomm, layout, rank,
                                     registry={None: stub, "gb_pe": stub},
                                     model=None)
            service.serve()
            return None

        out = world.run(body)
        head = min(r for r in out if out[r] is not None)
        return out[head]

    def test_two_ranks_are_bit_identical_to_the_serial_fit(self):
        got = self._fanout_run(2, self.d, grouped=True)
        assert_npz_identical(self, got, GOLDEN_GROUPED)

    def test_three_ranks_are_bit_identical_too(self):
        got = self._fanout_run(3, self.d, grouped=True)
        assert_npz_identical(self, got, GOLDEN_GROUPED)

    def test_single_group_split_is_bit_identical(self):
        got = self._fanout_run(2, self.d, grouped=False)
        assert_npz_identical(self, got, GOLDEN_SINGLE)

    def test_partials_are_deleted_after_assembly(self):
        self._fanout_run(2, self.d, grouped=True)
        parts = os.path.join(self.d, "fstat_grid_parts")
        leftovers = [f for f in os.listdir(parts) if f.endswith(".npy")] \
            if os.path.isdir(parts) else []
        self.assertEqual(leftovers, [], f"stage-B partials left behind: {leftovers}")

    def test_resume_with_the_same_rank_count_reuses_the_checkpoints(self):
        """A rank that dies mid-sweep resumes from its own progress file."""
        from lisatools.sampling import fstat_gridfit as GG

        calls = {"n": 0}
        real = GG.run_stage_b_group

        def flaky(spec, call_fstat, *, xp):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated rank death mid-sweep")
            return real(spec, call_fstat, xp=xp)

        with unittest.mock.patch.object(GG, "run_stage_b_group", flaky):
            with self.assertRaises(Exception):
                self._fanout_run(2, self.d, grouped=True)
        # the progress files from the surviving sweeps must still be there
        parts = os.path.join(self.d, "fstat_grid_parts")
        progress = [f for f in os.listdir(parts) if f.endswith(".progress.npz")]
        self.assertTrue(progress, "per-rank checkpoints must survive a death")
        self.assertTrue(any("_r" in f for f in progress),
                        f"checkpoints must be per rank, got {progress}")
        got = self._fanout_run(2, self.d, grouped=True)
        assert_npz_identical(self, got, GOLDEN_GROUPED)

    def test_a_different_rank_count_restarts_cleanly_with_the_same_result(self):
        self._fanout_run(2, self.d, grouped=True)
        os.remove(os.path.join(self.d, "fstat_grid_peaks_stacked.npz"))
        got = self._fanout_run(3, self.d, grouped=True)
        assert_npz_identical(self, got, GOLDEN_GROUPED)
```

Add `from unittest import mock` to the module imports and use `mock.patch.object` (the snippet above writes `unittest.mock.patch`; either is fine as long as the import exists).

- [ ] **Step 2: Run it**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t10a.log \
  python -m unittest tests.test_fstat_parallel_fit.ParallelStageBGateTest -v
cat .wtenv/t10a.log
```
Expected: initially FAIL. The likely first failures are mechanical, not design: the `ComputeService` registry key (`(stage, move_name)` with a bare-name fallback — `registry={None: stub, "gb_pe": stub}` covers both), the `WalkerFanout` `model=None` argument, and `build_layout`'s exact signature. Fix the TEST, not the production code, unless the failure is a genuine defect (assembly order, checkpoint names, partial cleanup) — in that case fix the production code and say so in the commit.

- [ ] **Step 3: Iterate until every assertion passes**

Run after each fix. The gate is not satisfied by skipping: if a test cannot be made to run, record why in `.superpowers/sdd/` and raise it in the task review rather than deleting the assertion.

- [ ] **Step 4: Run the whole F-stat and fan-out surface**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t10b.log \
  python -m unittest tests.test_fstat_parallel_fit tests.test_fstat_gridfit tests.test_fstat_ref_row_holder tests.test_fanout_fakecomm tests.test_fanout_passthrough tests.test_gb_rank_session -v
cat .wtenv/t10b.log
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```sh
git add tests/test_fstat_parallel_fit.py
git commit -m "$(cat <<'EOF'
test: the parallel F-stat acceptance gate -- 2 and 3 ranks bit-identical to serial

A real WalkerFanout/ComputeService over FakeWorld drives the production
sweep_runner and gb_fstat_stage_b body on a tiny analytic fixture: the
assembled stacked npz matches the serial golden byte for byte in both npz
formats, the partials are cleaned up, a rank death leaves per-rank
checkpoints that a same-count rerun resumes, and a different rank count
restarts cleanly with the same result.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Runbook and launch docs

**Files:**
- Modify: `docs/multirank-cluster-gates.md`
- Modify: `docs/global-fit-launch.md`

- [ ] **Step 1: Add the epoch-1 verification step to the cluster gates**

Append a new numbered step to `docs/multirank-cluster-gates.md`, matching the file's existing step formatting (read the last step first and mirror its headings, code-fence style and "what good looks like" phrasing):

```markdown
## Step N. Parallel F-stat epoch fit (WP7 addendum, 2026-09-16)

The first epoch that REFITS under the walker-block layout
(`GB_FSTAT_REFIT_EVERY=50`, so epoch 1 of the 6mo continuation) is the
cluster half of the parallel F-stat gate. Measured baseline: epoch 0 took
1 h 45 min, all of it stage B, on one of two GPUs.

What to collect from the head's `globalfit_run.log`:

1. `F-stat grid fit epoch 1 starting (walker_ref=<w> GLOBAL on rank <r> row
   <l>, ..., n_compute=<n>, ...)` — the reference is the GLOBAL argmax and
   names its owner. Cross-check `walker_ref` against `DONE.json`, which now
   also records `n_compute`.
2. `F-stat reference row replicated to <n> rank(s) ... <X> MB residual +
   <Y> MB invC per rank` — at 6mo expect roughly 18 MB and 54 MB.
   A wildly different size means `Nf_active` is not what the design assumed;
   record the real number.
3. One `[FSTAT_STAGEB] group g/G over <n> rank(s): boxes [...] | wall min
   ... max ... (imbalance ...%)` line per Mc group. Per-box cost is uniform
   within a group, so the residue is the sig-het reference-block build,
   which scales with f0 span. An imbalance above ~20% is the signal to
   weight the split by f0 span instead of box count (spec, Risks).
4. `[FANOUT] op=gb_fstat_stage_b ... head_s=... max_rank_s=... wait_s=...`
   per group — the transport view of the same balance.
5. `F-stat grid fit epoch 1 done in <W>s` — compare `W` against epoch 0's
   `wall_seconds` in `epoch_0000/DONE.json`. Expect roughly `1/n_compute`
   of the stage-B share plus the unchanged stage A and centre table.

Correctness, if a serial refit of the same state is affordable: run one with
`GF_LEGACY_RANK_LAYOUT=1` into a scratch fit root and diff the two
`fstat_grid_peaks_stacked.npz` files key by key (the `assert_npz_identical`
helper in `tests/test_fstat_parallel_fit.py` is the reference comparison).
They must be byte-identical.

Failure modes to watch for:

- `gb_fstat_stage_b arrived before gb_fstat_ref_row` — a rank missed the
  replication command; the run aborts through `RemoteWorkerError`.
- `stage-B partial gN rM changed under us` — shared-filesystem fault
  between the rank's reply and the head's read.
- `stage-B group N: assembled (...) from k partials, expected (...)` — a
  rank/range map mismatch; check that every rank sees the same
  `n_compute`.
- A rank dying mid-sweep aborts the run as for any op; the per-rank
  checkpoints (`<epoch>/fstat_grid_parts/stageb_g*_r*.progress.npz`) make
  the retry cheap AS LONG AS the relaunch uses the same `n_compute`. A
  changed rank count restarts each group cleanly — correct, but it pays the
  whole sweep again.
```

Replace `Step N` with the next number in the file.

- [ ] **Step 2: Note the `FSTAT_SIGHET_MULTIDEV` semantics change**

In `docs/global-fit-launch.md`, in whichever section covers the GB/F-stat environment knobs (grep for `FSTAT_` in that file; if there is no such section, add the note to the multi-rank section that covers `GPUS_PER_RANK`), add:

```markdown
- **`FSTAT_SIGHET_MULTIDEV=1` is inert under the walker-block layout.** It
  fans the F-stat scorer out over a rank's OWN devices, and the current
  pipeline gives each rank exactly one GPU — so it silently does nothing.
  The rank split is what replaces that parallelism: stage B is divided
  across compute ranks by contiguous box range (design spec
  `docs/superpowers/specs/2026-09-16-parallel-fstat-fit-design.md`). Leave
  the variable set; it costs nothing and re-engages if anyone ever runs
  `GPUS_PER_RANK > 1`.
```

- [ ] **Step 3: Sanity-check the markdown**

```sh
git diff --stat docs/
```
Expected: only the two doc files, no stray edits.

- [ ] **Step 4: Commit**

```sh
git add docs/multirank-cluster-gates.md docs/global-fit-launch.md
git commit -m "$(cat <<'EOF'
docs: the parallel F-stat epoch-1 cluster gate and the FSTAT_SIGHET_MULTIDEV note

What to read out of the first refit under the walker-block layout (global
walker_ref and its owner, row sizes, per-group per-rank stage-B balance,
wall vs epoch 0) plus the failure modes, and the fact that
FSTAT_SIGHET_MULTIDEV is inert at one GPU per rank -- the rank split
replaces it.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
EOF
)"
```

---

## Final verification (after Task 11, before the whole-branch review)

Run each command separately, honouring the one-python-process rule between them.

```sh
source /Users/mkatz/miniconda3/etc/profile.d/conda.sh && conda activate deving

# 1. The feature's own surface
.wtenv/wt_run.sh $PWD/src .wtenv/final_feature.log \
  python -m unittest tests.test_fstat_parallel_fit tests.test_fstat_ref_row_holder -v

# 2. The F-stat regression batch
.wtenv/wt_run.sh $PWD/src .wtenv/final_fstat.log \
  python -m unittest tests.test_fstat_gridfit tests.test_fstat_mc_groups tests.test_fstat_fdot_axis tests.test_fstat_ctr_epoch tests.test_fstat_ctr_audit tests.test_fstat_proposal -v

# 3. The multi-rank plumbing batch
.wtenv/wt_run.sh $PWD/src .wtenv/final_multirank.log \
  python -m unittest tests.test_fanout_fakecomm tests.test_fakecomm tests.test_fanout_passthrough tests.test_rank_layout tests.test_gb_rank_plumbing tests.test_gb_rank_session -v

# 4. The gated GB smoke, ALONE (~4.5 GB) -- nothing else running
RUN_GF_GB_SMOKE=1 .wtenv/wt_run.sh $PWD/src .wtenv/final_gb_smoke.log \
  python -m unittest tests.test_multirank_gb_smoke -v
```

`tests/test_gbspecial_flow.py` is NEVER run (10-26 GB). If a module name above does not exist, `ls tests | grep <stem>` and use the real name; never silently drop a batch.
</content>
</invoke>
