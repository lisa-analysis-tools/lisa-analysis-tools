# Multi-rank global fit, Plan 2: run.py + recipe integration (WP2, WP3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Plan 1's communication package into the run: every rank resolves its role and device before the build, the head broadcasts the initial state, every rank builds only its walker block of the residual, likelihoods are gathered as a setup-phase collective, the head runs the recipe while computation ranks serve commands, and move construction uses local walker counts. Single-rank runs stay byte-identical to today.

**Architecture:** `prepare_rank(fit, comm)` runs before `fit.build()` (driver hook in `StockGlobalFit.run`). `GlobalFit` derives roles from the `WalkerBlockLayout`; `prepare_main` (head) and `prepare_compute` (computation ranks) share `_build_acs_and_recipe`, which calls `setup_acs(..., walker_block=)` and gathers likelihoods with `WalkerFanout.allgather_walker_vector`. The head owns the HDF backend, midit checkpoint, engine and recipe; computation ranks hold an in-memory engine and run `ComputeService.serve()`. A startup guard refuses `n_compute > 1` while any move lacks a `gf_serve` override (Plans 4/5 add them), so this plan never produces a silently head-only run. Recipe builders size moves and `TemperatureControl` at the local block and block-gate the GB/VGB setup-time residual subtraction.

**Tech Stack:** Python 3.12, numpy, mpi4py (present in `deving`, imported by `run.py` at module level as today), eryn, the Plan 1 package `lisatools.globalfit.communication`, `unittest`, `FakeWorld` for in-process multi-rank tests.

**Spec:** `docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md` (Decisions, Architecture "Phases per rank", WP2, WP3) plus these two rulings recorded after Plan 1: (a) at size 2 the saver stays aliased to the head, BUT a `-n 2` launch on a pool that cannot host two compute ranks must warn, explain, and demote rank 1 to the dedicated saver instead of failing; (b) the HDF backend probe in `GlobalFitSetup.__init__` stays as is (it is a short read-only open at build time, before any writer exists), so compute ranks never open the store afterwards.

## Global Constraints

- Work ONLY in the worktree `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-multirank` (branch `multirank-walker-blocks`, fast-forwarded to `dev` b63a8041). Never touch the main `dev` checkout.
- Commits: one per task on this branch, never push, trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Never commit `.superpowers/` or `.wtenv/`.
- Tests through the worktree runner after `source /Users/mkatz/miniconda3/etc/profile.d/conda.sh && conda activate deving`:
  ```sh
  .wtenv/wt_run.sh $PWD/src .wtenv/<name>.log python -m unittest <modules> -v; tail -n 30 .wtenv/<name>.log
  ```
  One python process at a time (8 GB laptop), CPU only, never `tests/test_gbspecial_flow.py` or the full suite. The gated smokes need `RUN_GF_SMOKE=1` in the environment of the runner command.
- `black` is not installed: every line ≤ 100 characters (`awk 'length > 100'`). Repo rules: never store an array module on an instance; guard `__getattr__` against dunders; env knob = capitalized attribute name.
- **Single-rank invariance:** with `layout.is_single()` every code path added here must reduce to today's operations in today's order. `setup_acs(walker_block=None)` and `_global_likelihood` in single mode are the identity; `prepare_main` in single mode performs the same sequence as before.
- **Multi-rank guard:** `n_compute > 1` is refused at startup unless every leaf move either overrides `GlobalFitMove.gf_serve` or is a `FunctionMove` (head-only by design). Plans 4/5 lift the refusal move by move.
- Interfaces consumed from Plan 1 (exact): `communication.ranks.prepare_rank(fit, comm, *, logger=None, ...)`, `build_layout(comm, nwalkers, gpu_pool, *, gpus_per_rank=1, ranks_per_gpu=1, main_rank=0, legacy=None)`, `WalkerBlockLayout` (`head_rank`, `saver_rank`, `compute_ranks`, `worker_ranks`, `n_compute`, `is_single()`, `role_of`, `block_of`, `local_gpus`, `fanout_rank`, `make_fanout_comm`, `describe`, `digest`, `legacy`), `RankRole`, `derive_rank_seed(base_seed, layout, rank)`, `install_mpi_abort_on_error(comm)`, `rank_tag`, `prefix_stdout`; `communication.fanout.WalkerFanout(comm, layout, rank, *, model=None, logger=None)` with `run/ping/stop/allgather_walker_vector/enter_stage/note_iteration/clock`, `ComputeService(comm, layout, rank, *, registry, model=None, builtins=None, logger=None).serve()`, `concat_blocks`; `communication.walkerslice.slice_state(full, w0, w1, *, sub_states=...)`; `moves.globalfitmove.MoveBuildContext(layout=, fanout=, rank=, state_local=)` auto-filled from `curr.rank_layout / curr.fanout / curr.rank`; `GlobalFitMove.gf_serve` default raises `NotImplementedError`.

---

## File Structure

| File | Responsibility in this plan |
|---|---|
| `src/lisatools/globalfit/communication/ranks.py` (modify) | size-2 graceful fallback + `notes` on the layout (Task 1) |
| `src/lisatools/globalfit/run.py` (modify) | `GlobalFitSetup` rank attrs; `GlobalFit.__init__` roles/layout/fanout comm/logging/abort hook; `setup_acs(walker_block=)`; `_global_likelihood`; `prepare_main` split into shared pieces + `prepare_compute`; `run_global_fit` branches; `_release_rank_gpu_pool`; the fan-out readiness guard; seeds (Tasks 2-4) |
| `src/lisatools/globalfit/loginfo.py` (modify) | `setup_root_file_handler(..., filename=)` (Task 2) |
| `src/lisatools/globalfit/recipe.py` (modify) | `Recipe.fanout` hooks; local walker counts in `SingleSourcePEBuilder.build` / `build_noise_moves`; `_local_walker_block`; block-gated setup-time subtraction at six sites (Task 5) |
| `src/lisatools/globalfit/stock/base.py` (modify) | `StockGlobalFit.run` calls `prepare_rank` (Task 6) |
| `tests/test_rank_layout.py` (modify) | size-2 fallback tests (Task 1) |
| `tests/test_run_multirank_helpers.py` (create) | unit tests for the pure helpers extracted from run.py (Tasks 2-4) |
| `tests/test_recipe_local_block.py` (create) | `_local_walker_block` + builder sizing tests (Task 5) |
| `tests/test_multirank_blank_smoke.py` (create) | gated end-to-end: `erebor.blank` through `GlobalFit` on `FakeWorld(1)` and `FakeWorld(2)` (Task 7) |

---

### Task 1: Size-2 graceful fallback (`build_layout` warns and demotes rank 1 to the saver)

**Files:**
- Modify: `src/lisatools/globalfit/communication/ranks.py` (`WalkerBlockLayout` dataclass; `build_layout`; `prepare_rank` logging)
- Test: `tests/test_rank_layout.py`

**Interfaces:**
- Consumes: Plan 1 `build_layout`, `resolve_roles`, `RankRole`.
- Produces: `WalkerBlockLayout.notes: tuple[str, ...]` (default `()`), included in `describe()`; the fallback behaviour: when `size == 2`, not legacy, the pool is non-empty and `len(pool) * ranks_per_gpu // gpus_per_rank < 2`, then `saver_rank = 1`, `compute_ranks = (head,)`, a `UserWarning` is emitted, and the same text is in `layout.notes[0]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rank_layout.py` (inside `BuildLayoutTest`):

```python
    def test_size2_single_gpu_demotes_rank1_to_saver_with_warning(self):
        import warnings

        world = FakeWorld(2)

        def fn(rank, comm):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                lay = build_layout(comm, 4, [0], legacy=False)
            return lay, [str(w.message) for w in caught]

        out = world.run(fn)
        lay, messages = out[0]
        self.assertEqual(lay.compute_ranks, (0,))
        self.assertEqual(lay.saver_rank, 1)
        self.assertTrue(lay.is_single())
        self.assertEqual(lay.role_of(1), RankRole.SAVER)
        self.assertEqual(lay.block_of(0), (0, 4))
        self.assertEqual(lay.placements[0].devices, (0,))
        self.assertEqual(len(lay.notes), 1)
        self.assertIn("RANKS_PER_GPU=2", lay.notes[0])
        self.assertIn("-n 1", lay.notes[0])
        self.assertTrue(any("dedicated saver" in m for m in messages))
        self.assertIn("dedicated saver", lay.describe())
        self.assertEqual(out[1][0].describe(), lay.describe())

    def test_size2_with_enough_gpus_keeps_two_compute_ranks(self):
        lay = FakeWorld(2).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(lay.compute_ranks, (0, 1))
        self.assertEqual(lay.notes, ())

    def test_size2_ranks_per_gpu_2_on_one_gpu_keeps_two_compute_ranks(self):
        lay = FakeWorld(2).run(
            lambda r, c: build_layout(c, 4, [0], legacy=False, ranks_per_gpu=2)
        )[0]
        self.assertEqual(lay.compute_ranks, (0, 1))
        self.assertEqual(lay.notes, ())

    def test_size3_single_gpu_still_hard_errors(self):
        with self.assertRaises(RuntimeError):
            _layouts(FakeWorld(3), 4, [0])
```

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p2t1.log python -m unittest tests.test_rank_layout -v; tail -n 30 .wtenv/p2t1.log`
Expected: the first test fails (`compute_ranks == (0, 1)` / no `notes` attribute); the other three may already pass.

- [ ] **Step 3: Implement**

In `ranks.py`, add the field to `WalkerBlockLayout` after `legacy: bool = False`:

```python
    #: human-readable notes about non-default choices (e.g. the size-2 fallback)
    notes: tuple = ()
```

In `describe()`, after building `lines` and before `return`, add:

```python
        for note in self.notes:
            lines.append(f"  note: {note}")
```

In `build_layout`, right after `pool = [int(g) for g in (gpu_pool or [])]` and the `if legacy: compute = (head,)` block, insert the fallback (before the `nwalkers % n_compute` check, since it changes `n_compute`):

```python
    notes = []
    if size == 2 and not legacy and pool and len(pool) * m // k < 2:
        # A `-n 2` launch on a pool that cannot host two compute ranks: instead
        # of the over-subscription error, rank 1 becomes the dedicated saver
        # (user ruling 2026-09-15). The head then computes every walker exactly
        # as a single-rank run does.
        other = [r for r in range(size) if r != head][0]
        saver = other
        compute = (head,)
        note = (
            f"size-2 launch on a per-node GPU pool {pool} that supports only "
            f"{len(pool) * m // k} compute rank(s): rank {other} runs as the "
            "dedicated saver and the head computes all walkers. To use two "
            "compute ranks on this pool set RANKS_PER_GPU=2; for synchronous "
            "saves with no saver rank launch with -n 1."
        )
        notes.append(note)
        warnings.warn(note, UserWarning, stacklevel=2)
```

Add `import warnings` to the module imports. Pass `notes=tuple(notes)` into the `WalkerBlockLayout(...)` constructor call at the end of `build_layout`. In `prepare_rank`, extend the existing INFO log so each note is logged once (`layout.describe()` already carries them, so no separate call is needed; just confirm the describe output is what is logged).

Check the over-subscription check still runs for `size >= 3`: it does, because the fallback only rewrites `compute`/`saver` for `size == 2`.

- [ ] **Step 4: Run to verify pass**

Same command. Expected: `Ran 22 tests ... OK`.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/communication/ranks.py tests/test_rank_layout.py
git commit -m "feat(communication): size-2 launch on a too-small GPU pool warns and demotes rank 1 to the saver

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `GlobalFit` roles from the layout, per-rank logging, abort hook, pool release, run branches

**Files:**
- Modify: `src/lisatools/globalfit/run.py` (`GlobalFitSetup` class body ~line 151-184; `GlobalFit.resolve_rank_roles` 386-411; `GlobalFit.__init__` 413-467; `run_global_fit` 2214-2275; `_release_helper_gpu_pool` 2277-2312), `src/lisatools/globalfit/loginfo.py` (`setup_root_file_handler` 88-108)
- Test: `tests/test_run_multirank_helpers.py`

**Interfaces:**
- Consumes: `communication.ranks` (`build_layout`, `RankRole`, `rank_tag`, `prefix_stdout`, `install_mpi_abort_on_error`).
- Produces (module-level helpers in `run.py`, unit-tested): `_rank_log_filenames(layout, rank) -> (root_log, gf_log)`; `_fanout_unready_moves(runtime_moves) -> list[str]` (names of leaf moves with the default `gf_serve` that are not `FunctionMove`s); `_leaf_moves(moves)` (recursive flatten of `GFCombineMove.moves`, tuples unwrapped); on `GlobalFit`: `self.layout`, `self.role`, `self.compute_ranks`, `self.worker_ranks`, `self.fanout_comm`, `self.main_rank`, `self.results_rank`, `self.used_ranks`, `self.ranks_to_give` (legacy spares only); `_release_rank_gpu_pool()`; `run_global_fit` dispatching on `self.role`. `GlobalFitSetup` gains class-level defaults `rank_layout = None`, `fanout = None`, `rank = None`, `rank_device_mode = None`. `setup_root_file_handler(log_dir, level, filename="globalfit_run.log")`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_multirank_helpers.py`:

```python
"""Pure helpers run.py uses to wire the multi-rank roles (no build, no MPI)."""

import unittest

from eryn.moves import CombineMove

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.ranks import build_layout
from lisatools.globalfit.moves.functionmove import FunctionMove
from lisatools.globalfit.moves.globalfitmove import GlobalFitMove
from lisatools.globalfit.run import (
    _fanout_unready_moves,
    _leaf_moves,
    _rank_log_filenames,
)


class _Combine(CombineMove):
    """A bare eryn CombineMove holding the given moves (no proposal needed here)."""

    def __init__(self, moves):
        self.moves = moves


class _Served(GlobalFitMove):
    gf_move_name = "served"

    def gf_serve(self, op, payload, clock, model):
        return None


class _Unserved(GlobalFitMove):
    gf_move_name = "unserved"


def _fn(model, state):
    return state, None


class HelpersTest(unittest.TestCase):
    def test_leaf_moves_flattens_combines_and_tuples(self):
        a, b = _Served(name="a"), _Unserved(name="b")
        leaves = _leaf_moves([_Combine([a, (b, 0.5)]), _Combine([_Combine([a])])])
        self.assertEqual(leaves, [a, b, a])

    def test_fanout_unready_names_default_gf_serve_only(self):
        # FunctionMove(Move, GlobalFitMove, ErynMove) lives in moves/functionmove.py; read
        # its constructor (fn first, keyword name) and adapt this line if it differs.
        fn_move = FunctionMove(_fn, name="fn")
        plain_eryn = CombineMove(moves=[])  # a non-GlobalFitMove leaf: head-only, not refused
        names, head_only = _fanout_unready_moves(
            [_Combine([_Served(name="a"), _Unserved(name="b"), fn_move]), plain_eryn]
        )
        self.assertEqual(names, ["unserved"])
        self.assertEqual(head_only, ["CombineMove"])

    def test_rank_log_filenames(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(_rank_log_filenames(lay, 0), ("globalfit_run.log", "global_fit.log"))
        self.assertEqual(
            _rank_log_filenames(lay, 1), ("globalfit_run.rank1.log", "global_fit.rank1.log")
        )
        self.assertEqual(
            _rank_log_filenames(lay, 2), ("globalfit_run.rank2.log", "global_fit.rank2.log")
        )


if __name__ == "__main__":
    unittest.main()
```

(Check `FunctionMove`'s constructor in `moves/globalfitmove.py` line ~216: `FunctionMove(move, *, name=None, branch=None, debug=None)`; adjust the test's construction if the runtime wrapper class differs — the guard must recognise whatever class `Recipe._coerce_move` produces for a plain function AND its materialized runtime. Read `_coerce_move` and `FunctionMove.setup` before writing the guard; the test must construct the runtime object the recipe would install.)

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p2t2.log python -m unittest tests.test_run_multirank_helpers -v; tail -n 25 .wtenv/p2t2.log`
Expected: `ImportError: cannot import name '_fanout_unready_moves'`.

- [ ] **Step 3: Implement**

`loginfo.py`: change the signature to `def setup_root_file_handler(log_dir, level=logging.DEBUG, filename="globalfit_run.log"):` and use `filepath = os.path.join(log_dir, filename)`.

`run.py`, module-level helpers (place after the imports, near `_rss_mb`):

```python
def _leaf_moves(moves):
    """Flatten combine moves (recursively; ``(move, weight)`` tuples unwrapped).

    An EMPTY combine (no sub-moves) is itself returned as a leaf so a caller can
    still name it.
    """
    out = []
    for move in moves:
        if isinstance(move, tuple):
            move = move[0]
        inner = getattr(move, "moves", None) if isinstance(move, CombineMove) else None
        if inner:
            out.extend(_leaf_moves(inner))
        else:
            out.append(move)
    return out


def _move_label(move):
    return getattr(move, "gf_move_name", None) or getattr(move, "name", None) or type(move).__name__


def _fanout_unready_moves(moves):
    """``(unready, head_only)`` leaf-move names for a multi-rank run.

    ``unready``: ``GlobalFitMove`` leaves whose class still has the default
    ``gf_serve`` (they must be ported before running with several compute
    ranks). ``head_only``: leaves that are not ``GlobalFitMove`` at all (plain
    eryn moves): they run on the head against its walker block only, which is
    logged as a warning. ``FunctionMove`` is head-only by design and appears in
    neither list.
    """
    unready, head_only = [], []
    for move in _leaf_moves(moves):
        if isinstance(move, FunctionMove) or getattr(move, "gf_head_only", False):
            continue
        if not isinstance(move, GlobalFitMove):
            head_only.append(_move_label(move))
            continue
        if type(move).gf_serve is GlobalFitMove.gf_serve:
            unready.append(_move_label(move))
    return unready, head_only


def _rank_log_filenames(layout, rank):
    """(root lisatools log, GlobalFit log) for this rank; the head keeps today's names."""
    if int(rank) == int(layout.head_rank):
        return "globalfit_run.log", "global_fit.log"
    return f"globalfit_run.rank{int(rank)}.log", f"global_fit.rank{int(rank)}.log"
```

Imports for these helpers in `run.py`: `from eryn.moves import CombineMove` (`GFCombineMove` subclasses it and `moves/globalfitmove.py` already imports it), `from .moves.functionmove import FunctionMove` (`FunctionMove(Move, GlobalFitMove, ErynMove)` is its own runtime: `setup` returns `None`), `from .moves.globalfitmove import GlobalFitMove`. Check `run.py` for an import cycle with `moves/functionmove.py` (if `functionmove` imports from `run`, import `FunctionMove` lazily inside `_fanout_unready_moves`).

`GlobalFitSetup`: add class attributes right after the docstring, before `__init__`:

```python
    #: multi-rank runtime (set by ``communication.ranks.prepare_rank`` before the
    #: build and by ``GlobalFit`` at run time; ``None`` in single-process use)
    rank_layout = None
    fanout = None
    rank = None
    rank_device_mode = None
```

`GlobalFit.resolve_rank_roles`: keep as a compat wrapper with the same return shape, implemented via `communication.ranks.resolve_roles`: `head, saver, compute = resolve_roles(comm.Get_size(), main_rank); spares = [r for r in range(comm.Get_size()) if r not in compute and r != saver]; return head, saver, spares`. Update its docstring (spares are empty in the new layout; legacy layout re-creates them).

`GlobalFit.__init__` (replace lines 427-445):

```python
        self.comm = comm if comm is not None else MPI.COMM_SELF
        self.curr = curr
        self.rank = self.comm.Get_rank()
        self.nwalkers: int = self.curr.general_info.nwalkers
        self.ntemps: int = self.curr.general_info.ntemps
        self.all_ranks = list(range(self.comm.Get_size()))
        self.head_rank = self.curr.rank_info.head_rank
        # Layout: resolved before the build by prepare_rank (drivers /
        # StockGlobalFit.run). The late path covers fit.sample() and legacy
        # settings-file runs: single process, or CPU, or the legacy switch.
        layout = getattr(self.curr, "rank_layout", None)
        if layout is None:
            layout = build_layout(
                self.comm,
                self.nwalkers,
                list(self.curr.general_info.gpus or []),
                main_rank=int(self.curr.rank_info.main_rank),
            )
            if not layout.is_single() and self.curr.general_info.gpus:
                self.logger_early_warning = (
                    "multi-rank layout resolved AFTER the build: device pinning "
                    "did not run before the build allocated. Call "
                    "communication.ranks.prepare_rank(fit, comm) before fit.build()."
                )
            self.curr.rank_layout = layout
        self.layout = layout
        self.curr.rank = self.rank
        self.role = layout.role_of(self.rank)
        self.main_rank = layout.head_rank
        self.results_rank = layout.saver_rank
        self.compute_ranks = tuple(layout.compute_ranks)
        self.worker_ranks = tuple(layout.worker_ranks)
        self.ranks_to_give = [
            r for r in self.all_ranks if layout.role_of(r) == RankRole.SPARE
        ]
        self.used_ranks = list(self.compute_ranks)
        if self.results_rank != self.main_rank:
            self.used_ranks.append(self.results_rank)
        self.fanout_comm = layout.make_fanout_comm(self.comm) if not layout.is_single() else None
        if isinstance(self.comm, MPI.Comm) and self.comm.Get_size() > 1:
            install_mpi_abort_on_error(self.comm)
```

Then the logging block (replace 447-467):

```python
        level = logging.DEBUG
        name = "GlobalFit"
        self.verbose = bool(getattr(self.curr.general_info, "verbose", False))
        _progress = getattr(self.curr.general_info, "progress", None)
        self.progress = self.verbose if _progress is None else bool(_progress)
        artifacts_dir = self.curr.general_info.artifacts_file_dir
        root_log, gf_log = _rank_log_filenames(layout, self.rank)
        setup_root_file_handler(artifacts_dir, level=level, filename=root_log)
        self.logger = init_logger(
            filename=gf_log, level=level, name=name, log_dir=artifacts_dir,
            console=self.verbose,
        )
        if self.rank != self.main_rank:
            prefix_stdout(rank_tag(layout, self.rank))
        if getattr(self, "logger_early_warning", None):
            self.logger.warning(self.logger_early_warning)
        self.logger.info("%s\nthis rank: %s", layout.describe(), rank_tag(layout, self.rank))
        if self.rank == self.main_rank:
            dump_settings(self.curr.settings_dict, artifacts_dir)
```

Imports at the top of `run.py`: `from .communication.ranks import RankRole, build_layout, install_mpi_abort_on_error, prefix_stdout, rank_tag` and `from .moves.globalfitmove import FunctionMove, GlobalFitMove` (check for an existing import of `MoveBuildContext` from the same module and extend it).

`run_global_fit` (replace 2214-2275):

```python
    def run_global_fit(self):
        """Execute the run for this rank's role (head / compute / saver / legacy spare)."""
        backend_path = self.curr.general_info.main_file_path
        if self.role == RankRole.HEAD:
            self.prepare_main()
            self.sampler.run_mcmc(
                self.state, self.curr.general_info.num_iterations, thin_by=1,
                progress=self.progress, store=True,
            )
            if self.curr.general_info.submission_parent_folder is not None:
                self.logger.debug(
                    f"saving submission to {self.curr.general_info.submission_parent_folder}"
                )
                submission_writer = SubmissionWriter(
                    backend=self.run_backend, curr=self.curr, ess=20_000
                )
                submission_writer.write_submission(self.acs)
            logger.info("Residuals saved.")
            if self.fanout is not None:
                self.fanout.stop()
            if self.results_rank != self.main_rank:
                self.comm.send({"finish_run": True}, dest=self.results_rank)
        elif self.role == RankRole.SAVER:
            backend = GFHDFBackend(
                backend_path,
                sub_backend=self.engine_info.branch_backends,
                sub_state_bases=self.engine_info.branch_states,
            )
            plot_container = self.make_plot_container()
            self._release_rank_gpu_pool()
            save_to_backend_asynchronously_and_plot(
                backend, self.comm, self.main_rank,
                plot_container=plot_container, plot_iter=self._plot_iterations,
                backup_iter=self.curr.general_info.backup_iter,
            )
        elif self.role == RankRole.COMPUTE:
            self.prepare_compute()
            served = self.compute_service.serve()
            self._release_rank_gpu_pool()
            self.logger.info("compute rank %d served %d command(s); exiting.", self.rank, served)
        else:  # legacy SPARE: wait for the startup "stop" and exit
            self._release_rank_gpu_pool()
            info = self.comm.recv(source=self.main_rank)
            logger.info(f"Process {self.rank} finished ({info!r}).")
```

`_release_helper_gpu_pool` → rename to `_release_rank_gpu_pool` (keep the old name as an alias assignment `_release_helper_gpu_pool = _release_rank_gpu_pool` for any external caller) and iterate the rank's own devices:

```python
        devices = self.layout.local_gpus(self.rank) or []
        if not devices:
            return
        freed = 0
        for dev in devices:
            try:
                with cp.cuda.Device(int(dev)):
                    pool = cp.get_default_memory_pool()
                    freed += pool.total_bytes() - pool.used_bytes()
                    pool.free_all_blocks()
            except Exception:
                continue
        self.logger.info(
            "rank %d released ~%.2f GB of cached GPU pool blocks on device(s) %s.",
            self.rank, freed / 1e9, list(devices),
        )
```

(keep the `gc.collect()` / `import cupy` guard as today; drop the `getDeviceCount` loop).

In `prepare_main`, the legacy "stop the spare processes" loop (lines 2100-2103) becomes `for rank in self.ranks_to_give: self.comm.send("stop", dest=rank)` (empty in the new layout; identical behaviour in the legacy layout).

- [ ] **Step 4: Run to verify pass, plus the recipe/combine regression suites**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p2t2b.log python -m unittest tests.test_run_multirank_helpers tests.test_gf_combine_weighted tests.test_gf_combine_pe_rj_draw_one tests.test_stock_globalfit -v; tail -n 30 .wtenv/p2t2b.log`
Expected: all OK (`test_stock_globalfit` proves the pre-build fit still pickles/deepcopies with the new class attributes).

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/run.py src/lisatools/globalfit/loginfo.py tests/test_run_multirank_helpers.py
git commit -m "feat(run): rank roles from the walker-block layout, per-rank logs, role-dispatched run_global_fit

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `setup_acs(walker_block=)` and `_global_likelihood`

**Files:**
- Modify: `src/lisatools/globalfit/run.py` (`setup_acs` 1407-1684; the three `acs.likelihood` sites at 1826, 2057, 2173; `sample()` untouched)
- Test: `tests/test_run_multirank_helpers.py` (append), plus the gated smoke in Task 7

**Interfaces:**
- Produces: `setup_acs(self, state, rebuild_residuals=False, walker_block=None)`; module-level `_rebuild_state_view(state, walker_block)` returning `state` when `walker_block` is `None` else `slice_state(state, w0, w1, sub_states=[])`; `GlobalFit._global_likelihood(acs) -> np.ndarray (nwalkers,)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_multirank_helpers.py`:

```python
class RebuildViewTest(unittest.TestCase):
    def test_none_block_returns_the_same_state_object(self):
        from lisatools.globalfit.run import _rebuild_state_view
        from tests.test_gf_substate_roundtrip import make_state
        import numpy as np

        state = make_state(np.random.default_rng(3))
        self.assertIs(_rebuild_state_view(state, None), state)

    def test_block_returns_a_walker_slice_without_sub_states(self):
        from lisatools.globalfit.run import _rebuild_state_view
        from tests.test_gf_substate_roundtrip import make_state
        import numpy as np

        state = make_state(np.random.default_rng(3))
        part = _rebuild_state_view(state, (1, 3))
        self.assertEqual(part.branches["gb"].nwalkers, 2)
        np.testing.assert_array_equal(part.branches["gb"].coords, state.branches["gb"].coords[:, 1:3])
        self.assertTrue(all(v is None for v in part.sub_states.values()))
```

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p2t3.log python -m unittest tests.test_run_multirank_helpers -v; tail -n 25 .wtenv/p2t3.log`
Expected: `ImportError: cannot import name '_rebuild_state_view'`.

- [ ] **Step 3: Implement**

Module-level helper in `run.py`:

```python
def _rebuild_state_view(state, walker_block):
    """The state the residual rebuild reads: the full state, or this rank's walker slice."""
    if walker_block is None:
        return state
    from .communication.walkerslice import slice_state

    w0, w1 = walker_block
    return slice_state(state, w0, w1, sub_states=[])
```

`setup_acs` changes (keep everything else verbatim):

1. Signature: `def setup_acs(self, state, rebuild_residuals=False, walker_block=None):` and docstring line: "``walker_block``: ``(w0, w1)`` global walker range this rank owns; ``None`` builds every walker (single-process behaviour)."
2. After `general_info = self.curr.general_info`: 
   ```python
        w0, w1 = (0, self.nwalkers) if walker_block is None else (int(walker_block[0]), int(walker_block[1]))
        n_local = w1 - w0
        state_view = _rebuild_state_view(state, walker_block)
   ```
3. Walker→device map (1469-1476): split over `np.arange(w0, w1)` instead of `np.arange(self.nwalkers)` so the map is keyed by GLOBAL walker id but partitions only this rank's block across its device list (one device now: every local walker maps to `gpus[0]`).
4. Build loop (1545-1552): `for i, w in enumerate(range(w0, w1)):` with `_build_walker_ac(w)` unchanged (global `w` for the `walker_{w}` sensitivity name and the psd/galfor/sgwb rows read from the FULL `state`); log `n_local` instead of `self.nwalkers`; progress check on `i`.
5. Divisibility warning (1555-1563): use `n_local` in place of `self.nwalkers`.
6. Rebuild loop (1589-1631): `for i, ac in enumerate(acs.flatten()):` reading `state_view.branches_inds[name][0, i]` / `state_view.branches_coords[name][0, i]`, and `device_context(xp, _walker_device.get(w0 + i))`.
7. Bulk fallback (1651-1653): `get_templates(state_view, source_info, self.curr.general_info)`.

`_global_likelihood`:

```python
    def _global_likelihood(self, acs):
        """The (nwalkers,) likelihood vector: local rows, allgathered across compute ranks."""
        local = np.asarray(asnumpy(acs.likelihood(complex=False)))
        fanout = getattr(self, "fanout", None)
        if fanout is None or self.layout.is_single():
            return local
        return fanout.allgather_walker_vector(local)
```

Replace the first two sites: line 1826 `state.log_like[:] = self._global_likelihood(acs)` and line 2057 the same. The THIRD site (line 2173, after the engine is built) runs on the head only, so it must NOT be a collective: compute ranks have already left the setup phase. Keep single-mode identity and avoid the deadlock with

```python
        if self.layout.is_single():
            state.log_like[:] = acs.likelihood(complex=False)[None, :]
        else:
            # the residual has not changed since the post-setup gather (line 2057);
            # a fresh allgather here would have no partner on the compute ranks
            state.log_like[:] = self._ll_after_setup[None, :]
```

where `_build_acs_and_recipe` (Task 4) stores `self._ll_after_setup = np.array(state.log_like[0], copy=True)` right after the second gather. `sample()` (2314+) keeps its own direct call: it is single-process by construction. Collective count per role in the setup phase is then exactly: one `bcast`, two `allgather` (head and compute identical).

Note for the implementer: `asnumpy` is already imported in `run.py` (used at 1835); `acs.likelihood(complex=False)` returns a 1-D array of `acs_total_entries`.

- [ ] **Step 4: Run to verify pass**

Same command. Expected: `Ran 5 tests ... OK`.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/run.py tests/test_run_multirank_helpers.py
git commit -m "feat(run): setup_acs builds one walker block; likelihoods gathered across compute ranks

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: `prepare_main` split, `prepare_compute`, fan-out wiring, seeds, readiness guard

**Files:**
- Modify: `src/lisatools/globalfit/run.py` (`prepare_main` 1691-2212), `src/lisatools/globalfit/recipe.py` (`Recipe` class: `fanout` attribute + hooks in `setup_first_recipe_step` 563-581 and `__call__` 588-611)
- Test: `tests/test_run_multirank_helpers.py` (append), gated smoke in Task 7

**Interfaces:**
- Produces on `GlobalFit`: `_collect_priors_periodic() -> (priors, periodic)`; `_attach_walker_supplemental(state)`; `_build_acs_and_recipe(state, priors) -> (acs, like_mix)` (shared by both roles: `setup_acs(walker_block=self.layout.block_of(self.rank), rebuild_residuals=True)`, `_global_likelihood`, recipe `_init_runtime` + `setup_function`, `_global_likelihood` again, readiness guard); `_seed_rank_streams()`; `prepare_main()` (head); `prepare_compute()` (compute ranks) setting `self.compute_service`; `self.fanout` (a `WalkerFanout` on every compute rank incl. the head, `None` in single mode... see below); `self.curr.fanout`. Module-level `_stage_kind_of(step)`.
- `Recipe.fanout = None` attribute; `Recipe.setup_first_recipe_step` / `Recipe.__call__` call `self.fanout.enter_stage(name, kind)` after `setup_run` and `self.fanout.note_iteration(iteration)` when `self.fanout is not None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_multirank_helpers.py`:

```python
class RecipeFanoutHooksTest(unittest.TestCase):
    def test_recipe_forwards_stage_and_iteration_to_the_fanout(self):
        from lisatools.globalfit.recipe import Recipe

        calls = []

        class _Fanout:
            def enter_stage(self, name, kind):
                calls.append(("stage", name, kind))

            def note_iteration(self, i):
                calls.append(("it", i))

        class _Step:
            def __init__(self, kind):
                self.moves = [type("M", (), {"gf_stage_kind": kind})()]
                self.stops = [False, True]

            def setup_run(self, iteration, last_sample, sampler):
                calls.append(("setup_run", iteration))

            def stopping_function(self, iteration, last_sample, sampler):
                return self.stops.pop(0)

        class _Backend:
            def completed_recipe_step(self, name):
                calls.append(("done", name))

        recipe = Recipe()
        recipe._init_runtime()
        recipe.recipe = [
            {"name": "a", "adjust": _Step("search"), "status": False},
            {"name": "b", "adjust": _Step("pe"), "status": False},
        ]
        recipe.backend = _Backend()
        recipe.fanout = _Fanout()
        recipe.setup_first_recipe_step(0, None, None)
        self.assertEqual(calls[-1], ("stage", "a", "search"))
        self.assertFalse(recipe(1, None, None))  # step a not done yet
        self.assertIn(("it", 1), calls)
        self.assertFalse(recipe(2, None, None))  # a done -> b set up
        self.assertEqual(calls[-1], ("stage", "b", "pe"))
```

(Read `Recipe._init_runtime` and how `self.recipe` / `_current_iter` / `next(self)` work at recipe.py:203 and 563-611 before finalising this test; adapt the fixture so it drives the real `Recipe` object without a materialized fit. If `Recipe()` needs stages to construct, build the minimal declarative form and monkeypatch the materialized `recipe` list as shown.)

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p2t4.log python -m unittest tests.test_run_multirank_helpers -v; tail -n 25 .wtenv/p2t4.log`
Expected: FAIL (`Recipe` has no attribute `fanout`, or no `("stage", ...)` calls).

- [ ] **Step 3: Implement**

`recipe.py`: on `Recipe`, add `fanout = None` as a class attribute (documented: "multi-rank fan-out; the runner sets it before the first step"). Add a module-level helper:

```python
def _stage_kind_of(step):
    moves = getattr(step, "moves", None) or []
    first = moves[0][0] if moves and isinstance(moves[0], tuple) else (moves[0] if moves else None)
    return getattr(first, "gf_stage_kind", None)
```

In `setup_first_recipe_step`, after `self._current_recipe_step["adjust"].setup_run(...)`:

```python
        if self.fanout is not None:
            self.fanout.enter_stage(
                self._current_recipe_step["name"], _stage_kind_of(self._current_recipe_step["adjust"])
            )
```

In `__call__`, first line: `if self.fanout is not None: self.fanout.note_iteration(iteration)`; and after the `setup_run` inside the `if stop_here:` block, the same `enter_stage` call.

`run.py`, split `prepare_main`:

```python
    def _collect_priors_periodic(self):
        """priors/periodic dicts from source_info (moved verbatim from prepare_main)."""
        ...  # lines 1714-1746 verbatim, then: return priors, periodic

    def _attach_walker_supplemental(self, state):
        """supplemental['walker_inds'] = tile(arange(nwalkers)) (moved verbatim)."""
        ...  # lines 1751-1756 verbatim

    def _make_fanout(self, model):
        """A WalkerFanout for this compute rank (None in single mode)."""
        if self.layout.is_single():
            return None
        from .communication.fanout import WalkerFanout

        fanout = WalkerFanout(self.fanout_comm, self.layout, self.rank, model=model, logger=self.logger)
        fanout.clock["seed_base"] = getattr(self.curr.general_info, "random_seed", None)
        return fanout

    def _seed_rank_streams(self):
        """Distinct, deterministic RNG streams per compute rank (multi-rank only)."""
        if self.layout.is_single():
            return None
        from .communication.ranks import derive_rank_seed

        seed = derive_rank_seed(
            int(getattr(self.curr.general_info, "random_seed", 0) or 0), self.layout, self.rank
        )
        np.random.seed(seed)
        if _xp_is_cupy and self.curr.general_info.gpus:
            xp.random.seed(seed)
        self.logger.info("rank %d RNG streams seeded with %d", self.rank, seed)
        return seed

    def _build_acs_and_recipe(self, state, priors):
        """Shared by head and compute ranks: this rank's ACA block, the recipe, the likelihood."""
        acs = self.setup_acs(
            state, rebuild_residuals=True, walker_block=self.layout.block_of(self.rank)
        )
        self.logger.debug("acs setup done")
        state.log_like[:] = self._global_likelihood(acs)
        logger.info(f"initial log likelihood: {state.log_like[0]}")
        ...  # the non-finite diagnostic block (lines 1835-1853) verbatim
        like_mix = BasicResidualacsLikelihood(acs)
        recipe = self.curr.source_metadata.get("recipe")
        if recipe is None:
            recipe = Recipe()
        recipe._init_runtime()
        self.recipe = recipe
        self.curr.fanout = self.fanout
        self.recipe.fanout = self.fanout
        self.curr.settings_dict.setup_function(
            self.recipe, self.engine_info, self.curr, acs, priors, state
        )
        state.log_like[:] = self._global_likelihood(acs)
        logger.info(f"initial log likelihood (after recipe setup): {state.log_like[0]}")
        ...  # the GB_LAYER_CHI2 diagnostic block verbatim
        self._ll_after_setup = np.array(state.log_like[0], copy=True)
        if not self.layout.is_single():
            all_moves = [m for step in self.recipe.recipe for m in step["adjust"].moves]
            unready, head_only = _fanout_unready_moves(all_moves)
            if unready:
                raise RuntimeError(
                    f"multi-rank run with n_compute={self.layout.n_compute} but these moves do "
                    f"not serve fan-out commands yet: {unready}. Run with one compute rank "
                    "(or GF_LEGACY_RANK_LAYOUT=1) until they are ported."
                )
            if head_only:
                self.logger.warning(
                    "multi-rank run: these moves are not GlobalFitMoves and run on the HEAD "
                    "against its walker block only: %s", head_only,
                )
        return acs, like_mix
```

(`self.recipe.recipe` is the materialized step list: dicts with an `"adjust"` `RecipeStep` whose `.moves` is `[combined]` — confirm against `Recipe._init_runtime` / `Recipe.setup` at recipe.py:203 and 482-511 before relying on the key names.)

`prepare_main` (head), in order: `_collect_priors_periodic` → `state = self.load_info(priors)` → `if not single: state = self.fanout_comm.bcast(state, root=self.layout.fanout_rank(self.main_rank))` (the head passes its state, receives the same object back) → `_attach_walker_supplemental(state)` → **fan-out before setup** so setup-time builders can read `ctx.fanout`: `self.fanout = self._make_fanout(model=None)` (the model is attached after the engine exists: `self.fanout.model = GlobalFitInfo(acs, map, sampler_mix._random)`) → `acs, like_mix = self._build_acs_and_recipe(state, priors)` → backend construction + reset + domain settings + noise identity + midit arm (lines 1857-2022 verbatim) → legacy spare stop loop → engine construction (2105-2168 verbatim) → `state.log_like[:] = self._global_likelihood(acs)[None, :]` → `state.log_prior = zeros_like` → `self.recipe.setup_first_recipe_step(...)` → `self._seed_rank_streams()` → `if self.fanout is not None: self.fanout.model = sampler_mix.get_model(); self.fanout.ping()` → submission plotter → publish `self.sampler/state/priors/acs/run_backend/live_ctx` (the `MoveBuildContext` now also gets `state_local=slice_state(state, *self.layout.block_of(self.rank), sub_states=[])` when not single) → midit self-test.

`prepare_compute` (compute ranks):

```python
    def prepare_compute(self):
        """Build what a computation rank needs and hand it to the command loop."""
        priors, periodic = self._collect_priors_periodic()
        state = self.fanout_comm.bcast(None, root=self.layout.fanout_rank(self.main_rank))
        self._attach_walker_supplemental(state)
        self.fanout = self._make_fanout(model=None)
        acs, like_mix = self._build_acs_and_recipe(state, priors)
        from eryn.moves import StretchMove
        from eryn.utils import PeriodicContainer

        periodic_key_order = {key: value.key_order for key, value in priors.items()}
        if periodic and not isinstance(periodic, PeriodicContainer):
            periodic = PeriodicContainer(periodic, key_order=periodic_key_order)
        engine = GlobalFitEngine(
            acs, self.nwalkers, self.engine_info.ndims, like_mix, priors,
            tempering_kwargs={"ntemps": self.ntemps},
            nbranches=len(self.engine_info.branch_names),
            nleaves_max=self.engine_info.nleaves_max,
            nleaves_min=self.engine_info.nleaves_min,
            moves=StretchMove(live_dangerously=True), rj_moves=None, kwargs=None,
            backend=None, vectorize=True, periodic=periodic,
            branch_names=self.engine_info.branch_names,
            plot_generator=None, plot_iterations=-1,
            provide_groups=True, provide_supplemental=True, track_moves=False,
            stopping_fn=None,
        )
        for step in self.recipe.recipe:
            step["adjust"].setup_run(0, state, engine)
        seed = self._seed_rank_streams()
        rank_rng = np.random.RandomState(seed)
        model = GlobalFitInfo(acs, map, rank_rng)
        self.fanout.model = model
        registry = {}
        for move in _leaf_moves([m for step in self.recipe.recipe for m in step["adjust"].moves]):
            name = getattr(move, "gf_move_name", None)
            if name is not None:
                registry[name] = move
        from .communication.fanout import ComputeService

        self.compute_service = ComputeService(
            self.fanout_comm, self.layout, self.rank, registry=registry, model=model,
            builtins={
                "likelihood": lambda payload, clock, model: np.asarray(
                    asnumpy(model.analysis_container_arr.likelihood(complex=False))
                ),
            },
            logger=self.logger,
        )
        self.sampler = engine
        self.state = state
        self.priors = priors
        self.acs = acs
```

Imports: `GlobalFitInfo` from `.engine` (check the existing import list in run.py). `RJRecipeStep.setup_run` reads `sampler.backend.iteration` — the in-memory eryn backend exposes `iteration`; if it raises before the first `reset`, call `engine.backend.reset(...)`-free alternative: wrap the stamping loop so a missing backend attribute logs and continues (the stamp only sets `periodic`/`temperature_control` on moves).

The head's ping happens after the compute ranks enter `serve()`: make sure `prepare_main` calls `self.fanout.ping()` only after `setup_first_recipe_step`, and `prepare_compute` reaches `serve()` without further collectives after `_build_acs_and_recipe`'s second `_global_likelihood` (count the collectives: bcast, allgather, allgather on both roles — identical sequence).

- [ ] **Step 4: Run to verify pass**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p2t4b.log python -m unittest tests.test_run_multirank_helpers tests.test_gf_combine_weighted tests.test_gf_combine_pe_rj_draw_one -v; tail -n 30 .wtenv/p2t4b.log`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/run.py src/lisatools/globalfit/recipe.py tests/test_run_multirank_helpers.py
git commit -m "feat(run): prepare_main/prepare_compute over the walker-block layout with fan-out wiring and seeds

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Recipe builders at the local block; block-gated setup-time subtraction

**Files:**
- Modify: `src/lisatools/globalfit/recipe.py` (`build_noise_moves` 2119-2212; GB init block 2480-2579; VGB init block 3666-3736; neighbour subtraction 1917-1930; `SingleSourcePEBuilder.build` 3987-4102)
- Test: `tests/test_recipe_local_block.py`

**Interfaces:**
- Produces: module-level `_local_walker_block(curr, acs) -> (w0, w1)` (from `curr.rank_layout.block_of(curr.rank)` when both exist, else `(0, acs.acs_total_entries)`), `_local_nwalkers(acs) -> int` (`acs.acs_total_entries`); moves built with `nwalkers_local`; `move.fanout_branches` stamped (`[self.branch_name]` for addremove; the sampled noise branches for PSD).

- [ ] **Step 1: Write the failing test**

Create `tests/test_recipe_local_block.py`:

```python
"""Recipe builders size moves at the LOCAL walker block and read the block from the layout."""

import unittest

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.ranks import build_layout
from lisatools.globalfit.recipe import _local_nwalkers, _local_walker_block


class _Acs:
    def __init__(self, n):
        self.acs_total_entries = n


class _Curr:
    def __init__(self, layout=None, rank=None):
        self.rank_layout = layout
        self.rank = rank


class LocalBlockTest(unittest.TestCase):
    def test_no_layout_means_whole_range(self):
        self.assertEqual(_local_walker_block(_Curr(), _Acs(4)), (0, 4))
        self.assertEqual(_local_nwalkers(_Acs(4)), 4)

    def test_layout_gives_this_ranks_block(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        self.assertEqual(_local_walker_block(_Curr(lay, 0), _Acs(2)), (0, 2))
        self.assertEqual(_local_walker_block(_Curr(lay, 1), _Acs(2)), (2, 4))

    def test_block_must_match_the_aca_size(self):
        lay = FakeWorld(3).run(lambda r, c: build_layout(c, 4, [0, 1], legacy=False))[0]
        with self.assertRaises(ValueError):
            _local_walker_block(_Curr(lay, 1), _Acs(4))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p2t5.log python -m unittest tests.test_recipe_local_block -v; tail -n 20 .wtenv/p2t5.log`
Expected: `ImportError: cannot import name '_local_nwalkers'`.

- [ ] **Step 3: Implement**

`recipe.py` helpers (near `get_shared_dcga`):

```python
def _local_nwalkers(acs) -> int:
    """Walkers this rank's ACA holds (== the global count in a single-process run)."""
    return int(acs.acs_total_entries)


def _local_walker_block(curr, acs):
    """(w0, w1) of the global walkers this rank's ACA rows correspond to."""
    layout = getattr(curr, "rank_layout", None)
    rank = getattr(curr, "rank", None)
    n_local = _local_nwalkers(acs)
    if layout is None or rank is None:
        return 0, n_local
    w0, w1 = layout.block_of(rank)
    if w1 - w0 != n_local:
        raise ValueError(
            f"rank {rank}: layout block [{w0}, {w1}) has {w1 - w0} walkers but the ACA holds "
            f"{n_local} rows."
        )
    return int(w0), int(w1)
```

Then, site by site:

1. `build_noise_moves` (2119): `nwalkers: int = _local_nwalkers(acs)` (used by `TemperatureControl(effective_ndim, nwalkers, ...)` and both `accepted` arrays); after constructing the two moves: `search_move.fanout_branches = list(sampled_branches); pe_move.fanout_branches = list(sampled_branches)`.
2. `SingleSourcePEBuilder.build` (3990): `nwalkers = _local_nwalkers(acs)`; after `move = self.move_class(...)`: `move.fanout_branches = [self.branch_name]`. (`coords_shape`, `TemperatureControl(nwalkers=...)` inside the move, and `move.accepted` all follow.)
3. GB init subtraction (2480-2520): `w0, w1 = _local_walker_block(curr, acs)`; `inds_loc = state.branches["gb"].inds[0, w0:w1]`; `coords_out_gb = state.branches["gb"].coords[0, w0:w1][inds_loc]`; `walker_vals = np.tile(np.arange(w1 - w0), (nleaves_max_gb, 1)).transpose((1, 0))[inds_loc]`; the `.sum() > 0` gate uses `inds_loc`. Everything downstream (`data_index`, `factors`, `N_vals`, both domain branches) is unchanged.
4. GB WDM/FD path (2530-2579): unchanged (already reads `acs.linear_data_arr` / gathers the local ACA).
5. VGB init subtraction (3666-3694): same pattern with `state.branches["vgb"]` and `nleaves_max_vgb`; `leaf_inds = np.tile(np.arange(nleaves_max_vgb), (w1 - w0, 1))[inds0]`.
6. Neighbour subtraction (1917-1920): `nwalkers = _local_nwalkers(acs)` (the tiling is over ACA rows; no state read).

Search the file for any other `general_info.nwalkers` / `curr.general_info.nwalkers` used to SIZE a per-walker ACA operation or a move; leave the ones that describe the engine (e.g. `Stage.setup`'s `combined.accepted = np.zeros((ctx.ntemps, ctx.nwalkers))` stays global). List every site you changed in the report.

- [ ] **Step 4: Run to verify pass + regression**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p2t5b.log python -m unittest tests.test_recipe_local_block tests.test_gf_combine_weighted tests.test_gf_combine_pe_rj_draw_one tests.test_stock_globalfit -v; tail -n 30 .wtenv/p2t5b.log`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/recipe.py tests/test_recipe_local_block.py
git commit -m "feat(recipe): size moves at the local walker block; block-gate GB/VGB setup-time subtraction

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: `StockGlobalFit.run` prepares the rank before the build

**Files:**
- Modify: `src/lisatools/globalfit/stock/base.py` (`run` 837-851)
- Test: `tests/test_stock_globalfit.py` (append one test)

**Interfaces:**
- Produces: `StockGlobalFit.run(comm=None, **run_kwargs)` calls `communication.ranks.prepare_rank(self, comm)` before `self.build()` when `comm.Get_size() > 1` and the fit is not yet built (idempotent; a pre-prepared fit is left alone); `fit.sample()` untouched.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stock_globalfit.py` (read its imports and fixture style first; it constructs cheap unbuilt fits):

```python
class RunPreparesRankTest(unittest.TestCase):
    def test_run_calls_prepare_rank_before_build(self):
        from unittest import mock

        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.stock import erebor

        fit = erebor.blank(nwalkers=4, ntemps=2)
        comm = FakeWorld(2).comm(0)
        order = []

        def fake_prepare(fit_, comm_, **kw):
            order.append("prepare")
            fit_.rank_layout = "LAYOUT"
            return "LAYOUT"

        with mock.patch("lisatools.globalfit.communication.ranks.prepare_rank", fake_prepare), \
             mock.patch.object(type(fit), "build", lambda self_, *a, **k: order.append("build")), \
             mock.patch("lisatools.globalfit.run.GlobalFit") as gf:
            gf.return_value.run_global_fit.return_value = "ran"
            self.assertEqual(fit.run(comm=comm), "ran")
        self.assertEqual(order, ["prepare", "build"])
        gf.assert_called_once()
```

(`base.run` imports `GlobalFit` lazily inside the method as `from ..run import GlobalFit`; patch the attribute on `lisatools.globalfit.run` as shown. Adjust the `prepare_rank` patch target to the name `base.run` actually imports.)

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p2t6.log python -m unittest tests.test_stock_globalfit.RunPreparesRankTest -v; tail -n 20 .wtenv/p2t6.log`
Expected: FAIL (`order == ["build"]`).

- [ ] **Step 3: Implement**

`stock/base.py` `run`:

```python
    def run(self, comm=None, **run_kwargs):
        """One-shot pipeline: prepare this rank -> build (if needed) -> GlobalFit -> run_global_fit."""
        if comm is None:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
        if int(comm.Get_size()) > 1 and getattr(self, "rank_layout", None) is None and not self.built:
            from ..communication.ranks import prepare_rank

            prepare_rank(self, comm)
        self.build()
        from ..run import GlobalFit

        gf = GlobalFit(self, comm)
        return gf.run_global_fit(**run_kwargs)
```

- [ ] **Step 4: Run to verify pass**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p2t6b.log python -m unittest tests.test_stock_globalfit -v; tail -n 20 .wtenv/p2t6b.log`
Expected: all OK.

- [ ] **Step 5: Commit**

```bash
git add src/lisatools/globalfit/stock/base.py tests/test_stock_globalfit.py
git commit -m "feat(stock): StockGlobalFit.run prepares the rank layout and device before the build

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: End-to-end fake-communicator smoke on `erebor.blank`

**Files:**
- Create: `tests/test_multirank_blank_smoke.py` (gated `RUN_GF_SMOKE=1`)

**Interfaces:**
- Consumes everything above. Two scenarios: `FakeWorld(1)` (single: today's path) and `FakeWorld(2)` on CPU (compute ranks 0 and 1, saver aliased to the head so no concurrent HDF writer thread).

- [ ] **Step 1: Write the test**

```python
"""Multi-rank control plane end to end on the blank synthetic fit (fake communicator)."""

import os
import shutil
import tempfile
import unittest

import numpy as np

RUN_GF_SMOKE = os.environ.get("RUN_GF_SMOKE", "") not in ("", "0")


def _count_move(model, state):
    _count_move.calls += 1
    return state, None


_count_move.calls = 0


@unittest.skipUnless(RUN_GF_SMOKE, "set RUN_GF_SMOKE=1 to run the multi-rank blank smoke")
class MultiRankBlankSmokeTest(unittest.TestCase):
    def setUp(self):
        _count_move.calls = 0
        self.tmpdir = tempfile.mkdtemp(prefix="gf_multirank_smoke_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_fit(self, subdir):
        from eryn.prior import uniform_dist

        from lisatools.globalfit.stock import erebor

        fit = erebor.blank(
            nwalkers=4, ntemps=2, file_store_dir=os.path.join(self.tmpdir, subdir),
            make_diagnostic_plots=False,
        )
        fit.general.num_iterations = 3
        fit.add_branch(
            "line", ndim=2,
            priors={0: uniform_dist(0.0, 1.0), 1: uniform_dist(0.0, 1.0)},
            moves=[_count_move],
        )
        return fit

    def _run_world(self, size):
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.ranks import prepare_rank
        from lisatools.globalfit.run import GlobalFit

        def fn(rank, comm):
            fit = self._make_fit(f"n{size}")
            layout = prepare_rank(fit, comm)
            fit.build()
            gf = GlobalFit(fit, comm)
            gf.run_global_fit()
            out = {
                "role": layout.role_of(rank).value,
                "block": layout.block_of(rank),
                "acs_rows": int(gf.acs.acs_total_entries),
                "nwalkers_state": int(gf.state.branches["line"].nwalkers),
            }
            if hasattr(gf, "compute_service"):
                out["served"] = gf.compute_service_served
            return out

        return FakeWorld(size, timeout=600.0).run(fn)

    def test_single_rank_path(self):
        out = self._run_world(1)
        self.assertEqual(out[0]["role"], "head")
        self.assertEqual(out[0]["acs_rows"], 4)
        self.assertGreaterEqual(_count_move.calls, 3)

    def test_two_compute_ranks_share_the_walkers(self):
        from lisatools.globalfit.hdfbackend import GFHDFBackend

        out = self._run_world(2)
        self.assertEqual(out[0]["role"], "head")
        self.assertEqual(out[1]["role"], "compute")
        self.assertEqual(out[0]["block"], (0, 2))
        self.assertEqual(out[1]["block"], (2, 4))
        self.assertEqual(out[0]["acs_rows"], 2)
        self.assertEqual(out[1]["acs_rows"], 2)
        self.assertEqual(out[0]["nwalkers_state"], 4)
        self.assertGreaterEqual(out[1]["served"], 1)  # the startup ping
        store = os.path.join(self.tmpdir, "n2")
        h5 = [f for f in os.listdir(store) if f.endswith(".h5")]
        self.assertTrue(h5)
        reader = GFHDFBackend(os.path.join(store, h5[0]))
        self.assertGreaterEqual(reader.iteration, 1)
```

(Have `run_global_fit`'s COMPUTE branch store the served count on `self.compute_service_served` so the smoke can read it. Function moves are head-only, so `_count_move.calls >= 3` is asserted only in the single-rank case; in the two-rank case the head still runs them.)

- [ ] **Step 2: Run the smoke**

Run: `RUN_GF_SMOKE=1 .wtenv/wt_run.sh $PWD/src .wtenv/p2t7.log python -m unittest tests.test_multirank_blank_smoke -v; tail -n 60 .wtenv/p2t7.log`
Expected: `Ran 2 tests ... OK`. If the two-rank case hangs or fails, the log shows which collective mismatched (`bcast` / `allgather` counts differ between `prepare_main` and `prepare_compute`) or which move the readiness guard named; fix in the task that owns it and rerun. Allow up to 10 minutes.

- [ ] **Step 3: Run the existing sample smoke to prove single-process `fit.sample()` is unchanged**

Run: `RUN_GF_SMOKE=1 .wtenv/wt_run.sh $PWD/src .wtenv/p2t7b.log python -m unittest tests.test_globalfit_sample -v; tail -n 25 .wtenv/p2t7b.log`
Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_multirank_blank_smoke.py src/lisatools/globalfit/run.py
git commit -m "test(run): multi-rank blank smoke over the fake communicator (1 and 2 compute ranks)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 9: Drivers route through `prepare_rank` and the layout roles (execute BEFORE Task 8)

Added after the Task 2 review: both launch scripts still call `GlobalFit.resolve_rank_roles` and wait for a `"stop"` message that the new layout never sends, so a multi-rank launch would deadlock until this lands.

**Files:**
- Modify: `scripts/run_global.py` (stock path lines 130-141; legacy `-sfp` path 143-165; the final `GlobalFit(...)` at 167-168), `scripts/fstat_proposal/run_combined_staged.py` (lines 949-952 build/run; the completion print 965-984; the local `_install_mpi_abort_on_error` 988-1050 and its `__main__` use), `src/lisatools/globalfit/run.py` (`GlobalFit.resolve_rank_roles` docstring)
- Test: `tests/test_driver_scripts_layout.py` (create)

**Interfaces:**
- Consumes: `communication.ranks.prepare_rank(fit, comm)`, `install_mpi_abort_on_error(comm)`, `WalkerBlockLayout.role_of`, `RankRole`.
- Produces: launch scripts that never call `resolve_rank_roles` and never block on a startup `"stop"`; every rank builds; legacy settings files (`-sfp`) run under `GF_LEGACY_RANK_LAYOUT=1` unless the user set it explicitly.

- [ ] **Step 1: Write the failing test**

Create `tests/test_driver_scripts_layout.py`:

```python
"""The launch scripts route rank roles through the walker-block layout, not resolve_rank_roles."""

import os
import py_compile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = [
    os.path.join(ROOT, "scripts", "run_global.py"),
    os.path.join(ROOT, "scripts", "fstat_proposal", "run_combined_staged.py"),
]


class DriverScriptsTest(unittest.TestCase):
    def test_scripts_compile(self):
        for path in SCRIPTS:
            py_compile.compile(path, doraise=True)

    def test_no_resolve_rank_roles_and_no_startup_stop_wait(self):
        for path in SCRIPTS:
            with open(path) as fh:
                text = fh.read()
            self.assertNotIn("resolve_rank_roles", text, path)
            self.assertNotIn('recv(source=_main)', text, path)
            self.assertIn("prepare_rank", text, path)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p2t9.log python -m unittest tests.test_driver_scripts_layout -v; tail -n 20 .wtenv/p2t9.log`
Expected: `test_no_resolve_rank_roles_and_no_startup_stop_wait` FAILS (both scripts still reference `resolve_rank_roles`).

- [ ] **Step 3: Implement**

`scripts/run_global.py`, stock path: replace lines 130-141 with

```python
        # Every rank builds: in the walker-block layout the head and the
        # computation ranks each own a block of walkers, and the saver needs
        # the backend spec. prepare_rank resolves this rank's role and pins
        # its device BEFORE the build allocates on it (no-op at -n 1).
        from lisatools.globalfit.communication.ranks import prepare_rank

        prepare_rank(fit, MPI.COMM_WORLD)
        curr_info = fit.build()
```

Legacy `-sfp` path (143-165): before `curr_info = settings_function()` add

```python
        # Settings-file runs predate the walker-block layout: keep today's
        # roles (one sampling rank, stopped spares) unless the user opted in.
        if os.environ.setdefault("GF_LEGACY_RANK_LAYOUT", "1") == "1":
            print("[legacy settings file] GF_LEGACY_RANK_LAYOUT=1: single-compute layout.",
                  flush=True)
```

and at the top of `__main__`, install the library abort hook: `from lisatools.globalfit.communication.ranks import install_mpi_abort_on_error; install_mpi_abort_on_error(MPI.COMM_WORLD)`. Update the `--help` epilog lines (76-84) that describe the np layout: "np 1: single rank; np 2 on 1 GPU: head + saver (warns); np 2 on 2 GPUs: head + compute; np >= 3: head + compute ranks + saver (highest rank)". Remove the now-unused `GlobalFit.resolve_rank_roles` import if it was imported by name.

`scripts/fstat_proposal/run_combined_staged.py`: at 949-952

```python
    print("[combined] fit.build() ...", flush=True)
    from mpi4py import MPI
    from lisatools.globalfit.communication.ranks import prepare_rank

    layout = prepare_rank(fit, MPI.COMM_WORLD)
    fit.build()
    print("[combined] running", flush=True)
    fit.run()
```

Completion print (965-984): derive the role from `layout.role_of(MPI.COMM_WORLD.Get_rank())` and print `RUN COMPLETE` only for `RankRole.HEAD`, otherwise `[combined] rank {r} ({role.value}) exiting; the run continues on the head.` Replace the local `_install_mpi_abort_on_error()` with `from lisatools.globalfit.communication.ranks import install_mpi_abort_on_error` in `__main__` (`install_mpi_abort_on_error(MPI.COMM_WORLD)` returns the comm or `None` like the local one did); delete the local function.

`run.py` `GlobalFit.resolve_rank_roles` docstring: state that it is a legacy-compat wrapper over `communication.ranks.resolve_roles` (head, saver, and an always-empty spare list), that launchers must use `prepare_rank` + `layout.role_of`, and that legacy spares exist only in the `GF_LEGACY_RANK_LAYOUT=1` layout (`layout.role_of(r) == RankRole.SPARE`).

- [ ] **Step 4: Run to verify pass**

Same command. Expected: `Ran 2 tests ... OK`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_global.py scripts/fstat_proposal/run_combined_staged.py src/lisatools/globalfit/run.py tests/test_driver_scripts_layout.py
git commit -m "feat(scripts): launchers prepare the rank layout before the build; no startup stop wait

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Whole-plan verification (execute AFTER Task 9; add `tests.test_driver_scripts_layout` to the Step 1 module list)

- [ ] **Step 1: Every touched suite, one process**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/p2_all.log python -m unittest tests.test_rank_layout tests.test_run_multirank_helpers tests.test_recipe_local_block tests.test_stock_globalfit tests.test_gf_combine_weighted tests.test_gf_combine_pe_rj_draw_one tests.test_walkerslice_roundtrip tests.test_fanout_fakecomm tests.test_fanout_passthrough tests.test_fakecomm tests.test_move_build_context_layout tests.test_gf_substate_roundtrip tests.test_module_substate_reseed tests.test_coarse_nopsd_gate -v; tail -n 25 .wtenv/p2_all.log`
Expected: all OK.

- [ ] **Step 2: Both gated smokes**

Run: `RUN_GF_SMOKE=1 .wtenv/wt_run.sh $PWD/src .wtenv/p2_smokes.log python -m unittest tests.test_multirank_blank_smoke tests.test_globalfit_sample -v; tail -n 30 .wtenv/p2_smokes.log`
Expected: all OK.

- [ ] **Step 3: Line-length gate**

`awk 'length > 100 {print FILENAME": "FNR}' src/lisatools/globalfit/communication/ranks.py src/lisatools/globalfit/loginfo.py tests/test_run_multirank_helpers.py tests/test_recipe_local_block.py tests/test_multirank_blank_smoke.py` and `git diff b63a8041..HEAD -- src/lisatools/globalfit/run.py src/lisatools/globalfit/recipe.py src/lisatools/globalfit/stock/base.py | grep '^+' | awk 'length > 101'` → both empty.

- [ ] **Step 4: Report**

`git log --oneline b63a8041..HEAD`, `git status --short`, and the cluster runs now warranted (single-GPU `-n 1` and `-n 3` synthetic smokes for bit-identity against the pre-Plan-2 binary; the `RANKS_PER_GPU=2` shared-GPU run is blocked until Plans 4/5 lift the readiness guard).

---

## Self-review notes

- Spec coverage WP2: roles/layout/fanout comm in `__init__` (T2); `prepare_main` split + `prepare_compute` + shared `_build_acs_and_recipe` (T4); `setup_acs(walker_block)` (T3); `_global_likelihood` at the three sites (T3); `run_global_fit` branches + `_release_rank_gpu_pool` + legacy spare path (T2); seeds (T4); per-rank log files + prefixed stdout (T2); abort hook (T2, real MPI comms only); ping handshake (T4); `MoveBuildContext` population incl. `state_local` (T4); `clock["seed_base"]` (T4); `Recipe` stage/iteration hooks (T4). Deferred from the spec text by ruling: the lazy HDF probe (build-time read-only open; compute ranks never open the store afterwards). WP3: builders at B + `fanout_branches` + six block-gated sites (T5). Additional: size-2 fallback (T1), `StockGlobalFit.run` hook (T6), readiness guard (T4), end-to-end smoke (T7).
- Placeholder scan: Task 4's `_build_acs_and_recipe` sketch contains one expression flagged "remove the placeholder" — the implementer builds the flat move list explicitly; Task 2's `_leaf_moves` names the class-hierarchy check to verify. Both are instructions to resolve against the code, not TBDs.
- Type consistency: `walker_block=(w0, w1)` tuples everywhere; `_global_likelihood` returns `(nwalkers,)`; `fanout.model` is a `GlobalFitInfo`; `registry` keys are `gf_move_name`; `builtins["likelihood"](payload, clock, model)` matches `ComputeService.handle`'s call signature.
