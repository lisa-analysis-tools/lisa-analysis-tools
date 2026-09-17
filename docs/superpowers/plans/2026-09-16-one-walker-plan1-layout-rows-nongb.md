# One-walker replica mode, Plan 1: layout, row scatter, addremove/PSD seams, PSD eigen inner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a one-walker global fit run on several compute ranks: every rank holds the single walker's residual, the head runs the unchanged addremove/PSD bodies, and every likelihood batch scatters its rows over the replicas; PSD gets an eigen-axis inner proposal because the stretch move has no complement with one walker.

**Architecture:** `build_layout` enters *replica mode* when `nwalkers == 1` and `n_compute > 1` (every compute rank owns block `(0, 1)`). A new `RowFanout` primitive rides the existing `WalkerFanout.run` transport to scatter row chunks and broadcast replays. `WalkerFanoutMixin.propose` runs the single-process body on the head in replica mode; the moves' scoring seams (`compute_like` for addremove, a new `compute_psd_rows` for PSD) scatter through `RowFanout`; replicas serve `serve_<op>` methods and replay the residual-mutating steps. Nothing changes for `nwalkers > 1` or `n_compute == 1`.

**Tech Stack:** Python 3.12, numpy, mpi4py (pickle transport) with the in-repo `FakeWorld`/`FakeComm` test doubles, Eryn (`EigenAxisMove`, `MHMove`, `TemperatureControl`), `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-16-one-walker-replicas-design.md`

## Global Constraints

- Worktree `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-onewalker`, branch `one-walker-replicas`, base `dev` `aec22834`. Never touch the main checkout.
- **No `git commit`, no `git push`** in this session (standing rule). Every task ends with the tests green and a ledger line in `.superpowers/sdd/2026-09-16-one-walker-plan1/progress.md`; the work stays in the working tree.
- Tests run through the worktree shim, one python process at a time, CPU only:
  ```sh
  cd /Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-onewalker && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate deving && .wtenv/wt_run.sh $PWD/src .wtenv/<name>.log python -m unittest <modules> -v; tail -8 .wtenv/<name>.log
  ```
  Never run `tests/test_gbspecial_flow.py`. `python -m unittest` (not pytest).
- Behaviour for `nwalkers > 1` and for `n_compute == 1` must stay byte-identical: every existing multirank test module listed in Task 8 must stay green.
- Env knob = capitalized attribute name (`GF_ONE_WALKER_REPLICAS`, `{PREFIX}_LIKELIHOOD_FANOUT`, `{P}_INNER_MOVE_KIND`, `{P}_EIGEN_REFRESH`, `{P}_EIGEN_EPS_REL`).
- Op names are method names: a rank serves op `x` by calling `move.serve_x(payload, clock, model)`. Ops in this plan: `ll_rows`, `ll_rows_check`, `ar_replay`, `psd_rows`, `psd_replay`, plus the builtin `residual_hash`.
- Line numbers below are at `aec22834`; re-anchor with grep if an edit shifts them.

---

### Task 1: Replica mode in the layout

**Files:**
- Modify: `src/lisatools/globalfit/communication/ranks.py:30` (constants), `:55-72` (dataclass fields), `:74-98` (accessors), `:115-135` (`describe`), `:249-256` (divisibility), `:316-319` (placement), `:326-339` (`WalkerBlockLayout(...)` return)
- Test: `tests/test_rank_layout.py`

**Interfaces:**
- Produces: `WalkerBlockLayout.replica_mode: bool`, `WalkerBlockLayout.n_replicas -> int` (property), `WalkerBlockLayout.replica_index(rank) -> int`, constant `ONE_WALKER_ENV = "GF_ONE_WALKER_REPLICAS"`. In replica mode `block_of(r) == (0, 1)` for every compute rank and `block == 1`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_rank_layout.py`; add `import os` and `from unittest import mock` at the top if missing)

```python
class ReplicaModeTest(unittest.TestCase):
    def test_one_walker_on_two_compute_ranks_is_replica_mode(self):
        world = FakeWorld(3, nodes=[0, 0, 0])
        outs = _layouts(world, 1, [0, 1])
        lay = outs[0]
        self.assertTrue(lay.replica_mode)
        self.assertEqual(lay.compute_ranks, (0, 1))
        self.assertEqual(lay.n_replicas, 2)
        self.assertEqual(lay.block, 1)
        for r in lay.compute_ranks:
            self.assertEqual(lay.block_of(r), (0, 1))
        self.assertEqual([lay.replica_index(r) for r in lay.compute_ranks], [0, 1])
        self.assertIn("REPLICAS", lay.describe())
        for r in range(3):
            self.assertEqual(outs[r].digest(), lay.digest())

    def test_more_walkers_is_unchanged(self):
        lay = _layouts(FakeWorld(3), 4, [0, 1])[0]
        self.assertFalse(lay.replica_mode)
        self.assertEqual(lay.n_replicas, 1)
        self.assertEqual(lay.block_of(0), (0, 2))
        self.assertEqual(lay.block_of(1), (2, 4))
        self.assertEqual(lay.replica_index(1), 0)
        self.assertNotIn("REPLICAS", lay.describe())

    def test_one_walker_one_compute_rank_is_single(self):
        lay = _layouts(FakeWorld(1), 1, [0])[0]
        self.assertFalse(lay.replica_mode)
        self.assertTrue(lay.is_single())
        self.assertEqual(lay.block_of(0), (0, 1))

    def test_escape_hatch_refuses_replica_mode(self):
        with mock.patch.dict(os.environ, {"GF_ONE_WALKER_REPLICAS": "0"}):
            with self.assertRaises(RuntimeError):  # FakeWorld re-raises rank failures
                _layouts(FakeWorld(3), 1, [0, 1])

    def test_non_divisible_still_raises(self):
        with self.assertRaises(RuntimeError):
            _layouts(FakeWorld(3), 3, [0, 1])
```

- [ ] **Step 2: Run to verify they fail**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/t1.log python -m unittest tests.test_rank_layout.ReplicaModeTest -v; tail -8 .wtenv/t1.log`
Expected: FAIL / ERROR (`replica_mode` attribute missing; `nwalkers=1 is not divisible`).

- [ ] **Step 3: Implement**

In `ranks.py` after line 30 (`LEGACY_ENV = ...`):
```python
#: ``0`` refuses a one-walker run on several compute ranks (today's error)
ONE_WALKER_ENV = "GF_ONE_WALKER_REPLICAS"
```
Dataclass field after `notes: tuple = ()` (line 72):
```python
    #: nwalkers == 1 on several compute ranks: every compute rank holds the
    #: single walker (block (0, 1)); the replicas split the work inside moves
    replica_mode: bool = False
```
Accessors after `is_single` (line 83):
```python
    @property
    def n_replicas(self) -> int:
        return self.n_compute if self.replica_mode else 1

    def replica_index(self, rank) -> int:
        """Position among the replicas (head = 0); 0 outside replica mode."""
        return self.fanout_rank(rank) if self.replica_mode else 0
```
`describe` header (line ~120-125): append `{' REPLICAS' if self.replica_mode else ''}` right after the existing `{' LEGACY' if self.legacy else ''}`.

Replace the divisibility block (249-256) with:
```python
    nwalkers = int(nwalkers)
    n_compute = len(compute)
    replica_mode = False
    if nwalkers == 1 and n_compute > 1:
        if os.environ.get(ONE_WALKER_ENV, "1") != "1":
            raise ValueError(
                f"nwalkers=1 on {n_compute} compute ranks needs one-walker replica mode, "
                f"which {ONE_WALKER_ENV}=0 disables (unset it, or run one compute rank)."
            )
        replica_mode = True
        block = 1
    elif nwalkers % n_compute:
        raise ValueError(
            f"nwalkers={nwalkers} is not divisible by the compute-rank count {n_compute}: "
            "equal walker blocks are required (pick NWALKERS as a multiple of it)."
        )
    else:
        block = nwalkers // n_compute
```
Placement (316-319): replace with
```python
            bi = compute.index(r)
            w0, w1 = (0, 1) if replica_mode else (bi * block, (bi + 1) * block)
            placements[r] = RankPlacement(
                r, role, node, local_index, devices, slot, w0, w1
            )
```
In the `WalkerBlockLayout(...)` constructor call at the end of `build_layout` (326-339) add `replica_mode=replica_mode,`.

- [ ] **Step 4: Run the layout tests**

Run: `.wtenv/wt_run.sh $PWD/src .wtenv/t1.log python -m unittest tests.test_rank_layout tests.test_run_multirank_helpers -v; tail -8 .wtenv/t1.log`
Expected: OK.

- [ ] **Step 5: Record** — create `.superpowers/sdd/2026-09-16-one-walker-plan1/progress.md` with `- Task 1 DONE: replica mode in build_layout (tests: test_rank_layout.ReplicaModeTest)`.

---

### Task 2: `RowFanout` primitive

**Files:**
- Create: `src/lisatools/globalfit/communication/rowfanout.py`
- Test: `tests/test_rowfanout.py`

**Interfaces:**
- Consumes: `WalkerFanout.run(op, *, move, per_rank_payload, local_body, merge)` (`fanout.py:156`), `layout.compute_ranks`, `layout.fanout_rank(rank)`, `layout.head_rank`, `fanout.single`.
- Produces:
  ```python
  class RowFanout:
      def __init__(self, fanout, move): ...           # move: the move object (gf_move_name read lazily)
      active -> bool                                   # fanout is not None and not fanout.single
      def run(self, op, rows, *, local_body) -> dict   # rows: {name: array (N, ...)}; local_body(rows_chunk) -> {name: array (n, ...)}
      def replay(self, op, payload, *, local_body)     # broadcast; local_body(payload) on every rank incl. head
  ```
  Chunks are contiguous in compute-rank order with `np.array_split` sizes; results are concatenated per key along axis 0 in row order. `local_body` MUST return every key for an empty chunk too (zero-length arrays). Rank side: `ComputeService` routes op `x` to `move.gf_serve("x", payload, clock, model)`; `payload == {"rows": {...}}` for `run` and the raw payload for `replay`.

- [ ] **Step 1: Write the failing tests** — `tests/test_rowfanout.py`

```python
"""RowFanout: contiguous row chunks over the compute ranks, gathered in row order."""

import unittest

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.communication.rowfanout import RowFanout


class _StubMove:
    gf_move_name = "stub"

    def __init__(self, tag):
        self.tag = float(tag)
        self.replays = []

    def gf_serve(self, op, payload, clock, model):
        return getattr(self, f"serve_{op}")(payload, clock, model)

    def score(self, rows):
        coords = np.asarray(rows["coords"])
        return {"ll": coords.sum(axis=1) + self.tag, "idx": np.asarray(rows["data_index"])}

    def serve_ll_rows(self, payload, clock, model):
        return self.score(payload["rows"])

    def serve_ar_replay(self, payload, clock, model):
        self.replays.append(payload)
        return None


def _run(head_fn, size=3):
    world = FakeWorld(size, nodes=[0] * size)
    moves = {}

    def fn(rank, comm):
        layout = build_layout(comm, 1, [0, 1], legacy=False)  # nwalkers=1 -> replica mode
        fcomm = layout.make_fanout_comm(comm)
        if layout.role_of(rank) == RankRole.SAVER:
            return "saver"
        move = _StubMove(100.0 * rank)
        moves[rank] = move
        fo = WalkerFanout(fcomm, layout, rank, model=None)
        if layout.role_of(rank) == RankRole.HEAD:
            fo.enter_stage("pe", "pe")
            try:
                return head_fn(RowFanout(fo, move), move)
            finally:
                fo.stop()
        return ComputeService(fcomm, layout, rank, registry={("pe", "stub"): move}).serve()

    return world.run(fn), moves


class RowFanoutTest(unittest.TestCase):
    def test_rows_split_contiguously_and_gather_in_order(self):
        coords = np.arange(14.0).reshape(7, 2)
        idx = np.zeros(7, dtype=np.int32)

        def head(rf, move):
            self.assertTrue(rf.active)
            return rf.run("ll_rows", {"coords": coords, "data_index": idx}, local_body=move.score)

        out, _ = _run(head)
        # np.array_split(7, 2) -> rows 0..3 on the head (tag 0), rows 4..6 on rank 1 (tag 100)
        tags = np.array([0, 0, 0, 0, 100, 100, 100], dtype=float)
        np.testing.assert_array_equal(out[0]["ll"], coords.sum(axis=1) + tags)
        np.testing.assert_array_equal(out[0]["idx"], idx)

    def test_fewer_rows_than_ranks_and_zero_rows(self):
        def head(rf, move):
            one = rf.run("ll_rows", {"coords": np.ones((1, 2)), "data_index": np.zeros(1, np.int32)},
                         local_body=move.score)
            zero = rf.run("ll_rows", {"coords": np.ones((0, 2)), "data_index": np.zeros(0, np.int32)},
                          local_body=move.score)
            return one, zero

        out, _ = _run(head)
        one, zero = out[0]
        np.testing.assert_array_equal(one["ll"], [2.0])  # head-only chunk, rank 1 got an empty chunk
        self.assertEqual(zero["ll"].shape, (0,))

    def test_replay_reaches_every_rank_including_head(self):
        seen = []

        def head(rf, move):
            rf.replay("ar_replay", {"kind": "expose", "leaf": 3}, local_body=lambda p: seen.append(p))
            return len(seen)

        out, moves = _run(head)
        self.assertEqual(out[0], 1)
        self.assertEqual(seen, [{"kind": "expose", "leaf": 3}])
        self.assertEqual(moves[1].replays, [{"kind": "expose", "leaf": 3}])

    def test_single_rank_is_a_direct_call(self):
        move = _StubMove(7.0)
        rf = RowFanout(None, move)
        self.assertFalse(rf.active)
        out = rf.run("ll_rows", {"coords": np.ones((3, 2)), "data_index": np.zeros(3, np.int32)},
                     local_body=move.score)
        np.testing.assert_array_equal(out["ll"], [9.0, 9.0, 9.0])
        calls = []
        rf.replay("ar_replay", {"k": 1}, local_body=calls.append)
        self.assertEqual(calls, [{"k": 1}])

    def test_row_length_mismatch_raises(self):
        rf = RowFanout(None, _StubMove(0.0))
        with self.assertRaises(ValueError):
            rf.run("ll_rows", {"coords": np.ones((3, 2)), "data_index": np.zeros(2)}, local_body=lambda r: r)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure** — `python -m unittest tests.test_rowfanout -v` → ImportError (`rowfanout` missing).

- [ ] **Step 3: Implement** — `src/lisatools/globalfit/communication/rowfanout.py`

```python
"""Scatter independent rows over every compute rank; gather in row order.

One-walker replica mode (``WalkerBlockLayout.replica_mode``): every compute
rank holds the SAME walker's residual, so any scoring row (a parameter vector
plus the walker it scores against) can be evaluated on any rank.
:meth:`RowFanout.run` splits a row batch into contiguous chunks in
compute-rank order (``np.array_split`` sizes), ships each chunk through
:meth:`WalkerFanout.run` (the head evaluates its own chunk locally) and
concatenates the replies back in row order. :meth:`RowFanout.replay`
broadcasts a state-mutating step (expose a source, publish a noise model) so
every replica applies it, head included. With one compute rank both are
direct local calls -- no comm, no pickling.

Rank side: ``ComputeService`` routes op ``x`` to ``move.gf_serve("x", ...)``;
the payload is ``{"rows": {...}}`` for ``run`` and the raw payload for
``replay``. A ``run`` body must return EVERY key for an empty chunk too
(zero-length arrays) so the concatenation stays well-defined.
"""

import numpy as np

__all__ = ["RowFanout"]


class RowFanout:
    def __init__(self, fanout, move):
        self.fanout = fanout
        self.move = move

    @property
    def active(self) -> bool:
        return self.fanout is not None and not self.fanout.single

    @property
    def move_name(self):
        return getattr(self.move, "gf_move_name", None)

    @staticmethod
    def _bounds(n, n_chunks):
        sizes = [len(c) for c in np.array_split(np.arange(int(n)), int(n_chunks))]
        return np.concatenate([[0], np.cumsum(sizes)]).astype(int)

    def run(self, op, rows, *, local_body):
        """``{name: (N, ...)}`` -> ``{name: (N, ...)}``, rows evaluated over all compute ranks."""
        rows = {k: np.asarray(v) for k, v in rows.items()}
        if not rows:
            raise ValueError("RowFanout.run: no row arrays given")
        lengths = {k: int(v.shape[0]) for k, v in rows.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"RowFanout.run: row arrays disagree on length: {lengths}")
        n = next(iter(lengths.values()))
        if not self.active:
            return local_body(rows)
        layout = self.fanout.layout
        bounds = self._bounds(n, layout.n_compute)

        def chunk(rank, _w0, _w1):
            i = layout.fanout_rank(rank)
            s, e = int(bounds[i]), int(bounds[i + 1])
            return {"rows": {k: v[s:e] for k, v in rows.items()}}

        def body(payload, _model):
            return local_body(payload["rows"])

        def merge(results):
            keys = list(results[layout.head_rank].keys())
            return {
                k: np.concatenate(
                    [np.asarray(results[r][k]) for r in layout.compute_ranks], axis=0
                )
                for k in keys
            }

        return self.fanout.run(
            op, move=self.move_name, per_rank_payload=chunk, local_body=body, merge=merge
        )

    def replay(self, op, payload, *, local_body):
        """Apply ``payload`` on every compute rank (the head through ``local_body``)."""
        if not self.active:
            local_body(payload)
            return
        self.fanout.run(
            op,
            move=self.move_name,
            per_rank_payload=lambda _rank, _w0, _w1: payload,
            local_body=lambda p, _model: local_body(p),
            merge=lambda _results: None,
        )
```

- [ ] **Step 4: Run** — `python -m unittest tests.test_rowfanout tests.test_fanout_fakecomm -v` → OK.

- [ ] **Step 5: Record** in the ledger.

---

### Task 3: Mixin: replica branch, `serve_<op>` dispatch, knob

**Files:**
- Modify: `src/lisatools/globalfit/moves/walkerfanout.py:83-91` (class attrs), `:122-141` (`install_walker_fanout`), `:144-147` (`propose`), `:213-216` (`gf_serve` head), module docstring `:1-34`
- Test: `tests/test_walkerfanout_mixin.py`

**Interfaces:**
- Consumes: `RowFanout` (Task 2), `layout.replica_mode` (Task 1).
- Produces on `WalkerFanoutMixin`: class attrs `row_fanout = None`, `likelihood_fanout = True`; `fanout_knob_prefix(self) -> str` (default `str(self.branch_name).upper()`); `rows_active(self) -> bool`; `gf_serve` dispatching any op `x != "propose"` to `self.serve_x(payload, clock, model)` after setting `self.gf_clock`; `propose` running `propose_local` on the head in replica mode; `install_walker_fanout` building `self.row_fanout` and reading `{PREFIX}_LIKELIHOOD_FANOUT` in replica mode and NOT forcing `tc.adaptive = False` there.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_walkerfanout_mixin.py`; add `import os`, `from unittest import mock` if missing)

```python
class _ReplicaStub(WalkerFanoutMixin):
    gf_move_name = "stub"
    fanout_branches = ["mbh"]
    branch_name = "mbh"

    def __init__(self):
        self.calls = []
        self.tc = TemperatureControl(2, 1, ntemps=3, permute=False)
        self.tc.adaptive = True

    def fanout_temperature_controls(self):
        return [self.tc]

    def propose_local(self, model, state):
        self.calls.append((model, state))
        return state, np.zeros((3, 1), dtype=bool)

    def serve_echo(self, payload, clock, model):
        return {"echo": payload, "stage": clock.get("stage")}


class ReplicaModeMixinTest(unittest.TestCase):
    def _world(self, head_fn, env=None):
        world = FakeWorld(3, nodes=[0, 0, 0])
        moves = {}

        def fn(rank, comm):
            layout = build_layout(comm, 1, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return "saver"
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            move = _ReplicaStub()
            with mock.patch.dict(os.environ, env or {}):
                move.install_walker_fanout(_Curr(fo, rank))
            moves[rank] = move
            if layout.role_of(rank) == RankRole.HEAD:
                fo.enter_stage("pe", "pe")
                try:
                    return head_fn(move, fo)
                finally:
                    fo.stop()
            return ComputeService(fcomm, layout, rank, registry={("pe", "stub"): move}).serve()

        return world.run(fn), moves

    def test_head_runs_the_body_on_the_full_state_and_ranks_serve_nothing(self):
        sentinel = object()

        def head(move, fo):
            new, acc = move.propose("head-model", sentinel)
            return new is sentinel, acc.shape, move.calls[0][0]

        out, moves = self._world(head)
        self.assertEqual(out[0], (True, (3, 1), "head-model"))
        self.assertEqual(out[1], 0)  # the worker served no propose
        for r in (0, 1):
            self.assertIsNotNone(moves[r].row_fanout)
            self.assertTrue(moves[r].rows_active())
            self.assertTrue(moves[r].tc.adaptive)  # single-process semantics: ladders adapt in the body

    def test_knob_off_keeps_rows_local(self):
        out, moves = self._world(lambda move, fo: move.rows_active(),
                                 env={"MBH_LIKELIHOOD_FANOUT": "0"})
        self.assertFalse(out[0])
        self.assertFalse(moves[0].likelihood_fanout)

    def test_gf_serve_dispatches_serve_methods(self):
        move = _ReplicaStub()
        out = move.gf_serve("echo", {"a": 1}, {"stage": "pe"}, None)
        self.assertEqual(out, {"echo": {"a": 1}, "stage": "pe"})
        self.assertEqual(move.gf_clock, {"stage": "pe"})
        with self.assertRaises(ValueError):
            move.gf_serve("nope", {}, {}, None)
        with self.assertRaises(ValueError):
            move.gf_serve("_private", {}, {}, None)
```

- [ ] **Step 2: Run to verify failure** — `python -m unittest tests.test_walkerfanout_mixin.ReplicaModeMixinTest -v` → AttributeError (`rows_active`, `row_fanout`).

- [ ] **Step 3: Implement**

Class attrs (after line 91):
```python
    # ---- one-walker replica mode (row scatter) ---------------------------
    row_fanout = None  # RowFanout when the layout is in replica mode
    likelihood_fanout = True  # {PREFIX}_LIKELIHOOD_FANOUT: 0 = the head scores every row itself

    def fanout_knob_prefix(self) -> str:
        """Env prefix of this move's knobs (rule 0: the branch name, capitalized)."""
        return str(getattr(self, "branch_name", "gf")).upper()

    def rows_active(self) -> bool:
        """True when scoring rows scatter over the replicas (replica mode, knob on)."""
        return self.row_fanout is not None and bool(self.likelihood_fanout)
```
`install_walker_fanout`: after the `TypeError` guard (line 133) insert
```python
        if getattr(self.fanout.layout, "replica_mode", False):
            from ..communication.rowfanout import RowFanout

            self.row_fanout = RowFanout(self.fanout, self)
            env = os.environ.get(f"{self.fanout_knob_prefix()}_LIKELIHOOD_FANOUT")
            if env is not None:
                self.likelihood_fanout = env.strip() not in ("0", "false", "False", "")
            if not self.fanout.is_head and hasattr(self, "eigen_store_path"):
                self.eigen_store_path = None  # single-writer sidecar (head only)
            # the head runs the single-process body: its ladders adapt inside it
            return
```
(add `import os` at the top). `propose`:
```python
    def propose(self, model, state):
        if not self.fanout_active:
            return self.propose_local(model, state)
        if getattr(self.fanout.layout, "replica_mode", False):
            # one-walker replica mode: the head runs the unchanged body; the
            # body's scoring seams scatter rows over the replicas (RowFanout)
            return self.propose_local(model, state)
        return self.fanout_propose(model, state)
```
`gf_serve` head (replace lines 214-215):
```python
        if op != PROPOSE_OP:
            handler = None if str(op).startswith("_") else getattr(self, f"serve_{op}", None)
            if handler is None:
                raise ValueError(f"{type(self).__name__} serves no fan-out command {op!r}")
            self.gf_clock = dict(clock or {})
            return handler(payload, clock, model)
```
Module docstring: add a paragraph "One-walker replica mode (``layout.replica_mode``): ``propose`` runs ``propose_local`` on the head with the full one-walker state; the moves' scoring seams scatter rows through ``RowFanout``; ranks serve ``serve_<op>`` methods; none of the semantics above change because the body IS the single-process body."

- [ ] **Step 4: Run** — `python -m unittest tests.test_walkerfanout_mixin tests.test_addremove_fanout_hooks tests.test_psd_fanout_hooks tests.test_recipe_fanout_install -v` → OK.

- [ ] **Step 5: Record** in the ledger.

---

### Task 4: Representative gathers and the residual hash

**Files:**
- Modify: `src/lisatools/globalfit/communication/fanout.py:28-32` (ops), `:50-52` (`concat_blocks`), `:68-88` (`fanout_digest_line`), `:137-154` (after `gather_likelihood`), `:279-285` (`allgather_walker_vector`); `src/lisatools/globalfit/run.py:2953-2969` (builtins); `src/lisatools/globalfit/recipe.py:661-663` (digest emission)
- Test: `tests/test_fanout_fakecomm.py`, `tests/test_run_multirank_helpers.py`

**Interfaces:**
- Produces: `RESIDUAL_HASH_OP = "residual_hash"`, `residual_hash(acs) -> str`, `WalkerFanout.gather_residual_hashes(acs) -> dict[int, str]`, `fanout_digest_line(iteration, state, residual_hashes=None)`. In replica mode `concat_blocks` returns the head's block and `allgather_walker_vector` returns the head's vector on every rank.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_fanout_fakecomm.py`)

```python
class ReplicaGathersTest(unittest.TestCase):
    def _replica_world(self, head_fn):
        world = FakeWorld(3, nodes=[0, 0, 0])

        def fn(rank, comm):
            layout = build_layout(comm, 1, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return "saver"
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            if layout.role_of(rank) == RankRole.HEAD:
                fo.enter_stage("pe", "pe")
                try:
                    return head_fn(fo, layout)
                finally:
                    fo.stop()
            service = ComputeService(
                fcomm, layout, rank, registry={}, model=None,
                builtins={RESIDUAL_HASH_OP: lambda p, c, m: f"h{rank}"},
            )
            return service.serve()

        return world.run(fn)

    def test_concat_blocks_returns_the_head_block_in_replica_mode(self):
        out = self._replica_world(lambda fo, layout: concat_blocks({0: [1.5], 1: [9.9]}, layout))
        np.testing.assert_array_equal(out[0], [1.5])

    def test_allgather_walker_vector_returns_the_head_vector_everywhere(self):
        world = FakeWorld(3, nodes=[0, 0, 0])

        def fn(rank, comm):
            layout = build_layout(comm, 1, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            if layout.role_of(rank) == RankRole.SAVER:
                return "saver"
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            return fo.allgather_walker_vector(np.array([10.0 + rank]))

        out = world.run(fn)
        np.testing.assert_array_equal(out[0], [10.0])
        np.testing.assert_array_equal(out[1], [10.0])

    def test_gather_residual_hashes_and_digest_line(self):
        acs = _Acs([1.0])
        out = self._replica_world(lambda fo, layout: fo.gather_residual_hashes(acs))
        hashes = out[0]
        self.assertEqual(set(hashes), {0, 1})
        self.assertEqual(hashes[1], "h1")
        self.assertEqual(hashes[0], residual_hash(acs))
        line = fanout_digest_line(3, _DigestState(), residual_hashes={0: "aa", 1: "aa"})
        self.assertIn("residual=r0:aa,r1:aa replicas_agree=True", line)
        line = fanout_digest_line(3, _DigestState(), residual_hashes={0: "aa", 1: "bb"})
        self.assertIn("replicas_agree=False", line)
```
Add near the top of the module: `from lisatools.globalfit.communication.fanout import RESIDUAL_HASH_OP, concat_blocks, fanout_digest_line, residual_hash` (extend the existing import) and
```python
class _DigestState:
    log_like = np.zeros((1, 1))
    branches_coords = {"mbh": np.zeros((1, 1, 1, 2))}
    branches_inds = {"mbh": np.ones((1, 1, 1), dtype=bool)}
```
(check `fanout_digest_line` reads `state.log_like`, `state.branches_coords`, `state.branches_inds` at lines 80-88 and match the attribute names it uses.)

- [ ] **Step 2: Run to verify failure** — `python -m unittest tests.test_fanout_fakecomm.ReplicaGathersTest -v` → ImportError / AttributeError.

- [ ] **Step 3: Implement**

`fanout.py` after `LIKELIHOOD_OP` (line 32):
```python
#: builtin: sha1 of the rank's residual buffers (replica agreement check)
RESIDUAL_HASH_OP = "residual_hash"
```
after `_array_bytes` (line 65):
```python
def residual_hash(acs) -> str:
    """16-hex sha1 of the ACA's residual buffers (falls back to the likelihood vector)."""
    arrs = getattr(acs, "linear_data_arr", None)
    if arrs is None:
        arrs = [acs.likelihood(complex=False)]
    return _sha1_16(b"".join(_array_bytes(a) for a in arrs))
```
`concat_blocks`:
```python
def concat_blocks(results, layout):
    """Concatenate per-rank 1-D results in compute-rank (== walker) order.

    Replica mode (one walker on every compute rank): the head's block IS the
    whole vector; the replicas hold copies.
    """
    if getattr(layout, "replica_mode", False):
        return np.asarray(results[layout.head_rank])
    return np.concatenate([np.asarray(results[r]) for r in layout.compute_ranks])
```
`fanout_digest_line(iteration, state, residual_hashes=None)`: build the existing line, then
```python
    if residual_hashes:
        items = ",".join(f"r{int(r)}:{h}" for r, h in sorted(residual_hashes.items()))
        agree = len(set(residual_hashes.values())) == 1
        line += f" residual={items} replicas_agree={agree}"
    return line
```
After `gather_likelihood`:
```python
    def gather_residual_hashes(self, acs):
        """``{world_rank: residual_hash}`` over the compute ranks (head-only, sampling phase)."""
        return self.run(
            RESIDUAL_HASH_OP,
            move=None,
            per_rank_payload=lambda rank, w0, w1: None,
            local_body=lambda payload, model: residual_hash(acs),
            merge=lambda results: {int(r): str(h) for r, h in results.items()},
        )
```
`allgather_walker_vector`: after `parts = self.comm.allgather(local)` insert
```python
        if getattr(self.layout, "replica_mode", False):
            return np.asarray(parts[self.layout.fanout_rank(self.head)])
```
`run.py` builtins dict (2961-2967): add
```python
                RESIDUAL_HASH_OP: lambda payload, clock, model: residual_hash(
                    model.analysis_container_arr
                ),
```
and extend the import at 2953 to `from .communication.fanout import LIKELIHOOD_OP, RESIDUAL_HASH_OP, ComputeService, residual_hash`.
`recipe.py` (661-663): replace the `logger.info(fanout_digest_line(iteration, last_sample))` with
```python
            hashes = None
            fanout = getattr(self, "fanout", None)
            if (
                fanout is not None
                and getattr(fanout.layout, "replica_mode", False)
                and getattr(fanout, "model", None) is not None
            ):
                hashes = fanout.gather_residual_hashes(fanout.model.analysis_container_arr)
            logger.info(fanout_digest_line(iteration, last_sample, residual_hashes=hashes))
```

- [ ] **Step 4: Run** — `python -m unittest tests.test_fanout_fakecomm tests.test_run_multirank_helpers tests.test_functionmove_fanout -v` → OK.

- [ ] **Step 5: Record** in the ledger.

---

### Task 5: addremove seam, replays, check batch, fancy-swap gate

**Files:**
- Modify: `src/lisatools/globalfit/moves/addremovemove.py:1284-1303` (`compute_like`), `:1418-1427` (`_verify_prev_logl` call), `:1790` (expose), `:1802` (setup), `:2022-2026` (fancy gate), `:2109` (fold); `src/lisatools/globalfit/moves/sobbhspecialmove.py:276` (rename), `:603-612` (`_verify_prev_logl` call); `tests/test_addremove_verify_convention.py:34-47` (stub)
- Test: `tests/test_addremove_rows.py` (new), existing `tests/test_addremove_verify_convention.py`, `tests/test_sobbh_chunked_move.py`, `tests/test_eigen_refresh.py`

**Interfaces:**
- Consumes: `RowFanout.run/replay`, mixin `rows_active()`, `row_fanout`.
- Produces on `ResidualAddOneRemoveOneMove`: `compute_like(coords_in, data_index)` = scatter seam (sets `self._last_d_h/_last_h_h` from the gather); `compute_like_local(coords_in, data_index)` = the old body (subclasses override THIS); `_score_rows_local(rows) -> {"ll","d_h","h_h"}`; `serve_ll_rows`; `compute_check_like(coords_in, data_index)` + `_check_like_local` + `serve_ll_rows_check`; `_replay_cold_chain(kind, coords_in, leaf)` with kinds `expose|setup|fold`, `_apply_cold_chain_replay(payload)`, `serve_ar_replay`; `_fancy_swap_fires(repeat) -> bool`.

- [ ] **Step 1: Write the failing tests** — `tests/test_addremove_rows.py`

```python
"""addremove in one-walker replica mode: compute_like scatters rows, replays reach every rank."""

import os
import unittest
from unittest import mock

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.communication.rowfanout import RowFanout
from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove


def _stub(tag):
    m = ResidualAddOneRemoveOneMove.__new__(ResidualAddOneRemoveOneMove)
    m.branch_name = "mbh"
    m.gf_move_name = "mbh_pe"
    m._dcga = None
    m.waveform_like_kwargs = {}
    m.waveform_gen = object()
    m.likelihood_fanout = True
    m.row_fanout = None
    m.nwalkers = 1
    m.ntemps = 4
    m.num_repeats = 3
    m.permute_every = 1
    m._fancy_swap_clock = 1
    m.log = []
    m.compute_like_local = lambda coords, idx, _t=tag: np.asarray(coords).sum(axis=1) + _t
    m.compute_acs_like = lambda coords, data_index=None, signal_gen=None, **kw: (
        np.asarray(coords).sum(axis=1) * 10.0 + tag
    )
    m.remove_cold_chain_sources = lambda c: m.log.append(("expose", m._current_leaf, np.asarray(c).copy()))
    m.setup_likelihood_here = lambda c: m.log.append(("setup", m._current_leaf, np.asarray(c).copy()))
    m.add_back_in_cold_chain_sources = lambda c: m.log.append(("fold", m._current_leaf, np.asarray(c).copy()))
    return m


def _world(head_fn, knob=None):
    world = FakeWorld(3, nodes=[0, 0, 0])
    moves = {}

    def fn(rank, comm):
        layout = build_layout(comm, 1, [0, 1], legacy=False)
        fcomm = layout.make_fanout_comm(comm)
        if layout.role_of(rank) == RankRole.SAVER:
            return "saver"
        fo = WalkerFanout(fcomm, layout, rank, model=None)
        move = _stub(100.0 * rank)
        move.row_fanout = RowFanout(fo, move)
        if knob is not None:
            move.likelihood_fanout = knob
        moves[rank] = move
        if layout.role_of(rank) == RankRole.HEAD:
            fo.enter_stage("pe", "pe")
            try:
                return head_fn(move)
            finally:
                fo.stop()
        return ComputeService(fcomm, layout, rank, registry={("pe", "mbh_pe"): move}).serve()

    return world.run(fn), moves


COORDS = np.arange(10.0).reshape(5, 2)
IDX = np.zeros(5, dtype=np.int32)


class AddremoveRowsTest(unittest.TestCase):
    def test_compute_like_scatters_rows_contiguously(self):
        out, _ = _world(lambda m: (m.compute_like(COORDS, IDX), m._last_d_h))
        ll, d_h = out[0]
        tags = np.array([0, 0, 0, 100, 100], dtype=float)  # array_split(5, 2) -> 3 + 2
        np.testing.assert_array_equal(ll, COORDS.sum(axis=1) + tags)
        self.assertEqual(d_h.shape, (5,))
        self.assertTrue(np.all(np.isnan(d_h)))  # the base scorer has no side outputs

    def test_knob_off_scores_everything_on_the_head(self):
        out, _ = _world(lambda m: m.compute_like(COORDS, IDX), knob=False)
        np.testing.assert_array_equal(out[0], COORDS.sum(axis=1))

    def test_check_batch_scatters_through_the_container_path(self):
        out, _ = _world(lambda m: m.compute_check_like(COORDS, IDX))
        tags = np.array([0, 0, 0, 100, 100], dtype=float)
        np.testing.assert_array_equal(out[0], COORDS.sum(axis=1) * 10.0 + tags)

    def test_replays_reach_every_rank_in_order(self):
        c = np.array([[1.0, 2.0]])

        def head(m):
            m._replay_cold_chain("expose", c, 7)
            m._replay_cold_chain("setup", c, 7)
            m._replay_cold_chain("fold", c, 7)
            return m.log

        out, moves = _world(head)
        kinds = [(k, leaf) for k, leaf, _ in out[0]]
        self.assertEqual(kinds, [("expose", 7), ("setup", 7), ("fold", 7)])
        self.assertEqual([(k, leaf) for k, leaf, _ in moves[1].log], kinds)
        np.testing.assert_array_equal(moves[1].log[0][2], c)
        with self.assertRaises(ValueError):
            moves[1]._apply_cold_chain_replay({"kind": "bogus", "coords": c, "leaf": 0})

    def test_fancy_swap_never_fires_with_one_walker(self):
        m = _stub(0.0)
        self.assertFalse(m._fancy_swap_fires(m.num_repeats - 1))
        m.nwalkers = 2
        self.assertTrue(m._fancy_swap_fires(m.num_repeats - 1))
        self.assertFalse(m._fancy_swap_fires(0))

    def test_single_process_is_the_local_scorer(self):
        m = _stub(5.0)
        np.testing.assert_array_equal(m.compute_like(COORDS, IDX), COORDS.sum(axis=1) + 5.0)
        m._replay_cold_chain("expose", COORDS[:1], 2)
        self.assertEqual(m.log[0][:2], ("expose", 2))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure** — `python -m unittest tests.test_addremove_rows -v` → AttributeError (`compute_check_like`, `_replay_cold_chain`, `_fancy_swap_fires`).

- [ ] **Step 3: Implement in `addremovemove.py`**

Replace `compute_like` (1284-1303) with:
```python
    def compute_like(self, coords_in, data_index):
        """Score rows; in one-walker replica mode the rows scatter over the replicas.

        Every likelihood evaluation of this move funnels through here: the
        entry ``prev_logl``, the in-model scoring, the fancy-swap re-scoring
        (``log_like_for_fancy_swaping``) and the eigen-table builds. The
        rank-local scorer is :meth:`compute_like_local` (DCGA or container
        path) -- subclasses override THAT, never this. The gathered
        ``d_h``/``h_h`` side outputs land on ``self._last_d_h/_last_h_h``
        (the chunked SOBBH scorer's convention).
        """
        if not self.rows_active():
            return self.compute_like_local(coords_in, data_index)
        coords_np = np.atleast_2d(np.asarray(asnumpy(coords_in), dtype=np.float64))
        idx = np.atleast_1d(np.asarray(asnumpy(data_index))).astype(np.int32)
        out = self.row_fanout.run(
            "ll_rows", {"coords": coords_np, "data_index": idx}, local_body=self._score_rows_local
        )
        self._last_d_h = out["d_h"]
        self._last_h_h = out["h_h"]
        return out["ll"]

    def compute_like_local(self, coords_in, data_index):
        """Rank-local scorer (the pre-replica ``compute_like`` body)."""
        if self._dcga is not None:
            return self._compute_like_dcga(coords_in, data_index)
        return self.compute_acs_like(
            coords_in, data_index, **self.waveform_like_kwargs
        )

    def _score_rows_local(self, rows):
        """Serve one row chunk: ``{"ll", "d_h", "h_h"}`` in row order (empty-safe)."""
        coords = np.asarray(rows["coords"], dtype=np.float64)
        idx = np.asarray(rows["data_index"]).astype(np.int32)
        n = int(coords.shape[0])
        if n == 0:
            return {"ll": np.zeros(0), "d_h": np.zeros(0), "h_h": np.zeros(0)}
        self._last_d_h = None
        self._last_h_h = None
        ll = np.asarray(self.compute_like_local(coords, idx), dtype=float).reshape(n)

        def _side(arr):
            if arr is None:
                return np.full(n, np.nan)
            return np.asarray(arr, dtype=float).reshape(n)

        return {"ll": ll, "d_h": _side(self._last_d_h), "h_h": _side(self._last_h_h)}

    def serve_ll_rows(self, payload, clock, model):
        return self._score_rows_local(payload["rows"])

    def _check_like_local(self, coords_in, data_index):
        """The check_ll cross-check: the container path with the MOVE's own generator."""
        return self.compute_acs_like(
            coords_in,
            data_index=data_index,
            signal_gen=self.waveform_gen,
            **self.waveform_like_kwargs,
        )

    def compute_check_like(self, coords_in, data_index):
        if not self.rows_active():
            return np.asarray(self._check_like_local(coords_in, data_index))
        coords_np = np.atleast_2d(np.asarray(asnumpy(coords_in), dtype=np.float64))
        idx = np.atleast_1d(np.asarray(asnumpy(data_index))).astype(np.int32)
        out = self.row_fanout.run(
            "ll_rows_check", {"coords": coords_np, "data_index": idx},
            local_body=self._check_rows_local,
        )
        return out["ll"]

    def _check_rows_local(self, rows):
        n = int(np.shape(rows["coords"])[0])
        if n == 0:
            return {"ll": np.zeros(0)}
        ll = self._check_like_local(np.asarray(rows["coords"]), np.asarray(rows["data_index"]))
        return {"ll": np.asarray(ll, dtype=float).reshape(n)}

    def serve_ll_rows_check(self, payload, clock, model):
        return self._check_rows_local(payload["rows"])

    def _replay_cold_chain(self, kind, coords_in, leaf):
        """Expose / setup / fold the leaf's cold-chain source on EVERY replica.

        Single process (or knob off): a direct local call, identical to the
        pre-replica choreography.
        """
        payload = {
            "kind": str(kind),
            "coords": np.asarray(asnumpy(coords_in), dtype=np.float64),
            "leaf": int(leaf),
        }
        if not self.rows_active():
            self._apply_cold_chain_replay(payload)
            return
        self.row_fanout.replay("ar_replay", payload, local_body=self._apply_cold_chain_replay)

    def _apply_cold_chain_replay(self, payload):
        kind = payload["kind"]
        self._current_leaf = int(payload["leaf"])
        coords = np.asarray(payload["coords"], dtype=np.float64)
        if kind == "expose":
            self.remove_cold_chain_sources(coords)
        elif kind == "setup":
            self.setup_likelihood_here(coords)
        elif kind == "fold":
            self.add_back_in_cold_chain_sources(coords)
        else:
            raise ValueError(f"unknown cold-chain replay kind {kind!r}")

    def serve_ar_replay(self, payload, clock, model):
        self._apply_cold_chain_replay(payload)
        return None

    def _fancy_swap_fires(self, repeat):
        """Walker-permuting swap cadence; never with one walker (identity permutation)."""
        return (
            self.permute_every > 0
            and repeat == self.num_repeats - 1
            and self._fancy_swap_clock % self.permute_every == 0
            and int(self.nwalkers) > 1
        )
```
Call-site edits in `propose_local`:
- 1790 `self.remove_cold_chain_sources(removal_coords_in)` → `self._replay_cold_chain("expose", removal_coords_in, leaf)`
- 1802 `self.setup_likelihood_here(removal_coords_in)` → `self._replay_cold_chain("setup", removal_coords_in, leaf)`
- 2022-2026 → `fancy_swap = self._fancy_swap_fires(repeat)` (keep the comment block above it)
- 2109 `self.add_back_in_cold_chain_sources(add_coords_in)` → `self._replay_cold_chain("fold", add_coords_in, leaf)`
- `_verify_prev_logl` 1418-1427: replace the `self.compute_acs_like(old_coords_in, data_index=data_index_in, signal_gen=self.waveform_gen, **self.waveform_like_kwargs)` call with `self.compute_check_like(old_coords_in, data_index_in)` (keep `.reshape(prev_logl.shape).real`).

`sobbhspecialmove.py`: rename `def compute_like(self, coords_in, data_index):` (276) to `def compute_like_local(self, coords_in, data_index):` and change its docstring's first line to "Rank-local chunked-heterodyne scorer for one batch (the base ``compute_like`` scatters to it)."; in `_verify_prev_logl` (603-612) replace the `compute_acs_like(...)` call with `self.compute_check_like(old_coords_in, data_index_in)`.

`tests/test_addremove_verify_convention.py` `_stub` (34-47): add before `return s`:
```python
    s.row_fanout = None
    s.likelihood_fanout = True
    s.rows_active = ResidualAddOneRemoveOneMove.rows_active.__get__(s)
    s._check_like_local = ResidualAddOneRemoveOneMove._check_like_local.__get__(s)
    s.compute_check_like = ResidualAddOneRemoveOneMove.compute_check_like.__get__(s)
```

- [ ] **Step 4: Run** — `python -m unittest tests.test_addremove_rows tests.test_addremove_verify_convention tests.test_addremove_fanout_hooks tests.test_addremove_repeat_split tests.test_eigen_refresh tests.test_sobbh_chunked_move -v` → OK (the SOBBH module builds a tiny CPU WDM toy; it is laptop-safe).

- [ ] **Step 5: Record** in the ledger.

---

### Task 6: PSD seam, replays, fancy-swap gate

**Files:**
- Modify: `src/lisatools/globalfit/moves/psdmove.py:1842-2014` (`compute_log_like`), `:2208` (`do_fancy`), `:2389-2403` (begin), `:2527-2562` (publish), after `:2347` (hooks)
- Test: `tests/test_psd_rows.py` (new), existing `tests/test_noise_split_moves.py`, `tests/test_psd_move_batched.py`, `tests/test_psd_fanout_hooks.py`, `tests/test_psd_delayed_acceptance.py`

**Interfaces:**
- Produces on `PSDMove`: `_score_rows(walker_inds_keep, psd_coords, galfor_coords, sgwb_coords) -> (n,) float` (the tier dispatch, rank-local); `compute_psd_rows(...)` same signature = scatter seam; `_serve_psd_rows_local(rows)`, `serve_psd_rows`; `_replay_noise_begin()`, `_apply_noise_begin(payload)`, `_replay_noise_publish(new_state)`, `_apply_psd_replay(payload)` (kinds `begin|publish`), `serve_psd_replay`; `_fancy_swap_fires(move_i, nwalkers) -> bool`; `fanout_knob_prefix()` = first sampled branch, capitalized.

- [ ] **Step 1: Write the failing tests** — `tests/test_psd_rows.py`

```python
"""PSDMove in one-walker replica mode: rows scatter, begin/publish replay on every rank."""

import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.communication.rowfanout import RowFanout
from lisatools.globalfit.moves.psdmove import PSDMove


class _Container:
    def __init__(self):
        self.sens_mat = "old"


class _Acs:
    def __init__(self):
        self.c = [_Container()]
        self.resets = 0

    def __getitem__(self, i):
        return self.c[i]

    def __len__(self):
        return 1

    def reset_linear_psd_arr(self):
        self.resets += 1


def _stub(tag):
    m = PSDMove.__new__(PSDMove)
    m.sampled_branches = ["galfor"]
    m.gf_move_name = "galfor_pe"
    m.likelihood_fanout = True
    m.row_fanout = None
    m.permute_every = 1
    m.acs = _Acs()
    m.coarse_runtime = None
    m._fixed_noise_coords = {}
    m.log = []
    m._score_rows = lambda w, p, g, s, _t=tag: (
        np.asarray(p).sum(axis=1) + (0.0 if g is None else np.asarray(g).sum(axis=1)) + _t
    )
    m._prepare_fixed_component_covariances = lambda: m.log.append(("prep", dict(m._fixed_noise_coords)))
    m._build_sensitivity_for_walker = lambda w, p, g, s: ("sens", w, tuple(np.asarray(p)), g, s)
    return m


def _world(head_fn):
    world = FakeWorld(3, nodes=[0, 0, 0])
    moves = {}

    def fn(rank, comm):
        layout = build_layout(comm, 1, [0, 1], legacy=False)
        fcomm = layout.make_fanout_comm(comm)
        if layout.role_of(rank) == RankRole.SAVER:
            return "saver"
        fo = WalkerFanout(fcomm, layout, rank, model=None)
        move = _stub(100.0 * rank)
        move.row_fanout = RowFanout(fo, move)
        moves[rank] = move
        if layout.role_of(rank) == RankRole.HEAD:
            fo.enter_stage("pe", "pe")
            try:
                return head_fn(move)
            finally:
                fo.stop()
        return ComputeService(fcomm, layout, rank, registry={("pe", "galfor_pe")): move}).serve()

    return world.run(fn), moves


PSD = np.arange(10.0).reshape(5, 2)
GAL = np.ones((5, 3))
W = np.zeros(5, dtype=int)


class PSDRowsTest(unittest.TestCase):
    def test_rows_scatter_and_absent_branches_stay_none(self):
        out, _ = _world(lambda m: (m.compute_psd_rows(W, PSD, GAL, None), m.compute_psd_rows(W, PSD, None, None)))
        with_gal, without = out[0]
        tags = np.array([0, 0, 0, 100, 100], dtype=float)
        np.testing.assert_array_equal(with_gal, PSD.sum(axis=1) + 3.0 + tags)
        np.testing.assert_array_equal(without, PSD.sum(axis=1) + tags)

    def test_begin_and_publish_replay_on_every_rank(self):
        fixed = {"psd": np.array([[0.5, 0.25]])}

        def head(m):
            m._fixed_noise_coords = fixed
            m._replay_noise_begin()
            state = SimpleNamespace(branches_coords={
                "psd": np.array([[[[1.0, 2.0]]]]), "galfor": np.array([[[[3.0, 4.0, 5.0]]]]),
            })
            m._replay_noise_publish(state)
            return m.log, m.acs[0].sens_mat, m.acs.resets

        out, moves = _world(head)
        log, sens, resets = out[0]
        self.assertEqual(log[0][0], "prep")
        np.testing.assert_array_equal(log[0][1]["psd"], fixed["psd"])
        self.assertEqual(sens[:3], ("sens", 0, (1.0, 2.0)))
        np.testing.assert_array_equal(sens[3], [3.0, 4.0, 5.0])
        self.assertIsNone(sens[4])
        self.assertEqual(resets, 1)
        # the replica applied the same two steps
        self.assertEqual(moves[1].log[0][0], "prep")
        np.testing.assert_array_equal(moves[1]._fixed_noise_coords["psd"], fixed["psd"])
        self.assertEqual(moves[1].acs[0].sens_mat[:3], ("sens", 0, (1.0, 2.0)))
        self.assertEqual(moves[1].acs.resets, 1)

    def test_knob_prefix_and_fancy_gate(self):
        m = _stub(0.0)
        self.assertEqual(m.fanout_knob_prefix(), "GALFOR")
        self.assertFalse(m._fancy_swap_fires(0, 1))
        self.assertTrue(m._fancy_swap_fires(0, 4))
        self.assertTrue(m._fancy_swap_fires(1, 4))  # permute_every=1 -> every move_i
        m.permute_every = 2
        self.assertFalse(m._fancy_swap_fires(1, 4))
        self.assertTrue(m._fancy_swap_fires(2, 4))

    def test_single_process_paths_are_direct(self):
        m = _stub(1.0)
        np.testing.assert_array_equal(m.compute_psd_rows(W, PSD, None, None), PSD.sum(axis=1) + 1.0)
        m._fixed_noise_coords = {"psd": np.zeros((1, 2))}
        m._replay_noise_begin()
        self.assertEqual(m.log[0][0], "prep")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure** — `python -m unittest tests.test_psd_rows -v` → AttributeError.

- [ ] **Step 3: Implement in `psdmove.py`**

Extract the tier dispatch. Replace lines 1895-2014 of `compute_log_like` (from `if self._kernel_fast_path_available(...)` through `return logl, None`) with:
```python
        logl[logp_keep] = self.compute_psd_rows(
            walker_inds_keep,
            psd_coords,
            galfor_coords if has_galfor else None,
            sgwb_coords if has_sgwb else None,
        )
        self.prev_logl = logl.copy()
        return logl, None
```
and add these methods right after `compute_log_like`:
```python
    def _score_rows(self, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords):
        """Rank-local tier dispatch: ``(n,)`` log-likelihoods for merged noise rows.

        Row ``i`` scores the full noise model ``(psd, galfor, sgwb)[i]`` against
        walker ``walker_inds_keep[i]``'s residual on THIS rank's ACA. Tier order
        is unchanged from the pre-replica ``compute_log_like``: C++ kernel,
        galfor sub-band coarse, coarse batch, PSD_BATCH, container fallback.
        """
        from eryn.state import BranchSupplemental

        walker_inds_keep = np.asarray(walker_inds_keep).astype(int).reshape(-1)
        n = int(walker_inds_keep.shape[0])
        if n == 0:
            return np.zeros(0, dtype=float)
        has_galfor = galfor_coords is not None
        has_sgwb = sgwb_coords is not None

        if self._kernel_fast_path_available(has_sgwb=has_sgwb):
            input_args = [psd_coords, galfor_coords] if has_galfor else [psd_coords]
            supps_keep = BranchSupplemental(
                {"walker_inds": walker_inds_keep.copy()}, base_shape=(n,), copy=True
            )
            return np.asarray(
                self.psd_log_like(input_args, supps=supps_keep, **self.psd_kwargs), dtype=float
            ).reshape(n)

        if self._galfor_subband_fast_path_available():
            stat = self.acs.flatten()[0].coarse_stats
            tmp_logl = self._compute_galfor_subband_loglike(stat, walker_inds_keep, galfor_coords)
            if tmp_logl is not None:
                return np.asarray(tmp_logl, dtype=float).reshape(n)

        if self._coarse_batch_fast_path_available():
            <move the existing tier-2 block here verbatim, returning tmp_logl>

        if self._batched_route_ready():
            return np.asarray(
                self._compute_log_like_batched(
                    walker_inds_keep, psd_coords, galfor_coords, sgwb_coords
                ), dtype=float,
            ).reshape(n)

        <move the existing container-fallback try/finally here verbatim, returning tmp_logl>

    def compute_psd_rows(self, walker_inds_keep, psd_coords, galfor_coords, sgwb_coords):
        """The scoring seam: scatters the rows over the replicas in one-walker replica mode."""
        if not self.rows_active():
            return self._score_rows(walker_inds_keep, psd_coords, galfor_coords, sgwb_coords)
        rows = {"walker_inds": np.asarray(walker_inds_keep).astype(np.int32).reshape(-1)}
        for key, arr in (("psd", psd_coords), ("galfor", galfor_coords), ("sgwb", sgwb_coords)):
            if arr is not None:
                rows[key] = np.asarray(arr, dtype=np.float64)
        out = self.row_fanout.run("psd_rows", rows, local_body=self._serve_psd_rows_local)
        return out["ll"]

    def _serve_psd_rows_local(self, rows):
        n = int(np.shape(rows["walker_inds"])[0])
        if n == 0:
            return {"ll": np.zeros(0)}
        ll = self._score_rows(rows["walker_inds"], rows.get("psd"), rows.get("galfor"), rows.get("sgwb"))
        return {"ll": np.asarray(ll, dtype=float).reshape(n)}

    def serve_psd_rows(self, payload, clock, model):
        return self._serve_psd_rows_local(payload["rows"])

    # ---- replays: every replica mirrors the head's propose-begin prep and publish ----
    def _replay_noise_begin(self):
        payload = {
            "kind": "begin",
            "fixed_noise_coords": {
                k: np.asarray(v, dtype=np.float64) for k, v in self._fixed_noise_coords.items()
            },
        }
        if not self.rows_active():
            self._apply_psd_replay(payload)
            return
        self.row_fanout.replay("psd_replay", payload, local_body=self._apply_psd_replay)

    def _apply_noise_begin(self, payload):
        self._fixed_noise_coords = {
            k: np.asarray(v, dtype=np.float64) for k, v in payload["fixed_noise_coords"].items()
        }
        self._prepare_fixed_component_covariances()
        if self.coarse_sidecar_active:
            self.coarse_runtime.refresh_P(self.acs)
            self._prepare_fixed_component_covariances_coarse()

    def _replay_noise_publish(self, new_state):
        """Publish the cold-row noise model of walker 0 onto every replica's container."""
        bc = new_state.branches_coords

        def _row(key):
            return np.asarray(bc[key][0, 0, 0], dtype=np.float64) if key in bc else None

        payload = {"kind": "publish", "psd": _row("psd"), "galfor": _row("galfor"), "sgwb": _row("sgwb")}
        self.row_fanout.replay("psd_replay", payload, local_body=self._apply_psd_replay)

    def _apply_psd_replay(self, payload):
        kind = payload["kind"]
        if kind == "begin":
            self._apply_noise_begin(payload)
        elif kind == "publish":
            new_sens = self._build_sensitivity_for_walker(
                0, payload["psd"], payload["galfor"], payload["sgwb"]
            )
            self.acs[0].sens_mat = new_sens
            self.acs.reset_linear_psd_arr()
        else:
            raise ValueError(f"unknown noise replay kind {kind!r}")

    def serve_psd_replay(self, payload, clock, model):
        self._apply_psd_replay(payload)
        return None

    def _fancy_swap_fires(self, move_i, nwalkers):
        """Walker-permuting swap cadence; never with one walker (identity permutation)."""
        return (int(move_i) % int(self.permute_every) == 0) and int(nwalkers) > 1

    def fanout_knob_prefix(self):
        return str((self.sampled_branches or ["psd"])[0]).upper()
```
Note the `coarse_sidecar_active` property must exist on the stub-less path: it is a property at line 1153 reading `self.coarse_runtime`; the test stub sets `coarse_runtime = None`.

`propose_local` edits:
- 2394-2403: replace `self._prepare_fixed_component_covariances()` and the coarse block with one line `self._replay_noise_begin()`.
- 2208: `do_fancy = self._fancy_swap_fires(move_i, int(np.shape(new_state.log_like)[1]))` in `run_move`.
- 2551-2562: replace with
```python
        if self.rows_active():
            self._replay_noise_publish(new_state)  # walker 0 on every replica, head included
        else:
            self._run_rows_per_split(
                _publish_one, np.arange(nwalkers), np.arange(nwalkers)
            )
            self.acs.reset_linear_psd_arr()
```
(keep `_publish_one` and the comment about the repack; add a comment that the debug overlay params are captured only on the non-replica path).

- [ ] **Step 4: Run** — `python -m unittest tests.test_psd_rows tests.test_noise_split_moves tests.test_psd_fanout_hooks tests.test_psd_delayed_acceptance tests.test_psd_move_batched tests.test_psd_move_multi_shard -v` → OK (`test_psd_move_batched` skips its stock-stack class if mpi4py or the synthetic stack is unavailable; everything else must pass).

- [ ] **Step 5: Record** in the ledger.

---

### Task 7: PSD eigen inner proposal + settings + recipe plumbing

**Files:**
- Modify: `src/lisatools/globalfit/moves/psdmove.py:159-268` (`__init__`), `:2102` and `:2187` (inner propose calls), `:2423` (after `nt_mod, nwalkers_mod`); `src/lisatools/globalfit/stock/erebor/noise.py:86-133` (PSDSettings), `:338-372` (GalForSettings); `src/lisatools/globalfit/stock/erebor/stochastic.py:30-57` (SGWBSettings); `src/lisatools/globalfit/recipe.py:2290-2315` (`move_kwargs`)
- Test: `tests/test_psd_eigen_inner.py` (new), `tests/test_stock_globalfit.py` (env-default pattern, read-only reference)

**Interfaces:**
- Consumes: `eigen_refresh.eigen_tables_from_ll_batch(call_ll, x0s, widths, *, eps_rel)`, `eigen_refresh.prior_box_widths(container, ndim)`, `eryn.moves.EigenAxisMove(mode="axis", periodic=..., temperature_control=...)`, `MHMove.propose(model, state)`, `compute_psd_rows` (Task 6).
- Produces on `PSDMove`: ctor kwargs `inner_move_kind=None, eigen_refresh_every=10, eigen_eps_rel=1e-4`; `_resolve_inner_kind(nwalkers) -> "eigen"|"stretch"` (raises for stretch at one walker); `_eigen_inner_move() -> EigenAxisMove`; `_refresh_eigen_tables(tmp_branches_coords)`; `_inner_propose(model, state)`. Settings fields `inner_move_kind`, `eigen_refresh_every`, `eigen_eps_rel` on the three noise settings blocks with env `PSD_/GALFOR_/SGWB_` + `INNER_MOVE_KIND`, `EIGEN_REFRESH`, `EIGEN_EPS_REL`.

- [ ] **Step 1: Write the failing tests** — `tests/test_psd_eigen_inner.py`

```python
"""PSD eigen-axis inner proposal (one-walker regime): kind resolution, tables, one MH step."""

import os
import unittest
from unittest import mock

import numpy as np
from eryn.model import Model
from eryn.moves import TemperatureControl
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import BranchSupplemental

from lisatools.globalfit.moves.psdmove import PSDMove
from lisatools.globalfit.state import GFState

H = np.array([[4.0, 0.0], [0.0, 1.0]])  # curvature -> sigmas 0.5 and 1.0 along the axes
NT = 3


def _quad(p):
    p = np.atleast_2d(np.asarray(p, dtype=float))
    return -0.5 * np.einsum("ni,ij,nj->n", p, H, p)


def _like_fn(coords, inds=None, logp=None, supps=None, branch_supps=None):
    x = coords["psd"]
    return _quad(x.reshape(-1, 2)).reshape(x.shape[:2]), None


def _prior_fn(coords, *args, **kwargs):
    x = coords["psd"]
    return np.zeros(x.shape[:2])


def _move(kind=None, nwalkers=1):
    tc = TemperatureControl(2, nwalkers, ntemps=NT, permute=False)
    m = PSDMove(
        None, {"psd": ProbDistContainer({0: uniform_dist(-10, 10), 1: uniform_dist(-10, 10)})},
        sampled_branches=["psd"], temperature_control=tc, live_dangerously=True,
        inner_move_kind=kind, name="eigen test",
    )
    m.compute_log_like = _like_fn
    m.compute_log_prior = _prior_fn
    m._score_rows = lambda w, p, g, s: _quad(p)
    m._fixed_noise_coords = {}
    m.periodic = None
    m.accepted = np.zeros((NT, nwalkers))
    return m, tc


class InnerKindTest(unittest.TestCase):
    def test_default_is_eigen_at_one_walker_and_stretch_otherwise(self):
        m, _ = _move()
        self.assertEqual(m._resolve_inner_kind(1), "eigen")
        self.assertEqual(m._resolve_inner_kind(4), "stretch")

    def test_stretch_at_one_walker_raises_and_names_the_knob(self):
        m, _ = _move(kind="stretch")
        with self.assertRaisesRegex(ValueError, "INNER_MOVE_KIND"):
            m._resolve_inner_kind(1)
        self.assertEqual(m._resolve_inner_kind(2), "stretch")
        with self.assertRaises(ValueError):
            _move(kind="bogus")[0]._resolve_inner_kind(1)


class EigenTablesTest(unittest.TestCase):
    def test_tables_recover_the_quadratic_axes_per_rung(self):
        m, _ = _move()
        coords = {"psd": np.zeros((NT, 1, 1, 2))}
        m._refresh_eigen_tables(coords)
        axes, sigmas = m._eigen_inner._tables["psd"]
        self.assertEqual(axes.shape, (NT, 1, 1, 2, 2))
        self.assertEqual(sigmas.shape, (NT, 1, 1, 2))
        for t in range(NT):
            np.testing.assert_allclose(np.sort(sigmas[t, 0, 0]), [0.5, 1.0], rtol=1e-3)
            A = np.abs(axes[t, 0, 0])  # columns are +-e_i in some order
            np.testing.assert_allclose(A @ A.T, np.eye(2), atol=1e-3)
            np.testing.assert_allclose(np.sort(A.ravel()), [0.0, 0.0, 1.0, 1.0], atol=1e-3)

    def test_tables_tile_over_walkers(self):
        m, _ = _move(kind="eigen", nwalkers=3)
        m._refresh_eigen_tables({"psd": np.zeros((NT, 3, 1, 2))})
        axes, _ = m._eigen_inner._tables["psd"]
        self.assertEqual(axes.shape, (NT, 3, 1, 2, 2))


class EigenStepTest(unittest.TestCase):
    def test_run_move_takes_an_eigen_mh_step_with_one_walker(self):
        m, tc = _move()
        m._inner_kind = "eigen"
        m._refresh_eigen_tables({"psd": np.zeros((NT, 1, 1, 2))})
        m._tally_in_model_proposed = np.zeros(NT, dtype=int)
        m._tally_in_model_accepted = np.zeros(NT, dtype=int)
        m._tally_swaps_proposed = np.zeros(NT - 1, dtype=int)
        m._tally_swaps_accepted = np.zeros(NT - 1, dtype=int)
        rng = np.random.RandomState(3)
        coords = {"psd": rng.normal(size=(NT, 1, 1, 2))}
        supps = BranchSupplemental({"walker_inds": np.zeros((NT, 1), dtype=int)}, base_shape=(NT, 1), copy=True)
        state = GFState(coords, copy=True, supplemental=supps)
        state.log_prior = _prior_fn(coords)
        state.log_like = _like_fn(coords)[0]
        model = Model(None, _like_fn, _prior_fn, tc, map, rng)
        np.random.seed(5)
        new_state, accepted = m.run_move(0, model, state)
        self.assertEqual(np.asarray(accepted).shape, (NT, 1))
        np.testing.assert_allclose(new_state.log_like, _like_fn(new_state.branches_coords)[0])
        moved = np.any(new_state.branches_coords["psd"] != coords["psd"], axis=(2, 3))
        # rejected rows keep their coords, accepted rows moved (swaps may shuffle rungs, so compare sets)
        self.assertEqual(int(moved.sum()) >= int(np.asarray(accepted).sum()), True)


class SettingsKnobsTest(unittest.TestCase):
    def test_noise_settings_carry_the_eigen_fields(self):
        from lisatools.globalfit.stock.erebor.noise import GalForSettings, PSDSettings
        from lisatools.globalfit.stock.erebor.stochastic import SGWBSettings

        for cls in (PSDSettings, GalForSettings, SGWBSettings):
            s = cls()
            self.assertIsNone(s.inner_move_kind)
            self.assertEqual(s.eigen_refresh_every, 10)
            self.assertAlmostEqual(s.eigen_eps_rel, 1e-4)
        with mock.patch.dict(os.environ, {"GALFOR_INNER_MOVE_KIND": "eigen", "GALFOR_EIGEN_REFRESH": "3"}):
            s = GalForSettings()
            self.assertEqual(s.inner_move_kind, "eigen")
            self.assertEqual(s.eigen_refresh_every, 3)


if __name__ == "__main__":
    unittest.main()
```
If `PSDSettings()` needs constructor arguments in this tree, mirror how `tests/test_stock_globalfit.py` builds settings blocks and adjust the test accordingly (the assertions stay).

- [ ] **Step 2: Run to verify failure** — `python -m unittest tests.test_psd_eigen_inner -v` → TypeError (`inner_move_kind` unexpected kwarg).

- [ ] **Step 3: Implement**

`psdmove.py` `__init__`: add kwargs after `coarse_runtime=None,`:
```python
        inner_move_kind: str = None,
        eigen_refresh_every: int = 10,
        eigen_eps_rel: float = 1e-4,
```
and after `self.coarse_runtime = coarse_runtime` (line 208):
```python
        # inner proposal: eryn stretch (needs >= 2 walkers) or the eigen-axis MH
        # move fed by per-rung likelihood-difference info matrices (one-walker
        # default). Resolved per propose from the module ladder's walker count.
        self.inner_move_kind = inner_move_kind
        self.eigen_refresh_every = max(1, int(eigen_refresh_every or 10))
        self.eigen_eps_rel = float(eigen_eps_rel or 1e-4)
        self._inner_kind = None
        self._eigen_inner = None
        self._eigen_visits = 0
```
New methods (place after `run_move_max_likelihood`):
```python
    # ---- inner proposal: stretch or eigen-axis MH ----------------------------
    def _resolve_inner_kind(self, nwalkers) -> str:
        kind = self.inner_move_kind
        if kind is None:
            kind = "eigen" if int(nwalkers) == 1 else "stretch"
        kind = str(kind).strip().lower()
        if kind not in ("eigen", "stretch"):
            raise ValueError(f"PSDMove inner_move_kind {self.inner_move_kind!r}: use 'eigen' or 'stretch'")
        if kind == "stretch" and int(nwalkers) == 1:
            prefix = self.fanout_knob_prefix()
            raise ValueError(
                "PSDMove: the stretch inner proposal needs at least two walkers (there is no "
                f"complement with one); set {prefix}_INNER_MOVE_KIND=eigen"
            )
        return kind

    def _eigen_inner_move(self):
        if self._eigen_inner is None:
            from eryn.moves import EigenAxisMove

            self._eigen_inner = EigenAxisMove(mode="axis", periodic=self.periodic)
        # follow the move's control (None during the delayed-acceptance stage 1)
        self._eigen_inner.temperature_control = self.temperature_control
        return self._eigen_inner

    def _refresh_eigen_tables(self, tmp_branches_coords):
        """Per-branch, per-rung eigen tables from likelihood second differences at walker 0.

        ``call_ll`` for branch ``b`` varies only ``b``'s parameters; the other
        sampled branches sit at their own rung values (rows arrive as whole
        ``ntemps``-point blocks, the batching invariant of
        ``information_matrix_from_ll``) and the fixed branches at the cold row.
        Scoring goes through :meth:`compute_psd_rows`, so it scatters too.
        """
        from .eigen_refresh import eigen_tables_from_ll_batch, prior_box_widths

        names = list(tmp_branches_coords)
        first = np.asarray(tmp_branches_coords[names[0]])
        ntemps, nwalkers = int(first.shape[0]), int(first.shape[1])
        inner = self._eigen_inner_move()
        point = {
            b: np.asarray(tmp_branches_coords[b], dtype=np.float64)[:, 0, 0, :] for b in names
        }
        fixed = {k: np.asarray(v, dtype=np.float64)[0] for k, v in self._fixed_noise_coords.items()}
        for b in names:
            ndim_b = int(point[b].shape[1])
            widths = prior_box_widths(self.priors[b], ndim_b)

            def call_ll(x, _b=b):
                x = np.atleast_2d(np.asarray(x, dtype=np.float64))
                rung = np.tile(np.arange(ntemps), x.shape[0] // ntemps)
                rows = {}
                for key in self.NOISE_BRANCHES:
                    if key == _b:
                        rows[key] = x
                    elif key in point:
                        rows[key] = point[key][rung]
                    elif key in fixed:
                        rows[key] = np.tile(fixed[key], (x.shape[0], 1))
                return self.compute_psd_rows(
                    np.zeros(x.shape[0], dtype=np.int32),
                    rows.get("psd"), rows.get("galfor"), rows.get("sgwb"),
                )

            axes, sigmas = eigen_tables_from_ll_batch(
                call_ll, point[b], widths, eps_rel=self.eigen_eps_rel
            )
            axes5 = np.broadcast_to(
                axes[:, None, None], (ntemps, nwalkers, 1, ndim_b, ndim_b)
            ).copy()
            sig4 = np.broadcast_to(sigmas[:, None, None], (ntemps, nwalkers, 1, ndim_b)).copy()
            inner.set_axes(b, axes5, sig4)

    def _inner_propose(self, model, state):
        """One in-model step: the eigen-axis MH move or the vanilla eryn stretch.

        ``StretchMove.propose`` is named outright (it IS ``RedBlueMove.propose``)
        rather than reached through the MRO past the fan-out mixin.
        """
        if self._inner_kind == "eigen":
            return self._eigen_inner_move().propose(model, state)
        return StretchMove.propose(self, model, state)
```
Call sites: `run_move` line 2187 `new_state, accepted = StretchMove.propose(self, model, state)` → `new_state, accepted = self._inner_propose(model, state)`; `_propose_delayed_acceptance` line 2102 likewise (inside its existing `try/finally` that nulls `self.temperature_control`). In `propose_local`, right after the begin replay (the `self._replay_noise_begin()` line from Task 6) and after `nt_mod, nwalkers_mod = ...` (2423) insert:
```python
        self._inner_kind = self._resolve_inner_kind(nwalkers_mod)
        if self._inner_kind == "eigen":
            if self._eigen_visits % self.eigen_refresh_every == 0:
                self._refresh_eigen_tables(tmp_branches_coords)
            self._eigen_visits += 1
```
(the begin replay must run before it: the tables score merged rows that need the fixed-component covariances).

Settings — in `noise.py` `PSDSettings` (after `log_sampling`) add:
```python
    inner_move_kind: typing.Optional[str] = dataclasses.field(
        default_factory=env_default("PSD_INNER_MOVE_KIND", None, str)
    )
    eigen_refresh_every: int = dataclasses.field(
        default_factory=env_default("PSD_EIGEN_REFRESH", 10, int)
    )
    eigen_eps_rel: float = dataclasses.field(
        default_factory=env_default("PSD_EIGEN_EPS_REL", 1e-4, float)
    )
```
the same three fields in `GalForSettings` with the `GALFOR_` prefix and in `stochastic.py` `SGWBSettings` with `SGWB_` (check `typing`/`dataclasses`/`env_default` imports exist in each file). Field docstrings: "inner proposal of the PSDMove sampling this block: `eigen` (per-rung info-matrix axes; the one-walker default) or `stretch` (needs >= 2 walkers)".

`recipe.py` `move_kwargs` (2290-2315): add
```python
        inner_move_kind=getattr(lead_info, "inner_move_kind", None),
        eigen_refresh_every=int(getattr(lead_info, "eigen_refresh_every", 10) or 10),
        eigen_eps_rel=float(getattr(lead_info, "eigen_eps_rel", 1e-4) or 1e-4),
```
(`lead_info` is the settings block of the first sampled branch, resolved at 2253-2288).

- [ ] **Step 4: Run** — `python -m unittest tests.test_psd_eigen_inner tests.test_psd_rows tests.test_psd_delayed_acceptance tests.test_psd_fanout_hooks tests.test_noise_split_moves tests.test_stock_globalfit tests.test_recipe_fanout_install -v` → OK.

- [ ] **Step 5: Record** in the ledger.

---

### Task 8: Regression sweep, docs, ledger close

**Files:**
- Modify: `docs/superpowers/specs/2026-09-16-one-walker-replicas-design.md` (status line under the title: "Plan 1 landed in the working tree on <date>, uncommitted"), `src/lisatools/globalfit/moves/walkerfanout.py` docstring (done in Task 3; verify), `docs/codebase-map.md` (one line for `communication/rowfanout.py` under the globalfit communication package)
- Test: the whole multirank surface

- [ ] **Step 1: Run the regression sweep in one process**

```sh
.wtenv/wt_run.sh $PWD/src .wtenv/t8.log python -m unittest tests.test_rank_layout tests.test_rowfanout tests.test_fakecomm tests.test_fanout_fakecomm tests.test_fanout_passthrough tests.test_walkerfanout_mixin tests.test_walkerslice_roundtrip tests.test_run_multirank_helpers tests.test_recipe_fanout_install tests.test_functionmove_fanout tests.test_addremove_rows tests.test_addremove_fanout_hooks tests.test_addremove_repeat_split tests.test_addremove_verify_convention tests.test_eigen_refresh tests.test_sobbh_chunked_move tests.test_psd_rows tests.test_psd_eigen_inner tests.test_psd_fanout_hooks tests.test_psd_delayed_acceptance tests.test_noise_split_moves tests.test_psd_move_batched tests.test_psd_move_multi_shard tests.test_multirank_blank_smoke tests.test_multirank_noise_smoke tests.test_diagnostics_multirank -v; tail -12 .wtenv/t8.log
```
Expected: OK (skips allowed only where the module already skips on `dev`: the stock-stack classes without mpi4py/synthetic data). Any FAIL/ERROR is fixed before the task closes.

- [ ] **Step 2: Docs** — add the `rowfanout.py` line to `docs/codebase-map.md`; add to the spec under the title a status line; append a "Plan 1 complete" entry with the test count to the ledger.

- [ ] **Step 3: Diff review** — `git status --short && git diff --stat` and confirm no file outside `src/lisatools/globalfit/{communication,moves,stock/erebor,recipe.py,run.py}`, `tests/`, `docs/` changed. Do NOT commit.

---

## Self-review

- **Spec coverage.** Decision 1 → Task 1. Decision 2 → Task 1. Decision 3 (residual hash + digest) → Task 4; the addremove/PSD replays → Tasks 5, 6; the GB ledger and rebuild are Plan 2. Decision 4 → Tasks 3, 4, 5, 6. Decision 5 (knob) → Tasks 3 (mixin), 5, 6 (prefixes). Decisions 6-10 → Plan 2. Decision 11 → Task 7. Decision 12 → Tasks 5, 6. Decision 13 → runbook, Plan 3. Decision 14 laptop gates → every task's tests; cluster gates → Plan 3. Testing section items "residual after the propose equals the single-process propose's" (an end-to-end addremove run) is NOT in this plan: no CPU fixture drives `propose_local` today; the replay-ordering test stands in, and the end-to-end check moves to the Plan 3 cluster gate.
- **Placeholders.** Task 6 says "move the existing tier-2 block / container fallback here verbatim" with their exact line ranges (1922-1944, 1974-2010) — the code exists in the file; this is a move, not an invention.
- **Type consistency.** `rows_active()` is a METHOD everywhere (Tasks 3, 5, 6, 7 and the verify-convention stub). `RowFanout.run(op, rows, *, local_body)` with `local_body(rows_chunk) -> dict` everywhere; `replay(op, payload, *, local_body)` with `local_body(payload)`. Serve methods are `serve_ll_rows`, `serve_ll_rows_check`, `serve_ar_replay`, `serve_psd_rows`, `serve_psd_replay`. `compute_psd_rows(walker_inds_keep, psd, galfor, sgwb)` is the signature used by Task 7's `call_ll`.
