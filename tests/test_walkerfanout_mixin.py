"""WalkerFanoutMixin: direct call with one compute rank; slice/run/merge with several."""

import os
import unittest
from unittest import mock

import numpy as np
from eryn.moves.tempering import TemperatureControl

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.communication.fanout import ComputeService, WalkerFanout
from lisatools.globalfit.communication.ranks import RankRole, build_layout
from lisatools.globalfit.moves.walkerfanout import WalkerFanoutMixin, pooled_ladder_step
from lisatools.globalfit.state import GFState
from tests.test_gf_substate_roundtrip import NTEMPS, NWALKERS, make_state


class _Curr:
    def __init__(self, fanout, rank):
        self.fanout = fanout
        self.rank = rank


class _StubMove(WalkerFanoutMixin):
    """A move whose body is trivial but leaves fingerprints the merge must preserve."""

    gf_move_name = "stub"
    fanout_branches = ["mbh"]
    fanout_assigns_counters = True

    def __init__(self, tag, tc=None, assigns_counters=True):
        self.tag = float(tag)
        self.tc = tc if tc is not None else TemperatureControl(2, 2, ntemps=NTEMPS, permute=False)
        self.fanout_assigns_counters = assigns_counters
        self.eigen_store_path = "store.h5"
        self.applied = []
        self.merged = None
        self.seen_models = []

    def fanout_temperature_controls(self):
        return [self.tc]

    def fanout_payload_extra(self):
        return {"tick": 7}

    def fanout_apply_extra(self, extra):
        self.applied.append(extra["tick"])

    def fanout_reply_extra(self, part):
        return {"nw": int(part.branches["mbh"].coords.shape[1])}

    def fanout_merge_extra(self, replies, new_state):
        self.merged = replies

    def propose_local(self, model, state):
        self.seen_models.append(model)
        new = GFState(state, copy=True)
        new.branches["mbh"].coords[...] += 1.0
        nw = new.log_like.shape[1]
        new.log_like[...] = np.arange(nw)[None, :] + self.tag  # tag = which rank scored it
        sub = new.sub_states["mbh"]
        sub.in_model_accepted[...] = 1
        # accepted pattern is rank-dependent so a mis-ordered merge is visible
        accepted = np.full(new.log_like.shape, self.tag == 0.0, dtype=bool)
        return new, accepted


class MixinSingleTest(unittest.TestCase):
    def test_no_fanout_is_a_direct_call(self):
        move = _StubMove(0.0)
        state = make_state(np.random.default_rng(1))
        ref = GFState(state, copy=True)
        new, acc = move.propose("model", state)
        np.testing.assert_array_equal(new.branches["mbh"].coords, ref.branches["mbh"].coords + 1)
        self.assertEqual(acc.shape, (NTEMPS, NWALKERS))
        self.assertEqual(move.applied, [])  # no payload round trip
        self.assertFalse(move.fanout_active)

    def test_install_single_leaves_ladders_and_sidecar_alone(self):
        layout = FakeWorld(1).run(lambda r, c: build_layout(c, NWALKERS, [0], legacy=False))[0]
        fo = WalkerFanout(None, layout, 0)
        move = _StubMove(0.0)
        move.install_walker_fanout(_Curr(fo, 0))
        self.assertTrue(move.tc.adaptive)
        self.assertEqual(move.eigen_store_path, "store.h5")
        self.assertFalse(move.fanout_active)


class MixinFakeWorldTest(unittest.TestCase):
    SEED_COUNTER = 5  # pre-existing delta count on the head's incoming state

    def _run_two_ranks(self, assigns_counters):
        world = FakeWorld(3, nodes=[0, 0, 0])
        moves = {}

        def fn(rank, comm):
            layout = build_layout(comm, NWALKERS, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            role = layout.role_of(rank)
            if role == RankRole.SAVER:
                return "saver"
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            move = _StubMove(10.0 * rank, assigns_counters=assigns_counters)
            move.install_walker_fanout(_Curr(fo, rank))
            moves[rank] = move
            if role == RankRole.HEAD:
                fo.enter_stage("pe", "pe")
                # DELIBERATELY STALE: run_global_fit binds fanout.model once
                # at setup and never refreshes it, so the head body must use
                # the model ``propose`` was called with, not this one
                fo.model = "stale-fanout-model"
                state = make_state(np.random.default_rng(1))
                state.sub_states["mbh"].in_model_accepted[...] = self.SEED_COUNTER
                ref = GFState(state, copy=True)
                try:
                    new, acc = move.propose("head-model", state)
                finally:
                    fo.stop()
                return ref, new, acc
            return ComputeService(
                fcomm, layout, rank, registry={("pe", "stub"): move}, model="worker-model"
            ).serve()

        out = world.run(fn)
        return out, moves

    def test_two_compute_ranks_slice_run_merge(self):
        out, moves = self._run_two_ranks(assigns_counters=True)
        ref, new, acc = out[0]
        self.assertEqual(out[1], 1)  # one propose served
        w0, w1 = 0, NWALKERS // 2
        np.testing.assert_array_equal(new.branches["mbh"].coords, ref.branches["mbh"].coords + 1)
        # the head scored walkers [0, N/2) (tag 0) and the worker [N/2, N) (tag 10)
        head_cols = np.arange(w1 - w0) + 0.0
        worker_cols = np.arange(NWALKERS - w1) + 10.0
        np.testing.assert_array_equal(new.log_like[0], np.concatenate([head_cols, worker_cols]))
        # accepted: head block True, worker block False -> a mis-ordered merge shows
        self.assertEqual(acc.shape, (NTEMPS, NWALKERS))
        self.assertTrue(acc[:, w0:w1].all())
        self.assertFalse(acc[:, w1:].any())
        # the body ASSIGNS counters: the head copy (5) is zeroed, then 1 + 1 summed
        self.assertTrue(np.all(new.sub_states["mbh"].in_model_accepted == 2))
        # the input state is never mutated by the head's merge
        self.assertTrue(np.all(ref.sub_states["mbh"].in_model_accepted == self.SEED_COUNTER))
        # untouched sub-states are left alone
        for name, sub in new.sub_states.items():
            if name != "mbh" and sub is not None:
                np.testing.assert_array_equal(sub.coords, ref.sub_states[name].coords)
        # clock round trip on both ranks; extras merged per rank; command clock kept
        self.assertEqual(moves[0].applied, [7])
        self.assertEqual(moves[1].applied, [7])
        self.assertEqual(moves[0].merged, {0: {"nw": w1 - w0}, 1: {"nw": NWALKERS - w1}})
        self.assertEqual(moves[1].gf_clock["stage"], "pe")
        self.assertEqual(moves[1].gf_clock["move"], "stub")
        # ranks never adapt; the configured value is remembered on the control
        for rank in (0, 1):
            self.assertFalse(moves[rank].tc.adaptive)
            self.assertTrue(moves[rank].tc.gf_configured_adaptive)
        self.assertEqual(moves[0].eigen_store_path, "store.h5")  # head keeps the sidecar
        self.assertIsNone(moves[1].eigen_store_path)  # worker never writes it
        # the head body ran against propose()'s model, NOT fanout.model
        self.assertEqual(moves[0].seen_models, ["head-model"])
        # the worker keeps its ComputeService model (its own live one)
        self.assertEqual(moves[1].seen_models, ["worker-model"])

    def test_counters_accumulate_when_the_body_adds(self):
        out, _moves = self._run_two_ranks(assigns_counters=False)
        _ref, new, _acc = out[0]
        # no zeroing: the incoming 5 plus the two rank slices' 1 each
        self.assertTrue(np.all(new.sub_states["mbh"].in_model_accepted == self.SEED_COUNTER + 2))

    def test_gf_serve_rejects_other_ops(self):
        move = _StubMove(0.0)
        with self.assertRaises(ValueError):
            move.gf_serve("score", {}, {}, None)

    def test_gf_serve_refuses_an_initialized_unlisted_sub_state(self):
        # the body is handed a slice whose unlisted sub-states are None; a body
        # that INITIALIZES one of them (here: every branch, via a full state)
        # violates the fanout_branches contract and must not be silently dropped
        move = _StubMove(0.0)
        full = make_state(np.random.default_rng(2))  # every sub-state initialized
        with self.assertRaisesRegex(RuntimeError, "not in fanout_branches"):
            move.gf_serve("propose", {"state": full, "extra": {"tick": 1}}, {}, None)

    def test_gf_serve_nulls_bare_unlisted_sub_states(self):
        from lisatools.globalfit.communication.walkerslice import slice_state

        move = _StubMove(0.0)
        part = slice_state(make_state(np.random.default_rng(2)), 0, 2, sub_states=["mbh"])
        reply = move.gf_serve("propose", {"state": part, "extra": {"tick": 1}}, {}, None)
        for name, sub in reply["state"].sub_states.items():
            if name == "mbh":
                self.assertTrue(sub.tempered_initialized)
            else:
                self.assertIsNone(sub)


class _ProposeOverridingMove(_StubMove):
    """The trap: a subclass whose preamble lives in ``propose`` runs on the head only."""

    def propose(self, model, state):
        return self.propose_local(model, state)


class ProposeOverrideGuardTest(unittest.TestCase):
    def _fanout(self, compute_ranks):
        # one compute rank => a 1-rank world (a size-2 world would demote rank 1
        # to the saver and warn); several => + the dedicated saver rank
        size = 1 if len(compute_ranks) == 1 else len(compute_ranks) + 1
        layout = FakeWorld(size).run(
            lambda r, c: build_layout(c, NWALKERS, list(compute_ranks), legacy=False)
        )[0]
        return WalkerFanout(None, layout, 0)

    def test_multi_rank_install_refuses_a_propose_override(self):
        move = _ProposeOverridingMove(0.0)
        with self.assertRaisesRegex(TypeError, "propose_local"):
            move.install_walker_fanout(_Curr(self._fanout([0, 1]), 0))
        # refused before anything is touched: the ladder is left configured
        self.assertTrue(move.tc.adaptive)
        self.assertFalse(hasattr(move.tc, "gf_configured_adaptive"))

    def test_single_rank_install_allows_it(self):
        move = _ProposeOverridingMove(0.0)
        move.install_walker_fanout(_Curr(self._fanout([0]), 0))
        self.assertFalse(move.fanout_active)

    def test_the_ported_families_do_not_override_propose(self):
        from lisatools.globalfit.moves.addremovemove import ResidualAddOneRemoveOneMove
        from lisatools.globalfit.moves.mbhspecialmove import MBHSpecialMove
        from lisatools.globalfit.moves.psdmove import MultiGPUPSDMove, PSDMove

        for cls in (ResidualAddOneRemoveOneMove, MBHSpecialMove, PSDMove, MultiGPUPSDMove):
            self.assertIs(cls.propose, WalkerFanoutMixin.propose, cls.__name__)


class _ACSModel:
    """A model whose ACA reports a walker-row count."""

    def __init__(self, entries):
        self.analysis_container_arr = type("A", (), {"acs_total_entries": entries})()


class HeadACAWidthGuardTest(unittest.TestCase):
    """The mixin's ACA-width rule: the head's ACA is its BLOCK, not the ensemble.

    The raise branch was untested (re-review NEW-3) -- the other mixin tests
    pass a string model, so ``entries is None`` short-circuits the guard.
    """

    def _move(self):
        layout = FakeWorld(3).run(
            lambda r, c: build_layout(c, NWALKERS, [0, 1], legacy=False))[0]
        move = _StubMove(0.0)
        move.install_walker_fanout(_Curr(WalkerFanout(None, layout, 0), 0))
        self.assertTrue(move.fanout_active)
        return move, layout

    def test_an_ensemble_width_aca_raises_before_any_fan_out(self):
        move, layout = self._move()
        w0, w1 = layout.block_of(0)
        state = make_state(np.random.default_rng(1))
        with self.assertRaises(RuntimeError) as ctx:
            move.propose(_ACSModel(NWALKERS), state)  # block is NWALKERS // 2
        msg = str(ctx.exception)
        self.assertIn("_StubMove", msg)
        self.assertIn(f"{NWALKERS} walker rows", msg)
        self.assertIn(f"[{w0}, {w1}) ({w1 - w0} walkers)", msg)
        # raised before the (None) fan-out comm was ever touched
        self.assertEqual(move.seen_models, [])


class SharedControlInstallTest(unittest.TestCase):
    def test_second_install_keeps_the_configured_adaptive(self):
        # the PSD search and PE moves share ONE TemperatureControl: the second
        # install must remember the CONFIGURED value, not the already-False one
        layout = FakeWorld(3).run(lambda r, c: build_layout(c, NWALKERS, [0, 1], legacy=False))[0]
        fo = WalkerFanout(None, layout, 0)
        tc = TemperatureControl(2, 2, ntemps=NTEMPS, permute=False)
        self.assertTrue(tc.adaptive)
        first, second = _StubMove(0.0, tc=tc), _StubMove(0.0, tc=tc)
        first.install_walker_fanout(_Curr(fo, 0))
        second.install_walker_fanout(_Curr(fo, 0))
        self.assertFalse(tc.adaptive)
        self.assertTrue(tc.gf_configured_adaptive)


class PooledLadderStepTest(unittest.TestCase):
    def test_matches_one_eryn_adjustment_and_advances_time(self):
        tc = TemperatureControl(2, 4, ntemps=4, permute=False)
        tc.gf_configured_adaptive = True
        betas0 = np.array(tc.betas, copy=True)
        acc = np.array([3.0, 1.0, 2.0])
        prop = np.array([8.0, 8.0, 8.0])
        expect = betas0 + tc._get_ladder_adjustment(0, betas0.copy(), acc / prop)
        out = pooled_ladder_step(tc, betas0, acc, prop)
        np.testing.assert_allclose(out, expect)
        self.assertEqual(tc.time, 1)
        # a sequential per-rank adaptation (two half-size steps) differs from the pooled one
        seq = betas0 + tc._get_ladder_adjustment(0, betas0.copy(), np.array([2.0, 0.0, 1.0]) / 4)
        seq = seq + tc._get_ladder_adjustment(1, seq.copy(), np.array([1.0, 1.0, 1.0]) / 4)
        self.assertFalse(np.allclose(out, seq))

    def test_not_adaptive_or_single_rung_is_identity(self):
        tc = TemperatureControl(2, 4, ntemps=4, permute=False)
        tc.gf_configured_adaptive = False
        betas0 = np.array(tc.betas, copy=True)
        np.testing.assert_array_equal(
            pooled_ladder_step(tc, betas0, np.ones(3), np.ones(3)), betas0
        )
        self.assertEqual(tc.time, 1)
        one = TemperatureControl(2, 4, ntemps=1, permute=False)
        one.gf_configured_adaptive = True
        np.testing.assert_array_equal(
            pooled_ladder_step(one, np.array(one.betas), np.zeros(0), np.zeros(0)), one.betas
        )


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


if __name__ == "__main__":
    unittest.main()
