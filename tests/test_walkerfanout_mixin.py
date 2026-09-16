"""WalkerFanoutMixin: direct call with one compute rank; slice/run/merge with several."""

import unittest

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

    def __init__(self, tag):
        self.tag = float(tag)
        self.tc = TemperatureControl(2, 2, ntemps=NTEMPS, permute=False)
        self.eigen_store_path = "store.h5"
        self.applied = []
        self.merged = None

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
        new = GFState(state, copy=True)
        new.branches["mbh"].coords[...] += 1.0
        nw = new.log_like.shape[1]
        new.log_like[...] = np.arange(nw)[None, :] + self.tag  # tag = which rank scored it
        sub = new.sub_states["mbh"]
        sub.in_model_accepted[...] = 1
        return new, np.ones(new.log_like.shape, dtype=bool)


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
    def test_two_compute_ranks_slice_run_merge(self):
        world = FakeWorld(3, nodes=[0, 0, 0])
        moves = {}

        def fn(rank, comm):
            layout = build_layout(comm, NWALKERS, [0, 1], legacy=False)
            fcomm = layout.make_fanout_comm(comm)
            role = layout.role_of(rank)
            if role == RankRole.SAVER:
                return "saver"
            fo = WalkerFanout(fcomm, layout, rank, model=None)
            move = _StubMove(10.0 * rank)
            move.install_walker_fanout(_Curr(fo, rank))
            moves[rank] = move
            if role == RankRole.HEAD:
                fo.enter_stage("pe", "pe")
                fo.model = "head-model"
                state = make_state(np.random.default_rng(1))
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
        ref, new, acc = out[0]
        self.assertEqual(out[1], 1)  # one propose served
        w0, w1 = 0, NWALKERS // 2
        np.testing.assert_array_equal(new.branches["mbh"].coords, ref.branches["mbh"].coords + 1)
        # the head scored walkers [0, N/2) (tag 0) and the worker [N/2, N) (tag 10)
        head_cols = np.arange(w1 - w0) + 0.0
        worker_cols = np.arange(NWALKERS - w1) + 10.0
        np.testing.assert_array_equal(new.log_like[0], np.concatenate([head_cols, worker_cols]))
        self.assertEqual(acc.shape, (NTEMPS, NWALKERS))
        self.assertTrue(acc.all())
        # delta counters: the head copy is zeroed then the rank values are summed
        self.assertTrue(np.all(new.sub_states["mbh"].in_model_accepted == 2))
        # untouched sub-states are left alone
        for name, sub in new.sub_states.items():
            if name != "mbh" and sub is not None:
                np.testing.assert_array_equal(sub.coords, ref.sub_states[name].coords)
        # clock round trip on both ranks; extras merged per rank
        self.assertEqual(moves[0].applied, [7])
        self.assertEqual(moves[1].applied, [7])
        self.assertEqual(moves[0].merged, {0: {"nw": w1 - w0}, 1: {"nw": NWALKERS - w1}})
        # ranks never adapt; the configured value is remembered on the control
        for rank in (0, 1):
            self.assertFalse(moves[rank].tc.adaptive)
            self.assertTrue(moves[rank].tc.gf_configured_adaptive)
        self.assertEqual(moves[0].eigen_store_path, "store.h5")  # head keeps the sidecar
        self.assertIsNone(moves[1].eigen_store_path)  # worker never writes it

    def test_gf_serve_rejects_other_ops(self):
        move = _StubMove(0.0)
        with self.assertRaises(ValueError):
            move.gf_serve("score", {}, {}, None)


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


if __name__ == "__main__":
    unittest.main()
