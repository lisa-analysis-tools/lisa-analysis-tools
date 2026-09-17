"""GB per-unit ledger: snapshot/delta/exchange/apply over a FakeWorld fan-out comm."""

import unittest
from types import SimpleNamespace

import numpy as np

from lisatools.globalfit.communication.fakecomm import FakeWorld
from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase


def _sorter(coords_in, alive, N):
    return SimpleNamespace(
        coords_in=np.asarray(coords_in, dtype=float), inds=np.asarray(alive, dtype=bool),
        N_vals=np.asarray(N, dtype=int), xp=np,
    )


def _move(comm=None):
    m = GBSpecialBase.__new__(GBSpecialBase)
    m.fanout = None if comm is None else SimpleNamespace(comm=comm, single=False)
    m.fills = []
    m.nwalkers = 1  # one-walker replica mode: the ledger carries no walker identity
    m.waveform_kwargs = {}
    m._likelihood_engine = SimpleNamespace(
        fill_template=lambda acs, params, walkers, N, factor, **kw: m.fills.append(
            (int(factor), np.asarray(params).copy(), np.asarray(N).copy())
        )
    )
    return m


class LedgerLocalTest(unittest.TestCase):
    def test_delta_lists_only_changed_rows(self):
        m = _move()
        s0 = _sorter([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [1, 1, 0], [8, 8, 8])
        before = m._ledger_snapshot(s0, np.array([0, 1, 2]))
        s1 = _sorter([[1.0, 2.0], [3.5, 4.0], [7.0, 8.0]], [1, 1, 1], [8, 8, 8])  # row 1 moved, row 2 born
        d = m._ledger_delta(before, s1)
        np.testing.assert_array_equal(d["rows"], [1, 2])
        np.testing.assert_array_equal(d["old_alive"], [True, False])
        np.testing.assert_array_equal(d["new_alive"], [True, True])
        np.testing.assert_array_equal(d["new_coords_in"], [[3.5, 4.0], [7.0, 8.0]])

    def test_apply_removes_old_then_subtracts_new(self):
        m = _move()
        d = {"rows": np.array([1, 2]), "old_coords_in": np.array([[3.0, 4.0], [5.0, 6.0]]),
             "old_alive": np.array([True, False]), "new_coords_in": np.array([[3.5, 4.0], [7.0, 8.0]]),
             "new_alive": np.array([True, True]), "N": np.array([8, 8])}
        m._ledger_apply(SimpleNamespace(analysis_container_arr="acs"), [d])
        self.assertEqual([f[0] for f in m.fills], [+1, -1])
        np.testing.assert_array_equal(m.fills[0][1], [[3.0, 4.0]])          # only the alive old row
        np.testing.assert_array_equal(m.fills[1][1], [[3.5, 4.0], [7.0, 8.0]])
        m.fills.clear()
        m._ledger_apply(SimpleNamespace(analysis_container_arr="acs"), [{"rows": np.zeros(0, int),
            "old_coords_in": np.zeros((0, 2)), "old_alive": np.zeros(0, bool),
            "new_coords_in": np.zeros((0, 2)), "new_alive": np.zeros(0, bool), "N": np.zeros(0, int)}])
        self.assertEqual(m.fills, [])  # empty deltas make no engine calls

    def test_death_only_delta_makes_one_call(self):
        # Both rows died this unit (new_alive all False): only the +1
        # restore-the-old-template call fires; there is nothing alive left
        # to subtract, so the -1 call is skipped entirely.
        m = _move()
        d = {"rows": np.array([0, 1]), "old_coords_in": np.array([[1.0, 2.0], [3.0, 4.0]]),
             "old_alive": np.array([True, True]), "new_coords_in": np.array([[1.0, 2.0], [3.0, 4.0]]),
             "new_alive": np.array([False, False]), "N": np.array([8, 8])}
        m._ledger_apply(SimpleNamespace(analysis_container_arr="acs"), [d])
        self.assertEqual([f[0] for f in m.fills], [+1])
        np.testing.assert_array_equal(m.fills[0][1], [[1.0, 2.0], [3.0, 4.0]])

    def test_per_row_N_is_masked_with_alive(self):
        # Per-row N must be masked by EACH side's own alive mask independently
        # (old_alive for the +1 call, new_alive for the -1 call), not shipped
        # unmasked or masked by the wrong side's flags.
        m = _move()
        d = {"rows": np.array([0, 1]), "old_coords_in": np.array([[1.0, 2.0], [3.0, 4.0]]),
             "old_alive": np.array([False, True]), "new_coords_in": np.array([[1.5, 2.0], [3.0, 4.0]]),
             "new_alive": np.array([True, True]), "N": np.array([8, 16])}
        m._ledger_apply(SimpleNamespace(analysis_container_arr="acs"), [d])
        self.assertEqual([f[0] for f in m.fills], [+1, -1])
        np.testing.assert_array_equal(m.fills[0][2], [16])     # +1 call: only row 1 (N=16) was old-alive
        np.testing.assert_array_equal(m.fills[1][2], [8, 16])  # -1 call: both rows new-alive


class _FakeDev:
    """A minimal cupy stand-in: refuses to be indexed by a HOST numpy array.

    ``cupy.ndarray.__getitem__`` raises on a numpy index, and a raise on one
    rank between the ledger snapshot and the allgather deadlocks every other
    rank -- so the ledger must push its row index through the sorter's own
    ``xp.asarray`` before gathering.
    """

    def __init__(self, arr):
        self._a = np.asarray(arr)

    def __getitem__(self, idx):
        if isinstance(idx, _FakeDev):
            return _FakeDev(self._a[idx._a])
        if isinstance(idx, np.ndarray):
            raise TypeError("cannot index a device array with a host numpy array")
        return _FakeDev(self._a[idx])

    def get(self):          # what ``asnumpy`` uses to pull off "device"
        return self._a

    @property
    def shape(self):
        return self._a.shape

    @property
    def size(self):
        return self._a.size


class _FakeXP:
    """The namespace a ``_FakeDev`` sorter advertises as ``xp``."""

    @staticmethod
    def asarray(a):
        return a if isinstance(a, _FakeDev) else _FakeDev(a)


def _dev_sorter(coords_in, alive, N):
    return SimpleNamespace(
        coords_in=_FakeDev(np.asarray(coords_in, dtype=float)),
        inds=_FakeDev(np.asarray(alive, dtype=bool)),
        N_vals=_FakeDev(np.asarray(N, dtype=int)),
        xp=_FakeXP,
    )


class LedgerDeviceArrayTest(unittest.TestCase):
    def test_the_fake_device_array_really_refuses_a_host_index(self):
        # negative control for this test class's own premise
        with self.assertRaises(TypeError):
            _FakeDev(np.arange(3))[np.array([0, 1])]

    def test_snapshot_and_delta_work_on_device_arrays(self):
        m = _move()
        s0 = _dev_sorter([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [1, 1, 0], [8, 8, 16])
        before = m._ledger_snapshot(s0, np.array([0, 1, 2]))
        # the payload itself stays HOST numpy
        self.assertIsInstance(before["rows"], np.ndarray)
        self.assertIsInstance(before["coords_in"], np.ndarray)
        np.testing.assert_allclose(before["coords_in"], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(before["alive"], [True, True, False])
        np.testing.assert_array_equal(before["N"], [8, 8, 16])

        s1 = _dev_sorter([[1.0, 2.0], [3.5, 4.0], [7.0, 8.0]], [1, 1, 1], [8, 8, 16])
        d = m._ledger_delta(before, s1)
        np.testing.assert_array_equal(d["rows"], [1, 2])
        np.testing.assert_allclose(d["new_coords_in"], [[3.5, 4.0], [7.0, 8.0]])
        np.testing.assert_array_equal(d["new_alive"], [True, True])
        np.testing.assert_array_equal(d["N"], [8, 16])


class LedgerEmptySelectionTest(unittest.TestCase):
    def test_empty_selection_never_touches_the_sorter(self):
        # fast path: no transform call, no gather -- so a ``None`` sorter is
        # enough to prove nothing was read
        m = _move()
        snap = m._ledger_snapshot(None, np.zeros(0, dtype=int))
        self.assertEqual(int(snap["rows"].size), 0)
        self.assertEqual(int(snap["alive"].size), 0)
        self.assertEqual(int(snap["N"].size), 0)
        self.assertEqual(snap["coords_in"].ndim, 2)
        d = m._ledger_delta(snap, None)
        self.assertEqual(
            set(d),
            {"rows", "old_coords_in", "old_alive", "new_coords_in", "new_alive", "N"},
        )
        self.assertEqual(int(d["rows"].size), 0)
        m._ledger_apply(SimpleNamespace(analysis_container_arr="acs"), [d])
        self.assertEqual(m.fills, [])


class LedgerExchangeTest(unittest.TestCase):
    def test_allgather_returns_the_other_ranks_deltas(self):
        world = FakeWorld(2)

        def fn(rank, comm):
            m = _move(comm)
            mine = {"rows": np.array([rank]), "tag": rank}
            others = m._ledger_exchange(mine)
            return [o["tag"] for o in others]

        out = world.run(fn)
        self.assertEqual(out[0], [1])
        self.assertEqual(out[1], [0])
