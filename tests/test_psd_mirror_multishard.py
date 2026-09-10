"""Shared-psd MIRROR on a multi-shard (multi-GPU) buffer -- structural, CPU fakes.

The parent residual ACA is walker-sharded across devices; a mirror-mode
``SubBandBuffer`` (band-sharded) needs ONE replica of the parent's WHOLE
per-walker inverse-PSD plane on EACH of its devices. Pinned here on the
``tests/_multishard.py`` fakes (a recording ``xp`` whose ``cuda.Device``
contexts log the device every operation ran under):

* ``bind_psd_mirror`` produces one replica per device, each written inside
  its OWN device context, each equal to the host gather of the parent plane
  in global-AC order (so a walker row index means the same thing on every
  device -- plan risk R3);
* the replica is allocated ONCE per device and refreshed IN PLACE (same
  array object across binds; new bytes after the parent plane changes --
  plan risks R4/R9/R11), so the two buffers of one propose bind the same
  object;
* the mirror-mode ``psd_buffer`` gatherer's per-shard objects read each
  intra-shard row's walker/layers from that shard's replica, matching a
  direct NumPy gather from the host planes.
"""

from __future__ import annotations

import unittest

import numpy as np

try:
    from tests._multishard import FakeMultiShardACA
except ImportError:  # direct invocation from inside tests/
    from _multishard import FakeMultiShardACA

from lisatools.analysiscontainer import AnalysisContainerArray
from lisatools.globalfit.moves.gbbands import SubBandBuffer, _MirrorPsdSlots


def _wdm_settings():
    from lisatools.domains import WDMSettings

    try:
        return WDMSettings(Nf=8, Nt=16, dt=10.0, force_backend="cpu")
    except Exception as exc:  # pragma: no cover - environment guard
        raise unittest.SkipTest(f"WDMSettings construction failed: {exc}")


class _FakeParent(FakeMultiShardACA):
    """Walker-sharded PARENT residual ACA with the REAL psd-plane API bound.

    ``psd_mirror_for_device`` / ``gather_linear_psd_arr`` are the production
    methods (bound from ``AnalysisContainerArray``); only the storage is a
    fake. Walker ``w``'s plane is ``(1 + 0.7 w) * ramp`` -- distinct per
    walker so a wrong row is visible.
    """

    psd_version = AnalysisContainerArray.psd_version
    psd_mirror_for_device = AnalysisContainerArray.psd_mirror_for_device
    gather_linear_psd_arr = AnalysisContainerArray.gather_linear_psd_arr
    _gather_per_gpu_to_single = AnalysisContainerArray._gather_per_gpu_to_single

    def __init__(self, nwalkers, sens_shape, num_shards):
        super().__init__(sens_shape, nwalkers, num_shards, layout="blocked",
                         dtype=float)
        self.noise_dtype = float
        self._psd_storage = "per_ac"
        self._psd_version = 0
        self._psd_mirror = {}
        self.shape_sens = tuple(sens_shape[:-2])
        self.psd_data_length = int(np.prod(sens_shape[-2:]))
        self.repaint(scale=1.0)

    def repaint(self, scale):
        """Rewrite every walker's plane (a 'noise update')."""
        per_row = int(np.prod(self.per_band_shape))
        ramp = 1.0 + 1e-3 * np.arange(per_row)
        for s, rows in enumerate(self.gpu_splits):
            plane = self.linear_psd_arr[s].reshape(len(rows), per_row)
            for intra, w in enumerate(rows):
                plane[intra] = scale * (1.0 + 0.7 * int(w)) * ramp

    def host_plane(self):
        """(nwalkers, *shape_sens, Nf, Nt) in GLOBAL walker order."""
        per_row = int(np.prod(self.per_band_shape))
        out = np.zeros((self.acs_total_entries, per_row))
        for s, rows in enumerate(self.gpu_splits):
            plane = self.linear_psd_arr[s].reshape(len(rows), per_row)
            for intra, w in enumerate(rows):
                out[int(w)] = plane[intra]
        return out.reshape((self.acs_total_entries,) + self.per_band_shape)


class _FakeMirrorBuffer:
    """The slice of SubBandBuffer state ``bind_psd_mirror`` /
    ``_MirrorPsdSlots`` consume, for a band-sharded buffer."""

    def __init__(self, parent, ws, slots, num_shards, W, slab_min_f):
        self.xp = parent.xp
        self.gpus = list(range(num_shards))
        n = len(slots)
        self.num_bands_now = n
        self.alloc_capacity = None
        self.gpu_map = np.array([b % num_shards for b in range(n)], dtype=int)
        self.gpu_splits = [np.where(self.gpu_map == s)[0] for s in range(num_shards)]
        self.split_map = np.zeros(n, dtype=int)
        for s, rows in enumerate(self.gpu_splits):
            self.split_map[rows] = s
        self._psd_shared_mirror = True
        self._basis_settings = ws
        self.shape_sens = parent.shape_sens
        self.psd_row_index = np.asarray([w for (w, _b) in slots], dtype=np.int32)
        self.band_slab_Nf = W
        self.slab_min_f = None if slab_min_f is None else np.asarray(slab_min_f, dtype=np.int32)
        self.linear_psd_arr = [np.zeros(0) for _ in range(num_shards)]
        self.linear_data_arr = [np.zeros(0) for _ in range(num_shards)]

    bind_psd_mirror = SubBandBuffer.bind_psd_mirror

    @property
    def _n_slots_alloc(self):
        return int(self.num_bands_now)


class MultiShardMirrorBindTest(unittest.TestCase):
    NWALKERS = 5
    NUM_SHARDS = 2

    def setUp(self):
        self.ws = _wdm_settings()
        Nf_a, Nt_a = int(self.ws.Nf_active), int(self.ws.Nt_active)
        self.parent = _FakeParent(self.NWALKERS, (3, 3, Nf_a, Nt_a), self.NUM_SHARDS)
        self.W = min(3, Nf_a)
        lo = int(self.ws.ind_min_f)
        # 6 slots on 2 shards: (walker, band); walkers deliberately NOT the
        # slot order, origins vary within the active band
        self.slots = [(4, 0), (1, 1), (3, 2), (0, 3), (4, 4), (2, 5)]
        self.smf = [lo + (i % max(1, Nf_a - self.W + 1)) for i in range(6)]
        self.buf = _FakeMirrorBuffer(self.parent, self.ws, self.slots,
                                     self.NUM_SHARDS, self.W, self.smf)

    def test_one_replica_per_device_equal_to_host_gather(self):
        self.parent.xp.device_log.clear()
        self.buf.bind_psd_mirror(self.parent)
        host = self.parent.host_plane().reshape(self.NWALKERS, -1)
        ids = set()
        for s in range(self.NUM_SHARDS):
            rep = self.buf.linear_psd_arr[s]
            self.assertEqual(rep.size, host.size)
            np.testing.assert_array_equal(rep.reshape(self.NWALKERS, -1), host)
            ids.add(id(rep))
            self.assertIs(rep, self.parent._psd_mirror[s][0])
        self.assertEqual(len(ids), self.NUM_SHARDS, "replicas must be distinct per device")
        # every device was entered while building its replica
        self.assertTrue(set(range(self.NUM_SHARDS)) <= set(self.parent.xp.device_log))
        self.assertEqual(self.buf._psd_mirror_plane_rows, self.NWALKERS)

    def test_allocate_once_refresh_in_place_after_noise_update(self):
        self.buf.bind_psd_mirror(self.parent)
        first = [self.buf.linear_psd_arr[s] for s in range(self.NUM_SHARDS)]
        before = [r.copy() for r in first]
        v0 = self.parent.psd_version
        # a psd/galfor accept rewrites the parent plane + bumps the version
        self.parent.repaint(scale=1.3)
        self.parent._psd_version += 1
        self.buf.bind_psd_mirror(self.parent)
        host = self.parent.host_plane().reshape(self.NWALKERS, -1)
        for s in range(self.NUM_SHARDS):
            rep = self.buf.linear_psd_arr[s]
            self.assertIs(rep, first[s], "replica must be refreshed IN PLACE")
            np.testing.assert_array_equal(rep.reshape(self.NWALKERS, -1), host)
            self.assertFalse(np.array_equal(rep, before[s]))
        self.assertEqual(self.buf._psd_mirror_bound_version, v0 + 1)

    def test_refresh_always_even_without_version_bump(self):
        """bind_psd_mirror keeps 'snapshot at fill' semantics: a plane
        rewritten WITHOUT a version bump (no writer path) is still what the
        next bind reads."""
        self.buf.bind_psd_mirror(self.parent)
        self.parent.repaint(scale=2.0)
        self.buf.bind_psd_mirror(self.parent)
        host = self.parent.host_plane().reshape(self.NWALKERS, -1)
        for s in range(self.NUM_SHARDS):
            np.testing.assert_array_equal(
                self.buf.linear_psd_arr[s].reshape(self.NWALKERS, -1), host)

    def test_auto_refresh_is_version_gated(self):
        rep0 = self.parent.psd_mirror_for_device(0, refresh="auto")
        self.parent.repaint(scale=3.0)
        rep1 = self.parent.psd_mirror_for_device(0, refresh="auto")
        self.assertIs(rep1, rep0)
        host = self.parent.host_plane().reshape(self.NWALKERS, -1)
        self.assertFalse(np.array_equal(rep1.reshape(self.NWALKERS, -1), host),
                         "auto mode must NOT re-copy without a version bump")
        self.parent._psd_version += 1
        rep2 = self.parent.psd_mirror_for_device(0, refresh="auto")
        self.assertIs(rep2, rep0)
        np.testing.assert_array_equal(rep2.reshape(self.NWALKERS, -1), host)

    def test_two_buffers_share_one_replica_per_device(self):
        other = _FakeMirrorBuffer(self.parent, self.ws, self.slots[::-1],
                                  self.NUM_SHARDS, self.W, self.smf[::-1])
        self.buf.bind_psd_mirror(self.parent)
        other.bind_psd_mirror(self.parent)
        for s in range(self.NUM_SHARDS):
            self.assertIs(self.buf.linear_psd_arr[s], other.linear_psd_arr[s])

    def test_bad_row_or_origin_raises_before_binding(self):
        bad = _FakeMirrorBuffer(self.parent, self.ws,
                                [(self.NWALKERS, 0)] + self.slots[1:],
                                self.NUM_SHARDS, self.W, self.smf)
        with self.assertRaises(ValueError):
            bad.bind_psd_mirror(self.parent)
        self.assertEqual(bad.linear_psd_arr[0].size, 0, "nothing bound on failure")
        top = int(self.ws.ind_max_f) + 2 - self.W
        bad2 = _FakeMirrorBuffer(self.parent, self.ws, self.slots,
                                 self.NUM_SHARDS, self.W,
                                 [top] + self.smf[1:])
        with self.assertRaises(ValueError):
            bad2.bind_psd_mirror(self.parent)

    def test_mirror_psd_slots_per_shard_gather(self):
        """The likelihood's BandView branch reads ``psd_b._shards[s][rows]``
        with INTRA-shard rows: each must come out of shard s's replica at the
        slot's walker row and absolute layers."""
        self.buf.bind_psd_mirror(self.parent)
        gath = _MirrorPsdSlots(self.buf)
        host = self.parent.host_plane()
        lo0 = int(self.ws.ind_min_f)
        self.parent.xp.device_log.clear()
        for s, shard in enumerate(gath._shards):
            rows = np.arange(len(self.buf.gpu_splits[s]))
            got = shard[rows]
            exp = []
            for r in rows:
                slot = int(self.buf.gpu_splits[s][r])
                w, _ = self.slots[slot]
                lo = self.smf[slot] - lo0
                exp.append(host[w][:, :, lo:lo + self.W, :])
            np.testing.assert_array_equal(got, np.stack(exp))
            # single-row + slice forms agree with the array form
            np.testing.assert_array_equal(shard[0], got[0])
            np.testing.assert_array_equal(shard[0:2], got[0:2])
        # every gather entered a device context (the owning shard's)
        self.assertTrue(len(self.parent.xp.device_log) >= self.NUM_SHARDS)
        # and the global object routes slots to their owners
        allrows = gath[np.arange(len(self.slots))]
        for slot, (w, _) in enumerate(self.slots):
            lo = self.smf[slot] - lo0
            np.testing.assert_array_equal(allrows[slot], host[w][:, :, lo:lo + self.W, :])
        with self.assertRaises(TypeError):
            gath[0] = 0.0


if __name__ == "__main__":
    unittest.main()
