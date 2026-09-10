"""Shared-psd MIRROR at the allocation level (CPU, real objects, no GB comps).

Companion to ``tests/test_psd_mirror_multishard.py`` (which drives the
replica gather / refresh through FAKE parents). Here the PRODUCTION
objects are exercised:

* :class:`AnalysisContainerArray` with ``psd_storage="none"`` -- the
  layout the GB ``SubBandBuffer`` (and its template twin, which never reads
  invC) is built with in mirror mode: NO per-container inverse-PSD plane
  is allocated, the writers are no-ops, and every accidental reader of the
  plane raises instead of returning zeros.
* the inverse-PSD plane VERSION (``psd_version``): bumped by BOTH writers
  (``reset_linear_psd_arr`` / ``scatter_linear_psd_arr``), never by a
  direct write into ``linear_psd_arr`` -- which is exactly why the buffer
  binds with ``refresh="always"`` (snapshot-at-fill semantics) and the
  version-gated ``"auto"`` path is only used inside the parity gate.
* ``SubBandBuffer._build_band_ac_list`` in mirror mode: zero-storage
  per-slot invC + ``psd_storage="none"`` (the "twin psd: none" claim of the
  INFO line), against the per-slot allocation with the mirror OFF; and
  ``psd_row_index`` (per-slot parent walker row, padded to the ALLOCATED
  slot count like ``slab_min_f``).

The kernel-level bit-for-bit parity between the two layouts lives in
GBGPU (``tests/test_psd_mirror_kernels.py`` / ``test_psd_mirror_sighet.py``)
and, in production, in the ``GB_PSD_MIRROR_PARITY_PROPOSES`` shadow gate.
CPU-only; every fixture is a few kB.
"""

from __future__ import annotations

import unittest

import numpy as np


def _wdm_settings():
    from lisatools.domains import WDMSettings

    return WDMSettings(Nf=8, Nt=16, dt=10.0, force_backend="cpu")


def _make_acs(ws, n, nch=3, seed=0):
    """``n`` real WDM containers with DISTINCT XYZ (3x3) inverse-PSD slabs."""
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.sensitivity import SensitivityMatrixBase

    rng = np.random.default_rng(seed)
    Nf_a, Nt_a = int(ws.Nf_active), int(ws.Nt_active)
    acs = []
    for i in range(n):
        res = np.zeros((nch, Nf_a, Nt_a))
        dom = ws.associated_class(res, ws)
        sm = SensitivityMatrixBase(ws, skip_inv_det=True)
        sm.sens_mat = np.broadcast_to(np.zeros((), dtype=float), (nch, nch, Nf_a, Nt_a))
        sm.invC = (1.0 + i) * (1.0 + rng.random((nch, nch, Nf_a, Nt_a)))
        sm.channel_shape = (nch, nch)
        acs.append(AnalysisContainer(dom, sm))
    return acs


def _plane(aca):
    """The ACA's inverse-PSD plane in global AC order, from the containers."""
    return np.stack([np.asarray(ac.sens_mat.invC) for ac in aca.acs.ravel()])


class PsdStorageNoneTest(unittest.TestCase):
    """``psd_storage="none"``: what the mirror-mode buffer + twin are built with."""

    def setUp(self):
        try:
            from lisatools.analysiscontainer import AnalysisContainerArray
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"lisatools test deps not installed: {exc}")
        self.ws = _wdm_settings()
        self.acs = _make_acs(self.ws, 4)
        self.aca = AnalysisContainerArray(self.acs, gpus=None, complex_psd=False,
                                          psd_storage="none")

    def test_no_plane_is_allocated(self):
        Nf_a, Nt_a = int(self.ws.Nf_active), int(self.ws.Nt_active)
        self.assertEqual(len(self.aca.linear_psd_arr), 1)
        self.assertEqual(int(self.aca.linear_psd_arr[0].size), 0)
        # ...while the residual plane is the full per-container allocation
        self.assertEqual(int(self.aca.linear_data_arr[0].size), 4 * 3 * Nf_a * Nt_a)

    def test_writers_are_no_ops_and_never_bump(self):
        v0 = self.aca.psd_version
        self.aca.reset_linear_psd_arr()
        self.assertEqual(self.aca.psd_version, v0)
        self.assertEqual(int(self.aca.linear_psd_arr[0].size), 0)

    def test_readers_raise_instead_of_returning_zeros(self):
        with self.assertRaises(RuntimeError):
            self.aca.psd_shaped_view()
        with self.assertRaises(RuntimeError):
            self.aca.psd_mirror_for_device(None)

    def test_bad_storage_rejected(self):
        from lisatools.analysiscontainer import AnalysisContainerArray

        with self.assertRaises(ValueError):
            AnalysisContainerArray(_make_acs(self.ws, 1), gpus=None,
                                   complex_psd=False, psd_storage="shared")


class PsdVersionAndMirrorTest(unittest.TestCase):
    """The parent-side half of the mirror on a REAL per_ac plane."""

    def setUp(self):
        try:
            from lisatools.analysiscontainer import AnalysisContainerArray
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"lisatools test deps not installed: {exc}")
        self.ws = _wdm_settings()
        self.acs = _make_acs(self.ws, 5, seed=1)
        self.aca = AnalysisContainerArray(self.acs, gpus=None, complex_psd=False)

    def test_both_writers_bump_the_version(self):
        v0 = self.aca.psd_version
        self.aca.reset_linear_psd_arr()
        self.assertEqual(self.aca.psd_version, v0 + 1)
        flat = self.aca.gather_linear_psd_arr()
        self.aca.scatter_linear_psd_arr(flat)
        self.assertEqual(self.aca.psd_version, v0 + 2)

    def test_replica_is_the_plane_and_never_an_alias(self):
        m = self.aca.psd_mirror_for_device(None)
        expect = _plane(self.aca).reshape(-1)
        np.testing.assert_array_equal(np.asarray(m).reshape(-1), expect)
        self.assertFalse(np.shares_memory(np.asarray(m), np.asarray(self.aca.linear_psd_arr[0])))

    def test_noise_update_propagates_in_place(self):
        m = self.aca.psd_mirror_for_device(None)
        before = np.asarray(m).copy()
        # a noise update: a walker's inverse-PSD changes and is repacked
        self.acs[2].sens_mat.invC[...] *= 3.0
        self.aca.reset_linear_psd_arr()
        m2 = self.aca.psd_mirror_for_device(None, refresh="auto")
        self.assertIs(m2, m)  # allocate once, refresh IN PLACE
        np.testing.assert_array_equal(np.asarray(m2).reshape(-1),
                                      _plane(self.aca).reshape(-1))
        self.assertFalse(np.array_equal(np.asarray(m2), before))

    def test_auto_is_version_gated_always_is_not(self):
        m = self.aca.psd_mirror_for_device(None)
        stale = np.asarray(m).copy()
        # a direct write into the plane does NOT bump the version...
        v = self.aca.psd_version
        self.aca.linear_psd_arr[0][...] *= 2.0
        self.assertEqual(self.aca.psd_version, v)
        # ...so "auto" hands back the cached (stale) replica...
        np.testing.assert_array_equal(np.asarray(self.aca.psd_mirror_for_device(None, refresh="auto")),
                                      stale)
        # ...and "always" (what bind_psd_mirror uses) re-copies regardless.
        m3 = self.aca.psd_mirror_for_device(None, refresh="always")
        self.assertIs(m3, m)
        np.testing.assert_array_equal(np.asarray(m3), 2.0 * stale)

    def test_bad_refresh_rejected(self):
        with self.assertRaises(ValueError):
            self.aca.psd_mirror_for_device(None, refresh="sometimes")


def _make_min_buffer(mirror, n_bound=4, capacity=6, nch=3, walkers=(2, 0, 1, 2)):
    """A minimally-constructed WDM ``SubBandBuffer`` (full-band slabs) with
    just the state ``_build_band_ac_list`` / ``psd_row_index`` consume."""
    from lisatools.globalfit.moves.gbbands import SubBandBuffer
    from lisatools.utils.parallelbase import LISAToolsParallelModule

    ws = _wdm_settings()
    buf = SubBandBuffer.__new__(SubBandBuffer)
    buf.force_backend = "cpu"
    LISAToolsParallelModule.__init__(buf, force_backend="cpu")
    buf._psd_shared_mirror = bool(mirror)
    buf._basis_settings = ws
    # ``band_slab_Nf`` is a cached property over this knob: None = full-band
    # slabs (the layout the twin and full-band buffers use).
    buf._wdm_band_slab_layers = None
    buf.nchannels = nch
    buf.tdi_channel_setup = "XYZ"
    buf.keep_sens_mat = False
    buf.use_template_arr = False
    buf.alloc_capacity = int(capacity)
    buf.num_bands_now = int(n_bound)
    buf.gb = None
    # (temp, walker, band) per bound slot; walkers deliberately NOT the slot order
    buf.unique_band_combos = np.stack([
        np.zeros(n_bound, dtype=int),
        np.asarray(walkers[:n_bound], dtype=int),
        np.arange(n_bound, dtype=int),
    ], axis=1)
    return buf, ws


class BufferAllocationMirrorTest(unittest.TestCase):
    """``SubBandBuffer._build_band_ac_list`` + ``psd_row_index`` in both modes."""

    def setUp(self):
        try:
            import lisatools.globalfit.moves.gbbands  # noqa: F401
        except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
            self.skipTest(f"lisatools test deps not installed: {exc}")

    def test_mirror_mode_allocates_no_per_slot_invc(self):
        from lisatools.analysiscontainer import AnalysisContainerArray

        buf, ws = _make_min_buffer(mirror=True)
        ac_list, kw = buf._build_band_ac_list()
        Nf_a, Nt_a = int(ws.Nf_active), int(ws.Nt_active)
        self.assertEqual(len(ac_list), 6)  # capacity, not the bound count
        self.assertEqual(kw["psd_storage"], "none")
        for ac in ac_list:
            invc = ac.sens_mat.invC
            self.assertEqual(tuple(invc.shape), (3, 3, Nf_a, Nt_a))
            # zero-storage broadcast view: every stride is 0
            self.assertEqual(set(invc.strides), {0})
            self.assertEqual(tuple(ac.data_res_arr.shape), (3, Nf_a, Nt_a))
        # ...and the ACA built from it (buffer AND template twin take this
        # path) holds no psd plane at all, while the data plane is full size.
        AnalysisContainerArray.__init__(buf, ac_list, **kw)
        self.assertEqual(int(buf.linear_psd_arr[0].size), 0)
        self.assertEqual(int(buf.linear_data_arr[0].size), 6 * 3 * Nf_a * Nt_a)

    def test_off_mode_keeps_the_per_slot_invc(self):
        from lisatools.analysiscontainer import AnalysisContainerArray

        buf, ws = _make_min_buffer(mirror=False)
        ac_list, kw = buf._build_band_ac_list()
        Nf_a, Nt_a = int(ws.Nf_active), int(ws.Nt_active)
        self.assertEqual(kw["psd_storage"], "per_ac")
        for ac in ac_list:
            self.assertNotEqual(set(ac.sens_mat.invC.strides), {0})
        AnalysisContainerArray.__init__(buf, ac_list, **kw)
        self.assertEqual(int(buf.linear_psd_arr[0].size), 6 * 9 * Nf_a * Nt_a)
        self.assertIsNone(buf.psd_row_index)

    def test_psd_row_index_is_the_walker_column_padded_to_capacity(self):
        buf, _ = _make_min_buffer(mirror=True, walkers=(2, 0, 1, 2))
        rows = np.asarray(buf.psd_row_index)
        self.assertEqual(rows.dtype, np.int32)
        self.assertEqual(rows.shape, (6,))          # ALLOCATED count
        np.testing.assert_array_equal(rows[:4], [2, 0, 1, 2])
        np.testing.assert_array_equal(rows[4:], [0, 0])  # valid rows, never consumed
        # cached per bind, invalidated with the slab metadata
        self.assertIs(buf.psd_row_index, buf.psd_row_index)
        buf.__dict__.pop("_psd_row_index_cached", None)
        buf.unique_band_combos[:, 1] = [1, 1, 0, 0]
        np.testing.assert_array_equal(np.asarray(buf.psd_row_index)[:4], [1, 1, 0, 0])

    def test_mirror_psd_slots_are_read_only(self):
        from lisatools.globalfit.moves.gbbands import _MirrorPsdSlots

        buf, _ = _make_min_buffer(mirror=True)
        with self.assertRaises(TypeError):
            _MirrorPsdSlots(buf)[0] = 1.0


if __name__ == "__main__":
    unittest.main()
