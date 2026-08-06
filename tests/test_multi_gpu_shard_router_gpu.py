"""REAL multi-GPU tests for the GB shard router and per-device replicas.

``tests/test_gb_shard_router.py`` runs the same machinery against a NumPy
``FakeMultiShardACA`` whose ``cuda.Device(i)`` context merely records an id.
That proves the *dispatch* -- partitioning, intra-shard index translation,
scatter order, replica caching -- but it structurally cannot prove any of:

* that a replica's buffers actually LAND on the target device (the fake's
  ``_build_device`` is a bookkeeping integer over numpy arrays);
* that a cross-device kernel launch works at all (no kernel ever runs);
* that the real :class:`AnalysisContainerArray`'s own ``gpu_splits`` /
  ``ac_to_intra`` layout agrees with the router's ``_partition``;
* that 1-GPU and 2-GPU runs produce the SAME likelihood.

Those are exactly the failure modes the port exists to fix, so they get a
test that runs on real hardware. Skipped unless >= 2 CUDA devices are
visible; on 1 GPU or CPU the module is a no-op.

Reference results were established on 2x H100 NVL (P2P enabled) --
see ``_dev/merge_plan.md``, Step 5.
"""

from __future__ import annotations

import unittest

import numpy as np


def _n_devices() -> int:
    try:
        import cupy as cp
    except (ImportError, ModuleNotFoundError):
        return 0
    try:
        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


N_DEV = _n_devices()
requires_2gpu = unittest.skipUnless(
    N_DEV >= 2, f"requires >= 2 CUDA devices (found {N_DEV})")

BK = "cuda12x"


@requires_2gpu
class DeviceReplicaTest(unittest.TestCase):
    """The replica layer, on real device memory."""

    @classmethod
    def setUpClass(cls):
        import cupy as cp

        from gbgpu.gbcomps import GBWDMComputations
        from lisatools.domains import WDMSettings

        with cp.cuda.Device(0):
            cls.wdm = WDMSettings(Nf=32, Nt=64, dt=15.0, force_backend=BK)
            cls.comp0 = GBWDMComputations(
                cls.wdm, t_ref=0.0, Nt_sub=16, n_pad=2, N_sparse=64,
                tdi_config="1st generation", force_backend=BK)

    @staticmethod
    def _dev_of(arr):
        d = getattr(arr, "device", None)
        return None if d is None else int(getattr(d, "id", -1))

    def test_build_device_is_recorded_and_true(self):
        self.assertEqual(self.comp0._build_device, 0)
        self.assertEqual(self._dev_of(self.comp0.wdm_window), 0)

    def test_replica_buffers_live_on_the_target_device(self):
        """The whole point of the replica layer: not a bookkeeping flag, but
        memory that is genuinely resident on the other device."""
        import cupy as cp

        from lisatools.utils.devicereplicas import device_local_gb_comp

        rep = device_local_gb_comp(self.comp0, cp, 1, 0)
        self.assertIsNot(rep, self.comp0)
        self.assertEqual(rep._build_device, 1)
        self.assertEqual(self._dev_of(rep.wdm_window), 1)
        self.assertEqual(self._dev_of(rep.chunk_t_starts), 1)
        # prototype untouched
        self.assertEqual(self._dev_of(self.comp0.wdm_window), 0)
        # ... and numerically identical (deterministic in Nf/Nt/dt)
        np.testing.assert_allclose(cp.asnumpy(rep.wdm_window),
                                   cp.asnumpy(self.comp0.wdm_window))
        self.assertEqual(rep.backend.name, self.comp0.backend.name)

    def test_replica_is_allocate_once_and_primary_is_shared(self):
        import cupy as cp

        from lisatools.utils.devicereplicas import device_local_gb_comp

        a = device_local_gb_comp(self.comp0, cp, 1, 0)
        b = device_local_gb_comp(self.comp0, cp, 1, 0)
        self.assertIs(a, b)
        self.assertIs(device_local_gb_comp(self.comp0, cp, 0, 0), self.comp0)
        self.assertIs(device_local_gb_comp(self.comp0, cp, None, 0), self.comp0)

    def test_tdi_config_replica_is_not_the_primary_s(self):
        """A comp replica around the SHARED TDIConfig would still hold
        ``TDIConfigWrap`` pointers into the primary device's six link tables --
        the same bug one level down."""
        import cupy as cp

        from lisatools.utils.devicereplicas import (device_local_gb_comp,
                                                    device_local_tdi_config)

        t0 = self.comp0.tdi_config
        t1 = device_local_tdi_config(t0, cp, 1, 0)
        self.assertIsNot(t1, t0)
        self.assertIs(device_local_tdi_config(t0, cp, 1, 0), t1)
        self.assertIs(device_local_tdi_config(t0, cp, 0, 0), t0)
        rep = device_local_gb_comp(self.comp0, cp, 1, 0)
        self.assertIsNot(rep.tdi_config, self.comp0.tdi_config)

    def test_assert_comp_device_guard(self):
        import cupy as cp

        from lisatools.globalfit.moves.gbbands import _RoutedBandEngine
        from lisatools.utils.devicereplicas import device_local_gb_comp

        class _V:
            def __init__(self, d):
                self.device = d

        with self.assertRaises(RuntimeError):
            _RoutedBandEngine._assert_comp_device(self.comp0, _V(1))
        _RoutedBandEngine._assert_comp_device(self.comp0, _V(0))  # no raise
        rep = device_local_gb_comp(self.comp0, cp, 1, 0)
        _RoutedBandEngine._assert_comp_device(rep, _V(1))         # no raise
        self.assertEqual(_RoutedBandEngine._comp_build_device(self.comp0), 0)
        self.assertEqual(_RoutedBandEngine._comp_build_device(rep), 1)


def _build_fd_aca(gpus, n_acs=6, nch=3):
    """A real (len(gpus)-shard) FD AnalysisContainerArray."""
    import cupy as cp

    from lisatools.analysiscontainer import (AnalysisContainer,
                                             AnalysisContainerArray)
    from lisatools.domains import FDSettings
    from lisatools.sensitivity import SensitivityMatrixBase

    fd = FDSettings(N=512, df=1e-6, min_freq=1e-4, max_freq=3e-4,
                    force_backend=BK)
    acs = []
    for i in range(n_acs):
        with cp.cuda.Device(gpus[i % len(gpus)]):
            arr = cp.zeros((nch, fd.N_active), dtype=cp.complex128)
            arr[:] = float(i + 1)
            dom = fd.associated_class(arr, fd)
            sm = SensitivityMatrixBase(fd, skip_inv_det=True)
            sm.sens_mat = cp.ones((nch, fd.N_active), dtype=cp.float64)
            sm.invC = cp.ones((nch, fd.N_active), dtype=cp.float64)
            sm.channel_shape = (nch,)
            acs.append(AnalysisContainer(dom, sm))
    with cp.cuda.Device(gpus[0]):
        aca = AnalysisContainerArray(acs, gpus=list(gpus))
    return fd, aca


@requires_2gpu
class RealShardViewTest(unittest.TestCase):
    """``_ShardHolderView`` against a REAL 2-shard ACA.

    The CPU fake hand-writes ``gpu_splits`` / ``split_map`` / ``ac_to_intra``;
    here the ACA derives them itself, so a disagreement between the router's
    ``_partition`` and the ACA's own layout would show up.
    """

    @classmethod
    def setUpClass(cls):
        cls.fd, cls.aca = _build_fd_aca([0, 1])

    def test_aca_really_shards(self):
        self.assertEqual(len(self.aca.linear_data_arr), 2)
        self.assertEqual(int(self.aca.linear_data_arr[0].device.id), 0)
        self.assertEqual(int(self.aca.linear_data_arr[1].device.id), 1)

    def test_view_presents_one_shard_per_split(self):
        from lisatools.globalfit.moves.gbbands import _ShardHolderView

        for s in range(2):
            v = _ShardHolderView(self.aca, s)
            rows = np.asarray(self.aca.gpu_splits[s])
            self.assertEqual(len(v.linear_data_arr), 1)
            self.assertEqual(v.device, s)
            self.assertEqual(len(v), len(rows))
            self.assertEqual(v.acs_total_entries, len(rows))
            # zero-copy: the shard's live buffer, not a repack
            self.assertIs(v.linear_data_arr[0], self.aca.linear_data_arr[s])
            np.testing.assert_array_equal(
                np.asarray(v.ac_to_intra), np.arange(len(rows)))

    def test_router_partition_matches_the_aca_layout(self):
        from lisatools.globalfit.moves.gbbands import _RoutedBandEngine

        idx = np.arange(int(self.aca.acs_total_entries))
        parts = _RoutedBandEngine._partition(self.aca, idx)
        for s, (pos, intra, _) in enumerate(parts):
            rows = np.asarray(self.aca.gpu_splits[s])
            np.testing.assert_array_equal(np.sort(idx[pos]), np.sort(rows))
            # the router's intra ids must agree with the ACA's own
            np.testing.assert_array_equal(
                intra, np.asarray(self.aca.ac_to_intra)[idx[pos]])

    def test_as_wdm_holder_refuses_a_real_multi_shard_aca(self):
        """The guard that turns a silently wrong multi-GPU WDM likelihood
        into a raise."""
        from gbgpu.gbcomps import GBWDMComputations
        from lisatools.domains import WDMSettings

        import cupy as cp

        with cp.cuda.Device(0):
            comp = GBWDMComputations(
                WDMSettings(Nf=32, Nt=64, dt=15.0, force_backend=BK),
                t_ref=0.0, Nt_sub=16, n_pad=2, N_sparse=64,
                tdi_config="1st generation", force_backend=BK)
        with self.assertRaises(NotImplementedError) as ctx:
            comp._as_wdm_holder(self.aca)
        self.assertIn("single-shard", str(ctx.exception))

        _fd, single = _build_fd_aca([0], n_acs=2)
        self.assertIs(comp._as_wdm_holder(single), single)


@requires_2gpu
class RoutedInformationMatrixParityTest(unittest.TestCase):
    """1-GPU vs 2-GPU ``route_information_matrix`` must agree exactly."""

    def test_parity(self):
        import cupy as cp

        from gbgpu.gbcomps import GBFDComputations
        from lisatools.globalfit.moves.gbbands import _RoutedBandEngine

        n = 6
        fd2, aca2 = _build_fd_aca([0, 1], n_acs=n)
        fd1, aca1 = _build_fd_aca([0], n_acs=n)
        with cp.cuda.Device(0):
            # tdi_type="AET" so the FD binding's per-noise row size is
            # nchannels * n_rfft, matching the (nch, N_active) invC these
            # fixtures build (the "XYZ" default expects nchannels**2).
            comp = GBFDComputations(fd2, t_ref=0.0, N_sparse=64,
                                    tdi_config="1st generation",
                                    tdi_type="AET", force_backend=BK)

        p = np.zeros((n, 9))
        p[:, 0] = 1e-22
        p[:, 1] = np.linspace(1.5e-4, 2.5e-4, n)
        p[:, 2] = 1e-17
        p[:, 3:] = [0.5, 0.3, 0.4, 0.2, 0.1, 0.6]
        idx = np.arange(n, dtype=np.int32)

        with cp.cuda.Device(0):
            one = _RoutedBandEngine.route_information_matrix(
                comp, aca1, cp.asarray(p), inds=[0, 1, 2],
                noise_index=cp.asarray(idx))
        two = _RoutedBandEngine.route_information_matrix(
            comp, aca2, cp.asarray(p), inds=[0, 1, 2],
            noise_index=cp.asarray(idx))
        a, b = cp.asnumpy(one), cp.asnumpy(two)
        self.assertEqual(a.shape, (n, 3, 3))
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(a, b)


def _build_stft(gpus, orbits_mode="replica", nbands=6, nch=3, NT=8,
                big_dt=21600.0):
    """A real (len(gpus)-shard) STFT band ACA + engine.

    ``orbits_mode`` selects how each band's sensitivity backend gets its
    orbits: ``"replica"`` mirrors the production path in
    ``SubBandBuffer._build_band_ac_list``; ``"shared"`` reproduces the
    pre-fix behaviour (one Orbits for every device).
    """
    import cupy as cp

    from gbgpu.gbcomps import STFTGBComputations
    from lisatools.analysiscontainer import (AnalysisContainer,
                                             AnalysisContainerArray)
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.domains import STFTSettings
    from lisatools.globalfit.moves.gb_likelihood import STFTBandLikelihoodEngine
    from lisatools.sensitivity import XYZSensitivityBackend
    from lisatools.utils.devicereplicas import device_local_orbits

    settings = STFTSettings(
        t0=10.0 * 86400.0, dt=big_dt, df=1.0 / big_dt, NT=NT, NF=128,
        min_freq=4.16e-3, max_freq=4.26e-3, force_backend=BK)
    data_shape = (nch, settings.NT, settings.NF_active)
    sens_shape = (nch, nch, settings.NT, settings.NF_active)

    with cp.cuda.Device(gpus[0]):
        shared_orbits = EqualArmlengthOrbits(force_backend=BK)

    ac_list = []
    for b in range(nbands):
        dev = gpus[b % len(gpus)]
        with cp.cuda.Device(dev):
            if orbits_mode == "shared":
                orb = shared_orbits
            else:
                orb = device_local_orbits(shared_orbits, cp, int(gpus[0]))
            res_data = cp.zeros(data_shape, dtype=cp.complex128)
            res_data[:] = (b + 1) * 1e-21
            dom = settings.associated_class(res_data, settings)
            sm = XYZSensitivityBackend(orbits=orb, settings=settings,
                                       force_backend=BK)
            sm.sens_mat = cp.zeros(sens_shape, dtype=cp.complex128)
            invC = cp.zeros(sens_shape, dtype=cp.complex128)
            for j in range(nch):
                invC[j, j] = 1.0
            sm.invC = invC
            sm.channel_shape = sens_shape[: -len(settings.basis_shape_active)]
            ac_list.append(AnalysisContainer(dom, sm))

    with cp.cuda.Device(gpus[0]):
        aca = AnalysisContainerArray(
            ac_list, gpus=list(gpus),
            domain_group_kwargs=dict(tdi_type="XYZ", window_alpha=0.0,
                                     use_midpoint=False))
        gb = STFTGBComputations(
            stft_comps=aca.cpp_splits[0], T=NT * big_dt, t_ref=0.0,
            force_backend=BK, n_side_bins=3, window_factor=1.0,
            freq_from_tdi_phase=False)
        engine = STFTBandLikelihoodEngine(
            gb_stft_comp=gb, basis_settings=settings, nchannels=nch,
            tdi_channel_setup="XYZ")
    return settings, aca, gb, engine


def _stft_params(settings, n):
    f0 = (settings.ind_min + settings.NF_active // 2) * settings.df
    p = np.zeros((n, 9))
    p[:] = np.array([1.0e-21, f0, 1.0e-17, 0.0, 1.3, 0.6, 0.7, 2.0, 0.3])
    return p


@requires_2gpu
class STFTMultiGPUTest(unittest.TestCase):
    """The STFT arm -- this branch's primary GB path.

    It shards INSIDE the engine (``_split_plan``) rather than behind
    ``_RoutedBandEngine``, so none of the router tests touch it. These are the
    only tests that exercise those four per-split loops.
    """

    N = 6

    def test_one_vs_two_shard_lnL_agree(self):
        import cupy as cp

        s1, aca1, _gb1, eng1 = _build_stft([0])
        s2, aca2, _gb2, eng2 = _build_stft([0, 1])
        self.assertEqual(len(aca1.linear_data_arr), 1)
        self.assertEqual(len(aca2.linear_data_arr), 2)

        p = _stft_params(s1, self.N)
        idx = np.arange(self.N, dtype=np.int32)
        with cp.cuda.Device(0):
            ll1 = cp.asnumpy(eng1.get_ll(
                aca1, cp.asarray(p), data_index=cp.asarray(idx),
                noise_index=cp.asarray(idx), N_vals=None, waveform_kwargs={}))
            ll2 = cp.asnumpy(eng2.get_ll(
                aca2, cp.asarray(p), data_index=cp.asarray(idx),
                noise_index=cp.asarray(idx), N_vals=None, waveform_kwargs={}))
        self.assertTrue(np.all(np.isfinite(ll1)), f"{ll1}")
        np.testing.assert_array_equal(ll1, ll2)

    def test_comp_for_split_is_device_local_and_group_bound(self):
        import cupy as cp

        _s, aca, gb, eng = _build_stft([0, 1])
        self.assertEqual(gb._build_device, 0)
        for s in range(2):
            dev = aca.gpus[s]
            with cp.cuda.Device(dev):
                c = eng._comp_for_split(aca, s, dev)
            self.assertEqual(c._build_device, dev)
            self.assertIs(c.stft_comps, aca.cpp_splits[s])
        # the prototype's own split allocates nothing
        self.assertIs(eng._comp_for_split(aca, 0, 0), gb)

    def test_split_plan_checks_peer_access(self):
        """The per-split loops store into caller-device arrays, so they need
        peer access. Absent it, those stores are an illegal memory access with
        no diagnostic -- the guard turns that into a named error."""
        import cupy as cp

        from lisatools.utils import device as devmod

        _s, aca, _gb, eng = _build_stft([0, 1])
        idx = np.arange(self.N, dtype=np.int32)

        # Real topology here supports it, so the plan builds cleanly ...
        with cp.cuda.Device(0):
            plan = eng._split_plan(aca, cp.asarray(idx), cp.asarray(idx))
        self.assertEqual(len(plan), 2)

        # ... and with peer access reported unavailable it must refuse.
        devmod._PEER_ACCESS_OK.clear()
        real = cp.cuda.runtime.deviceCanAccessPeer
        cp.cuda.runtime.deviceCanAccessPeer = lambda a, b: 0
        try:
            with cp.cuda.Device(0):
                with self.assertRaises(RuntimeError) as ctx:
                    eng._split_plan(aca, cp.asarray(idx), cp.asarray(idx))
            self.assertIn("peer access", str(ctx.exception))
        finally:
            cp.cuda.runtime.deviceCanAccessPeer = real
            devmod._PEER_ACCESS_OK.clear()

    def test_shared_orbits_across_devices_is_what_the_fix_avoids(self):
        """Documents the defect ``_build_band_ac_list`` now avoids.

        Building every band's sensitivity backend around ONE ``Orbits``
        instance leaves bands on non-primary devices dereferencing the primary
        device's C++ orbit tables. Measured as an illegal memory access in
        ``Detector.cu`` -- on a node where P2P is ENABLED, so the orbit
        pointers are not peer-mapped and this is not merely a peer-access tax.

        Not executed: a CUDA illegal access poisons the whole context and
        would take the rest of the suite with it. The positive control is
        :meth:`test_one_vs_two_shard_lnL_agree`, which passes precisely
        because the production path uses ``device_local_orbits``.
        """
        import inspect

        from lisatools.globalfit.moves import gbbands

        src = inspect.getsource(gbbands.SubBandBuffer._build_band_ac_list)
        self.assertIn("device_local_orbits", src)
        # the swap must happen before the backend is constructed
        self.assertLess(src.index("device_local_orbits"),
                        src.index("type(parent_sb)(**sb_kwargs)"))


if __name__ == "__main__":
    unittest.main()
