"""Knobs, cfg resolution, getter and builder selection for ``EMRI_LIKELIHOOD=direct``;
the fit-side adapter (active-box crop, refusals, kwargs); per-row domain refusals in
``EMRIDirectWDM.batch``; the MPI-safe trajectory-pool start."""
from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

_EMRI_KNOBS = ("EMRI_LIKELIHOOD", "EMRI_BATCH_MAX_SIZE", "EMRI_DIRECT_TABLE",
               "EMRI_DIRECT_RESPONSE", "EMRI_TRAJ_WORKERS")


def _clean_env():
    return mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k not in _EMRI_KNOBS},
                           clear=True)


def _wdm_spec():
    from lisatools.domains import WDMSettings

    return WDMSettings.make_factory(Nf=32, Nt=128)


def _fd_spec():
    from lisatools.domains import FDSettings

    return FDSettings.make_factory()


def _emri(**kw):
    from lisatools.globalfit.stock.erebor.source_runtime import SourceEMRISettings

    with _clean_env():
        return SourceEMRISettings(**kw)


class _Table:
    """A real file for EMRI_DIRECT_TABLE's existence check."""

    def setUp(self):
        fd, self.table = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        self.addCleanup(os.remove, self.table)


class EMRIDirectKnobsTest(unittest.TestCase):
    def test_defaults(self):
        s = _emri()
        self.assertEqual((s.likelihood, s.batch_max_size, s.direct_table, s.direct_response,
                          s.traj_workers), ("auto", 8, None, "dense", 0))

    def test_env_knobs(self):
        from lisatools.globalfit.stock.erebor.source_runtime import SourceEMRISettings

        env = dict(EMRI_LIKELIHOOD="direct", EMRI_BATCH_MAX_SIZE="4", EMRI_DIRECT_TABLE="/x.h5",
                   EMRI_DIRECT_RESPONSE="spline", EMRI_TRAJ_WORKERS="3")
        with mock.patch.dict(os.environ, env):
            s = SourceEMRISettings()
        self.assertEqual((s.likelihood, s.batch_max_size, s.direct_table, s.direct_response,
                          s.traj_workers), ("direct", 4, "/x.h5", "spline", 3))

    def test_move_ctor_batch_default_matches_the_knob(self):
        import inspect

        from lisatools.globalfit.moves import EMRIDirectLikeMove

        p = inspect.signature(EMRIDirectLikeMove.__init__).parameters["batch_max_size"]
        self.assertEqual(p.default, _emri().batch_max_size)

    def test_settings_survive_deepcopy_and_pickle(self):
        import copy
        import pickle

        s = _emri(likelihood="direct", direct_table="/x.h5")
        t = pickle.loads(pickle.dumps(copy.deepcopy(s)))
        self.assertEqual((t.likelihood, t.direct_table), ("direct", "/x.h5"))


class ResolveEMRIDirectCfgTest(_Table, unittest.TestCase):
    def _resolve(self, domain=None, chan="XYZ", **kw):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_emri_direct_cfg

        return resolve_emri_direct_cfg(_emri(**kw), domain_settings=domain or _wdm_spec(),
                                       tdi_chan=chan)

    def test_full_passes_through_every_blocker(self):
        cfg = self._resolve(domain=_fd_spec(), chan="AET", direct_table=None, likelihood="full",
                            direct_response="bogus", batch_max_size=0)
        self.assertEqual(cfg["emri_likelihood"], "full")
        self.assertEqual(cfg["emri_batch_max_size"], 0)

    def test_direct_resolves(self):
        cfg = self._resolve(likelihood="direct", direct_table=self.table, batch_max_size=4,
                            traj_workers=2)
        self.assertEqual(cfg, dict(emri_likelihood="direct", emri_batch_max_size=4,
                                   emri_direct_table=self.table, emri_direct_response="dense",
                                   emri_traj_workers=2))

    def test_direct_needs_no_existing_table(self):
        """The table is found or built by the getter: neither an unset table (the run
        folder's canonical file) nor a pointer to a file not built yet blocks."""
        for table in (None, "/not/built/yet.h5"):
            cfg = self._resolve(likelihood="direct", direct_table=table)
            self.assertEqual((cfg["emri_likelihood"], cfg["emri_direct_table"]), ("direct", table))

    def test_auto_is_the_lookup_template_on_a_wdm_xyz_run(self):
        cfg = self._resolve(direct_table=self.table)                       # the default: auto
        self.assertEqual(cfg["emri_likelihood"], "direct")

    def test_auto_falls_back_to_full_with_one_info_line(self):
        import logging

        for extra, needle in ((dict(domain=_fd_spec()), "not WDM"), (dict(chan="AET"), "XYZ"),
                              (dict(domain=object()), "not identifiable")):
            with self.subTest(needle=needle), self.assertLogs(
                    "lisatools.globalfit.stock.erebor.source_runtime", logging.INFO) as cm:
                cfg = self._resolve(likelihood="auto", direct_table=self.table, **extra)
            self.assertEqual(cfg["emri_likelihood"], "full")
            self.assertEqual(len(cm.output), 1)
            self.assertIn(needle, cm.output[0])

    def test_direct_takes_an_unidentifiable_domain_at_its_word(self):
        cfg = self._resolve(domain=object(), likelihood="direct", direct_table=self.table)
        self.assertEqual(cfg["emri_likelihood"], "direct")

    def test_each_blocker_raises_and_is_named(self):
        cases = [
            (dict(domain=_fd_spec()), "not WDM"),
            (dict(chan="AET"), "XYZ"),
            (dict(direct_response="bogus"), "EMRI_DIRECT_RESPONSE"),
            (dict(batch_max_size=0), "EMRI_BATCH_MAX_SIZE"),
            (dict(traj_workers=-1), "EMRI_TRAJ_WORKERS"),
        ]
        for extra, needle in cases:
            kw = dict(likelihood="direct", direct_table=self.table)
            kw.update(extra)
            with self.subTest(needle=needle), self.assertRaisesRegex(ValueError, needle):
                self._resolve(**kw)

    def test_unknown_mode_raises(self):
        with self.assertRaisesRegex(ValueError, "EMRI_LIKELIHOOD"):
            self._resolve(likelihood="batched")

    def test_plain_emri_settings_block_is_full(self):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_emri_direct_cfg

        cfg = resolve_emri_direct_cfg(SimpleNamespace(), domain_settings=None, tdi_chan="AET")
        self.assertEqual(cfg["emri_likelihood"], "full")


class SourceSignalCfgTest(_Table, unittest.TestCase):
    def test_cfg_carries_the_resolved_emri_keys(self):
        from lisatools.globalfit.stock.erebor.source_runtime import (
            SourceMBHSettings,
            source_signal_cfg,
        )

        gs = SimpleNamespace(tdi_chan="XYZ", tdi_gen_str="2nd generation", nchannels=3,
                             data_mode="synthetic", sobbh_reference_time=0.0,
                             mbh_waveform_t0=0.0, min_freq=1e-4, max_freq=2.5e-2)
        emri = _emri(likelihood="direct", direct_table=self.table, batch_max_size=2)
        cfg = source_signal_cfg(gs, SourceMBHSettings(likelihood="full"), mock.MagicMock(), emri,
                                domain_settings=_wdm_spec())
        self.assertEqual((cfg["emri_likelihood"], cfg["emri_batch_max_size"],
                          cfg["emri_direct_table"]), ("direct", 2, self.table))
        # the MBH keys are untouched
        self.assertEqual(cfg["mbh_likelihood"], "full")


class EMRIDirectBuilderTest(unittest.TestCase):
    def test_builder_class_attrs(self):
        from lisatools.globalfit.moves import EMRIDirectLikeMove
        from lisatools.globalfit.recipe import EMRIDirectMoveBuilder, EMRIMoveBuilder

        self.assertTrue(issubclass(EMRIDirectMoveBuilder, EMRIMoveBuilder))
        self.assertIs(EMRIDirectMoveBuilder.move_class, EMRIDirectLikeMove)
        self.assertFalse(EMRIDirectMoveBuilder.use_dcga)
        # still strips the FEW-only key from the LIKELIHOOD kwargs
        self.assertIn("mode_selection_threshold", EMRIDirectMoveBuilder.like_kwargs_strip_keys)


class RuntimeSelectionTest(unittest.TestCase):
    def test_direct_selects_the_direct_builder(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        built = {}

        class _FakeBuilder:
            def __init__(self, **kw):
                built.update(kw)

            def build(self, *a):
                return [], ["the-move"]

        with mock.patch.object(sr, "EMRIDirectMoveBuilder", _FakeBuilder), \
                mock.patch.object(sr, "EMRIMoveBuilder") as stock:
            move = sr.build_emri_move_runtime(
                mock.MagicMock(), None, None, None,
                dict(emri_likelihood="direct", emri_batch_max_size=5))
        self.assertEqual(move, "the-move")
        stock.assert_not_called()
        self.assertEqual(built["batch_max_size"], 5)
        self.assertIsInstance(built["direct_gen"], sr.DeviceLocalWaveGen)
        self.assertIs(built["direct_gen"]._getter, sr.get_emri_direct_gen)
        self.assertIsInstance(built["wave_gen"], sr.DeviceLocalWaveGen)
        self.assertIs(built["wave_gen"]._getter, sr.get_emri_wave_wrap)

    def _engine_gen_calls(self, mode):
        from types import SimpleNamespace

        from lisatools.globalfit.stock.erebor import source_runtime as sr

        calls = []

        def getter(name):
            def get(general_info, cfg):
                return lambda *p, **kw: calls.append((name, p, kw)) or name
            return get

        gi = SimpleNamespace(force_backend="cpu", gpus=None, orbits=SimpleNamespace(xp=np))
        with mock.patch.object(sr, "get_emri_direct_gen", getter("direct")), \
                mock.patch.object(sr, "get_emri_wave_wrap", getter("production")):
            gen = sr.SourceSignalGen("emri", None, gi, dict(emri_likelihood=mode))
            out = gen(1.0, 2.0, apply_transform=False, mode_selection_threshold=1e-3)
        return out, calls

    def test_engine_template_generator_follows_the_likelihood(self):
        """direct: residual rebuilds and the move's expose/fold (the containers' installed
        generator) use the direct template, the same family the move scores with."""
        out, calls = self._engine_gen_calls("direct")
        self.assertEqual(out, "direct")
        self.assertEqual(calls, [("direct", (1.0, 2.0), {"mode_selection_threshold": 1e-3})])
        out, calls = self._engine_gen_calls("full")
        self.assertEqual(out, "production")

    def test_full_keeps_the_stock_builder(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        stock = mock.MagicMock()
        stock.return_value.build.return_value = ([], ["stock-move"])
        with mock.patch.object(sr, "EMRIDirectMoveBuilder") as direct, \
                mock.patch.object(sr, "EMRIMoveBuilder", stock):
            move = sr.build_emri_move_runtime(mock.MagicMock(), None, None, None,
                                              dict(emri_likelihood="full"))
        self.assertEqual(move, "stock-move")
        direct.assert_not_called()


class DirectGetterTest(unittest.TestCase):
    """``get_emri_direct_gen`` with the heavy classes stubbed."""

    def setUp(self):
        from lisatools.domains import WDMSettings
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        self.sr = sr
        sr._WAVE_WRAP_CACHE.clear()
        self.addCleanup(sr._WAVE_WRAP_CACHE.clear)
        # the run domain: 6-month-like crop (60 layers), tiny grid
        self.dom = WDMSettings(32, 256, 10.0, min_freq=2e-3, max_freq=2e-2, min_time=60 * 320.0,
                               max_time=196 * 320.0, force_backend="cpu")
        self.few_gen = object()
        wrap = SimpleNamespace(wave_gen=SimpleNamespace(
            waveform_gen=SimpleNamespace(waveform_generator=self.few_gen)))
        self.gi = SimpleNamespace(force_backend="cpu", data_t0=1.0e8, file_store_dir="/run/folder")
        self.cfg = dict(nchannels=3, tdi_gen_str="2nd generation", data_mode="mojito",
                        emri_mode_selection_threshold=1e-3, emri_direct_table="/t.h5",
                        emri_direct_response="dense", emri_traj_workers=2)
        self.table = SimpleNamespace(Nf=32, data_dt=10.0)
        self.direct_kw = {}

        class _Direct:
            def __init__(inner, few_gen, table, wdm, **kw):
                inner.few_gen, inner.table, inner.wdm, inner.pixel_edge = few_gen, table, wdm, 8
                self.direct_kw.update(kw, few_gen=few_gen, table=table, wdm=wdm)

        self.ensured = []
        self.patches = [
            mock.patch("lisatools.wdm_lookup_store.ensure_lookup_table",
                       lambda path, **kw: self.ensured.append((path, kw)) or "found"),
            mock.patch.object(sr, "_wrap_device_and_orbits",
                              lambda gi: (np, None, "orbits", self.dom)),
            mock.patch.object(sr, "get_emri_wave_wrap", lambda gi, cfg: wrap),
            mock.patch("lisatools.domains.WDMLookupTable.from_file",
                       lambda path, force_backend=None: self.table),
            mock.patch("lisatools.sources.emri.wdm_direct.EMRIDirectWDM", _Direct),
            mock.patch.object(sr, "TDIConfig", lambda s, force_backend=None: ("tdi", s)),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_builds_on_the_production_generator_epoch_and_full_grid(self):
        gen = self.sr.get_emri_direct_gen(self.gi, self.cfg)
        kw = self.direct_kw
        self.assertIs(kw["few_gen"], self.few_gen)          # ONE FEW construction per device
        self.assertIs(kw["table"], self.table)
        self.assertEqual((kw["wdm"].Nf, kw["wdm"].Nt, kw["wdm"].data_dt), (32, 256, 10.0))
        self.assertEqual((kw["wdm"].ind_min_t, kw["wdm"].Nt_active), (0, 256))   # FULL grid
        self.assertEqual(kw["t_start"], self.sr.MOJITO_REFERENCE_TIME)
        self.assertEqual(kw["data_t0"], 1.0e8)
        self.assertEqual(kw["response"], "dense")
        self.assertEqual(kw["orbits"], "orbits")
        self.assertIs(gen.domain_settings, self.dom)
        self.assertEqual(gen.runtime_kwargs, {"mode_selection_threshold": 1e-3})
        self.assertEqual((gen.nchannels, gen.traj_workers), (3, 2))
        # cached per (device, threshold, table, response)
        self.assertIs(self.sr.get_emri_direct_gen(self.gi, self.cfg), gen)
        other = self.sr.get_emri_direct_gen(self.gi, dict(self.cfg, emri_mode_selection_threshold=1e-5))
        self.assertIsNot(other, gen)

    def test_table_is_ensured_at_the_pointer_or_in_the_run_folder(self):
        """A pointer is found-or-built where it points; unset, the canonical file in the
        run's folder -- the same path on every restart."""
        self.sr.get_emri_direct_gen(self.gi, self.cfg)
        self.assertEqual(self.ensured[-1][0], "/t.h5")
        self.assertEqual((self.ensured[-1][1]["Nf"], self.ensured[-1][1]["dt"]), (32, 10.0))
        self.sr._WAVE_WRAP_CACHE.clear()
        self.sr.get_emri_direct_gen(self.gi, dict(self.cfg, emri_direct_table=None))
        self.assertEqual(self.ensured[-1][0],
                         "/run/folder/wdm_lookup_emri_cx_NF32_DT10_TL32_fd8x0p01_nld2.h5")

    def test_synthetic_epoch_is_the_data_start(self):
        self.sr.get_emri_direct_gen(self.gi, dict(self.cfg, data_mode="synthetic"))
        self.assertEqual(self.direct_kw["t_start"], 1.0e8)

    def test_table_grid_mismatch_raises(self):
        self.table.data_dt = 5.0
        with self.assertRaisesRegex(ValueError, "built for Nf=32, dt=5.0"):
            self.sr.get_emri_direct_gen(self.gi, self.cfg)

    def test_non_wdm_domain_raises(self):
        with mock.patch.object(self.sr, "_wrap_device_and_orbits",
                               lambda gi: (np, None, "orbits", object())):
            with self.assertRaisesRegex(ValueError, "WDM run domain"):
                self.sr.get_emri_direct_gen(self.gi, self.cfg)


# ---------------------------------------------------------------------------
# the fit-side adapter
# ---------------------------------------------------------------------------

class _FakeBatchDirect:
    """``EMRIDirectWDM.batch`` contract on the full grid: entry (r, c, m, n) encodes
    row r, channel c, pixel (m, n); rows whose column 0 is 1 are FEW-refused."""

    pixel_edge = 8

    def __init__(self, wdm, nch=3):
        self.wdm, self.nch = wdm, nch
        self.few_gen = SimpleNamespace(inspiral_kwargs={})
        self.calls = []
        self.last_failed_rows = []
        self.last_stats = {}

    def batch(self, rows, chunk_rows=16, consume=None, skip_domain_errors=False, **kw):
        self.calls.append((len(rows), chunk_rows, skip_domain_errors, dict(kw)))
        n = len(rows)
        r, c, m, t = np.meshgrid(np.arange(n), np.arange(self.nch), np.arange(self.wdm.Nf),
                                 np.arange(self.wdm.Nt), indexing="ij")
        arr = 1e6 * r + 1e5 * c + 1e3 * m + t + 1.0
        self.last_failed_rows = [i for i, p in enumerate(rows) if p[0] == 1.0]
        arr[self.last_failed_rows] = 0.0
        self.last_stats = dict(rows=n)
        return arr


class _AdapterBase:
    def setUp(self):
        from lisatools.domains import WDMSettings

        self.full = WDMSettings(32, 256, 10.0, force_backend="cpu")
        self.dom = WDMSettings(32, 256, 10.0, min_freq=2e-3, max_freq=2e-2, min_time=60 * 320.0,
                               max_time=196 * 320.0, force_backend="cpu")
        self.direct = _FakeBatchDirect(self.full)

    def _gen(self, **kw):
        from lisatools.sources.emri.direct_signal_gen import EMRIDirectWDMSignalGen

        return EMRIDirectWDMSignalGen(self.direct, self.dom, **kw)


class EMRIDirectAdapterTest(_AdapterBase, unittest.TestCase):
    def test_crops_to_the_active_box_and_reports_refusals(self):
        gen = self._gen(runtime_kwargs={"mode_selection_threshold": 1e-3})
        rows = np.array([[0.0] * 14, [1.0] * 14, [2.0] * 14])
        arr, ok = gen.templates(rows)
        self.assertEqual(arr.shape, (3, 3, self.dom.Nf_active, self.dom.Nt_active))
        np.testing.assert_array_equal(ok, [True, False, True])
        m0, n0 = self.dom.ind_min_f, self.dom.ind_min_t
        self.assertEqual(arr[2, 1, 0, 0], 2e6 + 1e5 + 1e3 * m0 + n0 + 1.0)
        self.assertEqual(arr[0, 2, -1, -1], 2e5 + 1e3 * self.dom.ind_max_f + self.dom.ind_max_t + 1.0)
        self.assertTrue(np.all(arr[1] == 0.0))
        # one batch call for the rows, refusals skipped, the threshold delivered
        self.assertEqual(self.direct.calls, [(3, 3, True, {"mode_selection_threshold": 1e-3})])

    def test_call_kwargs_win_over_runtime_kwargs(self):
        gen = self._gen(runtime_kwargs={"mode_selection_threshold": 1e-3})
        gen.templates(np.zeros((1, 14)), mode_selection_threshold=1e-5)
        self.assertEqual(self.direct.calls[-1][3], {"mode_selection_threshold": 1e-5})

    def test_nchannels_keeps_the_leading_channels(self):
        arr, _ = self._gen(nchannels=2).templates(np.zeros((1, 14)))
        self.assertEqual(arr.shape[1], 2)

    def test_single_template_call(self):
        from lisatools.domains import WDMSignal
        from lisatools.utils.exceptions import WaveformDomainError

        gen = self._gen()
        sig = gen(*np.full(14, 2.0))
        self.assertIsInstance(sig, WDMSignal)
        self.assertEqual(sig.arr.shape, (3, self.dom.Nf_active, self.dom.Nt_active))
        with self.assertRaises(WaveformDomainError):
            gen(*np.full(14, 1.0))

    def test_grid_and_edge_guards(self):
        from lisatools.domains import WDMSettings

        self.direct.wdm = WDMSettings(32, 128, 10.0, force_backend="cpu")
        with self.assertRaisesRegex(ValueError, "wavelet grid"):
            self._gen()
        self.direct.wdm = self.full
        self.dom = WDMSettings(32, 256, 10.0, min_freq=2e-3, max_freq=2e-2, min_time=4 * 320.0,
                               max_time=196 * 320.0, force_backend="cpu")
        with self.assertRaisesRegex(ValueError, "edge layers"):
            self._gen()


class _FakeBandDirect(_FakeBatchDirect):
    """The same contract with ``f_band``: batch returns layers [m_lo, m_hi) only."""

    f_band = None

    def batch(self, rows, **kw):
        arr = super().batch(rows, **kw)
        return arr if self.f_band is None else arr[:, :, self.f_band[0]:self.f_band[1]]


class AdapterBandTest(_AdapterBase, unittest.TestCase):
    """The adapter asks the direct generator for the run's active frequency band only (the
    template keeps ~1/8 of the 6-month grid) and crops time; same array as the full grid."""

    def test_band_is_requested_and_the_crop_is_unchanged(self):
        want, _ = self._gen().templates(np.array([[0.0] * 14, [2.0] * 14]))      # full-grid fake
        self.direct = _FakeBandDirect(self.full)
        gen = self._gen()
        sl = self.dom.active_slice_f
        self.assertEqual(self.direct.f_band, (sl.start, sl.stop))
        got, ok = gen.templates(np.array([[0.0] * 14, [2.0] * 14]))
        np.testing.assert_array_equal(got, want)
        self.assertTrue(np.all(ok))

    def test_a_generator_ignoring_the_band_is_caught(self):
        self.direct = _FakeBandDirect(self.full)
        gen = self._gen()
        self.direct.batch = lambda rows, **kw: _FakeBatchDirect.batch(self.direct, rows, **kw)   # full grid
        with self.assertRaisesRegex(ValueError, "band"):
            gen.templates(np.zeros((1, 14)))


class _FakeCache:
    """``TrajectoryCache`` stand-in: runs the capture fn per row, records clears."""

    def __init__(self, fail=None):
        self.precomputed, self.cleared, self.fail = [], 0, fail

    def precompute(self, fn, rows, pool):
        if self.fail is not None:
            raise self.fail
        for r in rows:
            fn(*r)
        self.precomputed.append(len(rows))
        return dict(rows=len(rows))

    def clear(self):
        self.cleared += 1


class AdapterTrajectoryPoolTest(_AdapterBase, unittest.TestCase):
    """The pool is used for chunks at least as large as the pool, captured through
    ``EMRIDirectWDM._mode_list`` with the call's kwargs, cleared after every batch,
    and disabled (serial from then on) on any failure."""

    def _pooled(self, workers, fail=None):
        gen = self._gen(traj_workers=workers, runtime_kwargs={"mode_selection_threshold": 1e-3})
        self.mode_list = []
        self.direct._mode_list = lambda p, kw: self.mode_list.append((float(p[0]), dict(kw)))
        cache = _FakeCache(fail)

        def start():
            gen._traj_pool, gen._traj_cache = object(), cache

        gen._start_pool = start
        gen.close_pool = mock.MagicMock(side_effect=lambda: setattr(gen, "_traj_cache", None))
        return gen, cache

    def test_chunk_at_least_the_pool_is_precomputed_then_cleared(self):
        gen, cache = self._pooled(2)
        with self.assertLogs("lisatools.sources.emri.direct_signal_gen", "INFO") as cm:
            arr, ok = gen.templates(np.array([[0.0] * 14, [2.0] * 14, [3.0] * 14]))
        self.assertIn("trajectory pool: 1 batches, 3 rows", cm.output[0])
        self.assertEqual(gen.pool_totals["rows"], 3)
        self.assertEqual(cache.precomputed, [3])
        self.assertEqual([m[0] for m in self.mode_list], [0.0, 2.0, 3.0])
        self.assertEqual(self.mode_list[0][1], {"mode_selection_threshold": 1e-3})
        self.assertEqual(cache.cleared, 1)
        self.assertTrue(np.all(ok))

    def test_chunk_smaller_than_the_pool_is_serial(self):
        gen, cache = self._pooled(4)
        gen.templates(np.zeros((3, 14)))
        gen.templates(np.zeros((1, 14)))
        self.assertEqual(cache.precomputed, [])

    def test_failure_disables_the_pool_and_the_batch_still_builds(self):
        gen, cache = self._pooled(2, fail=RuntimeError("BrokenProcessPool stand-in"))
        with self.assertLogs("lisatools.sources.emri.direct_signal_gen", "WARNING") as cm:
            arr, ok = gen.templates(np.zeros((2, 14)))
        self.assertIn("trajectory pool disabled", cm.output[0])
        self.assertEqual(gen.traj_workers, 0)
        gen.close_pool.assert_called_once()
        self.assertEqual(arr.shape[0], 2)


class _ChirpIntegrator:
    """A synthetic trajectory integrator: Phi_phi = w t + wddd t^4 / 24, Phi_theta = 0,
    Phi_r = 0.3 w t (FEW clock), with exact derivatives (the curvature trigger fires once
    (pi/3) |fddot| tau^3 passes 0.1 rad, fddot growing linearly in t)."""

    def __init__(self, w, wddd):
        self.w, self.wddd = w, wddd
        self.generating_trajectory = False
        self.massratio = 1.0

    def eval_integrator_spline(self, t):
        t = np.asarray(t, dtype=float)
        out = np.zeros((t.size, 6))
        out[:, 3] = self.w * t + self.wddd * t ** 4 / 24.0
        out[:, 5] = 0.3 * self.w * t
        return out

    def eval_integrator_derivative_spline(self, t, order=1):
        t = np.asarray(t, dtype=float)
        out = np.zeros((t.size, 6))
        if order == 1:
            out[:, 3] = self.w + self.wddd * t ** 3 / 6.0
            out[:, 5] = 0.3 * self.w
        elif order == 2:
            out[:, 3] = self.wddd * t ** 2 / 2.0
        elif order == 3:
            out[:, 3] = self.wddd * t
        return out


def _synthetic_holder(t_end, K=60):
    """A FEW-like knots holder ending at ``t_end``: modes (2,2,0,0) and (3,3,0,1) with smooth
    Teukolsky amplitudes, Ylms for the m >= 0 modes then their -m partners."""
    t = np.linspace(0.0, t_end, K)
    x = t / t_end
    teuk = np.stack([(1.0 + 0.2 * x) * np.exp(0.3j * x), (0.5 - 0.1 * x) * np.exp(-0.2j * x)], axis=1)
    return SimpleNamespace(t_arr=t, teuk_modes=teuk,
                           ylms=np.array([0.5 + 0.2j, 0.3 - 0.1j, 0.4 - 0.25j, -0.2 + 0.15j]),
                           ls=np.array([2, 3]), ms=np.array([2, 3]), ks=np.array([0, 0]), ns=np.array([0, 1]),
                           phases=None, freqs=None, integrate_backwards=False)


#: Synthetic rows (column 0 tags the row) -> (w, wddd, t_end) on the FEW clock of the
#: real-generator fixture (window from 2e4 s, 128 pixels of 3600 s): two rows hand off inside the
#: window (m = 3 harmonics, near pixels 88 and 98) and plunge at pixels 100 and 112; the third is
#: slow and outlasts the window (lookup only).
_SYNTH_ROWS = {1e6: (2 * np.pi * 1.5e-3, 2.0e-19, 2.0e4 + 100 * 3600.0),
               2e6: (2 * np.pi * 1.3e-3, 1.8e-19, 2.0e4 + 112 * 3600.0),
               3e6: (2 * np.pi * 1.2e-3, 0.0, 2.0e6)}


def _synthetic_direct():
    """The REAL EMRIDirectWDM on CPU (spline response, Python lookup, a real n_ref table, the real
    batch / chunk-row / single-template code) with only FEW replaced: ``_few_knots`` returns the
    synthetic holder of the row's tag and installs its integrator. Returns ``(gen, rows,
    refusing_row, tmpdir)``."""
    from lisatools.domains import WDMLookupTable, WDMSettings
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sources.emri import wdm_direct as wd

    wdm = WDMSettings(Nf=64, Nt=128, dt=56.25, force_backend="cpu")
    nf, md, mr = WDMLookupTable.apply_eps_frequency(0.005, wdm, m_ref=20, num_layers_diff=2)
    tmp = tempfile.TemporaryDirectory()
    table = WDMLookupTable(wdm, 1, m_ref=mr, norm_freq_single_layer=nf, m_diffs=md,
                           fdot_vals=WDMLookupTable.apply_eps_fdot(0.05, wdm, fdot_max_factor=0.5),
                           store_path=os.path.join(tmp.name, "t.h5"), batch_size_gen=32,
                           build_kind="n_ref_complex", time_layers=32)

    class _Synthetic(wd.EMRIDirectWDM):
        def _few_knots(self, few_args, few_kwargs, mode_selection=None):
            if float(few_args[0]) not in _SYNTH_ROWS:
                raise ValueError("p0 is outside of our domain of validity.")
            w, wddd, t_end = _SYNTH_ROWS[float(few_args[0])]
            self.few_gen.inspiral_generator.inspiral_generator = _ChirpIntegrator(w, wddd)
            return _synthetic_holder(t_end)

    few = SimpleNamespace(inspiral_generator=SimpleNamespace(inspiral_generator=None), inspiral_kwargs={})
    t_start = 1.0e7
    gen = _Synthetic(few, table, wdm, orbits=None, tdi_config=TDIConfig("2nd generation", force_backend="cpu"),
                     t_start=t_start, data_t0=t_start + 2.0e4, Nt_sub=32, fine_dt_plunge=240.0,
                     response="spline", lookup="python")
    rows = [np.array([m, 10.0, 0.9, 10.0, 0.3, 1.0, 1.0, qs, phis, 0.7, 4.0, 0.2, 0.0, 0.7])
            for m, qs, phis in ((1e6, 1.1, 2.3), (2e6, 0.6, 4.1), (3e6, 2.0, 0.5))]
    refuses = np.array([9e6, 10.0, 0.9, 10.0, 0.3, 1.0, 1.0, 1.4, 1.0, 0.7, 4.0, 0.2, 0.0, 0.7])
    return gen, rows, refuses, tmp


class BatchDomainRefusalTest(unittest.TestCase):
    """``EMRIDirectWDM.batch(skip_domain_errors=)``: a FEW domain refusal of one row; the
    plunge-chunk rows batched (``_chunk_rows_batch``) next to the lookup rows."""

    _synth = None

    @classmethod
    def tearDownClass(cls):
        if cls._synth is not None:
            cls._synth[3].cleanup()
            cls._synth = None

    def _direct(self, **kw):
        from lisatools.domains import WDMSettings
        from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

        class _Stub(EMRIDirectWDM):
            # column 0 tags the row: 1 -> FEW out-of-domain, 2 -> a real bug, anything else ->
            # a plunge-chunk row, batched through _chunk_rows_batch (filled with a constant)
            mode_calls = 0
            captured = []

            def _mode_list(self, few_args, few_kwargs):
                type(self).mode_calls += 1
                tag = int(few_args[0])
                if tag == 1:
                    raise ValueError("p0 is outside of our domain of validity.")
                if tag == 2:
                    raise RuntimeError("not a domain error")
                self._last_holder = SimpleNamespace(t_arr=np.array([0.0, 1.0e12]))
                return [], 0.0

            def _chunk_item(self, few_args, modes, chunk_start):
                type(self).captured.append((modes, chunk_start))     # the batch's own FEW call
                return float(few_args[0])

            def _chunk_rows_batch(self, out_arr, chunk_items, m_lo, m_hi, few_kwargs):
                for r, v in chunk_items.items():
                    out_arr[r] = v + 10.0
                n = len(chunk_items)
                return dict(rows=n, windows=0, tail_layers=[0] * n, lookup_pixels=0, chunk_pixels=0,
                            dropped_pixels=0, lookup="kernel")

            def _call_knots(self, few_args, few_kwargs, modes, reuse=None):
                raise AssertionError("a plunge-chunk row went through the single-template path")

        table = SimpleNamespace(fdot_vals=np.array([-1.0, 1.0]), INTERP_METHOD="spline")
        wdm = WDMSettings(32, 64, 10.0, force_backend="cpu")
        return _Stub(None, table, wdm, orbits=None, tdi_config=SimpleNamespace(nchannels=3),
                     t_start=0.0, data_t0=0.0, fine_dt=100.0, **kw)

    def _gen_with_two_chunk_rows_and_one_lookup_row(self):
        """The real generator (:func:`_synthetic_direct`, built once per class): ``self.rows`` =
        two plunge-chunk rows then one lookup-only row; ``self.row_few_refuses`` is out of FEW's
        domain."""
        cls = type(self)
        if cls._synth is None:
            cls._synth = _synthetic_direct()
        gen, self.rows, self.row_few_refuses, _ = cls._synth
        return gen

    def _gen_in_mode(self, chunk_batch, **kw):
        """The fixture's REAL generator rebuilt (the same synthetic FEW stand-in, table and grid)
        with ``EMRI_DIRECT_CHUNK_BATCH`` set to ``chunk_batch`` (read at construction) and the
        constructor overrides ``kw``."""
        with mock.patch.dict(os.environ, {"EMRI_DIRECT_CHUNK_BATCH": "1" if chunk_batch else "0"}):
            return self._sibling(**kw)

    def test_mixed_batch_matches_single_calls(self):
        """Both chunk-row modes (one call for all chunk rows, and one per row, the default): each
        batch row is the single template."""
        from lisatools.utils.utility import asnumpy

        for chunk_batch in (False, True):
            with self.subTest(chunk_batch=chunk_batch):
                gen = self._gen_in_mode(chunk_batch)
                self.assertEqual(gen.chunk_batch, chunk_batch)
                arr = gen.batch(self.rows, chunk_rows=4)
                # read before the single calls below replace last_stats
                self.assertEqual(gen.last_stats["chunk_rows"], 2)
                self.assertEqual(gen.last_stats["chunk_batch"], int(chunk_batch))
                self.assertEqual(gen.last_stats["alone"], 0)
                self.assertEqual(len(gen.last_stats["tail_layers"]), 2)
                self.assertTrue(all(n > 0 for n in gen.last_stats["tail_layers"]))   # both reach the chunk
                for r, p in enumerate(self.rows):
                    one = gen(*p).arr[:, gen._band()[0]:gen._band()[1]]
                    np.testing.assert_allclose(asnumpy(arr[r]), asnumpy(one), rtol=1e-10, atol=1e-13)

    def test_chunk_batch_off_builds_chunk_rows_one_per_call(self):
        """EMRI_DIRECT_CHUNK_BATCH off (the default, what the global fits run): every plunge-chunk
        row goes through its own _chunk_rows_batch call (a one-row {row: item}, in row order); the
        merged stats are the batched call's and the rows its numbers."""
        from lisatools.utils.utility import asnumpy

        on = self._gen_in_mode(True)
        want = asnumpy(on.batch(self.rows, chunk_rows=4))
        st_on = dict(on.last_stats)
        off = self._gen_in_mode(False)
        self.assertFalse(off.chunk_batch)
        with mock.patch.object(off, "_chunk_rows_batch", wraps=off._chunk_rows_batch) as spy:
            got = asnumpy(off.batch(self.rows, chunk_rows=4))
        st_off = dict(off.last_stats)
        self.assertEqual(spy.call_count, 2)                                   # one per chunk row
        self.assertEqual([sorted(c.args[1]) for c in spy.call_args_list], [[0], [1]])
        for k in ("chunk_rows", "windows", "lookup_pixels", "chunk_pixels", "dropped_pixels"):
            self.assertEqual(st_off[k], st_on[k], k)
        self.assertEqual(len(st_off["tail_layers"]), len(st_on["tail_layers"]))
        self.assertEqual(st_off["tail_layers"], st_on["tail_layers"])
        self.assertEqual((st_off["chunk_batch"], st_on["chunk_batch"]), (0, 1))
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-13)
        with mock.patch.object(on, "_chunk_rows_batch", wraps=on._chunk_rows_batch) as spy:
            on.batch(self.rows, chunk_rows=4)
        self.assertEqual(spy.call_count, 1)                                   # paired control
        self.assertEqual(sorted(spy.call_args.args[1]), [0, 1])

    def test_chunk_batch_knob_reads_the_env(self):
        """``chunk_batch``: explicit > env ``EMRI_DIRECT_CHUNK_BATCH`` (0 / 1, false / true, any
        case) > False; anything else raises naming the knob."""
        with mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "EMRI_DIRECT_CHUNK_BATCH"},
                             clear=True):
            self.assertFalse(self._direct().chunk_batch)
        for val, want in (("1", True), ("true", True), ("TRUE", True), ("0", False), ("False", False)):
            with mock.patch.dict(os.environ, {"EMRI_DIRECT_CHUNK_BATCH": val}):
                self.assertIs(self._direct().chunk_batch, want, val)
        with mock.patch.dict(os.environ, {"EMRI_DIRECT_CHUNK_BATCH": "1"}):
            self.assertFalse(self._direct(chunk_batch=False).chunk_batch)     # explicit wins
        for bad in ("no", "2", "", "yes"):
            with mock.patch.dict(os.environ, {"EMRI_DIRECT_CHUNK_BATCH": bad}):
                with self.assertRaisesRegex(ValueError, "EMRI_DIRECT_CHUNK_BATCH"):
                    self._direct()

    def _sibling(self, **kw):
        """A second REAL generator like the fixture's (the same synthetic FEW stand-in, table and
        grid; the env is read at this call) with the constructor overrides ``kw``."""
        gen = self._gen_with_two_chunk_rows_and_one_lookup_row()
        args = dict(orbits=None, tdi_config=gen.tdi_config, t_start=gen.t_start, data_t0=gen.data_t0,
                    Nt_sub=32, fine_dt_plunge=240.0, response="spline", lookup="python")
        args.update(kw)
        return type(gen)(gen.few_gen, gen.table, gen.wdm, **args)

    def test_auto_window_reaches_the_generator(self):
        """EMRI_DIRECT_NT_SUB=auto[:floor] on the real batch, in both chunk-row modes: each chunk
        row's tail in one window (one length per _chunk_rows_batch call: the batch's, or each
        row's own by default); with a floor that holds every tail the template is the fixed-length
        one, number for number."""
        from lisatools.utils.utility import asnumpy

        for chunk_batch in (False, True):
            with self.subTest(chunk_batch=chunk_batch):
                with mock.patch.dict(os.environ, {"EMRI_DIRECT_NT_SUB": "auto:96"}):
                    auto = self._sibling(Nt_sub=None, chunk_batch=chunk_batch)
                self.assertEqual((auto.Nt_sub, auto.nt_sub_floor), ("auto", 96))
                got = asnumpy(auto.batch(self.rows, chunk_rows=4))
                st = dict(auto.last_stats)
                fixed = self._sibling(Nt_sub=96, chunk_batch=chunk_batch)
                want = asnumpy(fixed.batch(self.rows, chunk_rows=4))
                self.assertEqual((st["nt_sub_eff"], fixed.last_stats["nt_sub_eff"]), (96, 96))
                self.assertEqual((st["chunk_rows"], st["windows"]), (2, 2))       # one window per chunk row
                np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)
                # a floor below the tails: the length grows past it, still one window per row
                # the fixture's tails: row 0 hands off at pixel 88 and ends at 100 (span 13), row 1
                # at 98 and 112 (span 15): ceil4(2 (13 + 8) + 4) = 48, ceil4(2 (15 + 8) + 4) = 52
                lean = self._sibling(Nt_sub="auto:32", chunk_batch=chunk_batch)
                lean.batch(self.rows, chunk_rows=4)
                st = dict(lean.last_stats)
                self.assertEqual(st["nt_sub_eff"], 52)       # the batch's / the largest per-row length
                self.assertEqual((st["chunk_rows"], st["windows"]), (2, 2))
                lean(*self.rows[0])                          # the single-template path: its own length
                self.assertEqual(lean.last_stats["nt_sub_eff"], 48)
                lean(*self.rows[1])
                self.assertEqual(lean.last_stats["nt_sub_eff"], 52)

    def test_tail_skip_knob_reaches_the_generator(self):
        """EMRI_DIRECT_TAIL_SKIP_PX above the fixture's tails, in both chunk-row modes: both chunk
        rows keep the lookup through their last pixel (no pairs, no chunk pixels, no tail entry;
        counted in tail_skipped); each batch row is the single template built with the same knob;
        the chunk rows differ from the default, tail-keeping template, the lookup-only row does
        not."""
        from lisatools.utils.utility import asnumpy

        default = self._gen_with_two_chunk_rows_and_one_lookup_row()
        want = asnumpy(default.batch(self.rows, chunk_rows=4))
        self.assertEqual(default.last_stats["tail_skipped"], 0)
        for chunk_batch in (False, True):
            with self.subTest(chunk_batch=chunk_batch):
                with mock.patch.dict(os.environ, {"EMRI_DIRECT_TAIL_SKIP_PX": "200"}):
                    skip = self._sibling(chunk_batch=chunk_batch)
                self.assertEqual(skip.tail_skip_px, 200)
                got = asnumpy(skip.batch(self.rows, chunk_rows=4))
                st = dict(skip.last_stats)
                self.assertEqual((st["chunk_rows"], st["tail_skipped"], st["chunk_pixels"], st["windows"]),
                                 (2, 2, 0, 0))
                self.assertEqual(st["tail_layers"], [])
                m_lo, m_hi = skip._band()
                for r, p in enumerate(self.rows):
                    np.testing.assert_allclose(got[r], asnumpy(skip(*p).arr[:, m_lo:m_hi]), rtol=1e-10, atol=1e-13)
                skip(*self.rows[0])                          # a chunk row alone: skipped too
                self.assertEqual((skip.last_stats["tail_skipped"], skip.last_stats["chunk_pixels"]), (1, 0))
                for r in (0, 1):
                    self.assertGreater(np.abs(got[r] - want[r]).max(), 1e-6 * np.abs(want[r]).max())
                np.testing.assert_allclose(got[2], want[2], rtol=1e-12, atol=1e-15)

    def test_chunk_rows_batch_records_phase_walls(self):
        """With EMRI_DIRECT_WALLS on, in BOTH chunk-row modes: ``last_stats["walls"]`` of one batch
        call, exclusive seconds per phase (capture, response, lookup, tail_dense, transform,
        splice, ...) that add up to the call's total; ``walls_per_call`` one entry per
        _batch_chunk call (one per ``chunk_rows`` group, whatever the mode; the mode shows in the
        _chunk_rows_batch calls: one per chunk row, or one for both); the response grid lengths
        (padded common length, per-row lengths); a single call records its own."""
        phases = {"capture", "response", "lookup", "tail_dense", "transform", "splice"}
        for chunk_batch in (False, True):
            with self.subTest(chunk_batch=chunk_batch):
                gen = self._gen_in_mode(chunk_batch, walls=True)
                self.assertTrue(gen.walls)
                with mock.patch.object(gen, "_chunk_rows_batch", wraps=gen._chunk_rows_batch) as spy:
                    gen.batch(self.rows, chunk_rows=4)
                self.assertEqual(spy.call_count, 1 if chunk_batch else 2)
                st = dict(gen.last_stats)
                walls = st["walls"]
                self.assertGreaterEqual(set(walls), phases | {"total"})
                self.assertTrue(all(v >= 0 for v in walls.values()), walls)
                parts = sum(v for k, v in walls.items() if k != "total")
                self.assertLessEqual(abs(parts - walls["total"]), 0.1 * walls["total"], walls)
                self.assertEqual(len(st["walls_per_call"]), 1)                 # 3 rows, 4 per call
                self.assertLessEqual(sum(st["walls_per_call"]), walls["total"])
                self.assertEqual(len(st["n_response_points_rows"]), 2)        # the two chunk rows
                self.assertEqual(st["n_response_points"], max(st["n_response_points_rows"]))
                self.assertTrue(all(isinstance(v, float) for v in st["walls_detail"].values()))
                gen.batch(self.rows, chunk_rows=2)
                self.assertEqual(len(gen.last_stats["walls_per_call"]), 2)    # rows 0-1, then row 2
                parts = sum(v for k, v in gen.last_stats["walls"].items() if k != "total")
                self.assertLessEqual(abs(parts - gen.last_stats["walls"]["total"]),
                                     0.1 * gen.last_stats["walls"]["total"])
                gen(*self.rows[0])
                walls = gen.last_stats["walls"]
                self.assertGreaterEqual(set(walls), phases | {"total"})
                parts = sum(v for k, v in walls.items() if k != "total")
                self.assertLessEqual(abs(parts - walls["total"]), 0.1 * walls["total"], walls)

    def test_walls_are_off_by_default_and_time_nothing(self):
        """EMRI_DIRECT_WALLS unset (the library and global-fit default): no phase clock is ever
        built, nothing is timed or synchronised, and ``last_stats`` has no walls keys -- batch
        (both chunk-row modes) and single call."""
        from lisatools.sources.emri import wdm_direct as wd

        for chunk_batch in (False, True):
            with self.subTest(chunk_batch=chunk_batch):
                with mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "EMRI_DIRECT_WALLS"},
                                     clear=True):
                    gen = self._gen_in_mode(chunk_batch)
                self.assertFalse(gen.walls)
                with mock.patch.object(wd, "_PhaseWalls", side_effect=AssertionError("a clock was built")) as pw:
                    gen.batch(self.rows, chunk_rows=4)
                    st = dict(gen.last_stats)
                    gen(*self.rows[0])
                    one = dict(gen.last_stats)
                self.assertEqual(pw.call_count, 0)
                for k in ("walls", "walls_detail", "walls_per_call"):
                    self.assertNotIn(k, st)
                    self.assertNotIn(k, one)
                self.assertEqual(st["chunk_rows"], 2)                         # the batch itself ran

    def test_alone_row_with_a_skipped_tail_has_no_entry_and_is_counted(self):
        """An ``alone`` row (feed="fine" / fixed n_fine: built through __call__) whose tail the skip
        replaced: counted in tail_skipped with no tail_layers entry, as a batched chunk row; one
        that kept its tail gets its entry (the single call's counters are stood in)."""
        from lisatools.domains import WDMSettings
        from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

        class _Alone(EMRIDirectWDM):
            def _mode_list(self, few_args, few_kwargs):
                self._last_holder = SimpleNamespace(t_arr=np.array([0.0, 1.0e12]))
                return [], 0.0                                   # hands off: an alone row on feed="fine"

            def _call(self, *few_args, **few_kwargs):
                skipped = int(few_args[0]) == 1
                self.last_stats = dict(lookup_pixels=10, chunk_pixels=0 if skipped else 5, dropped_pixels=0,
                                       windows=0 if skipped else 1, tail_layers=0 if skipped else 7,
                                       tail_skipped=int(skipped), nt_sub_eff=128)
                return SimpleNamespace(arr=np.zeros((3, self.wdm.Nf, self.wdm.Nt)))

        table = SimpleNamespace(fdot_vals=np.array([-1.0, 1.0]), INTERP_METHOD="spline")
        d = _Alone(None, table, WDMSettings(32, 64, 10.0, force_backend="cpu"), orbits=None,
                   tdi_config=SimpleNamespace(nchannels=3), t_start=0.0, data_t0=0.0, fine_dt=100.0, feed="fine")
        d.batch([np.full(14, 1.0), np.full(14, 0.0)], chunk_rows=4)
        st = d.last_stats
        self.assertEqual((st["alone"], st["tail_skipped"], st["tail_layers"], st["windows"]), (2, 1, [7], 1))
        self.assertEqual((st["lookup_pixels"], st["chunk_pixels"]), (20, 5))

    def test_refused_chunk_row_stays_zero(self):
        from lisatools.utils.utility import asnumpy

        gen = self._gen_with_two_chunk_rows_and_one_lookup_row()
        rows = list(self.rows); rows[1] = self.row_few_refuses      # a row inside the chunk group
        arr = gen.batch(rows, chunk_rows=4, skip_domain_errors=True)
        self.assertEqual(gen.last_failed_rows, [1])
        self.assertTrue(np.all(asnumpy(arr[1]) == 0))
        np.testing.assert_allclose(asnumpy(arr[0]), asnumpy(gen(*rows[0]).arr[:, gen._band()[0]:gen._band()[1]]), rtol=1e-10, atol=1e-13)

    def test_refused_row_is_zero_and_listed_across_chunks(self):
        d = self._direct()
        rows = [np.full(14, v) for v in (0.0, 1.0, 3.0, 1.0, 4.0)]
        out = d.batch(rows, chunk_rows=2, skip_domain_errors=True)
        self.assertEqual(d.last_failed_rows, [1, 3])
        for r, v in enumerate((0.0, 1.0, 3.0, 1.0, 4.0)):
            want = 0.0 if v == 1.0 else v + 10.0
            self.assertTrue(np.all(out[r] == want), r)
        self.assertEqual(d.last_stats["failed"], 2)

    def test_chunk_rows_reuse_the_first_few_call(self):
        """A plunge-chunk row is captured from the batch's own _mode_list: one FEW call per row,
        not two (the 720-day speed test paid ~80 ms per row for the second one), and the chunk
        rows are batched, never built through the single-template path."""
        d = self._direct()
        type(d).mode_calls, type(d).captured = 0, []
        d.batch([np.full(14, v) for v in (0.0, 3.0, 4.0)], chunk_rows=3)
        self.assertEqual(type(d).mode_calls, 3)
        self.assertEqual(type(d).captured, [([], 0.0)] * 3)
        self.assertEqual((d.last_stats["chunk_rows"], d.last_stats["alone"]), (3, 0))

    def _gen_with_one_chunk_row(self):
        """Tag-0 rows hand off to the plunge chunk and are batched through the REAL
        accumulate_harmonic_batch (each one harmonic tripping the curvature trigger at pixel 80
        of a 128-pixel grid, a fused-kernel stand-in counting the lookup pixels before it; the
        response stubbed out of _chunk_rows_batch); tag-5 rows stay on the lookup and are batched
        (response and kernel stubbed: 1000 lookup and 3 dropped pixels per batch); tag-7 rows end
        before the window (an exact zero, never a chunk row)."""
        from lisatools.domains import WDMSettings
        from lisatools.sources.emri import wdm_direct as wd

        class _Plunge(wd.EMRIDirectWDM):
            def _mode_list(self, few_args, few_kwargs):
                t_end = -1000.0 if int(few_args[0]) == 7 else 1.0e12      # tag 7: before the window
                self._last_holder = SimpleNamespace(t_arr=np.array([t_end - 1.0e4, t_end]))
                return [], (0.0 if int(few_args[0]) in (0, 7) else None)

            def _chunk_item(self, few_args, modes, chunk_start):
                return dict(modes=modes, chunk_start=chunk_start)

            def _chunk_rows_batch(self, out_arr, chunk_items, m_lo, m_hi, few_kwargs):
                # the real _chunk_rows_batch without FEW and the response: every chunk row one
                # harmonic, the batched tail in the pairs form, its counters as there
                wdm = self.wdm
                rows = sorted(chunk_items)
                n_ok = np.arange(self.pixel_edge, wdm.Nt - self.pixel_edge)
                tn = n_ok * wdm.layer_dt
                f = np.full_like(tn, 8.2 * wdm.layer_df)
                track = wd.HarmonicTrack((2, 2, 0, 0), tn, np.ones_like(tn), 2 * np.pi * f * tn, f,
                                         np.zeros_like(tn), np.where(n_ok >= 80, 1.0, 0.0))
                N = wdm.Nf * self.Nt_sub

                def tail_td(pairs):
                    ts = np.stack([self.data_t0 + (n0 * wdm.Nf + np.arange(N)) * wdm.data_dt
                                   for _, n0, _, _ in pairs])
                    return np.repeat(np.cos(2 * np.pi * f[0] * ts)[:, None, :], 3, axis=1)
                tail_td.summed, tail_td.by_pairs = True, True

                st = wd.accumulate_harmonic_batch(
                    out_arr, self.table, [track] * len(rows), None, n_ok, tail_td, Nf=wdm.Nf, Nt=wdm.Nt,
                    dt=wdm.data_dt, layer_dt=wdm.layer_dt, layer_df=wdm.layer_df, t0=self.data_t0,
                    Nt_sub=self.Nt_sub, fdot_axis_max=self.fdot_axis_max, pixel_edge=self.pixel_edge,
                    sub_row=np.asarray(rows), num_sub=len(rows), m_lo=m_lo,
                    lookup_kernel=lambda n_stop: dict(lookup_pixels=int(np.sum(3 * (n_stop - n_ok[0])))))
                return dict(rows=len(rows), windows=st["windows"],
                            tail_layers=[st["tail_layers"][r] for r in rows],
                            lookup_pixels=st["lookup_pixels"], chunk_pixels=st["chunk_pixels"],
                            dropped_pixels=st["dropped_pixels"], lookup="kernel")

            def _knots_feed(self, H, p, t_fine, fly):
                return None, np.zeros((2, 1)), None, None          # two harmonics

            def _kernel_backend(self, out):
                return "kernel"

            def _lookup_fn(self, be, out, acc, sub_row, n_ok, m_lo):
                return lambda n_stop: dict(lookup_pixels=1000, dropped_pixels=3)

        table = SimpleNamespace(fdot_vals=np.array([-1.0, 1.0]), INTERP_METHOD="spline")
        wdm = WDMSettings(32, 128, 10.0, force_backend="cpu")
        return _Plunge(None, table, wdm, orbits=None, tdi_config=SimpleNamespace(nchannels=3),
                       t_start=0.0, data_t0=0.0, fine_dt=100.0, Nt_sub=32)

    def test_batch_stats_carry_the_chunk_rows_counters(self):
        gen = self._gen_with_one_chunk_row()
        gen.batch([np.full(14, 0.0)], chunk_rows=4)
        st = gen.last_stats
        self.assertEqual((st["chunk_rows"], st["alone"]), (1, 0))
        self.assertGreater(st["chunk_pixels"], 0)      # the chunk row's chunk pixels are counted
        self.assertEqual(len(st["tail_layers"]), 1)
        self.assertGreaterEqual(st["tail_layers"][0], gen.Nt_sub // 2)  # at least one kept interior
        self.assertGreaterEqual(st["windows"], 1)

    def test_chunk_rows_counters_add_to_the_lookup_rows(self):
        """On the kernel path the lookup rows' counters are set by stats.update(**tot): the
        chunk rows' are ADDED to them, not overwritten."""
        gen = self._gen_with_one_chunk_row()
        gen.batch([np.full(14, 0.0)])
        chunk = dict(gen.last_stats)
        with mock.patch("lisatools.sources.emri.emritdionfly.EMRITDIonFly.sky",
                        return_value=(0, 0, 0.1, 0.2, 0.3)), \
                mock.patch("lisatools.sources.emri.emritdionfly.EMRITDIonFly.run_response",
                           return_value=SimpleNamespace(x=np.array([[-1.0, 1.0e12]]))):
            gen.batch([np.full(14, 5.0), np.full(14, 0.0)], chunk_rows=4)
        st = gen.last_stats
        self.assertEqual((st["chunk_rows"], st["subs"], st["lookup"]), (1, 2, "kernel"))
        self.assertGreater(chunk["lookup_pixels"], 0)
        self.assertEqual(st["lookup_pixels"], 1000 + chunk["lookup_pixels"])
        self.assertEqual(st["dropped_pixels"], 3 + chunk["dropped_pixels"])
        self.assertEqual(st["chunk_pixels"], chunk["chunk_pixels"])
        self.assertEqual((st["windows"], st["tail_layers"]),
                         (chunk["windows"], chunk["tail_layers"]))

    def test_rows_ending_before_the_window_are_not_chunk_rows(self):
        """A row whose inspiral ends before the window hands off (its chunk start is set) but is
        an exact zero that never touches the plunge chunk: counted in ended_before_window, not as
        a chunk row, no tail entry, nothing added to the pixel counters."""
        gen = self._gen_with_one_chunk_row()
        gen.batch([np.full(14, 0.0)])
        chunk = dict(gen.last_stats)
        with mock.patch("lisatools.sources.emri.emritdionfly.EMRITDIonFly.sky",
                        return_value=(0, 0, 0.1, 0.2, 0.3)), \
                mock.patch("lisatools.sources.emri.emritdionfly.EMRITDIonFly.run_response",
                           return_value=SimpleNamespace(x=np.array([[-1.0, 1.0e12]]))):
            gen.batch([np.full(14, 5.0)])
            look = dict(gen.last_stats)
            gen.batch([np.full(14, 5.0), np.full(14, 0.0), np.full(14, 7.0)], chunk_rows=4)
        st = gen.last_stats
        self.assertEqual((st["chunk_rows"], st["alone"]), (1, 0))
        self.assertEqual(len(st["tail_layers"]), 1)
        self.assertEqual(st["ended_before_window"], 1)
        for k in ("lookup_pixels", "chunk_pixels", "dropped_pixels"):
            self.assertEqual(st[k], look[k] + chunk[k], k)
        self.assertEqual((st["windows"], st["tail_layers"]),
                         (chunk["windows"], chunk["tail_layers"]))

    def test_tail_layers_concatenate_across_chunks(self):
        """batch() sums the chunks' counters; the per-row tail list is concatenated."""
        gen = self._gen_with_one_chunk_row()
        gen.batch([np.full(14, 0.0)] * 2, chunk_rows=1)
        st = gen.last_stats
        self.assertEqual(st["chunk_rows"], 2)
        self.assertEqual(len(st["tail_layers"]), 2)
        self.assertEqual(st["tail_layers"][0], st["tail_layers"][1])

    def test_without_skip_the_refusal_is_typed(self):
        from lisatools.utils.exceptions import WaveformDomainError

        d = self._direct()
        with self.assertRaises(WaveformDomainError):
            d.batch([np.full(14, 0.0), np.full(14, 1.0)])

    def test_inspiral_ending_before_the_window_is_an_exact_zero(self):
        """A hot-rung row that plunges before the data start: zero template, no
        response call (its response grid would be empty); one ending INSIDE the
        window still reaches the response (paired control)."""
        from lisatools.domains import WDMSettings
        from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

        class _Reached(Exception):
            pass

        class _Ended(EMRIDirectWDM):
            def _mode_list(self, few_args, few_kwargs):
                t_end = float(few_args[0])
                # knots every ~1 h: the in-window row's own response grid has enough points
                self._last_holder = SimpleNamespace(t_arr=np.linspace(0.0, t_end, 16))
                self._last_tracks, self._last_track_n = [], np.zeros(0, int)
                return [(2, 2, 0, 0)], t_end - 1.0      # stops in/before the window: a chunk row

            def _dense_response(self, items, t_fine):
                raise _Reached()

        table = SimpleNamespace(fdot_vals=np.array([-1.0, 1.0]), INTERP_METHOD="spline")
        wdm = WDMSettings(32, 64, 10.0, force_backend="cpu")
        few = SimpleNamespace(inspiral_generator=SimpleNamespace(inspiral_generator=None))
        d = _Ended(few, table, wdm, orbits=None, tdi_config=SimpleNamespace(nchannels=3),
                   t_start=0.0, data_t0=5.0e4, fine_dt=100.0, response="dense")
        out = d.batch([np.r_[5.0e4 - 700.0, np.zeros(13)]])          # ends 700 s before the start
        self.assertTrue(np.all(out == 0.0))
        self.assertEqual(d.last_failed_rows, [])
        st = d.last_stats                                   # not a plunge-chunk row
        self.assertEqual((st["alone"], st["ended_before_window"], st["tail_layers"]), (0, 1, []))
        with mock.patch("lisatools.sources.emri.emritdionfly.EMRITDIonFly.sky",
                        return_value=(0, 0, 0.1, 0.2, 0.3)), \
                mock.patch("lisatools.sources.emri.wdm_direct.dense_inputs_from_holder",
                           return_value=None):
            with self.assertRaises(_Reached):
                d.batch([np.r_[5.0e4 + 3600.0, np.zeros(13)]])       # ends inside the window

    def test_a_non_domain_error_always_propagates(self):
        d = self._direct()
        with self.assertRaisesRegex(RuntimeError, "not a domain error"):
            d.batch([np.full(14, 2.0)], skip_domain_errors=True)


# ---------------------------------------------------------------------------
# MPI-safe pool start
# ---------------------------------------------------------------------------

def _report_main_and_env():
    """Runs in a spawned worker: did the parent's main script get imported?"""
    import sys as _sys

    return ("__mp_main__" in _sys.modules and hasattr(_sys.modules["__mp_main__"], "MARKER"),
            os.getpid())


class SpawnWithoutMainTest(unittest.TestCase):
    def test_main_file_is_hidden_during_the_spawn_and_restored(self):
        import sys

        from lisatools.sources.emri import direct_signal_gen as mod

        main = sys.modules["__main__"]
        had = "__file__" in vars(main)
        saved = getattr(main, "__file__", None)
        main.__file__ = "/definitely/not/a/script.py"
        try:
            with mod._main_hidden_from_spawn():
                self.assertNotIn("__file__", vars(main))
            self.assertEqual(main.__file__, "/definitely/not/a/script.py")
        finally:
            if had:
                main.__file__ = saved
            else:
                del main.__file__

    def test_a_short_warm_up_is_refused(self):
        """A worker missing after the warm-up would be spawned later, outside the
        hidden-main window: refuse the pool (the adapter then runs serially)."""
        from lisatools.sources.emri import direct_signal_gen as mod

        killed = []

        class _Fut:
            def result(self, timeout=None):
                return None

        class _Ex:
            def __init__(self, max_workers, mp_context):
                self._processes = {1: SimpleNamespace(kill=lambda: killed.append(1))}

            def submit(self, fn, *a):
                return _Fut()

            def shutdown(self, wait=True, cancel_futures=False):
                pass

        with mock.patch.object(mod, "ProcessPoolExecutor", _Ex):
            with self.assertRaisesRegex(RuntimeError, "started 1 of 2"):
                mod.spawn_executor_without_main(2, len, ([],))
        self.assertEqual(killed, [1])

    def test_every_worker_starts_without_importing_the_main_script(self):
        """A parent main script that would poison every child (it raises on import)
        is never imported by the eagerly started workers."""
        import sys

        from lisatools.sources.emri.direct_signal_gen import _kill_executor, spawn_executor_without_main

        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "poison_main.py")
            with open(script, "w") as f:
                f.write("raise SystemExit('the parent main script was imported in a worker')\n")
            main = sys.modules["__main__"]
            had = "__file__" in vars(main)
            saved = getattr(main, "__file__", None)
            spec = getattr(main, "__spec__", None)
            main.__file__ = script
            main.__spec__ = None
            try:
                ex = spawn_executor_without_main(2, _report_main_and_env, timeout=120)
                try:
                    pids = {p.pid for p in ex._processes.values()}
                    self.assertEqual(len(pids), 2)
                    out = [ex.submit(_report_main_and_env).result(timeout=60) for _ in range(4)]
                    self.assertTrue(all(not imported for imported, _ in out))
                    # no worker was spawned after the warm-up
                    self.assertEqual({p.pid for p in ex._processes.values()}, pids)
                finally:
                    _kill_executor(ex)
            finally:
                main.__spec__ = spec
                if had:
                    main.__file__ = saved
                else:
                    del main.__file__


if __name__ == "__main__":
    unittest.main()
