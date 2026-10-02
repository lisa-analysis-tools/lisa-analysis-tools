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
                          s.traj_workers), ("full", 8, None, "dense", 0))

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
        cfg = self._resolve(domain=_fd_spec(), chan="AET", direct_table=None,
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


class BatchDomainRefusalTest(unittest.TestCase):
    """``EMRIDirectWDM.batch(skip_domain_errors=)``: a FEW domain refusal of one row."""

    def _direct(self):
        from lisatools.domains import WDMSettings
        from lisatools.sources.emri.wdm_direct import EMRIDirectWDM

        class _Stub(EMRIDirectWDM):
            # column 0 tags the row: 1 -> FEW out-of-domain, 2 -> a real bug,
            # anything else -> built "alone" (plunge-chunk path) as a constant
            def _mode_list(self, few_args, few_kwargs):
                tag = int(few_args[0])
                if tag == 1:
                    raise ValueError("p0 is outside of our domain of validity.")
                if tag == 2:
                    raise RuntimeError("not a domain error")
                return [], 0.0

            def __call__(self, *p, **kw):
                return SimpleNamespace(arr=np.full((3, self.wdm.Nf, self.wdm.Nt), p[0] + 10.0))

        table = SimpleNamespace(fdot_vals=np.array([-1.0, 1.0]), INTERP_METHOD="spline")
        wdm = WDMSettings(32, 64, 10.0, force_backend="cpu")
        return _Stub(None, table, wdm, orbits=None, tdi_config=SimpleNamespace(nchannels=3),
                     t_start=0.0, data_t0=0.0, fine_dt=100.0)

    def test_refused_row_is_zero_and_listed_across_chunks(self):
        d = self._direct()
        rows = [np.full(14, v) for v in (0.0, 1.0, 3.0, 1.0, 4.0)]
        out = d.batch(rows, chunk_rows=2, skip_domain_errors=True)
        self.assertEqual(d.last_failed_rows, [1, 3])
        for r, v in enumerate((0.0, 1.0, 3.0, 1.0, 4.0)):
            want = 0.0 if v == 1.0 else v + 10.0
            self.assertTrue(np.all(out[r] == want), r)
        self.assertEqual(d.last_stats["failed"], 2)

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
                self._last_holder = SimpleNamespace(t_arr=np.array([0.0, t_end]))
                self._last_tracks, self._last_track_n = [], np.zeros(0, int)
                return [(2, 2, 0, 0)], t_end - 1.0      # stops in/before the window: built alone

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
