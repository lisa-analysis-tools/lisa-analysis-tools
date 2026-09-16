"""GB fan-out end to end on the debug-preset gb_no_fg synthetic fit (fake communicator).

Gated by ``RUN_GF_GB_SMOKE=1`` (heavier than the noise/blank smokes). Two scenarios:
two compute ranks driving the three-command orchestrator, and the seeded PARITY of the
orchestrator at ONE compute rank (``GB_PROPOSE_ORCHESTRATE=1``, a direct-call
``WalkerFanout``) against ``_propose_legacy`` -- today's body.

THE TWO SCENARIOS USE DIFFERENT MODELS, ON PURPOSE
--------------------------------------------------
The two-rank scenario INJECTS an in-band GB source
(``general.gb_injection_params``), so the RJ search really births leaves and
the whole alive-source path -- birth, in-model repeats, write-back, the
head's block merge -- runs under real code on BOTH walker blocks. The parity
scenario does NOT inject: its fixture has to be REPRODUCIBLE, and a model
with alive leaves is not (measured below).

WHAT THE PARITY GATE CAN AND CANNOT COMPARE
-------------------------------------------
``log_like``, ``inds``, ``band_temps``, ``band_num_binaries`` and the alive
coords (``coords[inds]``) are compared bit for bit on the EMPTY-model fit.

Reproducibility of this fixture, measured with PAIRED NEGATIVE CONTROLS
(legacy vs legacy, same seed, same process -- an invariance claim without one
proves nothing):

* 2026-09-16, before the build seeds: ``coords`` diverged in the control
  exactly as it did legacy-vs-orchestrator (4000/7200 elements on the
  starting state, 5600/7200 after two iterations) while the four arrays above
  stayed bit-identical in BOTH pairings. Cause: the GB prior objects seeded
  themselves from OS entropy -- ``F0McGMMSampling`` (built via
  ``from_heatmap()`` in ``stock/erebor/gb.py``) and the 3-D galaxy
  sky/distance prior (``priors/galaxy_prior_3d.py``,
  ``priors/galaxy_sky_dist.py``), which between them fill 5 of the 9 GB
  columns. They now take PER-RANK sub-seeds derived from ``random_seed``
  (``communication/ranks.py::rank_build_seed`` -> ``GBSettings.build_seed``
  -> ``GBSetup._build_sub_seeds``).
* 2026-09-16, with the injection ON and the build seeds in: the control still
  diverges in ``log_like`` (all 4 walkers, O(100) nats), so the injected
  fixture STILL cannot support a bit-identity gate. Bisected (one legacy run
  vs a second legacy run in the same process, injected):
  ``start_log_like`` 0/4 differ, ``start_inds`` 0/800 differ (the model
  starts EMPTY), ``start_coords`` **3200/7200 = 800 leaves x exactly 4 of
  the 9 columns** differ -- down from 5 columns before the build seeds, i.e.
  the seeded (f0, Mc) + (dist, alpha, sin_delta) columns now reproduce and
  the remaining 4 are phi0 / cos_iota / psi / fdot_astro_ratio, the ones
  eryn's ``uniform_dist`` draws from the MODULE-level ``np.random`` stream.
  Its position at prior-draw time is not the same for the first and the
  second fit built in one process (both runs are otherwise exactly
  reproducible ACROSS processes -- run 1 and run 2 each give the same numbers
  every time -- so no OS entropy is left anywhere in the path). The dead-leaf
  fill itself is harmless, but the same stream feeds the birth containers'
  extrinsic columns during sampling, which is how it reaches ``log_like``.
  Closing this needs the global stream pinned at a defined point of the
  build/sample boundary (or the fixture rebuilt one-fit-per-process); it is
  NOT an entropy-seeded proposal object, so the birth-container seeding of
  this round does not address it.
* The RJ birth container itself IS now seeded (per rank, per F-stat epoch:
  ``build_gb_birth_distribution(seed=...)`` <- ``_birth_seed(k)`` <-
  ``fstat_fit_kwargs["build_seed"]``), unit-tested in
  ``tests/test_fstat_birth_seed.py``.

``GB_SMOKE_PARITY_CONTROL=1`` re-runs the control here and automatically
promotes the check to the FULL ``coords`` array (dead-leaf fill included)
once the fixture reproduces that too -- the gate is never weakened silently,
the control decides.

Memory: an 8 GB laptop is the budget. ``tearDown`` asserts the process stayed
under 5 GB; the whole module measured 4.0-4.7 GB peak RSS (two scenarios,
~2 min wall) over five runs, so run it on its own and not alongside another
python process. The fit is the ``gb_no_fg`` debug preset (3-day Tobs, tiny
chunked-het grids) with ``GB_DEBUG=0`` so the preset's *sizes* apply without
arming the debug instrumentation (round-trip verification + band plots), which
is neither wanted nor cheap here.
"""

import gc
import os
import resource
import shutil
import tempfile
import unittest

import numpy as np

RUN = os.environ.get("RUN_GF_GB_SMOKE", "") not in ("", "0")

#: Opt-in THIRD run of the parity scenario: the paired negative control
#: (legacy vs legacy). Off by default purely for MEMORY -- building this fit
#: costs ~2.1 GB and the allocator does not give it all back, so measured
#: 2026-09-16 the control takes the scenario to 5.13 GB peak RSS, past the
#: 5 GB laptop budget the module otherwise holds (without it the whole module
#: measured 4.0-4.7 GB over five runs; the parity scenario ALONE with the
#: control measured 3.0 GB / 128 s, so running just that test is the cheap
#: way to take this measurement). Turn it on
#: (``GB_SMOKE_PARITY_CONTROL=1``) to re-measure that the fixture still
#: reproduces itself; the last such measurement is in the module docstring.
CONTROL = os.environ.get("GB_SMOKE_PARITY_CONTROL", "") not in ("", "0")

#: peak-RSS ceiling asserted in ``tearDown``. The control run is an opt-in
#: FOURTH/FIFTH fit in the process, so it gets its own (measured) headroom --
#: the default gate stays at the 8 GB laptop's 5 GB.
RSS_BUDGET_GB = 6.0 if CONTROL else 5.0

#: arrays the orchestrator must reproduce bit for bit at one compute rank
#: (``coords`` is handled separately -- see the module docstring)
PARITY_KEYS = ("log_like", "inds", "band_temps", "band_num_binaries")


def _rss_gb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024.0 ** 3) if ru > 1e7 else ru / (1024.0 ** 2)  # macOS bytes / linux KB


def _reclaim():
    """Drop one finished fit's resident footprint before building the next.

    A production process builds ONE fit, so nothing evicts
    ``GBSpecialBase._shared_buffer_caches`` -- a CLASS-level staged-buffer
    cache keyed by ``id(computation object)``. This file builds FOUR fits in
    one process (the 2-rank world plus two single-rank worlds; five with
    ``GB_SMOKE_PARITY_CONTROL=1``), and without this the dead fits' buffers
    stay resident and the run is SIGKILLed on an 8 GB laptop -- measured.
    """
    try:
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase

        GBSpecialBase._shared_buffer_caches.clear()
        GBSpecialBase._branch_propose_counts.clear()
    except Exception:  # pragma: no cover - never let cleanup fail a test
        pass
    gc.collect()


@unittest.skipUnless(RUN, "set RUN_GF_GB_SMOKE=1 to run the multi-rank GB smoke")
class MultiRankGBSmokeTest(unittest.TestCase):
    #: two compute ranks -> a 2-walker block each
    NWALKERS = 4
    ITERATIONS = 2
    #: FakeWorld watchdog. Generous, but far below the "walk away" hour: a
    #: deadlocked fan-out must fail the run, not hold the laptop all night.
    TIMEOUT_S = 1800.0
    #: ``general.random_seed`` (eryn reseeds ``np.random`` from it in every
    #: ``Move.__init__``) AND the seed this fixture sets before each world --
    #: the run draws its STARTING state (prior draws) in ``load_info``, which
    #: happens BEFORE any move is constructed.
    SEED = 4242

    #: ONE in-band GB injection, GBGPU basis
    #: ``[A, f0, fdot, fddot, phi0, iota, psi, lam, beta]``
    #: (``injections.GB_INJECTION_PARAMS``' column order). ``f0 = 7.5 mHz``
    #: sits inside gb_no_fg's GB band [7.36, 7.78] mHz and ``A = 1e-21`` is
    #: loud enough at the 3-day debug Tobs that the RJ search births leaves
    #: within the two iterations. Used by the TWO-RANK scenario only -- the
    #: alive-source path was otherwise untested by real code (whole-plan
    #: review, I2); the parity scenario needs a reproducible fixture and an
    #: alive model is not one (module docstring).
    INJECTION = [[1e-21, 7.5e-3, 1e-16, 0.0, 1.2, 0.9, 1.0, 4.0, -0.6]]

    #: seeded through ``os.environ.setdefault`` (an explicit value in the
    #: caller's environment still wins, so the knobs stay tunable from a shell)
    SMOKE_ENV = {
        "USE_GPU": "0",
        "MAKE_DIAGNOSTIC_PLOTS": "0",
        # FakeWorld ranks are threads and the macOS GUI backend cannot draw off
        # the main thread. Best effort only: if matplotlib is already imported
        # by the time setUp runs, the setup heatmap still warns and skips --
        # harmless, and the same warning appears in the noise smoke.
        "MPLBACKEND": "Agg",
        # MEMORY KNOB (the reason this test fits an 8 GB laptop). gb_no_fg's
        # DATA band is [6, 25] mHz while its GB band is [7.36, 7.78] mHz, so
        # without the clip every per-walker ACA slab carries ~45x the
        # frequency range the GB moves ever touch. 8 layers is comfortably
        # above the 5 the chunked-het gating needs. It is NOT the dominant
        # term (one ``fit.build()`` costs ~2.1 GB on its own), but it is the
        # one knob this test owns: 4.64 GB unclipped vs 4.0-4.7 GB clipped
        # across repeated runs of the whole module (budget 5.0).
        "DATA_BAND_LAYERS": "8",
        # the debug *preset* is armed by the ``debug=True`` kwarg below; this
        # keeps the debug INSTRUMENTATION (residual round-trip checks, band
        # plots under GB_DEBUG_DIR) off -- it is expensive and writes to the cwd
        "GB_DEBUG": "0",
    }

    def setUp(self):
        self._env0 = dict(os.environ)
        for key, value in self.SMOKE_ENV.items():
            os.environ.setdefault(key, value)
        self.tmpdir = tempfile.mkdtemp(prefix="gf_multirank_gb_")
        # the per-branch setup logs ignore file_store_dir and land in the cwd;
        # remember whether those directories were already there so tearDown only
        # removes ones THIS test created
        self._strays = []
        for name in ("gf_output_gb_no_fg", "gf_output"):
            path = os.path.join(os.getcwd(), name)
            self._strays.append((path, os.path.isdir(path)))

    def tearDown(self):
        _reclaim()
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        for path, existed in self._strays:
            if not existed:
                shutil.rmtree(path, ignore_errors=True)
        # the stock variants seed env knobs with setdefault at construction
        # (the GB_DEBUG preset); do not leak them into the next test
        os.environ.clear()
        os.environ.update(self._env0)
        self.assertLess(_rss_gb(), RSS_BUDGET_GB,
                        f"GB smoke exceeded the {RSS_BUDGET_GB} GB budget")

    def _fit(self, subdir, inject):
        from lisatools.globalfit.stock import erebor

        fit = erebor.gb_no_fg(
            debug=True,  # 3-day Tobs, tiny chunked-het grids (sizes only)
            nwalkers=self.NWALKERS, ntemps=2, data_mode="synthetic",
            file_store_dir=os.path.join(self.tmpdir, subdir), make_diagnostic_plots=False,
        )
        fit.general.num_iterations = self.ITERATIONS
        fit.general.random_seed = self.SEED
        if inject:
            fit.general.gb_injection_params = self.INJECTION
        return fit

    def _run_world(self, size, env=None, subdir=None, inject=False):
        """Run ``size`` ranks over the fake world; return ``{rank: probe dict}``."""
        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.ranks import prepare_rank
        from lisatools.globalfit.run import GlobalFit

        store = subdir if subdir is not None else f"n{size}"

        def fn(rank, comm):
            fit = self._fit(store, inject)
            layout = prepare_rank(fit, comm)
            fit.build()
            gf = GlobalFit(fit, comm)
            gf.run_global_fit()
            out = {"role": layout.role_of(rank).value, "acs_rows": int(gf.acs.acs_total_entries)}
            if hasattr(gf, "compute_service"):
                out["served"] = gf.compute_service_served
                return out
            # ``run_global_fit`` throws away ``run_mcmc``'s return value, so
            # ``gf.state`` is still the STARTING state -- probing it would make
            # every assertion below vacuous. eryn keeps the last state for its
            # ``initial_state=None`` resume path; that in-memory object is
            # exactly what the moves produced (``get_last_sample()`` would
            # round-trip it through the HDF store instead).
            final = getattr(gf.sampler, "_previous_state", None)
            if final is None:  # pragma: no cover - eryn API fallback
                final = gf.sampler.get_last_sample()
            sub = final.sub_states["gb"]
            out["log_like"] = np.array(final.log_like[0], copy=True)
            out["coords"] = np.array(final.branches["gb"].coords, copy=True)
            out["inds"] = np.array(final.branches["gb"].inds, copy=True)
            out["band_temps"] = np.array(sub.band_info["band_temps"], copy=True)
            out["band_num_binaries"] = np.array(sub.band_info["band_num_binaries"], copy=True)
            return out

        # set OUTSIDE the rank threads: FakeWorld ranks are threads of one
        # process and share os.environ, so a per-rank write would race
        saved = {k: os.environ.get(k) for k in (env or {})}
        os.environ.update(env or {})
        try:
            # one shared global stream per world (FakeWorld ranks are threads)
            np.random.seed(self.SEED)
            return FakeWorld(size, timeout=self.TIMEOUT_S).run(fn)
        finally:
            _reclaim()
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_two_compute_ranks_run_the_gb_moves(self):
        out = self._run_world(2, inject=True)
        self.assertEqual((out[0]["role"], out[1]["role"]), ("head", "compute"))
        self.assertEqual((out[0]["acs_rows"], out[1]["acs_rows"]), (2, 2))
        # FLOOR, not the exact count: ping + the three commands of one GB
        # propose. A GB move can still take the whole-ensemble early return
        # (``rj_replace`` on a zero-leaf model) before issuing any command, so
        # the per-iteration command count is configuration-dependent.
        self.assertGreaterEqual(out[1]["served"], 1 + 3)
        self.assertTrue(np.all(np.isfinite(out[0]["log_like"])))
        self.assertEqual(out[0]["log_like"].shape, (self.NWALKERS,))
        # (num_bands, ntemps) ladder: betas non-increasing with temperature
        self.assertTrue(np.all(np.diff(out[0]["band_temps"], axis=-1) <= 0))
        # the injected source is found: the ALIVE-source path ran for real
        # (an empty model exercises only the neutral/early-return branches)
        inds = out[0]["inds"]
        self.assertGreater(int(inds.sum()), 0, "no GB leaf survived the run")
        # and BOTH walker blocks carry leaves on the cold chain -- the merge
        # writes each rank's block back into the head's full state, so a
        # block-shaped merge bug shows here and nowhere else
        half = self.NWALKERS // 2
        self.assertTrue(inds[0, :half].any(), "head block has no cold leaf")
        self.assertTrue(inds[0, half:].any(), "worker block has no cold leaf")

    def test_orchestrator_at_one_rank_matches_the_legacy_body(self):
        # NO injection here. The RJ birth container IS seeded now, but the
        # injected fixture still does not reproduce ITSELF: the measured
        # legacy-vs-LEGACY control diverges in ``log_like``, bisected to the
        # 4 columns eryn's uniform priors draw off the module-level
        # ``np.random`` stream (module docstring). Turning the injection on
        # here would make this gate red for a fixture reason rather than a
        # port one. Alive-source coverage lives in the two-rank scenario.
        legacy = self._run_world(1, env={"GB_PROPOSE_ORCHESTRATE": "0"}, subdir="legacy")[0]
        if CONTROL:
            # paired negative control FIRST: a second legacy run, same seed,
            # same process. Without it an invariance claim proves nothing --
            # it is what separates "the orchestrator diverges" from "this fit
            # does not reproduce", so it is asserted BEFORE the orchestrator
            # comparison rather than after it.
            ctl = self._run_world(1, env={"GB_PROPOSE_ORCHESTRATE": "0"}, subdir="control")[0]
            for key in PARITY_KEYS:
                np.testing.assert_array_equal(
                    ctl[key], legacy[key],
                    err_msg=f"CONTROL legacy-vs-legacy differs in {key}: the fixture is "
                            "non-reproducible here, so the orchestrator comparison "
                            "would prove nothing")
        orch = self._run_world(1, env={"GB_PROPOSE_ORCHESTRATE": "1"}, subdir="orch")[0]
        for key in PARITY_KEYS:
            np.testing.assert_array_equal(orch[key], legacy[key], err_msg=key)
        alive = legacy["inds"]
        np.testing.assert_array_equal(
            orch["coords"][alive], legacy["coords"][alive], err_msg="alive coords")
        # the control decides whether the gate tightens to the WHOLE coords
        # array (dead-leaf fill included): never weakened silently, never
        # promoted on a fixture that cannot reproduce it.
        if CONTROL and np.array_equal(ctl["coords"], legacy["coords"]):
            np.testing.assert_array_equal(orch["coords"], legacy["coords"], err_msg="coords")


if __name__ == "__main__":
    unittest.main()
