"""TRUTH-INJECTION NULL TEST wiring (user ruling 2026-09-14 evening).

The test the user asked for, replacing the nogb test config: "remove the psd
fitting, use best fit values for psd and galfor from the 3mo run. NO INJECTED
NOISE. Just psd modelling for the likelihood. Turn all the start factors for
SOBHB, MBHB, EMRI to zero. Only include sobhb, mbhb, emris in the fit. I want
to see our injection log-likelihood WITHOUT the noise term -- how close to
zero it is when we inject the truth (deviation from truth ~0)."

The physics: the data is source streams ONLY (no noise realization), every
branch starts EXACTLY at truth, and the likelihood is the source-only
convention ``-1/2 <d-h|d-h>`` weighted by a FIXED sensitivity built from the
3mo run's best-fit psd + galfor. At the null point the residual cancels and
lnL -> 0: sources synthesized from the catalogue null EXACTLY by
construction, while sources loaded from real mojito bricks leave whatever the
mojito <-> template waveform mismatch is -- THE number the test measures.

WHY THE BUILD ASSERTIONS RUN IN A SUBPROCESS. ``erebor``'s stock variants are
MODULE-LEVEL INSTANCES (``all_sources = AllSourcesGlobalFit()`` in
``stock/erebor/__init__.py``), so every env-backed settings field resolves
ONCE at import and ``erebor.all_sources(...)`` clones that snapshot. Driver
knobs read inside ``build_fit`` (REMOVE_BRANCHES, SOURCE_TYPES, the id lists,
the STAGE_* flags) therefore respond to an in-process ``os.environ`` edit,
but SETTINGS knobs (ADD_INSTRUMENT_NOISE, UNEQUAL_ARM,
LIKELIHOOD_SOURCE_ONLY) do NOT -- they are frozen by whichever test imported
erebor first. Probing in a fresh process is both order-independent and
faithful to the real launch, where the submit script exports everything
before ``python`` starts.

What is asserted here (composition level only -- no data built, no
materialization; the RemoveBranchesWiringTest pattern):

* the driver accepts ``psd`` in ``REMOVE_BRANCHES`` and composes
  source_search -> full_pe with the three source PE moves and NO psd move;
* the guards: psd cannot go while gb or galfor is still sampled, nor with
  injected instrument noise, nor with UNEQUAL_ARM, nor with a noise-stage
  flag set;
* ``PSD_FIXED_PARAMS`` / ``GALFOR_FIXED_PARAMS`` land on
  ``general.fixed_psd_kwargs`` (and an unset env leaves the default alone);
* ``likelihood_source_only`` / ``add_instrument_noise`` are env-backed
  (rule 0: LIKELIHOOD_SOURCE_ONLY / ADD_INSTRUMENT_NOISE);
* ``<BRANCH>_START_FACTOR = 0`` seeds EXACT truth.
"""
import json
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "fstat_proposal" / "run_combined_staged.py"

ALL_IDS = {"MBHB_IDS": "2,5,16,18",
           "EMRI_IDS": "0,1,2,3,4,5,6,7",
           "SOBHB_IDS": "0,1,2,3,4,5"}

# The null-test env contract exactly as submit_gf_6mo_v8_nogb_null.sh exports
# it: the three source branches armed, every other branch removed, and NO
# injected noise (the composition guards refuse anything else).
NULL_CONTRACT = dict(
    ALL_IDS,
    REMOVE_BRANCHES="gb,galfor,vgb,psd",
    ADD_INSTRUMENT_NOISE="0",
    UNEQUAL_ARM="0",
    LIKELIHOOD_SOURCE_ONLY="1",
    MBH_START_FACTOR="0",
    EMRI_START_FACTOR="0",
    SOBBH_START_FACTOR="0",
)

# Everything this suite drives, cleared from the child's env so an inherited
# shell value cannot silently change an assertion.
_CLEARED = tuple(NULL_CONTRACT) + (
    "SOURCE_TYPES", "GB_ONLY", "STAGE_SKIP_NOISE", "STAGE_NOISE_ONLY",
    "STAGE_NOISE_VGB_PE", "COMBINED_SMOKE", "TOBS_TARGET",
    "GB_WARM_START_COMPONENTS", "DATA_MODE", "PSD_FIXED_PARAMS",
    "GALFOR_FIXED_PARAMS",
)

_PROBE = textwrap.dedent(
    """
    import importlib.util, json, sys
    spec = importlib.util.spec_from_file_location("drv", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fit = mod.build_fit()
    gen = fit.general
    kw = gen.fixed_psd_kwargs or {}
    out = dict(
        branches=list(fit.branches),
        stages=[st.name for st in fit.recipe.stages],
        kinds=[st.kind for st in fit.recipe.stages],
        moves={st.name: [m.name for m in st.moves]
               for st in fit.recipe.stages},
        source_types=list(gen.source_types),
        psd_params=(None if kw.get("psd_params") is None
                    else [float(x) for x in kw["psd_params"]]),
        galfor_params=(None if kw.get("galfor_params") is None
                       else [float(x) for x in kw["galfor_params"]]),
        likelihood_source_only=bool(gen.likelihood_source_only),
        add_instrument_noise=gen.add_instrument_noise,
        unequal_arm=bool(gen.unequal_arm),
    )
    print("@@PROBE@@" + json.dumps(out))
    """
)


def _probe(env_updates, clear_all_ids=False):
    """Build the fit in a FRESH process under ``env_updates``.

    Returns ``(summary_dict_or_None, combined_output)``; the dict is None when
    the build raised, so a caller can assert on the message.
    """
    env = dict(os.environ)
    for k in _CLEARED:
        env.pop(k, None)
    if clear_all_ids:
        for k in ALL_IDS:
            env.pop(k, None)
    env.update({k: str(v) for k, v in env_updates.items()})
    env.setdefault("NWALKERS", "4")
    # House threading policy: exactly one thread per pool in the child.
    env.update(OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(SCRIPT)],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=1200,
    )
    blob = proc.stdout + proc.stderr
    for line in proc.stdout.splitlines():
        if line.startswith("@@PROBE@@"):
            return json.loads(line[len("@@PROBE@@"):]), blob
    return None, blob


class NullCompositionTest(unittest.TestCase):
    """REMOVE_BRANCHES=gb,galfor,vgb,psd -- the null-test composition."""

    @classmethod
    def setUpClass(cls):
        cls.summary, cls.out = _probe(NULL_CONTRACT)
        if cls.summary is None:
            raise AssertionError(
                f"the null composition failed to build:\n{cls.out}")

    def test_only_the_three_source_branches_remain(self):
        for b in ("gb", "galfor", "vgb", "psd"):
            self.assertNotIn(b, self.summary["branches"])
        for b in ("sobbh", "mbh", "emri"):
            self.assertIn(b, self.summary["branches"])

    def test_stage_ladder_is_source_search_then_full_pe(self):
        # no noise stages at all: there is no sampled noise branch left
        self.assertEqual(self.summary["stages"], ["source_search", "full_pe"])
        self.assertEqual(self.summary["kinds"], ["search", "pe"])

    def test_full_pe_is_exactly_the_three_source_moves(self):
        self.assertEqual(self.summary["moves"]["full_pe"],
                         ["sobbh_pe", "mbh_pe", "emri_pe"])

    def test_no_psd_galfor_vgb_or_gb_move_anywhere(self):
        for name, names in self.summary["moves"].items():
            for mv in names:
                for banned in ("psd", "galfor", "vgb"):
                    self.assertNotIn(banned, mv, f"{name}: {names}")
                self.assertFalse(
                    mv.startswith("rj_") or mv.startswith("gb_"),
                    f"{name}: {names}")

    def test_no_injected_noise_stream(self):
        # "NO INJECTED NOISE": the NOISE stream leaves the DEFAULT list with
        # the psd branch (nothing models it any more), and the resolved
        # instrument-noise knob is False
        self.assertEqual(self.summary["source_types"],
                         ["SOBHB", "MBHB", "EMRI"])
        self.assertIs(self.summary["add_instrument_noise"], False)

    def test_source_only_likelihood_survives_without_a_psd_branch(self):
        # run.py force-disables source-only when a psd branch is present;
        # here there is none, so the ruling's "WITHOUT the noise term" holds
        self.assertTrue(self.summary["likelihood_source_only"])

    def test_equal_arm_model(self):
        self.assertFalse(self.summary["unequal_arm"])


class NullCompositionGuardTest(unittest.TestCase):
    """The composition guards -- each a silent-wrongness trap if it passed."""

    def _expect_refusal(self, env_updates, needle):
        summary, out = _probe(env_updates)
        self.assertIsNone(
            summary, f"expected a refusal mentioning {needle!r}; built "
                     f"instead:\n{out}")
        self.assertIn("ValueError", out, out)
        self.assertIn(needle, out, out)

    def test_psd_removal_requires_galfor_removed(self):
        # galfor's 5-parameter foreground is a COMPONENT of the sampled psd
        # branch's sensitivity, so this composition is incoherent
        self._expect_refusal(
            dict(NULL_CONTRACT, REMOVE_BRANCHES="gb,vgb,psd"), "galfor")

    def test_psd_removal_requires_gb_removed(self):
        self._expect_refusal(
            dict(NULL_CONTRACT, REMOVE_BRANCHES="galfor,vgb,psd"), "gb")

    def test_psd_alone_still_rejected(self):
        # the pre-existing contract (RemoveBranchesWiringTest): a bare
        # REMOVE_BRANCHES=psd raises -- now because galfor/gb are present
        self._expect_refusal(dict(REMOVE_BRANCHES="psd"), "ValueError")

    def test_injected_noise_refused_without_psd(self):
        """NO INJECTED NOISE: unmodelled noise cannot null to zero."""
        self._expect_refusal(
            dict(NULL_CONTRACT, ADD_INSTRUMENT_NOISE="1"),
            "add_instrument_noise")

    def test_unequal_arm_refused_without_psd(self):
        """UNEQUAL_ARM installs ON the psd branch, which is gone."""
        self._expect_refusal(
            dict(NULL_CONTRACT, UNEQUAL_ARM="1"), "UNEQUAL_ARM")

    def test_noise_stage_flags_refused_without_psd(self):
        for flag in ("STAGE_NOISE_ONLY", "STAGE_NOISE_VGB_PE",
                     "STAGE_SKIP_NOISE"):
            with self.subTest(flag=flag):
                self._expect_refusal(
                    dict(NULL_CONTRACT, **{flag: "1"}), flag)


class NogbCompositionUnchangedTest(unittest.TestCase):
    """The EXISTING nogb composition (psd KEPT) is untouched by all of this.

    Regression guard on the psd-aware edits: with psd present the stage
    ladder, the noise moves and the NOISE stream are exactly what
    RemoveBranchesWiringTest already pins.
    """

    def test_psd_kept_composition(self):
        summary, out = _probe(
            dict(ALL_IDS, REMOVE_BRANCHES="gb,galfor", ADD_INSTRUMENT_NOISE="1"))
        self.assertIsNotNone(summary, out)
        self.assertIn("psd", summary["branches"])
        self.assertEqual(
            summary["stages"],
            ["source_search", "noise_search", "noise_vgb_search", "full_pe"])
        self.assertEqual(
            summary["moves"]["full_pe"],
            ["psd_pe", "sobbh_pe", "mbh_pe", "emri_pe", "vgb_pe"])
        self.assertEqual(summary["source_types"],
                         ["NOISE", "VGB", "SOBHB", "MBHB", "EMRI"])


class FixedPsdKwargsTest(unittest.TestCase):
    """PSD_FIXED_PARAMS / GALFOR_FIXED_PARAMS -> general.fixed_psd_kwargs.

    The fixed-sensitivity path (run.py setup_acs, no psd branch) builds every
    walker's AnalysisContainer from ``general.fixed_psd_kwargs`` and applies
    NO transform, so the values must already be PHYSICAL (linear). With the
    psd branch gone these two env knobs are the ONLY way the 3mo run's
    best-fit noise reaches the likelihood.
    """

    def test_env_params_land_on_general(self):
        summary, out = _probe(dict(
            NULL_CONTRACT,
            PSD_FIXED_PARAMS="1.2e-11,2.9e-15",
            GALFOR_FIXED_PARAMS="-39.0,1.1,-2.2,-3.3,-4.4",
        ))
        self.assertIsNotNone(summary, out)
        self.assertEqual(summary["psd_params"], [1.2e-11, 2.9e-15])
        self.assertEqual(summary["galfor_params"],
                         [-39.0, 1.1, -2.2, -3.3, -4.4])

    def test_absent_env_leaves_the_default(self):
        # engine.init_data_information fills the stock [15e-12, 3e-15] when
        # the field is still None; the driver must not invent anything.
        summary, out = _probe(NULL_CONTRACT)
        self.assertIsNotNone(summary, out)
        self.assertIsNone(summary["psd_params"])
        self.assertIsNone(summary["galfor_params"])

    def test_non_numeric_is_refused(self):
        summary, out = _probe(
            dict(NULL_CONTRACT, PSD_FIXED_PARAMS="1.2e-11,not_a_float"))
        self.assertIsNone(summary, out)
        self.assertIn("PSD_FIXED_PARAMS", out)


class EnvBackedNoiseFieldsTest(unittest.TestCase):
    """rule 0: the knob is the capitalized field name.

    Both of these were PLAIN dataclass defaults before the null test, i.e.
    unreachable from a submit script. Constructed directly here (not through
    the erebor module-level clone), so the factories run against this env.
    """

    def setUp(self):
        self._env = os.environ.copy()
        for k in ("LIKELIHOOD_SOURCE_ONLY", "ADD_INSTRUMENT_NOISE"):
            os.environ.pop(k, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._env)

    def _settings(self):
        from lisatools.globalfit.stock.erebor.variants.all_sources import (
            AllSourcesGeneralSettings,
        )

        return AllSourcesGeneralSettings

    def test_likelihood_source_only_env_backed(self):
        cls = self._settings()
        self.assertFalse(cls().likelihood_source_only)
        os.environ["LIKELIHOOD_SOURCE_ONLY"] = "1"
        self.assertTrue(cls().likelihood_source_only)

    def test_add_instrument_noise_takes_flag_or_mode(self):
        cls = self._settings()
        self.assertIs(cls().add_instrument_noise, True)
        os.environ["ADD_INSTRUMENT_NOISE"] = "0"
        self.assertIs(cls().add_instrument_noise, False)
        os.environ["ADD_INSTRUMENT_NOISE"] = "synthetic"
        self.assertEqual(cls().add_instrument_noise, "synthetic")

    def test_explicit_kwarg_still_beats_the_env(self):
        cls = self._settings()
        os.environ["ADD_INSTRUMENT_NOISE"] = "1"
        self.assertIs(cls(add_instrument_noise=False).add_instrument_noise,
                      False)


class NoNoiseStreamLoaderTest(unittest.TestCase):
    """The loader derives NO NOISE entry with add_instrument_noise=False.

    "NO INJECTED NOISE": the data must be source streams only. The hybrid
    loader derives its own stream list from the non-empty ``source_ids``
    classes and prepends NOISE only for ``add_instrument_noise="mojito"``
    (injections.py:1230-1235), so with the knob False there is no
    INSTRUMENT/L1 read at all -- and the grid + orbits still come from the
    FIRST source brick. Fixtures reused from test_hybrid_missing_bricks.
    """

    def setUp(self):
        import shutil
        import tempfile

        from tests.test_hybrid_missing_bricks import _make_fake_mojito

        self.tmp = tempfile.mkdtemp(prefix="null_no_noise_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        # a mojito folder with SOBHB catalogue + both bricks and NO
        # INSTRUMENT/L1 NOISE brick anywhere
        _make_fake_mojito(self.tmp, sobhb_bricks=(0, 1))

    def _build(self, **kw):
        """Construct the hybrid step, capturing the DERIVED source_types."""
        from unittest import mock

        from lisatools.globalfit.preprocessing import L1ProcessingStep
        from tests.test_hybrid_missing_bricks import (
            _FakeOrbits,
            _StubHybridStep,
        )

        seen = {}
        orig = L1ProcessingStep.__init__

        def _record(self, *a, **kwargs):
            seen.update(kwargs)
            return orig(self, *a, **kwargs)

        kw.setdefault("L1_folder", self.tmp)
        kw.setdefault("source_ids", {"SOBHB": [0, 1]})
        kw.setdefault("orbits_class", _FakeOrbits)
        kw.setdefault("verbose", False)
        with mock.patch.object(L1ProcessingStep, "__init__", _record):
            step = _StubHybridStep(**kw)
        return step, seen

    def test_no_noise_entry_and_orbits_from_first_brick(self):
        import numpy as np

        from tests.test_hybrid_missing_bricks import (
            DT_FAKE,
            N_FAKE,
            _FakeOrbits,
        )

        step, seen = self._build(add_instrument_noise=False)
        self.assertNotIn("NOISE", seen["source_types"])
        self.assertEqual(seen["source_types"], ["SOBHB"])
        # grid + orbits from the first real brick
        self.assertIsInstance(step.orbits, _FakeOrbits)
        self.assertEqual(step.dt, DT_FAKE)
        # the data is the two source bricks and NOTHING else: no instrument
        # noise summed on top
        np.testing.assert_allclose(step.data, 2.0 * np.ones((3, N_FAKE)))

    def test_mojito_noise_would_have_added_the_entry(self):
        # the paired control: the SAME fixture with the knob on 'mojito'
        # does derive a NOISE entry (and then fails on the absent brick),
        # so the assertion above is about the knob, not the fixture
        with self.assertRaises((FileNotFoundError, RuntimeError, OSError)):
            self._build(add_instrument_noise="mojito")

    def test_multi_class_order_has_no_noise(self):
        step, seen = self._build(
            source_ids={"SOBHB": [0, 1], "MBHB": [], "EMRI": []},
            add_instrument_noise=False,
        )
        self.assertNotIn("NOISE", seen["source_types"])


class StartFactorZeroTest(unittest.TestCase):
    """``<BRANCH>_START_FACTOR = 0`` seeds EXACT truth.

    The START_FACTOR convention is MULTIPLICATIVE ``x * (1 + factor * randn)``
    (``run.py::seed_injection_coords``, and the identical inline arithmetic in
    the mbh / emri / sobbh blocks at run.py:868 / :901 / :967), so a zero
    factor reproduces the injection bitwise for every walker and rung -- which
    is what "deviation from truth ~0" requires. Nothing floors or clips it,
    and there is no division by it.
    """

    def test_zero_factor_is_bitwise_truth(self):
        import numpy as np

        from lisatools.globalfit.run import seed_injection_coords

        inj = np.array([[1.0, -2.5, 3e-16], [4.0, 5.5, -6e-16]])
        out = seed_injection_coords(inj, 0.0, 3, 5)
        self.assertEqual(out.shape, (3, 5, 2, 3))
        for t in range(3):
            for w in range(5):
                np.testing.assert_array_equal(out[t, w], inj)

    def test_zero_factor_exact_with_additive_columns(self):
        import numpy as np

        from lisatools.globalfit.run import seed_injection_coords

        # the exactly-zero-truth exception still reproduces truth at 0
        inj = np.array([[1.0, 0.0, 3.0]])
        out = seed_injection_coords(
            inj, 0.0, 2, 2, additive_start_widths={1: 1e-16})
        for t in range(2):
            for w in range(2):
                np.testing.assert_array_equal(out[t, w], inj)

    def test_nonzero_factor_does_scatter(self):
        import numpy as np

        from lisatools.globalfit.run import seed_injection_coords

        inj = np.array([[1.0, -2.5, 3.0]])
        out = seed_injection_coords(inj, 1e-3, 2, 4)
        self.assertFalse(np.array_equal(out[0, 0], inj))
        np.testing.assert_allclose(out[0, 0], inj, rtol=1e-1)

    def test_all_walkers_identical_at_zero(self):
        # every walker starts at the SAME point; the run's inner moves are
        # EigenAxisMove by default (information-matrix tables, not ensemble
        # spread), which is what makes a zero-spread start viable
        import numpy as np

        from lisatools.globalfit.run import seed_injection_coords

        out = seed_injection_coords(np.array([[2.0, 3.0]]), 0.0, 2, 6)
        for w in range(6):
            np.testing.assert_array_equal(out[0, w], out[0, 0])

    def test_inner_move_kind_defaults_to_eigen(self):
        from lisatools.globalfit.stock.erebor.emri import EMRISettings
        from lisatools.globalfit.stock.erebor.mbh import MBHSettings
        from lisatools.globalfit.stock.erebor.sobbh import SOBBHSettings

        for cls in (MBHSettings, EMRISettings, SOBBHSettings):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls().inner_move_kind, "eigen")


if __name__ == "__main__":
    unittest.main()
