"""B7 sig-het trust region + diagnostics: gate math and STFT non-interference.

Two concerns, deliberately separated.

**1. Non-interference (the reason this file exists).** Every piece of the
B7 machinery lives behind ``sighet_active`` -- the truthiness of
``buffer_obj.setup_in_model_likelihood(...)``, which is the engine's
``setup_in_model`` hook. On this branch that hook is a hard ``return None``
for the FD comps (``GBFDComputations``), the chunked-het comps
(``WDMComputationsBase``) **and the STFT comps** (``STFTGBComputations``);
only ``gbgpu.GBSignalHetComputations`` -- reachable solely by handing a
sig-het comp to the *WDM* band engine -- returns truthy. So on the STFT arm
the trust region, the SNR gate, the anchor check and both end-of-block
audits are all structurally unreachable.

That matters because the gate math needs ``Tobs``, and **STFTSettings does
not have a Tobs attribute** (neither does FDSettings) -- only WDMSettings
and TDSettings do. An unconditional ``self._basis_settings.Tobs`` read
would ``AttributeError`` on every STFT and FD GB flow, which is exactly the
regression dev's ``f3abda7`` fixed. These tests pin all three facts: the
missing attribute, the no-op hook, and -- through the real STFT propose
flow with a booby-trapped ``Tobs`` -- that nothing on the STFT path reads
it.

**2. Gate math.** Because sig-het is dormant on this branch, the ported
arithmetic would otherwise be entirely uncovered. The helpers are pure
functions of ``self.xp`` / ``self.transform_fn`` / the ctor knobs, so they
are exercised directly against a minimal stand-in rather than through a
sig-het run that cannot be built here.
"""

from __future__ import annotations

import unittest

import numpy as np


def _have_gbgpu_stft() -> bool:
    try:
        from gbgpu.gbcomps import STFTGBComputations  # noqa: F401

        return True
    except (ImportError, ModuleNotFoundError):
        return False


# --------------------------------------------------------------------------
# 1. Non-interference
# --------------------------------------------------------------------------


class SigHetTobsGuardTest(unittest.TestCase):
    """``Tobs`` is not universal across domain settings -- hence the guard."""

    @staticmethod
    def _defines_tobs(settings_cls) -> bool:
        """Does ``cls`` provide ``.Tobs`` -- as a property or an instance attr?

        ``hasattr`` on the class is not enough: WDMSettings assigns
        ``self.Tobs`` in ``__init__`` (invisible on the class) while TDSettings
        uses a property. Read the class body for either shape.
        """
        import inspect
        import re

        if hasattr(settings_cls, "Tobs"):
            return True
        body = inspect.getsource(settings_cls)
        return bool(re.search(r"^\s*self\.Tobs\s*=", body, re.M))

    def test_only_wdm_and_td_settings_expose_tobs(self):
        from lisatools.domains import FDSettings, STFTSettings, TDSettings, WDMSettings

        for cls in (STFTSettings, FDSettings):
            self.assertFalse(
                self._defines_tobs(cls),
                f"{cls.__name__} unexpectedly exposes Tobs -- if this basis really "
                "grew a Tobs, the trust-region guard in _run_in_model_repeats can "
                "be simplified; until then the guard is load-bearing.",
            )
        for cls in (WDMSettings, TDSettings):
            self.assertTrue(
                self._defines_tobs(cls),
                f"{cls.__name__} lost Tobs; the sig-het drift metric reads it.",
            )

    def test_trust_knob_defaults_and_env_wiring(self):
        """Defaults match dev; the env vars are the only run-time surface."""
        import os

        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase

        import inspect

        defaults = {
            k: v.default
            for k, v in inspect.signature(GBSpecialBase.__init__).parameters.items()
        }
        self.assertEqual(defaults["sighet_trust_dlna"], 1.5)
        self.assertEqual(defaults["sighet_trust_dphase"], 0.5)
        self.assertEqual(defaults["sighet_trust_snr_c"], 30.0)
        self.assertEqual(defaults["sighet_trust_dlna_min"], 0.3)
        # Diagnostics cost an extra exact engine call per block -- off unless
        # explicitly asked for.
        self.assertIs(defaults["sighet_anchor_check"], False)
        self.assertIs(defaults["sighet_drift_check"], False)

        # The recipe reads these names; a rename here silently disarms every
        # existing run script.
        import lisatools.globalfit.recipe as _recipe

        src = open(_recipe.__file__).read()
        for var in (
            "GB_SIGHET_TRUST_DLNA",
            "GB_SIGHET_TRUST_DPHASE",
            "GB_SIGHET_TRUST_SNR_C",
            "GB_SIGHET_TRUST_DLNA_MIN",
            "GB_SIGHET_ANCHOR_CHECK",
            "GB_SIGHET_DRIFT_CHECK",
        ):
            self.assertIn(var, src, f"{var} not wired in recipe.build_gb_moves")
        del os


@unittest.skipUnless(_have_gbgpu_stft(), "requires gbgpu.gbcomps.STFTGBComputations")
class SigHetInertOnStftTest(unittest.TestCase):
    """The STFT arm can never arm sig-het, so B7 is structurally inert there."""

    def test_stft_comp_in_model_hook_is_noop(self):
        from gbgpu.gbcomps import GBFDComputations, STFTGBComputations

        from lisatools.chunked_het import WDMComputationsBase

        # ``sighet_active = bool(setup_in_model_likelihood(...))`` -- a hook
        # returning None keeps every B7 branch unreachable. Called unbound
        # with a dummy self: these overrides ignore every argument, and
        # building real comps here would need a full grid fixture.
        for comp_cls in (STFTGBComputations, GBFDComputations, WDMComputationsBase):
            self.assertIsNone(
                comp_cls.setup_in_model(
                    object(), None, None, None, N_vals=None
                ),
                f"{comp_cls.__name__}.setup_in_model is no longer a no-op; the "
                "B7 trust region would now arm on that arm and needs a Tobs.",
            )

    def test_in_model_repeats_never_read_tobs_on_stft(self):
        """Real STFT propose flow with a Tobs that explodes if touched.

        The end-to-end guard: booby-trap ``STFTSettings.Tobs`` and run the
        in-model move. This is the test that would have caught the dev
        regression f3abda7 fixed.
        """
        from lisatools.domains import STFTSettings
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialStretchMove

        from .test_gbspecial_flow_stft import build_fixture

        fx = build_fixture()
        move = GBSpecialStretchMove(
            *fx["move_args"], is_rj_prop=False, name="stretch_stft_tobs_guard",
            stretch_probability=0.5,
            # Trust region + both audits explicitly ON: even fully armed by
            # the knobs, the STFT arm must not reach the gate math.
            sighet_trust_dlna=1.5,
            sighet_anchor_check=True,
            sighet_drift_check=True,
            **fx["move_kwargs"],
        )
        move.temperature_control = fx["temperature_control"]
        move.time = 0

        touched = []

        def _boom(self):
            touched.append(True)
            raise AssertionError(
                "in-model repeats read _basis_settings.Tobs on the STFT arm"
            )

        self.assertFalse(hasattr(STFTSettings, "Tobs"))
        STFTSettings.Tobs = property(_boom)
        try:
            new_state, _ = move.propose(fx["model"], fx["state"])
        finally:
            del STFTSettings.Tobs

        self.assertEqual(touched, [])
        self.assertTrue(np.all(np.isfinite(new_state.log_like)))
        # ... and the block actually ran (an empty flow would pass vacuously).
        self.assertGreater(
            int(new_state.sub_states["gb"].band_info["band_num_proposed"].sum()), 0
        )


# --------------------------------------------------------------------------
# 2. Gate math
# --------------------------------------------------------------------------


class _GateStub:
    """Minimal stand-in exposing what the three B7 helpers actually touch."""

    _sighet_drift_metrics = None  # bound below
    _sighet_trust_dlna_vec = None
    _sighet_anchor_phys = None

    def __init__(self, Tobs=1e7, snr_c=30.0, dlna=1.5, dlna_min=0.3):
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase
        from lisatools.globalfit.stock.erebor import make_gb_transform_container

        self.xp = np
        self.transform_fn = make_gb_transform_container()
        self._basis_settings = type("S", (), {"Tobs": Tobs})()
        self.sighet_trust_snr_c = float(snr_c)
        self.sighet_trust_dlna = float(dlna)
        self.sighet_trust_dlna_min = float(dlna_min)
        for name in (
            "_sighet_drift_metrics",
            "_sighet_trust_dlna_vec",
            "_sighet_anchor_phys",
        ):
            setattr(self, name, getattr(GBSpecialBase, name).__get__(self))


class _HHStub:
    def __init__(self, h_h_out):
        self.h_h_out = np.asarray(h_h_out)


def _coords(lnA=None, f0_mhz=None, fdot=None, n=1):
    """Sampling-basis GB coords: [lnA, f0(mHz), fdot, phi0, cos_i, psi, lam, sin_b]."""
    c = np.zeros((n, 8))
    c[:, 0] = np.log(1e-21) if lnA is None else lnA
    c[:, 1] = 3.0 if f0_mhz is None else f0_mhz
    c[:, 2] = 0.0 if fdot is None else fdot
    c[:, 4] = 0.5
    return c


class SigHetDriftMetricTest(unittest.TestCase):
    """``_sighet_drift_metrics`` in the physical basis (dev 36bddb9)."""

    Tobs = 1e7

    def test_zero_drift_at_the_anchor(self):
        s = _GateStub(Tobs=self.Tobs)
        c = _coords(n=3)
        drift, damp = s._sighet_drift_metrics(c, c.copy())
        np.testing.assert_allclose(drift, 0.0, atol=0.0)
        np.testing.assert_allclose(damp, 0.0, atol=0.0)

    def test_carrier_phase_drift_uses_hz_not_mhz(self):
        """The mHz -> Hz conversion is the whole point of the physical basis.

        Sampling column 1 is mHz; the drift formula wants Hz. A 1e-9 Hz
        offset over Tobs = 1e7 s is 2*pi*1e-2 rad -- 1000x larger if the
        mHz value were used raw.
        """
        s = _GateStub(Tobs=self.Tobs)
        ref = _coords()
        cur = _coords(f0_mhz=3.0 + 1e-6)  # +1e-6 mHz == +1e-9 Hz
        drift, _ = s._sighet_drift_metrics(cur, ref)
        # rtol, not places: differencing two ~3e-3 Hz values to recover 1e-9 Hz
        # loses ~7 digits to cancellation. The point of the test is the 1e3
        # unit factor, which is 3 orders of magnitude away from that noise.
        np.testing.assert_allclose(
            drift[0], 2.0 * np.pi * 1e-9 * self.Tobs, rtol=1e-6
        )

    def test_fdot_term_is_quadratic_in_tobs(self):
        s = _GateStub(Tobs=self.Tobs)
        ref = _coords()
        cur = _coords(fdot=1e-18)
        drift, _ = s._sighet_drift_metrics(cur, ref)
        self.assertAlmostEqual(
            float(drift[0]), np.pi * 1e-18 * self.Tobs**2, places=12
        )

    def test_damp_is_the_log_amplitude_ratio(self):
        """Physical col 0 is A (=exp(lnA)); damp must come back as |dlnA|."""
        s = _GateStub()
        ref = _coords(lnA=np.log(1e-21))
        cur = _coords(lnA=np.log(1e-21) + 0.75)
        _, damp = s._sighet_drift_metrics(cur, ref)
        self.assertAlmostEqual(float(damp[0]), 0.75, places=12)
        # symmetric in the ratio
        _, damp_rev = s._sighet_drift_metrics(ref, cur)
        self.assertAlmostEqual(float(damp_rev[0]), 0.75, places=12)

    def test_matches_the_legacy_sampling_basis_form(self):
        """On this branch's full 8-col GB basis the rewrite is behavior-neutral.

        The old form read sampling columns directly (col 0 already lnA, col 1
        mHz/1e3). Equivalent here -- the rewrite buys basis-agnosticism, not a
        numeric change -- so any divergence beyond the cancellation floor means
        the transform or the column convention moved.

        The tolerance is ~1e-8, not machine epsilon: the physical path scales
        f0 to Hz *before* differencing while the legacy path differences in mHz
        and scales after, so the two lose different low-order bits to the
        cancellation. A convention error would show as a factor, not 1e-10.
        """
        s = _GateStub(Tobs=self.Tobs)
        rng = np.random.default_rng(7)
        ref = _coords(n=16)
        cur = _coords(n=16)
        cur[:, 0] += rng.normal(0.0, 0.4, 16)
        cur[:, 1] += rng.normal(0.0, 1e-5, 16)
        cur[:, 2] += rng.normal(0.0, 1e-18, 16)

        drift, damp = s._sighet_drift_metrics(cur, ref)

        df0_hz = np.abs(cur[:, 1] - ref[:, 1]) / 1e3
        dfdot = np.abs(cur[:, 2] - ref[:, 2])
        drift_legacy = (
            2.0 * np.pi * df0_hz * self.Tobs + np.pi * dfdot * self.Tobs**2
        )
        damp_legacy = np.abs(cur[:, 0] - ref[:, 0])

        np.testing.assert_allclose(drift, drift_legacy, rtol=1e-8)
        np.testing.assert_allclose(damp, damp_legacy, rtol=1e-12)


class SigHetTrustDlnaVecTest(unittest.TestCase):
    """``_sighet_trust_dlna_vec``: clip(C/snr_ref, dlna_min, dlna_cap)."""

    def test_snr_c_zero_gives_the_uniform_cap(self):
        s = _GateStub(snr_c=0.0, dlna=1.5)
        out = s._sighet_trust_dlna_vec(_HHStub([1.0, 1e4, 0.0]), 3)
        np.testing.assert_allclose(out, 1.5)
        self.assertEqual(out.shape, (3,))

    def test_scales_as_c_over_snr_between_the_clips(self):
        # snr_ref = sqrt(h_h) = 40 -> 30/40 = 0.75, inside [0.3, 1.5]
        s = _GateStub(snr_c=30.0, dlna=1.5, dlna_min=0.3)
        out = s._sighet_trust_dlna_vec(_HHStub([1600.0]), 1)
        self.assertAlmostEqual(float(out[0]), 0.75, places=12)

    def test_loud_sources_clip_at_dlna_min_weak_at_the_cap(self):
        s = _GateStub(snr_c=30.0, dlna=1.5, dlna_min=0.3)
        # snr 200 -> 0.15 -> clipped up to 0.3;  snr 4 -> 7.5 -> down to 1.5
        out = s._sighet_trust_dlna_vec(_HHStub([200.0**2, 4.0**2]), 2)
        self.assertAlmostEqual(float(out[0]), 0.3, places=12)
        self.assertAlmostEqual(float(out[1]), 1.5, places=12)

    def test_monotone_non_increasing_in_snr(self):
        s = _GateStub(snr_c=30.0)
        snr = np.array([1.0, 5.0, 20.0, 50.0, 100.0, 400.0])
        out = s._sighet_trust_dlna_vec(_HHStub(snr**2), len(snr))
        self.assertTrue(np.all(np.diff(out) <= 0.0))

    def test_zero_and_negative_hh_do_not_blow_up(self):
        """h_h can come back 0 (or -0/-eps from cancellation) for junk sources."""
        s = _GateStub(snr_c=30.0, dlna=1.5)
        out = s._sighet_trust_dlna_vec(_HHStub([0.0, -1e-30]), 2)
        self.assertTrue(np.all(np.isfinite(out)))
        np.testing.assert_allclose(out, 1.5)

    def test_accepts_complex_hh(self):
        """The buffer's h_h_out is complex on some engines; .real is taken."""
        s = _GateStub(snr_c=30.0, dlna=1.5, dlna_min=0.3)
        out = s._sighet_trust_dlna_vec(_HHStub(np.array([1600.0 + 3.0j])), 1)
        self.assertAlmostEqual(float(out[0]), 0.75, places=12)


class SigHetAnchorPhysTest(unittest.TestCase):
    """``_sighet_anchor_phys`` returns the anchor's (|A|, f0, fdot)."""

    def test_returns_physical_amplitude_f0_hz_and_fdot(self):
        s = _GateStub()
        ref = _coords(lnA=np.log(3e-21), f0_mhz=3.0, fdot=1e-17)
        A, f0, fdot = s._sighet_anchor_phys(ref)
        self.assertAlmostEqual(float(A[0]), 3e-21, places=30)
        self.assertAlmostEqual(float(f0[0]), 3e-3, places=15)
        self.assertAlmostEqual(float(fdot[0]), 1e-17, places=25)

    def test_is_a_copy_not_a_view_of_the_transform_output(self):
        """The block holds these across repeats; aliasing would drift silently."""
        s = _GateStub()
        ref = _coords(f0_mhz=3.0, n=2)
        A, f0, fdot = s._sighet_anchor_phys(ref)
        f0_before = f0.copy()
        ref[:, 1] = 4.0  # mutate the anchor coords afterwards
        s._sighet_anchor_phys(ref)
        np.testing.assert_allclose(f0, f0_before)

    def test_agrees_with_the_drift_metric_at_a_displaced_point(self):
        """Anchor-cache path and both-sides path must give the same gate value."""
        s = _GateStub(Tobs=1e7)
        ref = _coords(lnA=np.log(1e-21), f0_mhz=3.0, fdot=0.0, n=4)
        cur = _coords(lnA=np.log(1e-21) + 0.6, f0_mhz=3.0 + 2e-6, fdot=5e-19, n=4)

        drift_both, damp_both = s._sighet_drift_metrics(cur, ref)

        A_ref, f0_ref, fdot_ref = s._sighet_anchor_phys(ref)
        pc = s.transform_fn.both_transforms(cur, xp=np)
        damp_cache = np.abs(np.log(np.abs(pc[:, 0]) / A_ref))
        drift_cache = (
            2.0 * np.pi * np.abs(pc[:, 1] - f0_ref) * 1e7
            + np.pi * np.abs(pc[:, 2] - fdot_ref) * 1e7**2
        )

        np.testing.assert_allclose(drift_cache, drift_both, rtol=1e-12)
        np.testing.assert_allclose(damp_cache, damp_both, rtol=1e-12)


class SigHetGateRejectionTest(unittest.TestCase):
    """The gate's effect on ``new_logp``, as applied in the repeat loop."""

    Tobs = 1e7

    def _gate(self, s, new, anchor_phys, trust_dlna, dphase):
        pc = s.transform_fn.both_transforms(new, xp=np)
        damp = np.abs(np.log(np.abs(pc[:, 0]) / anchor_phys[0]))
        drift = (
            2.0 * np.pi * np.abs(pc[:, 1] - anchor_phys[1]) * self.Tobs
            + np.pi * np.abs(pc[:, 2] - anchor_phys[2]) * self.Tobs**2
        )
        return (damp > trust_dlna) | (drift > dphase)

    def test_the_anchor_itself_is_always_inside(self):
        """MH validity rests on this: x is in its own trust region."""
        s = _GateStub(Tobs=self.Tobs)
        ref = _coords(n=5)
        anchor_phys = s._sighet_anchor_phys(ref)
        trust = s._sighet_trust_dlna_vec(_HHStub(np.full(5, 100.0)), 5)
        rejected = self._gate(s, ref.copy(), anchor_phys, trust, 0.5)
        self.assertFalse(bool(rejected.any()))

    def test_amplitude_excursion_beyond_the_per_source_gate_is_rejected(self):
        s = _GateStub(Tobs=self.Tobs, snr_c=30.0, dlna=1.5, dlna_min=0.3)
        ref = _coords(lnA=np.log(1e-21), n=2)
        anchor_phys = s._sighet_anchor_phys(ref)
        # snr 40 -> gate 0.75 for both sources
        trust = s._sighet_trust_dlna_vec(_HHStub([1600.0, 1600.0]), 2)
        new = _coords(lnA=np.log(1e-21), n=2)
        new[0, 0] += 0.5   # inside
        new[1, 0] += 1.0   # outside
        rejected = self._gate(s, new, anchor_phys, trust, 0.5)
        self.assertFalse(bool(rejected[0]))
        self.assertTrue(bool(rejected[1]))

    def test_a_loud_source_gets_a_tighter_gate_than_a_weak_one(self):
        """Same |dlnA| excursion, opposite verdicts -- the point of af92a8a."""
        s = _GateStub(Tobs=self.Tobs, snr_c=30.0, dlna=1.5, dlna_min=0.3)
        ref = _coords(lnA=np.log(1e-21), n=2)
        anchor_phys = s._sighet_anchor_phys(ref)
        trust = s._sighet_trust_dlna_vec(_HHStub([200.0**2, 4.0**2]), 2)
        new = _coords(lnA=np.log(1e-21) + 0.5, n=2)
        rejected = self._gate(s, new, anchor_phys, trust, 0.5)
        self.assertTrue(bool(rejected[0]), "SNR-200 source should be gated at 0.3")
        self.assertFalse(bool(rejected[1]), "SNR-4 source keeps the 1.5 cap")

    def test_phase_drift_gate_fires_independently_of_amplitude(self):
        s = _GateStub(Tobs=self.Tobs, snr_c=0.0, dlna=1.5)
        ref = _coords(f0_mhz=3.0, n=1)
        anchor_phys = s._sighet_anchor_phys(ref)
        trust = s._sighet_trust_dlna_vec(_HHStub([100.0]), 1)
        # 0.5 rad / (2*pi*Tobs) = 7.96e-9 Hz; go well past it, amplitude fixed
        new = _coords(f0_mhz=3.0 + 1e-5, n=1)
        self.assertTrue(bool(self._gate(s, new, anchor_phys, trust, 0.5)[0]))
        # and just inside it
        near = _coords(f0_mhz=3.0 + 1e-9 * 1e3 * 0.5 / (2 * np.pi * self.Tobs) * 1e9, n=1)
        self.assertFalse(bool(self._gate(s, near, anchor_phys, trust, 0.5)[0]))

    def test_gate_is_symmetric_in_the_two_coordinates(self):
        """Symmetry of the indicator is what keeps the MH ratio unchanged."""
        s = _GateStub(Tobs=self.Tobs, snr_c=0.0, dlna=0.5)
        x = _coords(lnA=np.log(1e-21), f0_mhz=3.0)
        y = _coords(lnA=np.log(1e-21) + 0.8, f0_mhz=3.0)
        trust = s._sighet_trust_dlna_vec(_HHStub([100.0]), 1)
        x_from_y = self._gate(s, x, s._sighet_anchor_phys(y), trust, 1e9)
        y_from_x = self._gate(s, y, s._sighet_anchor_phys(x), trust, 1e9)
        self.assertEqual(bool(x_from_y[0]), bool(y_from_x[0]))
        self.assertTrue(bool(x_from_y[0]))


@unittest.skipUnless(_have_gbgpu_stft(), "requires gbgpu.gbcomps.STFTGBComputations")
class SigHetTrustDlnaDisableTest(unittest.TestCase):
    """``sighet_trust_dlna = 0`` is the documented disable switch.

    The move reads it as ``sighet_active and self.sighet_trust_dlna > 0.0``
    to decide whether ``anchor_phys`` exists at all, and gates the ``Tobs``
    read on ``anchor_phys`` rather than on ``sighet_active``. Both halves
    are checked behaviorally, by FORCING sig-het active on the STFT arm
    (which cannot arm it for real) over a basis that has no ``Tobs``:

    * ``dlna = 0`` -> gate never built -> Tobs never read -> flow completes.
    * ``dlna = 1.5`` -> gate built -> Tobs read -> AttributeError.

    The second case is what makes the first meaningful: it proves the guard
    is load-bearing rather than the read being unreachable for some other
    reason. It is also precisely the dev regression ``f3abda7`` fixed.
    """

    def _run_with_forced_sighet(self, trust_dlna):
        from lisatools.globalfit.moves.gbbands import SubBandBuffer
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialStretchMove

        from .test_gbspecial_flow_stft import build_fixture

        fx = build_fixture()
        move = GBSpecialStretchMove(
            *fx["move_args"], is_rj_prop=False,
            name=f"stretch_forced_sighet_{trust_dlna}",
            stretch_probability=0.5,
            sighet_trust_dlna=trust_dlna,
            # Keep the diagnostics off: they have their own Tobs reads and
            # would confound which branch tripped.
            sighet_anchor_check=False,
            sighet_drift_check=False,
            # 3 repeats vs refresh-every-20: the mid-block refresh (which
            # also needs Tobs) cannot fire, isolating the trust-region read.
            **fx["move_kwargs"],
        )
        move.temperature_control = fx["temperature_control"]
        move.time = 0

        real_setup = SubBandBuffer.setup_in_model_likelihood

        def _forced(self, *a, **kw):
            real_setup(self, *a, **kw)  # keep the engine call's side effects
            return True  # ... but claim a reference is active

        SubBandBuffer.setup_in_model_likelihood = _forced
        try:
            return move.propose(fx["model"], fx["state"])
        finally:
            SubBandBuffer.setup_in_model_likelihood = real_setup

    def test_dlna_zero_never_builds_the_gate_so_tobs_is_never_read(self):
        new_state, _ = self._run_with_forced_sighet(0.0)
        self.assertTrue(np.all(np.isfinite(new_state.log_like)))
        self.assertGreater(
            int(new_state.sub_states["gb"].band_info["band_num_proposed"].sum()), 0
        )

    def test_gate_enabled_over_a_tobs_less_basis_is_what_the_guard_prevents(self):
        with self.assertRaises(AttributeError) as ctx:
            self._run_with_forced_sighet(1.5)
        self.assertIn("Tobs", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
