"""``GB_SEARCH_3_OPT_SNR``: a per-stage optimal-SNR prior boundary.

Mike, 2026-09-29: "Let's test a gb search 3 rerun with snr limit 3 and
see what happens." This is the OPTIMAL-SNR boundary
(``opt_snr_rej_samp_limit``), not the F-stat peak floor -- ``peak_min_snr``
stays at 6.25.

Until now ``V9_SEARCH_STAGE_PROFILES`` hard-coded ``opt_snr=5.0`` for
gb_search_3 and ``SearchStageProfileStep._apply_profile`` writes it onto
every GB move on stage entry, so ``GB_OPT_SNR_LIMIT_SEARCH`` was
overridden and there was no per-stage way in.
"""

import importlib.util
import inspect
import os
import pathlib
import sys
import types
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "fstat_proposal" / "run_combined_staged.py"


_MOD = None


def _load():
    """Import the launcher module once; importing it costs ~10 s."""
    global _MOD
    if _MOD is None:
        spec = importlib.util.spec_from_file_location(
            "_rcs_under_test", SCRIPT)
        _MOD = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _MOD
        spec.loader.exec_module(_MOD)
    return _MOD


class ProfileResolverTest(unittest.TestCase):

    def setUp(self):
        self.R = _load()

    def _profiles(self, **env):
        with mock.patch.dict(os.environ, env):
            for k in ("GB_SEARCH_2_OPT_SNR", "GB_SEARCH_3_OPT_SNR"):
                if k not in env:
                    os.environ.pop(k, None)
            return {n: p for n, p, _ in self.R.search_stage_profiles()}

    def test_unset_is_the_shipped_table_byte_for_byte(self):
        got = self._profiles()
        want = {n: p for n, p, _ in self.R.V9_SEARCH_STAGE_PROFILES}
        self.assertEqual(got, want)

    def test_the_knob_lowers_gb_search_3_opt_snr(self):
        self.assertEqual(
            self._profiles(GB_SEARCH_3_OPT_SNR="3")["gb_search_3"]["opt_snr"],
            3.0)

    def test_it_does_NOT_touch_the_F_stat_peak_floor(self):
        """Mike's ruling moves the optimal-SNR boundary only; the peak
        floor stays 6.25 or stage 3 would also start accepting weaker
        F-stat peaks, which is a different experiment."""
        p = self._profiles(GB_SEARCH_3_OPT_SNR="3")["gb_search_3"]
        self.assertEqual(p["peak_min_snr"], 6.25)
        self.assertIs(p["phase_maximize"], False)

    def test_it_does_NOT_touch_the_other_stages(self):
        got = self._profiles(GB_SEARCH_3_OPT_SNR="3")
        self.assertEqual(got["gb_search_1"]["opt_snr"], 8.0)
        self.assertEqual(got["gb_search_2"]["opt_snr"], 5.0)

    def test_the_stage_2_symmetry_knob_works_the_same_way(self):
        got = self._profiles(GB_SEARCH_2_OPT_SNR="4.5")
        self.assertEqual(got["gb_search_2"]["opt_snr"], 4.5)
        self.assertEqual(got["gb_search_3"]["opt_snr"], 5.0)

    def test_an_empty_value_is_treated_as_unset(self):
        self.assertEqual(
            self._profiles(GB_SEARCH_3_OPT_SNR="")["gb_search_3"]["opt_snr"],
            5.0)

    def test_zero_or_negative_RAISES_rather_than_disabling_the_boundary(self):
        """``opt_snr_rej_samp_limit = 0`` is a legal 'off' elsewhere, so a
        fat-fingered 0 here would read as deliberate and admit every
        birth."""
        for bad in ("0", "-1"):
            with self.assertRaises(ValueError) as cm:
                self._profiles(GB_SEARCH_3_OPT_SNR=bad)
            self.assertIn("must be > 0", str(cm.exception))

    def test_a_non_number_RAISES(self):
        with self.assertRaises(ValueError):
            self._profiles(GB_SEARCH_3_OPT_SNR="three")

    def test_the_override_is_LOGGED_with_the_default_it_replaced(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            self._profiles(GB_SEARCH_3_OPT_SNR="3")
        out = buf.getvalue()
        self.assertIn("[V9-STAGE gb_search_3]", out)
        self.assertIn("opt_snr=3.0 from GB_SEARCH_3_OPT_SNR", out)
        self.assertIn("peak_min_snr=6.25 is NOT affected", out)


class BothAssembliesUseTheResolverTest(unittest.TestCase):
    """A knob wired into one of the two v9 stage assemblies is the
    "knob resolves, consuming path never runs" shape this file has
    already produced several defects in."""

    def setUp(self):
        self.src = SCRIPT.read_text()

    def test_only_the_RESOLVER_iterates_the_raw_table(self):
        """One raw iteration, inside search_stage_profiles itself. A
        second one is an assembly that bypassed the knob."""
        self.assertEqual(
            self.src.count("in V9_SEARCH_STAGE_PROFILES:"), 1)
        self.assertIn(
            "for name, prof, sampled in V9_SEARCH_STAGE_PROFILES:",
            inspect.getsource(_load().search_stage_profiles))

    def test_both_assemblies_call_the_resolver(self):
        self.assertEqual(self.src.count("_profiles = search_stage_profiles()"),
                         2)

    def test_neither_seed_row_reads_the_raw_table(self):
        self.assertNotIn("dict(V9_SEARCH_STAGE_PROFILES[0][1])", self.src)


class TheValueReachesTheMovesTest(unittest.TestCase):
    """End of the chain: the resolved profile has to land on the GB
    moves' ``opt_snr_rej_samp_limit``, and the truncated-SNR birth
    boundary has to follow it."""

    def _move(self, name):
        return types.SimpleNamespace(
            name=name, branch_name="gb", opt_snr_rej_samp_limit=5.0,
            _snr_lim_table=None, phase_maximize=False)

    def _apply(self, opt_snr, moves):
        from lisatools.globalfit.recipe import SearchStageProfileStep
        step = SearchStageProfileStep.__new__(SearchStageProfileStep)
        step.stage_name = "gb_search_3"
        step.profile = dict(phase_maximize=False, opt_snr=opt_snr,
                            peak_min_snr=6.25)
        step.moves = moves
        SearchStageProfileStep._apply_profile(step, 3)

    def test_every_gb_move_gets_the_resolved_boundary(self):
        moves = [self._move("rj_fstat_search"), self._move("in_model"),
                 self._move("rj_prior_removal")]
        self._apply(3.0, moves)
        for m in moves:
            self.assertEqual(m.opt_snr_rej_samp_limit, 3.0, m.name)

    def test_the_default_still_lands_as_5(self):
        moves = [self._move("rj_fstat_search")]
        self._apply(5.0, moves)
        self.assertEqual(moves[0].opt_snr_rej_samp_limit, 5.0)

    def test_the_truncated_birth_floor_FOLLOWS_the_boundary(self):
        """GB_RJ_SNR_TRUNC_DIST normalizes the birth distance draw
        against this boundary, so a stage that lowers it must lower the
        truncation too or the density is charged against a boundary the
        draw did not use."""
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase
        f = GBSpecialBase._snr_trunc_floor
        self.assertNotEqual(float(f(3.0)), float(f(5.0)))
        self.assertLess(float(f(3.0)), float(f(5.0)))

    def test_a_per_walker_SNR_TABLE_would_shadow_it_and_warns(self):
        """GB_SEARCH_STAGE_PER_WALKER=1 makes the table win over the
        scalar this knob sets; v9 expects it OFF."""
        m = self._move("rj_fstat_search")
        m._snr_lim_table = object()
        with self.assertLogs("lisatools.globalfit.recipe", "WARNING") as cm:
            self._apply(3.0, [m])
        self.assertIn("SHADOWS the scalar", "\n".join(cm.output))


if __name__ == "__main__":
    unittest.main()


class PriorRemovalKnobIsOverridableTest(unittest.TestCase):
    """The second half of the same stage-3 experiment.

    Mike, 2026-09-29: "let's also adjust the gb search 3 stage prior
    removal to be not just removal only." That is
    ``rj_prior_removal``'s ``rj_removal_only``
    (``GB_SEARCH_PRIOR_REMOVAL_ONLY``, default 1 = deaths only; 0 =
    prior births AND deaths). Both launchers hard-coded ``=1``, so an
    exported 0 could not reach the run at all.

    ⚠ NO PER-STAGE FORM, and the reason is not cost. ``rj_removal_only``
    IS read at propose time, so flipping it on stage entry would work --
    but ``_apply_profile`` writes its profile to EVERY GB move in the
    stage tree, and this attribute is not safe to broadcast: ``True`` on
    ``rj_fstat_search`` sets ``apply_inds`` (alive rows only, so NO
    births at all) and ``rj_removal_only`` with ``rj_replace`` raises.
    """

    SCRIPTS = ("submit_gf_6mo_v9_4gpu.sh", "submit_gf_3mo_v9_2gpu.sh")
    VAR = "GB_SEARCH_PRIOR_REMOVAL_ONLY"

    def _line(self, script):
        for ln in (REPO / "scripts" / "fstat_proposal" / script
                   ).read_text().splitlines():
            if ln.startswith(f"export {self.VAR}="):
                return ln
        self.fail(f"{script} does not export {self.VAR}")

    def _resolve(self, script, preset=None):
        """Run the export line in a real shell and read the result."""
        import subprocess
        env = dict(os.environ)
        env.pop(self.VAR, None)
        if preset is not None:
            env[self.VAR] = preset
        out = subprocess.run(
            ["bash", "-c", f'{self._line(script)}; echo "${self.VAR}"'],
            capture_output=True, text=True, env=env, check=True)
        return out.stdout.strip()

    def test_both_launchers_keep_the_default_when_it_is_unset(self):
        for s in self.SCRIPTS:
            self.assertEqual(self._resolve(s), "1", s)

    def test_a_command_line_value_REACHES_the_run(self):
        """The whole point: the hard-coded form swallowed this."""
        for s in self.SCRIPTS:
            self.assertEqual(self._resolve(s, preset="0"), "0", s)

    def test_the_hard_coded_form_is_gone_from_both(self):
        for s in self.SCRIPTS:
            self.assertNotEqual(self._line(s), f"export {self.VAR}=1", s)
            self.assertIn(":-1}", self._line(s), s)

    def test_the_resolved_value_is_echoed_for_the_run_log(self):
        for s in self.SCRIPTS:
            txt = (REPO / "scripts" / "fstat_proposal" / s).read_text()
            self.assertIn("[GB-PRIOR-REMOVAL]", txt, s)

    def test_the_twins_agree(self):
        self.assertEqual(self._line(self.SCRIPTS[0]),
                         self._line(self.SCRIPTS[1]))
