"""The one-walker gate driver mirrors the campaign script's sig-het settings."""

import contextlib
import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts", "diagnostics"))

import gate_run  # noqa: E402


class CampaignPinsTest(unittest.TestCase):
    def setUp(self):
        self.pins = gate_run.campaign_sighet_pins()

    def test_reads_the_production_sighet_pins(self):
        # the values the gates were missing (2026-09-17): finer stride, the
        # h_h-corruption n_cp fix, a refreshed reference, the drift check
        self.assertEqual(self.pins["SIGHET_NT_LAYER"], "120")
        self.assertEqual(self.pins["SIGHET_N_CP"], "256")
        self.assertEqual(self.pins["GB_SIGHET_REFRESH_EVERY"], "25")
        self.assertEqual(self.pins["GB_SIGHET_TRUST_PHASE_C"], "49")
        self.assertEqual(self.pins["GB_SIGHET_DRIFT_CHECK"], "1")
        self.assertEqual(self.pins["SIGHET_INFOMAT"], "1")
        self.assertEqual(self.pins["GB_SIGHET_INMODEL_WINDOWED"], "1")
        self.assertEqual(self.pins["GB_ORTHO_LL_CHECK"], "1")
        self.assertEqual(self.pins["GB_INMODEL_SETUP_BATCH"], "0")

    def test_diagnostic_sweeps_are_not_mirrored(self):
        for name in self.pins:
            self.assertNotRegex(name, r"DISSECT|SWEEP|TIER_SCAN", name)
        for name in self.pins:
            self.assertTrue(
                name.startswith(("SIGHET_", "GB_SIGHET_")) or name in gate_run._EXTRA, name
            )

    def test_values_are_unquoted_and_comment_free(self):
        for name, value in self.pins.items():
            self.assertNotIn("#", value, name)
            self.assertNotIn('"', value, name)
            self.assertNotIn(" ", value, name)

    def test_shell_value_wins_over_the_pin(self):
        env = {"SIGHET_NT_LAYER": "216"}
        applied, kept = gate_run.apply_pins(self.pins, environ=env)
        self.assertEqual(kept, {"SIGHET_NT_LAYER": "216"})
        self.assertEqual(env["SIGHET_NT_LAYER"], "216")
        self.assertEqual(env["SIGHET_N_CP"], "256")
        self.assertNotIn("SIGHET_NT_LAYER", applied)
        self.assertEqual(applied["SIGHET_N_CP"], "256")

    def test_print_only_reports_without_building(self):
        saved = dict(os.environ)
        buf = io.StringIO()
        try:
            for name in self.pins:
                os.environ.pop(name, None)
            with contextlib.redirect_stdout(buf):
                rc = gate_run.main(["--print-only"])
        finally:
            os.environ.clear()
            os.environ.update(saved)
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("[GATE] sig-het pins from scripts/fstat_proposal/submit_gf_6mo_v8.sh", out)
        self.assertIn("SIGHET_NT_LAYER=120", out)
        self.assertIn("shell overrides kept (none)", out)


if __name__ == "__main__":
    unittest.main()
