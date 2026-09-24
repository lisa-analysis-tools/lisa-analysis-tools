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
        # 2048, not the 0 this asserted until 2026-09-24. The campaign script
        # reads `export GB_INMODEL_SETUP_BATCH=${GB_INMODEL_SETUP_BATCH:-2048}`
        # -- the batched in-model setup that, with the 1 GiB fold cap, is what
        # fixed the 6-month sig-het OOM. This assertion could not have caught
        # the drift while it was written, because the parser was handing back
        # the LITERAL string "${GB_INMODEL_SETUP_BATCH:-2048}": the test was
        # already red on dev, just for a different reason than it looked.
        self.assertEqual(self.pins["GB_INMODEL_SETUP_BATCH"], "2048")

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

    def test_no_pin_is_an_unexpanded_shell_expression(self):
        """★ A pin carrying a literal ``$`` is a LANDMINE, not a value.

        The campaign script writes
        ``export GB_SIGHET_FOLD_MAX_BYTES=${GB_SIGHET_FOLD_MAX_BYTES:-1073741824}``.
        Copied through verbatim, that string reached GBGPU's module-level
        ``int(os.environ.get("GB_SIGHET_FOLD_MAX_BYTES", 1 << 30))`` and
        killed the import -- on the cluster, mid-gate, with a traceback a
        long way from its cause (2026-09-24). Every pin must be something
        its consumer can actually parse.
        """
        for name, value in self.pins.items():
            self.assertNotIn("$", value, f"{name}={value!r} is unexpanded")

    def test_a_parameter_expansion_resolves_to_its_default(self):
        pins = gate_run.campaign_sighet_pins(environ={})
        self.assertEqual(pins["GB_SIGHET_FOLD_MAX_BYTES"], "1073741824")
        self.assertEqual(int(pins["GB_SIGHET_FOLD_MAX_BYTES"]), 1 << 30)

    def test_a_parameter_expansion_prefers_the_shell(self):
        # ``${NAME:-default}`` means what apply_pins already means: the shell
        # wins. Resolving it here must not invent a second rule.
        pins = gate_run.campaign_sighet_pins(
            environ={"GB_SIGHET_FOLD_MAX_BYTES": "4096"})
        self.assertEqual(pins["GB_SIGHET_FOLD_MAX_BYTES"], "4096")

    def test_an_unmodelled_shell_expression_is_dropped_not_pinned(self):
        import tempfile, textwrap
        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
            fh.write(textwrap.dedent("""\
                export SIGHET_N_CP=256
                export SIGHET_NT_LAYER=$(compute_it)
                export GB_SIGHET_REFRESH_EVERY=${A}${B}
            """))
            path = fh.name
        pins = gate_run.campaign_sighet_pins(path, environ={})
        # the plain one survives; the two shell-y ones are left to the stock
        # default, which at least parses
        self.assertEqual(pins.get("SIGHET_N_CP"), "256")
        self.assertNotIn("SIGHET_NT_LAYER", pins)
        self.assertNotIn("GB_SIGHET_REFRESH_EVERY", pins)

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
