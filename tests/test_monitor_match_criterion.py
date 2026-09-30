"""The monitor page's headline match numbers are the phase-maximised overlap
(user request 2026-09-30), on by default.

The generator is a top-level script executed by ``build_monitor`` against a
run directory, so these are source-level pins: each names the exact line
whose removal would silently put the page back on the 2-df proxy or gate
the panels off again.
"""

import os
import re
import unittest

_GEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src", "lisatools", "globalfit", "monitor", "_generator.py")


class MatchCriterionSourceTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(_GEN, encoding="utf-8") as fh:
            cls.src = fh.read()

    def test_match_panels_are_on_by_default(self):
        self.assertIn(
            'SHOW_MATCH_STATS = os.environ.get("GF_MONITOR_MATCH_STATS", "1") != "0"',
            self.src)

    def test_headline_numbers_come_from_the_overlap_match(self):
        # the overlap-refined counts overwrite the proxy ones in SCI ...
        self.assertRegex(self.src, r"completeness=float\(FOUND_MM\.sum\(\)\) / NDET")
        self.assertRegex(self.src, r"purity=float\(MATCHED_MM\.sum\(\)\) / max\(REC9\.shape\[0\], 1\)")
        self.assertIn("match_is_overlap=True", self.src)
        # ... the proxy values survive under their own names for the caption
        self.assertIn("completeness_proxy=MI.size / NDET", self.src)

    def test_completeness_vs_snr_reads_the_overlap_found_set(self):
        m = re.search(r"np\.savez\(f\"gf_arm_\{ARM_TAG\}\.npz\".*?\)", self.src, re.S)
        self.assertIsNotNone(m)
        self.assertIn("ti=TI_OV", m.group(0))

    def test_kpi_cards_name_their_criterion(self):
        self.assertIn('completeness ({_crit_kpi})', self.src)
        self.assertIn('"phase-max overlap" if SCI and SCI.get("match_is_overlap")', self.src)


if __name__ == "__main__":
    unittest.main()
