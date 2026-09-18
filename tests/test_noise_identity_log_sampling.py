"""The noise-model identity must record the psd/galfor SAMPLING BASIS.

``psd.log_sampling`` / ``galfor.log_sampling`` change what the stored
numbers MEAN without changing a single array shape: a galfor amplitude of
2.5e-44 read back under log sampling is ``10**2.5e-44`` ~ 1. That is the
textbook "same shapes, different likelihood" resume the identity record
exists to refuse, and until 2026-09-18 it was the one noise knob the
record did not carry.

The second half of this file pins the BACKWARD-COMPATIBILITY rule that
had to come with it. The resume check iterates the configured identity and
compares each key against the store; a key added later is absent from
every existing store, so a naive ``stored.get(key) == value`` makes every
old store mismatch at once -- adding these two keys would have refused
both live production runs on their next resume. Absent-and-default is
therefore skipped, while absent-and-NON-default still raises, because that
is precisely the silent reinterpretation being guarded.
"""

import unittest

import numpy as np


def _mismatches(configured, stored):
    """The resume comparison from ``run.py::_open_run_backend``, verbatim."""
    mismatched = {}
    for key, value in configured.items():
        stored_value = stored.get(key)
        if key not in stored:
            if value in (False, "", 0):
                continue
            mismatched[key] = ("<not recorded>", value)
            continue
        if isinstance(value, float):
            same = np.isclose(float(stored_value), value, rtol=0.0, atol=1e-6)
        else:
            same = stored_value == value
        if not same:
            mismatched[key] = (stored_value, value)
    return mismatched


BASE = {
    "instrument_component": "UnequalArmInstrumentNoise",
    "unequal_arm": True,
    "wdm_psd_method": "layer_calibrated",
    "data_t0": 97729939.827664,
}


class BasisIsRecordedTest(unittest.TestCase):
    def test_a_basis_flip_is_refused(self):
        stored = {**BASE, "galfor_log_sampling": False, "psd_log_sampling": False}
        conf = {**BASE, "galfor_log_sampling": True, "psd_log_sampling": False}
        bad = _mismatches(conf, stored)
        self.assertIn("galfor_log_sampling", bad)
        self.assertEqual(bad["galfor_log_sampling"], (False, True))

    def test_a_flip_BACK_to_linear_is_also_refused(self):
        """Both directions reinterpret the stored numbers."""
        stored = {**BASE, "galfor_log_sampling": True}
        conf = {**BASE, "galfor_log_sampling": False}
        self.assertIn("galfor_log_sampling", _mismatches(conf, stored))

    def test_matching_bases_resume_cleanly(self):
        for flag in (False, True):
            stored = {**BASE, "galfor_log_sampling": flag, "psd_log_sampling": flag}
            self.assertEqual(_mismatches(dict(stored), stored), {}, flag)

    def test_psd_and_galfor_are_tracked_independently(self):
        stored = {**BASE, "galfor_log_sampling": False, "psd_log_sampling": False}
        conf = {**BASE, "galfor_log_sampling": False, "psd_log_sampling": True}
        bad = _mismatches(conf, stored)
        self.assertEqual(list(bad), ["psd_log_sampling"])


class BackwardCompatibilityTest(unittest.TestCase):
    """A store predating a key must still resume -- unless the value is live."""

    def test_an_old_store_resumes_when_the_new_flag_is_default(self):
        """The case that would have killed both production runs."""
        old = dict(BASE)                       # no *_log_sampling keys at all
        conf = {**BASE, "galfor_log_sampling": False, "psd_log_sampling": False}
        self.assertEqual(_mismatches(conf, old), {})

    def test_an_old_store_is_REFUSED_when_the_new_flag_is_on(self):
        """The dangerous direction: linear store, log sampling requested."""
        old = dict(BASE)
        conf = {**BASE, "galfor_log_sampling": True, "psd_log_sampling": False}
        bad = _mismatches(conf, old)
        self.assertEqual(bad["galfor_log_sampling"], ("<not recorded>", True))
        self.assertNotIn("psd_log_sampling", bad)

    def test_the_skip_does_not_mask_a_real_difference_on_a_present_key(self):
        old = {**BASE, "wdm_psd_method": "fold"}
        conf = {**BASE, "wdm_psd_method": "layer_calibrated"}
        self.assertIn("wdm_psd_method", _mismatches(conf, old))

    def test_float_keys_still_compare_with_tolerance(self):
        stored = {**BASE, "data_t0": 97729939.8276641}
        self.assertEqual(_mismatches({**BASE}, stored), {})
        self.assertIn("data_t0", _mismatches({**BASE, "data_t0": 1.0}, stored))

    def test_a_missing_nondefault_STRING_key_also_raises(self):
        """The rule is about default-ness, not about booleans."""
        old = dict(BASE)
        conf = {**BASE, "ltts_digest": "f1f3f00ea5d9cf13"}
        self.assertIn("ltts_digest", _mismatches(conf, old))
        self.assertEqual(_mismatches({**BASE, "ltts_digest": ""}, old), {})


if __name__ == "__main__":
    unittest.main()
