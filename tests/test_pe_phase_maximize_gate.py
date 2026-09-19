"""``rj_fstat_pe`` must not phase-maximize because of a knob set elsewhere.

Every move in the ``full_pe`` stack is hardcoded ``phase_maximize=False``
(``rj_warm_pe``, ``rj_prior_pe``, ``rj_replace_pe``, ``vgb_pe``) except
``rj_fstat_pe``, which used to read

    phase_maximize = GB_RJ_PHASE_MAXIMIZE if (gb_mode_search and not pe_strict)

so exporting ``GB_MODE=search`` -- a knob about which STAGE runs the search --
silently re-armed likelihood maximization inside PE. The standing policy is
that PE never maximizes, so the gate now needs ``GB_PE_PHASE_MAXIMIZE=1`` by
name.

The other half of the defect is why the obvious workaround was wrong:
``GB_PE_MOVES_STRICT=1`` closes the gate, but it ALSO nulls the PE leaf caps
and swaps the RJ flip fraction, so using it to disarm phase-max would
reconfigure the cap machinery of a production PE stage as a side effect.
That coupling is asserted here too, so nobody re-derives it as a fix.
"""

import os
import unittest

from lisatools.globalfit.recipe import pe_phase_maximize_on


class _Env:
    """Set/restore env vars around a block."""

    def __init__(self, **kw):
        self.kw = kw
        self.old = {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


_ALL_GATES = [(m, s) for m in (False, True) for s in (False, True)]


class PePhaseMaximizeGateTest(unittest.TestCase):
    def test_off_by_default_in_every_mode_combination(self):
        """The knob unset means PE never maximizes, whatever GB_MODE says."""
        with _Env(GB_PE_PHASE_MAXIMIZE=None):
            for mode_search, strict in _ALL_GATES:
                self.assertFalse(
                    pe_phase_maximize_on(mode_search, strict),
                    f"gb_mode_search={mode_search} pe_strict={strict}")

    def test_gb_mode_search_alone_does_not_arm_it(self):
        """The regression: GB_MODE=search used to be enough on its own."""
        with _Env(GB_PE_PHASE_MAXIMIZE=None, GB_RJ_PHASE_MAXIMIZE="1"):
            self.assertFalse(pe_phase_maximize_on(True, False))

    def test_explicit_opt_in_under_a_search_campaign(self):
        with _Env(GB_PE_PHASE_MAXIMIZE="1"):
            self.assertTrue(pe_phase_maximize_on(True, False))

    def test_opt_in_still_requires_the_search_campaign(self):
        """The knob re-arms the OLD behaviour; it does not invent a new one."""
        with _Env(GB_PE_PHASE_MAXIMIZE="1"):
            self.assertFalse(pe_phase_maximize_on(False, False))
            self.assertFalse(pe_phase_maximize_on(False, True))

    def test_strict_pe_still_wins_over_the_opt_in(self):
        with _Env(GB_PE_PHASE_MAXIMIZE="1"):
            self.assertFalse(pe_phase_maximize_on(True, True))

    def test_only_the_exact_string_one_arms_it(self):
        """'0', 'true', 'yes', '' must not arm a maximization knob."""
        for val in ("0", "", "true", "TRUE", "yes", "2", " 1"):
            with _Env(GB_PE_PHASE_MAXIMIZE=val):
                self.assertFalse(
                    pe_phase_maximize_on(True, False), repr(val))


class StrictPeIsNotAnOffSwitchTest(unittest.TestCase):
    """GB_PE_MOVES_STRICT carries cap side effects, so it is not the fix."""

    def test_strict_pe_also_nulls_the_pe_leaf_caps(self):
        import inspect

        from lisatools.globalfit import recipe

        src = inspect.getsource(recipe)
        self.assertIn('_pe_strict = os.environ.get("GB_PE_MOVES_STRICT"', src)
        # The same flag gates _pe_cap_off; if that coupling is ever removed
        # this assertion should be deleted along with the docstring claim in
        # pe_phase_maximize_on that depends on it.
        self.assertIn(
            '{"leaf_cap_start": None, "leaf_cap_update": False} if _pe_strict',
            src,
        )

    def test_the_docstring_names_the_replacement_knob(self):
        self.assertIn("GB_PE_PHASE_MAXIMIZE", pe_phase_maximize_on.__doc__)


if __name__ == "__main__":
    unittest.main()
