"""Every recipe step kind answers ``stopping_function`` (2026-10-04).

THE BUG. 3ac014a6 (2026-10-02) inserted the module-level helpers
``apply_rj_flip_fraction`` / ``apply_inmodel_repeats`` between
``PERecipeStep``'s methods and its ``stopping_function``. Python then read
that ``def`` as a nested function inside ``apply_inmodel_repeats``, after its
``return`` -- dead code -- and ``PERecipeStep`` inherited the abstract
``RecipeStep.stopping_function``. The 6mo run raised NotImplementedError on
its first full_pe iteration, right after replica_pe converged.
"""

from __future__ import annotations

import ast
import inspect
import unittest

from lisatools.globalfit import recipe as R


class PEStoppingTest(unittest.TestCase):
    def test_full_pe_never_stops_on_its_own(self):
        self.assertIn("stopping_function", vars(R.PERecipeStep))
        step = R.PERecipeStep.__new__(R.PERecipeStep)
        self.assertFalse(step.stopping_function(5, None, None))

    def test_no_step_kind_falls_back_to_the_abstract_method(self):
        base = R.RecipeStep.stopping_function
        for kind, cls in R._STEP_CLASSES.items():
            with self.subTest(kind=kind):
                self.assertIsNot(cls.stopping_function, base,
                                 f"{cls.__name__} inherits the abstract stop")
                self.assertIsNot(cls.setup_run, R.RecipeStep.setup_run,
                                 f"{cls.__name__} inherits the abstract setup")

    def test_no_module_function_swallows_a_step_method(self):
        """The class of mistake, not just this instance: a step method
        indented under a module-level function is dead code."""
        tree = ast.parse(inspect.getsource(R))
        bad = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                for sub in ast.walk(node):
                    if (sub is not node and isinstance(sub, ast.FunctionDef)
                            and sub.name in ("stopping_function", "setup_run")):
                        bad.append(f"{node.name} -> {sub.name}")
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
