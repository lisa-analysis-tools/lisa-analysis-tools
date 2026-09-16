"""The launch scripts route rank roles through the walker-block layout, not resolve_rank_roles."""

import os
import py_compile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = [
    os.path.join(ROOT, "scripts", "run_global.py"),
    os.path.join(ROOT, "scripts", "fstat_proposal", "run_combined_staged.py"),
]


class DriverScriptsTest(unittest.TestCase):
    def test_scripts_compile(self):
        for path in SCRIPTS:
            py_compile.compile(path, doraise=True)

    def test_no_resolve_rank_roles_and_no_startup_stop_wait(self):
        for path in SCRIPTS:
            with open(path) as fh:
                text = fh.read()
            self.assertNotIn("resolve_rank_roles", text, path)
            self.assertNotIn('recv(source=_main)', text, path)
            self.assertIn("prepare_rank", text, path)


if __name__ == "__main__":
    unittest.main()
