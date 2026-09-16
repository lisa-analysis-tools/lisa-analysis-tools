"""The 6mo campaign submit scripts size the rank count from the GPU count.

Both ``submit_gf_6mo_v8.sh`` (main) and ``submit_gf_6mo_v8_nogb_null.sh``
(null test) self-dispatch via ``exec sbatch ...`` before any heavy work
(environment activation, data preflight, ...) whenever ``SLURM_JOB_ID`` is
unset. That makes the pre-submit dispatch block cheap to exercise directly:
shadow ``sbatch`` with a stub that prints its argv and exits 0, run the
script with ``bash``, and inspect what the (never-actually-submitted) job
would have looked like.

Covers the CAMPAIGN SAFETY invariant: at the ``NGPUS=2`` default the
scripts must pin ``GF_LEGACY_RANK_LAYOUT=1`` (today's single-compute-rank
layout, ``--ntasks=3``) so a plain resubmit stays byte-identical in effect
until the WP7 cluster gates pass; ``NGPUS=4`` must force the walker-block
layout (``GF_LEGACY_RANK_LAYOUT=0``) across 2 nodes of 2 GPUs each, since
the legacy layout cannot span nodes.
"""

import os
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = [
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_6mo_v8.sh"),
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_6mo_v8_nogb_null.sh"),
]

# Env knobs the dispatch block reads; stripped from the inherited environment
# before each scenario applies its own overrides, so a stray value in the
# test runner's shell can never leak into a scenario that expects it unset.
_DISPATCH_ENV_KEYS = (
    "SLURM_JOB_ID",
    "NGPUS",
    "GPUS_PER_RANK",
    "RANKS_PER_GPU",
    "GF_LEGACY_RANK_LAYOUT",
)

_STUB_SBATCH = """#!/usr/bin/env bash
for a in "$@"; do printf '%s\\n' "$a"; done
exit 0
"""


class SubmitScriptsSyntaxTest(unittest.TestCase):
    def test_bash_syntax(self):
        for path in SCRIPTS:
            result = subprocess.run(
                ["bash", "-n", path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                result.returncode, 0, f"{path}: bash -n failed: {result.stderr}"
            )


class SubmitScriptsInJobBlockTest(unittest.TestCase):
    """The IN-JOB rank-layout block (after the `exec sbatch` self-dispatch).

    The stub-sbatch scenarios below can only observe the PRE-submit dispatch
    block -- everything after `exec sbatch` never runs without a real job --
    so these invariants are checked as text. They are the ones that only bite
    a launch that bypassed the dispatch block entirely (a manual
    `sbatch --nodes=2 --ntasks=5 <script>`), which is exactly the path no
    other test covers.
    """

    def _text(self, path):
        with open(path) as fh:
            return fh.read()

    def test_multi_node_forces_walker_block_layout_in_job(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn(
                    'if [ "${SLURM_NNODES:-1}" -gt 1 ] && '
                    '[ "${GF_LEGACY_RANK_LAYOUT}" = "1" ]; then',
                    text,
                )
                self.assertIn("FORCING GF_LEGACY_RANK_LAYOUT=0", text)

    def test_in_job_exports_all_three_layout_knobs(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn("export GPUS_PER_RANK RANKS_PER_GPU", text)
                self.assertIn("export GF_LEGACY_RANK_LAYOUT", text)

    def test_nwalkers_modulo_is_guarded_against_zero_compute_ranks(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn('[ "${N_COMPUTE_EFF}" -gt 0 ]', text)
                # the division itself must not be reachable unguarded
                self.assertNotIn(
                    'if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && '
                    "[ $(( NWALKERS % N_COMPUTE_EFF )) -ne 0 ]; then",
                    text,
                )

    def test_srun_launch_line_defaults_ntasks(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                self.assertIn(
                    'srun --ntasks="${SLURM_NTASKS:-3}" --distribution=cyclic',
                    self._text(path),
                )


class SubmitScriptsDispatchTest(unittest.TestCase):
    """Exercise the pre-submit `exec sbatch ...` dispatch block via a stub."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="submit_scripts_stub_")
        cls.stub_dir = cls._tmpdir.name
        stub_path = os.path.join(cls.stub_dir, "sbatch")
        with open(stub_path, "w") as fh:
            fh.write(_STUB_SBATCH)
        os.chmod(stub_path, 0o755)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _run_dispatch(self, script, overrides):
        env = dict(os.environ)
        for key in _DISPATCH_ENV_KEYS:
            env.pop(key, None)
        env["PATH"] = self.stub_dir + os.pathsep + env.get("PATH", "")
        env.update(overrides)
        result = subprocess.run(
            ["bash", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"{script} {overrides}: exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result.stdout.splitlines()

    def _assert_export_contains(self, lines, needle):
        export_lines = [l for l in lines if l.startswith("--export=")]
        self.assertTrue(
            export_lines, f"no --export= argv line found; stdout lines: {lines}"
        )
        self.assertTrue(
            any(needle in l for l in export_lines),
            f"{needle!r} not found in --export= line(s): {export_lines}",
        )

    def test_ngpus_2_default_is_legacy(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(script, {"NGPUS": "2"})
                self.assertIn("--ntasks=3", lines)
                self.assertIn("--nodes=1", lines)
                self.assertIn("--partition=gpu-80-spot", lines)
                self.assertIn("--gres=gpu:2", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=1")

    def test_ngpus_4_forces_walker_block_two_nodes(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(script, {"NGPUS": "4"})
                self.assertIn("--ntasks=5", lines)
                self.assertIn("--nodes=2", lines)
                self.assertIn("--gres=gpu:2", lines)
                self.assertIn("--partition=gpu-80-spot", lines)
                self.assertIn("--distribution=cyclic", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")

    def test_ngpus_2_explicit_walker_block(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(
                    script, {"NGPUS": "2", "GF_LEGACY_RANK_LAYOUT": "0"}
                )
                self.assertIn("--ntasks=3", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")

    def test_ngpus_2_gpus_per_rank_2_walker_block(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(
                    script,
                    {
                        "NGPUS": "2",
                        "GPUS_PER_RANK": "2",
                        "GF_LEGACY_RANK_LAYOUT": "0",
                    },
                )
                self.assertIn("--ntasks=2", lines)


if __name__ == "__main__":
    unittest.main()
