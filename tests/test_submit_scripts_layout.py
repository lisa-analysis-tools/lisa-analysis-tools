"""The 6mo campaign submit scripts size the rank count from the GPU count.

Both ``submit_gf_6mo_v8.sh`` (main) and ``submit_gf_6mo_v8_nogb_null.sh``
(null test) self-dispatch via ``exec sbatch ...`` before any heavy work
(environment activation, data preflight, ...) whenever ``SLURM_JOB_ID`` is
unset. That makes the pre-submit dispatch block cheap to exercise directly:
shadow ``sbatch`` with a stub that prints its argv and exits 0, run the
script with ``bash``, and inspect what the (never-actually-submitted) job
would have looked like.

Covers the layout invariants (user ruling 2026-09-16, after the WP7
transport gates passed on the cluster): at the ``NGPUS=2`` default the
scripts pin the walker-block layout (``GF_LEGACY_RANK_LAYOUT=0``: head +
1 compute + saver, still ``--ntasks=3``); ``GF_LEGACY_RANK_LAYOUT=1`` remains
the rollback knob and must still launch today's single-compute-rank shape;
``NGPUS=4`` must force the walker-block layout across 2 nodes of 2 GPUs each,
since the legacy layout cannot span nodes.
"""

import os
import re
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = [
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_6mo_v8.sh"),
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_6mo_v8_nogb_null.sh"),
    # the 3-month twin of the 6mo campaign script: same layout machinery,
    # Tobs-derived settings reverted, source branches and warm start off
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_3mo_v8_4gpu.sh"),
]

THREE_MO = SCRIPTS[2]
SIX_MO = SCRIPTS[0]

# Env knobs the dispatch block reads; stripped from the inherited environment
# before each scenario applies its own overrides, so a stray value in the
# test runner's shell can never leak into a scenario that expects it unset.
_DISPATCH_ENV_KEYS = (
    "SLURM_JOB_ID",
    "NGPUS",
    "NODES",
    "GPUS_PER_RANK",
    "RANKS_PER_GPU",
    "GF_LEGACY_RANK_LAYOUT",
)

_STUB_SBATCH = """#!/usr/bin/env bash
for a in "$@"; do printf '%s\\n' "$a"; done
exit 0
"""


def _exports(path):
    """``{KNOB: value}`` as bash resolves the script's export lines in order.

    Resolving them rather than regexing the file is the point: the values
    are ``${K:-default}`` forms and several are overridden further down, so
    a grep answers what the file SAYS and this answers what the run GETS.
    """
    src = open(path).read()
    lines = [l for l in src.split("\n") if re.match(r"^export [A-Z0-9_]+=", l)]
    env = dict(os.environ)
    for k in list(env):
        if k.isupper():
            env.pop(k, None)
    out = subprocess.run(["bash", "-c", "\n".join(lines) + "\nenv | sort\n"],
                         capture_output=True, text=True, env=env).stdout
    return dict(l.split("=", 1) for l in out.split("\n") if "=" in l)


class ThreeMonthTwinTest(unittest.TestCase):
    """``submit_gf_3mo_v8_4gpu.sh`` is the 6mo script at 3 months.

    Written 2026-09-18 to the spec "use all the updates and base it on the
    6mo run, but make sure the high level 3mo things are there (Tobs, no
    emris/sobhbs/mbhbs)" plus "(no warmstart)".

    The failure this guards is a silent one. The two files are 97% the
    same text, so the obvious way to carry a 6mo fix across is to copy the
    block -- and the blocks that must NOT be copied are exactly the ones
    that look like ordinary knobs (``TOBS_TARGET``, ``GB_N_SUBBANDS``). A
    3-month run that quietly analysed 6 months of data, or armed the
    source branches, would produce plausible output and waste the
    campaign.
    """

    def setUp(self):
        self.three = _exports(THREE_MO)
        self.six = _exports(SIX_MO)

    def test_tobs_and_its_derived_settings_are_the_3mo_values(self):
        self.assertEqual(self.three["TOBS_TARGET"], "7776000")
        self.assertEqual(self.three["GB_NLEAVES_MAX"], "10000")
        self.assertEqual(self.three["GB_N_SUBBANDS"], "32768")
        self.assertEqual(self.three["GB_RJ_INMODEL_CHUNK"], "65536")
        self.assertEqual(self.three["COARSE_Q"], "1")
        self.assertEqual(self.three["COARSE_GPU_MODE"], "off")
        self.assertEqual(self.three["BASE_FILE_NAME"], "gf_prod_3mo")
        # 6mo-only; the 3mo arm takes the defaults
        self.assertNotIn("SIGHET_NT_LAYER", self.three)
        self.assertNotIn("EDGE_CROP_WAVELETS", self.three)

    def test_no_source_branches(self):
        for knob in ("MBHB_IDS", "EMRI_IDS", "SOBHB_IDS"):
            self.assertEqual(
                self.three.get(knob, ""), "",
                f"{knob} must be EMPTY: _source_ids_from_env arms a branch "
                f"on any non-empty list, and an armed branch is sampled")
        self.assertEqual(self.three["SOURCE_TYPES"], "NOISE,GB,VGB")

    def test_the_source_search_skip_is_not_set(self):
        """It is an ERROR, not a no-op, with no armed sources.

        ``run_combined_staged`` raises "STAGE_SKIP_SOURCE_SEARCH=1 but no
        source branches are armed -- there is no source_search stage to
        skip." The 6mo script sets it; carrying it across would kill this
        run at startup, which is how it was found.
        """
        self.assertNotIn("STAGE_SKIP_SOURCE_SEARCH", self.three)

    def test_no_warm_start(self):
        self.assertEqual(
            self.three.get("GB_WARM_START_COMPONENTS", ""), "",
            "a 3-month run is the SOURCE of warm-start components, not a "
            "consumer; empty is the documented off switch")

    def test_every_other_knob_matches_the_6mo_script(self):
        """The whole point of the merge: only the listed knobs differ."""
        allowed = {
            "TOBS_TARGET", "GB_NLEAVES_MAX", "GB_N_SUBBANDS",
            "GB_RJ_INMODEL_CHUNK", "SIGHET_NT_LAYER", "EDGE_CROP_WAVELETS",
            "COARSE_Q", "COARSE_GPU_MODE", "BASE_FILE_NAME",
            "SOURCE_TYPES", "MBHB_IDS", "EMRI_IDS", "SOBHB_IDS",
            "GB_WARM_START_COMPONENTS", "GB_WARM_START_SOURCE_STORE",
            "STAGE_SKIP_SOURCE_SEARCH",
        }
        keys = (set(self.three) | set(self.six)) - {"_", "SHLVL", "PWD"}
        diff = {k for k in keys
                if self.three.get(k, "<unset>") != self.six.get(k, "<unset>")}
        unexpected = diff - allowed
        self.assertEqual(
            unexpected, set(),
            f"these knobs drifted apart and are not on the 3mo reversion "
            f"list: {sorted(unexpected)}")

    def test_the_multirank_and_correctness_updates_came_across(self):
        """The reason to derive from the 6mo file rather than the old 3mo one."""
        for knob, value in (("VGB_CHIRP_MASS_BASIS", "1"),
                            ("VGB_SIGHET_INMODEL", "1"),
                            ("VGB_INMODEL_PROPOSAL", "observable"),
                            ("GB_INMODEL_OBSERVABLE_EIGEN", "full"),
                            ("GB_LEAF_CAP_MIN_ITERS", "3"),
                            ("NWALKERS", "4"),
                            ("SIGHET_TUKEY_ALPHA", "0.01"),
                            ("SOBBH_EIGEN_SCOPE", "walker_max")):
            self.assertEqual(self.three.get(knob), value, knob)


class WarmStartPathSeparatorTest(unittest.TestCase):
    """The warm-start npz must land INSIDE the run store (2026-09-18).

    The default was written ``${STORE_DIR}warmstart/...`` with no
    separator, which relies on ``STORE_DIR`` ending in a slash. The
    built-in default does; a command-line override
    (``STORE_DIR=/.../gf_prod_6mo_v8_4gpu ./submit...``) does not, so the
    npz went to a SIBLING directory ``gf_prod_6mo_v8_4gpuwarmstart/`` --
    outside the snapshot zips, and not rebuilt when the store is reset.
    """

    def test_store_dir_and_warmstart_are_separated(self):
        for path in SCRIPTS:
            with open(path) as fh:
                src = fh.read()
            for m in re.finditer(r"\$\{STORE_DIR\}(?!/)(\S{0,24})", src):
                self.assertNotIn(
                    "warmstart", m.group(1),
                    f"{os.path.basename(path)}: "
                    f"${{STORE_DIR}}{m.group(1)} has no path separator, so "
                    f"an override without a trailing slash puts the warm "
                    f"start outside the store")


class SobbhKnobsSingleExportTest(unittest.TestCase):
    """One effective export per SOBBH knob (2026-09-17).

    ``SOBBH_EIGEN_SCOPE=walker_max`` (user ruling 2026-09-16) was exported
    once and then silently overridden by a second, older
    ``export SOBBH_EIGEN_SCOPE=per_walker`` further down the same file, so
    the ruling never reached a run. ``SOBBH_NTEMPS`` must be overridable
    from the environment (a store born at 8 rungs is resumed with
    ``SOBBH_NTEMPS=8``; the stored count wins either way, but the knob
    should not lie in run_settings.log).
    """

    def _exports(self, path, name):
        with open(path) as fh:
            return [
                line.strip() for line in fh
                if line.lstrip().startswith(f"export {name}=")
            ]

    def test_sobbh_eigen_scope_exported_once(self):
        for path in SCRIPTS:
            lines = self._exports(path, "SOBBH_EIGEN_SCOPE")
            self.assertLessEqual(
                len(lines), 1,
                f"{path}: SOBBH_EIGEN_SCOPE exported {len(lines)} times: {lines}",
            )

    def test_campaign_sobbh_eigen_scope_is_walker_max(self):
        # The ruling applies to the campaign script; the null-test script
        # keeps its own (per-walker) setting and is only held to one export.
        lines = self._exports(SCRIPTS[0], "SOBBH_EIGEN_SCOPE")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("walker_max", lines[0])
        self.assertNotIn("per_walker", lines[0])

    def test_sobbh_repeats_is_env_overridable_and_defaults_to_10(self):
        # User ruling 2026-09-18. Repeats are the ONLY knob that moves the
        # SOBBH cost: [SOBBH_LL_TIMING] measured a flat 1.73 s per scoring
        # call regardless of rows, and calls come from repeats, not walkers
        # or rungs. 20 -> 10 halves the dominant per-iteration cost.
        lines = self._exports(SCRIPTS[0], "SOBBH_NUM_PROP_REPEATS")
        self.assertEqual(len(lines), 1, lines)
        self.assertEqual(
            lines[0], "export SOBBH_NUM_PROP_REPEATS=${SOBBH_NUM_PROP_REPEATS:-10}")

    def test_sobbh_ntemps_is_env_overridable_and_defaults_to_8(self):
        # User ruling 2026-09-17: 8 in the scripts (the 4-GPU store is an
        # 8-rung store; resumed 12-rung stores keep 12 via the store-wins
        # rule). Both campaign scripts.
        for path in SCRIPTS:
            lines = self._exports(path, "SOBBH_NTEMPS")
            self.assertEqual(len(lines), 1, f"{path}: {lines}")
            self.assertEqual(
                lines[0], "export SOBBH_NTEMPS=${SOBBH_NTEMPS:-8}", path)


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

    def test_one_walker_is_exempt_from_the_divisibility_fix(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn('[ "${NWALKERS}" -eq 1 ]', text)
                self.assertIn("one-walker replica mode", text)
                # The sampler-shape export must honour the submitting shell's
                # NWALKERS (carried by --export=ALL), or the branch below it is
                # unreachable: a hard `export NWALKERS=<n>` above the `-eq 1`
                # test would silently run n walkers for `NWALKERS=1 ./submit`.
                # The DEFAULT VALUE is deliberately not pinned here -- it is a
                # campaign choice that moves (10 at the 2026-09-11 rebase, 4
                # for the walker-block store on 2026-09-18); what this test
                # protects is the overridable FORM and its position.
                m = re.search(r"^export NWALKERS=\$\{NWALKERS:-\d+\}",
                              text, re.M)
                self.assertIsNotNone(
                    m, "NWALKERS must be exported as ${NWALKERS:-<default>}")
                self.assertNotRegex(text, r"\nexport NWALKERS=\d+\s")
                self.assertLess(
                    m.start(), text.index('[ "${NWALKERS}" -eq 1 ]'))
                # the rounding branch survives for NWALKERS > 1
                self.assertIn(
                    "(NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF", text
                )
                # the exemption is tested BEFORE the rounding branch
                self.assertLess(
                    text.index('[ "${NWALKERS}" -eq 1 ]'),
                    text.index(
                        "(NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF"
                    ),
                )

    def test_multinode_launch_line_is_hydra_round_robin(self):
        # WP7 Step 0 (2026-09-16): on this cluster `srun --mpi=pmix` bootstraps
        # Intel MPI but its OFI address exchange fails, and a bare `srun` gives
        # three size-1 worlds; the launcher that works is hydra with the SLURM
        # bootstrap, `-ppn 1` (round-robin over hosts = cyclic placement) and
        # the tcp fabric provider. Pin all of it so a refactor cannot drift
        # back to `srun`.
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn(
                    'mpiexec -n "${SLURM_NTASKS:-3}" -ppn 1 python', text,
                )
                self.assertIn(
                    "export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi "
                    "FI_PROVIDER=tcp",
                    text,
                )
                self.assertNotIn('srun --ntasks="${SLURM_NTASKS', text)


class SubmitScriptsNwalkersBlockTest(unittest.TestCase):
    """Execute the in-job NWALKERS divisibility/exemption block with bash.

    The text checks in ``SubmitScriptsInJobBlockTest`` confirm the exemption
    line and message exist and come first; this class actually runs the
    extracted `if ... fi` block under bash to confirm the one-walker branch
    leaves NWALKERS untouched and the rounding branch still fires (and still
    rounds correctly) for every other NWALKERS value.
    """

    _START_MARKER = (
        'if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && '
        '[ "${N_COMPUTE_EFF}" -gt 1 ] && [ "${NWALKERS}" -eq 1 ]; then'
    )

    def _text(self, path):
        with open(path) as fh:
            return fh.read()

    def _extract_block(self, path):
        text = self._text(path)
        start = text.index(self._START_MARKER)
        end = text.index("\nfi", start)
        return text[start : end + len("\nfi")]

    def _run_block(self, path, overrides):
        block = self._extract_block(path)
        env = {"PATH": os.environ.get("PATH", "")}
        env.update(overrides)
        result = subprocess.run(
            ["bash", "-c", block + "\necho NW=${NWALKERS}"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"{path} {overrides}: exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result.stdout

    def test_nwalkers_1_is_one_walker_replica_mode_and_unchanged(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                stdout = self._run_block(
                    path,
                    {
                        "GF_LEGACY_RANK_LAYOUT": "0",
                        "N_COMPUTE_EFF": "4",
                        "NWALKERS": "1",
                    },
                )
                self.assertIn("one-walker replica mode", stdout)
                self.assertIn("NW=1", stdout.splitlines())

    def test_nwalkers_6_still_rounds_up_to_8(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                stdout = self._run_block(
                    path,
                    {
                        "GF_LEGACY_RANK_LAYOUT": "0",
                        "N_COMPUTE_EFF": "4",
                        "NWALKERS": "6",
                    },
                )
                self.assertIn("is not a multiple of N_COMPUTE", stdout)
                self.assertIn("NW=8", stdout.splitlines())


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

    def test_ngpus_2_explicit_legacy_still_launches_today_shape(self):
        # GF_LEGACY_RANK_LAYOUT=1 is the rollback knob: same resource request,
        # --ntasks=3, mpiexec -n 3, the single-compute-rank layout.
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(
                    script, {"NGPUS": "2", "GF_LEGACY_RANK_LAYOUT": "1"}
                )
                self.assertIn("--ntasks=3", lines)
                self.assertIn("--nodes=1", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=1")

    def test_ngpus_2_default_is_walker_block(self):
        # user ruling 2026-09-16 (WP7 Steps 0-2/4 green): the walker-block
        # layout is the default at NGPUS=2 -- head + 1 compute + saver, still
        # --ntasks=3, but GF_LEGACY_RANK_LAYOUT=0 is what the job sees.
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(script, {"NGPUS": "2"})
                self.assertIn("--ntasks=3", lines)
                self.assertIn("--nodes=1", lines)
                self.assertIn("--partition=gpu-80-spot", lines)
                self.assertIn("--gres=gpu:2", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")

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

    def test_nodes_knob_spreads_the_gpus_one_per_node(self):
        # one-walker replica gates (user ruling 2026-09-16: test ACROSS nodes):
        # NGPUS=2 NODES=2 -> 2 nodes x gpu:1, head + 1 compute + saver, cyclic
        for script in SCRIPTS:
            with self.subTest(script=script, nodes=2):
                lines = self._run_dispatch(script, {"NGPUS": "2", "NODES": "2"})
                self.assertIn("--nodes=2", lines)
                self.assertIn("--gres=gpu:1", lines)
                self.assertIn("--ntasks=3", lines)
                self.assertIn("--distribution=cyclic", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")
            with self.subTest(script=script, nodes=4):
                lines = self._run_dispatch(script, {"NGPUS": "4", "NODES": "4"})
                self.assertIn("--nodes=4", lines)
                self.assertIn("--gres=gpu:1", lines)
                self.assertIn("--ntasks=5", lines)
                self.assertIn("--distribution=cyclic", lines)
            with self.subTest(script=script, nodes="unset"):
                # unset NODES = the NGPUS table, unchanged
                lines = self._run_dispatch(script, {"NGPUS": "2"})
                self.assertIn("--nodes=1", lines)
                self.assertIn("--gres=gpu:2", lines)

    def test_nodes_knob_must_divide_ngpus(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                env = {k: v for k, v in os.environ.items() if k not in _DISPATCH_ENV_KEYS}
                env["PATH"] = self.stub_dir + os.pathsep + env.get("PATH", "")
                env.update({"NGPUS": "2", "NODES": "3"})
                result = subprocess.run(
                    ["bash", script], env=env, capture_output=True, text=True, timeout=60
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("must be >= 1 and divide NGPUS=2", result.stdout)

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
