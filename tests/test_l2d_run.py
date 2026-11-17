"""Tests for the L2D run entry point: ``build_curr_info``, the snapshot, and ``scripts/run_global.py --config``.

``globalfit/run.py`` imports mpi4py, which cannot load its MPI library on every dev machine: a stand-in
``mpi4py`` (one rank, chosen per test) is installed while the engine is imported and the script runs.
The data is the CPU white-noise stand-in ``FakeL1``; nothing is sampled.
"""

# Pre-existing env shims (see tests/test_quintic_response.py): the installed
# scienceplots raises AttributeError at import, and eryn.utils.plot uses a bare
# ``typing`` name without importing it. Both must run before importing lisatools.
import sys as _sys

_sys.modules["scienceplots"] = None

import builtins as _builtins
import typing as _typing

_builtins.typing = _typing

import contextlib
import importlib.util
import io
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

import yaml

from lisatools.globalfit.stock.l2d import load_settings
from lisatools.globalfit.stock.l2d.io import load_run
from lisatools.globalfit.stock.l2d.load_settings import SNAPSHOT_NAME, build_curr_info, write_snapshot

from tests.test_l2d_blocks import NOISE_PRIOR, cpu_raw
from tests.test_l2d_load_settings import FakeL1

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_global.py"


class FakeComm:
    def __init__(self, rank):
        self.rank = rank

    def Get_rank(self):
        return self.rank


def fake_mpi4py(rank=0):
    mpi = types.SimpleNamespace(COMM_WORLD=FakeComm(rank), Comm=FakeComm)
    module = types.ModuleType("mpi4py")
    module.MPI = mpi
    return {"mpi4py": module, "mpi4py.MPI": mpi}


def load_script():
    """``scripts/run_global.py`` as a module (its import-time CUDA pre-init sees no --config / -sfp)."""
    spec = importlib.util.spec_from_file_location("run_global_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_run_folder(root: Path, psd=None, **global_changes) -> Path:
    """A CPU run folder: global.yaml + blocks/psd.yaml."""
    raw = cpu_raw()
    raw.pop("blocks")
    raw.update(global_changes)
    (root / "blocks").mkdir(parents=True)
    (root / "global.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    (root / "blocks" / "psd.yaml").write_text(yaml.safe_dump(psd or {"type": "psd", "priors": NOISE_PRIOR}))
    return root


class _Run(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for patcher in (
            mock.patch.dict(load_settings.LOADERS, {"mojito_l1": FakeL1}),
            mock.patch.dict(_sys.modules, fake_mpi4py(rank=0)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        warnings.simplefilter("ignore", UserWarning)
        self.addCleanup(warnings.resetwarnings)


class BuildCurrInfoTest(_Run):
    def test_curr_info_holds_every_block_and_the_recipe(self):
        from lisatools.globalfit.run import CurrentInfoGlobalFit
        from lisatools.globalfit.stock.erebor import PSDSetup

        cfg = load_run(write_run_folder(self.root / "run"))
        curr = build_curr_info(cfg)
        self.assertIsInstance(curr, CurrentInfoGlobalFit)
        self.assertEqual(curr.branch_names, ["psd"])
        self.assertIsInstance(curr.source_info["psd"], PSDSetup)
        self.assertEqual(curr.engine_info.ndims, {"psd": 2})
        settings = curr.settings_dict
        self.assertEqual(settings.setup_function.__name__, "setup_recipe")
        self.assertEqual(list(settings.source_metadata), ["psd"])
        self.assertEqual(
            (settings.rank_info.head_rank, settings.rank_info.main_rank),
            (cfg.compute.head_rank, cfg.compute.main_rank),
        )
        # the engine works on a deep copy; the setup function is kept by reference
        self.assertIs(curr.current_info.setup_function, settings.setup_function)

    def test_disabled_blocks_are_left_out(self):
        cfg = load_run(write_run_folder(self.root / "run", psd={"type": "psd", "priors": NOISE_PRIOR, "enabled": False}))
        self.assertEqual(build_curr_info(cfg).branch_names, [])

    def test_nothing_is_written_but_the_general_setup_artifacts(self):
        cfg = load_run(write_run_folder(self.root / "run"))
        curr = build_curr_info(cfg)
        self.assertFalse(Path(curr.general_info.main_file_path).exists())  # no backend file


class SnapshotTest(_Run):
    def test_snapshot_reloads_to_the_same_config(self):
        cfg = load_run(write_run_folder(self.root / "run"))
        general = build_curr_info(cfg).general_info
        path = write_snapshot(cfg, general)
        self.assertEqual(path, Path(general.artifacts_file_dir) / SNAPSHOT_NAME)
        self.assertEqual(load_run(path), cfg)

    def test_prior_output_file_is_written_whole_and_leaves_no_temp_files(self):
        out = self.root / "run" / "priors" / "psd_prior.yaml"
        folder = write_run_folder(self.root / "run", psd={"type": "psd", "priors": {**NOISE_PRIOR, "output_file": "priors/psd_prior.yaml"}})
        out.parent.mkdir()
        build_curr_info(load_run(folder))
        self.assertEqual(yaml.safe_load(out.read_text())["branch_name"], "psd")
        self.assertEqual(sorted(p.name for p in out.parent.iterdir()), ["psd_prior.yaml"])


class ScriptTest(_Run):
    def setUp(self):
        super().setUp()
        self.script = load_script()

    def main(self, *argv):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = self.script.main(list(argv))
        return code, stdout.getvalue()

    def test_dry_run_builds_reports_and_snapshots(self):
        folder = write_run_folder(self.root / "run")
        code, out = self.main("--config", str(folder), "--dry-run")
        self.assertEqual(code, 0)
        self.assertIn("dry run: full_run_1_fd built, nothing sampled", out)
        self.assertIn("psd: 1 leaves x 2 params, ladder 5, no search, pe_move MultiGPUPSDMove", out)
        self.assertIn("recipe: StockRecipe", out)
        snapshot = next((folder / "out").rglob(SNAPSHOT_NAME))
        self.assertIn(f"resolved config: {snapshot}", out)
        self.assertFalse(list((folder / "out").rglob("*.h5")))  # nothing sampled, no backend

    def test_dry_run_on_another_rank_neither_prints_nor_snapshots(self):
        folder = write_run_folder(self.root / "run")
        with mock.patch.dict(_sys.modules, fake_mpi4py(rank=1)):
            code, out = self.main("--config", str(folder), "--dry-run")
        self.assertEqual(code, 0)
        self.assertNotIn("dry run:", out)  # (library import notices may still print)
        self.assertFalse(list((folder / "out").rglob(SNAPSHOT_NAME)))

    def test_config_runs_the_global_fit_with_the_built_info(self):
        folder = write_run_folder(self.root / "run")
        with mock.patch.object(self.script, "run") as run:
            code, _ = self.main("--config", str(folder))
        self.assertEqual(code, 0)
        (curr, comm), _ = run.call_args
        self.assertEqual(curr.branch_names, ["psd"])
        self.assertEqual(comm.Get_rank(), 0)

    def test_invalid_config_fails_before_building(self):
        folder = write_run_folder(self.root / "run", sampler={"nwalkers": 4})  # ntemps missing
        with mock.patch.object(load_settings, "load_general_setup") as general:
            with self.assertRaises(Exception) as ctx:
                self.main("--config", str(folder), "--dry-run")
        self.assertIn("sampler.ntemps", str(ctx.exception))
        general.assert_not_called()

    def test_exactly_one_source_of_settings(self):
        for argv in ([], ["--config", "a", "-sfp", "b.py"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.script.make_parser().parse_args(argv)

    def test_dry_run_needs_a_config(self):
        with self.assertRaisesRegex(SystemExit, "--dry-run needs --config"):
            self.script.main(["-sfp", "settings.py", "--dry-run"])

    def test_legacy_settings_file_path(self):
        settings = self.root / "legacy_settings.py"
        settings.write_text("def get_global_fit_settings():\n    return 'legacy curr info'\n")
        with mock.patch.object(self.script, "run") as run:
            self.script.main(["-sfp", str(settings)])
        self.assertEqual(run.call_args.args[0], "legacy curr info")


class PreInitCudaTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.script = load_script()

    def device_set(self, argv, parent=None):
        cdll = mock.MagicMock()
        with mock.patch.object(self.script.sys, "argv", ["run_global.py", *argv]), \
                mock.patch.object(self.script.ctypes, "CDLL", cdll), \
                mock.patch.object(self.script.multiprocessing, "parent_process", return_value=parent):
            self.script._pre_init_cuda()
        calls = cdll.return_value.cudaSetDevice.call_args_list
        return [c.args[0] for c in calls]

    def test_config_folder_and_resolved_file(self):
        folder = self.root / "run"
        folder.mkdir()
        (folder / "global.yaml").write_text("compute: {backend: cuda12x, gpus: [4, 5]}\nrun: {name: x}\n")
        resolved = self.root / "resolved.yaml"
        resolved.write_text("compute:\n  gpus: [2]\n")
        self.assertEqual(self.device_set(["--config", str(folder)]), [4])
        self.assertEqual(self.device_set(["-c", str(resolved), "--dry-run"]), [2])

    def test_merge_keys_are_understood(self):
        folder = self.root / "run"
        folder.mkdir()
        (folder / "global.yaml").write_text("base: &b {backend: cuda12x, gpus: [6]}\ncompute: {<<: *b}\n")
        self.assertEqual(self.device_set(["--config", str(folder)]), [6])

    def test_no_device_for_cpu_missing_or_broken_configs(self):
        folder = self.root / "run"
        folder.mkdir()
        (folder / "global.yaml").write_text("compute: {backend: cpu}\n")
        broken = self.root / "broken.yaml"
        broken.write_text("compute: [unclosed\n")
        for argv in (["--config", str(folder)], ["--config", str(self.root / "nope")], ["--config", str(broken)], []):
            with self.subTest(argv=argv):
                self.assertEqual(self.device_set(argv), [])

    def test_spawned_children_never_set_a_device(self):
        folder = self.root / "run"
        folder.mkdir()
        (folder / "global.yaml").write_text("compute: {gpus: [4]}\n")
        self.assertEqual(self.device_set(["--config", str(folder)], parent=object()), [])

    def test_legacy_settings_file(self):
        settings = self.root / "settings.py"
        settings.write_text("def get_settings():\n    gpus = [3, 1]\n    return gpus\n")
        self.assertEqual(self.device_set(["-sfp", str(settings)]), [3])


if __name__ == "__main__":
    unittest.main()
