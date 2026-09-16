"""Multi-rank-aware diagnostics tooling (Plan 5 Task 4).

Covers: ``discover_run_logs()`` in both ``scripts/diagnostics/gf_monitor_gen.py``
and ``scripts/diagnostics/gf_run_log_digest.py`` (head log + per-rank logs,
head first), ``summarize_fanout()`` in ``gf_run_log_digest.py``,
``fanout_digest_line()`` in ``communication/fanout.py``, and
``scripts/diagnostics/gf_state_digest.py``'s per-array hashing + CLI diff
behaviour.
"""
import importlib.util
import os
import re
import tempfile
import unittest

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIAG_DIR = os.path.join(_REPO_ROOT, "scripts", "diagnostics")


def _load_diagnostics_module(filename, modname):
    """Load a ``scripts/diagnostics/*.py`` file without running its main body.

    ``gf_monitor_gen.py`` / ``gf_run_log_digest.py`` guard everything after
    their testable helpers with ``if __name__ != "__main__": raise
    SystemExit(0)``; ``gf_state_digest.py`` has no unguarded top-level side
    effects at all (its executable part is already under ``if __name__ ==
    "__main__":``). Either way this loads cleanly without a real run dir.
    """
    path = os.path.join(_DIAG_DIR, filename)
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    return module


# Mirrors gf_monitor_gen.py's RJ_SPLIT_RE (defined after that script's
# SystemExit guard, so not importable from here -- kept in sync manually).
RJ_SPLIT_RE = re.compile(
    r"\[GB_ACCEPT rj-split (\w+)\] births (\d+): viable (\d+) "
    r"\(acc (\d+)[^|]*\| gated: prior (\d+) oob (\d+) capped (\d+) "
    r"\| scored-dropped: snr (\d+) kernel (\d+)"
)

_RJ_SPLIT_LINE = (
    "[GB_ACCEPT rj-split gb] births 1: viable 1 (acc 1/1, foo=bar) "
    "| gated: prior 0 oob 0 capped 0 | scored-dropped: snr 0 kernel 0\n"
)


class DiscoverRunLogsTest(unittest.TestCase):
    """gf_monitor_gen.py / gf_run_log_digest.py: multi-rank log discovery."""

    @classmethod
    def setUpClass(cls):
        cls.gf_monitor_gen = _load_diagnostics_module(
            "gf_monitor_gen.py", "_test_gf_monitor_gen"
        )
        cls.gf_run_log_digest = _load_diagnostics_module(
            "gf_run_log_digest.py", "_test_gf_run_log_digest"
        )

    def _make_run_dir(self, tmpdir):
        head = os.path.join(tmpdir, "globalfit_run.log")
        rank1 = os.path.join(tmpdir, "globalfit_run.rank1.log")
        with open(head, "w") as f:
            f.write(_RJ_SPLIT_LINE)
        with open(rank1, "w") as f:
            f.write(_RJ_SPLIT_LINE)
        return head, rank1

    def test_gf_monitor_gen_discovers_head_and_rank_logs_head_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            head, rank1 = self._make_run_dir(tmpdir)
            found = self.gf_monitor_gen.discover_run_logs(tmpdir)
            self.assertEqual(found, [head, rank1])

    def test_gf_monitor_gen_concatenated_logs_sum_rj_split_matches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_run_dir(tmpdir)
            found = self.gf_monitor_gen.discover_run_logs(tmpdir)
            text = "".join(open(p).read() for p in found)
            self.assertEqual(len(RJ_SPLIT_RE.findall(text)), 2)

    def test_gf_run_log_digest_discovers_head_and_rank_logs_head_first(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            head, rank1 = self._make_run_dir(tmpdir)
            found = self.gf_run_log_digest.discover_run_logs(tmpdir)
            self.assertEqual(found, [head, rank1])

    def test_discover_run_logs_multi_digit_rank_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            head = os.path.join(tmpdir, "globalfit_run.log")
            r2 = os.path.join(tmpdir, "globalfit_run.rank2.log")
            r10 = os.path.join(tmpdir, "globalfit_run.rank10.log")
            for p in (head, r2, r10):
                with open(p, "w") as f:
                    f.write(_RJ_SPLIT_LINE)
            found = self.gf_monitor_gen.discover_run_logs(tmpdir)
            self.assertEqual(found, [head, r2, r10])

    def test_discover_run_logs_no_head_still_orders_ranks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = os.path.join(tmpdir, "globalfit_run.rank1.log")
            r2 = os.path.join(tmpdir, "globalfit_run.rank2.log")
            for p in (r2, r1):
                with open(p, "w") as f:
                    f.write(_RJ_SPLIT_LINE)
            found = self.gf_run_log_digest.discover_run_logs(tmpdir)
            self.assertEqual(found, [r1, r2])


class SummarizeFanoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gf_run_log_digest = _load_diagnostics_module(
            "gf_run_log_digest.py", "_test_gf_run_log_digest_fanout"
        )

    def test_summarize_fanout_three_lines(self):
        lines = [
            "2026-09-16 - x - DEBUG - [FANOUT] op=addremove move=mv1 "
            "head_s=1.000 max_rank_s=2.000 wait_s=0.500",
            "2026-09-16 - x - DEBUG - [FANOUT] op=addremove move=mv1 "
            "head_s=3.000 max_rank_s=4.000 wait_s=1.500",
            "2026-09-16 - x - DEBUG - [FANOUT] op=psd move=mv2 "
            "head_s=0.100 max_rank_s=0.200 wait_s=0.050",
        ]
        out = self.gf_run_log_digest.summarize_fanout(lines)
        self.assertEqual(set(out), {("addremove", "mv1"), ("psd", "mv2")})
        a = out[("addremove", "mv1")]
        self.assertEqual(a["count"], 2)
        self.assertAlmostEqual(a["mean_head_s"], 2.0)
        self.assertAlmostEqual(a["max_rank_s"], 4.0)
        self.assertAlmostEqual(a["mean_wait_s"], 1.0)
        b = out[("psd", "mv2")]
        self.assertEqual(b["count"], 1)
        self.assertAlmostEqual(b["mean_head_s"], 0.1)

    def test_summarize_fanout_empty_without_fanout_lines(self):
        out = self.gf_run_log_digest.summarize_fanout(["no fanout here\n"])
        self.assertEqual(out, {})


class FanoutDigestLineTest(unittest.TestCase):
    def test_deterministic_on_make_state(self):
        from lisatools.globalfit.communication.fanout import fanout_digest_line
        from tests.test_gf_substate_roundtrip import make_state

        rng = np.random.default_rng(7)
        state = make_state(rng)
        line1 = fanout_digest_line(3, state)
        line2 = fanout_digest_line(3, state)
        self.assertEqual(line1, line2)
        self.assertTrue(line1.startswith("[FANOUT_DIGEST] it=3 "))
        self.assertIn("log_like=", line1)
        self.assertIn("coords=", line1)
        self.assertIn("inds=", line1)

    def test_changed_coord_changes_coords_hash_only(self):
        from lisatools.globalfit.communication.fanout import fanout_digest_line
        from tests.test_gf_substate_roundtrip import make_state

        state_a = make_state(np.random.default_rng(7))
        state_b = make_state(np.random.default_rng(7))
        state_b.branches["gb"].coords[0, 0, 0, 0] += 1.0

        line_a = fanout_digest_line(5, state_a)
        line_b = fanout_digest_line(5, state_b)
        self.assertNotEqual(line_a, line_b)

        def _field(line, name):
            for tok in line.split():
                if tok.startswith(f"{name}="):
                    return tok.split("=", 1)[1]
            raise AssertionError(f"{name} not found in {line!r}")

        self.assertEqual(_field(line_a, "log_like"), _field(line_b, "log_like"))
        self.assertNotEqual(_field(line_a, "coords"), _field(line_b, "coords"))


class GfStateDigestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gf_state_digest = _load_diagnostics_module(
            "gf_state_digest.py", "_test_gf_state_digest"
        )

    def test_array_sha1_deterministic(self):
        arr = np.arange(12.0).reshape(3, 4)
        gsd = self.gf_state_digest
        self.assertEqual(gsd.array_sha1(arr), gsd.array_sha1(arr.copy()))
        self.assertNotEqual(gsd.array_sha1(arr), gsd.array_sha1(arr + 1))

    def test_state_arrays_on_make_state(self):
        from tests.test_gf_substate_roundtrip import make_state

        state = make_state(np.random.default_rng(11))
        arrays = self.gf_state_digest.state_arrays(state)
        for expected in (
            "log_like", "log_prior", "betas",
            "coords/gb", "inds/gb",
            "substate/gb/chain", "substate/gb/band_edges",
        ):
            self.assertIn(expected, arrays)

    def _write_tiny_store(self, tmpdir, seed, ntemps=2, nwalkers=2, num_bands=2):
        from lisatools.globalfit.hdfbackend import GBHDFBackend, GFHDFBackend, ModuleSubBackend
        from lisatools.globalfit.state import GBState, GFState, ModuleSubState

        branch_shapes = {"gb": (2, 8), "psd": (1, 4)}
        band_edges = np.linspace(1e-3, 7e-3, num_bands + 1)
        rng = np.random.default_rng(seed)
        coords = {
            name: rng.standard_normal((ntemps, nwalkers, nleaves, ndim))
            for name, (nleaves, ndim) in branch_shapes.items()
        }
        inds = {
            name: np.ones((ntemps, nwalkers, nleaves), dtype=bool)
            for name, (nleaves, _) in branch_shapes.items()
        }
        sub_state_bases = {"gb": GBState, "psd": ModuleSubState}
        state = GFState(
            coords, inds=inds,
            log_like=rng.standard_normal((ntemps, nwalkers)),
            log_prior=rng.standard_normal((ntemps, nwalkers)),
            betas=np.linspace(1.0, 0.1, ntemps),
            random_state=np.random.RandomState(seed).get_state(),
            sub_state_bases=sub_state_bases,
        )
        band_temps = np.tile(np.linspace(1.0, 0.1, ntemps), (num_bands, 1))
        state.sub_states["gb"].initialize_band_information(
            nwalkers, ntemps, band_edges, band_temps
        )
        for name, sub in state.sub_states.items():
            sub.pull_from_main(state, name)

        fp = os.path.join(tmpdir, f"tiny_store_{seed}.h5")
        backend = GFHDFBackend(
            fp,
            sub_backend={"gb": GBHDFBackend, "psd": ModuleSubBackend},
            sub_state_bases=sub_state_bases,
        )
        ndims = {name: shape[1] for name, shape in branch_shapes.items()}
        nleaves_max = {name: shape[0] for name, shape in branch_shapes.items()}
        sub_reset_kwargs = {
            "gb": dict(
                nleaves_max=branch_shapes["gb"][0], ndim=branch_shapes["gb"][1],
                num_bands=num_bands, band_edges=band_edges,
            ),
            "psd": dict(nleaves_max=branch_shapes["psd"][0], ndim=branch_shapes["psd"][1]),
        }
        backend.reset(
            nwalkers, ndims, nleaves_max=nleaves_max, ntemps=ntemps,
            branch_names=list(branch_shapes.keys()), nbranches=len(branch_shapes),
            rj=False, moves=None, sub_reset_kwargs=sub_reset_kwargs,
        )
        backend.grow(1, None)
        accepted = np.ones((ntemps, nwalkers))
        backend.save_step(state, accepted)
        return fp

    def test_digest_store_deterministic_and_matches_itself(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = self._write_tiny_store(tmpdir, seed=1)
            d1 = self.gf_state_digest.digest_store(fp)
            d2 = self.gf_state_digest.digest_store(fp)
            self.assertEqual(d1, d2)
            self.assertIn("substate/gb/chain", d1)
            self.assertIn("substate/gb/band_edges", d1)
            self.assertIn("coords/gb", d1)

    def test_cli_main_identical_store_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fp = self._write_tiny_store(tmpdir, seed=2)
            rc = self.gf_state_digest.main([fp, fp])
            self.assertEqual(rc, 0)

    def test_cli_main_different_stores_exits_one(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fp1 = self._write_tiny_store(tmpdir, seed=3)
            fp2 = self._write_tiny_store(tmpdir, seed=4)
            rc = self.gf_state_digest.main([fp1, fp2])
            self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
