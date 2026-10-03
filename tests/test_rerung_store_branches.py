"""Re-rung a store's branches in place (scripts/fstat_proposal/rerung_store_branches.py).

User request 2026-10-02: add temperatures to the MBHBs and EMRIs of the 6mo
store (2 rungs each) by duplicating the existing rungs up the ladder, the count
adjustable per branch, and be able to take an existing branch (SOBHB) DOWN.
The synthetic store below mirrors the real layout read off the job-695 store:
per-leaf-ladder branches (betas_all, log_like (nleaves, ntemps, nwalkers)),
flat-ladder branches (betas, log_like (ntemps, nwalkers)), and a banded branch
that must be refused.
"""

import os
import shutil
import sys
import tempfile
import unittest

import h5py
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir,
                    "scripts", "fstat_proposal"))

import rerung_store_branches as R  # noqa: E402


def _ms(shape):
    """The backend's maxshape: the ITERATION axis 0 unlimited, the rest fixed
    (the 2026-10-03 production abort came from a fixture that pinned axis 0)."""
    return (None,) + tuple(int(x) for x in shape[1:])

ROWS, NW = 6, 4


def _per_leaf_group(root, name, nt, nl, nd, leaf_count_name):
    g = root["sub_backend"].create_group(name)
    g.attrs.update({"ndim": nd, "nleaves_max": nl, "ntemps": nt, "nwalkers": NW,
                    leaf_count_name: nl})
    kw = dict(compression="gzip", compression_opts=4)
    t = np.arange(nt)
    chain = (1000.0 * t[None, :, None, None, None] + 100.0 * np.arange(NW)[None, None, :, None, None]
             + 10.0 * np.arange(nl)[None, None, None, :, None] + np.arange(nd)[None, None, None, None, :]
             + 1e4 * np.arange(ROWS)[:, None, None, None, None])
    g.create_dataset("chain", data=chain, chunks=(3, 1, 1, 2, nd), maxshape=_ms(chain.shape), **kw)
    inds = np.ones((ROWS, nt, NW, nl), dtype=bool)
    inds[:, :, :, -1] = False
    g.create_dataset("inds", data=inds, chunks=(3, 1, 2, nl), maxshape=_ms(inds.shape), **kw)
    ll = (-1000.0 * t[None, None, :, None] - 10.0 * np.arange(nl)[None, :, None, None]
          - np.arange(NW)[None, None, None, :] - 1e4 * np.arange(ROWS)[:, None, None, None])
    g.create_dataset("log_like", data=ll, chunks=(3, 2, 1, 2), maxshape=_ms(ll.shape), **kw)
    g.create_dataset("log_prior", data=0.5 * ll, chunks=(3, 2, 1, 2), maxshape=_ms(ll.shape), **kw)
    betas = np.tile((1.0 / 1.5 ** t)[None, None, :], (ROWS, nl, 1))
    g.create_dataset("betas_all", data=betas, chunks=(3, 2, 1), maxshape=_ms(betas.shape), **kw)
    for cname in ("in_model_accepted", "in_model_proposed", "rj_accepted", "rj_proposed"):
        c = (7 + t[None, None, :] + 100 * np.arange(nl)[None, :, None]
             + 0 * np.arange(ROWS)[:, None, None]).astype(np.int64)
        g.create_dataset(cname, data=c, chunks=(3, 2, 1), maxshape=_ms(c.shape), **kw)
    for cname in ("swaps_accepted", "swaps_proposed"):
        s = (3 + np.arange(nt - 1)[None, None, :] + 100 * np.arange(nl)[None, :, None]
             + 0 * np.arange(ROWS)[:, None, None]).astype(np.int64)
        g.create_dataset(cname, data=s, chunks=(3, 2, 1), maxshape=_ms(s.shape), **kw)
    for dname in ("d_h", "h_h"):
        d = np.arange(ROWS * NW * nl, dtype=float).reshape(ROWS, NW, nl)
        g.create_dataset(dname, data=d, chunks=(3, 1, nl), maxshape=_ms(d.shape), **kw)
    return g


def _flat_group(root, name, nt, nd):
    g = root["sub_backend"].create_group(name)
    g.attrs.update({"ndim": nd, "nleaves_max": 1, "ntemps": nt, "nwalkers": NW})
    kw = dict(compression="gzip", compression_opts=4)
    t = np.arange(nt)
    chain = (1000.0 * t[None, :, None, None, None] + 100.0 * np.arange(NW)[None, None, :, None, None]
             + np.arange(nd)[None, None, None, None, :] + 0 * np.arange(ROWS)[:, None, None, None, None])
    g.create_dataset("chain", data=chain, chunks=(3, 2, 2, 1, 1), maxshape=_ms(chain.shape), **kw)
    inds = np.ones((ROWS, nt, NW, 1), dtype=bool)
    g.create_dataset("inds", data=inds, chunks=(3, 2, 2, 1), maxshape=_ms(inds.shape), **kw)
    ll = -1000.0 * t[None, :, None] - np.arange(NW)[None, None, :] + 0 * np.arange(ROWS)[:, None, None]
    g.create_dataset("log_like", data=ll, chunks=(3, 2, 1), maxshape=_ms(ll.shape), **kw)
    g.create_dataset("log_prior", data=0.5 * ll, chunks=(3, 2, 1), maxshape=_ms(ll.shape), **kw)
    betas = np.tile((1.0 / 2.0 ** t)[None, :], (ROWS, 1))
    g.create_dataset("betas", data=betas, chunks=(3, 2), maxshape=_ms(betas.shape), **kw)
    for cname in ("in_model_accepted", "in_model_proposed", "rj_accepted", "rj_proposed"):
        c = (5 + t[None, :] + 0 * np.arange(ROWS)[:, None]).astype(np.int64)
        g.create_dataset(cname, data=c, chunks=(3, 2), maxshape=_ms(c.shape), **kw)
    for cname in ("swaps_accepted", "swaps_proposed"):
        s = (2 + np.arange(nt - 1)[None, :] + 0 * np.arange(ROWS)[:, None]).astype(np.int64)
        g.create_dataset(cname, data=s, chunks=(3, 2), maxshape=_ms(s.shape), **kw)
    for dname in ("d_h", "h_h"):
        d = np.ones((ROWS, NW, 1))
        g.create_dataset(dname, data=d, chunks=(3, 1, 1), maxshape=_ms(d.shape), **kw)
    return g


def make_store(path):
    with h5py.File(path, "w") as f:
        root = f.create_group("global_fit")
        root.attrs.update({"iteration": ROWS, "nwalkers": NW, "ntemps": 1, "nsamplers": 1,
                           "has_recipe": False})
        root.create_group("sub_backend")
        inds = root.create_group("inds")
        chain = root.create_group("chain")
        mbh = _per_leaf_group(root, "mbh", nt=2, nl=4, nd=11, leaf_count_name="num_mbhs")
        _per_leaf_group(root, "sobbh", nt=8, nl=6, nd=11, leaf_count_name="num_sobbhs")
        _flat_group(root, "psd", nt=3, nd=2)
        # the main (cold) group mirrors rung 0 of every branch: the resume cross-checks it
        for b in ("mbh", "sobbh", "psd"):
            sub = root["sub_backend"][b]
            inds.create_dataset(b, data=sub["inds"][:, 0][:, None, None])
            chain.create_dataset(b, data=sub["chain"][:, 0][:, None, None])
        gb = root["sub_backend"].create_group("gb")
        gb.attrs.update({"ndim": 9, "nleaves_max": 10, "ntemps": 4, "nwalkers": NW, "num_bands": 3})
        gb.create_dataset("band_temps", data=np.ones((ROWS, 3, 4)))
        return mbh


class _Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rerung_")
        self.path = os.path.join(self.tmp, "run_testing.h5")
        make_store(self.path)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read(self, branch, name):
        with h5py.File(self.path, "r") as f:
            return f["global_fit/sub_backend"][branch][name][...]

    def _attr(self, branch, key):
        with h5py.File(self.path, "r") as f:
            return f["global_fit/sub_backend"][branch].attrs[key]


class FillMapTest(unittest.TestCase):
    def test_modes(self):
        self.assertEqual(R.fill_map(5, 2, "cycle"), [0, 1, 0, 1, 0])
        self.assertEqual(R.fill_map(5, 2, "hottest"), [0, 1, 1, 1, 1])
        self.assertEqual(R.fill_map(5, 2, "coldest"), [0, 1, 0, 0, 0])
        self.assertEqual(R.fill_map(3, 8, "cycle"), [0, 1, 2])           # shrink keeps the coldest
        self.assertEqual(R.fill_map(2, 2, "cycle"), [0, 1])
        with self.assertRaises(ValueError):
            R.fill_map(5, 2, "random")

    def test_ladders(self):
        np.testing.assert_allclose(R.new_ladder(4, True, 11), 1 / 1.2 ** np.arange(4))
        np.testing.assert_allclose(R.new_ladder(3, True, 11, ratio=2.0), [1, 0.5, 0.25])
        np.testing.assert_allclose(R.new_ladder(3, False, 2, betas=[1, 0.3, 0.1]), [1, 0.3, 0.1])
        flat = R.new_ladder(5, False, 2)                                   # eryn make_ladder
        self.assertEqual(flat.shape, (5,))
        self.assertEqual(flat[0], 1.0)
        self.assertTrue(np.all(np.diff(flat) < 0))
        with self.assertRaises(ValueError):
            R.new_ladder(3, True, 11, betas=[1, 0.5])                      # wrong length
        with self.assertRaises(ValueError):
            R.new_ladder(3, True, 11, betas=[1, 0.5, 0.6])                 # not decreasing
        with self.assertRaises(ValueError):
            R.new_ladder(3, True, 11, ratio=0.9)


class GrowPerLeafTest(_Base):
    """mbh 2 -> 5 rungs, cycle fill: the job-695 request."""

    def test_states_copied_counters_zeroed_ladder_rebuilt(self):
        old = {k: self._read("mbh", k) for k in ("chain", "inds", "log_like", "log_prior",
                                                  "in_model_accepted", "swaps_accepted", "d_h")}
        plans = R.plan(self.path, {"mbh": 5})
        self.assertEqual(len(plans), 1)
        bp = plans[0]
        self.assertEqual((bp.old_nt, bp.new_nt, bp.src), (2, 5, [0, 1, 0, 1, 0]))
        self.assertEqual({d.name for d in bp.datasets},
                         {"chain", "inds", "log_like", "log_prior", "betas_all", "in_model_accepted",
                          "in_model_proposed", "rj_accepted", "rj_proposed", "swaps_accepted",
                          "swaps_proposed"})
        R.apply_plan(self.path, plans, log=lambda *_: None)
        chain = self._read("mbh", "chain")
        self.assertEqual(chain.shape, (ROWS, 5, NW, 4, 11))
        np.testing.assert_array_equal(chain[:, :2], old["chain"])            # rungs 0, 1 untouched
        np.testing.assert_array_equal(chain[:, 2], old["chain"][:, 0])
        np.testing.assert_array_equal(chain[:, 3], old["chain"][:, 1])
        np.testing.assert_array_equal(chain[:, 4], old["chain"][:, 0])
        inds = self._read("mbh", "inds")
        np.testing.assert_array_equal(inds[:, 3], old["inds"][:, 1])
        ll = self._read("mbh", "log_like")                                    # (rows, nl, T, nw)
        self.assertEqual(ll.shape, (ROWS, 4, 5, NW))
        np.testing.assert_array_equal(ll[:, :, 4, :], old["log_like"][:, :, 0, :])
        np.testing.assert_array_equal(ll[:, :, 1, :], old["log_like"][:, :, 1, :])
        np.testing.assert_array_equal(self._read("mbh", "log_prior")[:, :, 3, :],
                                      old["log_prior"][:, :, 1, :])
        c = self._read("mbh", "in_model_accepted")                            # (rows, nl, T)
        self.assertEqual(c.shape, (ROWS, 4, 5))
        np.testing.assert_array_equal(c[:, :, :2], old["in_model_accepted"])
        self.assertTrue((c[:, :, 2:] == 0).all())
        s = self._read("mbh", "swaps_accepted")                               # (rows, nl, T-1)
        self.assertEqual(s.shape, (ROWS, 4, 4))
        np.testing.assert_array_equal(s[:, :, 0], old["swaps_accepted"][:, :, 0])
        self.assertTrue((s[:, :, 1:] == 0).all())
        b = self._read("mbh", "betas_all")
        self.assertEqual(b.shape, (ROWS, 4, 5))
        np.testing.assert_allclose(b, np.broadcast_to(1 / 1.2 ** np.arange(5), b.shape))
        np.testing.assert_array_equal(self._read("mbh", "d_h"), old["d_h"])  # no temperature axis
        self.assertEqual(int(self._attr("mbh", "ntemps")), 5)
        self.assertEqual(int(self._attr("mbh", "num_mbhs")), 4)
        # idempotent: re-planning at 5 reads as unchanged, and the verify step agrees
        self.assertTrue(R.plan(self.path, {"mbh": 5})[0].unchanged)
        self.assertIn("OK", R.verify(self.path, {"mbh": 5}))
        # the cold rung still matches the main group (the resume's cross-check)
        with h5py.File(self.path, "r") as f:
            np.testing.assert_array_equal(f["global_fit/inds/mbh"][:, 0, 0],
                                          f["global_fit/sub_backend/mbh/inds"][:, 0])

    def test_hottest_fill_and_explicit_betas(self):
        old = self._read("mbh", "chain")
        plans = R.plan(self.path, {"mbh": 4}, fill="hottest", betas=[1.0, 0.5, 0.2, 0.05])
        R.apply_plan(self.path, plans, log=lambda *_: None)
        chain = self._read("mbh", "chain")
        np.testing.assert_array_equal(chain[:, 2], old[:, 1])
        np.testing.assert_array_equal(chain[:, 3], old[:, 1])
        np.testing.assert_allclose(self._read("mbh", "betas_all")[0, 0], [1.0, 0.5, 0.2, 0.05])

    def test_compression_and_chunks_survive(self):
        R.apply_plan(self.path, R.plan(self.path, {"mbh": 6}), log=lambda *_: None)
        with h5py.File(self.path, "r") as f:
            ds = f["global_fit/sub_backend/mbh/chain"]
            self.assertEqual(ds.compression, "gzip")
            self.assertEqual(ds.chunks[0], 3)
            # the temperature (and every other) axis fixed, the ITERATION axis
            # unlimited -- the backend's convention
            self.assertEqual(ds.maxshape, (None,) + ds.shape[1:])

    def test_the_rewritten_store_can_still_grow(self):
        # 2026-10-03 production abort: the re-rung 6mo store refused eryn's
        # first grow ("dimension cannot exceed the existing maximal size (new:
        # 2085 max: 2084)") because every rewritten dataset had its iteration
        # axis pinned at the allocated row count. Every dataset of a re-rung
        # branch must resize along axis 0, and verify() must say so.
        R.apply_plan(self.path, R.plan(self.path, {"mbh": 6, "sobbh": 3}), log=lambda *_: None)
        self.assertEqual(R.fixed_axis0(self.path, ["mbh", "sobbh"]), [])
        self.assertIn("OK", R.verify(self.path, {"mbh": 6, "sobbh": 3}))
        with h5py.File(self.path, "r+") as f:
            for b in ("mbh", "sobbh"):
                g = f["global_fit/sub_backend"][b]
                for name in g:
                    ds = g[name]
                    self.assertIsNone(ds.maxshape[0], f"{b}/{name}")
                    ds.resize(ds.shape[0] + 1, axis=0)       # eryn's grow

    def test_repair_unlimited_mends_a_store_written_by_the_old_tool(self):
        # reproduce the old tool: fixed iteration axis on a rewritten dataset
        with h5py.File(self.path, "r+") as f:
            g = f["global_fit/sub_backend/mbh"]
            for name in ("chain", "log_like"):
                ds = g[name]
                R._recreate(g, name, ds, ds.shape, ds[...], maxshape=tuple(ds.shape))
            before = {name: g[name][...].copy() for name in g}
        self.assertEqual(R.fixed_axis0(self.path, ["mbh"]), ["mbh/chain", "mbh/log_like"])
        with self.assertRaises(RuntimeError):
            R.verify(self.path, {"mbh": 2})
        done = R.repair_unlimited(self.path, ["mbh"], log=lambda *_: None)
        self.assertEqual(done, ["mbh/chain", "mbh/log_like"])
        self.assertEqual(R.fixed_axis0(self.path, ["mbh"]), [])
        with h5py.File(self.path, "r+") as f:
            g = f["global_fit/sub_backend/mbh"]
            for name in g:
                np.testing.assert_array_equal(g[name][...], before[name], name)
                self.assertIsNone(g[name].maxshape[0], name)
            g["chain"].resize(g["chain"].shape[0] + 1, axis=0)
        # idempotent
        self.assertEqual(R.repair_unlimited(self.path, ["mbh"], log=lambda *_: None), [])

    def test_repair_cli_dry_run_then_apply(self):
        with h5py.File(self.path, "r+") as f:
            g = f["global_fit/sub_backend/sobbh"]
            ds = g["inds"]
            R._recreate(g, "inds", ds, ds.shape, ds[...], maxshape=tuple(ds.shape))
        self.assertEqual(R.main([self.path, "--repair-unlimited", "sobbh"]), 0)      # dry run
        self.assertEqual(R.fixed_axis0(self.path, ["sobbh"]), ["sobbh/inds"])
        self.assertEqual(R.main([self.path, "--repair-unlimited", "sobbh", "--apply",
                                 "--no-backup"]), 0)
        self.assertEqual(R.fixed_axis0(self.path, ["sobbh"]), [])


class ShrinkTest(_Base):
    def test_sobbh_8_to_3_keeps_the_coldest_rungs_and_their_stored_ladder(self):
        old = {k: self._read("sobbh", k) for k in ("chain", "log_like", "betas_all",
                                                    "rj_proposed", "swaps_proposed")}
        plans = R.plan(self.path, {"sobbh": 3}, ladder_mode="keep")
        self.assertIsNone(plans[0].ladder)
        R.apply_plan(self.path, plans, log=lambda *_: None)
        np.testing.assert_array_equal(self._read("sobbh", "chain"), old["chain"][:, :3])
        np.testing.assert_array_equal(self._read("sobbh", "log_like"), old["log_like"][:, :, :3, :])
        np.testing.assert_array_equal(self._read("sobbh", "betas_all"), old["betas_all"][:, :, :3])
        np.testing.assert_array_equal(self._read("sobbh", "rj_proposed"), old["rj_proposed"][:, :, :3])
        np.testing.assert_array_equal(self._read("sobbh", "swaps_proposed"),
                                      old["swaps_proposed"][:, :, :2])
        self.assertEqual(int(self._attr("sobbh", "ntemps")), 3)
        self.assertIn("OK", R.verify(self.path, {"sobbh": 3}))

    def test_shrink_with_a_fresh_ladder(self):
        plans = R.plan(self.path, {"sobbh": 3}, ladder_mode="fresh")
        R.apply_plan(self.path, plans, log=lambda *_: None)
        np.testing.assert_allclose(self._read("sobbh", "betas_all")[-1, 2], 1 / 1.2 ** np.arange(3))

    def test_flat_family_psd_3_to_2_and_2_to_5(self):
        old = {k: self._read("psd", k) for k in ("chain", "log_like", "betas", "swaps_accepted")}
        R.apply_plan(self.path, R.plan(self.path, {"psd": 2}), log=lambda *_: None)
        np.testing.assert_array_equal(self._read("psd", "chain"), old["chain"][:, :2])
        np.testing.assert_array_equal(self._read("psd", "log_like"), old["log_like"][:, :2])
        np.testing.assert_array_equal(self._read("psd", "betas"), old["betas"][:, :2])
        np.testing.assert_array_equal(self._read("psd", "swaps_accepted"), old["swaps_accepted"][:, :1])
        R.apply_plan(self.path, R.plan(self.path, {"psd": 5}), log=lambda *_: None)
        chain = self._read("psd", "chain")
        self.assertEqual(chain.shape, (ROWS, 5, NW, 1, 2))
        np.testing.assert_array_equal(chain[:, 4], old["chain"][:, 0])      # cycle over the 2 kept
        b = self._read("psd", "betas")
        self.assertEqual(b.shape, (ROWS, 5))
        self.assertEqual(b[0, 0], 1.0)
        self.assertTrue(np.all(np.diff(b[0]) < 0))                            # eryn make_ladder
        self.assertEqual(self._read("psd", "swaps_accepted").shape, (ROWS, 4))
        self.assertEqual(int(self._attr("psd", "ntemps")), 5)


class RefusalsTest(_Base):
    def test_banded_branch_unknown_branch_unknown_dataset(self):
        with self.assertRaises(ValueError):
            R.plan(self.path, {"gb": 8})
        with self.assertRaises(ValueError):
            R.plan(self.path, {"emri": 8})                                    # not in this store
        with h5py.File(self.path, "r+") as f:
            f["global_fit/sub_backend/mbh"].create_dataset("mystery_w", data=np.zeros((ROWS, 2)))
        with self.assertRaises(ValueError) as cm:
            R.plan(self.path, {"mbh": 5})
        self.assertIn("mystery_w", str(cm.exception))

    def test_unchanged_count_is_a_no_op(self):
        before = self._read("mbh", "chain")
        plans = R.plan(self.path, {"mbh": 2})
        self.assertTrue(plans[0].unchanged)
        R.apply_plan(self.path, plans, log=lambda *_: None)
        np.testing.assert_array_equal(self._read("mbh", "chain"), before)


class CLITest(_Base):
    def test_dry_run_writes_nothing_apply_writes_and_backs_up(self):
        import io
        from contextlib import redirect_stdout

        before = self._read("mbh", "chain")
        out = io.StringIO()
        with redirect_stdout(out):
            rc = R.main([self.path, "--set", "mbh=4", "--set", "psd=3"])
        self.assertEqual(rc, 0)
        self.assertIn("DRY RUN", out.getvalue())
        self.assertIn("psd: 3 -> 3 rung(s)   (UNCHANGED", out.getvalue())
        self.assertIn("MBH_NTEMPS=4", out.getvalue())
        np.testing.assert_array_equal(self._read("mbh", "chain"), before)
        out = io.StringIO()
        with redirect_stdout(out):
            rc = R.main([self.path, "--set", "mbh=4", "--apply"])
        self.assertEqual(rc, 0, out.getvalue())
        self.assertEqual(self._read("mbh", "chain").shape, (ROWS, 4, NW, 4, 11))
        self.assertIn("launch with: MBH_NTEMPS=4", out.getvalue())
        baks = [p for p in os.listdir(self.tmp) if ".pre_rerung-" in p]
        self.assertEqual(len(baks), 1)
        with h5py.File(os.path.join(self.tmp, baks[0]), "r") as f:
            self.assertEqual(f["global_fit/sub_backend/mbh/chain"].shape, (ROWS, 2, NW, 4, 11))

    def test_refusals_exit_2(self):
        import io
        from contextlib import redirect_stdout

        out = io.StringIO()
        with redirect_stdout(out):
            self.assertEqual(R.main([self.path, "--set", "gb=8"]), 2)
        self.assertIn("REFUSING", out.getvalue())


if __name__ == "__main__":
    unittest.main()
