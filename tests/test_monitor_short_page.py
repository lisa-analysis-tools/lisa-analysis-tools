"""The SHORT monitor page (``lisatools.globalfit.monitor._short``).

User request 2026-10-03: "let's make a short version of the html as well.
That picks out the most important plots and topline info. phase maximized
overlap. current psd measurements. nleaves over time, likelihood over time."

Pinned here:
  * it renders from a small synthetic run directory, with the topline fields
    and the section headings in order, as one self-contained, well-formed
    document;
  * it DEGRADES rather than fails when the head log, the GPU CSV or the truth
    set is absent;
  * the overlap section, given a truth set equal to the model, scores every
    pair at overlap 1 -- the real GBGPU path, end to end;
  * the generator's own helpers are compiled out of its syntax tree, not run,
    and the overlap loop it mirrors is executed against the generator's
    inline block so the two cannot drift;
  * the CLI flag, the in-run knob and from_tar select it, and the full page
    keeps its name.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import textwrap
import unittest
from datetime import datetime
from html.parser import HTMLParser
from unittest import mock

import h5py
import numpy as np

import lisatools.globalfit.monitor as mon
from lisatools.globalfit.monitor import _short as sh

NIT, CAP, NW, NLEAF = 12, 16, 4, 40
GATES = {6: "nudge", 9: "moved"}       # rows saved after the gated noise step
NOW = datetime(2026, 10, 3, 11, 12)
GALFOR0 = np.array([-43.84, -2.6, 5.0, -2.0, -2.85])      # log10 basis, alpha linear


def _make_run(root, name="gf_prod_test", *, log=True, csv=True):
    """A run directory the short page can read: the store plus, optionally,
    the head log and the head node's GPU CSV."""
    run = os.path.join(root, name)
    os.makedirs(run)
    rng = np.random.default_rng(7)
    ll = np.zeros((CAP, 1, 1, NW))
    for r in range(NIT):
        ll[r, 0, 0] = 1.0e8 + 1000.0 * r - (500.0 if r == 6 else 0.0) + rng.uniform(0, 5, NW)
    ll[NIT - 1, 0, 0, 0] += 50.0                           # walker 0 is the best at the end
    alive = np.zeros((CAP, 1, 1, NW, NLEAF), bool)
    for r in range(NIT):
        alive[r, 0, 0, :, :5 + 2 * r] = True
    chain = np.zeros((CAP, 1, 1, NW, NLEAF, 9))
    f0 = np.geomspace(1.2, 15.0, NLEAF) * (1.0 + 0.01 * np.arange(NLEAF) / NLEAF)
    base = np.column_stack([
        rng.uniform(0.5, 5.0, NLEAF), f0, rng.uniform(0.3, 0.8, NLEAF),
        rng.uniform(0, 2 * np.pi, NLEAF), rng.uniform(-0.9, 0.9, NLEAF),
        rng.uniform(0, np.pi, NLEAF), rng.uniform(0, 2 * np.pi, NLEAF),
        rng.uniform(-0.9, 0.9, NLEAF), rng.uniform(-2, 2, NLEAF)])
    chain[:NIT, 0, 0] = base[None, None]
    chain[~alive] = 0.0
    gal = np.zeros((CAP, 2, NW, 1, 5))
    psd = np.zeros((CAP, 2, NW, 1, 2))
    g_now = np.tile(GALFOR0, (NW, 1))
    p_now = np.column_stack([1.5e-11 * (1 + 1e-4 * rng.standard_normal(NW)),
                             3.0e-15 * (1 + 1e-3 * rng.standard_normal(NW))])
    for r in range(NIT):
        if r == 6:                                         # the forced step
            g_now = g_now + np.array([-0.05, -0.1, 0.0, 0.0, -0.15])
        elif r == 9:                                       # released and sampled
            g_now = g_now + 0.02 * rng.standard_normal((NW, 5))
            p_now = p_now * (1 + 1e-4 * rng.standard_normal((NW, 2)))
        gal[r, :, :, 0] = g_now
        psd[r, :, :, 0] = p_now
    saved = [("noise_ratchet_search" if r in GATES else
              ("in_model", "in_model_fstat", "in_model_removal")[r % 3]) for r in range(NIT)]
    with h5py.File(os.path.join(run, f"{name}_testing.h5"), "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = NIT
        g.create_dataset("log_like", data=ll)
        g.create_dataset("inds/gb", data=alive)
        g.create_dataset("chain/gb", data=chain)
        for b, n in (("vgb", 3), ("psd", 1), ("galfor", 1)):
            g.create_dataset(f"inds/{b}", data=np.ones((CAP, 1, 1, NW, n), bool))
        g.create_dataset("saved_after", data=np.array(saved, dtype="S32"))
        g.create_dataset("sub_backend/psd/chain", data=psd)
        g.create_dataset("sub_backend/galfor/chain", data=gal)
        g.create_group("noise_model_identity").attrs["galfor_log_sampling"] = True
        a = g.create_group("domain_settings").create_group("args").attrs
        a["0"], a["1"], a["2"] = 1440, 4320, 2.5
        rec = g.create_group("recipe")
        s1 = rec.create_group("stage_a").attrs
        s1["order num"], s1["status"], s1["completed_iteration"] = 1, True, 4
        s2 = rec.create_group("stage_b").attrs
        s2["order num"], s2["status"], s2["start_iteration"] = 2, False, 4
        s2["move_order"] = json.dumps(["noise_ratchet_search", "in_model",
                                       "in_model_fstat", "in_model_removal"])
        s3 = rec.create_group("stage_c").attrs             # stale stamp from a reset
        s3["order num"], s3["status"], s3["start_iteration"] = 3, False, 10
    if log:
        art = os.path.join(run, f"{name}_artifacts")
        os.makedirs(art)
        t = "2026-10-03 {} - lisatools.globalfit.{} - INFO - {}\n"
        with open(os.path.join(art, "globalfit_run.log"), "w") as fh:
            for hm in ("10:00:00,100", "10:30:00,100", "11:10:00,100"):
                fh.write(t.format(hm, "hdfbackend", "[SAVE] save_step 0.13 s (handoff to rank 4)"))
            fh.write(t.format("11:10:00,200", "recipe", "[GALFOR_RATCHET stage_b] iteration 3: "
                              "NUDGE (cycle 2 of 20) -- galfor shift [-0.05 -0.1 0 0 -0.15]"))
            fh.write(t.format("11:10:01,000", "recipe", "[GALFOR_RATCHET stage_b] after RELEASE 1: "
                              "foreground drop -3.20% (threshold 1.00%) at 3-5 mHz; continues"))
    if csv:
        with open(os.path.join(run, "gpu_util_123.csv"), "w") as fh:
            for s in (20, 40, 60):
                for i, u in ((0, 55), (1, 80)):
                    fh.write(f"2026/10/03 11:09:{s % 60:02d}.000, {i}, NVIDIA H100 NVL, {u}, 0, "
                             f"30000, 95830, 120.5, 45\n")
    return run


class _WellFormed(HTMLParser):
    VOID = {"meta", "img", "br", "hr", "input", "link", "area", "base", "col",
            "embed", "source", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self.stack, self.errors = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.errors.append((tag, self.stack[-3:]))


class _Cwd:
    """Run from an empty directory: the truth lookup also searches the CWD."""

    def setUp(self):
        self._cwd = os.getcwd()
        self.d = tempfile.mkdtemp()
        os.chdir(self.d)

    def tearDown(self):
        os.chdir(self._cwd)
        shutil.rmtree(self.d, ignore_errors=True)


class ShortPageRenderTest(unittest.TestCase):
    """One render of the full fixture, many readings of it."""

    @classmethod
    def setUpClass(cls):
        # erebor's import chain restyles matplotlib (eryn); take that hit
        # BEFORE snapshotting, so the rcParams test measures this module only
        import matplotlib

        import lisatools.globalfit.stock.erebor.noise  # noqa: F401

        cls._cwd = os.getcwd()
        cls.d = tempfile.mkdtemp()
        os.chdir(cls.d)
        cls.rundir = _make_run(cls.d)
        cls.rc_before = dict(matplotlib.rcParams)
        cls.path = mon.build_short_monitor(cls.rundir, now=NOW)
        cls.rc_after = dict(matplotlib.rcParams)
        with open(cls.path, encoding="utf-8") as fh:
            cls.html = fh.read()

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._cwd)
        shutil.rmtree(cls.d, ignore_errors=True)

    def test_it_lands_beside_the_run_under_its_own_name(self):
        self.assertEqual(self.path, os.path.abspath(self.rundir) + "_monitor_short.html")
        self.assertNotEqual(self.path, mon.default_out_path(self.rundir))
        self.assertFalse(os.path.exists(mon.default_out_path(self.rundir)),
                         "the short page must never write the full page's file")

    def test_the_sections_are_there_in_order(self):
        heads = ["<h2>Topline</h2>", "<h2>Log-likelihood</h2>", "<h2>GB Leaf Count</h2>",
                 "<h2>Phase-Maximised Overlap</h2>", "<h2>Noise: Current Measurement</h2>",
                 "<h2>Ratchet Gates</h2>", "<h2>Notes</h2>"]
        pos = [self.html.index(h) for h in heads]
        self.assertEqual(pos, sorted(pos))

    def test_the_topline_carries_every_requested_field(self):
        top = self.html[self.html.index('<section id="topline">'):
                        self.html.index('<section id="lnl">')]
        for want in ("last saved row", f"<b>{NIT - 1}</b>", "saved after",
                     "<code>in_model_removal</code>",         # row 11: 11 % 3 = 2
                     "next leg <code>noise_ratchet_search</code>",   # the cycle wraps
                     "<b>stage_b</b>", "step 2 of 3", "started row 4, 8 rows in",
                     "galfor ratchet", "iteration 3: NUDGE (cycle 2 of 20)",
                     "foreground drop -3.20% vs 1.00%",
                     "row cadence: last 40 min, median of 2 = 40 min",
                     "gpu (head node)", "GPU0 util 55%", "GPU1 util 80%",
                     "head node only", "29.3/93.6 GiB",
                     f"cold lnL, row {NIT - 1}", "change vs row", "spread",
                     "cold leaves", f"gb min <b>{5 + 2 * (NIT - 1)}</b>",
                     "fixed-size branches", "vgb 3", "instrument noise"):
            self.assertIn(want, top, want)

    def test_the_lnl_and_leaf_panels_are_embedded_images(self):
        for alt in ("cold log-likelihood", "GB leaf count", "current sensitivity per walker",
                    "instrument-noise posteriors", "noise parameters vs row"):
            self.assertRegex(self.html, rf'<img src="data:image/png;base64,[A-Za-z0-9+/=]{{200,}}" '
                                        rf'alt="{alt}">', alt)

    def test_no_truth_set_degrades_the_overlap_section(self):
        sec = self.html[self.html.index('<section id="overlap">'):
                        self.html.index('<section id="noise">')]
        self.assertIn('class="missing"', sec)
        self.assertIn("No usable truth set", sec)
        self.assertIn("--build-truth", sec)

    def test_the_ratchet_gates_are_read_off_the_store(self):
        sec = self.html[self.html.index('<section id="ratchet">'):
                        self.html.index('<section id="notes">')]
        self.assertRegex(sec, r"<td>6</td><td class='l'[^>]*>nudge</td>")
        self.assertRegex(sec, r"<td>9</td><td class='l'[^>]*>moved</td>")

    def test_it_is_one_well_formed_self_contained_document(self):
        self.assertTrue(self.html.startswith("<!doctype html>"))
        p = _WellFormed()
        p.feed(self.html)
        p.close()
        self.assertEqual(p.errors, [])
        self.assertEqual(p.stack, [])
        for bad in ('src="http', "href=\"http", "<script src", "<link "):
            self.assertNotIn(bad, self.html)

    def test_the_callers_matplotlib_style_is_left_alone(self):
        # it runs on the saver rank: rcParams are set inside rc_context only
        changed = {k for k in self.rc_before if k != "backend"   # resolved lazily
                   and str(self.rc_before[k]) != str(self.rc_after.get(k))}
        self.assertEqual(changed, set())

    def test_it_is_small(self):
        self.assertLess(os.path.getsize(self.path), 3 * 1024 * 1024)


class ShortPageDegradesTest(_Cwd, unittest.TestCase):

    def test_no_log_and_no_csv_still_renders(self):
        run = _make_run(self.d, log=False, csv=False)
        html = sh.render_short_page(run, now=NOW)
        self.assertIn("n/a (no head log or job stdout in the run dir)", html)
        self.assertIn("n/a (no gpu_util_*.csv in the run dir)", html)
        notes = html[html.index('<section id="notes">'):]
        self.assertIn("no head log", notes)
        self.assertIn("no gpu_util_*.csv", notes)
        self.assertIn('alt="cold log-likelihood"', html)

    def test_a_failed_build_leaves_the_previous_page_alone(self):
        out = os.path.join(self.d, "page.html")
        with open(out, "w") as fh:
            fh.write("previous")
        empty = os.path.join(self.d, "empty_run")
        os.makedirs(empty)
        self.assertIsNone(sh.build_short_monitor(empty, out, check=False))
        with open(out) as fh:
            self.assertEqual(fh.read(), "previous")
        self.assertFalse(os.path.exists(out + ".tmp"))

    def test_match_stats_knob_suppresses_the_overlap_section(self):
        run = _make_run(self.d, log=False, csv=False)
        with mock.patch.dict(os.environ, {"GF_MONITOR_MATCH_STATS": "0"}):
            html = sh.render_short_page(run, now=NOW)
        self.assertIn("SUPPRESSED (GF_MONITOR_MATCH_STATS=0)", html)


class ShortPageOverlapTest(_Cwd, unittest.TestCase):
    """The real waveform path: a truth set that IS the model scores 1."""

    def test_a_truth_set_equal_to_the_model_matches_everything_at_overlap_one(self):
        from lisatools.globalfit.stock.erebor.transforms import make_gb_transform_container

        run = _make_run(self.d, log=False, csv=False)
        store = os.path.join(run, "gf_prod_test_testing.h5")
        with h5py.File(store, "r") as f:
            al = f["global_fit/inds/gb"][NIT - 1, 0, 0, 0]
            rec = f["global_fit/chain/gb"][NIT - 1, 0, 0, 0][al]
        tc = make_gb_transform_container(use_chirp_mass=True, use_fdot_astro=True,
                                         use_distance=True, mc_lims=(0.001, 1.0))
        phys = tc.both_transforms(np.asarray(rec, float).copy())
        np.savez(os.path.join(run, mon.TRUTH_NAME), f0=phys[:, 1], amp=phys[:, 0],
                 snr=np.full(len(phys), 20.0), det=np.ones(len(phys), bool), phys=phys,
                 band=np.array([0.8e-3, 21.94e-3]), tobs=1440 * 4320 * 2.5,
                 orbits="mojito_l1")
        with mock.patch.object(sh, "_l1_orbits_from", return_value=(None, "test: none")):
            html = sh.render_short_page(run, now=NOW)
        sec = html[html.index('<section id="overlap">'):html.index('<section id="noise">')]
        n = int(al.sum())
        self.assertIn(f"<b>{n:,}</b><span>matched, overlap &ge; 0.80</span>", sec)
        self.assertIn("<b>100.0%</b><span>completeness", sec)
        self.assertIn("<b>1.000</b><span>median overlap</span>", sec)
        self.assertRegex(sec, r'<img src="data:image/png;base64,[^"]{200,}" '
                              r'alt="phase-maximised overlap">')
        self.assertIn("analytic DefaultOrbits", sec)


class PhaseMaxOverlapTest(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.nw = 32
        self.a = rng.standard_normal(self.nw) + 1j * rng.standard_normal(self.nw)
        self.e = rng.standard_normal(self.nw) + 1j * rng.standard_normal(self.nw)
        self.SA = rng.uniform(1.0, 2.0, 400)
        self.SE = rng.uniform(1.0, 2.0, 400)

    def _mm(self, a2, e2, s1, s2):
        return sh.phase_max_overlaps(self.a[None], self.e[None], np.array([s1]),
                                     a2[None], e2[None], np.array([s2]),
                                     self.SA, self.SE, self.nw)[0]

    def test_identical_waveforms_overlap_one(self):
        self.assertAlmostEqual(self._mm(self.a, self.e, 10, 10), 1.0, places=12)

    def test_an_overall_phase_and_amplitude_are_maximised_away(self):
        z = 3.0 * np.exp(1j * 0.7)
        self.assertAlmostEqual(self._mm(z * self.a, z * self.e, 10, 10), 1.0, places=12)

    def test_disjoint_windows_overlap_zero(self):
        self.assertAlmostEqual(self._mm(self.a, self.e, 10, 10 + self.nw), 0.0, places=12)

    def test_a_window_off_the_grid_scores_zero(self):
        self.assertEqual(self._mm(self.a, self.e, 390, 390), 0.0)

    def test_it_is_the_generators_own_loop(self):
        """Execute the generator's inline block on the same inputs."""
        with open(mon.generator_path(), encoding="utf-8") as fh:
            src = fh.read()
        head, tail = "        MM = np.zeros(MI.size)\n", "        MM = np.clip(MM, 0.0, 1.0)\n"
        block = textwrap.dedent(src[src.index(head):src.index(tail) + len(tail)])
        rng = np.random.default_rng(5)
        n, nw = 25, self.nw
        Ar, Er, At, Et = (rng.standard_normal((n, nw)) + 1j * rng.standard_normal((n, nw))
                          for _ in range(4))
        sr = rng.integers(0, 300, n)
        st = np.clip(sr + rng.integers(-40, 40, n), 0, None)
        st[0], sr[1] = 395, -3                                 # both skip paths
        ns = dict(np=np, MI=np.arange(n), _Ar=Ar, _Er=Er, _sr=sr, _At=At, _Et=Et,
                  _st=st, NW_=nw, SA_G=self.SA, SE_G=self.SE)
        exec(block, ns)                                        # noqa: S102
        got = sh.phase_max_overlaps(Ar, Er, sr, At, Et, st, self.SA, self.SE, nw)
        np.testing.assert_array_equal(got, ns["MM"])
        self.assertTrue(np.any(got > 0))


class GeneratorAdapterTest(unittest.TestCase):

    def test_the_named_helpers_are_compiled_and_nothing_else_runs(self):
        ns = sh._generator_functions({"np": np})
        for name in sh.GEN_HELPERS:
            self.assertTrue(callable(ns.get(name)), name)
        for script_global in ("RUN_DIR", "OUT", "IMGS", "f", "g", "html"):
            self.assertNotIn(script_global, ns)
        mi, ti, d = ns["_match_pairs"](np.array([1.0, 2.0]), np.array([2.05, 0.98]), 0.1)
        self.assertEqual(list(mi), [0, 1])
        self.assertEqual(list(ti), [1, 0])

    def test_a_missing_helper_is_an_error_not_a_silent_gap(self):
        with self.assertRaisesRegex(RuntimeError, "no longer defines"):
            sh._generator_functions({"np": np}, names=("no_such_helper",))

    def test_stage_bounds_skip_stale_stamps_of_steps_not_reached(self):
        steps = [dict(name="a", order=1, status=True, start=None, completed=4),
                 dict(name="b", order=2, status=False, start=4, completed=9),
                 dict(name="c", order=3, status=False, start=10, completed=None)]
        self.assertEqual(sh.stage_bounds(steps), [(4, "a done -> b")])

    def test_verdict_phrases(self):
        self.assertEqual(sh.verdict_phrase(
            "after iteration 2 (RELEASE): cold-mean galfor [1 2]; no pre-nudge reference "
            "captured yet"), "after iteration 2 (RELEASE): no pre-nudge reference yet")
        self.assertEqual(sh.verdict_phrase("after RELEASE 0: recorded as the reference"),
                         "after RELEASE 0: reference recorded")


class ShortPageCliTest(unittest.TestCase):

    def setUp(self):
        import lisatools.globalfit.monitor.__main__ as m

        self.m = m
        self.d = tempfile.mkdtemp()
        self.rundir = os.path.join(self.d, "gf_prod_run")
        os.makedirs(self.rundir)
        self.addCleanup(shutil.rmtree, self.d, True)

    def test_the_flag_builds_the_short_page_INSTEAD_of_the_full_one(self):
        with mock.patch.object(self.m, "build_monitor") as bm, \
                mock.patch.object(self.m, "build_short_monitor") as bs:
            self.assertEqual(self.m.main(["--short-page", self.rundir]), 0)
        bm.assert_not_called()
        bs.assert_called_once()
        self.assertEqual(bs.call_args[0][1], os.path.abspath(self.rundir) + "_monitor_short.html")

    def test_without_the_flag_the_full_page_is_unchanged(self):
        with mock.patch.object(self.m, "build_monitor") as bm, \
                mock.patch.object(self.m, "build_short_monitor") as bs:
            self.m.main([self.rundir])
        bs.assert_not_called()
        self.assertEqual(bm.call_args[0][1], os.path.abspath(self.rundir) + "_monitor.html")

    def test_an_explicit_out_path_is_honoured(self):
        out = os.path.join(self.d, "s.html")
        with mock.patch.object(self.m, "build_short_monitor") as bs:
            self.m.main(["--short-page", self.rundir, out])
        self.assertEqual(bs.call_args[0][1], out)

    def test_it_composes_with_snapshot_but_not_snapshot_only(self):
        with mock.patch.object(self.m, "build_short_monitor") as bs, \
                mock.patch.object(self.m, "build_snapshot", return_value="/t.tar.gz") as bt:
            self.assertEqual(self.m.main(["--short-page", "--snapshot", self.rundir]), 0)
        bs.assert_called_once()
        bt.assert_called_once()
        with mock.patch.object(self.m, "build_short_monitor"), \
                contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit) as cm:
            self.m.main(["--short-page", "--snapshot-only", self.rundir])
        self.assertEqual(cm.exception.code, 2)

    def test_from_tar_passes_it_through(self):
        from lisatools.globalfit.monitor import from_tar as ft

        _make_run(self.d, "gf_prod_tar")
        tar = os.path.join(self.d, "snap.tar.gz")
        open(tar, "w").close()
        with mock.patch.object(mon, "build_short_monitor") as bs, \
                mock.patch.object(mon, "build_monitor") as bm, \
                mock.patch.object(ft.os.path, "getsize", return_value=1), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            rc = ft.main([tar, "--short-page", "--run-dir", os.path.join(self.d, "gf_prod_tar")])
        self.assertEqual(rc, 0)
        bm.assert_not_called()
        self.assertTrue(bs.call_args[0][1].endswith("gf_prod_tar_monitor_short.html"))


class ShortPageHookTest(unittest.TestCase):
    """GF_MONITOR_PAGE_SHORT=1 swaps the page the saver rank builds."""

    class _Comm:
        def __init__(self, payloads):
            self._p = list(payloads)

        def recv(self, source=None):
            return self._p.pop(0)

        def iprobe(self, source=None):
            return False

    class _Reader:
        filename = "/run/dir/gf_prod.h5"

        def save_step_main(self, *a, **k):
            pass

    def _loop(self, **env):
        from lisatools.globalfit import hdfbackend as hb

        base = {"GF_MONITOR_AFTER_SAVE": "1", "GF_MONITOR_SNAPSHOT": "0"}
        with mock.patch.dict(os.environ, {**base, **env}), \
                mock.patch.object(hb, "_atomic_backup_copy"), \
                mock.patch.object(mon, "build_monitor") as bm, \
                mock.patch.object(mon, "build_short_monitor") as bs:
            for k in ("GF_MONITOR_PAGE", "GF_MONITOR_PAGE_SHORT", "GF_MONITOR_ITER"):
                if k not in env:
                    os.environ.pop(k, None)
            hb.save_to_backend_asynchronously_and_plot(
                self._Reader(), self._Comm([{"save_args": (), "save_kwargs": {}},
                                            {"finish_run": True}]),
                main_rank=0, plot_container=None)
        return bm, bs

    def test_the_knob_builds_the_short_page_instead(self):
        bm, bs = self._loop(GF_MONITOR_PAGE_SHORT="1")
        bm.assert_not_called()
        bs.assert_called_once()
        self.assertEqual(bs.call_args[0][0], "/run/dir")

    def test_the_full_page_stays_the_default(self):
        bm, bs = self._loop()
        bm.assert_called_once()
        bs.assert_not_called()

    def test_page_off_turns_the_short_page_off_too(self):
        bm, bs = self._loop(GF_MONITOR_PAGE_SHORT="1", GF_MONITOR_PAGE="0")
        bm.assert_not_called()
        bs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
