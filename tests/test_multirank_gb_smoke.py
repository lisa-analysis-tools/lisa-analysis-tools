"""GB fan-out end to end on the debug-preset gb_no_fg synthetic fit (fake communicator).

Gated by ``RUN_GF_GB_SMOKE=1`` (heavier than the noise/blank smokes). Two scenarios:
two compute ranks driving the three-command orchestrator, and the seeded PARITY of the
orchestrator at ONE compute rank (``GB_PROPOSE_ORCHESTRATE=1``, a direct-call
``WalkerFanout``) against ``_propose_legacy`` -- today's body.

BOTH SCENARIOS NOW CARRY A LIVE SOURCE
--------------------------------------
Both inject one in-band GB source (``general.gb_injection_params``), so the RJ
search really births leaves and the whole alive-source path -- birth, in-model
repeats, write-back, the head's block merge -- runs under real code. The parity
gate asserts the legacy arm ended with alive leaves, so a green gate provably
covers accepted RJ steps and not just the neutral/early-return branches.

ONE FIT PER PROCESS (the parity scenario), 2026-09-16
-----------------------------------------------------
The two-rank scenario runs in THIS process (its ranks are threads). The parity
scenario does not: each of its arms (legacy, control, orchestrator) runs in a
FRESH child interpreter via ``_run_arm_subprocess`` -> ``python -m
tests.test_multirank_gb_smoke --arm ...``, and the parent only loads the npz the
child writes. That is not a memory trick, it is what makes the fixture
comparable at all:

* 2026-09-16, round 2 (both arms in ONE process, injected): the PAIRED NEGATIVE
  CONTROL (legacy vs legacy, same seed, same process) diverged in ``log_like``
  on all 4 walkers, O(100) nats. Bisected to the STARTING state: ``start_inds``
  0/800 differ (the model starts empty), ``start_coords`` 3200/7200 differ =
  800 leaves x exactly 4 of the 9 GB columns -- ``phi0``, ``cos_iota``, ``psi``,
  ``fdot_astro_ratio``, the ones eryn's ``UniformDistribution`` draws from the
  MODULE-level ``np.random`` stream. The SECOND ``fit.build()`` in a process
  consumes a different NUMBER of draws from that stream than the first
  (per-process registries/caches skip work the second time), so re-seeding
  ``np.random`` before each arm -- which the fixture does -- cannot align them.
  The dead-leaf fill is harmless on its own, but the same stream feeds the birth
  containers' extrinsic columns during sampling, which is how it reaches
  ``log_like`` once leaves are alive.
* The same round measured that every arm is exactly reproducible ACROSS
  processes (run 1's legacy arm equals run 2's legacy arm to the last digit, and
  run 1's orchestrator arm equals run 2's control arm), i.e. no OS entropy is
  left anywhere in this path -- only the "first vs second build in one process"
  difference. One fit per process therefore gives every arm the identical stream
  position, and the gate can carry the injection.
* The other entropy sources were closed earlier the same day: the GB prior
  objects (``F0McGMMSampling`` via ``from_heatmap()``, the 3-D galaxy
  sky/distance prior) take per-rank sub-seeds derived from ``random_seed``
  (``communication/ranks.py::rank_build_seed`` -> ``GBSettings.build_seed`` ->
  ``GBSetup._build_sub_seeds``), and the RJ birth container is seeded per rank
  and per F-stat epoch (``build_gb_birth_distribution(seed=...)``, unit-tested in
  ``tests/test_fstat_birth_seed.py``).

WHAT THE PARITY GATE COMPARES
-----------------------------
``log_like``, the FULL ``coords`` array (dead-leaf fill included), ``inds``,
``band_temps`` and ``band_num_binaries``, bit for bit. ``_run_world`` exposes no
separate ``d_h``/``h_h`` probe; ``log_like`` is the likelihood-side check.
``GB_SMOKE_PARITY_CONTROL`` is ON by default and asserts the same arrays for a
legacy-vs-LEGACY pair FIRST -- an invariance claim without a paired negative
control proves nothing, and a red control means "this fixture does not
reproduce", never "the orchestrator diverges". ``GB_SMOKE_PARITY_CONTROL=0``
skips it (one child fewer).

MEASURED 2026-09-16, first run of this harness: the gate is RED, the control is
GREEN
-----------------------------------------------------------------------------
Recorded here because the numbers are the finding, not the harness:

* CONTROL (legacy vs legacy, two fresh interpreters, injected, 11 alive
  leaves): **bit-identical on all five arrays, ``coords`` included**. The
  fixture reproduces itself perfectly once each arm is its own process -- which
  is what this round set out to establish, and which retires the old
  "self-promotion" branch (the gate now compares full ``coords`` outright).
* GATE (legacy vs orchestrator, same harness): ``inds``, ``band_temps`` and
  ``band_num_binaries`` bit-identical; ``log_like`` differs on 1 of 4 cold
  walkers by 1.99e-13 (rel 2.3e-15); ``coords`` differs in 390 of 400 cold-chain
  leaves -- 389 DEAD slots plus ONE alive leaf (cold walker 1, leaf 1) whose 9
  columns are a different accepted birth. The hot chain is untouched.
* The orchestrated arm is internally deterministic: two orchestrated arms in
  separate processes agree bit for bit on all five arrays.
* On the EMPTY model (no injection) through the same harness: ``log_like``,
  ``inds``, ``band_temps``, ``band_num_binaries`` bit-identical, and ``coords``
  differs in ALL 400 cold-chain (dead) leaves.

Read together: the two bodies make the same decisions but do not consume the
module-level ``np.random`` proposal stream identically, so the rejected-birth
fill left behind in dead leaf slots differs wholesale even with nothing alive,
and on a live model that difference occasionally lands a different accepted
birth. Dead-leaf content is not part of the model (eryn ignores ``coords``
where ``inds`` is False), but the alive-leaf difference is real. Whether the
fix is to make ``_propose_orchestrated`` draw in the legacy order, or to
redefine parity on a live model, is a controller ruling -- so the gate is left
STRICT and red rather than quietly relaxed.

Memory: an 8 GB laptop is the budget. The parity arms cost the PARENT nothing
now (it builds no fit there; each child peaks ~3 GB and exits before the next
starts), so ``tearDown``'s ceiling is driven by the in-process two-rank
scenario. The fit is the ``gb_no_fg`` debug preset (3-day Tobs, tiny chunked-het
grids) with ``GB_DEBUG=0`` so the preset's *sizes* apply without arming the debug
instrumentation (round-trip verification + band plots), which is neither wanted
nor cheap here.
"""

import argparse
import gc
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

RUN = os.environ.get("RUN_GF_GB_SMOKE", "") not in ("", "0")

#: The paired negative control arm (legacy vs legacy, fresh interpreter each).
#: ON by default: now that every arm is its own child process it costs one more
#: sequential ~2 min child and no parent memory at all, which is cheap against
#: what it buys -- it is the only thing that separates "the orchestrator
#: diverges" from "this fixture does not reproduce". ``=0`` skips it.
CONTROL = os.environ.get("GB_SMOKE_PARITY_CONTROL", "1") not in ("", "0")

#: peak-RSS ceiling asserted in ``tearDown`` on THIS process. The parity arms
#: are child interpreters, so they no longer enter this number (the children's
#: own peak is read via ``RUSAGE_CHILDREN`` and printed per arm); what is left
#: here is the in-process two-rank world on an 8 GB laptop.
RSS_BUDGET_GB = 5.0

#: arrays the orchestrator must reproduce bit for bit at one compute rank, and
#: the exact set each arm's child writes into its npz
PARITY_KEYS = ("log_like", "coords", "inds", "band_temps", "band_num_binaries")

#: repo root: the child is started as ``python -m tests.test_multirank_gb_smoke``
#: with this as its cwd, so ``tests`` imports from the worktree under test
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: prefix the child prints its identity/measurement lines with; the parent
#: echoes exactly those lines into the test log (the child's full log lives in
#: the tmpdir and is dumped only when the arm fails)
MARKER = "[gbsmoke-arm]"

#: two compute ranks -> a 2-walker block each
NWALKERS = 4
ITERATIONS = 2
#: FakeWorld watchdog, and the per-arm subprocess timeout. Generous, but far
#: below the "walk away" hour: a deadlocked fan-out must fail the run, not hold
#: the laptop all night.
TIMEOUT_S = 1800.0
#: ``general.random_seed`` (eryn reseeds ``np.random`` from it in every
#: ``Move.__init__``) AND the seed this fixture sets before each world -- the run
#: draws its STARTING state (prior draws) in ``load_info``, which happens BEFORE
#: any move is constructed.
SEED = 4242

#: ONE in-band GB injection, GBGPU basis
#: ``[A, f0, fdot, fddot, phi0, iota, psi, lam, beta]``
#: (``injections.GB_INJECTION_PARAMS``' column order). ``f0 = 7.5 mHz`` sits
#: inside gb_no_fg's GB band [7.36, 7.78] mHz and ``A = 1e-21`` is loud enough at
#: the 3-day debug Tobs that the RJ search births leaves within the two
#: iterations -- the alive-source path was otherwise untested by real code
#: (whole-plan review, I2).
INJECTION = [[1e-21, 7.5e-3, 1e-16, 0.0, 1.2, 0.9, 1.0, 4.0, -0.6]]

#: seeded through ``os.environ.setdefault`` (an explicit value in the caller's
#: environment still wins, so the knobs stay tunable from a shell). The child
#: interpreters inherit the parent's copy AND re-apply this map, so a child
#: launched by hand behaves the same way.
SMOKE_ENV = {
    "USE_GPU": "0",
    "MAKE_DIAGNOSTIC_PLOTS": "0",
    # FakeWorld ranks are threads and the macOS GUI backend cannot draw off the
    # main thread. Best effort only: if matplotlib is already imported by the
    # time setUp runs, the setup heatmap still warns and skips -- harmless, and
    # the same warning appears in the noise smoke.
    "MPLBACKEND": "Agg",
    # MEMORY KNOB (the reason this test fits an 8 GB laptop). gb_no_fg's DATA
    # band is [6, 25] mHz while its GB band is [7.36, 7.78] mHz, so without the
    # clip every per-walker ACA slab carries ~45x the frequency range the GB
    # moves ever touch. 8 layers is comfortably above the 5 the chunked-het
    # gating needs. It is NOT the dominant term (one ``fit.build()`` costs
    # ~2.1 GB on its own), but it is the one knob this test owns.
    "DATA_BAND_LAYERS": "8",
    # the debug *preset* is armed by the ``debug=True`` kwarg below; this keeps
    # the debug INSTRUMENTATION (residual round-trip checks, band plots under
    # GB_DEBUG_DIR) off -- it is expensive and writes to the cwd
    "GB_DEBUG": "0",
}


def _rss_gb(who=resource.RUSAGE_SELF):
    ru = resource.getrusage(who).ru_maxrss
    return ru / (1024.0 ** 3) if ru > 1e7 else ru / (1024.0 ** 2)  # macOS bytes / linux KB


def _reclaim():
    """Drop one finished fit's resident footprint before building the next.

    A production process builds ONE fit, so nothing evicts
    ``GBSpecialBase._shared_buffer_caches`` -- a CLASS-level staged-buffer cache
    keyed by ``id(computation object)``. The two-rank scenario builds one fit per
    rank thread in THIS process, and without this the dead fits' buffers stay
    resident and the run is SIGKILLed on an 8 GB laptop -- measured. (The parity
    arms each build their one fit in a child interpreter that then exits, which
    is the only complete way to give the memory back.)
    """
    try:
        from lisatools.globalfit.moves.gbspecialstretch import GBSpecialBase

        GBSpecialBase._shared_buffer_caches.clear()
        GBSpecialBase._branch_propose_counts.clear()
    except Exception:  # pragma: no cover - never let cleanup fail a test
        pass
    gc.collect()


def _build_fit(store_dir, inject):
    """The ONE build recipe both the in-process and the child path use."""
    from lisatools.globalfit.stock import erebor

    fit = erebor.gb_no_fg(
        debug=True,  # 3-day Tobs, tiny chunked-het grids (sizes only)
        nwalkers=NWALKERS, ntemps=2, data_mode="synthetic",
        file_store_dir=store_dir, make_diagnostic_plots=False,
    )
    fit.general.num_iterations = ITERATIONS
    fit.general.random_seed = SEED
    if inject:
        fit.general.gb_injection_params = INJECTION
    return fit


def _world_probe(size, store_dir, inject):
    """Run ``size`` ranks over the fake world; return ``{rank: probe dict}``.

    Module level on purpose: the unittest path and the ``--arm`` child path must
    run byte-identical code (same seed, same build, same iteration count), so
    there is exactly one copy of it.
    """
    from lisatools.globalfit.communication.fakecomm import FakeWorld
    from lisatools.globalfit.communication.ranks import prepare_rank
    from lisatools.globalfit.run import GlobalFit

    def fn(rank, comm):
        fit = _build_fit(store_dir, inject)
        layout = prepare_rank(fit, comm)
        fit.build()
        gf = GlobalFit(fit, comm)
        gf.run_global_fit()
        out = {"role": layout.role_of(rank).value, "acs_rows": int(gf.acs.acs_total_entries)}
        if hasattr(gf, "compute_service"):
            out["served"] = gf.compute_service_served
            return out
        # ``run_global_fit`` throws away ``run_mcmc``'s return value, so
        # ``gf.state`` is still the STARTING state -- probing it would make every
        # assertion below vacuous. eryn keeps the last state for its
        # ``initial_state=None`` resume path; that in-memory object is exactly
        # what the moves produced (``get_last_sample()`` would round-trip it
        # through the HDF store instead).
        final = getattr(gf.sampler, "_previous_state", None)
        if final is None:  # pragma: no cover - eryn API fallback
            final = gf.sampler.get_last_sample()
        sub = final.sub_states["gb"]
        out["log_like"] = np.array(final.log_like[0], copy=True)
        out["coords"] = np.array(final.branches["gb"].coords, copy=True)
        out["inds"] = np.array(final.branches["gb"].inds, copy=True)
        out["band_temps"] = np.array(sub.band_info["band_temps"], copy=True)
        out["band_num_binaries"] = np.array(sub.band_info["band_num_binaries"], copy=True)
        return out

    # one shared global stream per world (FakeWorld ranks are threads)
    np.random.seed(SEED)
    return FakeWorld(size, timeout=TIMEOUT_S).run(fn)


def _parity_signature(actual, desired, alive):
    """One line per compared array: how much differs, and WHERE.

    A bare ``assert_array_equal`` on ``coords`` says "3510 of 7200 elements"
    and nothing else; the split that matters is alive leaves (real model
    content) vs dead slots (rejected-birth fill eryn ignores), because the two
    have completely different meanings for a port defect.
    """
    lines = []
    for key in PARITY_KEYS:
        x, y = np.asarray(actual[key]), np.asarray(desired[key])
        bad = int((x != y).sum())
        note = ""
        if bad and key == "coords" and x.shape[:-1] == alive.shape:
            leaf = (x != y).any(axis=-1)
            note = (f" -> {int((leaf & alive).sum())} of {int(alive.sum())} ALIVE leaves, "
                    f"{int((leaf & ~alive).sum())} of {int((~alive).sum())} dead")
        if bad and x.dtype.kind == "f":
            note += f" max|delta|={np.abs(x - y).max():.3e}"
        lines.append(f"      {key}: {bad}/{x.size} elements differ{note}")
    return "\n".join(lines)


def _arm_main(args):
    """``python -m tests.test_multirank_gb_smoke --arm ...``: ONE arm, ONE fit.

    Writes ``PARITY_KEYS`` to ``--out`` and prints its interpreter identity and
    peak RSS on ``MARKER`` lines the parent echoes.
    """
    for key, value in SMOKE_ENV.items():
        os.environ.setdefault(key, value)
    os.environ["GB_PROPOSE_ORCHESTRATE"] = args.orchestrate

    import lisatools

    print(f"{MARKER} child {args.arm}: lisatools={lisatools.__file__} "
          f"orchestrate={args.orchestrate} inject={args.inject}", flush=True)
    started = time.time()
    probe = _world_probe(1, os.path.join(args.store, args.arm), args.inject == "1")[0]
    np.savez(args.out, **{key: probe[key] for key in PARITY_KEYS})
    print(f"{MARKER} child {args.arm}: alive leaves={int(np.asarray(probe['inds']).sum())} "
          f"wall={time.time() - started:.1f}s peak_rss={_rss_gb():.2f}GB", flush=True)


@unittest.skipUnless(RUN, "set RUN_GF_GB_SMOKE=1 to run the multi-rank GB smoke")
class MultiRankGBSmokeTest(unittest.TestCase):

    def setUp(self):
        self._env0 = dict(os.environ)
        for key, value in SMOKE_ENV.items():
            os.environ.setdefault(key, value)
        self.tmpdir = tempfile.mkdtemp(prefix="gf_multirank_gb_")
        # the per-branch setup logs ignore file_store_dir and land in the cwd;
        # remember whether those directories were already there so tearDown only
        # removes ones THIS test created (the arm children share this cwd)
        self._strays = []
        for name in ("gf_output_gb_no_fg", "gf_output"):
            path = os.path.join(os.getcwd(), name)
            self._strays.append((path, os.path.isdir(path)))

    def tearDown(self):
        _reclaim()
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        for path, existed in self._strays:
            if not existed:
                shutil.rmtree(path, ignore_errors=True)
        # the stock variants seed env knobs with setdefault at construction
        # (the GB_DEBUG preset); do not leak them into the next test
        os.environ.clear()
        os.environ.update(self._env0)
        self.assertLess(_rss_gb(), RSS_BUDGET_GB,
                        f"GB smoke exceeded the {RSS_BUDGET_GB} GB budget")

    def _run_world(self, size, env=None, subdir=None, inject=False):
        """In-process world (two-rank scenario). Returns ``{rank: probe dict}``."""
        store = subdir if subdir is not None else f"n{size}"
        # set OUTSIDE the rank threads: FakeWorld ranks are threads of one
        # process and share os.environ, so a per-rank write would race
        saved = {k: os.environ.get(k) for k in (env or {})}
        os.environ.update(env or {})
        try:
            return _world_probe(size, os.path.join(self.tmpdir, store), inject)
        finally:
            _reclaim()
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _run_arm_subprocess(self, subdir, orchestrate, inject):
        """One single-rank arm in a FRESH interpreter; returns the probe dict.

        The parent builds NO fit here -- each arm's ~3 GB lives and dies in its
        own child, and (the point of the exercise) every arm is the FIRST
        ``fit.build()`` of its process, so they share the module-level
        ``np.random`` stream position (module docstring).
        """
        out_npz = os.path.join(self.tmpdir, f"{subdir}.npz")
        log_path = os.path.join(self.tmpdir, f"{subdir}.log")
        flag = "1" if orchestrate else "0"
        env = os.environ.copy()
        env["GB_PROPOSE_ORCHESTRATE"] = flag
        cmd = [
            sys.executable, "-m", "tests.test_multirank_gb_smoke",
            "--arm", subdir, "--orchestrate", flag, "--inject", "1" if inject else "0",
            "--out", out_npz, "--store", self.tmpdir,
        ]
        started = time.time()
        timed_out = False
        with open(log_path, "w") as handle:
            try:
                proc = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                      env=env, cwd=REPO_ROOT, check=False,
                                      timeout=TIMEOUT_S)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out, returncode = True, -1
        with open(log_path) as handle:
            log = handle.read()
        for line in log.splitlines():
            if MARKER in line:
                print(line, flush=True)
        print(f"{MARKER} parent {subdir}: rc={returncode} wall={time.time() - started:.1f}s "
              f"children_peak_rss={_rss_gb(resource.RUSAGE_CHILDREN):.2f}GB", flush=True)
        if timed_out or returncode != 0:
            tail = "\n".join(log.splitlines()[-40:])
            self.fail(f"parity arm {subdir!r} "
                      + ("TIMED OUT" if timed_out else f"failed (rc={returncode})")
                      + f"; child log tail:\n{tail}")
        with np.load(out_npz) as data:
            return {key: data[key] for key in PARITY_KEYS}

    def test_two_compute_ranks_run_the_gb_moves(self):
        out = self._run_world(2, inject=True)
        self.assertEqual((out[0]["role"], out[1]["role"]), ("head", "compute"))
        self.assertEqual((out[0]["acs_rows"], out[1]["acs_rows"]), (2, 2))
        # FLOOR, not the exact count: ping + the three commands of one GB
        # propose. A GB move can still take the whole-ensemble early return
        # (``rj_replace`` on a zero-leaf model) before issuing any command, so
        # the per-iteration command count is configuration-dependent.
        self.assertGreaterEqual(out[1]["served"], 1 + 3)
        self.assertTrue(np.all(np.isfinite(out[0]["log_like"])))
        self.assertEqual(out[0]["log_like"].shape, (NWALKERS,))
        # (num_bands, ntemps) ladder: betas non-increasing with temperature
        self.assertTrue(np.all(np.diff(out[0]["band_temps"], axis=-1) <= 0))
        # the injected source is found: the ALIVE-source path ran for real
        # (an empty model exercises only the neutral/early-return branches)
        inds = out[0]["inds"]
        self.assertGreater(int(inds.sum()), 0, "no GB leaf survived the run")
        # and BOTH walker blocks carry leaves on the cold chain -- the merge
        # writes each rank's block back into the head's full state, so a
        # block-shaped merge bug shows here and nowhere else
        half = NWALKERS // 2
        self.assertTrue(inds[0, :half].any(), "head block has no cold leaf")
        self.assertTrue(inds[0, half:].any(), "worker block has no cold leaf")

    def test_orchestrator_at_one_rank_matches_the_legacy_body(self):
        # WITH the injection: every arm is a fresh interpreter, so all three see
        # the identical module-level ``np.random`` stream position and the
        # fixture reproduces itself with alive leaves (module docstring).
        legacy = self._run_arm_subprocess("legacy", orchestrate=False, inject=True)
        if CONTROL:
            # paired negative control FIRST: a second legacy run, same seed,
            # fresh interpreter. Without it an invariance claim proves nothing --
            # it is what separates "the orchestrator diverges" from "this fit
            # does not reproduce", so it is asserted BEFORE the orchestrator
            # comparison rather than after it.
            ctl = self._run_arm_subprocess("control", orchestrate=False, inject=True)
            signature = _parity_signature(ctl, legacy, legacy["inds"])
            for key in PARITY_KEYS:
                with self.subTest(arm="control", key=key):
                    np.testing.assert_array_equal(
                        ctl[key], legacy[key],
                        err_msg=f"CONTROL legacy-vs-legacy differs in {key}: the fixture is "
                                "non-reproducible here, so the orchestrator comparison "
                                f"would prove nothing.\n    signature:\n{signature}")
        orch = self._run_arm_subprocess("orch", orchestrate=True, inject=True)
        # the gate only means something if the run really birthed sources: an
        # empty model never reaches the block merge or the write-back
        self.assertGreater(int(legacy["inds"].sum()), 0,
                           "no GB leaf survived the legacy arm: the parity gate would cover "
                           "only the neutral/early-return branches")
        # every key is compared (subTest) rather than stopping at the first:
        # which arrays survive is the whole diagnosis -- identical ``inds`` +
        # ``band_num_binaries`` with differing ``coords`` means "same decisions,
        # different draws", and the reverse would mean a real merge defect.
        signature = _parity_signature(orch, legacy, legacy["inds"])
        for key in PARITY_KEYS:
            with self.subTest(arm="orch", key=key):
                np.testing.assert_array_equal(
                    orch[key], legacy[key],
                    err_msg=f"orchestrator vs legacy differs in {key}\n"
                            f"    signature:\n{signature}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ONE single-rank parity arm in this interpreter and save its probe.")
    parser.add_argument("--arm", default=None,
                        help="arm name; doubles as the file_store_dir subdirectory")
    parser.add_argument("--orchestrate", default="0", choices=("0", "1"))
    parser.add_argument("--inject", default="1", choices=("0", "1"))
    parser.add_argument("--out", default=None, help="npz path to write")
    parser.add_argument("--store", default=None, help="parent tmpdir holding the arm stores")
    _args, _rest = parser.parse_known_args()
    if _args.arm is not None:
        if _args.out is None or _args.store is None:
            parser.error("--arm requires --out and --store")
        _arm_main(_args)
    else:
        unittest.main()
