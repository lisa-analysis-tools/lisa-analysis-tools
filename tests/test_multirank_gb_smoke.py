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
skips it (one child fewer). The control is asserted with a PLAIN loop, not
``subTest``: a red control must short-circuit, or the orchestrator arm still
runs and its meaningless comparison is reported beside it.

Every arm also ASSERTS that the ``lisatools`` it imported lives under this
worktree's ``src`` (and so does the parent, in ``setUpClass``, before anything
is spawned) -- printing the path was not enough, see
:func:`_worktree_import_problem`.

WHAT THIS GATE DOES **NOT** COVER: THE TEMPERING DRAWS (2026-09-16, round 5)
---------------------------------------------------------------------------
``GBSpecialBase._temper_rng`` -- the Generator whose single-rank derivation
fix round 5 unified between ``_propose_legacy`` and the orchestrator
(``gf_temper_seed_base``) -- has exactly one consumer: the per-repeat VERTICAL
band-temperature swap, off by default and armed with ``GB_TEMPER_VERTICAL=1``.
Arming it here was measured and rejected: one legacy arm went 59.8 s -> 63-67 s
(~10%), the census ran over 5 in-model blocks, and **0 of 1600 rows ever had a
vertical partner** -- at 4 walkers, 2 temperatures and ~11 alive leaves no
(walker, band) cell is ever co-resident at two adjacent rungs, so no swap is
proposed, ``_temper_rng.random`` is never called and the compared arrays carry
no draw from it. So the knob is NOT in ``SMOKE_ENV``; the stream is covered by
the unit tests (``tests/test_gb_rank_session.py::MakeTemperRngTest``) and, at
production scale, by the cluster gate in ``docs/multirank-cluster-gates.md``.
Each arm still reports what the sweep did (``_VertCounter`` -> the
``vert_*`` npz keys and the child's ``[gbsmoke-arm]`` line), so the number
above is re-measurable by setting ``GB_TEMPER_VERTICAL=1`` in the shell.

WHAT THIS GATE CAUGHT, AND THE FIX (2026-09-16)
-----------------------------------------------
The first run of this harness was RED with a GREEN control, and the numbers are
worth keeping because they are what named the defect:

* CONTROL (legacy vs legacy, two fresh interpreters, injected, 11 alive
  leaves): bit-identical on all five arrays, ``coords`` included. The fixture
  reproduces itself perfectly once each arm is its own process -- which retired
  the old "self-promotion" branch (the gate compares full ``coords`` outright).
* GATE (legacy vs orchestrator): ``inds``, ``band_temps`` and
  ``band_num_binaries`` bit-identical; ``coords`` differed in 390 of 400
  cold-chain leaves (389 DEAD slots plus one alive leaf) and ``log_like`` on
  1 of 4 cold walkers by 1.99e-13. On the EMPTY model: every physics array
  bit-identical and ALL 400 cold-chain (dead) leaves different.

The first reading -- that the two bodies consume the module-level ``np.random``
stream differently -- was WRONG, and RNG-state probes at nine matched
checkpoints in both bodies refuted it: the MT19937 position and key hash agree
exactly at every one of them (entry, after ``setup()``, after the sorter build,
after ``run_proposal``, after ``run_tempering``, after the write-back, at exit),
and so does eryn's ``model.random``. The real defect was a MERGE gap:

    a ``keep_all_inds`` ``BandSorter`` takes
    ``xp.asarray(gb_branch.coords.reshape(-1, ndim))`` as its coords, which on
    a CPU-resolved run is a VIEW of the branch array, so the rejected-birth
    fill it writes into the dead slots lands in the state ``_propose_legacy``
    returns. ``gb_finish`` exported only the ALIVE leaves, so the head's dead
    slots kept their pre-propose values.

Fixed by having ``gb_finish`` ship the block's whole branch
(``block_coords``/``block_inds``) and the head write ``work.coords[:, w0:w1]``
/ ``work.inds[:, w0:w1]`` per block (a neutral block ships ``None`` and keeps
what the head sliced). All five arrays -- full ``coords`` included -- are now
bit-identical on the empty model and with the live source.

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
import logging
import os
import re
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

#: also written to the npz, NOT compared: the vertical-swap coverage counters
#: (see :class:`_VertCounter`). They exist to prove the fixture reaches the
#: ``_temper_rng`` draws at all, not to be part of the parity claim.
COVERAGE_KEYS = ("vert_proposed", "vert_blocks")

#: repo root: the child is started as ``python -m tests.test_multirank_gb_smoke``
#: with this as its cwd, so ``tests`` imports from the worktree under test
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: the ``lisatools`` package THIS module's worktree owns. Derived from the test
#: file's own location, never from an env var, because it is exactly the env
#: that can be wrong (see :func:`_worktree_import_problem`).
SRC_ROOT = os.path.join(REPO_ROOT, "src")

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
    # NOT set here: ``GB_TEMPER_VERTICAL``. See the coverage note in the
    # module docstring -- arming it costs ~10% wall and still fires no swap
    # at this fixture size, so it buys the gate nothing. It stays tunable
    # from a shell (this map is applied with ``setdefault``) and the arms
    # report what the sweep did either way (``_VertCounter``).
}


class _VertCounter(logging.Handler):
    """Count vertical-swap proposals off the block-end ``[GB_VERT]`` census.

    The census is an INFO record on the ``lisatools`` logger, whose level
    ``globalfit.loginfo.init_logger`` sets to DEBUG during ``fit.build()``
    while leaving its managed console handler at WARNING -- so attaching this
    handler reads the line and adds no output. Counting records beats reading
    the run's log file: no path guessing, and it works whatever the run's
    verbosity knob does.
    """

    _PAT = re.compile(
        r"\[GB_VERT[^\]]*\].*\((\d+)/(\d+) rows had a partner.*"
        r"proposed (\d+) accepted (\d+)")

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.blocks = 0
        self.rows = 0
        self.paired = 0
        self.proposed = 0
        self.accepted = 0

    def emit(self, record):  # pragma: no cover - exercised only in the arms
        try:
            match = self._PAT.search(record.getMessage())
        except Exception:
            return
        if match is None:
            return
        self.blocks += 1
        self.paired += int(match.group(1))
        self.rows += int(match.group(2))
        self.proposed += int(match.group(3))
        self.accepted += int(match.group(4))


def _worktree_import_problem():
    """``None`` when ``lisatools`` came from THIS worktree, else the message.

    The three parity arms compare an orchestrator against a legacy body, and
    that only means anything if both are the ones in the tree under test.
    Printing the path was not enough: this repo has a documented mechanism
    that silently defeats it -- the deving env installs lisaanalysistools
    editable through scikit-build-core, whose redirecting finder hard-maps
    every ``lisatools.*`` module to the MAIN working tree, and a meta-path
    finder beats ``sys.path``, so ``PYTHONPATH`` alone cannot shadow it. Run
    without ``.wtenv/wt_run.sh`` (which exports ``LAT_WORKTREE_SRC`` and the
    ``sitecustomize.py`` that undoes the redirect) and both the parent and
    every child import the main checkout, compare it against itself and
    report GREEN for a tree that need not contain the change at all.
    """
    import lisatools

    got = os.path.realpath(lisatools.__file__)
    root = os.path.realpath(SRC_ROOT)
    if got.startswith(root + os.sep):
        return None
    return (f"imported {got}, expected under {root} -- run this module through "
            "`.wtenv/wt_run.sh $PWD/src <log> python -m unittest ...` so the "
            "worktree's lisatools wins over the editable install's redirecting "
            "finder; the arms would otherwise compare the MAIN checkout "
            "against itself")


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

    problem = _worktree_import_problem()
    if problem is not None:
        # the parent fails the arm on any non-zero return code and dumps this
        # child's log tail, so the reason lands in the test output
        raise SystemExit(f"{MARKER} child {args.arm}: {problem}")
    print(f"{MARKER} child {args.arm}: lisatools={lisatools.__file__} "
          f"orchestrate={args.orchestrate} inject={args.inject}", flush=True)
    counter = _VertCounter()
    logging.getLogger("lisatools").addHandler(counter)
    started = time.time()
    probe = _world_probe(1, os.path.join(args.store, args.arm), args.inject == "1")[0]
    probe["vert_proposed"] = np.int64(counter.proposed)
    probe["vert_blocks"] = np.int64(counter.blocks)
    np.savez(args.out, **{key: probe[key] for key in PARITY_KEYS + COVERAGE_KEYS})
    print(f"{MARKER} child {args.arm}: alive leaves={int(np.asarray(probe['inds']).sum())} "
          f"vert_swaps proposed={counter.proposed} accepted={counter.accepted} "
          f"(pairs {counter.paired}/{counter.rows} rows over {counter.blocks} block(s)) "
          f"wall={time.time() - started:.1f}s peak_rss={_rss_gb():.2f}GB", flush=True)


@unittest.skipUnless(RUN, "set RUN_GF_GB_SMOKE=1 to run the multi-rank GB smoke")
class MultiRankGBSmokeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # BEFORE anything is built or spawned: a run that is importing the
        # wrong tree must fail here, not produce a green comparison of the
        # main checkout against itself (the children inherit this env)
        problem = _worktree_import_problem()
        if problem is not None:
            raise AssertionError(f"tests.test_multirank_gb_smoke parent {problem}")

    def setUp(self):
        self._env0 = dict(os.environ)
        for key, value in SMOKE_ENV.items():
            os.environ.setdefault(key, value)
        self.tmpdir = tempfile.mkdtemp(prefix="gf_multirank_gb_")
        # the per-branch setup logs ignore file_store_dir and land in the
        # process's cwd; remember whether those directories were already there
        # so tearDown only removes ones THIS test created. Keyed on REPO_ROOT,
        # not ``os.getcwd()``: the arm children are spawned with
        # ``cwd=REPO_ROOT``, so that is where their strays land however the
        # parent was launched.
        self._strays = []
        for name in ("gf_output_gb_no_fg", "gf_output"):
            path = os.path.join(REPO_ROOT, name)
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
        # ``ru_maxrss`` over RUSAGE_CHILDREN is the max over ALL reaped
        # children, so it never decreases: this is the running high-water mark
        # of the arms so far, NOT this arm's own peak. The per-arm truth is the
        # child's own ``peak_rss=`` line, echoed just above.
        print(f"{MARKER} parent {subdir}: rc={returncode} wall={time.time() - started:.1f}s "
              f"children_peak_rss_max={_rss_gb(resource.RUSAGE_CHILDREN):.2f}GB", flush=True)
        if timed_out or returncode != 0:
            # the child ran with stderr merged into this log, so the tail
            # carries its traceback / SystemExit message verbatim
            tail = "\n".join(log.splitlines()[-40:])
            self.fail(f"parity arm {subdir!r} "
                      + ("TIMED OUT" if timed_out else f"failed (rc={returncode})")
                      + f"; child log tail (stdout+stderr):\n{tail}")
        with np.load(out_npz) as data:
            return {key: data[key] for key in PARITY_KEYS + COVERAGE_KEYS}

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
            # a PLAIN loop, not subTest: a red control must SHORT-CIRCUIT, or
            # the ~60 s orchestrator arm still runs and its (meaningless)
            # comparison is reported beside the control's failure as if the two
            # were independent findings
            for key in PARITY_KEYS:
                if not np.array_equal(ctl[key], legacy[key]):
                    self.fail(
                        f"CONTROL legacy-vs-legacy differs in {key}: the fixture is "
                        "non-reproducible here, so the orchestrator comparison would "
                        f"prove nothing (not run).\n    signature:\n{signature}")
        orch = self._run_arm_subprocess("orch", orchestrate=True, inject=True)
        # the gate only means something if the run really birthed sources: an
        # empty model never reaches the block merge or the write-back
        self.assertGreater(int(legacy["inds"].sum()), 0,
                           "no GB leaf survived the legacy arm: the parity gate would cover "
                           "only the neutral/early-return branches")
        # NOT asserted: a fired vertical swap. ``_temper_rng``'s draws stay
        # outside this gate at laptop scale -- see the coverage note in the
        # module docstring; the arms print what the sweep did.
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
