"""Every store write of a running job goes through the SAVER rank.

USER RULING 2026-10-07 (Mike): "In general, let's make sure all file saves
go through the saver rank."

WHY. The head hands each sample row to the dedicated saver rank
(``GFHDFBackend.save_step`` -> ``save_to_backend_asynchronously_and_plot``),
but the recipe's stage-boundary stamps (``completed_recipe_step``), the
``GF_PERSIST_STAGE_START`` stamp (``stamp_stage_start``) and the galfor
ratchet's stop flag (``stamp_stage_flag``) were written by the HEAD, which
opened the store in append mode itself. When one coincided with the saver
writing a row, two processes wrote one HDF5 file and the recipe group was
torn: 3mo 2026-09-28, and 6mo replica_pe -> full_pe 2026-10-04 (salvaged with
scripts/fstat_proposal/salvage_recipe_group.py).

THE CONTRACT pinned here:

* with a saver rank (communicator size >= 3, the same test ``save_step``
  uses) the head sends those writes to the saver as a ``store_write``
  request and NEVER opens the store for writing itself;
* the saver runs them IN ARRIVAL ORDER with the rows, so a stage stamp lands
  after the row it closes and before the next one, and acknowledges each;
* the head returns only after the acknowledgement, so a read right after a
  routed write (``stage_start_iteration`` in the next stage's
  ``setup_run``) sees it;
* a write that fails on the saver is logged there with its method name and
  arguments, and re-raised on the head (exactly where the direct write
  would have raised) -- never silently dropped;
* without a saver (single process, ``fit.sample()``, size < 3) the same
  methods write directly, as before.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from unittest import mock

import h5py

from lisatools.globalfit import hdfbackend as hb
from lisatools.globalfit.hdfbackend import GFHDFBackend

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAD, SAVER = 0, 2
STEPS = ("gb_search_2", "gb_search_3", "full_pe")


def _store(tmp, iteration=0):
    """A minimal store: the run group, its row counter, a three-step recipe."""
    path = os.path.join(tmp, "gf_prod_parameter_estimation_main.h5")
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = iteration
        r = g.create_group("recipe")
        for i, n in enumerate(STEPS):
            s = r.create_group(n)
            s.attrs["status"] = False
            s.attrs["order num"] = i + 1
    return path


class _Pipe:
    """An in-process MPI stand-in for one head <-> saver pair.

    head -> saver messages share ONE queue (MPI's non-overtaking order for a
    receive on any tag); saver -> head messages are queued per tag.
    """

    def __init__(self):
        self.to_saver = queue.Queue()
        self._to_head = {}
        self._lock = threading.Lock()

    def to_head(self, tag):
        with self._lock:
            return self._to_head.setdefault(tag, queue.Queue())


class _End:
    def __init__(self, pipe, rank, size=3):
        self.pipe, self.rank, self.size = pipe, rank, size
        self.sent = []

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank

    def send(self, obj, dest, tag=0):
        self.sent.append((dest, tag, obj))
        if dest == SAVER:
            self.pipe.to_saver.put(obj)
        else:
            self.pipe.to_head(tag).put(obj)

    def recv(self, source=None, tag=None):
        if self.rank == SAVER:
            return self.pipe.to_saver.get(timeout=30)
        return self.pipe.to_head(tag).get(timeout=30)

    def iprobe(self, source=None, tag=None):
        return not self.pipe.to_saver.empty()


class _HeadSpy(GFHDFBackend):
    """The head's backend, recording every mode it opens the store in."""

    def open(self, mode="r"):
        self.__dict__.setdefault("modes", []).append(mode)
        return super().open(mode)


class _RowSaver(GFHDFBackend):
    """The saver's backend with a stand-in row write (one row = iteration+1).

    ``row_delay`` makes each row slow, so a head that did NOT wait for the
    acknowledgement would read the store before the stamp landed.
    """

    def __init__(self, *a, log=None, row_delay=0.0, **k):
        super().__init__(*a, **k)
        self.log = log if log is not None else []
        self.row_delay = row_delay

    def save_step_main(self, tag, *a, **k):
        time.sleep(self.row_delay)
        with self.open("a") as f:
            g = f[self.name]
            g.attrs["iteration"] = int(g.attrs["iteration"]) + 1
        self.log.append(("row", tag))

    def completed_recipe_step(self, *a, **k):
        self.log.append(("completed_recipe_step",) + a)
        return super().completed_recipe_step(*a, **k)

    def stamp_stage_start(self, *a, **k):
        self.log.append(("stamp_stage_start",) + a)
        return super().stamp_stage_start(*a, **k)

    def stamp_stage_flag(self, *a, **k):
        self.log.append(("stamp_stage_flag",) + a)
        return super().stamp_stage_flag(*a, **k)


class _Harness:
    """A head backend and a real saver loop on a thread, over one ``_Pipe``."""

    def __init__(self, path, row_delay=0.0):
        self.pipe = _Pipe()
        self.head_comm = _End(self.pipe, HEAD)
        self.saver_comm = _End(self.pipe, SAVER)
        self.head = _HeadSpy(path, comm=self.head_comm, save_plot_rank=SAVER)
        self.head.__dict__["modes"] = []
        self.log = []
        self.saver = _RowSaver(path, log=self.log, row_delay=row_delay)
        self.errors = []
        self.thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self):
        try:
            hb.save_to_backend_asynchronously_and_plot(
                self.saver, self.saver_comm, main_rank=HEAD, plot_container=None)
        except BaseException as e:  # noqa: BLE001 -- surfaced by finish()
            self.errors.append(e)

    def start(self):
        self.thread.start()
        return self

    def finish(self):
        self.head_comm.send({"finish_run": True}, dest=SAVER)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise AssertionError("the saver loop did not finish")
        if self.errors:
            raise self.errors[0]


class _QuietSaverEnv(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        for k in ("GF_MONITOR_AFTER_SAVE", "GF_MONITOR_RANK"):
            os.environ.pop(k, None)
        p = mock.patch.object(hb, "_atomic_backup_copy")
        p.start()
        self.addCleanup(p.stop)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)


class RoutedWriteTest(_QuietSaverEnv):
    """With a saver rank the head never opens the store for writing."""

    def test_head_never_opens_the_store_for_writing_with_a_saver(self):
        path = _store(self.tmp)
        h = _Harness(path).start()
        h.head.save_step("r1")
        h.head.completed_recipe_step("gb_search_2", next_step_name="gb_search_3")
        h.head.stamp_stage_start("gb_search_3", 1)
        self.assertTrue(h.head.stamp_stage_flag("gb_search_3", "galfor_ratchet_done", 1))
        h.head.save_step("r2")
        h.finish()
        self.assertEqual([m for m in h.head.modes if m != "r"], [],
                         "the head opened the store for writing")
        # every write ran on the saver, and landed
        self.assertEqual([e[0] for e in h.log],
                         ["row", "completed_recipe_step", "stamp_stage_start",
                          "stamp_stage_flag", "row"])
        with h5py.File(path, "r") as f:
            r = f["global_fit/recipe"]
            self.assertTrue(bool(r["gb_search_2"].attrs["status"]))
            self.assertEqual(int(r["gb_search_2"].attrs["completed_iteration"]), 1)
            self.assertEqual(int(r["gb_search_3"].attrs["start_iteration"]), 1)
            self.assertEqual(int(r["gb_search_3"].attrs["galfor_ratchet_done"]), 1)
            self.assertEqual(int(f["global_fit"].attrs["iteration"]), 2)

    def test_the_request_is_a_small_named_payload_on_the_row_channel(self):
        """The routed request rides the SAME head -> saver channel (default
        tag) as the rows, which is what orders it; the ack comes back on its
        own tag so it can never be mistaken for anything else."""
        path = _store(self.tmp)
        h = _Harness(path).start()
        h.head.completed_recipe_step("gb_search_2", next_step_name="gb_search_3")
        h.finish()
        req = [(d, t, o) for d, t, o in h.head_comm.sent
               if isinstance(o, dict) and hb.STORE_WRITE_KEY in o]
        self.assertEqual(len(req), 1)
        dest, tag, obj = req[0]
        self.assertEqual((dest, tag), (SAVER, 0))
        self.assertEqual(obj[hb.STORE_WRITE_KEY], "completed_recipe_step")
        self.assertEqual(tuple(obj["args"]), ("gb_search_2",))
        self.assertEqual(obj["kwargs"], {"next_step_name": "gb_search_3"})
        acks = [(d, t) for d, t, _o in h.saver_comm.sent]
        self.assertEqual(acks, [(HEAD, hb.STORE_WRITE_ACK_TAG)])


class OrderingTest(_QuietSaverEnv):
    def test_stamp_lands_between_the_rows_it_separates(self):
        """Two rows, the boundary, one row: the stamp says 2 (it landed after
        both rows of the closing stage and before the next stage's first)."""
        path = _store(self.tmp)
        h = _Harness(path, row_delay=0.2).start()
        h.head.save_step("r1")
        h.head.save_step("r2")
        h.head.completed_recipe_step("gb_search_2", next_step_name="gb_search_3")
        h.head.save_step("r3")
        h.finish()
        self.assertEqual(h.log, [("row", "r1"), ("row", "r2"),
                                 ("completed_recipe_step", "gb_search_2"),
                                 ("row", "r3")])
        with h5py.File(path, "r") as f:
            r = f["global_fit/recipe"]
            self.assertEqual(int(r["gb_search_2"].attrs["completed_iteration"]), 2)
            self.assertEqual(int(r["gb_search_3"].attrs["start_iteration"]), 2)
            self.assertEqual(int(f["global_fit"].attrs["iteration"]), 3)

    def test_a_read_right_after_a_routed_write_sees_it(self):
        """The next stage's setup_run reads stage_start_iteration (and the
        live iteration) right after the boundary stamp. The rows are SLOW
        here: a head that returned before the saver's acknowledgement would
        read None / 0."""
        path = _store(self.tmp)
        h = _Harness(path, row_delay=0.3).start()
        h.head.save_step("r1")
        h.head.save_step("r2")
        h.head.completed_recipe_step("gb_search_2", next_step_name="gb_search_3")
        got_start = h.head.stage_start_iteration("gb_search_3")
        got_iter = int(h.head.iteration)
        h.finish()
        self.assertEqual(got_start, 2)
        self.assertEqual(got_iter, 2)


class _ScriptedComm:
    """The saver's side only: scripted payloads, every iprobe drains."""

    def __init__(self, payloads, drain=True):
        self._payloads = list(payloads)
        self.drain = drain        # False: every recv is a batch of its own
        self.sent = []

    def recv(self, source=None):
        return self._payloads.pop(0)

    def iprobe(self, source=None):
        return self.drain and bool(self._payloads)

    def send(self, obj, dest, tag=0):
        self.sent.append((dest, tag, obj))


class _LogReader:
    filename = "/run/dir/gf_prod.h5"

    def __init__(self, fail=None):
        self.log = []
        self.fail = fail

    @property
    def iteration(self):
        return sum(1 for e in self.log if e[0] == "row")

    def save_step_main(self, tag, *a, **k):
        self.log.append(("row", tag))

    def completed_recipe_step(self, step_name, next_step_name=None):
        if self.fail:
            raise KeyError(f"recipe step {step_name!r} unreadable")
        self.log.append(("completed_recipe_step", step_name))

    def stamp_stage_flag(self, step_name, key, value):
        self.log.append(("stamp_stage_flag", step_name))
        return True


def _row(tag):
    return {"save_args": (tag,), "save_kwargs": {}}


def _write(method, *args, **kwargs):
    return {hb.STORE_WRITE_KEY: method, "args": args, "kwargs": kwargs}


class SaverLoopTest(_QuietSaverEnv):
    """The loop itself: arrival order, acks, coalescing, failures."""

    def _loop(self, payloads, reader, **kw):
        comm = _ScriptedComm(payloads + [{"finish_run": True}])
        hb.save_to_backend_asynchronously_and_plot(
            reader, comm, main_rank=HEAD, plot_container=None, **kw)
        return comm

    def test_one_batch_runs_in_arrival_order_and_acks_the_write(self):
        reader = _LogReader()
        comm = self._loop([_row("r1"), _row("r2"),
                           _write("completed_recipe_step", "gb_search_2",
                                  next_step_name="gb_search_3"),
                           _row("r3")], reader)
        self.assertEqual(reader.log, [("row", "r1"), ("row", "r2"),
                                      ("completed_recipe_step", "gb_search_2"),
                                      ("row", "r3")])
        self.assertEqual(len(comm.sent), 1)
        dest, tag, ack = comm.sent[0]
        self.assertEqual((dest, tag), (HEAD, hb.STORE_WRITE_ACK_TAG))
        self.assertTrue(ack["ok"])

    def test_the_ack_carries_the_methods_result(self):
        reader = _LogReader()
        comm = self._loop([_write("stamp_stage_flag", "gb_search_3",
                                  "galfor_ratchet_done", 1)], reader)
        self.assertEqual(comm.sent[0][2], {"ok": True, "result": True})

    def test_coalescing_never_drops_the_row_a_write_closes(self):
        """Backpressure keeps the newest row -- and the row right before each
        write, so the stamp still lands after the row it closes."""
        reader = _LogReader()
        self._loop([_row("r1"), _row("r2"), _row("r3"),
                    _write("completed_recipe_step", "gb_search_2"),
                    _row("r4"), _row("r5")], reader, coalesce_threshold=1)
        self.assertEqual(reader.log, [("row", "r3"),
                                      ("completed_recipe_step", "gb_search_2"),
                                      ("row", "r5")])

    def test_a_failing_write_is_logged_acked_and_the_saver_keeps_saving(self):
        reader = _LogReader(fail=True)
        with self.assertLogs(hb.logger, level="ERROR") as cm:
            comm = self._loop([_write("completed_recipe_step", "gb_search_2",
                                      next_step_name="gb_search_3"),
                               _row("r1")], reader)
        text = "\n".join(cm.output)
        self.assertIn("completed_recipe_step", text)
        self.assertIn("gb_search_2", text)
        self.assertIn("gb_search_3", text)
        ack = comm.sent[0][2]
        self.assertFalse(ack["ok"])
        self.assertIn("KeyError", ack["error"])
        self.assertEqual(reader.log, [("row", "r1")])

    def test_a_write_alone_does_not_rebuild_the_page_for_the_same_save(self):
        """A batch holding only a routed write saved no row: the page/snapshot
        hook (cadence counted in saves) must not fire a second time for the
        save count it already built at."""
        from lisatools.globalfit.monitor import hooks as mh

        os.environ["GF_MONITOR_AFTER_SAVE"] = "1"
        os.environ["GF_MONITOR_ITER"] = "1"
        comm = _ScriptedComm([_row("r1"),
                              _write("stamp_stage_flag", "gb_search_3", "k", 1),
                              {"finish_run": True}], drain=False)
        with mock.patch.object(mh, "build_products", return_value=1.0) as bp:
            hb.save_to_backend_asynchronously_and_plot(
                _LogReader(), comm, main_rank=HEAD, plot_container=None)
        self.assertEqual(bp.call_count, 1)

    def test_an_unknown_method_is_refused_not_run(self):
        reader = _LogReader()
        reader.reset = mock.Mock()
        with self.assertLogs(hb.logger, level="ERROR"):
            comm = self._loop([_write("reset", 4)], reader)
        reader.reset.assert_not_called()
        self.assertFalse(comm.sent[0][2]["ok"])


class HeadFailureTest(_QuietSaverEnv):
    def test_a_write_the_saver_could_not_do_raises_on_the_head(self):
        """The direct write would have raised on the head (completed_recipe_step
        has no try); the routed one raises there too, naming the method."""
        path = _store(self.tmp)
        h = _Harness(path).start()
        with self.assertLogs(hb.logger, level="ERROR") as logs, \
                self.assertRaises(RuntimeError) as cm:
            h.head.completed_recipe_step("no_such_step")
        h.finish()
        self.assertIn("completed_recipe_step", str(cm.exception))
        self.assertIn("no_such_step", str(cm.exception))
        # ... and the saver said so in its own log, with the arguments
        self.assertIn("no_such_step", "\n".join(logs.output))


class _NoSendComm:
    def __init__(self, size):
        self.size = size

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return 0

    def send(self, *a, **k):
        raise AssertionError("no saver rank: nothing may be sent")

    def recv(self, *a, **k):
        raise AssertionError("no saver rank: nothing may be received")


class NoSaverFallbackTest(_QuietSaverEnv):
    """Single process / fit.sample() / size < 3: the methods write directly."""

    def _check_direct(self, be, path):
        be.completed_recipe_step("gb_search_2", next_step_name="gb_search_3")
        be.stamp_stage_start("full_pe", 5)
        self.assertTrue(be.stamp_stage_flag("gb_search_3", "galfor_ratchet_done", 1))
        with h5py.File(path, "r") as f:
            r = f["global_fit/recipe"]
            self.assertTrue(bool(r["gb_search_2"].attrs["status"]))
            self.assertEqual(int(r["gb_search_2"].attrs["completed_iteration"]), 7)
            self.assertEqual(int(r["gb_search_3"].attrs["start_iteration"]), 7)
            self.assertEqual(int(r["full_pe"].attrs["start_iteration"]), 5)
            self.assertEqual(int(r["gb_search_3"].attrs["galfor_ratchet_done"]), 1)

    def test_no_comm_writes_directly(self):
        path = _store(self.tmp, iteration=7)
        self._check_direct(GFHDFBackend(path), path)

    def test_a_two_rank_run_writes_directly_like_its_rows(self):
        """Below 3 ranks the rows are a synchronous write on the head (the
        saver is aliased to it, or idle in the size-2 dedicated-saver case),
        so the stamps must stay with the rows on the head."""
        for size in (1, 2):
            with self.subTest(size=size):
                tmp = tempfile.mkdtemp(dir=self.tmp)
                path = _store(tmp, iteration=7)
                self._check_direct(
                    GFHDFBackend(path, comm=_NoSendComm(size), save_plot_rank=0), path)


_MPI_SCRIPT = textwrap.dedent('''
    import json, os, sys, time
    os.environ.pop("GF_MONITOR_AFTER_SAVE", None)
    os.environ.pop("GF_MONITOR_RANK", None)
    from mpi4py import MPI
    import h5py
    from lisatools.globalfit import hdfbackend as hb
    from lisatools.globalfit.hdfbackend import GFHDFBackend
    out_dir, path = sys.argv[1], sys.argv[2]
    comm = MPI.COMM_WORLD
    me, size = comm.Get_rank(), comm.Get_size()
    saver = size - 1
    hb._atomic_backup_copy = lambda *a, **k: None

    class RowSaver(GFHDFBackend):
        def save_step_main(self, tag, *a, **k):
            time.sleep(0.2)
            with self.open("a") as f:
                f[self.name].attrs["iteration"] = int(f[self.name].attrs["iteration"]) + 1

    class HeadSpy(GFHDFBackend):
        def open(self, mode="r"):
            self.__dict__.setdefault("modes", []).append(mode)
            return super().open(mode)

    if me == saver:
        hb.save_to_backend_asynchronously_and_plot(RowSaver(path), comm, main_rank=0)
    elif me == 0:
        head = HeadSpy(path, comm=comm, save_plot_rank=saver)
        head.save_step("r1")
        head.save_step("r2")
        head.completed_recipe_step("gb_search_2", next_step_name="gb_search_3")
        start = head.stage_start_iteration("gb_search_3")
        flag = head.stamp_stage_flag("gb_search_3", "galfor_ratchet_done", 1)
        head.save_step("r3")
        comm.send({"finish_run": True}, dest=saver)
        with open(os.path.join(out_dir, "head.json"), "w") as fh:
            json.dump({"modes": head.__dict__.get("modes", []),
                       "start": start, "flag": bool(flag)}, fh)
    comm.Barrier()
''')


RUN_GF_SMOKE = os.environ.get("RUN_GF_SMOKE", "") not in ("", "0")


def _warm_move(model, state):
    return state, None


def _line_move(model, state):
    return state, None


@unittest.skipUnless(RUN_GF_SMOKE, "set RUN_GF_SMOKE=1 to run the GlobalFit stage-boundary smoke")
class FakeWorldStageBoundaryTest(unittest.TestCase):
    """The REAL GlobalFit wiring on three in-process ranks (head 0, compute 1,
    saver 2): a ``search`` stage ends after its first iteration, so the run
    crosses a real stage boundary, and from the head's first row handoff on
    the head thread never opens the store for writing."""

    def test_a_stage_boundary_is_written_by_the_saver(self):
        from eryn.backends import HDFBackend as ErynHDF
        from eryn.prior import uniform_dist

        from lisatools.globalfit.communication.fakecomm import FakeWorld
        from lisatools.globalfit.communication.ranks import prepare_rank
        from lisatools.globalfit.recipe import Stage
        from lisatools.globalfit.run import GlobalFit
        from lisatools.globalfit.stock import erebor

        tmp = tempfile.mkdtemp(prefix="gf_routed_writes_")
        self.addCleanup(shutil.rmtree, tmp, True)
        lock = threading.Lock()
        opens, routed, sampling = [], [], {"head": False}
        _open, _save_step = ErynHDF.open, GFHDFBackend.save_step
        _run_write = hb._run_routed_store_write

        def rec_open(be, mode="r"):
            with lock:
                opens.append((threading.current_thread().name, mode,
                              os.path.basename(str(be.filename)), sampling["head"]))
            return _open(be, mode)

        def rec_save_step(be, *a, **k):
            out = _save_step(be, *a, **k)
            sampling["head"] = True      # the first row is handed off: sampling
            return out

        def rec_write(gb_reader, comm, main_rank, payload):
            routed.append((threading.current_thread().name, payload[hb.STORE_WRITE_KEY],
                           tuple(payload["args"]), dict(payload["kwargs"])))
            return _run_write(gb_reader, comm, main_rank, payload)

        def fn(rank, comm):
            fit = erebor.blank(nwalkers=4, ntemps=2, file_store_dir=tmp,
                               make_diagnostic_plots=False)
            fit.general.num_iterations = 4
            fit.add_branch("line", ndim=2,
                           priors={0: uniform_dist(0.0, 1.0), 1: uniform_dist(0.0, 1.0)},
                           moves=[_line_move])
            fit.recipe.add_stage(Stage("warm", kind="search", moves=[_warm_move]),
                                 before="main")
            prepare_rank(fit, comm)
            fit.build()
            gf = GlobalFit(fit, comm)
            gf.run_global_fit()
            return gf.curr.general_info.main_file_path

        with mock.patch.object(ErynHDF, "open", rec_open), \
                mock.patch.object(GFHDFBackend, "save_step", rec_save_step), \
                mock.patch.object(hb, "_run_routed_store_write", rec_write), \
                mock.patch.object(hb, "_atomic_backup_copy"):
            out = FakeWorld(3, timeout=600.0).run(fn)
        store = out[0]
        name = os.path.basename(store)
        head_writes = [o for o in opens
                       if o[0] == "fake-rank-0" and o[1] != "r" and o[2] == name and o[3]]
        self.assertEqual(head_writes, [], "the head wrote the store while sampling")
        self.assertTrue(any(o[0] == "fake-rank-0" and o[1] != "r" for o in opens),
                        "control: the head's launch-time writes (reset/grow) were not seen")
        self.assertEqual(routed, [("fake-rank-2", "completed_recipe_step", ("warm",),
                                   {"next_step_name": "main"})])
        with h5py.File(store, "r") as f:
            r = f["global_fit/recipe"]
            self.assertTrue(bool(r["warm"].attrs["status"]))
            self.assertEqual(int(r["warm"].attrs["completed_iteration"]), 1)
            self.assertEqual(int(r["main"].attrs["start_iteration"]), 1)
            self.assertEqual(int(f["global_fit"].attrs["iteration"]), 4)


@unittest.skipUnless(shutil.which("mpiexec"), "mpiexec not available")
class RealMpiRoutingTest(unittest.TestCase):
    """mpiexec -n 3: head 0, an idle compute stand-in 1, saver 2."""

    def test_routed_stamp_over_real_mpi(self):
        import json

        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        path = _store(d)
        script = os.path.join(d, "mpi_routed_writes.py")
        with open(script, "w") as fh:
            fh.write(_MPI_SCRIPT)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.path.join(ROOT, "src") + os.pathsep + env.get(
            "PYTHONPATH", "")
        r = subprocess.run(
            ["mpiexec", "--oversubscribe", "-n", "3", sys.executable, script, d, path],
            env=env, capture_output=True, text=True, timeout=180)
        self.assertEqual(r.returncode, 0, r.stdout[-2000:] + r.stderr[-2000:])
        with open(os.path.join(d, "head.json")) as fh:
            head = json.load(fh)
        self.assertEqual([m for m in head["modes"] if m != "r"], [], head)
        self.assertEqual(head["start"], 2)
        self.assertTrue(head["flag"])
        with h5py.File(path, "r") as f:
            r_ = f["global_fit/recipe"]
            self.assertEqual(int(r_["gb_search_2"].attrs["completed_iteration"]), 2)
            self.assertEqual(int(r_["gb_search_3"].attrs["start_iteration"]), 2)
            self.assertEqual(int(f["global_fit"].attrs["iteration"]), 3)


if __name__ == "__main__":
    unittest.main()
