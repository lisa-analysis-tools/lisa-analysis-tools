"""A DEDICATED MONITOR RANK: the page + snapshot tar off the saver rank.

USER RULING 2026-10-04 (Mike): "adjust this so they are separated and you
can have a separate rank run the html".

WHY
---
The saver rank is the run's only writer, and the head's ``save_step`` is a
BLOCKING pickled send to it. When the page and the tar were built on the
saver (:func:`.hooks.after_save`), the saver stopped receiving for the whole
build, so the head's NEXT save waited for the build -- and every compute
rank waits on the head. Measured on the 6mo replica_pe (job 717,
2026-10-04): build 575-600 s page + ~175 s tar = ~775 s against a 321 s
save interval; the save after every build took 398-413 s instead of
0.13 s, with every GPU idle -- about 25-30 % of the wall at
``GF_MONITOR_ITER=3``. Building in the writer's process was also the route
of the 2026-09-28 store corruption (``project_monitor_saver_corruption``).

HOW
---
* :func:`split_monitor_rank` (driver, before ANYTHING else touches MPI)
  takes the HIGHEST world rank out of the run. The run gets a communicator
  of exactly the size it had before the launcher added the extra task, and
  ``Split`` keeps the world order, so every run rank keeps its number, its
  node placement, its role and its walker block. The layout code never
  sees the monitor rank.
* The saver (:func:`notify_saved`, called from
  ``hdfbackend.save_to_backend_asynchronously_and_plot``) posts a tiny
  non-blocking notice on the WORLD communicator after its saves and goes
  straight back to ``recv``. Nothing on the saver waits for the monitor.
* The monitor rank (:func:`run_monitor_rank`) coalesces notices to the
  newest, builds on the ``GF_MONITOR_ITER`` cadence from the files on disk
  (the live store, read through the generator's own backup-copy fallbacks
  for rows the saver is writing), and leaves when every run rank has joined
  the closing non-blocking barrier (:func:`finish_run_ranks`). It never
  raises: an exception escaping it would hit the driver's MPI-abort hook
  and take the run down.

``GF_MONITOR_RANK=1`` arms it (the 6mo/1yr v9 launchers add the task and
set it). Off, or with no split configured, the saver keeps the old
in-process hook.
"""

from __future__ import annotations

import logging
import os
import sys
import time

logger = logging.getLogger(__name__)

__all__ = [
    "MONITOR_TAG",
    "monitor_rank_enabled",
    "split_monitor_rank",
    "active",
    "notify_saved",
    "finish_run_ranks",
    "run_monitor_rank",
]

# Notices travel on the WORLD communicator, a different MPI context from the
# run's communicator, so this tag cannot collide with any run message.
MONITOR_TAG = 7801

# Per-process state, set by split_monitor_rank on the RUN ranks only.
_STATE = {"world": None, "monitor_rank": None, "reqs": []}


def _any_source():
    try:
        from mpi4py import MPI

        return MPI.ANY_SOURCE
    except Exception:  # noqa: BLE001 -- test doubles without mpi4py
        return -1


def monitor_rank_enabled(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get("GF_MONITOR_RANK", "0")).strip() in (
        "1", "true", "True", "yes", "on")


def _reset_state():
    _STATE["world"] = None
    _STATE["monitor_rank"] = None
    _STATE["reqs"] = []


def split_monitor_rank(world, environ=None):
    """``(run_comm, monitor_rank)``. COLLECTIVE on ``world`` when armed.

    Off (or a single process): ``(world, None)`` and nothing changes. Armed:
    the highest world rank is the monitor; it receives ``(None, rank)`` and
    must call :func:`run_monitor_rank`. Every other rank receives the run
    communicator (world order kept) and is registered to notify / finish.
    """
    _reset_state()
    size = int(world.Get_size())
    if not monitor_rank_enabled(environ) or size == 1:
        return world, None
    if size < 4:
        raise ValueError(
            f"GF_MONITOR_RANK=1 needs at least 4 MPI ranks (head + compute + "
            f"saver + monitor); this job has {size}. Launch one more task or "
            "set GF_MONITOR_RANK=0.")
    mon = size - 1
    me = int(world.Get_rank())
    run_comm = world.Split(1 if me == mon else 0, me)
    if me == mon:
        return None, mon
    _STATE["world"] = world
    _STATE["monitor_rank"] = mon
    _STATE["reqs"] = []
    return run_comm, mon


def active() -> bool:
    """Whether this process notifies a dedicated monitor rank."""
    return _STATE["world"] is not None and _STATE["monitor_rank"] is not None


def notify_saved(store_path, i, dropped=0) -> None:
    """Saver side: post a NON-BLOCKING notice that save ``i`` is on disk.

    Completed sends are released here; the list holds at most the sends
    the monitor has not drained yet. Never raises into the saver loop.
    """
    try:
        world, mon = _STATE["world"], _STATE["monitor_rank"]
        reqs = [r for r in _STATE["reqs"] if not r.Test()]
        reqs.append(world.isend(
            {"saved": int(i), "store": str(store_path),
             "dropped": int(dropped)},
            dest=mon, tag=MONITOR_TAG))
        _STATE["reqs"] = reqs
    except Exception as e:  # noqa: BLE001 -- a notice must never cost a save
        logger.warning("[GF_MONITOR_RANK] notice for save %s not sent "
                       "(%s: %s); saves are unaffected.", i,
                       type(e).__name__, e)


def finish_run_ranks() -> None:
    """EVERY run rank, at the end of the driver: complete this rank's
    outstanding notices, then join the closing barrier the monitor rank
    waits on. A no-op when no monitor rank is configured."""
    if not active():
        return
    for r in _STATE["reqs"]:
        r.Wait()
    _STATE["reqs"] = []
    _STATE["world"].Ibarrier().Wait()
    _reset_state()


def _due(i, last_built, every):
    """Build on the GF_MONITOR_ITER cadence, counted in SAVES like the
    saver-rank hook (``i % every == 0``), but robust to coalescing: build
    once the newest save has crossed the next multiple of ``every``."""
    every = max(int(every), 1)
    return int(i) > 0 and (int(i) // every) > (int(last_built) // every)


def _every(environ):
    try:
        return max(int(float(environ.get("GF_MONITOR_ITER", "") or 1)), 1)
    except ValueError:
        return 1


def _configure_logging():
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO, stream=sys.stdout,
            format="%(asctime)s - [monitor-rank] %(name)s - %(levelname)s "
                   "- %(message)s")


def run_monitor_rank(world, *, build=None, environ=None, poll_s=2.0,
                     sleep=time.sleep, hide_gpus=True) -> int:
    """The monitor rank's whole life. Returns the number of builds.

    ``build(run_dir)`` defaults to :func:`.hooks.build_products`. The loop
    polls (``iprobe`` + the closing barrier's ``Test``) rather than
    blocking in ``recv``, because the barrier -- not a message -- is what
    says the run is over: it completes only once every run rank, the saver
    included, has finished.
    """
    env = os.environ if environ is None else environ
    _configure_logging()
    if hide_gpus:
        # This process needs no GPU. Hide the node's devices before
        # anything initialises CUDA, so no context lands on a compute
        # rank's card.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if build is None:
        from .hooks import build_products as build
    src = _any_source()
    barrier = world.Ibarrier()
    latest = None
    last_built = 0
    builds = 0
    logger.info("[GF_MONITOR_RANK] rank %d: monitor rank up (GF_MONITOR_ITER="
                "%d, GF_MONITOR_AFTER_SAVE=%s).", int(world.Get_rank()),
                _every(env), env.get("GF_MONITOR_AFTER_SAVE", "0"))
    while True:
        try:
            while world.iprobe(source=src, tag=MONITOR_TAG):
                latest = world.recv(source=src, tag=MONITOR_TAG)
        except Exception as e:  # noqa: BLE001
            logger.warning("[GF_MONITOR_RANK] receive failed (%s: %s).",
                           type(e).__name__, e)
        if barrier.Test():
            # Every run rank has finished. A notice can still be in flight
            # (an eager send completes on the sender before it is
            # received), so drain once more rather than leave it unmatched
            # at MPI_Finalize.
            try:
                while world.iprobe(source=src, tag=MONITOR_TAG):
                    world.recv(source=src, tag=MONITOR_TAG)
            except Exception:  # noqa: BLE001
                pass
            break
        want = (latest is not None
                and env.get("GF_MONITOR_AFTER_SAVE", "0") == "1"
                and _due(latest["saved"], last_built, _every(env)))
        if not want:
            sleep(poll_s)
            continue
        i = int(latest["saved"])
        run_dir = os.path.dirname(os.path.abspath(latest["store"]))
        last_built = i
        st = time.perf_counter()
        try:
            out = build(run_dir)
        except (Exception, SystemExit) as e:  # noqa: BLE001 -- never abort the run
            logger.warning("[GF_MONITOR_RANK] build after save %d failed "
                           "(%s: %s); the run is unaffected.", i,
                           type(e).__name__, e)
            continue
        if out is None:
            continue
        builds += 1
        logger.info("[GF_MONITOR_RANK] page+snapshot after save %d in %.1f s "
                    "(%d built; no save waits on this rank).", i,
                    time.perf_counter() - st, builds)
    logger.info("[GF_MONITOR_RANK] run finished; monitor rank exiting after "
                "%d build(s).", builds)
    return builds
