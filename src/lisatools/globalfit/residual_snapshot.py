"""The run's own residual, written beside the store for the monitor page.

``GF_RESIDUAL_SNAPSHOT_EVERY=N`` (default 3; 0 turns it off): after every
N-th iteration the head writes
``<store stem>_residual_snapshot.npz`` holding, in the run's own WDM
analysis domain (active band and time span only):

* ``data`` -- the data exactly as the run loaded it
  (``general_info.input_data_residual_array``; for a mojito COMBINED run the
  pre-summed stream with every source class and the noise in it);
* ``residual`` -- the max-lnL cold walker's residual, i.e. that data minus
  EVERY fitted signal (GB, VGB, MBH, EMRI, SOBBH): the array its likelihood
  is computed from;
* ``f_edges`` / ``t_edges`` (Hz / s, active band), ``iteration``,
  ``walker``, ``lnl`` (every cold walker), ``nf`` / ``nt`` / ``dt``.

Why the run writes it rather than the monitor rebuilding it (Mike
2026-10-07): the monitor's data / template / residual panel used to subtract
only GB + VGB from per-type bricks it picked itself, so every MBH merger
in the window showed up as a "residual" (an MBH at 111 d on the 6mo page).
Regenerating every source class outside the run would need the run's exact
generators and settings; the run already holds the exact answer.

Arrays are stored as float32 (complex64 for a complex WDM basis): this is a
plotting product, not an analysis input. Multi-rank: the residual comes from
the walker's owning rank through the ``residual_snapshot`` fan-out builtin
(see :data:`RESIDUAL_SNAPSHOT_OP`); the data from the head's own copy.

Nothing here may break the run: :func:`maybe_snapshot` logs and carries on.
"""
from __future__ import annotations

import logging
import os
import time

import numpy as np

logger = logging.getLogger(__name__)

#: Sidecar file name: ``<store without .h5>`` + this suffix.
SNAPSHOT_SUFFIX = "_residual_snapshot.npz"
#: Cadence knob, in iterations; 0 = off.
SNAPSHOT_EVERY_ENV = "GF_RESIDUAL_SNAPSHOT_EVERY"
#: Default cadence: every 3rd iteration (~15 min on the 6mo run; one ~1 s
#: fan-out + a ~20 MB file, overwritten in place).
SNAPSHOT_EVERY_DEFAULT = 3
#: Fan-out builtin: the owning rank returns one walker's residual array.
RESIDUAL_SNAPSHOT_OP = "residual_snapshot"


def snapshot_every() -> int:
    """``GF_RESIDUAL_SNAPSHOT_EVERY`` as an int (default
    :data:`SNAPSHOT_EVERY_DEFAULT`; 0 when invalid)."""
    try:
        return max(0, int(os.environ.get(SNAPSHOT_EVERY_ENV,
                                         str(SNAPSHOT_EVERY_DEFAULT)) or 0))
    except ValueError:
        logger.warning("%s=%r is not an integer; residual snapshots are off.",
                       SNAPSHOT_EVERY_ENV, os.environ.get(SNAPSHOT_EVERY_ENV))
        return 0


def snapshot_path(store_path: str) -> str:
    """The sidecar path for the store at ``store_path``."""
    stem = store_path[:-3] if store_path.endswith(".h5") else store_path
    return stem + SNAPSHOT_SUFFIX


def _plot_array(arr) -> np.ndarray:
    """Host copy in a plotting dtype (float32, or complex64 if complex)."""
    a = arr.get() if hasattr(arr, "get") else np.asarray(arr)
    return a.astype(np.complex64 if np.iscomplexobj(a) else np.float32)


def walker_residual(model, payload):
    """Fan-out builtin body: this rank's walker ``payload["local"]``'s
    residual, or ``None`` on a rank that was not asked."""
    if payload is None:
        return None
    ac = model.analysis_container_arr[int(payload["local"])]
    return _plot_array(ac.data.arr)


def take_snapshot(*, acs, data_holder, store_path, iteration, fanout=None):
    """Write the sidecar for the max-lnL cold walker; returns its path.

    ``acs`` is the head's AnalysisContainerArray (its own block under the
    walker-block layout), ``data_holder`` the run's
    ``general_info.input_data_residual_array``.
    """
    if fanout is None:
        _ll = acs.likelihood(complex=False)
        lls = np.asarray(_ll.get() if hasattr(_ll, "get") else _ll, float)
    else:
        lls = np.asarray(fanout.gather_likelihood(acs), float)
    finite = np.where(np.isfinite(lls), lls, -np.inf)
    w = int(np.argmax(finite))

    if fanout is None:
        residual = _plot_array(acs[w].data.arr)
    else:
        owner, local = fanout.layout.owner_of(w)

        def _payload(rank, w0, w1):
            return {"local": int(local)} if int(rank) == int(owner) else None

        def _local(payload, model):
            return None if payload is None else _plot_array(
                acs[int(payload["local"])].data.arr)

        def _merge(results):
            got = [v for v in results.values() if v is not None]
            if len(got) != 1:
                raise RuntimeError(
                    f"residual snapshot: expected one rank to answer for "
                    f"walker {w}, got {len(got)}")
            return got[0]

        residual = fanout.run(RESIDUAL_SNAPSHOT_OP, move=None,
                              per_rank_payload=_payload, local_body=_local,
                              merge=_merge)

    data = _plot_array(data_holder.data_res_arr.arr)
    if data.shape != residual.shape:
        raise RuntimeError(f"residual snapshot: data {data.shape} and residual "
                           f"{residual.shape} differ")
    s = acs[0].data.settings
    path = snapshot_path(store_path)
    tmp = path + ".tmp.npz"
    np.savez(
        tmp, data=data, residual=residual,
        f_edges=np.asarray(_plot_array(s.f_arr_edges), float),
        t_edges=np.asarray(_plot_array(s.t_arr_edges), float),
        iteration=int(iteration), walker=w, lnl=lls,
        nf=int(s.Nf), nt=int(s.Nt), dt=float(s.data_dt),
        written_at=time.time())
    os.replace(tmp, path)   # a reader never sees a partial file
    return path


def maybe_snapshot(*, iteration, acs, data_holder, store_path, fanout=None):
    """Cadence gate + never-raise wrapper around :func:`take_snapshot`.

    ``iteration`` is the index of the iteration that just ran; a snapshot is
    written after iterations ``N-1, 2N-1, ...``. Returns the path or None.
    """
    every = snapshot_every()
    if every <= 0 or (int(iteration) + 1) % every:
        return None
    if acs is None or data_holder is None or not store_path:
        return None
    try:
        t0 = time.perf_counter()
        path = take_snapshot(acs=acs, data_holder=data_holder,
                             store_path=store_path, iteration=iteration,
                             fanout=fanout)
        logger.info("[RESIDUAL_SNAPSHOT] iteration %d -> %s (%.1f s)",
                    int(iteration), os.path.basename(path),
                    time.perf_counter() - t0)
        return path
    except Exception as exc:  # noqa: BLE001 - a monitor product never breaks the run
        logger.warning("[RESIDUAL_SNAPSHOT] iteration %d failed (%r); the run "
                       "continues.", int(iteration), exc, exc_info=True)
        return None
