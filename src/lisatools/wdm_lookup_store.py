"""Find-or-build store for the n_ref WDM lookup tables (``WDMLookupTable``).

A table depends only on the wavelet grid's ``Nf`` and sample step ``dt`` (the layer
duration ``Nf*dt``) and on the build recipe; it is built once and reused across runs and
restarts:

* :func:`lookup_table_path` -- an explicit path wins; otherwise the recipe's canonical file
  name inside a directory (the run's folder in the global fit).
* :func:`ensure_lookup_table` -- returns that path, building and saving the table there
  first when it does not exist. A found table is checked against the requested grid. The
  build writes a temporary file and renames it into place, so a reader never sees a partial
  table; an ``<path>.lock`` file makes concurrent callers (MPI ranks, a launcher preflight)
  wait for one builder instead of building twice.

The default recipe :data:`EMRI_TABLE_RECIPE` is the one every EMRI direct-to-WDM table so
far was built with (``scripts/wdm/build_wdm_lookup_gpu.py --build-kind n_ref_complex
--m-ref 21 --eps-freq 0.005 --num-layers-diff 2 --eps-fdot 0.01 --fdot-max-factor 8
--time-layers 32 --nchannels 1``), so its canonical names are the existing files'
(``wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5`` on the 6-month grid).
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import time
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "EMRI_TABLE_RECIPE",
    "lookup_table_name",
    "lookup_table_path",
    "build_lookup_table",
    "ensure_lookup_table",
]

#: build recipe of the EMRI direct-to-WDM tables (see the module docstring)
EMRI_TABLE_RECIPE = dict(
    prefix="wdm_lookup_emri_cx",
    build_kind="n_ref_complex",
    m_ref=21,
    eps_freq=0.005,
    num_layers_diff=2,
    eps_fdot=0.01,
    fdot_max_factor=8.0,
    time_layers=32,
    nchannels=1,
    batch_size=64,
    # the build grid's band and length do not enter the table (it is a single-pixel
    # response at layer m_ref); kept as the original builds had them
    min_freq=1e-4,
    max_freq=2.5e-2,
    Nt=1024,
)

#: seconds a caller waits for another process's build before giving up
WAIT_TIMEOUT = 6 * 3600.0


def _tag(x) -> str:
    return f"{float(x):g}".replace(".", "p")


def lookup_table_name(Nf: int, dt: float, recipe: Optional[dict] = None) -> str:
    """Canonical file name of a ``recipe`` table on the ``(Nf, dt)`` grid."""
    r = dict(EMRI_TABLE_RECIPE, **(recipe or {}))
    return (f"{r['prefix']}_NF{int(Nf)}_DT{_tag(dt)}_TL{int(r['time_layers'])}"
            f"_fd{_tag(r['fdot_max_factor'])}x{_tag(r['eps_fdot'])}_nld{int(r['num_layers_diff'])}.h5")


def lookup_table_path(table: Optional[str], table_dir: Optional[str], Nf: int, dt: float,
                      recipe: Optional[dict] = None) -> str:
    """``table`` when given (a pointer to a specific file), else ``table_dir/<canonical name>``."""
    if table:
        return os.path.abspath(os.path.expanduser(str(table)))
    if not table_dir:
        raise ValueError("lookup_table_path: neither a table path nor a directory to keep it in")
    return os.path.abspath(os.path.join(os.path.expanduser(str(table_dir)),
                                        lookup_table_name(Nf, dt, recipe)))


def build_lookup_table(path: str, *, Nf: int, dt: float, force_backend: str = "cpu",
                       recipe: Optional[dict] = None, verbose: bool = False) -> str:
    """Build the ``recipe`` table on the ``(Nf, dt)`` grid and save it at ``path`` (atomic)."""
    from .domains import WDMLookupTable, WDMSettings

    r = dict(EMRI_TABLE_RECIPE, **(recipe or {}))
    settings = WDMSettings(Nf=int(Nf), Nt=int(r["Nt"]), dt=float(dt), min_freq=r["min_freq"],
                           max_freq=r["max_freq"], force_backend=force_backend)
    norm_f, m_diffs, m_ref = WDMLookupTable.apply_eps_frequency(
        r["eps_freq"], settings, m_ref=r["m_ref"], num_layers_diff=r["num_layers_diff"])
    fdot_vals = WDMLookupTable.apply_eps_fdot(r["eps_fdot"], settings,
                                              fdot_max_factor=r["fdot_max_factor"])
    tmp = f"{path}.building.{socket.gethostname()}.{os.getpid()}"
    if os.path.exists(tmp):
        os.remove(tmp)
    try:
        WDMLookupTable(settings, nchannels=int(r["nchannels"]), m_ref=m_ref,
                       norm_freq_single_layer=norm_f, m_diffs=m_diffs, fdot_vals=fdot_vals,
                       store_path=tmp, batch_size_gen=int(r["batch_size"]),
                       build_kind=r["build_kind"], time_layers=int(r["time_layers"]),
                       verbose=verbose)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def _check_grid(path: str, Nf: int, dt: float) -> None:
    import h5py

    from .domains import WDMLookupTable

    try:
        table = WDMLookupTable.from_file(path, force_backend="cpu")
    except (OSError, KeyError, h5py.HDF5ExtError) as exc:
        raise ValueError(f"lookup table {path} cannot be read ({exc}); move it aside to rebuild") from exc
    if int(table.Nf) != int(Nf) or abs(float(table.data_dt) - float(dt)) > 1e-9:
        raise ValueError(
            f"lookup table {path} is built for Nf={table.Nf}, dt={table.data_dt}; this grid is "
            f"Nf={Nf}, dt={dt}. Point to the right table or move this one aside."
        )


def ensure_lookup_table(path: str, *, Nf: int, dt: float, force_backend: str = "cpu",
                        recipe: Optional[dict] = None, wait_timeout: float = WAIT_TIMEOUT,
                        poll: float = 10.0) -> str:
    """Return ``"found"``, ``"built"`` or ``"waited"`` once a valid table is at ``path``.

    Missing: build it there (creating the directory) under ``<path>.lock``; a caller that
    finds the lock held waits for the table to appear (another rank or the launcher is
    building it). A lock whose builder died without a table is taken over once it has not
    been touched for ``wait_timeout`` seconds. A found table on another grid raises
    ``ValueError`` (never rebuilt over).
    """
    if os.path.exists(path):
        _check_grid(path, Nf, dt)
        return "found"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lock = path + ".lock"
    waited = False
    t_start = time.monotonic()
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if os.path.exists(path):
                _check_grid(path, Nf, dt)
                return "waited"
            try:
                age = time.time() - os.path.getmtime(lock)
            except FileNotFoundError:
                continue                                  # the builder just finished or failed
            if age > wait_timeout:
                logger.warning("lookup table lock %s untouched for %.0f s: taking it over.", lock, age)
                with contextlib.suppress(FileNotFoundError):
                    os.remove(lock)
                continue
            if not waited:
                logger.info("lookup table %s is being built elsewhere (%s); waiting.", path, lock)
                waited = True
            if time.monotonic() - t_start > wait_timeout:
                raise TimeoutError(f"waited {wait_timeout:.0f} s for {path} (lock {lock})")
            time.sleep(poll)
            continue
        try:
            os.write(fd, f"{socket.gethostname()} {os.getpid()}\n".encode())
            os.close(fd)
            if os.path.exists(path):                      # built between our check and the lock
                _check_grid(path, Nf, dt)
                return "waited"
            t0 = time.monotonic()
            logger.info("lookup table %s not found: building it (Nf=%d, dt=%g, %s).",
                        path, int(Nf), float(dt), force_backend)
            build_lookup_table(path, Nf=Nf, dt=dt, force_backend=force_backend, recipe=recipe)
            logger.info("lookup table %s built in %.0f s.", path, time.monotonic() - t0)
            return "built"
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(lock)

