"""Automatic warm-start component building (user ruling 2026-09-14).

The warm-start refit proposal (``rj_warm_search`` / ``rj_warm_pe``) needs a
REFEREED components npz built from the previous run's store by the three-step
pipeline fit -> referee -> apply (``scripts/gb/warmstart_fit_from_store.py``
-> ``warmstart_match_referee.py`` -> ``warmstart_referee_apply.py``). The 6mo
campaign's first launch failed on that npz being missing, so
:func:`ensure_warm_start_components` now runs the pipeline automatically at
recipe build when the npz is absent and ``GB_WARM_START_SOURCE_STORE`` names
the source store.

The steps run IN-PROCESS -- each script module is imported and its
``main()`` called under a swapped ``sys.argv`` (user ruling 2026-09-14:
"python imports right? not like calling bash") -- no subprocess, no
interpreter spawn.

MPI-safe by a lock DIRECTORY next to the target: ``run_combined_staged.py``
builds on every rank before roles resolve, so one rank wins ``os.mkdir`` on
the lock and builds while the others poll for the npz. A crashed builder
releases the lock in ``finally``; a lock left behind by a killed process is
taken over once it is older than the timeout.

Env knobs (read here, not Settings fields -- they configure a one-shot
offline build, not the sampler): ``GB_WARM_START_SOURCE_STORE`` (the
previous run's FULL FINAL h5 -- never a make_snapshots extract),
``GB_WARM_START_LAST_K`` (default 10), ``GB_WARM_START_SOURCE_TOBS`` (the
SOURCE store's Tobs, default 7776000.0 = 3 months -- NOT this run's Tobs;
``WarmStartComponents.from_npz(new_tobs=...)`` does the rescale at load),
``GB_WARM_START_BUILD_TIMEOUT`` (seconds a waiting rank polls, default 7200).
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
import typing

__all__ = ["ensure_warm_start_components"]

logger = logging.getLogger(__name__)

_MODULE_CACHE: dict = {}

_SCRIPTS = (
    "warmstart_fit_from_store.py",
    "warmstart_match_referee.py",
    "warmstart_referee_apply.py",
)


def _manual_recipe(path: str, store: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    fit = os.path.join(os.path.dirname(path) or ".", stem + "_fit.npz")
    ref = os.path.splitext(fit)[0] + "_referee.npz"
    return (
        f"python scripts/gb/warmstart_fit_from_store.py --store {store} "
        f"--last-k 10 --tobs 7776000 --out {fit}\n"
        f"python scripts/gb/warmstart_match_referee.py --npz {fit} "
        f"--store {store}\n"
        f"python scripts/gb/warmstart_referee_apply.py --fit {fit} "
        f"--referee {ref} --out {path}"
    )


def _script_path(name: str) -> str:
    """Resolve ``scripts/gb/<name>`` -- repo root (editable src layout:
    ``<repo>/src/lisatools/...``) first, then the current directory (the
    staged driver runs from the repo root)."""
    import lisatools

    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(lisatools.__file__))))
    for base in (repo, os.getcwd()):
        cand = os.path.join(base, "scripts", "gb", name)
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError(
        f"cannot locate scripts/gb/{name} (looked under {repo!r} and the "
        "current directory) -- the warm-start auto-build needs the LAT "
        "repo checkout, not just the installed package."
    )


def _run_script(script_path: str, argv: typing.Sequence[str]) -> None:
    """Run one pipeline script IN-PROCESS: import the module (cached) and
    call its ``main()`` under a temporarily swapped ``sys.argv``.

    User ruling 2026-09-14 ("python imports right? not like calling
    bash"): no subprocess, no interpreter spawn -- the step executes
    inside the calling rank's python, which sidesteps cluster fork/exec
    restrictions and reuses the already-imported heavy stack. All three
    scripts are ``if __name__ == "__main__"`` guarded (import runs no
    work) and expose a zero-argument argparse ``main()``.
    """
    script_path = os.path.abspath(script_path)
    mod = _MODULE_CACHE.get(script_path)
    if mod is None:
        name = "_warmstart_step_" + os.path.splitext(
            os.path.basename(script_path))[0]
        spec = importlib.util.spec_from_file_location(name, script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODULE_CACHE[script_path] = mod
    old_argv = list(sys.argv)
    sys.argv = [script_path] + [str(a) for a in argv]
    try:
        mod.main()
    except SystemExit as exc:  # argparse errors / explicit exits
        if exc.code not in (None, 0):
            raise RuntimeError(
                f"{os.path.basename(script_path)} exited with code "
                f"{exc.code}."
            ) from exc
    finally:
        sys.argv = old_argv


def _default_runner(cmd: typing.Sequence[str]) -> None:
    logger.info("[WARMSTART-BUILD] running (in-process): %s",
                " ".join(map(str, cmd)))
    _run_script(str(cmd[0]), [str(a) for a in cmd[1:]])


def _require(path: str, step: str) -> None:
    if not os.path.exists(path):
        raise RuntimeError(
            f"[WARMSTART-BUILD] {step} completed without producing "
            f"{path!r} -- read its output above."
        )


def ensure_warm_start_components(
    path: str,
    *,
    store: typing.Optional[str] = None,
    last_k: typing.Optional[int] = None,
    tobs: typing.Optional[float] = None,
    timeout: typing.Optional[float] = None,
    poll: float = 5.0,
    runner: typing.Optional[typing.Callable[[list], None]] = None,
    log: typing.Optional[logging.Logger] = None,
) -> str:
    """The refereed warm-start npz at ``path`` -- built if missing.

    Returns ``path``. When the file already exists this is a pure check.
    Otherwise the fit -> referee -> apply pipeline runs against ``store``
    (or ``GB_WARM_START_SOURCE_STORE``), under a lock directory so that
    concurrent MPI ranks build exactly once; the output lands atomically
    (tmp + ``os.replace``), so a waiting rank never sees a partial npz.
    """
    log = log or logger
    if os.path.exists(path):
        return path

    store = (store or os.environ.get("GB_WARM_START_SOURCE_STORE", "")
             or "").strip()
    if not store:
        raise FileNotFoundError(
            f"warm-start components {path!r} do not exist and "
            "GB_WARM_START_SOURCE_STORE is not set, so they cannot be "
            "built automatically. Either set GB_WARM_START_SOURCE_STORE "
            "to the previous run's FULL FINAL store h5 (never a "
            "make_snapshots extract) or build the npz by hand:\n"
            + _manual_recipe(path, "<previous-run store h5>")
        )
    if not os.path.exists(store):
        raise FileNotFoundError(
            f"warm-start source store {store!r} does not exist "
            "(GB_WARM_START_SOURCE_STORE)."
        )

    last_k = int(last_k if last_k is not None
                 else os.environ.get("GB_WARM_START_LAST_K", "10"))
    tobs = float(tobs if tobs is not None
                 else os.environ.get("GB_WARM_START_SOURCE_TOBS",
                                     "7776000.0"))
    timeout = float(timeout if timeout is not None
                    else os.environ.get("GB_WARM_START_BUILD_TIMEOUT",
                                        "7200"))
    runner = runner or _default_runner

    workdir = os.path.dirname(path) or "."
    os.makedirs(workdir, exist_ok=True)
    lock = path + ".build.lock"

    try:
        os.mkdir(lock)
    except FileExistsError:
        # another rank (or process) is building; wait for the npz.
        log.info(
            "[WARMSTART-BUILD] %s is being built elsewhere (lock %s); "
            "waiting up to %.0f s.", path, lock, timeout)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if os.path.exists(path):
                return path
            # stale lock (builder killed without cleanup): take over.
            try:
                if time.time() - os.path.getmtime(lock) > timeout:
                    os.rmdir(lock)
                    return ensure_warm_start_components(
                        path, store=store, last_k=last_k, tobs=tobs,
                        timeout=timeout, poll=poll, runner=runner, log=log)
            except OSError:
                pass
            time.sleep(poll)
        raise RuntimeError(
            f"timed out after {timeout:.0f} s waiting for another process "
            f"to build {path!r} (lock {lock}). If no builder is alive, "
            "remove the lock directory and relaunch."
        )

    stem = os.path.splitext(os.path.basename(path))[0]
    fit_npz = os.path.join(workdir, stem + "_fit.npz")
    referee_npz = os.path.splitext(fit_npz)[0] + "_referee.npz"
    tmp_out = path + ".building.npz"
    try:
        log.warning(
            "[WARMSTART-BUILD] %s missing -- building it now from %s "
            "(last_k=%d, source tobs=%.0f s). fit -> referee -> apply; "
            "other ranks wait on the lock.", path, store, last_k, tobs)
        t0 = time.perf_counter()
        runner([_script_path(_SCRIPTS[0]),
                "--store", store, "--last-k", str(last_k),
                "--tobs", str(tobs), "--out", fit_npz])
        _require(fit_npz, "warmstart_fit_from_store.py")
        runner([_script_path(_SCRIPTS[1]),
                "--npz", fit_npz, "--store", store])
        _require(referee_npz, "warmstart_match_referee.py")
        runner([_script_path(_SCRIPTS[2]),
                "--fit", fit_npz, "--referee", referee_npz,
                "--out", tmp_out])
        _require(tmp_out, "warmstart_referee_apply.py")
        os.replace(tmp_out, path)
        log.warning(
            "[WARMSTART-BUILD] built %s in %.1f s.",
            path, time.perf_counter() - t0)
        return path
    finally:
        try:
            os.remove(tmp_out)
        except OSError:
            pass
        try:
            os.rmdir(lock)
        except OSError:
            pass
