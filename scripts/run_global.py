import importlib.util
import multiprocessing
import sys
import argparse

import numpy as np
import os
import warnings
from copy import deepcopy
from pathlib import Path

import ast
import ctypes
import sys

import yaml

# NOTE (spawn safety): multiprocessing 'spawn' children (e.g. the eryn flow
# trainer) re-import this module at module level WITH the parent's sys.argv.
# Everything heavy or CUDA-touching must therefore live behind the
# parent-process guard in _pre_init_cuda or inside the __main__ block below —
# otherwise every spawned child creates a ~400 MB CUDA context on gpus[0].


def _argv_value(*flags):
    """The value after the first of ``flags`` in sys.argv, or None (argparse has not run yet)."""
    return next((sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1]) if a in flags), None)


def _gpus_from_settings_file(sfp):
    """The ``gpus = [...]`` literal assigned inside a function of a settings file (AST only, no execution)."""
    try:
        with open(sfp) as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "gpus":
                            try:
                                return ast.literal_eval(stmt.value)
                            except Exception:
                                pass
    return None


def _gpus_from_run_config(run):
    """``compute.gpus`` of a run folder (its global.yaml) or of a resolved config file; plain YAML, no lisatools."""
    path = Path(run)
    try:
        with open(path / "global.yaml" if path.is_dir() else path) as f:
            raw = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return None
    compute = raw.get("compute") if isinstance(raw, dict) else None
    return compute.get("gpus") if isinstance(compute, dict) else None


def _pre_init_cuda() -> None:
    """Set the CUDA device before any cupy/GPU import.

    Reads the ``gpus`` list from the run config (``--config``) or from the
    settings file (``-sfp``, via AST, no code execution), then calls
    ``cudaSetDevice`` through ctypes so that the CUDA runtime initialises on
    the correct device before cupy is imported anywhere in the module-level
    import chain.

    No-op in multiprocessing children: spawn propagates the parent's argv, so
    without the guard a spawned trainer would set its device to gpus[0] and
    end up with a stray CUDA context on a sampling GPU.
    """
    if multiprocessing.parent_process() is not None:
        return
    run = _argv_value("-c", "--config")
    if run is not None:
        gpus = _gpus_from_run_config(run)
    else:
        sfp = _argv_value("-sfp", "--settings_file_path")
        gpus = None if sfp is None else _gpus_from_settings_file(sfp)
    if not gpus:
        return
    try:
        ctypes.CDLL("libcudart.so").cudaSetDevice(gpus[0])
    except Exception:
        pass


_pre_init_cuda() # avoid allocating GPU memory on unrequested devices.


def curr_info_from_settings_file(file_path, function_name):
    """Legacy path: import a Python settings file and call its settings function."""
    if file_path[-3:] != ".py":
        raise ValueError("Imported settings file must be a python file (.py).")

    module_name = file_path.split("/")[-1].split(".py")[0]

    # Create a module specification from the file location
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    # Create a new module object from the specification
    my_module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules (optional, but good practice for caching)
    sys.modules[module_name] = my_module

    # Execute the module's code
    spec.loader.exec_module(my_module)

    settings_function = getattr(my_module, function_name)
    return settings_function()


def curr_info_from_run_config(run, rank):
    """L2D path: load and validate the run folder, build everything, snapshot the resolved config on the main rank."""
    from lisatools.globalfit.stock.l2d.io import load_run
    from lisatools.globalfit.stock.l2d.load_settings import build_curr_info, write_snapshot

    cfg = load_run(run)
    curr_info = build_curr_info(cfg)
    snapshot = write_snapshot(cfg, curr_info.general_info) if rank == cfg.compute.main_rank else None
    return cfg, curr_info, snapshot


def dry_run_report(cfg, curr_info, snapshot):
    """What a dry run built, in a few lines."""
    general = curr_info.general_info
    lines = [
        f"dry run: {cfg.run.name} built, nothing sampled",
        f"  data: Tobs {general.Tobs:.6g} s, dt {general.dt:.6g} s, band [{general.start_freq:.3g}, "
        f"{general.end_freq:.3g}] Hz, backend {general.force_backend}",
        f"  sampler: {cfg.sampler.nwalkers} walkers x {cfg.sampler.ntemps} temperatures",
    ]
    for name, setup in curr_info.source_info.items():
        block = cfg.blocks[name]
        search = "no search" if block.search is None else f"search {block.search.search_type}"
        flow = getattr(block, "flow", None)
        if flow is not None:
            params = len(flow.parameters) if flow.parameters else setup.ndim
            complement = f" + {flow.complement.class_name}" if flow.complement else ""
            search += f", flow {flow.model.class_name} on {params} params{complement}"
        move = f"pe_move {block.pe_move.class_name}" if hasattr(block, "pe_move") else "sampled by the psd move"
        lines.append(
            f"  {name}: {setup.nleaves_max} leaves x {setup.ndim} params, ladder {len(setup.betas)}, {search}, {move}"
        )
    recipe = cfg.recipe
    lines.append(
        f"  recipe: {recipe.file}:{recipe.function}" if hasattr(recipe, "file")
        else f"  recipe: {recipe.class_name} {recipe.kwargs or ''}".rstrip()
    )
    lines.append(f"  backend file (not created): {general.main_file_path}")
    if snapshot is not None:
        lines.append(f"  resolved config: {snapshot}")
    return "\n".join(lines)


def run(curr_info, comm):
    from lisatools.globalfit.run import GlobalFit

    gf = GlobalFit(curr_info, comm)
    try:
        gf.run_global_fit()
    except KeyboardInterrupt:
        # Attempt a normal (clean) shutdown, but with a deadline: if any
        # thread/finalizer wedges during teardown (e.g. a GIL-released CUDA
        # call, or an mp.Queue feeder flushing to a dead trainer child), a
        # daemon watchdog hard-exits so Ctrl+C always terminates the process.
        # State is safe either way (backups every backup_iter; the flow
        # checkpoint is written atomically).
        import threading

        print(
            "\nKeyboardInterrupt — shutting down (hard exit in 10 s if teardown hangs).",
            file=sys.stderr,
            flush=True,
        )
        watchdog = threading.Timer(10.0, lambda: os._exit(130))
        watchdog.daemon = True
        watchdog.start()
        sys.exit(130)


def make_parser():
    parser = argparse.ArgumentParser(description="Run the LISA Global Fit with LISA Analysis Tools.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-c", "--config", help="An L2D run folder (global.yaml + blocks/) or a resolved config file.")
    source.add_argument("-sfp", "--settings_file_path", help="A Python settings file (legacy).")
    parser.add_argument("-sff", "--settings_function", default="get_global_fit_settings", help="The function in the settings file that will import the settings information (legacy).") # Optional flag
    parser.add_argument("--dry-run", action="store_true", help="With --config: validate and build everything, then exit without sampling.")
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    if args.dry_run and args.config is None:
        raise SystemExit("--dry-run needs --config.")

    # Heavy imports stay in here: spawned children re-import this module but
    # must not pull in mpi4py/lisatools/cupy (each would add startup cost and
    # a CUDA context to every trainer process).
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if args.config is None:
        curr_info = curr_info_from_settings_file(args.settings_file_path, args.settings_function)
    else:
        cfg, curr_info, snapshot = curr_info_from_run_config(args.config, comm.Get_rank())
        if args.dry_run:
            if comm.Get_rank() == cfg.compute.main_rank:
                print(dry_run_report(cfg, curr_info, snapshot))
            return 0
    run(curr_info, comm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
