import os

# MPI-only parallelism (2026-07 policy): pin every threading backend to a
# single thread unless the user overrides explicitly. OMP threading in the
# C++ kernels has caused OOM kills on dev machines, and BLAS pools must be
# set BEFORE numpy loads — hence this block precedes all other imports.
# Parallelism comes from MPI ranks (and GPUs, when configured).
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

import importlib.util
import sys
import argparse

import numpy as np
from mpi4py import MPI
import warnings
from copy import deepcopy

import ast
import ctypes


def _pre_init_cuda() -> None:
    """Set the CUDA device before any cupy/GPU import.

    Parses the settings file path from sys.argv via AST (no code execution)
    to extract the ``gpus`` list, then calls ``cudaSetDevice`` through ctypes
    so that the CUDA runtime initialises on the correct device before cupy
    is imported anywhere in the module-level import chain.
    """
    sfp = next(
        (sys.argv[i + 1] for i, a in enumerate(sys.argv[:-1])
         if a in ("-sfp", "--settings_file_path")),
        None,
    )
    if sfp is None:
        return
    try:
        with open(sfp) as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "gpus":
                            try:
                                gpus = ast.literal_eval(stmt.value)
                                ctypes.CDLL("libcudart.so").cudaSetDevice(gpus[0])
                                return
                            except Exception:
                                pass


_pre_init_cuda() # avoid allocating GPU memory on unrequested devices.

from lisatools.globalfit.run import CurrentInfoGlobalFit, GlobalFit


if __name__ == "__main__":

    # Installed BEFORE prepare_rank (and before argparse/build): a pre-collective
    # crash on one rank must not leave the others hanging in the layout's allgather.
    from lisatools.globalfit.communication.ranks import install_mpi_abort_on_error

    install_mpi_abort_on_error(MPI.COMM_WORLD)

    import argparse
    parser = argparse.ArgumentParser(
        description="Run the LISA Global Fit with LISA Analysis Tools.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "MPI launch matrix (walker-block layout; see docs/global-fit-launch.md):\n"
            "  np 1: single rank. np 2 on 1 GPU: head + saver (warns). np 2 on 2\n"
            "  GPUs: head + compute. np >= 3: head + compute ranks + saver (highest\n"
            "  rank). prepare_rank resolves this rank's role before the build.\n"
            "  python scripts/run_global.py --stock <name>\n"
            "  mpiexec -n 3 python scripts/run_global.py --stock <name>\n"
            "  srun -n 3 --gpus-per-node=<G> python scripts/run_global.py --stock <name>\n"
            "GPU count is a knob, not a rank: GPUS=0,1 selects the local devices the\n"
            "main rank drives (USE_GPU=0 forces CPU). Common env: VERBOSE, NWALKERS, NTEMPS,\n"
            "NUM_ITERATIONS, DATA_MODE, TOBS_TARGET, MAKE_DIAGNOSTIC_PLOTS."
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-sfp", "--settings_file_path", help="A settings file (legacy path-loading).")
    group.add_argument(
        "--stock",
        help=(
            "A stock run variant from lisatools.globalfit.stock.erebor "
            "(e.g. gb_no_fg, all_sources, full_year_combined). Knobs come "
            "from the variant's env-backed defaults; adjust anything else "
            "through the class API in a driver script."
        ),
    )
    parser.add_argument("-sff", "--settings_function", default="get_global_fit_settings", help="The function in the settings file that will import the settings information.") # Optional flag

    args = parser.parse_args()

    if args.stock is not None:
        from lisatools.globalfit.stock import erebor

        fit = erebor.get_stock(args.stock)  # cheap: validation + defaults only
        # TOBS_TARGET honored (ported from run_combined_staged.py's
        # 2026-08-13 fix, which fixed this for the staged driver only).
        # Several variants pin a FIXED WDM grid via the legacy (nf, nt)
        # override -- all_sources 720x2160, noise_mojito 1440x2160, both
        # 90 d -- and by the settings contract that BEATS
        # general.tobs_target (fit.py: "if gs.nf is not None and gs.nt is
        # not None"). So TOBS_TARGET was SILENTLY IGNORED on this driver,
        # even though the --help epilog advertises it: the noise-only PSD
        # run's TOBS_TARGET=7776000 export only "worked" because
        # 1440*2160*2.5 s is 90 d anyway. When the env asks for a Tobs,
        # clear the fixed grid so the build derives (Nf, Nt) from
        # tobs_target + the wavelet-duration bounds -- the same machinery
        # that reproduces 1440x2160 exactly at the 90-d default, so this
        # is a no-op for every run that was already asking for 90 d.
        # Unset env -> behavior unchanged.
        if os.environ.get("TOBS_TARGET", "").strip():
            fit.general.nf = None
            fit.general.nt = None
            print(
                f"[stock] TOBS_TARGET={fit.general.tobs_target:.6g} s: "
                "cleared any fixed-grid (nf, nt) override so the WDM grid "
                "derives from tobs_target.",
                flush=True,
            )
        # Every rank builds: in the walker-block layout the head and the
        # computation ranks each own a block of walkers, and the saver needs
        # the backend spec. prepare_rank resolves this rank's role and pins
        # its device BEFORE the build allocates on it (no-op at -n 1).
        from lisatools.globalfit.communication.ranks import prepare_rank

        prepare_rank(fit, MPI.COMM_WORLD)
        curr_info = fit.build()
    else:
        # Define the module name and the full path to the Python file
        file_path = args.settings_file_path
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

        # Now you can access functions, classes, or variables from the imported module
        settings_function = getattr(my_module, args.settings_function)

        # Settings-file runs predate the walker-block layout: keep today's
        # roles (one sampling rank, stopped spares) unless the user opted in.
        if os.environ.setdefault("GF_LEGACY_RANK_LAYOUT", "1") == "1":
            print("[legacy settings file] GF_LEGACY_RANK_LAYOUT=1: single-compute layout.",
                  flush=True)

        curr_info = settings_function()

    gf = GlobalFit(curr_info, MPI.COMM_WORLD)
    gf.run_global_fit()
    #breakpoint()