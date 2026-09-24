"""Real-MPI layout preflight: ``build_layout`` across N processes, then stop.

Gate G1 of ``docs/gpu-routing-test-campaign.md`` -- the cheapest test that
exercises the layout OUTSIDE ``FakeWorld``: real mpi4py, real processes, a
real ``allgather``. It catches anything that only works because FakeWorld's
``Split``/``allgather`` are in-process, and it costs about ten seconds.

    export GF_GPU_ROUTING=1                   # the routing is OPT-IN
    export GF_LAYOUT_DRY_RUN=1 GPUS=""        # CPU: no GPU-pool capacity check
    NWALKERS=2 mpiexec --oversubscribe -n 5 python gf_layout_preflight_mpi.py

``LAT_SRC`` is OPTIONAL and normally unset: on a plain checkout the installed
``lisatools`` is the one under test, and pinning a path would be a way to
preflight code the run will not use.

    export LAT_SRC=<worktree>/src
    # IN A WORKTREE, the .wtenv shim is REQUIRED: the editable install's
    # meta-path finder hard-maps every ``lisatools.*`` module to the MAIN
    # checkout and beats sys.path, so without these two the script silently
    # imports the installed package and tests the wrong code.
    export PYTHONPATH=<worktree>/.wtenv LAT_WORKTREE_SRC=$LAT_SRC

Prints the resolved factorization, asserts every rank agrees on
``layout.digest()``, and dumps the per-rank table. ``GPUS=0,1`` instead
exercises the per-node GPU-pool capacity check (which will refuse more
compute ranks than the pool can host -- that refusal is the check working).

``GF_GPU_ROUTING`` is honoured exactly as the engine honours it, because
preflighting a rule the run will not apply is worse than not preflighting at
all. Unset, a shape needing the unified factorization REFUSES here just as it
would at launch, and the printed line says ``gpu_routing=OFF(legacy)``.

EVERY OUTCOME PRINTS A ``###`` LINE ON STDOUT, INCLUDING FAILURE. Runbooks
pipe this through ``grep '^###'``, and a filter that only matches success
makes a crash, a refused shape and a silent pass look identical -- an empty
result then reads as "nothing happened" when it actually means "it broke".
So the layout build is wrapped: on any exception rank 0 prints
``### FAILED ...`` and the process exits non-zero.

Nothing here builds a fit or touches a store.
"""
import os, sys, traceback
from mpi4py import MPI
if os.environ.get("LAT_SRC"):
    sys.path.insert(0, os.environ["LAT_SRC"])
from lisatools.globalfit.communication.ranks import build_layout, layout_dry_run

comm = MPI.COMM_WORLD
_rank, _size = comm.Get_rank(), comm.Get_size()
nw = int(os.environ.get("NWALKERS", "4"))
pool = [int(x) for x in os.environ.get("GPUS", "0,1").split(",") if x != ""]
rpb = os.environ.get("RANKS_PER_BLOCK") or None
# A SIZE-1 WORLD IS THE COMMONEST CLUSTER FAILURE and it is not an error
# here -- build_layout resolves it happily as a single-process run -- so it
# has to be SAID. It means the launcher started N independent jobs rather
# than one N-rank world (Intel MPI without I_MPI_HYDRA_BOOTSTRAP=slurm is
# the usual cause), and every agreement result below would be vacuous.
if _size == 1:
    print(f"### WARNING: MPI world size is 1 -- the launcher did NOT make one "
          f"multi-rank world. Nothing below tests agreement. Check the MPI "
          f"bootstrap pins.", flush=True)
try:
    lay = build_layout(comm, nw, pool, legacy=False,
                       ranks_per_block=None if rpb is None else int(rpb))
except Exception as exc:
    if _rank == 0:
        print(f"### FAILED nwalkers={nw} size={_size} "
              f"GF_GPU_ROUTING={os.environ.get('GF_GPU_ROUTING', '<unset>')} "
              f"RANKS_PER_BLOCK={rpb or 'AUTO'}: {type(exc).__name__}: {exc}",
              flush=True)
        traceback.print_exc()
    # NO Barrier HERE. build_layout raises the same ValueError on every rank
    # (the rule is pure and every rank has identical inputs), but if that ever
    # became one-sided, a Barrier in the error path would turn a clear failure
    # into a HANG -- strictly worse than interleaved output.
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(1)
if _rank == 0:
    print(f"### nwalkers={nw} n_compute={lay.n_compute} "
          f"n_blocks={lay.n_blocks} R={lay.ranks_per_block} block={lay.block} "
          f"gpu_routing={'on' if lay.gpu_routing else 'OFF(legacy)'}", flush=True)
digests = comm.allgather(lay.digest())
if _rank == 0:
    print("### all ranks agree on the layout:", len(set(digests)) == 1, flush=True)


def _say(*a):
    """FLUSHING printer for layout_dry_run.

    Every print in this script flushes, and that is not fussiness. Python
    block-buffers stdout when it is a PIPE rather than a tty, so
    ``mpiexec ... | grep`` buffers a few hundred bytes and holds them -- and
    if the job then dies (an MPI abort SIGKILLs the siblings), the buffer
    goes with it. The command prints NOTHING, which reads as "no output, so
    nothing happened" when it actually means "it died before flushing".
    Exactly how this script appeared to do nothing on the cluster on
    2026-09-23. ``python -u`` fixes it from the caller's side; this fixes it
    from the script's, so no runbook has to remember.
    """
    print(*a, flush=True)


layout_dry_run(lay, comm, out=_say)
sys.stdout.flush()
