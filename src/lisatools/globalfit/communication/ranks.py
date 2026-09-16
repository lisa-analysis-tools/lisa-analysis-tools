"""Rank roles, walker-block layout, per-rank device pinning and seeds.

Roles: HEAD (rank 0: sequences the recipe, owns the full host state, AND
computes walker block 0), COMPUTE (one device list + one walker block each),
SAVER (highest rank, unchanged; aliased to the head below 3 ranks). The
mapping between ranks and GPUs is general in both directions:
``gpus_per_rank`` (a rank owns several devices, sharded in-process) and
``ranks_per_gpu`` (several ranks share one device); at most one exceeds 1.

``GF_LEGACY_RANK_LAYOUT=1`` restores today's roles (one compute rank owning
the whole pool, other non-saver ranks are stopped SPARE ranks).

``mpi4py`` is imported lazily and only for real communicators.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import os
import sys
import threading
import traceback
import warnings

import numpy as np

LEGACY_ENV = "GF_LEGACY_RANK_LAYOUT"


class RankRole(enum.Enum):
    HEAD = "head"
    COMPUTE = "compute"
    SAVER = "saver"
    SPARE = "spare"  # legacy layout only


@dataclasses.dataclass(frozen=True)
class RankPlacement:
    rank: int
    role: RankRole
    node: str
    local_index: int
    #: the rank's device list in the per-node pool's numbering (empty on CPU);
    #: several ranks list the same device when ranks_per_gpu > 1
    devices: tuple
    #: this rank's index among the ranks sharing its device
    device_slot: int
    w0: int
    w1: int


@dataclasses.dataclass(frozen=True)
class WalkerBlockLayout:
    size: int
    head_rank: int
    saver_rank: int
    compute_ranks: tuple
    nwalkers: int
    block: int
    placements: dict
    gpus_per_rank: int = 1
    ranks_per_gpu: int = 1
    legacy: bool = False
    #: human-readable notes about non-default choices (e.g. the size-2 fallback)
    notes: tuple = ()

    @property
    def n_compute(self) -> int:
        return len(self.compute_ranks)

    @property
    def worker_ranks(self) -> tuple:
        return tuple(r for r in self.compute_ranks if r != self.head_rank)

    def is_single(self) -> bool:
        return self.n_compute == 1

    def role_of(self, rank) -> RankRole:
        return self.placements[int(rank)].role

    def block_of(self, rank) -> tuple:
        p = self.placements[int(rank)]
        return p.w0, p.w1

    def local_gpus(self, rank):
        devices = self.placements[int(rank)].devices
        return list(devices) if devices else None

    def fanout_rank(self, rank) -> int:
        """This rank's index in the fan-out communicator (== compute-rank order)."""
        return self.compute_ranks.index(int(rank))

    def ranks_on_device(self, node, device) -> tuple:
        return tuple(
            r
            for r, p in sorted(self.placements.items())
            if p.node == str(node)
            and int(device) in p.devices
            and p.role in (RankRole.HEAD, RankRole.COMPUTE)
        )

    def make_fanout_comm(self, comm):
        """Split ``comm`` into the head+compute communicator (saver: null comm)."""
        rank = int(comm.Get_rank())
        color = 0 if rank in self.compute_ranks else _undefined(comm)
        return comm.Split(color, key=rank)

    def describe(self) -> str:
        head = (
            f"walker-block layout: size={self.size} n_compute={self.n_compute} "
            f"nwalkers={self.nwalkers} block={self.block} "
            f"gpus_per_rank={self.gpus_per_rank} ranks_per_gpu={self.ranks_per_gpu}"
            f"{' LEGACY' if self.legacy else ''}"
        )
        lines = [head]
        for r in range(self.size):
            p = self.placements[r]
            lines.append(
                f"  r{r:<3d} {p.role.value:<7s} node={p.node} local={p.local_index} "
                f"devices={list(p.devices)} slot={p.device_slot} walkers=[{p.w0},{p.w1})"
            )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def digest(self) -> str:
        return hashlib.sha256(self.describe().encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# communicator shims (mpi4py only when the comm is real)
# --------------------------------------------------------------------------


def _comm_type_shared(comm):
    value = getattr(comm, "COMM_TYPE_SHARED", None)
    if value is not None:
        return value
    from mpi4py import MPI

    return MPI.COMM_TYPE_SHARED


def _undefined(comm):
    value = getattr(comm, "UNDEFINED", None)
    if value is not None:
        return value
    from mpi4py import MPI

    return MPI.UNDEFINED


def _proc_name(comm) -> str:
    fn = getattr(comm, "Get_processor_name", None)
    if fn is not None:
        return str(fn())
    from mpi4py import MPI

    return str(MPI.Get_processor_name())


# --------------------------------------------------------------------------
# roles + layout
# --------------------------------------------------------------------------


def resolve_roles(size, main_rank=0):
    """``(head, saver, compute_ranks)``: saver = highest non-head rank at
    ``size >= 3`` (aliased to the head below that, as today); every other
    rank computes, the head included. No spares."""
    size = int(size)
    if size < 1:
        raise ValueError("communicator size must be >= 1")
    head = int(main_rank)
    if not (0 <= head < size):
        raise ValueError(f"main_rank {head} is not a rank of a size-{size} communicator")
    others = [r for r in range(size) if r != head]
    saver = others[-1] if size >= 3 else head
    compute = tuple(r for r in range(size) if r == head or r != saver)
    return head, saver, compute


def build_layout(
    comm,
    nwalkers,
    gpu_pool,
    *,
    gpus_per_rank=1,
    ranks_per_gpu=1,
    main_rank=0,
    legacy=None,
):
    """Resolve the identical layout on every rank (one collective ``allgather``).

    ``gpu_pool`` is the PER-NODE device list (the ``GPUS`` setting). Compute
    ranks on a node are ordered by world rank and assigned blocked:
    ``gpus_per_rank = k > 1`` -> devices ``pool[i*k:(i+1)*k]``;
    ``ranks_per_gpu = m > 1`` -> device ``pool[i // m]``, slot ``i % m``.
    """
    if legacy is None:
        legacy = os.environ.get(LEGACY_ENV, "0") == "1"
    size = int(comm.Get_size())
    rank = int(comm.Get_rank())
    head, saver, compute = resolve_roles(size, main_rank)
    k, m = int(gpus_per_rank), int(ranks_per_gpu)
    if k < 1 or m < 1:
        raise ValueError("gpus_per_rank and ranks_per_gpu must both be >= 1")
    if k > 1 and m > 1:
        raise ValueError("at most one of gpus_per_rank / ranks_per_gpu may exceed 1")
    pool = [int(g) for g in (gpu_pool or [])]
    if legacy:
        compute = (head,)
    notes = []
    if size == 2 and not legacy and pool and len(pool) * m // k < 2:
        # A `-n 2` launch on a pool that cannot host two compute ranks: instead
        # of the over-subscription error, rank 1 becomes the dedicated saver
        # (user ruling 2026-09-15). The head then computes every walker exactly
        # as a single-rank run does.
        other = [r for r in range(size) if r != head][0]
        saver = other
        compute = (head,)
        note = (
            f"size-2 launch on a per-node GPU pool {pool} that supports only "
            f"{len(pool) * m // k} compute rank(s): rank {other} runs as the "
            "dedicated saver and the head computes all walkers. To use two "
            "compute ranks on this pool set RANKS_PER_GPU=2; for synchronous "
            "saves with no saver rank launch with -n 1."
        )
        notes.append(note)
        warnings.warn(note, UserWarning, stacklevel=2)
    nwalkers = int(nwalkers)
    n_compute = len(compute)
    if nwalkers % n_compute:
        raise ValueError(
            f"nwalkers={nwalkers} is not divisible by the compute-rank count {n_compute}: "
            "equal walker blocks are required (pick NWALKERS as a multiple of it)."
        )
    block = nwalkers // n_compute

    if size == 1 or not hasattr(comm, "Split_type"):
        table = [(_proc_name(comm), 0)]
    else:
        node_comm = comm.Split_type(_comm_type_shared(comm), key=rank)
        table = comm.allgather((_proc_name(comm), int(node_comm.Get_rank())))
        free = getattr(node_comm, "Free", None)
        if free is not None:
            free()

    by_node = {}
    for r in range(size):
        by_node.setdefault(str(table[r][0]), []).append(r)

    placements = {}
    for node, ranks in by_node.items():
        comp_here = [r for r in ranks if r in compute]
        if pool and not legacy:
            capacity = len(pool) * m // k
            if len(comp_here) > capacity:
                raise ValueError(
                    f"node {node}: {len(comp_here)} compute ranks but the per-node GPU pool "
                    f"{pool} supports at most {capacity} "
                    f"(gpus_per_rank={k}, ranks_per_gpu={m})"
                )
        for r in ranks:
            local_index = int(table[r][1])
            if r == head:
                role = RankRole.HEAD
            elif r == saver and saver != head:
                role = RankRole.SAVER
            elif r in compute:
                role = RankRole.COMPUTE
            else:
                role = RankRole.SPARE
            if role in (RankRole.SAVER, RankRole.SPARE):
                # builds on a device like everyone else, then releases it
                devices = (pool[local_index % len(pool)],) if pool else ()
                placements[r] = RankPlacement(r, role, node, local_index, devices, 0, 0, 0)
                continue
            i = comp_here.index(r)
            if not pool:
                devices, slot = (), 0
            elif legacy:
                devices, slot = tuple(pool), 0
            elif k > 1:
                devices, slot = tuple(pool[i * k : (i + 1) * k]), 0
            else:
                devices, slot = (pool[i // m],), i % m
            bi = compute.index(r)
            placements[r] = RankPlacement(
                r, role, node, local_index, devices, slot, bi * block, (bi + 1) * block
            )
    return WalkerBlockLayout(
        size=size,
        head_rank=head,
        saver_rank=saver,
        compute_ranks=tuple(compute),
        nwalkers=nwalkers,
        block=block,
        placements=placements,
        gpus_per_rank=k,
        ranks_per_gpu=m,
        legacy=bool(legacy),
        notes=tuple(notes),
    )


def derive_rank_seed(base_seed, layout, rank) -> int:
    """Per-compute-rank seed: ``SeedSequence(base).spawn(n_compute)[i]``, deterministic."""
    sequence = np.random.SeedSequence([int(base_seed), 0x5AFE])
    children = sequence.spawn(layout.n_compute)
    child = children[layout.fanout_rank(rank)]
    return int(child.generate_state(1, dtype=np.uint32)[0])


# --------------------------------------------------------------------------
# device pinning (must run BEFORE any CUDA initialisation on the rank)
# --------------------------------------------------------------------------

_CUDART_NAMES = (
    "libcudart.so",
    "libcudart.so.13",
    "libcudart.so.12",
    "libcudart.so.11.0",
    "libcudart.dylib",
)


def _load_cudart():
    import ctypes

    for name in _CUDART_NAMES:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _cudart_device_count():
    """``cudaGetDeviceCount`` via ctypes, or ``None`` when no runtime is loadable."""
    import ctypes

    lib = _load_cudart()
    if lib is None:
        return None
    count = ctypes.c_int(0)
    if lib.cudaGetDeviceCount(ctypes.byref(count)) != 0:
        return None
    return int(count.value)


def _cudart_set_device(device):
    lib = _load_cudart()
    if lib is None:
        raise RuntimeError("cudaSetDevice fallback requested but no CUDA runtime is loadable")
    if lib.cudaSetDevice(int(device)) != 0:
        raise RuntimeError(f"cudaSetDevice({device}) failed")


def select_rank_device(
    layout,
    rank,
    *,
    environ=None,
    device_count_fn=None,
    set_device_fn=None,
    logger=None,
):
    """Pin this process to its devices. Returns ``(gpus, mode)``.

    ``gpus`` is what ``general_info.gpus`` must become on this rank
    (``None`` on CPU). Modes: ``"cpu"``; ``"legacy"`` (pool untouched);
    ``"visible"`` (``CUDA_VISIBLE_DEVICES`` narrowed, the rank sees its
    devices as ``0..k-1``); ``"setdevice"`` (the runtime was already
    initialised, e.g. by a CUDA-aware MPI, so the env is restored and
    ``cudaSetDevice`` pins the first device; the rank keeps the pool ids).
    """
    environ = os.environ if environ is None else environ
    device_count_fn = _cudart_device_count if device_count_fn is None else device_count_fn
    set_device_fn = _cudart_set_device if set_device_fn is None else set_device_fn
    placement = layout.placements[int(rank)]
    if not placement.devices:
        return None, "cpu"
    if layout.legacy:
        return list(placement.devices), "legacy"

    previous = environ.get("CUDA_VISIBLE_DEVICES")
    if previous is None:
        # the env var is genuinely unset: the pool ids are physical device ids
        physical = [str(d) for d in placement.devices]
    else:
        # SET (possibly to "", meaning explicitly NO devices visible): pool ids
        # index the CURRENTLY visible set (Slurm may already have narrowed it,
        # e.g. via --gpus-per-task/--gpu-bind, down to as little as one device)
        visible = [v.strip() for v in previous.split(",") if v.strip()]
        try:
            physical = [visible[d] for d in placement.devices]
        except IndexError:
            raise ValueError(
                f"rank {rank}: device pool ids {list(placement.devices)} index the "
                f"inherited CUDA_VISIBLE_DEVICES={previous!r} (visible devices={visible!r}), "
                "which does not have that many entries. This usually means the launcher "
                "already bound this task to its own device(s) (Slurm "
                "--gpus-per-task/--gpu-bind), so the layout's pool ids double-count that "
                "binding. Fix by either launching without per-task GPU binding "
                "(--gpu-bind=none) so every rank sees the full node pool, or by setting "
                "GPUS to the per-task pool this rank should see (matching what "
                "CUDA_VISIBLE_DEVICES already narrowed it to)."
            ) from None
    environ["CUDA_VISIBLE_DEVICES"] = ",".join(physical)

    count = device_count_fn()
    if count is None or count == len(placement.devices):
        return list(range(len(placement.devices))), "visible"

    # the runtime saw the pool before we narrowed the env: fall back to cudaSetDevice
    if previous is None:
        environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        environ["CUDA_VISIBLE_DEVICES"] = previous
    set_device_fn(placement.devices[0])
    environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if logger is not None:
        logger.warning(
            "rank %d: CUDA runtime already initialised (%d devices visible); pinned device %d "
            "with cudaSetDevice instead of CUDA_VISIBLE_DEVICES",
            rank,
            count,
            placement.devices[0],
        )
    return list(placement.devices), "setdevice"


def prepare_rank(
    fit,
    comm,
    *,
    logger=None,
    environ=None,
    device_count_fn=None,
    set_device_fn=None,
):
    """THE driver hook: resolve the layout and pin this rank's device BEFORE ``fit.build()``.

    Idempotent (returns the stored layout on a second call). Reads the
    pre-build settings ``fit.general.nwalkers`` / ``.gpus`` (the per-node
    pool) / ``.gpus_per_rank`` / ``.ranks_per_gpu``; writes the rank-local
    ``fit.general.gpus``, ``fit.rank_layout`` and ``fit.rank_device_mode``.
    """
    layout = getattr(fit, "rank_layout", None)
    if layout is not None:
        return layout
    if int(comm.Get_size()) > 1 and bool(getattr(fit, "built", False)):
        raise RuntimeError(
            "prepare_rank must run BEFORE fit.build(): the build allocates on the device"
        )
    general = fit.general
    pool = list(general.gpus) if getattr(general, "gpus", None) else []
    layout = build_layout(
        comm,
        int(general.nwalkers),
        pool,
        gpus_per_rank=int(getattr(general, "gpus_per_rank", 1) or 1),
        ranks_per_gpu=int(getattr(general, "ranks_per_gpu", 1) or 1),
        main_rank=int(getattr(fit, "main_rank", 0) or 0),
    )
    # capture BEFORE select_rank_device narrows/restores it in place
    env = os.environ if environ is None else environ
    inherited_cvd = env.get("CUDA_VISIBLE_DEVICES", "<unset>")
    gpus, mode = select_rank_device(
        layout,
        int(comm.Get_rank()),
        environ=environ,
        device_count_fn=device_count_fn,
        set_device_fn=set_device_fn,
        logger=logger,
    )
    if pool:
        general.gpus = gpus
    fit.rank_layout = layout
    fit.rank_device_mode = mode
    if logger is not None:
        logger.info(
            "%s\nrank %d device mode: %s (inherited CUDA_VISIBLE_DEVICES=%s)",
            layout.describe(),
            comm.Get_rank(),
            mode,
            inherited_cvd,
        )
    return layout


# --------------------------------------------------------------------------
# failure + logging helpers
# --------------------------------------------------------------------------


def install_mpi_abort_on_error(comm):
    """Route any rank's uncaught exception (main thread or threads) through ``comm.Abort``.

    Under ``mpiexec`` a crashed rank otherwise leaves the survivors blocked in
    their receive loops for the whole allocation. Returns ``comm`` when it
    installed the hooks, ``None`` for a single-process run (normal Python
    exception behaviour is kept there).
    """
    if comm is None or int(comm.Get_size()) < 2:
        return None
    rank = int(comm.Get_rank())
    size = int(comm.Get_size())

    def _hook(exc_type, exc, tb):
        try:
            print(
                f"\n[MPI-ABORT] rank {rank} of {size} raised {exc_type.__name__}: {exc}\n"
                "[MPI-ABORT] aborting ALL ranks so the job fails fast instead of hanging.",
                file=sys.stderr,
                flush=True,
            )
            if exc_type is not KeyboardInterrupt:
                traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
            sys.stderr.flush()
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 - never mask the abort
            pass
        finally:
            try:
                comm.Abort(1)
            except Exception:  # noqa: BLE001
                os._exit(1)

    sys.excepthook = _hook
    if hasattr(threading, "excepthook"):

        def _thread_hook(args):
            _hook(args.exc_type, args.exc_value, args.exc_traceback)

        threading.excepthook = _thread_hook
    return comm


def rank_tag(layout, rank) -> str:
    role = layout.role_of(rank)
    if role == RankRole.HEAD:
        return f"r{int(rank)}/head"
    if role == RankRole.SAVER:
        return f"r{int(rank)}/saver"
    if role == RankRole.SPARE:
        return f"r{int(rank)}/spare"
    return f"r{int(rank)}/c{layout.fanout_rank(rank)}"


class _PrefixedStream:
    """Line-prefixing wrapper for a text stream (worker-rank stdout)."""

    def __init__(self, stream, prefix):
        self._stream = stream
        self._prefix = prefix
        self._at_line_start = True

    def write(self, text):
        out = []
        for chunk in str(text).splitlines(keepends=True):
            if self._at_line_start:
                out.append(self._prefix)
            out.append(chunk)
            self._at_line_start = chunk.endswith("\n")
        self._stream.write("".join(out))

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return getattr(self._stream, name)


def prefix_stdout(tag):
    """Prefix every stdout line of this process with ``[tag] `` (idempotent)."""
    if not isinstance(sys.stdout, _PrefixedStream):
        sys.stdout = _PrefixedStream(sys.stdout, f"[{tag}] ")
    return sys.stdout
