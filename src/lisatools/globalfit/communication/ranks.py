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
            if p.node == str(node) and int(device) in p.devices and p.role != RankRole.SAVER
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
    )


def derive_rank_seed(base_seed, layout, rank) -> int:
    """Per-compute-rank seed: ``SeedSequence(base).spawn(n_compute)[i]``, deterministic."""
    sequence = np.random.SeedSequence([int(base_seed), 0x5AFE])
    children = sequence.spawn(layout.n_compute)
    child = children[layout.fanout_rank(rank)]
    return int(child.generate_state(1, dtype=np.uint32)[0])
