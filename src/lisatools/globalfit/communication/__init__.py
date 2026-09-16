"""Multi-rank communication layer of the global fit.

One MPI rank per GPU (general in both directions via ``gpus_per_rank`` /
``ranks_per_gpu``), equal static walker blocks, a head rank that sequences
the recipe and also computes block 0, computation ranks that serve
head-directed commands, and the unchanged saver rank. See
``docs/superpowers/specs/2026-09-15-multirank-walker-blocks-design.md``.

``mpi4py`` is imported lazily and only for real communicators; the
:mod:`fakecomm` simulator runs the same code in one process for tests.
"""

from .fakecomm import FakeAbort, FakeComm, FakeCommNull, FakeWorld
from .fanout import (
    LIKELIHOOD_OP,
    PING_OP,
    STOP_OP,
    ComputeService,
    RemoteWorkerError,
    WalkerFanout,
    concat_blocks,
)
from .ranks import (
    LEGACY_ENV,
    RankPlacement,
    RankRole,
    WalkerBlockLayout,
    build_layout,
    derive_rank_seed,
    install_mpi_abort_on_error,
    prefix_stdout,
    prepare_rank,
    rank_tag,
    resolve_roles,
    select_rank_device,
)
from .walkerslice import WALKER_INDS_KEY, merge_state, slice_state

__all__ = [
    "WALKER_INDS_KEY",
    "merge_state",
    "slice_state",
    "FakeAbort",
    "FakeComm",
    "FakeCommNull",
    "FakeWorld",
    "LIKELIHOOD_OP",
    "PING_OP",
    "STOP_OP",
    "ComputeService",
    "RemoteWorkerError",
    "WalkerFanout",
    "concat_blocks",
    "LEGACY_ENV",
    "RankPlacement",
    "RankRole",
    "WalkerBlockLayout",
    "build_layout",
    "derive_rank_seed",
    "install_mpi_abort_on_error",
    "prefix_stdout",
    "prepare_rank",
    "rank_tag",
    "resolve_roles",
    "select_rank_device",
]
