"""Head-directed fan-out over walker blocks + the compute-rank service loop.

Sampling phase: the head calls :meth:`WalkerFanout.run` inside a proposal;
each computation rank receives one command, runs the same body on its own
walker block and replies; the head merges. With ONE compute rank ``run`` is
a direct call (no communicator, no pickling, no copies). Setup phase:
:meth:`WalkerFanout.allgather_walker_vector` is a symmetric collective every
compute rank calls at the same program point.

Command / reply schemas (host numpy, pickled by mpi4py ``send``/``recv``)::

    cmd   = {"seq", "op", "move", "clock", "payload", "shared"}
    reply = {"seq", "rank", "ok", "result", "wall_s",
             "error": None | {"type", "msg", "traceback"}}
"""

from __future__ import annotations

import collections
import hashlib
import logging
import time
import traceback

import numpy as np

# Module logger: propagates to the ``lisatools`` file handler, i.e. the head's
# ``globalfit_run.log`` -- the file the diagnostics (``summarize_fanout``) and
# the WP7 runbook read. The ``GlobalFit`` logger handed in at construction
# writes ``global_fit.log`` with propagation OFF, which is where the
# ``[FANOUT]`` load-balance line hid until 2026-09-16 (cluster Step 4).
_LOG = logging.getLogger(__name__)

from ...utils.utility import asnumpy

STOP_OP = "stop"
PING_OP = "ping"
#: builtin every ComputeService installs: the rank's current per-walker
#: likelihood over its block (see WalkerFanout.gather_likelihood).
LIKELIHOOD_OP = "likelihood"
#: builtin: sha1 of the rank's residual buffers (replica agreement check)
RESIDUAL_HASH_OP = "residual_hash"


class RemoteWorkerError(RuntimeError):
    """A computation rank's command failed; carries the remote traceback text."""

    def __init__(self, rank, op, move, error):
        self.rank = int(rank)
        self.op = op
        self.move = move
        self.error = dict(error or {})
        self.remote_traceback = self.error.get("traceback", "")
        super().__init__(
            f"rank {self.rank} failed op={op!r} move={move!r}: "
            f"{self.error.get('type')}: {self.error.get('msg')}\n{self.remote_traceback}"
        )


def concat_blocks(results, layout):
    """Concatenate per-rank 1-D results in compute-rank (== walker) order.

    Replica mode (one walker on every compute rank): the head's block IS the
    whole vector; the replicas hold copies.
    """
    if getattr(layout, "replica_mode", False):
        return np.asarray(results[layout.head_rank])
    return np.concatenate([np.asarray(results[r]) for r in layout.compute_ranks])


def _reply(seq, rank, ok, result, wall_s, error=None):
    return {"seq": seq, "rank": rank, "ok": ok, "result": result, "wall_s": wall_s, "error": error}


def _sha1_16(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


def _array_bytes(arr) -> bytes:
    """Host bytes for one array (device arrays pulled via ``asnumpy``)."""
    return np.ascontiguousarray(asnumpy(arr)).tobytes()


def residual_hash(acs) -> str:
    """16-hex sha1 of the ACA's residual AND noise-model buffers.

    Hashes ``linear_data_arr`` (the residual) *and* ``linear_psd_arr`` (the
    noise covariance) when present, so a PSD-only drift between replicas is
    caught too, not just a residual drift; falls back to the likelihood
    vector when neither buffer is available.
    """
    arrs = (
        list(getattr(acs, "linear_data_arr", []) or [])
        + list(getattr(acs, "linear_psd_arr", []) or [])
    )
    if not arrs:
        arrs = [acs.likelihood(complex=False)]
    return _sha1_16(b"".join(_array_bytes(a) for a in arrs))


def fanout_digest_line(iteration, state, residual_hashes=None) -> str:
    """``[FANOUT_DIGEST] it=<n> log_like=<sha1> coords=<sha1> inds=<sha1>``.

    Cluster-gate tool (``docs/multirank-cluster-gates.md`` Step 1): the head
    logs this once per iteration under ``GF_FANOUT_DIGEST=1`` so three
    transport-parity layouts (shared-GPU / multi-GPU-one-node / multi-node)
    can be diffed line-for-line for bit-identical merged state. Host numpy
    only; ``coords``/``inds`` each hash the byte-concatenation of every
    branch's array in branch-name-sorted order (branches differ in shape, so
    concatenating raw bytes -- not ``np.concatenate`` -- is what makes a
    single hash possible across all of them), independent of dict order.

    ``residual_hashes`` (one-walker replica mode): ``{world_rank: hash}`` from
    :meth:`WalkerFanout.gather_residual_hashes`; when given, the line gains
    `` residual=r<rank>:<hash>,... replicas_agree=<bool>`` so replica drift is
    visible on the same line as the merged-state digest.
    """
    log_like_hash = _sha1_16(_array_bytes(state.log_like))
    coords = state.branches_coords
    coords_hash = _sha1_16(b"".join(_array_bytes(coords[name]) for name in sorted(coords)))
    inds = state.branches_inds
    inds_hash = _sha1_16(b"".join(_array_bytes(inds[name]) for name in sorted(inds)))
    line = (
        f"[FANOUT_DIGEST] it={int(iteration)} log_like={log_like_hash} "
        f"coords={coords_hash} inds={inds_hash}"
    )
    if residual_hashes:
        items = ",".join(f"r{int(r)}:{h}" for r, h in sorted(residual_hashes.items()))
        agree = len(set(residual_hashes.values())) == 1
        line += f" residual={items} replicas_agree={agree}"
    return line


class WalkerFanout:
    """Head-side fan-out (usable on every compute rank for the collectives)."""

    def __init__(self, comm, layout, rank, *, model=None, logger=None):
        self.comm = comm
        self.layout = layout
        self.rank = int(rank)
        self.model = model
        self.logger = logger
        self.head = layout.head_rank
        self.is_head = self.rank == self.head
        self.single = layout.is_single()
        if not self.single and comm is not None:
            # `comm` must be the compute-only fan-out comm (layout.make_fanout_comm),
            # never the world comm; `None` is a test-only placeholder that skips this
            # check entirely (see tests/test_fanout_fakecomm.py::test_run_on_a_worker_raises).
            size = int(comm.Get_size())
            if size != layout.n_compute:
                raise ValueError(
                    f"WalkerFanout comm size {size} != layout.n_compute {layout.n_compute}; "
                    "pass the compute-only fan-out comm (layout.make_fanout_comm(comm)), "
                    "not the world/full comm"
                )
        self.seq = 0
        self.last_wait_s = 0.0
        self._stopped = False
        self.clock = {
            "iteration": 0,
            "stage": None,
            "stage_kind": None,
            "move": None,
            "call_index": 0,
            "seq": 0,
            "seed_base": None,
        }
        self._call_index = collections.Counter()

    # -- clock -----------------------------------------------------------
    def enter_stage(self, name, kind):
        self.clock["stage"] = name
        self.clock["stage_kind"] = kind

    def note_iteration(self, iteration):
        self.clock["iteration"] = int(iteration)

    # -- commands --------------------------------------------------------
    def gather_likelihood(self, acs):
        """Current per-walker likelihood over ALL walkers (head-only, sampling phase).

        The head scores its own block from ``acs`` (its B-row ACA); every worker
        answers through the ``LIKELIHOOD_OP`` builtin its ComputeService installs.
        Any head-side "all walkers" read of the residual during sampling must go
        through here -- the head's ACA holds only its block. With one compute
        rank this is a direct call.
        """
        return self.run(
            LIKELIHOOD_OP,
            move=None,
            per_rank_payload=lambda rank, w0, w1: None,
            local_body=lambda payload, model: np.asarray(
                asnumpy(acs.likelihood(complex=False))
            ),
            merge=lambda results: concat_blocks(results, self.layout),
        )

    def gather_residual_hashes(self, acs):
        """``{world_rank: residual_hash}`` over the compute ranks (head-only, sampling phase).

        One-walker replica mode watch: every rank hashes its OWN residual
        buffers (``acs`` for the head, the ``RESIDUAL_HASH_OP`` builtin for
        each worker) so drift between replicas shows up as disagreeing
        hashes -- unlike :meth:`gather_likelihood` / :func:`concat_blocks`,
        this is per-rank, never merged down to the head's single block.
        """
        return self.run(
            RESIDUAL_HASH_OP,
            move=None,
            per_rank_payload=lambda rank, w0, w1: None,
            local_body=lambda payload, model: residual_hash(acs),
            merge=lambda results: {int(r): str(h) for r, h in results.items()},
        )

    def run(self, op, *, move=None, per_rank_payload, local_body, merge, shared=None):
        if not self.is_head:
            raise RuntimeError("WalkerFanout.run is head-only; compute ranks serve commands")
        key = (move, op)
        self._call_index[key] += 1
        clock = dict(self.clock, move=move, call_index=self._call_index[key], seq=self.seq)
        w0, w1 = self.layout.block_of(self.head)
        if self.single:
            self.clock.update(clock)
            return merge({self.head: local_body(per_rank_payload(self.head, w0, w1), self.model)})

        self.seq += 1
        clock["seq"] = self.seq
        self.clock.update(clock)
        requests = []
        sent_workers = []
        wait_s = 0.0
        try:
            for r in self.layout.worker_ranks:
                rw0, rw1 = self.layout.block_of(r)
                cmd = {
                    "seq": self.seq,
                    "op": op,
                    "move": move,
                    "clock": clock,
                    "payload": per_rank_payload(r, rw0, rw1),
                    "shared": shared,
                }
                req = self.comm.isend(cmd, dest=self.layout.fanout_rank(r))
                requests.append(req)
                sent_workers.append(r)
            # Wait IMMEDIATELY: under real MPI, large payloads run the rendezvous
            # protocol, which only makes progress when the sender calls wait/test.
            # Every worker is already parked in recv, so this cannot block on them;
            # without this, the transfer stalls until the head reaches this point
            # anyway, but AFTER local_body -- serializing head and worker compute.
            t_wait0 = time.perf_counter()
            for req in requests:
                req.wait()
            wait_s = time.perf_counter() - t_wait0
            self.last_wait_s = wait_s
            t0 = time.perf_counter()
            local = local_body(per_rank_payload(self.head, w0, w1), self.model)
            head_wall = time.perf_counter() - t0
        except BaseException:
            # Drain: a worker that was sent a command WILL reply. If we don't
            # receive it here, it sits in the channel and corrupts the next
            # run()'s bookkeeping (or blocks stop()'s send). A worker that never
            # received a command must NOT be recv'ed from (it never sent a reply).
            for req in requests:
                try:
                    req.wait()
                except Exception:  # noqa: BLE001 - best-effort drain, never masks the raise
                    pass
            for r in sent_workers:
                try:
                    self.comm.recv(source=self.layout.fanout_rank(r))
                except Exception:  # noqa: BLE001 - best-effort drain, never masks the raise
                    pass
            raise
        replies = {self.head: _reply(self.seq, self.head, True, local, head_wall)}
        for r in self.layout.worker_ranks:
            replies[r] = self.comm.recv(source=self.layout.fanout_rank(r))
        # seq check FIRST: a stale reply (left over from an earlier, aborted run())
        # must be reported as a sequence error, not misread as a fresh remote failure.
        bad_seq = [rep["rank"] for rep in replies.values() if rep["seq"] != self.seq]
        if bad_seq:
            raise RuntimeError(f"fan-out sequence mismatch from ranks {bad_seq} (op={op!r})")
        failures = [rep for rep in replies.values() if not rep["ok"]]
        if failures:
            if self.logger is not None:
                for rep in failures:
                    self.logger.error(
                        "rank %d failed op=%r move=%r:\n%s",
                        rep["rank"],
                        op,
                        move,
                        (rep.get("error") or {}).get("traceback", ""),
                    )
            first = failures[0]
            raise RemoteWorkerError(first["rank"], op, move, first.get("error"))
        # INFO on the module logger (not DEBUG on ``self.logger``): one line per
        # fan-out op, the load-balance signal the cluster gates diff.
        worst = max(rep["wall_s"] for rep in replies.values())
        _LOG.info(
            "[FANOUT] op=%s move=%s head_s=%.3f max_rank_s=%.3f wait_s=%.3f",
            op,
            move,
            head_wall,
            worst,
            wait_s,
        )
        return merge({r: rep["result"] for r, rep in replies.items()})

    def ping(self):
        """Every compute rank's layout digest, keyed by world rank; raises on mismatch."""
        digests = self.run(
            PING_OP,
            per_rank_payload=lambda r, w0, w1: None,
            local_body=lambda payload, model: self.layout.digest(),
            merge=lambda results: results,
        )
        if len(set(digests.values())) != 1:
            raise RuntimeError(f"rank layouts disagree: {digests}")
        return digests

    def stop(self):
        if self.single or not self.is_head or self._stopped:
            return
        self._stopped = True
        for r in self.layout.worker_ranks:
            self.comm.send(
                {
                    "seq": self.seq,
                    "op": STOP_OP,
                    "move": None,
                    "clock": dict(self.clock),
                    "payload": None,
                    "shared": None,
                },
                dest=self.layout.fanout_rank(r),
            )

    # -- setup-phase collective ------------------------------------------
    def allgather_walker_vector(self, local_1d):
        """Concatenate every compute rank's 1-D block vector in walker order (collective).

        Replica mode: every rank holds a copy of the same single walker, so
        the head's vector (not a concatenation of the replicas) is the
        answer, returned identically on every rank.
        """
        local = np.asarray(local_1d)
        if self.single:
            return local
        parts = self.comm.allgather(local)  # fan-out comm ranks == compute-rank order
        if getattr(self.layout, "replica_mode", False):
            return np.asarray(parts[self.layout.fanout_rank(self.head)])
        return np.concatenate([np.asarray(p) for p in parts])


class ComputeService:
    """Compute-rank command loop: ``recv`` is the clock, ``{"op": "stop"}`` exits."""

    def __init__(self, comm, layout, rank, *, registry, model=None, builtins=None, logger=None):
        self.comm = comm
        self.layout = layout
        self.rank = int(rank)
        self.registry = dict(registry)
        self.model = model
        self.logger = logger
        self.builtins = {PING_OP: lambda payload, clock, model: self.layout.digest()}
        self.builtins.update(builtins or {})
        self._head_local = layout.fanout_rank(layout.head_rank)

    def handle(self, cmd):
        t0 = time.perf_counter()
        seq = op = move_name = None
        try:
            seq = cmd.get("seq")
            op = cmd.get("op")
            move_name = cmd.get("move")
            clock = cmd.get("clock") or {}
            if op in self.builtins:
                result = self.builtins[op](cmd.get("payload"), clock, self.model)
            else:
                # Registry keys are ``(stage_name, move_name)`` (run.py
                # _serve_registry: names recur across stages) with a bare-name
                # fallback for stage-less registries (tests, hand-built).
                move = self.registry.get((clock.get("stage"), move_name))
                if move is None:
                    move = self.registry[move_name]
                if "stage_kind" in clock:
                    move.gf_stage_kind = clock["stage_kind"]
                result = move.gf_serve(op, cmd.get("payload"), clock, self.model)
            return _reply(seq, self.rank, True, result, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - reported to the head, never swallowed
            error = {
                "type": type(exc).__name__,
                "msg": str(exc),
                "traceback": traceback.format_exc(),
            }
            try:
                if self.logger is not None:
                    self.logger.exception(
                        "rank %d: command %r for move %r failed", self.rank, op, move_name
                    )
            except Exception:  # noqa: BLE001 - a failing logger must never mask the reply
                pass
            return _reply(seq, self.rank, False, None, time.perf_counter() - t0, error)

    def serve(self):
        served = 0
        while True:
            cmd = self.comm.recv(source=self._head_local)
            if cmd.get("op") == STOP_OP:
                return served
            reply = self.handle(cmd)
            try:
                self.comm.send(reply, dest=self._head_local)
            except Exception as exc:  # noqa: BLE001 - a bad reply must not escape the loop
                error = {
                    "type": "SendError",
                    "msg": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                degraded = _reply(reply["seq"], self.rank, False, None, reply["wall_s"], error)
                self.comm.send(degraded, dest=self._head_local)
            served += 1
