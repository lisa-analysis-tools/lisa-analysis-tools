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
import time
import traceback

import numpy as np

STOP_OP = "stop"
PING_OP = "ping"


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
    """Concatenate per-rank 1-D results in compute-rank (== walker) order."""
    return np.concatenate([np.asarray(results[r]) for r in layout.compute_ranks])


def _reply(seq, rank, ok, result, wall_s, error=None):
    return {"seq": seq, "rank": rank, "ok": ok, "result": result, "wall_s": wall_s, "error": error}


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
        self.seq = 0
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
            requests.append(self.comm.isend(cmd, dest=self.layout.fanout_rank(r)))
        t0 = time.perf_counter()
        local = local_body(per_rank_payload(self.head, w0, w1), self.model)
        head_wall = time.perf_counter() - t0
        for req in requests:
            req.wait()
        replies = {self.head: _reply(self.seq, self.head, True, local, head_wall)}
        for r in self.layout.worker_ranks:
            replies[r] = self.comm.recv(source=self.layout.fanout_rank(r))
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
        bad_seq = [rep["rank"] for rep in replies.values() if rep["seq"] != self.seq]
        if bad_seq:
            raise RuntimeError(f"fan-out sequence mismatch from ranks {bad_seq} (op={op!r})")
        if self.logger is not None:
            worst = max(rep["wall_s"] for rep in replies.values())
            self.logger.debug(
                "[FANOUT] op=%s move=%s head_s=%.3f max_rank_s=%.3f", op, move, head_wall, worst
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
        if self.single or not self.is_head:
            return
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
        """Concatenate every compute rank's 1-D block vector in walker order (collective)."""
        local = np.asarray(local_1d)
        if self.single:
            return local
        parts = self.comm.allgather(local)  # fan-out comm ranks == compute-rank order
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
        seq = cmd.get("seq")
        op = cmd["op"]
        move_name = cmd.get("move")
        clock = cmd.get("clock") or {}
        t0 = time.perf_counter()
        try:
            if op in self.builtins:
                result = self.builtins[op](cmd.get("payload"), clock, self.model)
            else:
                move = self.registry[move_name]
                if "stage_kind" in clock:
                    move.gf_stage_kind = clock["stage_kind"]
                result = move.gf_serve(op, cmd.get("payload"), clock, self.model)
            return _reply(seq, self.rank, True, result, time.perf_counter() - t0)
        except Exception as exc:  # noqa: BLE001 - reported to the head, never swallowed
            if self.logger is not None:
                self.logger.exception(
                    "rank %d: command %r for move %r failed", self.rank, op, move_name
                )
            error = {
                "type": type(exc).__name__,
                "msg": str(exc),
                "traceback": traceback.format_exc(),
            }
            return _reply(seq, self.rank, False, None, time.perf_counter() - t0, error)

    def serve(self):
        served = 0
        while True:
            cmd = self.comm.recv(source=self._head_local)
            if cmd.get("op") == STOP_OP:
                return served
            reply = self.handle(cmd)
            self.comm.send(reply, dest=self._head_local)
            served += 1
