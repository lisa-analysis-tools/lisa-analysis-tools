"""In-process stand-in for the mpi4py communicator subset the global fit uses.

``FakeWorld(size)`` runs one Python thread per rank; ``FakeComm`` implements
``Get_rank / Get_size / Get_processor_name / send / isend / recv / iprobe /
bcast / Bcast / allgather / barrier / Split / Split_type / Abort / Free``
with pickle-copy transport, so the SAME fan-out code runs on a laptop with
no MPI installed, in one python process. A test harness, not a performance
tool.

What the fake does NOT model:

- ``tag`` is ignored: there is one FIFO queue per ``(source, dest)`` pair,
  not per ``(source, dest, tag)``.
- ``Split_type`` ignores its ``split_type`` argument and always groups by
  node (the only grouping this codebase needs).
- ``send``/``isend`` are a blocking pickle-copy into an unbounded queue:
  no rendezvous handshake, no backpressure. Ordering bugs that only appear
  under real MPI's rendezvous protocol for large messages (see fanout.py's
  ``WalkerFanout.run`` isend/wait ordering, finding F1) cannot be
  reproduced here.
"""

from __future__ import annotations

import pickle
import queue
import threading
import time
import traceback

import numpy as np

#: mirrors of the mpi4py constants the layout code needs (only identity matters)
COMM_TYPE_SHARED = 1
UNDEFINED = -32766


class FakeAbort(RuntimeError):
    """Raised in every rank's thread once any rank aborts the world."""


def _pcopy(obj):
    return pickle.loads(pickle.dumps(obj))


class _Request:
    def wait(self):
        return None

    def test(self):
        return True, None


class _Group:
    """Shared plumbing of one communicator: inboxes + collective slots."""

    def __init__(self, members):
        self.members = tuple(members)  # world ranks, in local-rank order
        n = len(self.members)
        self.inbox = {(d, s): queue.Queue() for d in range(n) for s in range(n)}
        self.barrier = threading.Barrier(n) if n > 0 else None
        self.slots = [None] * n


class FakeCommNull:
    """What ``Split`` returns to a rank with colour ``UNDEFINED``."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise RuntimeError("FakeCommNull: this rank is not a member of the communicator")


class FakeComm:
    COMM_TYPE_SHARED = COMM_TYPE_SHARED
    UNDEFINED = UNDEFINED

    def __init__(self, world, group, world_rank):
        self._world = world
        self._group = group
        self._world_rank = int(world_rank)
        self._rank = group.members.index(self._world_rank)

    # -- topology ---------------------------------------------------------
    def Get_rank(self):
        return self._rank

    def Get_size(self):
        return len(self._group.members)

    def Get_processor_name(self):
        return self._world.node_name(self._world_rank)

    def Free(self):
        return None

    # -- point to point ---------------------------------------------------
    def send(self, obj, dest, tag=0):
        self._check_abort()
        self._group.inbox[(int(dest), self._rank)].put(_pcopy(obj))

    def isend(self, obj, dest, tag=0):
        self.send(obj, dest, tag)
        return _Request()

    def recv(self, source, tag=0):
        q = self._group.inbox[(self._rank, int(source))]
        while True:
            self._check_abort()
            try:
                return q.get(timeout=0.02)
            except queue.Empty:
                continue

    def iprobe(self, source, tag=0):
        return not self._group.inbox[(self._rank, int(source))].empty()

    # -- collectives (every member calls them in the same order) ----------
    def _wait_barrier(self):
        self._check_abort()
        try:
            self._group.barrier.wait(timeout=self._world.timeout)
        except threading.BrokenBarrierError:
            raise FakeAbort(
                f"rank {self._world_rank}: collective broken (abort or timeout)"
            ) from None

    def _exchange(self, value):
        group = self._group
        group.slots[self._rank] = value
        self._wait_barrier()
        out = list(group.slots)
        self._wait_barrier()  # nobody overwrites the slots before everyone has read
        return out

    def bcast(self, obj, root=0):
        out = self._exchange(_pcopy(obj) if self._rank == int(root) else None)
        return _pcopy(out[int(root)])

    def Bcast(self, buf, root=0):
        """Buffer broadcast (uppercase MPI form): fills ``buf`` IN PLACE.

        mpi4py's ``Bcast`` moves contiguous buffers without pickling, which
        is what the F-stat reference row pair (tens of MB) uses. The fake
        moves the bytes through the existing slot exchange and writes them
        into each non-root rank's own array, so a caller that allocates its
        own receive buffer -- the real usage -- takes the same shape here.

        The receive buffer must be C-CONTIGUOUS: ``reshape(-1)`` returns a
        view only then, and on a strided array it would copy, so the write
        would land in a temporary and the broadcast would vanish with no
        error at all. Refused loudly instead. (Every array the F-stat op
        broadcasts is a fresh ``np.empty``, so this never fires in
        production -- it exists so a future caller cannot fail silently.)

        What this does NOT model, on top of the module-level list:

        - the ROOT's buffer is never validated. Real mpi4py raises
          ``BufferError`` for a non-contiguous send buffer; here the root
          goes through ``tobytes()``, which happily linearizes a strided
          array. So a strided buffer is rejected by BOTH implementations but
          for different reasons and on different ranks -- receiver-side here,
          sender-side there.
        - the ``[buf, count, datatype]`` list form is accepted, but only
          ``buf`` is honoured: an explicit ``count`` or MPI datatype is
          dropped, and the whole array is sent at its own numpy dtype.
        """
        arr = buf[0] if isinstance(buf, (list, tuple)) else buf
        arr = np.asarray(arr)
        payload = self._exchange(
            arr.tobytes() if self._rank == int(root) else None)[int(root)]
        if self._rank != int(root):
            if not arr.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    "Bcast needs a C-contiguous receive buffer; got shape "
                    f"{arr.shape} strides {arr.strides}, whose reshape(-1) "
                    "is a COPY -- the broadcast would be silently lost")
            flat = arr.reshape(-1)
            got = np.frombuffer(payload, dtype=flat.dtype)
            if got.size != flat.size:
                raise ValueError(
                    f"Bcast buffer size mismatch: root sent {got.size} "
                    f"elements, this rank's buffer holds {flat.size}")
            flat[:] = got
        return None

    def allgather(self, obj):
        return [_pcopy(v) for v in self._exchange(_pcopy(obj))]

    def barrier(self):
        self._wait_barrier()

    Barrier = barrier

    def Split(self, color, key=0):
        colors = self._exchange((color, key))
        gid = self._exchange(self._world.next_group_id() if self._rank == 0 else None)[0]
        if color == UNDEFINED:
            return FakeCommNull()
        order = sorted(
            (i for i, (c, _k) in enumerate(colors) if c == color),
            key=lambda i: (colors[i][1], self._group.members[i]),
        )
        members = tuple(self._group.members[i] for i in order)
        return FakeComm(self._world, self._world.group(gid, members), self._world_rank)

    def Split_type(self, split_type, key=0):
        return self.Split(color=self._world.node_of(self._world_rank), key=key)

    # -- failure ----------------------------------------------------------
    def Abort(self, errorcode=1):
        self._world.abort(errorcode, self._world_rank)
        raise FakeAbort(f"rank {self._world_rank} called Abort({errorcode})")

    def _check_abort(self):
        if self._world.aborted is not None:
            code, who = self._world.aborted
            raise FakeAbort(f"rank {self._world_rank}: world aborted by rank {who} (code {code})")


class FakeWorld:
    """``size`` ranks on ``nodes`` (e.g. ``[0, 1, 0, 1, 0]``), one thread each."""

    def __init__(self, size, nodes=None, timeout=60.0):
        self.size = int(size)
        self.nodes = list(nodes) if nodes is not None else [0] * self.size
        if len(self.nodes) != self.size:
            raise ValueError("nodes must have one entry per rank")
        self.timeout = float(timeout)
        self.aborted = None
        self._groups = {}
        self._lock = threading.Lock()
        self._gid = 0
        self._world_group = self.group(0, tuple(range(self.size)))

    def node_of(self, rank):
        return int(self.nodes[int(rank)])

    def node_name(self, rank):
        return f"node{self.node_of(rank)}"

    def next_group_id(self):
        with self._lock:
            self._gid += 1
            return self._gid

    def group(self, gid, members):
        with self._lock:
            key = (int(gid), tuple(members))
            if key not in self._groups:
                self._groups[key] = _Group(members)
            return self._groups[key]

    def comm(self, rank):
        return FakeComm(self, self._world_group, int(rank))

    def abort(self, code, rank):
        self.aborted = (int(code), int(rank))
        with self._lock:
            for g in self._groups.values():
                if g.barrier is not None:
                    g.barrier.abort()

    def run(self, fn, ranks=None, timeout=None):
        """Run ``fn(rank, comm)`` on one thread per rank; return ``{rank: result}``.

        Re-raises the first NON-abort failure (the root cause) as a
        ``RuntimeError`` carrying the remote traceback text, after aborting
        the world so no other thread stays blocked.
        """
        ranks = list(range(self.size)) if ranks is None else list(ranks)
        results, errors = {}, {}

        def _target(r):
            try:
                results[r] = fn(r, self.comm(r))
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors[r] = (exc, traceback.format_exc())
                if self.aborted is None:
                    self.abort(1, r)

        threads = [
            threading.Thread(target=_target, args=(r,), name=f"fake-rank-{r}", daemon=True)
            for r in ranks
        ]
        for t in threads:
            t.start()
        deadline = time.time() + (self.timeout if timeout is None else float(timeout))
        for t in threads:
            t.join(max(0.0, deadline - time.time()))
        alive = [t.name for t in threads if t.is_alive()]
        if alive and not errors:
            self.abort(1, -1)
            raise TimeoutError(f"FakeWorld.run: ranks still running at timeout: {alive}")
        if errors:
            root = [r for r, (e, _) in errors.items() if not isinstance(e, FakeAbort)]
            r = min(root) if root else min(errors)
            exc, tb = errors[r]
            raise RuntimeError(f"rank {r} failed:\n{tb}") from exc
        return results
