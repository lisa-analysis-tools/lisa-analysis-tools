"""Scatter independent rows over every compute rank; gather in row order.

One-walker replica mode (``WalkerBlockLayout.replica_mode``): every compute
rank holds the SAME walker's residual, so any scoring row (a parameter vector
plus the walker it scores against) can be evaluated on any rank.
:meth:`RowFanout.run` splits a row batch into contiguous chunks in
compute-rank order (``np.array_split`` sizes), ships each chunk through
:meth:`WalkerFanout.run` (the head evaluates its own chunk locally) and
concatenates the replies back in row order. :meth:`RowFanout.replay`
broadcasts a state-mutating step (expose a source, publish a noise model) so
every replica applies it, head included. With one compute rank both are
direct local calls -- no comm, no pickling.

Rank side: ``ComputeService`` routes op ``x`` to ``move.gf_serve("x", ...)``;
the payload is ``{"rows": {...}}`` for ``run`` and the raw payload for
``replay``. A ``run`` body must return EVERY key for an empty chunk too
(zero-length arrays) so the concatenation stays well-defined.
"""

import numpy as np

__all__ = ["RowFanout"]


class RowFanout:
    def __init__(self, fanout, move):
        self.fanout = fanout
        self.move = move

    @property
    def active(self) -> bool:
        return self.fanout is not None and not self.fanout.single

    @property
    def move_name(self):
        return getattr(self.move, "gf_move_name", None)

    @staticmethod
    def _bounds(n, n_chunks):
        sizes = [len(c) for c in np.array_split(np.arange(int(n)), int(n_chunks))]
        return np.concatenate([[0], np.cumsum(sizes)]).astype(int)

    def run(self, op, rows, *, local_body):
        """``{name: (N, ...)}`` -> ``{name: (N, ...)}``, rows evaluated over all compute ranks."""
        rows = {k: np.asarray(v) for k, v in rows.items()}
        if not rows:
            raise ValueError("RowFanout.run: no row arrays given")
        lengths = {k: int(v.shape[0]) for k, v in rows.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"RowFanout.run: row arrays disagree on length: {lengths}")
        n = next(iter(lengths.values()))
        if not self.active:
            return local_body(rows)
        layout = self.fanout.layout
        bounds = self._bounds(n, layout.n_compute)

        def chunk(rank, _w0, _w1):
            i = layout.fanout_rank(rank)
            s, e = int(bounds[i]), int(bounds[i + 1])
            return {"rows": {k: v[s:e] for k, v in rows.items()}}

        def body(payload, _model):
            return local_body(payload["rows"])

        def merge(results):
            keys = list(results[layout.head_rank].keys())
            return {
                k: np.concatenate(
                    [np.asarray(results[r][k]) for r in layout.compute_ranks], axis=0
                )
                for k in keys
            }

        return self.fanout.run(
            op, move=self.move_name, per_rank_payload=chunk, local_body=body, merge=merge
        )

    def replay(self, op, payload, *, local_body):
        """Apply ``payload`` on every compute rank (the head through ``local_body``)."""
        if not self.active:
            local_body(payload)
            return
        self.fanout.run(
            op,
            move=self.move_name,
            per_rank_payload=lambda _rank, _w0, _w1: payload,
            local_body=lambda p, _model: local_body(p),
            merge=lambda _results: None,
        )
