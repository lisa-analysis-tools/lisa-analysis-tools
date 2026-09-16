"""Whole-``propose`` fan-out over walker blocks (addremove family, PSD family).

Head: ``propose`` slices the FULL state into per-rank blocks, every compute rank
(the head included) runs the UNCHANGED body ``propose_local`` on its block
against its own ACA, the head merges the blocks back, SUM-reduces the per-ladder
swap tallies and adapts each ladder ONCE. With one compute rank ``propose`` is a
direct call of ``propose_local`` -- bit-identical to the pre-port code.

Semantics that change only when several compute ranks exist (design spec,
"Semantics that change only when n_compute > 1"):

* Eryn ladders adapt once per propose from the pooled swap ratio
  (sum accepted / sum proposed over ranks and repeats); ``tc.time`` advances once
  per propose instead of once per repeat. Ranks never adapt.
* The mid-iteration checkpoint granularity drops from per leaf to per propose.
* Eigen / info-matrix tables are rank-local proposal shapes; the sidecar is
  written by the head only.
* Stretch complements (PSD RedBlue splits) are the rank's local block.
* Cross-rank tempering swaps: none (within-rank); see the spec's WP8.
* Sub-state delta counters of the fanned-out branches equal the SUM over ranks
  of what the body wrote (the head's copy is zeroed first when the body assigns
  them, ``fanout_assigns_counters``).
"""

from __future__ import annotations

import numpy as np

from ..communication.walkerslice import merge_state, slice_state
from ..state import GFState

__all__ = ["PROPOSE_OP", "WalkerFanoutMixin", "pooled_ladder_step"]

PROPOSE_OP = "propose"


def pooled_ladder_step(tc, betas0, acc, prop):
    """One head-side ladder adaptation from pooled swap tallies; advances ``tc.time``.

    Returns the new ladder (``betas0`` when the control is not configured
    adaptive, has one rung, is past ``stop_adaptation``, or has no proposals).
    Mirrors ``TemperatureControl.adapt_temps`` (Eryn tempering.py ~867-897) with
    ``ratios = acc / prop`` supplied by the caller.
    """
    betas0 = np.array(betas0, dtype=float, copy=True)
    acc = np.asarray(acc, dtype=float).ravel()
    prop = np.asarray(prop, dtype=float).ravel()
    adaptive = bool(getattr(tc, "gf_configured_adaptive", getattr(tc, "adaptive", False)))
    stop = int(getattr(tc, "stop_adaptation", -1))
    running = stop < 0 or int(tc.time) < stop
    new = betas0
    if adaptive and int(tc.ntemps) > 1 and acc.size and np.all(prop > 0) and running:
        new = betas0 + tc._get_ladder_adjustment(int(tc.time), betas0.copy(), acc / prop)
    tc.time += 1
    return new


class WalkerFanoutMixin:
    """First in the MRO of a move whose whole ``propose`` runs per walker block.

    Subclass contract: rename the body to ``propose_local(model, state)``; the
    recipe stamps ``fanout_branches`` and calls ``install_walker_fanout(curr)``
    after construction; override the hooks below.
    """

    fanout = None  # the run's WalkerFanout (None single-process)
    gf_rank = None
    fanout_branches = None  # sub-states shipped and merged
    fanout_assigns_counters = False  # body ASSIGNS sub-state delta counters -> zero, then sum
    _fanout_body = False  # True while propose_local runs as a rank body

    # ---- subclass hooks --------------------------------------------------
    def fanout_temperature_controls(self):
        """Every ``TemperatureControl`` this move adapts (ranks never adapt them)."""
        raise NotImplementedError

    def fanout_payload_extra(self):
        """Head -> rank clock values (ladders, propose counters). Host numpy only."""
        return {}

    def fanout_apply_extra(self, extra):
        """Rank side: install the shipped clock values before the body runs."""

    def fanout_reply_extra(self, part):
        """Rank -> head: swap tallies and anything the body left rank-local."""
        return {}

    def fanout_merge_extra(self, replies, new_state):
        """Head: pooled ladder adaptation etc. ``replies`` = {rank: extra}."""

    # ---- wiring ------------------------------------------------------------
    @property
    def fanout_active(self):
        return self.fanout is not None and not self.fanout.single

    def install_walker_fanout(self, curr):
        """Bind the run's fan-out (``curr.fanout``) and apply the multi-rank rules."""
        self.fanout = getattr(curr, "fanout", None)
        self.gf_rank = getattr(curr, "rank", None)
        if not self.fanout_active:
            return
        for tc in self.fanout_temperature_controls():
            if tc is None:
                continue
            if not hasattr(tc, "gf_configured_adaptive"):  # shared controls: stamp once
                tc.gf_configured_adaptive = bool(getattr(tc, "adaptive", False))
            tc.adaptive = False  # ranks never adapt; the head adapts once per propose
        if not self.fanout.is_head and hasattr(self, "eigen_store_path"):
            self.eigen_store_path = None  # single-writer sidecar (head only)

    # ---- propose -------------------------------------------------------------
    def propose(self, model, state):
        if not self.fanout_active:
            return self.propose_local(model, state)
        return self.fanout_propose(model, state)

    def fanout_propose(self, model, state):
        fanout = self.fanout
        layout = fanout.layout
        branches = list(self.fanout_branches or [])
        extra = self.fanout_payload_extra()

        def payload(rank, w0, w1):
            return {"state": slice_state(state, w0, w1, sub_states=branches), "extra": extra}

        def body(p, model_local):
            return self.gf_serve(PROPOSE_OP, p, fanout.clock, model_local)

        def merge(replies):
            new_state = GFState(state, copy=True)
            if self.fanout_assigns_counters:
                for name in branches:
                    sub = (getattr(new_state, "sub_states", None) or {}).get(name)
                    if sub is None or not getattr(sub, "tempered_initialized", False):
                        continue
                    for cname in sub.delta_counter_names:
                        arr = getattr(sub, cname, None)
                        if arr is not None:
                            arr[...] = 0
            for rank in layout.compute_ranks:
                w0, w1 = layout.block_of(rank)
                merge_state(new_state, replies[rank]["state"], w0, w1)
            accepted = np.concatenate(
                [np.asarray(replies[r]["accepted"]) for r in layout.compute_ranks], axis=1
            )
            self.fanout_merge_extra({r: replies[r]["extra"] for r in layout.compute_ranks},
                                    new_state)
            return new_state, accepted

        return fanout.run(
            PROPOSE_OP,
            move=getattr(self, "gf_move_name", None),
            per_rank_payload=payload,
            local_body=body,
            merge=merge,
        )

    def gf_serve(self, op, payload, clock, model):
        if op != PROPOSE_OP:
            raise ValueError(f"{type(self).__name__} serves only {PROPOSE_OP!r}, got {op!r}")
        self.fanout_apply_extra(payload.get("extra") or {})
        self._fanout_body = True
        try:
            part, accepted = self.propose_local(model, payload["state"])
        finally:
            self._fanout_body = False
        # `propose_local` commonly re-wraps its input as ``GFState(state, copy=True)``;
        # that constructor instantiates a BARE (non-None, not tempered_initialized)
        # sub-state for every branch in ``sub_state_bases``, even ones ``slice_state``
        # deliberately left ``None`` (branches outside ``fanout_branches``). Restore
        # that invariant on the reply so the head's ``merge_state`` only ever touches
        # the branches this move actually fans out -- a bare sub-state reaching
        # ``merge_walkers`` on an already-initialized full branch raises there.
        fanned = set(self.fanout_branches or [])
        for name in list(getattr(part, "sub_states", None) or {}):
            if name not in fanned:
                part.sub_states[name] = None
        return {
            "state": part,
            "accepted": np.asarray(accepted),
            "extra": self.fanout_reply_extra(part),
        }
