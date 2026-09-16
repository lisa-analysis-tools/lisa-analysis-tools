"""Cross-device stream-ordering edges in ``_RoutedBandEngine._dispatch_shards``.

2026-09-15 shard-1 accounting defect: routed shard work is enqueued on each
shard device's CUDA stream, and ``_dispatch_shards`` returned when the host
futures completed -- i.e. when the LAUNCHES returned, not the kernels. The
caller (device 0) then read shard-1 output (assembly scatters, parent
residual rows at unit close, the ortho/drift checks) with no happens-before
edge. Production signature: ``|direct - credited|`` exactly 0.0 for the
caller-device walker block and 1e2-1e3 lnL for the other shard's walkers
(vgb_pe / rj_warm_search, 6mo run), plus the row-13 GB birth asymmetry
(334 vs 240 leaves/walker across the device boundary).

The fix records an event on each shard's stream right after its worker's
launches and makes the caller's stream wait on all of them before
``_dispatch_shards`` returns. These tests pin that CONTRACT on CPU with a
stream-capable ``RecordingXp`` -- real multi-GPU semantics are validated on
the cluster (GB_MULTIGPU_SYNC_DEBUG=1 discriminator).
"""

from __future__ import annotations

import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest import mock

import numpy as np

try:
    from tests._multishard import RecordingXp
except ImportError:  # pragma: no cover - direct invocation from tests/
    from _multishard import RecordingXp

from lisatools.globalfit.moves.gbbands import _RoutedBandEngine


class _FakeEvent:
    def __init__(self, device, seq):
        self.device = device
        self.seq = seq


class StreamRecordingXp(RecordingXp):
    """RecordingXp + a cupy-like per-device current stream.

    ``cuda.get_current_stream()`` returns a stream bound to the CURRENT
    device (thread-local, like cupy); ``stream.record()`` appends
    ("record", device, seq) and returns the event; ``stream.wait_event(ev)``
    appends ("wait", waiting_device, ev.device, ev.seq, seq). A shared
    monotone ``seq`` orders records, waits and worker marks across threads.
    """

    def __init__(self):
        super().__init__()
        self.stream_log = []
        self._seq_lock = threading.Lock()
        self._seq = 0

        outer = self

        class _Stream:
            def __init__(self, device):
                self.device = device

            def record(self, event=None):
                seq = outer.next_seq()
                ev = _FakeEvent(self.device, seq)
                outer.stream_log.append(("record", self.device, seq))
                return ev

            def wait_event(self, ev):
                seq = outer.next_seq()
                outer.stream_log.append(
                    ("wait", self.device, ev.device, ev.seq, seq))

        self.cuda.get_current_stream = (
            lambda: _Stream(self.cuda.runtime.getDevice()))

    def next_seq(self):
        with self._seq_lock:
            self._seq += 1
            return self._seq


def _make_holder(xp, threaded=True):
    return SimpleNamespace(
        xp=xp,
        thread_pool=ThreadPoolExecutor(max_workers=2) if threaded else None,
    )


def _make_items(xp, devices, work_log, seq_source):
    """Items shaped like every production call site: an int, then the shard
    view (the first element carrying ``.device``), then payload."""
    items = [
        (si, SimpleNamespace(device=dev), "payload")
        for si, dev in enumerate(devices)
    ]

    def worker(si, view, payload):
        seq = seq_source()
        work_log.append(("work", si, view.device, seq))

    return items, worker


class DispatchStreamEdgeTest(unittest.TestCase):
    def test_event_per_shard_and_caller_waits_after_all_work(self):
        xp = StreamRecordingXp()
        holder = _make_holder(xp)
        work_log = []
        items, worker = _make_items(xp, [0, 1], work_log, xp.next_seq)
        _RoutedBandEngine._dispatch_shards(
            holder, items, worker, state_ids=[10, 11])

        records = [e for e in xp.stream_log if e[0] == "record"]
        waits = [e for e in xp.stream_log if e[0] == "wait"]
        self.assertEqual(len(records), 2, xp.stream_log)
        self.assertEqual(sorted(r[1] for r in records), [0, 1])
        # one wait per recorded event, issued on the CALLER's stream (dev 0)
        self.assertEqual(len(waits), 2, xp.stream_log)
        self.assertTrue(all(w[1] == 0 for w in waits), waits)
        self.assertEqual(sorted(w[2] for w in waits), [0, 1])
        # ordering: every record after its shard's work; every wait after
        # every record and every work mark
        work_seq = {w[1]: w[3] for w in work_log}
        for rec in records:
            si = [it[0] for it in items
                  if it[1].device == rec[1]][0]
            self.assertGreater(rec[2], work_seq[si])
        last_pre_wait = max([r[2] for r in records]
                            + [w[3] for w in work_log])
        for w in waits:
            self.assertGreater(w[4], last_pre_wait)

    def test_serial_path_gets_the_same_edges(self):
        xp = StreamRecordingXp()
        holder = _make_holder(xp, threaded=False)
        work_log = []
        items, worker = _make_items(xp, [0, 1], work_log, xp.next_seq)
        with mock.patch.dict(os.environ, {"GB_ROUTER_THREADED": "0"}):
            _RoutedBandEngine._dispatch_shards(holder, items, worker)
        self.assertEqual(
            len([e for e in xp.stream_log if e[0] == "record"]), 2)
        self.assertEqual(
            len([e for e in xp.stream_log if e[0] == "wait"]), 2)

    def test_none_device_item_records_no_event(self):
        xp = StreamRecordingXp()
        holder = _make_holder(xp)
        work_log = []
        items, worker = _make_items(xp, [None, 1], work_log, xp.next_seq)
        _RoutedBandEngine._dispatch_shards(
            holder, items, worker, state_ids=[0, 1])
        records = [e for e in xp.stream_log if e[0] == "record"]
        waits = [e for e in xp.stream_log if e[0] == "wait"]
        self.assertEqual([r[1] for r in records], [1])
        self.assertEqual(len(waits), 1)
        self.assertEqual(len(work_log), 2)  # both workers still ran

    def test_xp_without_streams_is_a_noop(self):
        """RecordingXp has ``cuda`` but no ``get_current_stream`` (like the
        pre-fix fakes): dispatch must run workers and add no edges."""
        xp = RecordingXp()
        holder = _make_holder(xp)
        work_log = []
        items, worker = _make_items(
            xp, [0, 1], work_log, iter(range(100)).__next__)
        _RoutedBandEngine._dispatch_shards(
            holder, items, worker, state_ids=[0, 1])
        self.assertEqual(len(work_log), 2)

    def test_plain_numpy_holder_is_a_noop(self):
        holder = SimpleNamespace(xp=np, thread_pool=None)
        ran = []
        items = [(0, SimpleNamespace(device=None), "p")]
        _RoutedBandEngine._dispatch_shards(
            holder, items, lambda *a: ran.append(a))
        self.assertEqual(len(ran), 1)


if __name__ == "__main__":
    unittest.main()
