"""Cross-device stream-ordering edges on the DIRECT peer copies.

2026-09-16 shard-1 RJ accounting defect. ``beaa60f5`` closed the ordering
gap *inside* the GB shard router (``_dispatch_shards``) and ``e3dde390``
its ACA twin (``_run_per_split``). Both record an event on each shard's
stream and make the CALLER's current stream wait before returning. That
leaves three holes, and the GB reversible-jump path walks straight
through all of them:

1. **Direction.** The edge is shard -> caller only. Routed call sites
   stage their per-shard inputs on the CALLER's device (the default
   ``GB_ROUTER_DEVICE_RESIDENT=1`` path keeps params device-resident and
   pre-slices them with ``_slice_rows``), and each worker then does
   ``xp.asarray(part)`` inside its own device context -- a peer READ of
   caller-device memory on the shard's stream, ordered against nothing.

2. **Which stream waits.** ``wait_event`` is applied to the caller's
   current stream on the caller's CURRENT device. Any work the caller
   issues afterwards inside ``with Device(other)`` -- i.e. every
   ``BandView`` per-shard loop and ``SubBandBuffer.likelihood``'s
   per-shard reduce -- is on a different stream and waits on nothing.

3. **Copies that never reach a dispatcher at all.** The buffer fill
   (``fill_buffer_residual_and_psd_from_acs``) reads the WALKER-sharded
   parent through ``BandView.__getitem__`` and writes the BAND-sharded
   buffer through ``BandView.accumulate``. Both desugar to raw
   ``xp.asarray`` peer copies issued from the caller thread across two
   different devices, with no event anywhere.

Hole 3 is device-topology-only (a device-0 stream reading device-1
memory), so it bites regardless of whether the build uses the legacy
default stream or per-thread default streams.

These tests pin the CONTRACT on CPU with the stream-capable fakes; real
multi-GPU semantics are validated on the cluster.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

try:
    from tests._multishard import (
        DeviceArray, FakeMultiShardACA, StreamRecordingXp, stamp_device,
    )
except ImportError:  # pragma: no cover - direct invocation from tests/
    from _multishard import (
        DeviceArray, FakeMultiShardACA, StreamRecordingXp, stamp_device,
    )

from lisatools.analysiscontainer import BandView
from lisatools.utils.device import order_after_array


class OrderAfterArrayTest(unittest.TestCase):
    """The primitive: make the CURRENT device's stream wait on ``arr``'s."""

    def test_records_on_source_and_waits_on_current(self):
        xp = StreamRecordingXp()
        arr = stamp_device(np.zeros(4), 1)
        with xp.cuda.Device(0):
            order_after_array(xp, arr)
        records = [e for e in xp.stream_log if e[0] == "record"]
        waits = [e for e in xp.stream_log if e[0] == "wait"]
        self.assertEqual(len(records), 1, xp.stream_log)
        self.assertEqual(records[0][1], 1)          # recorded on dev 1
        self.assertEqual(len(waits), 1, xp.stream_log)
        self.assertEqual(waits[0][1], 0)            # waited on dev 0
        self.assertEqual(waits[0][2], 1)            # for dev 1's event
        # the wait must follow the record
        self.assertGreater(waits[0][4], records[0][2])

    def test_same_device_is_a_noop(self):
        xp = StreamRecordingXp()
        arr = stamp_device(np.zeros(4), 0)
        with xp.cuda.Device(0):
            order_after_array(xp, arr)
        self.assertEqual(xp.stream_log, [])

    def test_host_array_is_a_noop(self):
        xp = StreamRecordingXp()
        with xp.cuda.Device(0):
            order_after_array(xp, np.zeros(4))
        self.assertEqual(xp.stream_log, [])

    def test_plain_numpy_xp_is_a_noop(self):
        order_after_array(np, np.zeros(4))  # must not raise

    def test_streamless_xp_is_a_noop(self):
        """A cupy-like xp without get_current_stream (older fakes)."""
        try:
            from tests._multishard import RecordingXp
        except ImportError:  # pragma: no cover
            from _multishard import RecordingXp
        xp = RecordingXp()
        arr = stamp_device(np.zeros(4), 1)
        with xp.cuda.Device(0):
            order_after_array(xp, arr)  # must not raise


def _device_stamped_aca(num_acs=6, num_shards=2):
    """Multi-shard fake whose shard buffers carry a real ``.device``."""
    aca = FakeMultiShardACA((2, 4), num_acs=num_acs, num_shards=num_shards,
                            layout="blocked", dtype=float)
    aca.xp = StreamRecordingXp()
    aca.linear_data_arr = [
        stamp_device(buf, s).view(DeviceArray)
        for s, buf in enumerate(aca.linear_data_arr)
    ]
    for s, buf in enumerate(aca.linear_data_arr):
        buf._device_id = s
    return aca


class BandViewCrossDeviceEdgeTest(unittest.TestCase):
    """Every direct peer copy in BandView must carry an ordering edge."""

    def test_row_read_orders_staging_copy_behind_each_shard(self):
        """``view[rows]`` gathers on gpus[0] from every shard's device."""
        aca = _device_stamped_aca()
        view = BandView(aca, kind="data")
        with mock.patch(
            "lisatools.analysiscontainer.order_after_array"
        ) as spy:
            view[np.array([0, 1, 4, 5])]
        # rows 4,5 live on shard 1 -> its gathered slice must be ordered
        # before the gpus[0] staging copy reads it.
        self.assertTrue(spy.called, "no ordering edge on the read path")

    def test_accumulate_orders_shard_add_behind_the_payload_slice(self):
        """``accumulate`` slices on the payload's device, adds on the shard's."""
        aca = _device_stamped_aca()
        view = BandView(aca, kind="data")
        payload = stamp_device(np.ones((4,) + aca.per_band_shape), 0)
        with mock.patch(
            "lisatools.analysiscontainer.order_after_array"
        ) as spy:
            view.accumulate(np.array([0, 1, 4, 5]), payload)
        self.assertTrue(spy.called, "no ordering edge on the accumulate path")

    def test_scatter_orders_shard_write_behind_the_payload(self):
        aca = _device_stamped_aca()
        view = BandView(aca, kind="data")
        payload = stamp_device(np.ones((4,) + aca.per_band_shape), 0)
        with mock.patch(
            "lisatools.analysiscontainer.order_after_array"
        ) as spy:
            view[np.array([0, 1, 4, 5])] = payload
        self.assertTrue(spy.called, "no ordering edge on the scatter path")

    def test_cpu_aca_adds_no_edges(self):
        """``gpus is None`` must stay a pure-NumPy no-op."""
        aca = FakeMultiShardACA((2, 4), num_acs=6, num_shards=1,
                                layout="blocked", dtype=float)
        aca.gpus = None
        view = BandView(aca, kind="data")
        with mock.patch(
            "lisatools.analysiscontainer.order_after_array"
        ) as spy:
            view[np.array([0, 1])]
        spy.assert_not_called()


class DispatchRemainingEdgeTest(unittest.TestCase):
    """The two halves ``beaa60f5`` left open in ``_dispatch_shards``."""

    def _run(self, devices, caller_dev=0):
        from concurrent.futures import ThreadPoolExecutor
        from types import SimpleNamespace

        from lisatools.globalfit.moves.gbbands import _RoutedBandEngine

        xp = StreamRecordingXp()
        holder = SimpleNamespace(
            xp=xp, thread_pool=ThreadPoolExecutor(max_workers=2))
        items = [(si, SimpleNamespace(device=d), "payload")
                 for si, d in enumerate(devices)]
        with xp.cuda.Device(caller_dev):
            _RoutedBandEngine._dispatch_shards(
                holder, items, lambda *a: None,
                state_ids=list(range(len(devices))))
        return xp

    def test_inbound_edge_orders_each_shard_behind_caller_staging(self):
        """Workers read params staged on the CALLER's stream, so every
        shard stream must wait on a caller event BEFORE its launches."""
        xp = self._run([0, 1])
        waits = [e for e in xp.stream_log if e[0] == "wait"]
        records = [e for e in xp.stream_log if e[0] == "record"]
        # a staging event recorded on the caller's device before any work
        stage = [r for r in records if r[1] == 0]
        self.assertTrue(stage, xp.stream_log)
        first_stage_seq = min(r[2] for r in stage)
        # shard 1's stream waits on that caller event
        inbound = [w for w in waits if w[1] == 1 and w[2] == 0
                   and w[3] == first_stage_seq]
        self.assertTrue(inbound, f"no inbound edge: {xp.stream_log}")

    def test_outbound_edge_reaches_every_shard_device(self):
        """Later caller work is issued inside ``with Device(other)``, so
        each shard device's stream -- not only the caller's -- must be
        ordered behind all shard work when dispatch returns."""
        xp = self._run([0, 1])
        waits = [e for e in xp.stream_log if e[0] == "wait"]
        waiting_devices = {w[1] for w in waits}
        self.assertIn(0, waiting_devices, xp.stream_log)
        self.assertIn(1, waiting_devices, xp.stream_log)

    def test_single_device_dispatch_adds_no_cross_edges(self):
        """One populated shard on the caller's own device: nothing to order."""
        xp = self._run([0])
        cross = [e for e in xp.stream_log
                 if e[0] == "wait" and e[1] != e[2]]
        self.assertEqual(cross, [], xp.stream_log)


if __name__ == "__main__":
    unittest.main()
