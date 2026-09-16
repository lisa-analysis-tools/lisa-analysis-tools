"""Cross-device stream-ordering edges in ``AnalysisContainerArray._run_per_split``.

The ACA twin of the GB shard-router defect fixed in ``beaa60f5``
(``gbbands.py::_RoutedBandEngine._dispatch_shards``). ``_run_per_split``
dispatches one worker per walker shard, each entering its own device
context; the futures complete when each worker's HOST code returns, with
that shard's CUDA kernels still queued on its own device's stream. The
caller then reads shard output from ITS OWN device's stream.

For :meth:`AnalysisContainerArray._vectorized_dispatch` the per-row
``asnumpy`` self-synchronises, but :meth:`signal_operation` (and
``apply_signal_from_params`` behind it) leaves the result purely
device-side: an in-place ``add_signal`` into the shard's slice of
``linear_data_arr``. Nothing ordered the caller behind the other shard's
residual writes, which is the same silent cross-shard accounting defect
the router carried.

These are CPU structural contracts on the stream-capable ``RecordingXp``
fake: records/waits are logged, never executed.
"""

import unittest

import numpy as np

from lisatools.analysiscontainer import AnalysisContainerArray

try:
    from tests._multishard import FakeMultiShardACA, StreamRecordingXp
except ImportError:  # running from inside tests/
    from _multishard import FakeMultiShardACA, StreamRecordingXp


def _aca(num_shards=2, num_acs=4, threaded=True, streams=True):
    aca = FakeMultiShardACA(
        (2, 8), num_acs, num_shards, layout="blocked", run_threaded=threaded
    )
    if streams:
        aca.xp = StreamRecordingXp()
    return aca


def _worker_factory(aca, work_log):
    """Worker that marks the device it ran on, in the shared sequence."""

    def worker(split, rows):
        with aca.xp.cuda.Device(int(aca.gpus[split])):
            work_log.append(
                ("work", int(aca.xp.cuda.runtime.getDevice()), aca.xp.next_seq())
            )

    return worker


class SplitStreamEdgeTest(unittest.TestCase):
    """The fix: one event per shard, caller waits on all before returning."""

    def test_threaded_two_shards_record_and_caller_waits_after_all_work(self):
        aca = _aca(threaded=True)
        work_log = []
        rows = aca._split_rows(np.arange(aca.acs_total_entries))
        AnalysisContainerArray._run_per_split(
            aca, _worker_factory(aca, work_log), rows
        )

        log = aca.xp.stream_log
        records = [e for e in log if e[0] == "record"]
        waits = [e for e in log if e[0] == "wait"]

        # one event recorded per populated shard, on that shard's device
        self.assertEqual(len(records), 2, log)
        self.assertEqual(sorted(r[1] for r in records), [0, 1])
        # one wait per recorded event, all issued on the CALLER's device (0)
        self.assertEqual(len(waits), 2, log)
        self.assertTrue(all(w[1] == 0 for w in waits), waits)
        self.assertEqual(sorted(w[2] for w in waits), [0, 1])

        # ordering: each shard's record follows its own work mark, and every
        # wait follows every record and every work mark.
        self.assertEqual(len(work_log), 2, work_log)
        for rec in records:
            own_work = [w for w in work_log if w[1] == rec[1]]
            self.assertEqual(len(own_work), 1, work_log)
            self.assertLess(own_work[0][2], rec[2])
        last_pre_wait = max([r[2] for r in records] + [w[2] for w in work_log])
        for wait in waits:
            self.assertGreater(wait[4], last_pre_wait, log)

    def test_serial_path_gets_the_same_edges(self):
        """run_threaded=False still crosses devices — the edge is not optional."""
        aca = _aca(threaded=False)
        work_log = []
        rows = aca._split_rows(np.arange(aca.acs_total_entries))
        AnalysisContainerArray._run_per_split(
            aca, _worker_factory(aca, work_log), rows
        )

        log = aca.xp.stream_log
        self.assertEqual(len([e for e in log if e[0] == "record"]), 2, log)
        self.assertEqual(len([e for e in log if e[0] == "wait"]), 2, log)

    def test_single_populated_split_takes_no_edge(self):
        """One shard == the caller's own stream: no event, no added overhead."""
        aca = _aca(threaded=True, num_acs=4)
        work_log = []
        # rows 0,1 are shard 0 under the blocked layout -> one populated split
        rows = aca._split_rows(np.array([0, 1]))
        self.assertEqual(len(rows), 1, rows)
        AnalysisContainerArray._run_per_split(
            aca, _worker_factory(aca, work_log), rows
        )

        self.assertEqual(len(work_log), 1, work_log)
        self.assertEqual(aca.xp.stream_log, [])

    def test_cpu_path_is_a_noop(self):
        """gpus=None (CPU): worker still runs, no stream traffic."""
        aca = _aca(threaded=False)
        aca.gpus = None
        work_log = []
        ran = []

        def worker(split, rows):
            ran.append(int(split))

        AnalysisContainerArray._run_per_split(
            aca, worker, {0: np.arange(2), 1: np.arange(2, 4)}
        )
        self.assertEqual(sorted(ran), [0, 1])
        self.assertEqual(aca.xp.stream_log, [])

    def test_xp_without_streams_is_a_noop(self):
        """A stream-less ``xp`` (plain RecordingXp) must not raise."""
        aca = _aca(threaded=True, streams=False)
        ran = []

        def worker(split, rows):
            ran.append(int(split))

        rows = aca._split_rows(np.arange(aca.acs_total_entries))
        AnalysisContainerArray._run_per_split(aca, worker, rows)
        self.assertEqual(sorted(ran), [0, 1])


class FakeDelegatesToRealRunnerTest(unittest.TestCase):
    """The shared fake must not carry its own copy of the runner.

    A duplicated ``_run_per_split`` in ``tests/_multishard.py`` would let
    every multi-shard consumer test pass against a runner that never got
    this fix.
    """

    def test_fake_uses_the_production_runner(self):
        self.assertIs(
            FakeMultiShardACA._run_per_split,
            AnalysisContainerArray._run_per_split,
        )


if __name__ == "__main__":
    unittest.main()
