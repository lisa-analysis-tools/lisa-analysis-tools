"""Developer sanity check for the two-GPU STFT GB path. Not part of the standard test suite.

Every coherence check compares a two-shard computation against a ONE-shard computation of the same
case, so it fails on any difference that sharding introduces, whatever the physics. Everything
skips unless ``STFT_MULTI_GPU_CHECK=1`` is set and at least two CUDA devices are visible.

Run it inside a GPU job step, on two free devices, from ``lisa-analysis-tools/``::

    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES=<two free physical GPUs>
    STFT_MULTI_GPU_CHECK=1 python -m unittest tests.test_stft_multi_gpu -v

Switches:

    STFT_MULTI_GPU_CHECK=1        enable the checks
    STFT_MULTI_GPU_LOAD=0         leave the second device idle during the coherence checks
    STFT_MULTI_GPU_REPEATS=<n>    fresh-buffer trials per coherence check (default 5)
    GB_TEST_GPUS=<a>,<b>          logical devices to shard over (default 0,1)

Two groups of checks:

* **Coherence**, with a busy process on the second device by default. Likelihoods, swap
  likelihoods, templates, buffer contents, per-cell source terms and the tempering swap difference
  must equal the one-shard values. Each trial builds a fresh buffer, because the build injects
  sources through the same cross-device paths.
* **Concurrency**, with the second device idle, because load would slow one shard and skew the
  timing. A launch must return well before its kernel finishes, and a two-shard call must take less
  than ``OVERLAP_LIMIT`` of the serial sum of its shards.

The load and the repeats are there because the defects this file guards against are races: on
2026-09-14 one of them passed with the second device idle and failed every trial with it loaded.
A single clean run proves little.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
import unittest

import numpy as np

from . import test_stft_store_windows as harness

CHECK_ENABLED = os.environ.get("STFT_MULTI_GPU_CHECK", "0") == "1"
LOAD_ENABLED = os.environ.get("STFT_MULTI_GPU_LOAD", "1") != "0"
REPEATS = int(os.environ.get("STFT_MULTI_GPU_REPEATS", "5"))
# ? A two-shard call must take less than this fraction of the serial sum of its shards.
OVERLAP_LIMIT = 0.8
# ? Rows per shard in the concurrency checks: enough that a kernel lasts milliseconds.
TIMING_ROWS_PER_SHARD = 2000


def _device_count() -> int:
    try:
        import cupy as cp

        return int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return 0


def _skip_reason():
    """Why the checks cannot run, or None. Cheap when disabled: nothing is imported."""
    if not CHECK_ENABLED:
        return "developer check: set STFT_MULTI_GPU_CHECK=1 to run it"
    if not harness._have_gbgpu_stft():
        return "requires gbgpu.gbcomps.STFTGBComputations"
    if harness._cuda_backend_name() is None:
        return "requires a usable CUDA backend"
    count = _device_count()
    if count < 2:
        return f"requires two visible CUDA devices, found {count}"
    return None


SKIP_REASON = _skip_reason()


def _device_pair() -> tuple[int, int]:
    """The two logical devices to shard over: the first two of GB_TEST_GPUS, else 0 and 1."""
    requested = [part for part in os.environ.get("GB_TEST_GPUS", "").split(",") if part.strip()]
    if len(requested) >= 2:
        return int(requested[0]), int(requested[1])
    return 0, 1


@contextlib.contextmanager
def _shard_over(devices: str):
    """Set GB_TEST_GPUS, which the harness reads when it builds, and restore it afterwards."""
    previous = os.environ.get("GB_TEST_GPUS")
    os.environ["GB_TEST_GPUS"] = devices
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GB_TEST_GPUS", None)
        else:
            os.environ["GB_TEST_GPUS"] = previous


_LOAD_SCRIPT = """
import sys
import cupy as cp
with cp.cuda.Device(int(sys.argv[1])):
    matrix = cp.random.random((4096, 4096))
    print("ready", flush=True)
    while True:
        matrix = (matrix @ matrix) / cp.linalg.norm(matrix)
        cp.cuda.runtime.deviceSynchronize()
"""


@contextlib.contextmanager
def _busy_device(device: int, enabled: bool):
    """Keep ``device`` busy while the context is open, from a process with its own CUDA context."""
    if not enabled:
        yield
        return
    process = subprocess.Popen(
        [sys.executable, "-c", _LOAD_SCRIPT, str(device)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    try:
        if process.stdout.readline().strip() != "ready":
            raise RuntimeError(f"the load process on device {device} did not start")
        yield
    finally:
        process.terminate()
        process.wait(timeout=30)


def _drain(devices) -> None:
    import cupy as cp

    for device in devices:
        with cp.cuda.Device(int(device)):
            cp.cuda.runtime.deviceSynchronize()


class _Case:
    """The store-window harness case sharded over ``devices``, with its live sources."""

    def __init__(self, devices: str):
        self.devices = devices
        with _shard_over(devices):
            self.data = harness.build_case(harness._cuda_backend_name())
        sorter = harness.make_sorter(self.data, self.data["windows"])
        self.xp = sorter.xp
        self.ids = sorter.xp.arange(sorter.num_sources)[sorter.inds]
        self.specials = sorter.special_band_inds[self.ids]
        self.params = sorter.coords[self.ids]
        self.walkers = sorter.walker_inds[self.ids]
        self.n_vals = sorter.band_N_vals[sorter.band_inds[self.ids]]
        self.cells = sorter.xp.asarray(np.unique(harness.host(self.specials)))

    def fresh_buffer(self, **kwargs):
        """A new buffer over every live cell, and the slot of each source in it."""
        with _shard_over(self.devices):
            buffer = harness.make_sorter(self.data, self.data["windows"]).get_buffer(
                self.data["aca"], self.cells, **kwargs)
        return buffer, buffer.get_index(self.specials)

    def buffer_with_templates(self):
        """A tempering-style buffer with every source injected into its template twin."""
        buffer, slots = self.fresh_buffer(use_template_arr=True)
        buffer.add_sources_to_template_buffer(self.params, slots, self.n_vals)
        return buffer, slots


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class StftMultiGpuCoherenceCheck(unittest.TestCase):
    """Two shards must reproduce one shard, over repeated fresh buffers, with the second device busy.

    Nothing here drains a device before reading a result: a drain would order away exactly the races
    these checks exist to catch. Results are read through the same paths the sampler uses.
    """

    @classmethod
    def setUpClass(cls):
        first, second = _device_pair()
        cls.one = _Case(f"{first}")
        cls.two = _Case(f"{first},{second}")
        cls._load = contextlib.ExitStack()
        cls._load.enter_context(_busy_device(second, LOAD_ENABLED))

    @classmethod
    def tearDownClass(cls):
        cls._load.close()

    def _trials(self, compute, compare, what):
        """Compare ``compute(two-shard case)`` against the one-shard value, REPEATS times."""
        reference = compute(self.one)
        for trial in range(REPEATS):
            with self.subTest(trial=trial):
                compare(compute(self.two), reference,
                        f"{what}: two shards left the one-shard value in trial {trial}")

    # ---------------- kernels through the engine ----------------

    def test_get_ll(self):
        def compute(case):
            buffer, slots = case.fresh_buffer()
            return harness.host(buffer.get_ll(case.params, slots, slots, case.n_vals))

        self._trials(compute, np.testing.assert_array_equal, "get_ll")

    def test_get_swap_ll(self):
        def compute(case):
            buffer, slots = case.fresh_buffer()
            shifted = case.params * case.xp.asarray(
                np.where(np.arange(harness.NDIM) == 1, 1.00001, 1.0))
            return harness.host(buffer.get_swap_ll(case.params, shifted, slots, case.n_vals))

        self._trials(compute, np.testing.assert_array_equal, "get_swap_ll")

    def test_fill_template(self):
        def compute(case):
            buffer, slots = case.buffer_with_templates()
            return harness.host(buffer._materialize(buffer.template_buffer))[harness.host(slots)]

        self._trials(compute, np.testing.assert_array_equal, "template fill")

    def test_buffer_contents(self):
        def compute(case):
            # * The build fills each cell from the parent and injects its sources, both across devices.
            buffer, slots = case.fresh_buffer()
            order = harness.host(slots)
            return (harness.host(buffer._materialize(buffer.band_buffer))[order],
                    harness.host(buffer._materialize(buffer.psd_buffer))[order])

        def compare(got, reference, message):
            np.testing.assert_array_equal(got[0], reference[0], err_msg=f"residual. {message}")
            np.testing.assert_array_equal(got[1], reference[1], err_msg=f"inverse CSD. {message}")

        self._trials(compute, compare, "buffer contents")

    def test_information_matrix(self):
        from lisatools.globalfit.moves.gbbands import _RoutedBandEngine

        # Sampled parameters of the move; fddot (index 3) is fixed by the fill dict.
        test_inds = [0, 1, 2, 4, 5, 6, 7, 8]

        def compute(case):
            # * The router launches every shard before collecting, so the shard Fisher kernels overlap.
            physical = case.data["transform"].both_transforms(case.params, xp=case.xp)
            info = _RoutedBandEngine.route_information_matrix(
                case.data["comp"], case.data["aca"], physical,
                inds=test_inds, noise_index=case.walkers.astype(case.xp.int32))
            return harness.host(info)

        self._trials(compute, np.testing.assert_array_equal, "information matrix")

    # ---------------- reductions and tempering ----------------

    def test_source_terms(self):
        def compute(case):
            buffer, slots = case.buffer_with_templates()
            per_cell = harness.host(buffer.band_likelihoods(source_only=True))
            columns = np.arange(int(buffer.num_bands_now)).reshape(-1, harness.NTEMPS).T
            selected = harness.host(buffer.band_likelihoods(source_only=True, cells=columns))
            return per_cell[harness.host(slots)], selected, per_cell[columns]

        def compare(got, reference, message):
            # ? Tolerance, not equality: a shard reduces fewer cells per batch, which can move the
            # ? last bits of a contraction without changing its value.
            np.testing.assert_allclose(got[0], reference[0], rtol=1e-12, atol=0,
                                       err_msg=f"per-cell source terms. {message}")
            np.testing.assert_allclose(got[1], got[2], rtol=1e-12, atol=0,
                                       err_msg=f"column selection against the full terms. {message}")

        self._trials(compute, compare, "source terms")

    def test_tempering_swap_difference(self):
        def compute(case):
            buffer, _slots = case.buffer_with_templates()
            pairs = {}
            for slot, (temp, walker, band) in enumerate(harness.host(buffer.unique_band_combos)):
                pairs.setdefault((int(walker), int(band)), {})[int(temp)] = slot
            keys = sorted(key for key, temps in pairs.items() if 0 in temps and 1 in temps)
            cold = np.asarray([pairs[key][0] for key in keys])
            hot = np.asarray([pairs[key][1] for key in keys])
            before = harness.host(buffer.band_likelihoods(source_only=True))
            buffer.swap_template_slots(buffer.xp.asarray(cold), buffer.xp.asarray(hot))
            after = harness.host(buffer.band_likelihoods(source_only=True))
            delta = after - before
            return keys, np.stack([delta[cold], delta[hot]], axis=1), np.abs(before).max()

        def compare(got, reference, message):
            self.assertEqual(got[0], reference[0], msg=f"different temperature pairs. {message}")
            np.testing.assert_allclose(got[1], reference[1], rtol=0, atol=1e-12 * reference[2],
                                       err_msg=message)

        self._trials(compute, compare, "tempering swap difference")


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class StftMultiGpuConcurrencyCheck(unittest.TestCase):
    """The shards of one call must run at the same time.

    Built at production NT and stencil so a kernel lasts milliseconds, and run with the second
    device idle: a loaded shard runs slower, which pushes a perfectly overlapped call toward the
    serial sum.
    """

    @classmethod
    def setUpClass(cls):
        import cupy as cp

        first, second = _device_pair()
        cls.devices = [first, second]
        # ! The harness reads these module constants when it builds; restore them straight after,
        # ! so the small case the other checks use is untouched.
        saved = (harness.NT_STFT, harness.N_SIDE_BINS, harness.TOBS)
        harness.NT_STFT, harness.N_SIDE_BINS = 273, 10
        harness.TOBS = harness.NT_STFT * harness.BIG_DT
        try:
            with _shard_over(f"{first},{second}"):
                data = harness.build_case(harness._cuda_backend_name())
                sorter = harness.make_sorter(data, None)
                cells = sorter.xp.asarray(np.unique(harness.host(sorter.special_band_inds[sorter.inds])))
                cls.buffer = sorter.get_buffer(data["aca"], cells)
        finally:
            harness.NT_STFT, harness.N_SIDE_BINS, harness.TOBS = saved

        buffer, xp = cls.buffer, cls.buffer.xp
        ids = harness.host(sorter.xp.arange(sorter.num_sources)[sorter.inds])
        slots = harness.host(buffer.get_index(sorter.special_band_inds[sorter.xp.asarray(ids)]))
        n_vals = harness.host(sorter.band_N_vals[sorter.band_inds[sorter.xp.asarray(ids)]])
        split_of_row = np.asarray(buffer.split_map)[slots]

        def rows(splits, count):
            """Tile the live sources onto the cells of ``splits`` up to ``count`` rows."""
            take = np.resize(np.nonzero(np.isin(split_of_row, splits))[0], count)
            return (sorter.coords[sorter.xp.asarray(ids[take])], xp.asarray(slots[take]),
                    xp.asarray(n_vals[take]))

        cls.rows_first = rows([0], TIMING_ROWS_PER_SHARD)
        cls.rows_second = rows([1], TIMING_ROWS_PER_SHARD)
        cls.rows_both = rows([0, 1], 2 * TIMING_ROWS_PER_SHARD)

        # * Bare kernel launches, one per shard, prepared on their own devices so nothing else is timed.
        engine = buffer._likelihood_engine
        cls.launches = []
        for split, device in enumerate(buffer.gpus):
            params, slot_rows, _n = rows([split], TIMING_ROWS_PER_SHARD)
            intra_host = np.asarray(buffer.ac_to_intra)[harness.host(slot_rows)]
            with cp.cuda.Device(int(device)):
                comp = engine._comp_for_split(buffer, split, device)
                physical = buffer.transform_fn.both_transforms(xp.asarray(harness.host(params)), xp=xp)
                intra = xp.asarray(intra_host, dtype=xp.int32)
                starts = engine._split_start_inds(buffer, split)
            cls.launches.append((int(device), comp, physical, intra, starts))

    def _mean_call_seconds(self, call, repeats=5):
        for _ in range(2):
            call()
        _drain(self.buffer.gpus)
        start = time.perf_counter()
        for _ in range(repeats):
            call()
        _drain(self.buffer.gpus)
        return (time.perf_counter() - start) / repeats

    def test_launch_returns_before_its_kernel(self):
        import cupy as cp

        for device, comp, params, intra, starts in self.launches:
            def launch():
                with cp.cuda.Device(device):
                    comp.get_ll_stft(params, data_index=intra, noise_index=intra,
                                     start_freq_inds=starts)

            launch()
            _drain(self.buffer.gpus)
            start = time.perf_counter()
            launch()
            returned = time.perf_counter() - start
            _drain(self.buffer.gpus)
            finished = time.perf_counter() - start
            print(f"\n  device {device}: launch returned after {returned * 1e3:.2f} ms, "
                  f"kernel finished after {finished * 1e3:.2f} ms")
            if finished < 2e-3:
                self.skipTest(f"kernel on device {device} too short to time ({finished * 1e3:.2f} ms)")
            self.assertLess(
                returned, 0.5 * finished,
                msg=f"a launch on device {device} blocked until its kernel finished, so the shards "
                    "of one call cannot overlap",
            )

    def test_two_shard_call_overlaps(self):
        def get_ll(rows):
            params, slots, n_vals = rows
            return lambda: self.buffer.get_ll(params, slots, slots, n_vals)

        first = self._mean_call_seconds(get_ll(self.rows_first))
        second = self._mean_call_seconds(get_ll(self.rows_second))
        both = self._mean_call_seconds(get_ll(self.rows_both))
        serial = first + second
        print(f"\n  shard 0 alone {first * 1e3:.2f} ms, shard 1 alone {second * 1e3:.2f} ms, "
              f"both {both * 1e3:.2f} ms: {both / serial:.2f} of the serial sum "
              f"(limit {OVERLAP_LIMIT}, full overlap {max(first, second) / serial:.2f})")
        self.assertLess(
            both, OVERLAP_LIMIT * serial,
            msg=f"a two-shard get_ll took {both / serial:.2f} of its shards' serial sum; the shards "
                "are not running at the same time",
        )


if __name__ == "__main__":
    unittest.main()
