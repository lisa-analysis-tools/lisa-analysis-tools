"""Shared CPU stand-ins for multi-shard (multi-GPU) structural tests.

CPU hosts cannot build a true multi-GPU ``AnalysisContainerArray`` (it enters
``cp.cuda.Device`` contexts when ``gpus`` is set), and even on a GPU node a
test suite cannot assume more than one visible device. So multi-shard logic is
exercised against duck-typed fakes whose ``gpu_map`` / ``gpu_splits`` /
``split_map`` / linear buffers look multi-shard but whose backing arrays are
NumPy. Device-context entries are recorded (not executed), so tests can assert
WHICH device each call ran under.

Ported from ``dev``'s ``tests/_multishard.py``, minus the F-stat comp entry
(the F-stat stack is B1, deferred on this branch).

.. note::
   ``tests/test_band_view_multi_shard.py`` keeps its own, older
   ``_FakeMultiShardACA``. It is deliberately NOT folded in here: that fixture
   aliases ``psd_shaped`` onto the data buffers so the ``kind="psd"`` path
   reads the same seeded values, whereas :class:`FakeMultiShardACA` below
   gives the PSD its own all-ones buffer (what the engines actually expect).
   Merging them would silently change what that test asserts.
"""

from __future__ import annotations

import threading

import numpy as np


class RecordingXp:
    """NumPy-backed fake ``xp`` with a cupy-like ``cuda`` namespace.

    ``cuda.Device(i)`` context entries push/pop a current-device stack and
    append to :attr:`device_log`, so a test can assert the device each
    operation was routed to.

    The current-device stack is **thread-local** (like cupy's real current
    device), so a threaded per-split dispatch records each worker on its own
    device without the workers clobbering each other. :attr:`device_log` stays
    shared (append is atomic under the GIL) for aggregate assertions.
    """

    __name__ = "numpy"

    def __init__(self):
        self.device_log = []
        self._tl = threading.local()

        outer = self

        class _Runtime:
            @staticmethod
            def getDevice():
                return outer._stack()[-1]

            @staticmethod
            def setDevice(gpu):
                outer._stack()[-1] = int(gpu)

            @staticmethod
            def deviceSynchronize():
                return None

        class _Cuda:
            runtime = _Runtime()

            class device:
                @staticmethod
                def Device(gpu):
                    return outer._device_ctx(gpu)

            @staticmethod
            def Device(gpu):
                return outer._device_ctx(gpu)

        self.cuda = _Cuda()

    def _stack(self):
        """Per-thread current-device stack (mirrors cupy thread-locality)."""
        stack = getattr(self._tl, "stack", None)
        if stack is None:
            stack = self._tl.stack = [0]
        return stack

    def _device_ctx(self, gpu):
        outer = self

        class _Ctx:
            def __enter__(self_inner):
                outer._stack().append(int(gpu))
                outer.device_log.append(int(gpu))
                return None

            def __exit__(self_inner, exc_type, exc, tb):
                outer._stack().pop()
                return False

        return _Ctx()

    @property
    def current_device(self):
        return self._stack()[-1]

    def get_default_memory_pool(self):
        """No-op cupy-style pool (the real ACA frees it after GPU applies)."""

        class _Pool:
            @staticmethod
            def free_all_blocks():
                return None

        return _Pool()

    # numpy passthrough for everything else (asarray, zeros, where, ...)
    def __getattr__(self, name):
        return getattr(np, name)


class FakeDeviceComp:
    """Duck-typed GB comp recording the device it was constructed on.

    Implements the replica contract the real comps record -- ``_build_device``
    plus ``args`` / ``kwargs`` (``lisatools.chunked_het.WDMComputationsBase``,
    ``gbgpu.gbcomps.GBFDComputations``, ``gbgpu.gbcomps.STFTGBComputations``)
    -- so ``lisatools.utils.devicereplicas.device_local_gb_comp`` can rebuild
    it and the router's per-shard replica dispatch is exercisable without a
    second GPU. ``information_matrix`` stamps the comp's OWN build device into
    its output, so a test can prove a shard ran against a device-local comp
    rather than the prototype.
    """

    def __init__(self, xp, tag: str = "comp", ndim: int = 3):
        from lisatools.utils.device import current_device

        self.xp = xp
        self.tag = tag
        self.ndim = int(ndim)
        self._build_device = current_device(xp)
        self.calls = []

    @property
    def args(self) -> tuple:
        # Empty: everything this comp needs to be rebuilt is keyword-only,
        # which also keeps the helper's "first positional is the domain
        # settings" branch out of the way.
        return ()

    @property
    def kwargs(self) -> dict:
        return dict(xp=self.xp, tag=self.tag, ndim=self.ndim)

    def _record(self, kind, holder, **info):
        self.calls.append(
            dict(kind=kind, holder=holder, build_device=self._build_device,
                 **info)
        )

    def information_matrix(self, params, holder, *, inds, noise_index,
                           **swap_kwargs):
        intra = np.asarray(noise_index)
        self._record("info", holder, intra=intra.copy())
        nd = len(inds) if inds is not None else self.ndim
        out = np.zeros((len(intra), nd, nd), dtype=float)
        for r in range(len(intra)):
            np.fill_diagonal(out[r], float(self._build_device))
        return out


class FakeMultiShardACA:
    """Duck-typed multi-shard ACA over NumPy arrays.

    Exposes the attributes consumed by the LAT shard router
    (``_ShardHolderView`` / ``_RoutedBandEngine``) and the ACA gather/scatter
    helpers: ``acs_total_entries``, ``gpus``, ``gpu_map``, ``gpu_splits``,
    ``split_map``, ``xp`` (a :class:`RecordingXp`), ``linear_data_arr``,
    ``linear_psd_arr``, ``data_shaped``, ``psd_shaped``, and (optionally)
    ``min_freq_inds``.

    Args:
        per_band_shape: per-row shape, e.g. ``(nchannels, data_length)``.
        num_acs: total rows (cells / walkers).
        num_shards: shard count; ``gpus = list(range(num_shards))``.
        layout: ``"striped"`` (row ``b`` on shard ``b % num_shards``) or
            ``"blocked"`` (contiguous ``np.array_split`` blocks, possibly
            uneven -- the main-ACA walker layout).
        with_min_freq_inds: attach a per-row int32 ``min_freq_inds`` store.
    """

    def __init__(self, per_band_shape: tuple, num_acs: int, num_shards: int,
                 layout: str = "striped", with_min_freq_inds: bool = False,
                 dtype=complex, run_threaded: bool = False):
        self.xp = RecordingXp()
        self.acs_total_entries = int(num_acs)
        self._num_acs = int(num_acs)
        self.gpus = list(range(num_shards))
        self.run_threaded = bool(run_threaded)
        self._thread_pool = None
        if layout == "striped":
            self.gpu_map = np.array(
                [b % num_shards for b in range(num_acs)], dtype=int
            )
            self.gpu_splits = [
                np.where(self.gpu_map == s)[0] for s in range(num_shards)
            ]
        elif layout == "blocked":
            self.gpu_splits = [
                np.asarray(chunk, dtype=int)
                for chunk in np.array_split(np.arange(num_acs), num_shards)
            ]
            self.gpu_map = np.empty(num_acs, dtype=int)
            for s, rows in enumerate(self.gpu_splits):
                self.gpu_map[rows] = s
        else:
            raise ValueError(f"unknown layout {layout!r}")
        self.split_map = np.zeros(num_acs, dtype=int)
        for s_i, split in enumerate(self.gpu_splits):
            self.split_map[split] = s_i
        # Intra-shard rank per global row (the real ACA's ac_to_intra).
        self.ac_to_intra = np.empty(num_acs, dtype=np.int32)
        for rows in self.gpu_splits:
            self.ac_to_intra[rows] = np.arange(len(rows), dtype=np.int32)

        self.per_band_shape = tuple(per_band_shape)
        per_row = int(np.prod(per_band_shape))
        # Flat per-shard buffers (row-major in intra-shard order), seeded so
        # row ``b`` holds the constant ``b + 1`` -- routing is verifiable.
        self.linear_data_arr = []
        self.linear_psd_arr = []
        for rows in self.gpu_splits:
            buf = np.zeros(len(rows) * per_row, dtype=dtype)
            psd = np.zeros(len(rows) * per_row, dtype=float)
            for intra, ac_i in enumerate(rows):
                buf[intra * per_row:(intra + 1) * per_row] = float(ac_i + 1)
                psd[intra * per_row:(intra + 1) * per_row] = 1.0
            self.linear_data_arr.append(buf)
            self.linear_psd_arr.append(psd)

        self.min_freq_inds = (
            np.arange(100, 100 + num_acs, dtype=np.int32)
            if with_min_freq_inds else None
        )
        self.start_freq_ind = np.arange(num_acs, dtype=np.int32) * 10

    @property
    def data_shaped(self):
        return [
            buf.reshape((len(rows),) + self.per_band_shape)
            for buf, rows in zip(self.linear_data_arr, self.gpu_splits)
        ]

    @property
    def psd_shaped(self):
        return [
            buf.reshape((len(rows),) + self.per_band_shape)
            for buf, rows in zip(self.linear_psd_arr, self.gpu_splits)
        ]

    def __len__(self):
        # Mirror AnalysisContainerArray.__len__ (== number of containers):
        # the engines read len(holder) as num_data/num_noise.
        return self._num_acs

    # --- per-split runner (mirrors AnalysisContainerArray) -------------
    @property
    def thread_pool(self):
        from concurrent.futures import ThreadPoolExecutor
        if self._thread_pool is None:
            self._thread_pool = ThreadPoolExecutor(
                max_workers=max(1, len(self.gpu_splits))
            )
        return self._thread_pool

    def _split_rows(self, index_arr) -> dict:
        """Group flat row positions by owning split: ``{split: rows}``."""
        split_per_row = self.split_map[np.asarray(index_arr, dtype=int)]
        return {
            int(s): np.where(split_per_row == s)[0]
            for s in np.unique(split_per_row)
        }

    def _run_per_split(self, worker, split_to_rows: dict,
                       run_threaded=None) -> None:
        """Run ``worker(split, rows)`` once per populated split."""
        if run_threaded is None:
            run_threaded = self.run_threaded
        items = [(s, rows) for s, rows in split_to_rows.items() if len(rows)]
        if run_threaded and len(items) > 1:
            futures = [
                self.thread_pool.submit(worker, s, rows) for s, rows in items
            ]
            for f in futures:
                f.result()
        else:
            for s, rows in items:
                worker(s, rows)

    def reference_rows(self):
        """(num_acs, *per_band_shape) reference in global row order."""
        out = np.zeros(
            (self.acs_total_entries,) + self.per_band_shape,
            dtype=self.linear_data_arr[0].dtype,
        )
        for s_i, split in enumerate(self.gpu_splits):
            shaped = self.data_shaped[s_i]
            for intra, ac_i in enumerate(split):
                out[int(ac_i)] = shaped[intra]
        return out
