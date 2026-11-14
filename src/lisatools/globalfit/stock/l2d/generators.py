"""A block's waveform, built on first use, and the params-based generators that call it.

``Settings.signal_gen`` is a :class:`SignalGen`, not a closure. ``CurrentInfoGlobalFit`` deep-copies the
settings once, and a deep copy keeps a function as it is (with its closure). An object is copied in the same
pass, so its :class:`WaveformHandle` points at the copied orbits and data settings that the engine uses
(``curr.general_info``). The waveform itself is built after the copy, on the first call (``setup_acs`` has
selected the device by then). The PE move, ``RecipeContext.signal_gens`` and the TD generator share it.
"""
from __future__ import annotations

from typing import Any
import warnings

import numpy as np


class WaveformHandle:
    """``waveform_class(**init_kwargs)``, built once, on the first :meth:`get`."""

    def __init__(self, waveform_class: type, init_kwargs: dict[str, Any]):
        self.waveform_class = waveform_class
        self.init_kwargs = init_kwargs
        self._waveform = None

    @property
    def built(self) -> bool:
        return self._waveform is not None

    def get(self):
        if self._waveform is None:
            self._waveform = self.waveform_class(**self.init_kwargs)
        return self._waveform

    def __deepcopy__(self, memo):
        # copying a built waveform would copy its device arrays: copy only an unbuilt handle
        if self.built:
            raise RuntimeError("a built WaveformHandle is not copied; deep-copy the settings before the first call.")
        from copy import deepcopy

        copy = WaveformHandle(self.waveform_class, deepcopy(self.init_kwargs, memo))
        memo[id(self)] = copy
        return copy


class SignalGen:
    """``fn(*sampling_params, **kwargs)``: the sampling->waveform transform, then ``waveform.<method>``.

    ``waveform`` is a waveform object or a :class:`WaveformHandle` (resolved on the first call). Scalar params
    (one leaf) give one signal, the ``Settings.signal_gen`` contract; 1-D params (a batch, e.g. over walkers)
    give a ``DomainBaseArray``. ``kwargs`` are added to ``runtime_kwargs`` (e.g. ``td_settings``).
    """

    # run.py build_acs: one call per leaf with 1-D params over the walkers (as subtract_initial_signal)
    supports_batch = True

    def __init__(self, waveform, method: str, transform, runtime_kwargs: dict[str, Any]):
        self.source = waveform
        self.method = method
        self.transform = transform
        self.runtime_kwargs = dict(runtime_kwargs)

    @property
    def waveform(self):
        return self.source.get() if isinstance(self.source, WaveformHandle) else self.source

    def __call__(self, *params, **kwargs):
        waveform = self.waveform
        xp = getattr(waveform, "xp", np)
        single = all(np.ndim(p) == 0 for p in params)
        coords = np.stack([np.atleast_1d(np.asarray(p, dtype=float)) for p in params], axis=-1)
        if self.transform is not None:
            coords = self.transform.both_transforms(coords)
        # contiguous columns, as subtract_initial_signal passes them
        columns = [xp.ascontiguousarray(col) for col in xp.asarray(coords).T]
        columns_host = [np.ascontiguousarray(col) for col in coords.T]
        try:
            out = getattr(waveform, self.method)(*columns, **{**self.runtime_kwargs, **kwargs})
        except TypeError as e:
            out = getattr(waveform, self.method)(*columns_host, **{**self.runtime_kwargs, **kwargs})
            warnings.warn(f"waveform.{self.method} failed on device arrays, ran on host instead: {e}")
        
        return out[0] if single else out
