"""Device-pinning helpers: the canonical multi-GPU discipline for LAT.

The rule (see ``docs/conventions.md``): pin the run's *main* device once at
operation entry (:func:`pin_main_device`); every multi-shard structure
(``AnalysisContainerArray``, ``DomainComputationGroupArray``, sub-band
buffers) enters and restores its own per-shard device contexts
(:func:`device_context`); kernels always launch from inside the owning
context. CuPy's current device is thread-local, so per-split worker threads
must enter their shard's context themselves.

Everything here is a stateless function on purpose: device identity must
never live on ``Backend`` singletons (they re-resolve by registry name on
deepcopy/pickle) nor on pre-build settings objects.

All helpers are CPU-safe no-ops when ``xp`` is numpy (or any module without
a ``cuda`` attribute) so call sites need no ``uses_cupy`` guards.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Sequence, Union

__all__ = [
    "device_context",
    "pin_main_device",
    "current_device",
    "jax_device_context",
    "assert_peer_access",
]


def _first_gpu(gpus: Union[int, Sequence[int], None]) -> Optional[int]:
    if gpus is None:
        return None
    if isinstance(gpus, int):
        return gpus
    return int(gpus[0])


def device_context(xp, device: Optional[int]):
    """Context manager placing work on ``device`` under ``xp``.

    Returns ``xp.cuda.Device(device)`` on the CuPy path and a
    ``nullcontext()`` when ``device`` is None or ``xp`` has no ``cuda``
    (numpy / CPU path).
    """
    if device is None or not hasattr(xp, "cuda"):
        return nullcontext()
    return xp.cuda.Device(int(device))


def pin_main_device(xp, gpus: Union[int, Sequence[int], None]) -> Optional[int]:
    """Pin the process-current device to the run's main GPU (``gpus[0]``).

    No-op on CPU (``gpus is None`` or numpy ``xp``). Returns the previous
    device id (None when nothing was pinned) so callers that need to restore
    can ``xp.cuda.runtime.setDevice(prev)`` afterwards.
    """
    first = _first_gpu(gpus)
    if first is None or not hasattr(xp, "cuda"):
        return None
    prev = int(xp.cuda.runtime.getDevice())
    xp.cuda.runtime.setDevice(first)
    return prev


def current_device(xp) -> Optional[int]:
    """The process-current device id under ``xp`` (None on the CPU path)."""
    if not hasattr(xp, "cuda"):
        return None
    return int(xp.cuda.runtime.getDevice())


#: ``(sorted device tuple) -> None`` for peer-access sets already verified.
#: Peer-access support is a fixed property of the machine's topology, so the
#: query runs once per device set per process rather than per kernel launch.
_PEER_ACCESS_OK: set = set()


def assert_peer_access(xp, devices, *, context: str = "") -> None:
    """Raise unless every ordered pair in ``devices`` supports peer access.

    Some multi-shard code paths write their per-shard results straight into an
    output array that lives on the CALLER's device, from inside the shard's
    own ``device_context``. cupy makes that work by enabling peer access
    automatically -- but only where the topology supports it. Where it does
    not, the same code is an *illegal memory access* from a CUDA kernel, which
    surfaces as a bare ``GPUassert`` with no indication of the cause.

    Peer access is NOT guaranteed on a multi-GPU node. It is absent across
    CPU sockets on most chipsets (``nvidia-smi topo -m`` showing ``SYS``),
    under some IOMMU/virtualisation configurations, and with MIG enabled.
    ``nvidia-smi topo -p2p r`` reports the machine's matrix.

    No-op on the CPU path, for fewer than two distinct devices, and on any
    ``xp`` whose runtime does not expose ``deviceCanAccessPeer`` (the fake
    ``xp`` used by the CPU multi-shard tests).
    """
    if not hasattr(xp, "cuda"):
        return
    devs = sorted({int(d) for d in devices if d is not None})
    if len(devs) < 2:
        return
    key = tuple(devs)
    if key in _PEER_ACCESS_OK:
        return
    runtime = getattr(getattr(xp, "cuda", None), "runtime", None)
    can = getattr(runtime, "deviceCanAccessPeer", None)
    if can is None:
        return
    for a in devs:
        for b in devs:
            if a == b:
                continue
            if not bool(can(a, b)):
                where = f" ({context})" if context else ""
                raise RuntimeError(
                    f"peer access is unavailable from device {a} to device "
                    f"{b}, but this code path{where} writes across devices "
                    "and needs it. This can result in OOM errors. Run on a "
                    "single GPU (GPUS=<one id>), or shard only across devices "
                    "that report OK in `nvidia-smi topo -p2p r`. Peer access is "
                    "commonly absent across CPU sockets (topo 'SYS'), under "
                    "some IOMMU/virtualisation setups, and with MIG enabled."
                )
    _PEER_ACCESS_OK.add(key)


def jax_device_context(device: Optional[int], *, kind: str = "gpu"):
    """Pin JAX's default device to match cupy ``device`` (JAX-backed gens).

    CuPy's :func:`device_context` does NOT move JAX ops: JAX placement is
    governed by ``jax.default_device`` / ``jax.device_put``. So any
    per-device JAX generation (the phentax MBH waveform) must ALSO enter this
    context, otherwise scalar ``device_put(x, device=None)`` inputs and the
    traced kernels land on JAX's *default* device (gpu0) regardless of the
    surrounding cupy :func:`device_context`. This is the JAX twin of
    :func:`device_context`; the two are entered together at each per-shard
    source-generation site.

    Assumes JAX and cupy enumerate CUDA devices in the same order (the DLPack
    precondition already documented in ``lisatools/jax/jaxbase.py``): the
    cupy device id indexes ``jax.devices(kind)`` directly.

    Returns ``nullcontext()`` when ``device`` is None (CPU / single-device /
    the run's primary device), when JAX is unavailable, or when JAX cannot
    see a device at ``jax.devices(kind)[device]`` -- so call sites need no
    guards and single-GPU/CPU behaviour is unchanged.
    """
    if device is None:
        return nullcontext()
    try:
        import jax
    except (ImportError, ModuleNotFoundError):
        return nullcontext()
    try:
        return jax.default_device(jax.devices(kind)[int(device)])
    except (RuntimeError, IndexError):
        # JAX on CPU while cupy on GPU, or fewer JAX devices than cupy sees.
        return nullcontext()
