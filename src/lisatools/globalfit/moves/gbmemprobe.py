"""GPU memory probe for the GB block.

Off by default. With ``GBDebugSettings.mem_probe`` every ``_ProposeTimer`` span
records the CuPy pool peak, the pool and device bytes at entry and exit, and the
call sites of large allocations. One table is logged per propose. With
``mem_probe_file`` set, the same record is appended as one JSON line.

The probe only observes; it never frees, collects or reorders anything.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import weakref
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# * Used when a probe is built without settings, and by the dataclass defaults.
SITE_BYTES_DEFAULT = 1024**2
SITE_DEPTH_DEFAULT = 3

_GB = 1024.0**3
# ? Frames from these paths are skipped when locating the caller of an allocation.
_SKIP_PATH_PARTS = ("/cupy/", "/cupyx/", "/cupy_backends/", os.path.abspath(__file__))

try:
    from cupy.cuda import memory_hook as _cupy_memory_hook

    _HookBase = _cupy_memory_hook.MemoryHook
except Exception:  # CPU-only environment: the probe stays disabled
    _HookBase = object


def _caller_site(depth: int) -> str:
    frame = sys._getframe(2)
    parts = []
    while frame is not None and len(parts) < depth:
        if not any(part in frame.f_code.co_filename for part in _SKIP_PATH_PARTS):
            parts.append(f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}:{frame.f_code.co_name}")
        frame = frame.f_back
    return " < ".join(parts) if parts else "<unknown>"


class _SpanCall:
    """Memory record of one span call."""

    __slots__ = (
        "name", "entry_used", "entry_total", "peak_used", "peak_total",
        "exit_used", "exit_total", "entry_device_used", "exit_device_used",
        "largest_alloc", "sites", "failed_request", "error",
    )

    def __init__(self, name, used, total, device_used):
        self.name = name
        self.entry_used = self.peak_used = self.exit_used = used
        self.entry_total = self.peak_total = self.exit_total = total
        self.entry_device_used = self.exit_device_used = device_used
        self.largest_alloc = 0
        self.sites = {}
        self.failed_request = None
        self.error = None


class _PoolPeakHook(_HookBase):
    """Tracks pool bytes with running counters, resynchronised at every span boundary."""

    name = "GBMemProbeHook"

    def __init__(self, probe):
        self.probe = probe
        self.used = 0
        self.total = 0
        self.last_device_request = 0
        # * Live large allocations: pointer -> (bytes, site, span name). Lets a check name what survives.
        self.live = {}

    def malloc_postprocess(self, **kwargs):
        if not kwargs.get("mem_ptr"):
            return
        mem_size = kwargs["mem_size"]
        self.used += mem_size
        stack = self.probe.stack
        for call in stack:
            if self.used > call.peak_used:
                call.peak_used = self.used
        if stack:
            innermost = stack[-1]
            if mem_size > innermost.largest_alloc:
                innermost.largest_alloc = mem_size
            if mem_size >= self.probe.site_bytes:
                site = _caller_site(self.probe.site_depth)
                key = (site, mem_size)
                innermost.sites[key] = innermost.sites.get(key, 0) + 1
                self.live[kwargs["mem_ptr"]] = (mem_size, site, innermost.name)

    def free_postprocess(self, **kwargs):
        self.used -= kwargs["mem_size"]
        self.live.pop(kwargs.get("mem_ptr"), None)

    def alloc_preprocess(self, **kwargs):
        self.last_device_request = kwargs["mem_size"]

    def alloc_postprocess(self, **kwargs):
        if not kwargs.get("mem_ptr"):
            return
        self.total += kwargs["mem_size"]
        for call in self.probe.stack:
            if self.total > call.peak_total:
                call.peak_total = self.total


class _SpanSummary:
    """Aggregate of all calls of one span name within one propose."""

    def __init__(self, name):
        self.name = name
        self.calls = 0
        self.max_call = None
        self.max_rise = 0
        self.largest_alloc = 0
        self.sites = {}

    def add(self, call: _SpanCall):
        self.calls += 1
        if self.max_call is None or call.peak_used > self.max_call.peak_used:
            self.max_call = call
        self.max_rise = max(self.max_rise, call.peak_used - call.entry_used)
        self.largest_alloc = max(self.largest_alloc, call.largest_alloc)
        for key, count in call.sites.items():
            self.sites[key] = self.sites.get(key, 0) + count
        call.sites = {}

    def as_dict(self, top_sites: int = 12) -> dict:
        call = self.max_call
        sites = sorted(self.sites.items(), key=lambda kv: -kv[0][1] * kv[1])[:top_sites]
        return dict(
            name=self.name,
            calls=self.calls,
            peak_used_bytes=call.peak_used,
            peak_total_bytes=call.peak_total,
            entry_used_bytes=call.entry_used,
            exit_used_bytes=call.exit_used,
            entry_total_bytes=call.entry_total,
            exit_total_bytes=call.exit_total,
            entry_device_used_bytes=call.entry_device_used,
            exit_device_used_bytes=call.exit_device_used,
            max_rise_bytes=self.max_rise,
            largest_alloc_bytes=self.largest_alloc,
            sites=[
                dict(site=site, alloc_bytes=size, count=count) for (site, size), count in sites
            ],
        )


class GBMemProbe:
    """Per-propose memory probe; see the module docstring."""

    _active = None

    def __init__(self, xp, move_name: str, context: dict | None = None, settings=None):
        self.xp = xp
        self.move_name = move_name
        self.context = dict(context or {})
        self.probe_file = getattr(settings, "mem_probe_file", None)
        self.site_bytes = int(getattr(settings, "mem_probe_site_bytes", SITE_BYTES_DEFAULT))
        self.site_depth = int(getattr(settings, "mem_probe_site_depth", SITE_DEPTH_DEFAULT))
        self.pool = xp.get_default_memory_pool()
        self.stack: list[_SpanCall] = []
        self.summaries: dict[str, _SpanSummary] = {}
        self.lifetime: list[dict] = []
        self._watched: dict[str, weakref.ref] = {}
        self._reported_errors: set[int] = set()
        self.hook = _PoolPeakHook(self)
        self._install()

    # ------------------------------------------------------------------
    def _install(self):
        if GBMemProbe._active is not None:
            GBMemProbe._active._uninstall()
        self._resync()
        self.hook.__enter__()
        GBMemProbe._active = self

    def _uninstall(self):
        if GBMemProbe._active is self:
            try:
                self.hook.__exit__(None, None, None)
            except Exception:
                pass
            GBMemProbe._active = None

    def _device_used(self) -> int:
        free, total = self.xp.cuda.runtime.memGetInfo()
        return int(total - free)

    def _resync(self):
        self.hook.used = int(self.pool.used_bytes())
        self.hook.total = int(self.pool.total_bytes())

    # ------------------------------------------------------------------
    @contextmanager
    def span(self, name: str):
        self._resync()
        call = _SpanCall(name, self.hook.used, self.hook.total, self._device_used())
        self.stack.append(call)
        try:
            yield
        except BaseException as exc:
            call.error = type(exc).__name__
            if "OutOfMemory" in call.error:
                call.failed_request = self.hook.last_device_request
            if id(exc) not in self._reported_errors:
                # * The innermost span sees the error first; log the open stack once.
                self._reported_errors.add(id(exc))
                self._resync()
                for open_call in self.stack:
                    open_call.exit_used = self.hook.used
                    open_call.exit_total = self.hook.total
                self._log_error(exc)
            raise
        finally:
            self.stack.pop()
            if call.error is None:
                self._resync()
                call.exit_used = self.hook.used
                call.exit_total = self.hook.total
            call.exit_device_used = self._device_used()
            summary = self.summaries.get(name)
            if summary is None:
                summary = self.summaries[name] = _SpanSummary(name)
            summary.add(call)

    # ------------------------------------------------------------------
    def watch(self, tag: str, obj, span_name: str) -> None:
        """Keep a weak reference to ``obj`` and the live allocations made so far in ``span_name``.

        :meth:`check_released` then reports whether the object and those allocations outlive its ``del``.
        """
        try:
            ref = weakref.ref(obj)
        except TypeError:
            logger.warning("[GB_MEMPROBE %s] %s is not weak-referenceable", self.move_name, tag)
            ref = None
        pointers = {ptr for ptr, (_, _, span) in self.hook.live.items() if span == span_name}
        self._watched[tag] = (ref, pointers)

    def check_released(self, tag: str) -> None:
        watched = self._watched.pop(tag, None)
        if watched is None:
            return
        ref, pointers = watched
        survivors = {}
        for ptr in pointers & self.hook.live.keys():
            size, site, _ = self.hook.live[ptr]
            count, total = survivors.get(site, (0, 0))
            survivors[site] = (count + 1, total + size)
        self.lifetime.append(
            dict(
                tag=tag,
                alive=None if ref is None else ref() is not None,
                pool_used_bytes=int(self.pool.used_bytes()),
                survivor_bytes=sum(total for _, total in survivors.values()),
                survivor_sites=[
                    dict(site=site, count=count, bytes=total)
                    for site, (count, total) in sorted(survivors.items(), key=lambda kv: -kv[1][1])[:10]
                ],
                time=time.time(),
            )
        )

    # ------------------------------------------------------------------
    def _record(self) -> dict:
        return dict(
            move=self.move_name,
            time=time.time(),
            context=self.context,
            device_total_bytes=int(self.xp.cuda.runtime.memGetInfo()[1]),
            spans=[s.as_dict() for s in self.summaries.values()],
            lifetime=self.lifetime,
        )

    def _write(self, record: dict) -> None:
        if not self.probe_file:
            return
        with open(self.probe_file, "a") as handle:
            handle.write(json.dumps(record) + "\n")

    def _log_error(self, exc) -> None:
        failed = self.stack[-1].failed_request if self.stack else None
        open_spans = [
            dict(
                name=c.name, entry_used_bytes=c.entry_used, peak_used_bytes=c.peak_used,
                peak_total_bytes=c.peak_total, entry_device_used_bytes=c.entry_device_used,
                sites=[dict(site=s, alloc_bytes=b, count=n) for (s, b), n in
                       sorted(c.sites.items(), key=lambda kv: -kv[0][1] * kv[1])[:12]],
            )
            for c in self.stack
        ]
        record = self._record()
        record.update(error=type(exc).__name__, failed_request_bytes=failed, open_spans=open_spans)
        self._write(record)
        logger.error(
            "[GB_MEMPROBE %s] %s inside %s; failed device request %s GB; device used %.2f GB",
            self.move_name, type(exc).__name__, " > ".join(c.name for c in self.stack),
            "n/a" if failed is None else f"{failed / _GB:.2f}", self._device_used() / _GB,
        )

    def finish(self) -> None:
        """Log the per-span table, append the JSON record, and remove the hook."""
        self._uninstall()
        record = self._record()
        self._write(record)
        header = (
            f"{'span':<24}{'calls':>7}{'peak used':>11}{'peak pool':>11}{'max rise':>10}"
            f"{'used in':>9}{'used out':>10}{'dev in':>8}{'dev out':>9}{'largest':>10}"
        )
        lines = [header]
        for span in sorted(record["spans"], key=lambda s: -s["peak_used_bytes"]):
            lines.append(
                f"{span['name']:<24}{span['calls']:>7}"
                f"{span['peak_used_bytes'] / _GB:>11.2f}{span['peak_total_bytes'] / _GB:>11.2f}"
                f"{span['max_rise_bytes'] / _GB:>10.2f}"
                f"{span['entry_used_bytes'] / _GB:>9.2f}{span['exit_used_bytes'] / _GB:>10.2f}"
                f"{span['entry_device_used_bytes'] / _GB:>8.2f}{span['exit_device_used_bytes'] / _GB:>9.2f}"
                f"{span['largest_alloc_bytes'] / _GB:>10.3f}"
            )
        alive = [entry for entry in self.lifetime if entry["alive"]]
        lines.append(
            f"buffer lifetime checks: {len(self.lifetime)}, still alive at next build: {len(alive)}"
            + (f" ({sorted({e['tag'] for e in alive})})" if alive else "")
        )
        logger.info("[GB_MEMPROBE %s] GB units, pool = CuPy default pool\n%s", self.move_name, "\n".join(lines))


def make_probe(xp, move_name: str, context: dict | None = None, settings=None):
    """Return a :class:`GBMemProbe` when enabled on a CuPy backend, else ``None``.

    ``settings`` is a :class:`GBDebugSettings`, read by attribute so the probe does
    not import the debug module.
    """
    if not getattr(settings, "mem_probe", False):
        return None
    if _HookBase is object or getattr(xp, "__name__", "") != "cupy":
        return None
    return GBMemProbe(xp, move_name, context, settings=settings)
