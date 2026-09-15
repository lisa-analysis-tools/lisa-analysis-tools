"""Per-device replicas of the objects that pin buffers at construction.

Multi-GPU discipline in LAT is "enter the owning shard's device context, then
launch" (see :mod:`lisatools.utils.device`). That is necessary but not
sufficient: a kernel launched under shard 1's context still reads whatever
pointers the *object* it was called on holds, and most of our configuration
objects allocate their device buffers **once**, on whatever CUDA device was
current at construction, and never migrate them.

So a routed launch on a non-primary shard dereferences the primary device's
pointers -- a silent peer-access tax where P2P is enabled, an illegal access
where it is not. The fix is one cached replica per ``(prototype, device)``,
rebuilt from the prototype's own recorded constructor arguments inside the
target device context.

Four object families need it, in dependency order:

``Orbits``
    grid arrays consumed by every response evaluation.
``DomainSettingsBase``
    the WDM analysis ``window`` / ``omega`` a ``.transform()`` multiplies by.
``TDIConfig``
    the six link/sign/channel tables ``__init__`` uploads; a comp's
    ``cpp_tdi_config = backend.TDIConfigWrap(*tdi_config.pytdiconfig_args)``
    captures those pointers verbatim, so replicating a comp around a SHARED
    TDIConfig reproduces the bug one level down.
``GB computation objects``
    ``GBWDMComputations`` / ``GBFDComputations`` / ``STFTGBComputations`` and
    the ``GBSignalHetComputations`` wrapper -- chunk geometry, WDM window,
    ``OrbitsWrap`` / ``TDIConfigWrap`` pointer fields, and for sig-het the
    whole heterodyne reference stash.

**Invariants.** The CPU path (``device is None``) and the prototype's own
build device always get the SHARED object back, so single-GPU and CPU
behaviour is byte-identical to the pre-multi-GPU path and allocates nothing.
Replicas are built once per ``(prototype, device)`` and kept for the whole run
(allocate-once, persist) -- they deliberately do NOT follow a proposal's
lifetime, and they never reach the settings tree (pickle-safety rule).

**Cache keys** hold only ``id(prototype)``, so every cache VALUE keeps a
strong reference to the prototype: a garbage-collected prototype could
otherwise let a recycled id alias a stale replica.

.. note::
   On the ``dev`` branch these helpers live in
   ``globalfit/stock/erebor/source_runtime.py``, which arrives with B8's
   ``stock/erebor`` package split. They are placed here instead because they
   are generic device infrastructure with no stock-recipe content, and
   because ``globalfit/moves/gbbands.py`` consumes them -- importing a
   ``stock`` module from a ``moves`` module inverts the layering (``stock``
   builds moves, not the other way round) and forces a lazy-import dance to
   dodge the cycle. When this branch lands on dev, ``source_runtime`` can re-export from
   here.
"""

from __future__ import annotations

from .device import current_device, device_context

__all__ = [
    "device_local_orbits",
    "device_local_domain_settings",
    "device_local_domain_settings_on",
    "device_local_tdi_config",
    "device_local_gb_comp",
]


# ---------------------------------------------------------------------------
# Orbits
# ---------------------------------------------------------------------------

_DEVICE_ORBITS_REPLICAS: dict = {}


def device_local_orbits(orbits, xp, primary_device):
    """An orbits replica resident on the cupy *current* device.

    CPU / single-GPU / the run's primary device reuse the shared ``orbits``
    (zero extra memory -- identical to the pre-multi-GPU path). A non-primary
    device gets a lazily-built, cached ``orbits.__class__(*args, **kwargs)``
    replica whose grid arrays land on the current device (the
    :class:`~lisatools.domaincomputation.DomainComputationGroupArray`
    ``build_cpp_objects`` pattern), so anything built around it reads orbit
    data locally instead of via peer access off the primary device.
    """
    dev = current_device(xp)
    if dev is None or primary_device is None or dev == int(primary_device):
        return orbits
    if orbits is None:
        return orbits
    key = (id(orbits), dev)
    hit = _DEVICE_ORBITS_REPLICAS.get(key)
    if hit is not None:
        return hit[1]
    with device_context(xp, dev):
        replica = orbits.__class__(*orbits.args, **orbits.kwargs)
    _DEVICE_ORBITS_REPLICAS[key] = (orbits, replica)
    return replica


# ---------------------------------------------------------------------------
# Domain settings
# ---------------------------------------------------------------------------

# The wave wraps project raw TD channels onto the run's domain settings via
# ``.transform()``, which multiplies by ``settings.window`` (the WDM analysis
# window). That window lives on the primary device, so a shard-1 walker's
# device-local template * primary-device window trips peer access on EVERY
# generation (``domains.py``: ``before_ifft[:] *= base_window``).
_DEVICE_DOMAIN_REPLICAS: dict = {}


def device_local_domain_settings(settings, xp, primary_device):
    """A domain-settings replica whose device arrays live on the current device.

    CPU / single-GPU / the primary device reuse the shared ``settings``
    (byte-identical to the pre-multi-GPU path). A non-primary device rebuilds
    ``settings.__class__(*args, **kwargs)`` with the device-resident WDM
    ``window`` / ``omega`` DROPPED, so ``__init__`` regenerates them on THIS
    device via ``setup_window()`` -- deterministic in ``(Nf, Nt, dt,
    oversample)``, hence numerically identical to the primary-device window
    (lnL parity preserved), just with no peer access off the primary device.
    Domain types without those keys (FD / STFT / TD) rebuild harmlessly from
    their scalar args.
    """
    dev = current_device(xp)
    if dev is None or primary_device is None or dev == int(primary_device):
        return settings
    if not (hasattr(settings, "args") and hasattr(settings, "kwargs")):
        return settings  # unknown settings type -> leave shared
    key = (id(settings), dev)
    hit = _DEVICE_DOMAIN_REPLICAS.get(key)
    if hit is not None:
        return hit[1]
    kw = dict(settings.kwargs)
    # Regenerate the WDM window/omega on the target device (no-op key pops
    # for non-WDM domain types).
    kw.pop("window", None)
    kw.pop("omega", None)
    with device_context(xp, dev):
        replica = settings.__class__(*settings.args, **kw)
    _DEVICE_DOMAIN_REPLICAS[key] = (settings, replica)
    return replica


def device_local_domain_settings_on(settings, xp, device, primary_device):
    """:func:`device_local_domain_settings` with an EXPLICIT target device.

    Callers already sitting inside their shard's ``device_context`` can use
    the current-device form; the GB replica builder resolves a device it was
    handed, so it needs the explicit spelling. Same cache, same "primary
    reuses the shared object" rule.
    """
    if device is None or primary_device is None or int(device) == int(primary_device):
        return settings
    with device_context(xp, int(device)):
        return device_local_domain_settings(settings, xp, primary_device)


# ---------------------------------------------------------------------------
# TDI config
# ---------------------------------------------------------------------------

_DEVICE_TDI_CONFIG_REPLICAS: dict = {}


def device_local_tdi_config(tdi_config, xp, device, primary_device):
    """A TDI-config replica whose link tables live on ``device``.

    CPU / the comp's own build device reuse the shared object (byte-identical
    to the single-GPU path, zero extra memory). Anything else gets one cached
    replica built from ``tdi_config.tdi_combinations`` -- the very list the
    constructor consumes -- inside the target device context, so the six
    ``pytdiconfig_args`` tables (and therefore the ``TDIConfigWrap`` pointer
    fields a comp builds from them) are device-local.
    """
    if tdi_config is None or device is None or primary_device is None:
        return tdi_config
    if int(device) == int(primary_device):
        return tdi_config
    if not hasattr(tdi_config, "tdi_combinations"):
        return tdi_config  # a string / unknown spec -> the comp rebuilds it
    key = (id(tdi_config), int(device))
    hit = _DEVICE_TDI_CONFIG_REPLICAS.get(key)
    if hit is not None:
        return hit[1]
    with device_context(xp, int(device)):
        replica = type(tdi_config)(
            tdi_config.tdi_combinations, force_backend=tdi_config.backend
        )
    _DEVICE_TDI_CONFIG_REPLICAS[key] = (tdi_config, replica)
    return replica


# ---------------------------------------------------------------------------
# GB computation objects
# ---------------------------------------------------------------------------

_DEVICE_GB_COMP_REPLICAS: dict = {}

#: ``GBSignalHetComputations.for_band_engine`` knob name -> the key it is
#: recorded under in the instance's ``_g`` dict. Deriving the replica's knobs
#: from ``_g`` (rather than a second stash) keeps the sig-het class free of
#: replica-only state; the recorded values are already RESOLVED (snapped
#: ``nt_layer``, resolved ``n_cp_build``), and both resolutions are
#: idempotent, so a rebuild reproduces the prototype exactly.
#:
#: Only the knobs this branch's ``for_band_engine`` actually accepts are
#: listed. ``dev`` additionally carries ``n_cp_build`` / ``v3_n_nodes`` /
#: ``v4_knots`` / ``v4_band`` / ``v5``; our signature is
#: ``(nt_layer, n_sparse_fd, m_active_half_width, max_r)``, and the builder
#: below intersects with the live signature anyway, so a later merge that
#: widens ``for_band_engine`` needs no change here.
_SIGHET_REPLICA_KNOBS = {
    "nt_layer": "nt_layer",
    "n_sparse_fd": "n_sparse_fd",
    "m_active_half_width": "m_half",
    "max_r": "max_r",
    "n_cp_build": "n_cp_build",
    "v3_n_nodes": "v3_n_nodes",
    "v4_knots": "v4_knots",
    "v4_band": "v4_band",
    "v5": "v5",
}


def release_gb_comp_groups(comp, parent_acs) -> None:
    """Point ``comp`` and its replicas back at ``parent_acs``'s groups.

    The band engine binds a buffer's group onto the shared comp on every call and never
    clears it, so a finished buffer's arrays stay reachable through the comp. Callers release
    when a buffer is finished with; the next engine call rebinds to the next buffer.
 """
    if comp is None or parent_acs is None or not hasattr(comp, "stft_comps"):
        return
    splits = parent_acs.cpp_splits
    if not splits:
        return
    gpus = list(parent_acs.gpus) if getattr(parent_acs, "gpus", None) else []
    comp.stft_comps = splits[0]
    for (_comp_id, device_id), (prototype, replica) in _DEVICE_GB_COMP_REPLICAS.items():
        if prototype is not comp or not hasattr(replica, "stft_comps"):
            continue
        split_index = gpus.index(device_id) if device_id in gpus else 0
        replica.stft_comps = splits[split_index]


def device_local_gb_comp(comp, xp, device, primary_device):
    """A GB comp replica resident on ``device``.

    ``primary_device`` is the device the prototype's own buffers live on
    (``comp._build_device`` where the router can read it, else ``gpus[0]``):
    that device -- and the CPU path (``device is None``) -- reuses the shared
    object, so single-GPU behaviour is byte-identical and allocates nothing.
    Any other device gets ONE lazily-built, cached replica.

    Three comp shapes are handled, all by duck-typing so this module keeps no
    import dependency on gbgpu:

    * the sig-het wrapper (``GBSignalHetComputations``) -- built through
      ``for_band_engine(chunked_comp, **knobs)``, never ``__class__(*args)``,
      so it is rebuilt the same way around a device-local delegate with the
      knobs recovered from the instance's own ``_g`` grid/knob dict;
    * comps that record their constructor -- ``WDMComputationsBase``
      subclasses (``GBWDMComputations``), ``GBFDComputations`` and
      ``STFTGBComputations`` -- rebuilt from ``comp.args`` / ``comp.kwargs``
      with the device-resident domain-settings / ``orbits`` / ``tdi_config``
      arguments swapped for their own device-local replicas;
    * anything else -- returned shared, unchanged (no silent half-fix).
    """
    if comp is None or device is None or primary_device is None:
        return comp
    if int(device) == int(primary_device):
        return comp
    key = (id(comp), int(device))
    hit = _DEVICE_GB_COMP_REPLICAS.get(key)
    if hit is not None:
        return hit[1]
    with device_context(xp, int(device)):
        replica = _build_gb_comp_replica(comp, xp, int(device), int(primary_device))
    if replica is comp:
        return comp
    _DEVICE_GB_COMP_REPLICAS[key] = (comp, replica)
    return replica


def _sighet_knobs(comp, g):
    """Knob kwargs for rebuilding a sig-het wrapper, intersected with its factory.

    Guards against a ``_SIGHET_REPLICA_KNOBS`` entry that the *installed*
    ``for_band_engine`` does not accept (the dev/branch signature skew noted
    on the table above): passing one would ``TypeError`` mid-proposal on a
    non-primary shard only -- the worst possible place to find out.
    """
    import inspect

    try:
        accepted = set(
            inspect.signature(type(comp).for_band_engine).parameters
        )
    except (TypeError, ValueError):
        accepted = None
    return {
        name: g[gkey]
        for name, gkey in _SIGHET_REPLICA_KNOBS.items()
        if gkey in g and (accepted is None or name in accepted)
    }


def _build_gb_comp_replica(comp, xp, device, primary_device):
    """Construct one GB comp replica (called inside ``device``'s context)."""
    # --- sig-het wrapper: rebuilt through its own factory -----------------
    chunked = getattr(comp, "chunked", None)
    if chunked is not None and hasattr(type(comp), "for_band_engine"):
        g = getattr(comp, "_g", None)
        if g is None:
            return comp
        return type(comp).for_band_engine(
            device_local_gb_comp(chunked, xp, device, primary_device),
            **_sighet_knobs(comp, g),
        )

    # --- comps that record their constructor ------------------------------
    if not (hasattr(comp, "args") and hasattr(comp, "kwargs")):
        return comp
    args = list(comp.args)
    kw = dict(comp.kwargs)

    # The first positional is the domain settings for the WDM/FD comps
    # (WDMSettings / FDSettings) -- WDMSettings carries a device-resident
    # analysis ``window``, and the domain replica helper regenerates it on
    # THIS device from the same deterministic (Nf, Nt, dt, oversample), so
    # values are unchanged. It is NOT domain settings for STFTGBComputations,
    # whose first positional is an STFTComputationGroup: the isinstance test
    # skips it, and the engine rebinds ``stft_comps`` to the owning split's
    # own group on every call anyway.
    if args:
        from ..domains import DomainSettingsBase

        if isinstance(args[0], DomainSettingsBase):
            args[0] = device_local_domain_settings_on(
                args[0], xp, device, primary_device
            )

    # ``orbits=None`` is fine: the comp's setter builds EqualArmlengthOrbits
    # and configures it HERE, i.e. on this device.
    if kw.get("orbits") is not None:
        kw["orbits"] = device_local_orbits(kw["orbits"], xp, primary_device)
    if kw.get("tdi_config") is not None:
        kw["tdi_config"] = device_local_tdi_config(
            kw["tdi_config"], xp, device, primary_device
        )
    return type(comp)(*args, **kw)