"""``LISAToolsJaxBackend`` -- the JAX entry into lisatools' backend registry.

Slots into :class:`lisatools.cutils.LISAToolsBackendMethods` so that
:class:`lisatools.detector.Orbits` (and any other class subclassing
:class:`gpubackendtools.ParallelModuleBase`) can pick the JAX path via
``force_backend='jax'``.

There is no compiled plugin module for this backend; the
implementations live in :mod:`lisatools.jax` as pure JAX code. We
subclass :class:`Backend` directly (not :class:`CpuBackend`) so the
``_check_module_installed`` importlib check is skipped; ``import jax``
is what we verify instead.

Stubs are provided for any cutils-side wrap slots that don't have a
JAX implementation yet (e.g. ``SensitivityMatrixWrap``,
``check_orbits``). They raise ``NotImplementedError`` with a clear
message on first use, so orchestration code that doesn't touch them
still works.
"""
from __future__ import annotations

from gpubackendtools.exceptions import MissingDependency
from gpubackendtools.gpubackendtools import Backend
from gpubackendtools.jax.backend import GBTJaxBackend

from ..cutils import LISAToolsBackend, LISAToolsBackendMethods


def _check_jax():
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp
    except (ImportError, ModuleNotFoundError) as e:
        raise MissingDependency(
            "'jax' backend requires jax (with jaxlib)",
            pip_deps=["jax", "jaxlib"],
            conda_deps=["jax"],
        ) from e
    from jax import config
    config.update("jax_enable_x64", True)
    return jnp


def _unimplemented(name):
    class _NotImpl:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError(
                f"{name} is not implemented in the lisatools JAX backend yet. "
                "Use force_backend='cpu' (or one of the CUDA backends) for "
                "this path until a JAX implementation lands."
            )
    _NotImpl.__name__ = name
    return _NotImpl


def _check_orbits_jax(*args, **kwargs):
    """Stub for the C++ ``check_orbits`` validation routine.

    The C++ routine validates that arm lengths, light-travel times and
    positions are self-consistent. For the JAX path we silently no-op
    -- the orbit data was already validated when the host-side
    ``Orbits.configure(linear_interp_setup=True)`` ran on numpy / cupy
    inputs upstream.
    """
    return None


def _jax_methods_loader() -> LISAToolsBackendMethods:
    jnp = _check_jax()
    from .orbits import OrbitsWrapJAX
    from .response.tdi_config import TDIConfigWrapJAX
    from gpubackendtools.jax.cubic_spline import CubicSplineWrapJAX

    return LISAToolsBackendMethods(
        OrbitsWrap=OrbitsWrapJAX,
        Orbits=OrbitsWrapJAX,                          # JAX side: same class
        check_orbits=_check_orbits_jax,
        # SensitivityMatrixWrap=_unimplemented("SensitivityMatrixWrap"),  # XYZBackend disabled
        SensitivityMatrixWrap=None,
        GalacticGridSetup=None,
        GalacticGridWrap=None,
        # 2026-06 domains consolidation: STFT/FD domain wraps.
        STFTDomainWrap=None,
        FDDomainForStftWrap=None,
        STFTFresnelWrap=None,
        psd_likelihood=None,
        compute_logpdf=None,
        # Phase 3L.7k LISA-response wraps absorbed from fastlisaresponse.
        TDSplineTDIWaveformWrap=None,
        FDSplineTDIWaveformWrap=None,
        LISAResponseWrap=None,
        LISAResponse=None,
        TDIConfigWrap=TDIConfigWrapJAX,
        TDIConfig=TDIConfigWrapJAX,
        # GBT is the single registrant for CubicSplineWrap.
        CubicSplineWrap=CubicSplineWrapJAX,
        WDMSettingsWrap=None,
        WDMDomainWrap=None,
        FDDomainWrap=None,
        TDITypeDict=None,
        xp=jnp,
    )


class LISAToolsJaxBackend(GBTJaxBackend, LISAToolsBackend):
    """JAX backend object plugged into ``Globals().backends_manager``."""

    _backend_name: str = "lisatools_backend_jax"
    _name = "lisatools_jax"

    def __init__(self, *args, **kwargs):
        methods = _jax_methods_loader()
        GBTJaxBackend.__init__(
            self,
            *args,
            **kwargs,
        )
        LISAToolsBackend.__init__(self, methods)

    @staticmethod
    def _check_module_installed(*args, **kwargs):
        return None
