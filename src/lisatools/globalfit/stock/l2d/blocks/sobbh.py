"""Stellar-origin BBH block: the gwtf time-frequency waveform (``GWTF_generator``) and ``SOBBHSpecialMove``.

Differs from the MBH/EMRI blocks in three places:

* ``GWTF_generator`` is built from the analysis STFT grid itself, so the block fills its own constructor
  arguments (:meth:`SOBBHBlock.engine_waveform_kwargs`); the separate TD generator
  (``SOBBHTaylorT3TDIWaveform``) still gets the shared engine arguments.
* The waveforms are referenced to the data start (gwtf's ``f0`` is the frequency at ``settings.t0``), and
  ``SOBHBSearchInterface`` evolves the catalogue seeds there, so a catalog search runs with
  ``time_reference: data_start``.
* The special move sets its own waveform methods and there is no generic builder (gwtf has no likelihood
  ``__call__``).
"""
from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field, model_validator

from .base import (
    SPECIAL_MOVES, SPECIAL_SOURCE_MOVE_FILLED, BuildContext, ClassSpec, PEMove,
    PriorRef, SourceBlock, StockPrior, TDWaveformSpec, WaveformSpec,
)


class SOBBHBlock(SourceBlock):
    """Block for stellar-origin binary black hole inference with the gwtf waveform."""

    type: Literal["sobbh"]
    CATALOGUE_KEY: ClassVar[str] = "SOBHB"
    SOURCE_IDS_KEY: ClassVar[str] = "sobhb"
    WAVEFORM_MODULES: ClassVar[tuple[str, ...]] = (
        "lisatools.sources.sobbh.waveform_gwtf", "lisatools.sources.sobbh.waveform",
    )
    SOURCE_TYPE: ClassVar[str] = "SOBHB"
    SETTINGS_NAME: ClassVar[str] = "SOBBHTFSettings"
    SETUP_NAME: ClassVar[str] = "SOBBHTFSetup"
    SEARCH_INTERFACE: ClassVar[str] = "SOBHBSearchInterface"
    # seeds come from the catalogue's own TimeReferenceSSBFrame, evolved to the data start
    NEEDS_REFERENCE_TIME: ClassVar[bool] = False
    TD_WAVEFORM_METHOD: ClassVar[str | None] = "get_td_signals"
    # GWTF_generator constructor arguments the block fills (engine_waveform_kwargs)
    ENGINE_WAVEFORM_KWARGS: ClassVar[frozenset[str]] = frozenset({"settings", "orbits", "T_obs", "use_GPU", "tukey_alpha"})
    PE_MOVES: ClassVar[dict[str, PEMove]] = {
        "SOBBHSpecialMove": PEMove(SPECIAL_MOVES, SPECIAL_SOURCE_MOVE_FILLED),
    }

    priors: PriorRef = Field(default_factory=lambda: StockPrior.model_validate({"class": "mojitolight.SOBHBPrior"}))
    waveform: WaveformSpec = Field(default_factory=lambda: WaveformSpec.model_validate({
        "class": "GWTF_generator", "kwargs": dict(fresnel_kernel_width=16),
    }))
    # the injection/residual TD generator: same parameters, phi_coal at coalescence, referenced to the data start
    td_waveform: TDWaveformSpec | None = Field(default_factory=lambda: TDWaveformSpec.model_validate({
        "class": "SOBBHTaylorT3TDIWaveform", "kwargs": dict(phase_at_coalescence=True, order=25, buffer_time=10000),
    }))
    time_reference: Literal["data_start"] = "data_start"
    pe_move: ClassSpec = Field(default_factory=lambda: ClassSpec.model_validate({
        "class": "SOBBHSpecialMove",
        "kwargs": dict(
            permute_every=0, pad_out_of_prior=True, run_async=True, run_threaded=True, randomize_split=True,
            tc_min_days=60,
        ),
    }))

    @model_validator(mode="after")
    def _catalog_is_referenced_to_mojito(self):
        # replaces the base check: SOBHBSearchInterface evolves the catalogue seeds to the data start itself
        return self

    @model_validator(mode="after")
    def _proposal_priors_follow_the_source_ids(self):
        if self.proposal_priors is not None and self.proposal_priors.kwargs.get("sort_with_chirp_mass"):
            raise ValueError("proposal_priors: sort_with_chirp_mass orders the priors by chirp mass, not by source id.")
        return self

    @model_validator(mode="after")
    def _no_runtime_kwargs(self):
        if self.waveform.runtime_kwargs:
            raise ValueError("waveform.runtime_kwargs: GWTF_generator.get_signals_for_residuals takes none.")
        return self

    def engine_waveform_kwargs(self, ctx: BuildContext) -> dict[str, Any]:
        """``GWTF_generator``'s builder-filled arguments: the analysis STFT grid, orbits, observation time, the
        device, and the Tukey kernel matching the data window (box-car without a Tukey window)."""
        g = ctx.general
        tukey = g.window_type == "tukey" and g.window_alpha > 0
        return dict(
            settings=g.domain_settings,
            orbits=g.gpu_orbits if g.force_backend == g.gpu_backend else g.orbits,  # as engine_waveform_kwargs
            T_obs=g.Tobs,
            use_GPU=g.force_backend != "cpu",
            tukey_alpha=float(g.window_alpha) if tukey else None,
        )
