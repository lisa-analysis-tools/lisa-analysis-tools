"""EMRI block: everything is in ``SourceBlock``; only the class variables differ from the MBH block."""
from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from .base import (
    GENERIC_MOVES, GENERIC_SOURCE_MOVE_FILLED, RESIDUAL_MOVE, SPECIAL_MOVES, SPECIAL_SOURCE_MOVE_FILLED,
    ClassSpec, PEMove, SourceBlock,
)


class EMRIBlock(SourceBlock):
    """Block for extreme mass-ratio inspiral inference."""

    type: Literal["emri"]
    CATALOGUE_KEY: ClassVar[str] = "EMRI"
    SOURCE_IDS_KEY: ClassVar[str] = "emri"
    WAVEFORM_MODULES: ClassVar[tuple[str, ...]] = ("lisatools.sources.emri.waveform",)
    SOURCE_TYPE: ClassVar[str] = "EMRI"
    SETTINGS_NAME: ClassVar[str] = "EMRISettings"
    SETUP_NAME: ClassVar[str] = "EMRISetup"
    SEARCH_INTERFACE: ClassVar[str] = "EMRISearchInterface"
    WAVEFORM_METHODS: ClassVar[dict[str, str]] = dict(
        waveform_gen_method="get_signals_for_residuals", waveform_like_method="__call__"
    )
    LIKE_KWARGS_FROM_WAVEFORM_KWARGS: ClassVar[bool] = True
    PE_MOVES: ClassVar[dict[str, PEMove]] = {
        "EMRISpecialMove": PEMove(SPECIAL_MOVES, SPECIAL_SOURCE_MOVE_FILLED),
        "EMRIMoveBuilder": PEMove(GENERIC_MOVES, GENERIC_SOURCE_MOVE_FILLED, forwards_to=RESIDUAL_MOVE),
    }

    pe_move: ClassSpec = Field(default_factory=lambda: ClassSpec.model_validate({
        "class": "EMRISpecialMove",
        "kwargs": dict(
            permute_every=25, pad_out_of_prior=True, run_async=True, run_threaded=True, randomize_split=True,
            batch_size_per_gpu=5,
        ),
    }))
