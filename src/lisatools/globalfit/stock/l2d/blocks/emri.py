"""EMRI block: everything is in ``SourceBlock``; only the class variables differ from the MBH block."""
from __future__ import annotations

from typing import ClassVar, Literal

from .base import SourceBlock


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
