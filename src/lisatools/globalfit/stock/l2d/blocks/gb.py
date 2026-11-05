"""GB block: the galactic binaries, GBs and VGBs together, in one reversible-jump source branch (STFT domain).

It is set up like ``mbh.yaml``. ``pe_move`` names a GB special move and its knobs; the recipe fills the
engine-owned arguments (``recipe.prepare_gb_moves``). The in-model proposals belong to that move: its
``stretch_probability`` sets how often a proposal is a stretch instead of a Fisher-matrix one, so the block
takes no ``inner_moves`` and no ``flow``. ``waveform.runtime_kwargs`` and ``stft`` hold the knobs of
``GBWaveformSettings`` and ``GBSTFTSettings``; the builder fills their time grid, window, orbits and domain
from the ``GeneralSetup``.
"""
from __future__ import annotations

import dataclasses
import importlib
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, ClassVar, Literal

from pydantic import Field, PositiveInt, model_validator

from .base import (
    SPECIAL_MOVES, BlockBuild, BuildContext, ClassSpec, PEMove, SourceBlock, TemperingSpec, WaveformSpec,
    make_betas, make_search_interface,
)
from .utility import check_kwargs, find_class

# GBSpecialBase arguments the recipe fills (recipe.prepare_gb_moves and build_recipe.gb_moves)
GB_SPECIAL_MOVE_FILLED = frozenset({
    "gb", "priors", "start_freq_ind", "data_length", "acs", "band_edges", "band_N_vals", "gpu_priors",
    "waveform_kwargs", "parameter_transforms", "provide_betas", "skip_supp_names_update", "random_seed",
    "force_backend", "nfriends", "temperature_control", "gb_wdm_comp", "gb_fd_comp", "gb_stft_comp", "orbits",
    "tdi_config", "t_ref", "ntemps", "stft_store_windows", "rj_proposal_distribution", "num_repeat_proposals",
    "name",
})
# GBWaveformSettings and GBSTFTSettings fields the builder fills from the GeneralSetup
GB_WAVEFORM_FILLED = frozenset({"dt", "T", "start_freq_ind", "window", "window_alpha", "orbits", "t0", "force_backend"})
GB_STFT_FILLED = frozenset({"domain_settings", "band_alpha"})
# pe_move kwargs the YAML gives as mappings, and the dataclass each one becomes
GB_MOVE_DATACLASSES = {
    "debug_settings": ("lisatools.globalfit.moves.gbdebug", "GBDebugSettings"),
    "leaf_cap_settings": ("lisatools.globalfit.moves.gbspecialmove", "GBLeafCapSettings"),
    "sighet_settings": ("lisatools.globalfit.moves.gbspecialmove", "GBSigHetSettings"),
}


class GBBlock(SourceBlock):
    """Block for the galactic binaries: GBs and VGBs share the ``gb`` branch, which may hold no source."""

    type: Literal["gb"]
    CATALOGUE_KEY: ClassVar[str] = "GB"
    SOURCE_IDS_KEY: ClassVar[str] = "gb"
    WAVEFORM_MODULES: ClassVar[tuple[str, ...]] = ("gbgpu.gbgpu",)
    SOURCE_TYPE: ClassVar[str] = "gb"
    SETTINGS_NAME: ClassVar[str] = "GBSettings"
    SETUP_NAME: ClassVar[str] = "GBSetup"
    SEARCH_INTERFACE: ClassVar[str] = "GBSearchInterface"
    SEARCH_TYPES: ClassVar[tuple[str, ...]] = ("catalog", "point_estimate")
    FULL_LADDER: ClassVar[bool] = True  # the band temperatures span every temperature
    # ? The STFT computations take the parameters at the mojito catalogue epoch, not at data.reference_time.
    NEEDS_REFERENCE_TIME: ClassVar[bool] = False
    PE_MOVES: ClassVar[dict[str, PEMove]] = {
        "GBSpecialRJPriorMove": PEMove(SPECIAL_MOVES, GB_SPECIAL_MOVE_FILLED),
    }

    nleaves: PositiveInt = 8000  # the most sources at once (nleaves_max); the least is 0
    num_prop_repeats: PositiveInt = 50  # the move's num_repeat_proposals
    waveform: WaveformSpec = Field(default_factory=lambda: WaveformSpec.model_validate({"class": "GBGPU"}))
    stft: dict[str, Any] = Field(default_factory=dict)  # GBSTFTSettings knobs
    # * The reference ladder: ratio 1.2, with the hottest rung at T = 1e4.
    tempering: TemperingSpec = Field(default_factory=lambda: TemperingSpec(Tmax=1e4))
    # * The rj_prior move of recipe.build_gb_moves; stretch_probability 0: every in-model proposal is a
    # * Fisher-matrix one.
    pe_move: ClassSpec = Field(default_factory=lambda: ClassSpec.model_validate({
        "class": "GBSpecialRJPriorMove",
        "kwargs": dict(
            stretch_probability=0.0, num_stationary_fim_steps=1, run_swaps=True, use_prior_removal=False,
            phase_maximize=False, ranks_needed=0,
        ),
    }))

    @model_validator(mode="after")
    def _proposals_belong_to_the_pe_move(self):
        if self.inner_moves is not None or self.flow is not None:
            raise ValueError(
                "the gb block takes no inner_moves and no flow: its pe_move's stretch_probability sets the "
                "in-model proposals."
            )
        return self

    @model_validator(mode="after")
    def _groups_fit_the_settings(self):
        if self.waveform.kwargs:
            raise ValueError(
                "the gb block fills GBGPU's constructor arguments (orbits, t0, force_backend); give the waveform "
                "knobs in waveform.runtime_kwargs."
            )
        erebor = importlib.import_module("lisatools.globalfit.stock.erebor")
        check_kwargs(self.waveform.runtime_kwargs, erebor.GBWaveformSettings, GB_WAVEFORM_FILLED)
        check_kwargs(self.stft, erebor.GBSTFTSettings, GB_STFT_FILLED)
        for name, (module, class_name) in GB_MOVE_DATACLASSES.items():
            value = self.pe_move.kwargs.get(name)
            if value is None:
                continue
            if not isinstance(value, dict):
                raise ValueError(f"pe_move kwargs {name} must be a mapping of {class_name} fields.")
            check_kwargs(value, find_class(class_name, (module,)), frozenset())
        return self

    @model_validator(mode="after")
    def _catalog_reads_a_file(self):
        if self.search is not None and self.search.search_type == "catalog" and self.search.search_path is None:
            raise ValueError(
                "the gb catalog search needs search_path, e.g. gb_candidates.hdf5: the loaded catalogue holds "
                "the GB population as one entry of arrays, which the interface cannot unpack per source."
            )
        return self

    def build_source(self, ctx: BuildContext) -> BlockBuild:
        """The GB setup, from the grouped ``GBSettings`` (``GBSettings.from_general_setup``)."""
        from lisatools.globalfit.stock.erebor import (
            GBMoveSettings, GBSettings, GBSetup, GBSTFTSettings, GBWaveformSettings,
        )

        general_setup = ctx.general
        info = self.parse_prior_information(general_setup)
        # * The setup records the move knobs it has fields for; the recipe builds the move from pe_move itself.
        move_fields = {settings_field.name for settings_field in dataclasses.fields(GBMoveSettings)}
        moves = GBMoveSettings(
            num_repeat_proposals=self.num_prop_repeats,
            **{name: value for name, value in gb_move_kwargs(self).items() if name in move_fields},
        )
        search_interface = make_search_interface(self, info)
        settings = GBSettings.from_general_setup(
            general_setup,
            priors=info["priors"],
            nleaves_max=self.nleaves,
            nleaves_min=0,
            betas=make_betas(self.tempering, ctx.cfg.sampler.ntemps),
            waveform=GBWaveformSettings(**deepcopy(self.waveform.runtime_kwargs)),
            stft=GBSTFTSettings(**deepcopy(self.stft)),
            moves=moves,
            search_interface=search_interface,
        )
        setup = GBSetup(settings)
        if search_interface is not None and search_interface.search_type == "catalog":
            # * As SourceBlock.build_source: the plots' truths, set here so a restart (which does not seed) keeps
            # * them. GBSettings has no injection field; the plots read the setup's attribute if present.
            setup.injection = search_interface.catalog_truths(SimpleNamespace(general_info=general_setup))
        metadata = setup.make_source_metadata(
            waveform_model_code_link=self.links.waveform_model_code_link,
            prior_model_code_link=self.links.prior_model_code_link,
            prior_model_config_file=self.links.prior_model_config_file,
        )
        return BlockBuild(setup, metadata, find_class(self.waveform.class_name, self.WAVEFORM_MODULES))


def gb_move_kwargs(block: GBBlock) -> dict[str, Any]:
    """The pe_move kwargs, with their mappings made into the move's settings dataclasses."""
    kwargs = deepcopy(block.pe_move.kwargs)
    for name, (module, class_name) in GB_MOVE_DATACLASSES.items():
        if isinstance(kwargs.get(name), dict):
            kwargs[name] = find_class(class_name, (module,))(**kwargs[name])
    return kwargs
