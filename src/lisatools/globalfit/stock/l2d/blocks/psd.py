from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field

from .base import (
    GENERIC_MOVES, GENERIC_PSD_MOVE_FILLED, SPECIAL_MOVES, SPECIAL_PSD_MOVE_FILLED,
    BlockBuild, BuildContext, ClassSpec, MoveBlock, PEMove, make_betas, make_search_interface,
)

REFERENCE_TMAX = 1e6  # top of the reference psd ladder (mojito_input/emri_mbh_psd_settings.py)
STOCHASTIC = ("psd", "galfor")  # the branches the psd move samples (one ladder, one search)


class PSDBlock(MoveBlock):
    """Block for PSD estimation."""

    type: Literal["psd"]
    SEARCH_INTERFACE: ClassVar[str] = "NoiseSearchInterface"
    SEARCH_TYPES: ClassVar[tuple[str, ...]] = ("prior", "samples", "point_estimate")  # no catalogue of noise
    FULL_LADDER: ClassVar[bool] = True  # the psd moves run on every temperature
    PE_MOVES: ClassVar[dict[str, PEMove]] = {
        "MultiGPUPSDMove": PEMove(SPECIAL_MOVES, SPECIAL_PSD_MOVE_FILLED),
        "PSDMoveBuilder": PEMove(GENERIC_MOVES, GENERIC_PSD_MOVE_FILLED),
    }

    # builds both the search (max_logl_mode) and the PE move
    pe_move: ClassSpec = Field(default_factory=lambda: ClassSpec.model_validate({
        "class": "MultiGPUPSDMove",
        "kwargs": dict(permute_every=100, live_dangerously=True, run_async=True, run_threaded=True),
    }))


def psd_betas(block: PSDBlock, ntemps: int):
    """The ladder of the psd move, shared by psd and galfor: the block's tempering, else the reference
    ``make_ladder(ndim, ntemps, Tmax=1e6)``. With ``ntemps`` and a finite ``Tmax`` that is the geometric
    ladder from 1 to 1/Tmax whatever ``ndim``, so galfor does not change it."""
    from eryn.moves.tempering import make_ladder

    if block.tempering is not None:
        return make_betas(block.tempering, ntemps)
    return make_ladder(1, ntemps=ntemps, Tmax=REFERENCE_TMAX)


def build(block: PSDBlock, ctx: BuildContext) -> BlockBuild:
    from lisatools.globalfit.postprocessing import StochasticMetadata
    from lisatools.globalfit.stock.erebor import PSDSettings, PSDSetup

    g = ctx.general

    info = block.parse_prior_information(g)

    prior_model = info["prior_model"]
    priors = info["priors"]
    transform_fn = info["transform_fn"]
    ndim = info["ndim"]
    injection = info["injection"]

    settings = PSDSettings(
        Tobs=g.Tobs, dt=g.dt, initialize_kwargs={}, log_dir=g.artifacts_file_dir,
        priors=priors, ndim=ndim, transform=transform_fn,
        injection=injection,
        num_prop_repeats=block.num_prop_repeats,
        # one ladder for seeding and both psd moves (the Setup's own default is 24 rungs)
        betas=psd_betas(block, ctx.cfg.sampler.ntemps),
        search_interface=make_search_interface(block, info),
    )

    setup = PSDSetup(settings)

    metadata = StochasticMetadata(
        model_config=dict(use_splines=False, num_params=ndim),
        frequency_ranges=[(g.start_freq, g.end_freq)],
        prior_model=prior_model,
        prior_model_code_link=block.links.prior_model_code_link,
        prior_model_config_file=block.links.prior_model_config_file,
    )
    return BlockBuild(setup, metadata, None)
