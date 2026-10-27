from __future__ import annotations

from typing import ClassVar, Literal

from .base import BlockBase, BlockBuild, BuildContext, make_betas, make_search_interface


class PSDBlock(BlockBase):
    """Block for PSD estimation."""

    type: Literal["psd"]
    SEARCH_INTERFACE: ClassVar[str] = "NoiseSearchInterface"
    SEARCH_TYPES: ClassVar[tuple[str, ...]] = ("prior", "samples", "point_estimate")  # no catalogue of noise


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
        betas=make_betas(block.tempering, ctx.cfg.sampler.ntemps),
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
