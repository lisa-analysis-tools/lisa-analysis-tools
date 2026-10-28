from __future__ import annotations

from typing import ClassVar, Literal

from .base import BlockBase, BlockBuild, BuildContext, make_search_interface
from .psd import psd_betas


class GalForBlock(BlockBase):
    """Block for the galactic foreground.

    It has no move of its own: the psd block's move samples psd and galfor together, on the psd ladder
    and with the psd repeats. The sensitivity needs ``galactic_grid_kwargs`` (checked at load).
    """

    type: Literal["galfor"]
    SEARCH_INTERFACE: ClassVar[str] = "GalacticForegroundSearchInterface"
    SEARCH_TYPES: ClassVar[tuple[str, ...]] = ("prior", "samples", "point_estimate")  # no catalogue of the galaxy


def build(block: GalForBlock, ctx: BuildContext) -> BlockBuild:
    from lisatools.globalfit.postprocessing import StochasticMetadata
    from lisatools.globalfit.stock.erebor import GalForSettings, GalForSetup

    g = ctx.general
    info = block.parse_prior_information(g)

    settings = GalForSettings(
        Tobs=g.Tobs, dt=g.dt, initialize_kwargs={}, log_dir=g.artifacts_file_dir,
        priors=info["priors"], ndim=info["ndim"], transform=info["transform_fn"],
        betas=psd_betas(ctx.cfg.blocks["psd"], ctx.cfg.sampler.ntemps),  # seeded on the ladder of the move that samples it
        search_interface=make_search_interface(block, info),
    )
    setup = GalForSetup(settings)
    # GalForSettings has no injection field; the diagnostic plots read the Setup's attribute if present
    setup.injection = info["injection"]

    metadata = StochasticMetadata(
        model_config=dict(
            num_params=info["ndim"],
            galactic_grid_kwargs=ctx.cfg.sensitivity.kwargs.get("galactic_grid_kwargs", {}),
        ),
        frequency_ranges=[(g.start_freq, g.end_freq)],
        prior_model=info["prior_model"],
        prior_model_code_link=block.links.prior_model_code_link,
        prior_model_config_file=block.links.prior_model_config_file,
    )
    return BlockBuild(setup, metadata, None)
