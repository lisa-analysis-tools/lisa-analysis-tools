"""The run's recipe: seed the state from the blocks, build their moves, add the steps.

``make_setup_function`` returns the engine's ``setup_function`` (called by ``run_global_fit`` with
``(recipe, engine_info, curr, acs, priors, state)``). The library always seeds first (source inds off,
search interfaces, initial-signal subtraction); then a stock recipe, or the run's own recipe file, adds
the steps. Heavy imports stay inside functions: ``io.py`` imports this module to validate ``recipe:``.
"""
from __future__ import annotations

import importlib.util
import inspect
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .blocks.base import GENERIC_MOVES, SourceBlock
from .blocks.utility import check_kwargs, find_class

if TYPE_CHECKING:
    from .blocks.base import BlockBuild
    from .io import RunConfig

SETUP_ARGS = ("recipe", "engine_info", "curr", "acs", "priors", "state")


class BranchMoves(NamedTuple):
    search: list  # one-shot moves run before PE (the psd max-likelihood search)
    pe: list


@dataclass
class RecipeContext:
    """What a recipe needs beyond the engine's arguments; a recipe file gets it as ``l2d``."""

    cfg: RunConfig
    wave_gens: dict[str, Any]  # seeded source branch -> waveform generator (shared with its PE move)

    def block_moves(self, engine_info, curr, acs, priors, state) -> dict[str, BranchMoves]:
        return build_block_moves(self, engine_info, curr, acs, priors, state)


# --- the recipes --- #

class StockRecipe:
    """psd search (when the seeding rules need it), optional burn-in, then joint PE until ``run.num_iterations``."""

    def __init__(self, burn_in_iterations: int | None = None):
        if burn_in_iterations is not None and (
            isinstance(burn_in_iterations, bool) or not isinstance(burn_in_iterations, int) or burn_in_iterations < 1
        ):
            raise ValueError(f"burn_in_iterations must be a positive integer, not {burn_in_iterations!r}.")
        self.burn_in_iterations = burn_in_iterations

    def add_steps(self, recipe, moves: dict[str, BranchMoves]) -> None:
        from lisatools.globalfit.moves import GFCombineMove
        from lisatools.globalfit.recipe import IterationCountRecipeStep, PERecipeStep, SearchRecipeStep

        search = [m for branch in moves.values() for m in branch.search]
        pe = [m for branch in moves.values() for m in branch.pe]
        if not pe:
            raise ValueError("nothing to sample: no block has a PE move (source blocks get one only with a search).")

        # step names are recorded by the backend; a restart skips the completed ones
        if search:
            recipe.add_recipe_component(SearchRecipeStep(moves=search), name="psd search")
        combined = GFCombineMove(moves=pe, share_temperature_control=False)
        if self.burn_in_iterations is not None:
            recipe.add_recipe_component(
                IterationCountRecipeStep(moves=[combined], num_iters=self.burn_in_iterations), name="burn-in"
            )
        recipe.add_recipe_component(PERecipeStep(moves=[combined]), name="pe")


STOCK_RECIPES: dict[str, type] = {"StockRecipe": StockRecipe}


def make_stock_recipe(name: str, kwargs: dict[str, Any]):
    cls = STOCK_RECIPES.get(name)
    if cls is None:
        raise ValueError(f"unknown recipe {name!r}; use one of {sorted(STOCK_RECIPES)}.")
    check_kwargs(kwargs, cls, frozenset())
    return cls(**deepcopy(kwargs))


def load_recipe_function(path: Path, name: str):
    """Import a recipe file and return its ``name`` function, checked against the engine's call."""
    if not path.is_file():
        raise ValueError(f"recipe file {path} does not exist.")
    spec = importlib.util.spec_from_file_location(f"l2d_recipe_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as err:  # pydantic only turns ValueError into a validation error
        raise ValueError(f"cannot import recipe file {path}: {err!r}") from err

    fn = getattr(module, name, None)
    if not callable(fn):
        raise ValueError(f"recipe file {path} defines no function {name!r}.")
    signature = inspect.signature(fn)
    try:
        signature.bind(*SETUP_ARGS, **({"l2d": None} if "l2d" in signature.parameters else {}))
    except TypeError as err:
        raise ValueError(f"{name} in {path} must accept ({', '.join(SETUP_ARGS)}): {err}.") from err
    return fn


# --- seeding --- #

def enabled_blocks(cfg: RunConfig) -> dict[str, Any]:
    return {name: block for name, block in cfg.blocks.items() if block.enabled}


def pad_ladder(betas, ntemps: int) -> np.ndarray:
    """A subset ladder padded with its last rung to ``ntemps`` (the state keeps every temperature)."""
    betas = np.asarray(betas, dtype=float)
    return np.concatenate([betas, np.full(ntemps - len(betas), betas[-1])])


def psd_needs_search(block) -> bool:
    """The max-likelihood psd search runs unless the psd is seeded from samples or a point estimate."""
    return block.search is None or block.search.search_type == "prior"


def seed_branches(cfg: RunConfig, waveform_classes: dict[str, type], curr, acs, priors, state) -> dict[str, Any]:
    """Seed the state from the blocks; return the waveform generators of the seeded source branches.

    Every source branch starts empty; a branch with a search interface is filled by it and its initial
    signals are subtracted from the residuals. The interfaces get the branch's own ladder, padded to the
    sampler's temperatures.
    """
    from lisatools.globalfit.recipe import subtract_initial_signal

    ntemps = cfg.sampler.ntemps
    blocks = enabled_blocks(cfg)
    sources = {name: block for name, block in blocks.items() if isinstance(block, SourceBlock)}

    for name in sources:
        state.branches_inds[name][:] = False
        info = curr.source_info[name]
        state.sub_states[name].betas_all = np.tile(pad_ladder(info.betas, ntemps), (info.nleaves_max, 1))

    for name, block in blocks.items():
        if block.search is not None:
            info = curr.source_info[name]
            info.search_interface.populate_state(curr, state, pad_ladder(info.betas, ntemps), priors=priors)

    wave_gens = {}
    for name, block in sources.items():
        if block.search is None:
            continue
        info = curr.source_info[name]
        wave_gens[name] = waveform_classes[name](**info.initialize_kwargs)
        subtract_initial_signal(acs, state, wave_gens[name].get_signals_for_residuals, name, info)
    return wave_gens


# --- moves --- #

def on_gpu(general) -> bool:
    return general.force_backend == general.gpu_backend


def is_generic(block) -> bool:
    """The block's pe_move is a ``recipe.py`` builder (generic family), not a special move."""
    return block.PE_MOVES[block.pe_move.class_name].module == GENERIC_MOVES


def build_block_moves(ctx: RecipeContext, engine_info, curr, acs, priors, state) -> dict[str, BranchMoves]:
    """Each block's moves from its ``pe_move``; source branches without a search get none."""
    moves = {}
    for name, block in enabled_blocks(ctx.cfg).items():
        # checked at load (check_pe_move); here it is only looked up
        cls = find_class(block.pe_move.class_name, (block.PE_MOVES[block.pe_move.class_name].module,))
        if isinstance(block, SourceBlock):
            if block.search is not None:
                moves[name] = source_moves(block, cls, ctx, engine_info, curr, acs, priors, state)
        else:
            moves[name] = psd_moves(block, cls, engine_info, curr, acs, priors, state)
    return moves


def source_moves(block, cls, ctx: RecipeContext, engine_info, curr, acs, priors, state) -> BranchMoves:
    name = block.type
    info = curr.source_info[name]
    g = curr.general_info
    betas = np.asarray(info.betas, dtype=float)
    wave_gen = ctx.wave_gens[name]
    kwargs = deepcopy(block.pe_move.kwargs)

    if is_generic(block):
        # the builder reads the rest from curr.source_info
        builder = cls(wave_gen=wave_gen.get_signals_for_residuals, betas=betas, move_name=f"{name} pe move", **kwargs)
        search, pe = builder.build(engine_info, curr, acs, priors, state)
        return BranchMoves(search, pe)

    like_kwargs = info.waveform_kwargs.copy() if block.LIKE_KWARGS_FROM_WAVEFORM_KWARGS else {}
    move = cls(
        dcga=acs,
        waveform_gen=wave_gen,
        branch_name=name,
        coords_shape=(len(betas), g.nwalkers, info.nleaves_max, info.ndim),  # may be a subset ladder
        waveform_gen_kwargs=info.waveform_kwargs.copy(),
        waveform_like_kwargs=like_kwargs,
        num_repeats=info.num_prop_repeats,
        transform_fn=info.transform,
        priors=priors,
        inner_moves=info.inner_moves,
        betas_all=np.tile(betas, (info.nleaves_max, 1)),
        **block.WAVEFORM_METHODS,
        **kwargs,
    )
    move.accepted = np.zeros((g.ntemps, g.nwalkers))
    return BranchMoves([], [move])


def psd_moves(block, cls, engine_info, curr, acs, priors, state) -> BranchMoves:
    """The psd search and PE moves from one ``pe_move`` (only ``max_logl_mode`` differs)."""
    from eryn.moves.tempering import TemperatureControl

    info = curr.source_info[block.type]
    g = curr.general_info
    betas = np.asarray(info.betas, dtype=float)
    kwargs = deepcopy(block.pe_move.kwargs)

    if is_generic(block):
        search, pe = cls(num_repeats=info.num_prop_repeats, betas=betas, **kwargs).build(
            engine_info, curr, acs, priors, state
        )
    else:
        common = dict(
            num_repeats=info.num_prop_repeats,
            psd_transform_fn=info.transform,
            temperature_control=TemperatureControl(
                engine_info.ndims[block.type], g.nwalkers, ntemps=len(betas), betas=betas, permute=False
            ),
            use_gpu=on_gpu(g),
            **kwargs,
        )
        search = [cls(acs, priors, max_logl_mode=True, name="psd search move", **common)]
        pe = [cls(acs, priors, max_logl_mode=False, name="psd pe move", **common)]
        for move in search + pe:
            move.accepted = np.zeros((g.ntemps, g.nwalkers))
    return BranchMoves(search if psd_needs_search(block) else [], pe)


# --- the engine's setup_function --- #

def select_device(general) -> None:
    if on_gpu(general):
        import cupy as cp

        cp.cuda.runtime.setDevice(general.gpus[0])


def make_setup_function(cfg: RunConfig, builds: dict[str, BlockBuild]):
    """``setup_recipe(recipe, engine_info, curr, acs, priors, state)`` for ``GlobalFitSettings.setup_function``.

    A closure, not an object: ``CurrentInfoGlobalFit`` deep-copies its settings, and functions are copied
    by reference.
    """
    waveform_classes = {name: build.waveform_class for name, build in builds.items()}

    def setup_recipe(recipe, engine_info, curr, acs, priors, state):
        select_device(curr.general_info)
        ctx = RecipeContext(cfg, seed_branches(cfg, waveform_classes, curr, acs, priors, state))
        spec = cfg.recipe
        if hasattr(spec, "file"):
            fn = load_recipe_function(spec.file, spec.function)
            extra = {"l2d": ctx} if "l2d" in inspect.signature(fn).parameters else {}
            return fn(recipe, engine_info, curr, acs, priors, state, **extra)
        make_stock_recipe(spec.class_name, spec.kwargs).add_steps(
            recipe, ctx.block_moves(engine_info, curr, acs, priors, state)
        )

    return setup_recipe
