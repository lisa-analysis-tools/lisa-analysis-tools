"""The run's recipe: build the blocks' moves and add the steps.

``make_setup_function`` returns the engine's ``setup_function`` (called by ``run_global_fit`` with
``(recipe, engine_info, curr, acs, priors, state)``). By then the engine has seeded the state through the
blocks' search interfaces (``load_info``, fresh starts only) and built the residuals from it through each
seeded branch's ``Settings.signal_gen`` (``setup_acs``, ``GlobalFitSettings.rebuild_residuals``). The library
sets the per-leaf ladders, attaches the generators and starts the flow trainers; then a stock recipe, or the
run's own recipe file, adds the steps. Heavy imports stay inside functions: ``io.py`` imports this module to validate ``recipe:``.
"""
from __future__ import annotations

import importlib.util
import inspect
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .blocks.base import GENERIC_MOVES, MoveBlock, SourceBlock
from .blocks.psd import STOCHASTIC
from .blocks.utility import check_kwargs, find_class
from .generators import SignalGen

from ...loginfo import init_logger

if TYPE_CHECKING:
    from .blocks.base import BlockBuild
    from .io import RunConfig

SETUP_ARGS = ("recipe", "engine_info", "curr", "acs", "priors", "state")

logger = init_logger(__name__, level="DEBUG", name="RecipeBuilder")
class BranchMoves(NamedTuple):
    search: list  # one-shot moves run before PE (the psd max-likelihood search)
    pe: list


@dataclass
class RecipeContext:
    """What a recipe needs beyond the engine's arguments; a recipe file gets it as ``l2d``."""

    cfg: RunConfig
    wave_gens: dict[str, Any]  # seeded source branch -> waveform generator (shared with its PE move)
    # seeded source branch -> its Settings.signal_gen (analysis domain; setup_acs built the residuals with it)
    signal_gens: dict[str, Any] = field(default_factory=dict)
    # seeded source branch -> ``fn(*sampling_params, td_settings=None) -> TDSignal`` (unwindowed TDI)
    td_signal_gens: dict[str, Any] = field(default_factory=dict)
    td_ac: Any = None  # the shared CPU time-domain AnalysisContainer (data.keep_td), walker 0's residual

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
            logger.info("Adding psd search")
            recipe.add_recipe_component(SearchRecipeStep(moves=search), name="psd search")
        combined = GFCombineMove(moves=pe, share_temperature_control=False)
        if self.burn_in_iterations is not None:
            recipe.add_recipe_component(
                IterationCountRecipeStep(moves=[combined], num_iters=self.burn_in_iterations), name="burn-in"
            )
        logger.info("Adding joint PE")
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


# --- branches and generators (the state is seeded by load_info, the residuals built by setup_acs) --- #

def enabled_blocks(cfg: RunConfig) -> dict[str, Any]:
    return {name: block for name, block in cfg.blocks.items() if block.enabled}


def psd_needs_search(block) -> bool:
    """The max-likelihood psd search runs unless psd (and galfor) are seeded from samples or a point estimate."""
    return block.search is None or block.search.search_type == "prior"


def prepare_branches(cfg: RunConfig, curr, state) -> dict[str, Any]:
    """Set the source branches' per-leaf ladders; return the waveforms of the seeded source branches.

    The state is seeded by ``load_info`` (the blocks' search interfaces) and the residuals are built from it by
    ``setup_acs`` through each seeded branch's ``Settings.signal_gen``. The waveform returned here is that
    generator's (built by then), so the PE move, the residual rebuild and the TD generator share one object.
    """
    from lisatools.globalfit.searches import pad_ladder

    ntemps = cfg.sampler.ntemps
    wave_gens = {}
    for name, block in enabled_blocks(cfg).items():
        if not isinstance(block, SourceBlock):
            continue
        info = curr.source_info[name]
        state.sub_states[name].betas_all = np.tile(pad_ladder(info.betas, ntemps), (info.nleaves_max, 1))
        if info.signal_gen is not None:  # seeded (build_source registers it for blocks with a search)
            wave_gens[name] = info.signal_gen.waveform
    return wave_gens


def attach_signal_gens(ctx: RecipeContext, td_generators: dict, curr, state) -> None:
    """The seeded branches' analysis-domain and TD generators, and the shared TD residual (``data.keep_td``).

    The analysis-domain generator is the branch's ``Settings.signal_gen`` (the one ``setup_acs`` built the
    residuals with); the TD generator reuses its waveform (a block's ``td_waveform.class`` gets its own).
    """
    for name, wave_gen in ctx.wave_gens.items():
        info = curr.source_info[name]
        ctx.signal_gens[name] = info.signal_gen
        td = td_generators.get(name)
        if td is None:
            continue
        td_wave = wave_gen if td.waveform_class is None else td.waveform_class(**td.init_kwargs)
        ctx.td_signal_gens[name] = SignalGen(td_wave, td.method, info.transform, td.runtime_kwargs)

    if not ctx.cfg.data.keep_td:
        return
    if not ctx.td_signal_gens:
        logger.warning("data.keep_td is set, but no seeded block has a TD generator: no TD residual.")
        return
    ctx.td_ac = setup_td_ac(curr.general_info, ctx.td_signal_gens)
    subtract_td_signals(ctx.td_ac, ctx.td_signal_gens, state)


def setup_td_ac(general, td_signal_gens: dict):
    """One CPU ``AnalysisContainer`` holding the preprocessed, unwindowed TD data (no sensitivity)."""
    from lisatools.analysiscontainer import AnalysisContainer
    from lisatools.domains import TDSettings, TDSignal

    grid = general.data_td_settings
    arr = general.data_processor.td_signal.arr
    arr = arr.get() if hasattr(arr, "get") else np.array(arr, copy=True)
    if arr.shape[-1] != grid.N:
        raise ValueError(f"TD data has {arr.shape[-1]} samples, the data grid {grid.N}.")
    settings = TDSettings(grid.N, grid.dt, t0=grid.t0, force_backend="cpu")
    return AnalysisContainer(TDSignal(arr, settings), signal_gen=dict(td_signal_gens))


def subtract_td_signals(td_ac, td_signal_gens: dict, state) -> None:
    """Subtract walker 0's seeded leaves (cold chain) from the TD residual, one leaf at a time."""
    for name, gen in td_signal_gens.items():
        inds = state.branches_inds[name][0, 0]
        for leaf in np.flatnonzero(inds):
            td_ac.data.subtract_signal(gen(*state.branches_coords[name][0, 0, leaf]))
        logger.info(f"Subtracted {int(inds.sum())} {name} signals from the TD residual")


def attach_flow_trainers(cfg: RunConfig, curr) -> None:
    """Start the trainer of each seeded branch's flow move (``blocks.base.make_flow_move`` keeps its arguments).

    Here and not at build time: a live ``ProcessExecutor`` (queues + a spawned worker) does not survive
    ``CurrentInfoGlobalFit``'s deepcopy, and only the process that samples should spawn trainers.
    """
    from eryn.moves import CombineMove

    for name, block in enabled_blocks(cfg).items():
        if getattr(block, "flow", None) is None or block.search is None:  # unseeded: no moves, no trainer
            continue
        for move, _weight in curr.source_info[name].inner_moves:
            for sub in move.sub_moves if isinstance(move, CombineMove) else [move]:
                if hasattr(sub, "executor_init_kwargs") and sub.executor is None:
                    from eryn.flows import ProcessExecutor

                    sub.executor = ProcessExecutor(sub.flow, **sub.executor_init_kwargs)


# --- moves --- #

def on_gpu(general) -> bool:
    return general.force_backend == general.gpu_backend


def is_generic(block) -> bool:
    """The block's pe_move is a ``recipe.py`` builder (generic family), not a special move."""
    return block.PE_MOVES[block.pe_move.class_name].module == GENERIC_MOVES


def build_block_moves(ctx: RecipeContext, engine_info, curr, acs, priors, state) -> dict[str, BranchMoves]:
    """Each block's moves from its ``pe_move``; source branches without a search get none, galfor none (the
    psd move samples it)."""
    blocks = enabled_blocks(ctx.cfg)
    moves = {}
    for name, block in blocks.items():
        if not isinstance(block, MoveBlock):
            continue
        # checked at load (check_pe_move); here it is only looked up
        cls = find_class(block.pe_move.class_name, (block.PE_MOVES[block.pe_move.class_name].module,))
        if isinstance(block, SourceBlock):
            if block.search is not None:
                moves[name] = source_moves(block, cls, ctx, engine_info, curr, acs, priors, state)
        else:
            search = any(psd_needs_search(b) for n, b in blocks.items() if n in STOCHASTIC)
            moves[name] = psd_moves(block, cls, engine_info, curr, acs, priors, state, search=search)
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
    logger.debug(f"{name} PE move kwargs: {move.__dict__}")
    logger.info(f"Built {name} PE move {move} with {len(move.moves)} inner moves")
    move.accepted = np.zeros((g.ntemps, g.nwalkers))
    return BranchMoves([], [move])


def psd_moves(block, cls, engine_info, curr, acs, priors, state, search: bool = True) -> BranchMoves:
    """The psd search and PE moves from one ``pe_move`` (only ``max_logl_mode`` differs).

    They sample galfor too when it is a branch (the generic builder finds it in ``curr.source_info`` itself).
    """
    from eryn.moves.tempering import TemperatureControl

    info = curr.source_info[block.type]
    g = curr.general_info
    betas = np.asarray(info.betas, dtype=float)
    kwargs = deepcopy(block.pe_move.kwargs)

    if is_generic(block):
        search_moves, pe = cls(num_repeats=info.num_prop_repeats, betas=betas, **kwargs).build(
            engine_info, curr, acs, priors, state
        )
    else:
        galfor = curr.source_info.get("galfor")
        if galfor is not None:
            logger.info(f"psd move will sample galfor with and {galfor.ndim} parameters")
        common = dict(
            num_repeats=info.num_prop_repeats,
            psd_transform_fn=info.transform,
            galfor_transform_fn=None if galfor is None else galfor.transform,
            temperature_control=TemperatureControl(
                engine_info.ndims[block.type], g.nwalkers, ntemps=len(betas), betas=betas, permute=False
            ),
            use_gpu=on_gpu(g),
            **kwargs,
        )
        search_moves = [cls(acs, priors, max_logl_mode=True, name="psd search move", **common)]
        pe = [cls(acs, priors, max_logl_mode=False, name="psd pe move", **common)]
        for move in search_moves + pe:
            move.accepted = np.zeros((g.ntemps, g.nwalkers))
    return BranchMoves(search_moves if search else [], pe)


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
    td_generators = {
        name: build.td_generator for name, build in builds.items() if getattr(build, "td_generator", None)
    }

    def setup_recipe(recipe, engine_info, curr, acs, priors, state):
        # the state was seeded by load_info and the residuals built from it by setup_acs
        select_device(curr.general_info)
        ctx = RecipeContext(cfg, prepare_branches(cfg, curr, state))
        attach_signal_gens(ctx, td_generators, curr, state)
        attach_flow_trainers(cfg, curr)
        spec = cfg.recipe
        if hasattr(spec, "file"):
            fn = load_recipe_function(spec.file, spec.function)
            extra = {"l2d": ctx} if "l2d" in inspect.signature(fn).parameters else {}
            return fn(recipe, engine_info, curr, acs, priors, state, **extra)
        make_stock_recipe(spec.class_name, spec.kwargs).add_steps(
            recipe, ctx.block_moves(engine_info, curr, acs, priors, state)
        )

    return setup_recipe
