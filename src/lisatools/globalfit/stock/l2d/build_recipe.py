"""The run's recipe: build the blocks' moves and add the steps.

``make_setup_function`` returns the engine's ``setup_function`` (called by ``run_global_fit`` with
``(recipe, engine_info, curr, acs, priors, state)``). By then the engine has seeded the state through the
blocks' search interfaces (``load_info``, fresh starts only) and built the residuals from it through each
seeded branch's ``Settings.signal_gen`` (``setup_acs``, ``GlobalFitSettings.rebuild_residuals``). The library
sets the per-leaf ladders, attaches the generators, sets up the GB band machinery (which subtracts the GB start
sources itself) and starts the flow trainers; then a stock recipe, or the run's own recipe file, adds the
steps. Heavy imports stay inside functions: ``io.py`` imports this module to validate ``recipe:``.
"""
from __future__ import annotations

import importlib.util
import inspect
from copy import deepcopy
from functools import partial
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import numpy as np

from .blocks.base import GENERIC_MOVES, MoveBlock, SourceBlock, make_proposal_priors
from .blocks.gb import GBBlock, gb_move_kwargs
from .blocks.utility import check_kwargs, find_class
from .generators import SignalGen

from ...loginfo import init_logger

if TYPE_CHECKING:
    from .blocks.base import BlockBuild
    from .io import RunConfig

SETUP_ARGS = ("recipe", "engine_info", "curr", "acs", "priors", "state")

logger = init_logger(__name__, level="DEBUG", name="RecipeBuilder")

class Alternative(NamedTuple):
    """One of a branch's PE moves that take turns, with its shares of the PE and the burn-in iterations."""

    move: Any
    weight: float
    burn_in_weight: float


class BranchMoves(NamedTuple):
    search: list  # one-shot moves run before PE (the psd max-likelihood search)
    pe: list
    # PE moves that take (random) turns as the branch's PE move, one per iteration; None: pe runs every iteration
    alternatives: list[Alternative] | None = None


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
    # stage name -> {branch: apply_settings kwargs} from the blocks' ``stages`` (a recipe step's move_settings)
    move_settings: Callable[[str], dict[str, dict]] | None = None

    def block_moves(self, engine_info, curr, acs, priors, state) -> dict[str, BranchMoves]:
        return build_block_moves(self, engine_info, curr, acs, priors, state)


# --- the recipes --- #

def _positive_int(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 1


class StockRecipe:
    """psd max-likelihood search (when there is a psd block), then the joint PE in ``stages``.

    ``stages`` is a list of ``{name, iterations}``: each stage runs ``iterations`` iterations, the last one has
    none and runs until ``run.num_iterations``. All stages share one joint PE move; what changes between them
    are the blocks' ``stages`` settings (proposal repeats, inner-move weights). Default: one stage, ``pe``.
    ``burn_in_iterations: N`` is short for ``[{name: burn-in, iterations: N}, {name: pe}]``.
    """

    RESERVED = ("psd search",)  # step names the recipe uses itself

    def __init__(self, stages: list[dict[str, Any]] | None = None, burn_in_iterations: int | None = None):
        if burn_in_iterations is not None:
            if not _positive_int(burn_in_iterations):
                raise ValueError(f"burn_in_iterations must be a positive integer, not {burn_in_iterations!r}.")
            if stages is not None:
                raise ValueError("give stages or burn_in_iterations, not both (burn-in is a stage).")
            stages = [{"name": "burn-in", "iterations": burn_in_iterations}, {"name": "pe"}]
        self.stages = self._checked(stages if stages is not None else [{"name": "pe"}])

    @classmethod
    def _checked(cls, stages) -> list[dict[str, Any]]:
        if not isinstance(stages, list) or not stages:
            raise ValueError(f"stages must be a non-empty list of {{name, iterations}}, not {stages!r}.")
        out = []
        for i, stage in enumerate(stages):
            if not isinstance(stage, dict) or set(stage) - {"name", "iterations"} or "name" not in stage:
                raise ValueError(f"stage {stage!r} must be {{name, iterations}} (iterations left out on the last).")
            name, iterations = stage["name"], stage.get("iterations")
            if not isinstance(name, str) or not name or name in cls.RESERVED:
                raise ValueError(f"stage name {name!r} must be a non-empty string other than {list(cls.RESERVED)}.")
            last = i == len(stages) - 1
            if last and iterations is not None:
                raise ValueError(f"the last stage ({name!r}) runs until run.num_iterations; leave out its iterations.")
            if not last and not _positive_int(iterations):
                raise ValueError(f"stage {name!r} needs iterations (a positive integer), not {iterations!r}.")
            out.append({"name": name, "iterations": iterations})
        names = [stage["name"] for stage in out]
        if len(set(names)) != len(names):
            raise ValueError(f"stage names must be unique: {names}.")
        return out

    @property
    def stage_names(self) -> list[str]:
        return [stage["name"] for stage in self.stages]

    def add_steps(self, recipe, moves: dict[str, BranchMoves], move_settings=None) -> None:
        """Add the steps; ``move_settings(stage name) -> {branch: settings}`` gives each stage's move settings."""
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
        #from here
        alternating = [name for name, branch in moves.items() if branch.alternatives]
        if not alternating:
            combined = GFCombineMove(moves=pe, share_temperature_control=False)
            if self.burn_in_iterations is not None:
                recipe.add_recipe_component(
                    IterationCountRecipeStep(moves=[combined], num_iters=self.burn_in_iterations), name="burn-in"
                )
            logger.info("Adding joint PE")
            recipe.add_recipe_component(PERecipeStep(moves=[combined]), name="pe")
            return

        # * A GFCombineMove runs every move it holds, so a branch whose PE moves take turns gets one combined move per
        # * alternative, each with every other branch's PE moves in the usual order. Each step picks one per iteration
        # * with the alternatives' shares of its phase; an alternative without a share in a phase is left out of it.
        #! Coen: "this should be updated in the future, now keeping same behaviour as previously."
        if len(alternating) > 1:
            raise ValueError(f"only one block may alternate its PE moves, not {alternating}.")
        name = alternating[0]
        alternatives = moves[name].alternatives
        combined = []
        for alternative in alternatives:
            listed = [m for other, branch in moves.items() for m in ([alternative.move] if other == name else branch.pe)]
            combined.append(GFCombineMove(moves=listed, share_temperature_control=False))

        def phase(share: str) -> tuple[list, list]:
            kept = [index for index, alternative in enumerate(alternatives) if getattr(alternative, share) > 0.0]
            return [combined[index] for index in kept], [getattr(alternatives[index], share) for index in kept]

        if self.burn_in_iterations is not None:
            burn_in_moves, burn_in_weights = phase("burn_in_weight")
            logger.info(f"Adding burn-in, the {name} PE moves taking turns with weights {burn_in_weights}")
            recipe.add_recipe_component(
                IterationCountRecipeStep(
                    moves=burn_in_moves, weights=burn_in_weights, num_iters=self.burn_in_iterations
                ),
                name="burn-in",
            )
        elif any(alternative.weight == 0.0 for alternative in alternatives):
            raise ValueError(
                f"a {name} PE move has a share of the burn-in only, but the recipe has no burn-in step "
                "(burn_in_iterations)."
            )
        pe_moves, pe_weights = phase("weight")
        logger.info(f"Adding joint PE, the {name} PE moves taking turns with weights {pe_weights}")
        recipe.add_recipe_component(PERecipeStep(moves=pe_moves, weights=pe_weights), name="pe")
        #to here
        combined = GFCombineMove(moves=pe, share_temperature_control=False)
        for stage in self.stages:
            name, iterations = stage["name"], stage["iterations"]
            settings = move_settings(name) if move_settings is not None else None
            if iterations is None:
                step = PERecipeStep(moves=[combined], move_settings=settings)
            else:
                step = IterationCountRecipeStep(moves=[combined], num_iters=iterations, move_settings=settings)
            logger.info(f"Adding joint PE stage {name!r}" + ("" if iterations is None else f" ({iterations} iterations)"))
            recipe.add_recipe_component(step, name=name)
         # to here


STOCK_RECIPES: dict[str, type] = {"StockRecipe": StockRecipe}


def stage_move_settings(cfg: RunConfig, curr, stage: str) -> dict[str, dict]:
    """``{branch: apply_settings kwargs}`` for one stage, for every block with ``stages`` and a PE move.

    The weights follow the PE move's inner moves (``Settings.inner_moves``: the YAML ``inner_moves``, or the
    Setup's default, then the flow move); a stage's left-out keys take the block's own values.
    """
    out = {}
    for name, block in enabled_blocks(cfg).items():
        if not getattr(block, "stages", None) or block.search is None:  # no stages, or no moves (unseeded)
            continue
        stage_spec = block.stages.get(stage)
        info = curr.source_info[name]
        weights = [w for _move, w in info.inner_moves]
        repeats = block.num_prop_repeats
        if stage_spec is not None:
            if stage_spec.inner_move_weights is not None:
                weights[: len(stage_spec.inner_move_weights)] = stage_spec.inner_move_weights
            if stage_spec.flow_weight is not None:
                weights[-1] = stage_spec.flow_weight  # the flow move is appended last (blocks.base.build_source)
            repeats = stage_spec.num_prop_repeats or repeats
        out[name] = {"num_repeats": repeats, "weights": weights}
    return out


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


def prepare_gb_branches(
    cfg: RunConfig, waveform_classes: dict[str, type], engine_info, curr, acs, priors, state
) -> dict[str, Any]:
    """The band machinery of each seeded GB branch (``recipe.prepare_gb_moves``), which ``gb_moves`` runs on.

    GBGPU has no per-source ``Settings.signal_gen``, so ``setup_acs`` leaves the GB start sources in the
    residuals: ``prepare_gb_moves`` subtracts them with the band likelihood engine and sets the band temperatures.
    """
    from lisatools.globalfit.recipe import prepare_gb_moves

    gb_parts = {}
    for name, block in enabled_blocks(cfg).items():
        if isinstance(block, GBBlock) and block.search is not None:  # unseeded: no moves, no machinery
            gb_parts[name] = prepare_gb_moves(
                engine_info, curr, acs, priors, state, waveform_class=waveform_classes[name]
            )
    return gb_parts


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
                # * The GB move runs on the band machinery prepare_gb_branches set up, not on a waveform generator.
                make_moves = gb_moves if isinstance(block, GBBlock) else source_moves
                moves[name] = make_moves(block, cls, ctx, engine_info, curr, acs, priors, state)
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
    if block.proposal_priors is not None:
        # this branch's move proposes under per-leaf priors; the sampler and the other branches keep the shared dict
        priors = {**priors, name: make_proposal_priors(block, ctx.cfg, priors[name])}

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


def gb_moves(block, cls, ctx: RecipeContext, engine_info, curr, acs, priors, state) -> BranchMoves:
    """The GB PE move: ``block.pe_move``'s class on the band machinery ``prepare_gb_branches`` set up
    (``prepare_gb_moves``, which also subtracted the start sources and set the band temperatures)."""
    from lisatools.globalfit.moves.gbdebug import GBDebugSettings
    from lisatools.globalfit.recipe import default_gb_debug_settings

    parts = ctx.wave_gens[block.type]
    kwargs = gb_move_kwargs(block)
    # * As the settings files: the debug defaults, with plots in the run's artifacts.
    kwargs["debug_settings"] = default_gb_debug_settings(
        kwargs.get("debug_settings") or GBDebugSettings(), curr.general_info.artifacts_file_dir
    )
    move = cls(
        *parts.args,
        **parts.kwargs,
        rj_proposal_distribution=parts.gpu_priors,
        num_repeat_proposals=block.num_prop_repeats,
        # ! GBSpecialBase reads "fstat" and "refit" in its name; the prior RJ move needs neither.
        name=f"{block.type} pe move",
        **kwargs,
    )
    move.accepted = np.zeros((parts.ntemps, parts.nwalkers))
    logger.info(f"Built {block.type} PE move {type(move).__name__} with {block.num_prop_repeats} proposals")
    if not block.extra_moves:
        return BranchMoves([], [move])

    from lisatools.globalfit.recipe import default_fstat_settings

    extras = []
    for key, spec in block.extra_moves.items():
        entry = block.EXTRA_MOVES[spec.class_name]
        # * the pe_move's knobs (its in-model proposals among them) with the extra move's on top
        extra_kwargs = gb_move_kwargs(block, {**block.pe_move.kwargs, **spec.kwargs})
        extra_kwargs["debug_settings"] = default_gb_debug_settings(
            extra_kwargs.get("debug_settings") or GBDebugSettings(), curr.general_info.artifacts_file_dir
        )
        if spec.class_name == "GBFStatRJMove":
            # * as the settings files: the search's log and records in the run's artifacts
            extra_kwargs["fstat_settings"] = default_fstat_settings(
                extra_kwargs.get("fstat_settings"), curr.general_info.artifacts_file_dir
            )
        # * the same band machinery as the pe_move, so the moves share the GB ladder
        extra = find_class(spec.class_name, (entry.module,))(
            *parts.args,
            **parts.kwargs,
            rj_proposal_distribution=parts.gpu_priors if entry.prior_proposal else None,
            num_repeat_proposals=block.num_prop_repeats,
            name=f"{block.type} {key} move",
            **extra_kwargs,
        )
        extra.accepted = np.zeros((parts.ntemps, parts.nwalkers))
        extras.append(Alternative(extra, spec.weight, spec.burn_in_share))
        logger.info(
            f"Built {block.type} extra move {key} ({spec.class_name}), run in a share {spec.weight} of the PE "
            f"iterations and {spec.burn_in_share} of the burn-in"
        )
    # * pe_move takes what the extra moves leave of each phase, more than 0 by the block's check
    pe_share = 1.0 - sum(alternative.weight for alternative in extras)
    burn_in_share = 1.0 - sum(alternative.burn_in_weight for alternative in extras)
    return BranchMoves([], [move], [Alternative(move, pe_share, burn_in_share), *extras])


def psd_moves(block, cls, engine_info, curr, acs, priors, state) -> BranchMoves:
    """The psd max-likelihood search and PE moves from one ``pe_move`` (only ``max_logl_mode`` differs).

    The search runs however psd is seeded (prior, samples or a point estimate). The moves sample galfor too
    when it is a branch (the generic builder finds it in ``curr.source_info`` itself).
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
    return BranchMoves(search_moves, pe)


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
    td_generators = {
        name: build.td_generator for name, build in builds.items() if getattr(build, "td_generator", None)
    }

    def setup_recipe(recipe, engine_info, curr, acs, priors, state):
        # the state was seeded by load_info and the residuals built from it by setup_acs
        select_device(curr.general_info)
        ctx = RecipeContext(cfg, prepare_branches(cfg, curr, state), move_settings=partial(stage_move_settings, cfg, curr))
        attach_signal_gens(ctx, td_generators, curr, state)
        # * After attach_signal_gens: a GB entry holds the band machinery, not a waveform with a signal_gen.
        ctx.wave_gens.update(prepare_gb_branches(cfg, waveform_classes, engine_info, curr, acs, priors, state))
        attach_flow_trainers(cfg, curr)
        spec = cfg.recipe
        if hasattr(spec, "file"):
            fn = load_recipe_function(spec.file, spec.function)
            extra = {"l2d": ctx} if "l2d" in inspect.signature(fn).parameters else {}
            return fn(recipe, engine_info, curr, acs, priors, state, **extra)
        make_stock_recipe(spec.class_name, spec.kwargs).add_steps(
            recipe, ctx.block_moves(engine_info, curr, acs, priors, state), ctx.move_settings
        )

    return setup_recipe
