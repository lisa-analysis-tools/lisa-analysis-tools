"""Pieces shared by every block type: schema parts, class lookup + kwarg checks, build helpers."""
from __future__ import annotations

import importlib
import inspect
from copy import deepcopy
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple, Annotated, TYPE_CHECKING, List
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt, model_validator, Discriminator, Tag, field_validator   

from .utility import find_class, accepted_kwargs, check_kwargs, prior_class
from ..typing import RunPath

if TYPE_CHECKING:
    from ereborpriors.base import GFBasePrior
    from lisatools.globalfit.engine import GeneralSetup

# Waveform constructor arguments the builder fills from GeneralSetup / data.reference_time.
ENGINE_WAVEFORM_KWARGS = frozenset({
    "waveform_t0", "data_td_settings", "sampling_frequency", "orbits", "tukey_alpha",
    "stft_dt", "freq_min", "freq_max", "force_backend",
})
MOVE_MODULES = ("eryn.moves", "lisatools.sampling.moves.skymodehop")
# filled by the block builder (e.g. MBHBlock.filled_move_kwargs); the YAML gives only `which` and the weight
ENGINE_MOVE_KWARGS = {"SkyMove": frozenset({"ind_map", "coord_frame", "to_lisa_frame", "from_lisa_frame"})}
ENGINE_PRIOR_KWARGS = frozenset({"observation_time",})  # filled from GeneralSetup

class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassSpec(_Spec):
    class_name: str = Field(alias="class")
    kwargs: dict[str, Any] = Field(default_factory=dict)


class WaveformSpec(ClassSpec):
    runtime_kwargs: dict[str, Any] = Field(default_factory=dict)  # -> Settings.waveform_kwargs


class MoveSpec(ClassSpec):
    weight: PositiveFloat = 1.0

class TemperingSpec(_Spec):
    ntemps: PositiveInt | None = None  # None -> sampler.ntemps
    ratio: float = Field(default=1.2, gt=1.0)  # geometric ladder: betas = ratio ** -arange(ntemps)

class SearchSpec(_Spec):
    """How a block's walkers are seeded: the arguments of the branch's search interface.

    The interface (``lisatools.globalfit.searches``) gets the prior's transform and periodicity from
    the builder and seeds through the inverse transform. A block without ``search`` has no
    interface: source branches then start empty, psd/galfor run the max-likelihood search.
    """

    search_type: Literal["prior", "distribution", "samples", "point_estimate", "catalog"] = "catalog"
    search_path: RunPath | None = None  # samples / point estimates: a file or a folder (one file per source)
    parameters_searched: list[str] | Literal["all"] = "all"  # names in the prior's input (sampling) basis
    # scalar, per-parameter (1-D), full covariance (2-D) or one covariance per leaf (3-D)
    spread: float | list[float] | list[list[float]] | list[list[list[float]]] | None = None
    is_eryn_backend: bool = False  # search results are an eryn backend, already in the sampling basis

    @model_validator(mode="after")
    def _same_rules_as_the_interface(self):
        if self.search_type not in ("catalog", "prior") and self.search_path is None:
            raise ValueError(f"search_type {self.search_type!r} needs search_path.")
        if self.search_type in ("catalog", "point_estimate") and self.spread is None:
            raise ValueError(f"search_type {self.search_type!r} needs spread.")
        if self.search_path is not None and not self.search_path.exists():
            raise ValueError(f"search_path {self.search_path} does not exist.")
        return self

class StockPrior(_Spec):                                                                                                                              
    """A GFBasePrior subclass from ereborpriors.stock and its kwargs."""                                                                              
    class_name: str = Field(alias="class")                                                                                                            
    kwargs: dict[str, Any] = Field(default_factory=dict) 
    output_file: RunPath | None = None  # where to write the prior's YAML; relative paths are relative to the run folder                                                                                             
                                                                                                                                                    
    @model_validator(mode="after")                                                                                                                    
    def _class_and_kwargs(self):
        cls = prior_class(self.class_name)
        factory = getattr(cls, "prior_factory", cls)                                                                                                                 
        check_kwargs(self.kwargs, factory, ENGINE_PRIOR_KWARGS)                                                                  
        return self                                                                                                                                   
                                                                                                                                                    
                                                                                                                                                    
class PriorFile(_Spec):                                                                                                                               
    """A prior file for ``GFBasePrior.from_yaml``; relative paths are relative to the run folder."""                                                  
    file: RunPath
    output_file: RunPath | None = None  # where to write the prior's YAML; relative paths are relative to the run folder                                                                                                                                
                                                                                                                                                    
    @field_validator("file")                                                                                                                          
    @classmethod                                                                                                                                      
    def _exists(cls, v):                                                                                                                              
        if not v.is_file():                                                                                                                           
            raise ValueError(f"prior file {v} does not exist.")                                                                                       
        return v                                                                                                                                                                                                                                                             
                                                                                                                                                    
def _prior_kind(v):                                                                                                                                   
    if isinstance(v, dict):                                                                                                                           
        return "file" if "file" in v else "class"                                                                                                     
    return "file" if isinstance(v, PriorFile) else "class"                                                                                            
                                                                                                                                                    
                                                                                                                                                    
PriorRef = Annotated[                                                                                                                                 
    Annotated[StockPrior, Tag("class")] | Annotated[PriorFile, Tag("file")],                                                                          
    Discriminator(_prior_kind),                                                                                                                       
]                                                                                                                                                     

class Links(_Spec):
    prior_model_code_link: str = ""
    prior_model_config_file: str = ""

class SourceLinks(Links):
    waveform_model_code_link: str = ""

class BlockBase(_Spec):
    """Base class for all block specs: schema, class lookup, kwarg checks, build helpers."""

    NEEDS_REFERENCE_TIME: ClassVar[bool] = False  # True -> data.reference_time is required
    SEARCH_INTERFACE: ClassVar[str]  # class in lisatools.globalfit.searches that seeds this branch
    # "distribution" is not implemented by the interface yet
    SEARCH_TYPES: ClassVar[tuple[str, ...]] = ("prior", "samples", "point_estimate", "catalog")

    type: str  # block type == file stem == branch name (source_info / priors / recipe key)
    priors: PriorRef
    enabled: bool = True
    tempering: TemperingSpec | None = None
    nleaves: PositiveInt = 1  # source blocks override: "auto" -> len(data.source_ids[SOURCE_IDS_KEY])
    num_prop_repeats: PositiveInt = 100  # how many times to repeat the proposal per walker per iteration
    inner_moves: list[MoveSpec] | None = None  # None -> the Setup's default (StretchMove)
    links: Links = Field(default_factory=Links)
    injection: dict[str, float] | None = None  # for diagnostic plots, keyed like the prior basis
    search: SearchSpec | None = None  # None -> no search interface (see SearchSpec)

    @model_validator(mode="after")
    def _search_type_is_available(self):
        if self.search is not None and self.search.search_type not in self.SEARCH_TYPES:
            raise ValueError(
                f"search_type {self.search.search_type!r} is not available for the {self.type!r} block; "
                f"use one of {list(self.SEARCH_TYPES)}."
            )
        return self

    def parse_prior_information(self, g: GeneralSetup) -> dict[str, Any]:
        """Return the prior, transform container, periodicity and basis for this block's priors."""
        priors_obj = make_priors(self, g)
        branch = priors_obj.branch_name
        if branch != self.type:
            raise ValueError(
                f"{type(priors_obj).__name__} is the prior of the {branch!r} branch, "
                f"but this is the {self.type!r} block."
            )
        prior_model = priors_obj.__class__.__name__
        priors = {branch: priors_obj}
        transform_fn = priors_obj.get_transforms()
        periodic = priors_obj.get_periodic()
        periodic = {branch: periodic} if periodic else None  # the engine keys periodicity by branch
        basis = priors_obj.get_input_basis()
        ndim = priors_obj.ndim

        if self.injection is not None:
            missing = set(basis) - set(self.injection)
            extra = set(self.injection) - set(basis)
            if missing or extra:
                raise ValueError(f"injection keys {sorted(self.injection)} do not match prior basis {sorted(basis)}.")
            injection = np.array([self.injection[n] for n in basis])
        else:
            injection = None

        return dict(
            prior_model=prior_model,
            priors=priors,
            transform_fn=transform_fn,
            periodic=periodic,
            basis=basis,
            ndim=ndim,
            injection=injection,
        )

def resolve_class(name: str, modules: tuple[str, ...]) -> type:
    """``find_class`` as a validation error: pydantic only turns ValueError into one, not ImportError."""
    try:
        return find_class(name, modules)
    except ImportError as err:
        raise ValueError(f"cannot import {list(modules)} to look up {name!r}: {err}") from err


class SourceBlock(BlockBase):
    """A source branch (mbh, emri, ...): leaves, waveform, sampler knobs, simple inner moves.

    A subclass declares the class variables and, if its inner moves need builder-filled
    kwargs, overrides :meth:`filled_move_kwargs`; the build itself is shared (:meth:`build_source`).
    """

    CATALOGUE_KEY: ClassVar[str]      # key of the data catalogue entries (e.g. "MBHB")
    SOURCE_IDS_KEY: ClassVar[str]     # key in data.source_ids / data.source_types (e.g. "mbhb")
    WAVEFORM_MODULES: ClassVar[tuple[str, ...]]
    SOURCE_TYPE: ClassVar[str]        # SourceMetadata.source_type
    SETTINGS_NAME: ClassVar[str]      # Settings / Setup classes in lisatools.globalfit.stock.erebor
    SETUP_NAME: ClassVar[str]
    NEEDS_REFERENCE_TIME: ClassVar[bool] = True

    nleaves: Literal["auto"] | PositiveInt = "auto"
    waveform: WaveformSpec
    links: SourceLinks = Field(default_factory=SourceLinks)  # adds waveform_model_code_link
    # Always a ladder (default: ratio 1.2 over sampler.ntemps, as the reference settings pass). Not None: the Setup's
    # own default is 24 rungs, which cannot seed a state with fewer temperatures.
    tempering: TemperingSpec = Field(default_factory=TemperingSpec)

    @model_validator(mode="after")
    def _classes_and_kwargs(self):
        cls = resolve_class(self.waveform.class_name, self.WAVEFORM_MODULES)
        check_kwargs(self.waveform.kwargs, cls, ENGINE_WAVEFORM_KWARGS)
        for move in self.inner_moves or []:
            cls = resolve_class(move.class_name, MOVE_MODULES)
            if cls.__name__ == "ConditionalFlowMove":
                raise ValueError("flow moves are not supported yet (they come with the recipe step).")
            check_kwargs(move.kwargs, cls, ENGINE_MOVE_KWARGS.get(cls.__name__, frozenset()))
        return self

    def resolve_nleaves(self, cfg) -> int:
        if self.nleaves != "auto":
            return self.nleaves
        return len(cfg.data.source_ids[self.SOURCE_IDS_KEY])

    def filled_move_kwargs(self, ctx: BuildContext, info: dict[str, Any]) -> dict[str, dict]:
        """Builder-filled kwargs of the inner moves, by move class name (e.g. ``SkyMove``)."""
        return {}

    def build_source(self, ctx: BuildContext) -> BlockBuild:
        """Build this branch's Setup and metadata (the waveform class is resolved, not constructed)."""
        erebor = importlib.import_module("lisatools.globalfit.stock.erebor")
        g = ctx.general

        info = self.parse_prior_information(g)
        nleaves = self.resolve_nleaves(ctx.cfg)
        init_kwargs = {**deepcopy(self.waveform.kwargs), **engine_waveform_kwargs(ctx)}

        settings = getattr(erebor, self.SETTINGS_NAME)(
            log_dir=g.artifacts_file_dir,
            Tobs=g.Tobs,
            dt=g.dt,
            initialize_kwargs=init_kwargs,
            transform=info["transform_fn"],
            periodic=info["periodic"],
            priors=info["priors"],
            waveform_kwargs=deepcopy(self.waveform.runtime_kwargs),
            nleaves_max=nleaves,
            nleaves_min=nleaves,
            ndim=info["ndim"],
            num_prop_repeats=self.num_prop_repeats,
            betas=make_betas(self.tempering, ctx.cfg.sampler.ntemps),
            inner_moves=make_inner_moves(self, self.filled_move_kwargs(ctx, info)),
            injection=info["injection"],
            search_interface=make_search_interface(self, info),
        )
        setup = getattr(erebor, self.SETUP_NAME)(settings)
        waveform_class = find_class(self.waveform.class_name, self.WAVEFORM_MODULES)
        return BlockBuild(setup, source_metadata(self, init_kwargs, g), waveform_class)

# --- build helpers (heavy imports stay inside functions) ---

class BuildContext(NamedTuple):
    cfg: Any      # RunConfig
    general: Any  # built GeneralSetup


class BlockBuild(NamedTuple):
    setup: Any                   # the branch Setup (PSDSetup, MBHSetup, ...)
    metadata: Any                # SourceMetadata / StochasticMetadata
    waveform_class: type | None  # constructed later by the recipe

def make_betas(tempering: TemperingSpec | None, default_ntemps: int):
    import numpy as np

    if tempering is None:
        return None
    ntemps = tempering.ntemps or default_ntemps
    return 1.0 / tempering.ratio ** np.arange(ntemps)

def make_search_interface(block: BlockBase, info: dict[str, Any]):
    """The block's search interface, or ``None`` without ``search``.

    The prior supplies what the interface needs to seed through the inverse transform: its transform and
    its (branch-keyed) periodicity.
    """
    spec = block.search
    if spec is None:
        return None

    parameters = None if spec.parameters_searched == "all" else list(spec.parameters_searched)
    unknown = sorted(set(parameters or []) - set(info["basis"]))
    if unknown:
        raise ValueError(f"parameters_searched {unknown} are not in the prior basis {info['basis']}.")

    searches = importlib.import_module("lisatools.globalfit.searches")
    return getattr(searches, block.SEARCH_INTERFACE)(
        search_type=spec.search_type,
        transform=info["transform_fn"],
        parameters_searched=parameters,
        search_path=None if spec.search_path is None else str(spec.search_path),
        spread=None if spec.spread is None else np.asarray(spec.spread, dtype=float),
        is_eryn_backend=spec.is_eryn_backend,
        periodic=info["periodic"],
    )

def engine_waveform_kwargs(ctx: BuildContext) -> dict[str, Any]:
    """The waveform constructor arguments the engine owns (see ``ENGINE_WAVEFORM_KWARGS``)."""
    g = ctx.general
    return dict(
        waveform_t0=ctx.cfg.data.reference_time,
        data_td_settings=g.data_td_settings,
        sampling_frequency=1.0 / g.dt,
        orbits=g.gpu_orbits if g.force_backend == g.gpu_backend else g.orbits,  # as GeneralSetup does
        tukey_alpha=g.window_alpha,
        stft_dt=g.stft_dt,
        freq_min=g.start_freq,
        freq_max=g.end_freq,
        force_backend=g.force_backend,
    )

def waveform_metadata_config(init_kwargs: dict) -> dict:
    """What SourceMetadata records: no live objects, CPU backend (as the settings files did)."""
    config = {k: v for k, v in init_kwargs.items() if k not in ("data_td_settings", "orbits")}
    config["force_backend"] = "cpu"
    return config


def make_inner_moves(block: BlockBase, filled: dict[str, dict]) -> list[tuple[Any, float]]:
    """``[(move, weight)]``; ``filled`` maps class name -> builder-filled kwargs (e.g. SkyMove ind_map)."""
    if block.inner_moves is None:
        return None
    moves = []
    for spec in block.inner_moves:
        cls = find_class(spec.class_name, MOVE_MODULES)
        moves.append((cls(**deepcopy(spec.kwargs), **filled.get(cls.__name__, {})), spec.weight))
    return moves

def make_priors(block: BlockBase, general) -> GFBasePrior:
    """``{name: prior}`` for the block's priors, with builder-filled kwargs (e.g. observation_time)."""
    from ereborpriors.base import GFBasePrior

    if isinstance(block.priors, PriorFile):
        # TODO(l2d): GFBasePrior.from_yaml returns the base class, which has no get_transforms / get_periodic.
        raise NotImplementedError("priors: {file: ...} is not supported yet; use priors: {class: ...}.")

    else:
        cls = prior_class(block.priors.class_name)
        factory = getattr(cls, "prior_factory", cls)
        engine = {"observation_time": general.Tobs}
        accepted = accepted_kwargs(factory)

        kwargs = deepcopy(block.priors.kwargs)
        for k, v in engine.items():
            if accepted is None or k in accepted:
                kwargs[k] = v

        prior_cls = factory(**kwargs)

    if block.priors.output_file is not None:
        prior_cls.to_yaml(str(block.priors.output_file))

        if hasattr(block, "links") and isinstance(block.links, Links):
            block.links.prior_model_config_file = str(block.priors.output_file)

    return prior_cls

def source_metadata(block: SourceBlock, init_kwargs: dict, general):
    from lisatools.globalfit.postprocessing import SourceMetadata

    return SourceMetadata(
        source_type=block.SOURCE_TYPE,
        frequency_ranges=[(general.start_freq, general.end_freq)],
        waveform_model=block.waveform.class_name,
        waveform_model_code_link=block.links.waveform_model_code_link,
        waveform_model_config=waveform_metadata_config(init_kwargs),
        prior_model_code_link=block.links.prior_model_code_link,
        prior_model_config_file=block.links.prior_model_config_file,
    )

