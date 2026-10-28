"""Read an L2D run folder (``global.yaml`` + ``blocks/*.yaml``) into a validated config."""

from __future__ import annotations

import inspect
import os

import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict,PositiveInt, PositiveFloat, Field, field_validator, model_validator

from typing import Annotated
from pydantic import BeforeValidator, AfterValidator, ValidationInfo, Discriminator, Tag

from lisatools.utils.constants import YRSID_SI

from .blocks import BlockSpec
from .blocks.base import SourceBlock
from .typing import Duration, RunPath

SCHEMA_VERSION = 1


class _Loader(yaml.SafeLoader):
    """SafeLoader that also reads ``1e-4`` as a float (YAML 1.2 rule; PyYAML gives a str)."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":  # `<<: *anchor`, expanded by SafeLoader
                continue
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"found duplicate key {key!r}", key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)

_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"^[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+$"),
    list("-+0123456789"),
)


def read_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML file; an empty file is an empty mapping."""
    with open(path) as f:
        data = yaml.load(f, Loader=_Loader)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping, got {type(data).__name__}.")
    return data


class _Section(BaseModel):
    """Base for config sections: unknown keys are errors, never silently ignored."""

    model_config = ConfigDict(extra="forbid")


class RunSection(_Section):
    name: str
    output_dir: RunPath
    seed: int | None = None
    num_iterations: PositiveInt = 500
    backup_iter: PositiveInt = 5
    submission_dir: RunPath | None = None


class SamplerSection(_Section):
    nwalkers: PositiveInt
    ntemps: PositiveInt


class ComputeSection(_Section):
    gpus: list[int] | None = None  # None -> CPU run
    backend: Literal["cpu", "cuda11x", "cuda12x", "cuda13x"] = "cpu"
    head_rank: int = 1
    main_rank: int = 0
    flow_gpu: int | None = None  # where every flow trainer runs; never a sampling GPU (None on cpu -> "cpu")

    @model_validator(mode="after")
    def _cpu_means_no_gpus(self):
        if (self.backend == "cpu") != (self.gpus is None):
            raise ValueError("use backend: cpu with no gpus, or a cuda backend with gpus: [...]")
        if self.flow_gpu is not None and self.backend == "cpu":
            raise ValueError("flow_gpu needs a cuda backend; on cpu the flow trainers run on the cpu.")
        if self.flow_gpu is not None and self.flow_gpu in self.gpus:
            raise ValueError(f"flow_gpu {self.flow_gpu} is a sampling GPU (gpus: {self.gpus}); use another one.")
        return self

    @model_validator(mode="after")
    def _validate_gpus(self):
        if self.gpus is not None:
            import jax
            jax.config.update("jax_cuda_visible_devices", ",".join(str(gpu) for gpu in self.gpus))

            if len(self.gpus) != len(jax.devices("gpu")):
                raise ValueError(
                    f"gpus {self.gpus} does not match the GPUs JAX sees: {jax.devices('gpu')}. "
                    "Check CUDA_VISIBLE_DEVICES or jax_cuda_visible_devices."
                )
        return self

class BlockConfig(BaseModel):
    """One source/noise block. Type-specific fields are validated in a later step."""

    model_config = ConfigDict(extra="allow")

    type: str
    enabled: bool = True

# --- Data section --- #

class FilterSpec(_Section):
    cutoff: PositiveFloat  # Hz
    order: PositiveInt = 2
    zero_phase: bool = True

class TrimSpec(_Section):
    duration: Duration  # seconds, or a fraction when is_percent
    is_percent: bool = False
    trimming_type: Literal["from_each_end", "keep_from_start", "discard_from_start"] = "from_each_end"

class DownsampleSpec(_Section):
    """Resample to fs=1/dt with dt taken from the data section."""

    window: tuple[str, float] = ("kaiser", 31.0)

class L1OrbitsSpec(_Section):
    """Orbits read from the mojito L1 data folder (``lisatools.detector.L1Orbits``)."""

    type: Literal["l1"]
    frame: Literal["icrs", "ecliptic"] = "icrs"
    armlength: PositiveFloat | None = None  # None -> L1Orbits' own default
    force_backend: Literal["cpu", "cuda11x", "cuda12x", "cuda13x"] = "cpu"


class EqualArmlengthOrbitsSpec(_Section):
    """Packaged equal-armlength orbit file (``EqualArmlengthOrbits``)."""

    type: Literal["equal_armlength"]
    force_backend: Literal["cpu", "cuda11x", "cuda12x", "cuda13x"] = "cpu"


class ESAOrbitsSpec(_Section):
    """Packaged ESA trailing orbit file (``ESAOrbits``)."""

    type: Literal["esa"]
    force_backend: Literal["cpu", "cuda11x", "cuda12x", "cuda13x"] = "cpu"

OrbitsSpec = Annotated[
    L1OrbitsSpec | EqualArmlengthOrbitsSpec | ESAOrbitsSpec, Field(discriminator="type")
]

# Which orbit types each loader can build (L1DataLoader passes the L1 orbit file to the class).
_ORBITS_PER_LOADER = {"mojito_l1": ("l1",)}

class DataSection(_Section):
    loader: Literal["mojito_l1"] = "mojito_l1"
    path: RunPath
    source_types: list[str]
    source_ids: dict[str, list[int]] = Field(default_factory=dict)
    orbits: OrbitsSpec = Field(default_factory=lambda: L1OrbitsSpec(type="l1"))
    reference_time: float | None = None  # absolute SSB time (s) the source waveforms are referenced to
    Tobs: Duration
    dt: Duration                      # seconds: `dt: 5.0` or `dt: 5 s`
    highpass: FilterSpec | None = None
    lowpass: FilterSpec | None = None
    downsample: DownsampleSpec | None = None
    trim: TrimSpec | None = None
    alias_correction: bool = True  # whether to apply the alias correction to the data (default: True)

    @model_validator(mode="after")
    def _orbits_match_loader(self):
        allowed = _ORBITS_PER_LOADER[self.loader]
        if self.orbits.type not in allowed:
            raise ValueError(
                f"loader {self.loader!r} can build orbits of type {allowed}, got {self.orbits.type!r}."
            )
        return self

    @model_validator(mode="after")
    def _filters_need_explicit_trim(self):
        if (self.highpass is not None or self.lowpass is not None) and self.trim is None:
            raise ValueError(
                "filtering without data.trim makes process() trim 100 h from each end; set data.trim."
            )
        return self

# --- Domain section --- #

class _Band(_Section):
    min_freq: PositiveFloat
    max_freq: PositiveFloat

    @model_validator(mode="after")
    def _ordered(self):
        if self.min_freq >= self.max_freq:
            raise ValueError(f"min_freq ({self.min_freq}) must be below max_freq ({self.max_freq}).")
        return self


class FDDomain(_Band):
    type: Literal["fd"]


class STFTDomain(_Band):
    type: Literal["stft"]
    segment_duration: Duration  # -> STFTSettings.make_factory(big_dt=...)


Domain = Annotated[FDDomain | STFTDomain, Field(discriminator="type")]

# --- Window section --- #

WindowType = Literal[
    "tukey", "planck", "kaiser", "hamming", "hanning", "welch", "rectangular",
    "blackman-harris", "bh92", "nuttall", "nuttall3", "nuttall3a", "nuttall3b",
    "nuttall4", "nuttall4a", "nuttall4b", "nuttall4c",
]


class WindowSection(_Section):
    type: WindowType = "tukey"
    taper_duration: Literal["auto"] | Duration = "auto"  # auto -> 1/domain.min_freq (builder)
    normalize: bool = False  # pass the window to the sensitivity backend (window_values)

# --- Sensitivity section --- #

# Constructor arguments GeneralSetup fills in itself; never given in the YAML.
_ENGINE_FILLED = frozenset({"settings", "orbits", "force_backend", "window_values"})


def _sensitivity_class(name: str) -> type:
    from lisatools import sensitivity  # local: only needed to check this section

    cls = getattr(sensitivity, name, None)
    if not isinstance(cls, type):
        raise ValueError(f"lisatools.sensitivity has no class {name!r}.")
    return cls

def _init_params(cls: type) -> set[str] | None:
    """Names ``cls.__init__`` accepts; ``None`` if it takes ``**kwargs`` (anything goes)."""
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return None
    return set(params) - {"self"}

class SensitivitySection(_Section):
    """A class from ``lisatools.sensitivity`` and the kwargs for its ``__init__``."""

    class_name: str = Field(alias="class")
    kwargs: dict[str, Any] = Field(default_factory=dict)
    fixed_psd_kwargs: dict[str, Any] | None = None  # call kwargs; used only without a psd block

    @model_validator(mode="after")
    def _kwargs_match_class(self):
        given = set(self.kwargs)
        if given & _ENGINE_FILLED:
            raise ValueError(f"{sorted(given & _ENGINE_FILLED)} are filled in by the engine; remove them.")
        accepted = _init_params(_sensitivity_class(self.class_name))
        if accepted is not None and given - accepted:
            raise ValueError(
                f"{self.class_name} does not accept {sorted(given - accepted)}; "
                f"it takes {sorted(accepted - _ENGINE_FILLED)}."
            )
        return self

#--- Global metadata section ---#

class MetadataSection(_Section):
    codename: str | None = "Erebor"
    version: str | None = Field(default=None, pattern=r"^[^_]+_.+$")  # "<run_type>_<run_id>"
    contact: str | None = None
    code_link: str | None = None
    input_data_link: str | None = None
    input_reference: str | None = None
    noise_model: str | None = None
    noise_model_code_link: str | None = None
    comment: str | None = None


#--- Recipe section ---#

class StockRecipeSpec(_Section):
    """A recipe from ``build_recipe.STOCK_RECIPES`` and its kwargs."""

    class_name: str = Field(default="StockRecipe", alias="class")
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _known_recipe(self):
        from .build_recipe import make_stock_recipe  # local: build_recipe is the runtime side

        make_stock_recipe(self.class_name, self.kwargs)
        return self


class RecipeFile(_Section):
    """A Python file whose ``function`` adds the recipe steps once the library has seeded the state."""

    file: RunPath
    function: str = "setup_recipe"

    @model_validator(mode="after")
    def _function_fits(self):
        from .build_recipe import load_recipe_function

        load_recipe_function(self.file, self.function)
        return self


def _recipe_kind(v):
    if isinstance(v, dict):
        return "file" if "file" in v else "class"
    return "file" if isinstance(v, RecipeFile) else "class"


RecipeRef = Annotated[
    Annotated[StockRecipeSpec, Tag("class")] | Annotated[RecipeFile, Tag("file")],
    Discriminator(_recipe_kind),
]


class RunConfig(_Section):
    schema_version: int
    run: RunSection
    sampler: SamplerSection
    compute: ComputeSection = Field(default_factory=ComputeSection)
    blocks: dict[str, BlockSpec] = Field(default_factory=dict)
    data: DataSection
    domain: Domain
    metadata: MetadataSection = Field(default_factory=MetadataSection)
    window: WindowSection = Field(default_factory=WindowSection)
    sensitivity: SensitivitySection
    recipe: RecipeRef = Field(default_factory=StockRecipeSpec)  # absent -> the stock recipe, no kwargs

    @field_validator("schema_version")
    @classmethod
    def _known_version(cls, v: int) -> int:
        if v != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version {v}; this code reads {SCHEMA_VERSION}.")
        return v

    @model_validator(mode="after")
    def _normalize_needs_window_values(self):
        if self.window.normalize:
            accepted = _init_params(_sensitivity_class(self.sensitivity.class_name))
            if accepted is not None and "window_values" not in accepted:
                raise ValueError(
                    f"window.normalize has no effect: {self.sensitivity.class_name} takes no window_values."
                )
        return self

    @model_validator(mode="after")
    def _blocks_fit_the_run(self):
        """Check the blocks against the data section (reference time, source ids and types)."""
        data = self.data
        source_types = {s.lower() for s in data.source_types}
        for name, block in self.blocks.items():
            if name != block.type:
                raise ValueError(f"blocks/{name} has type {block.type!r}; name the file {block.type}.yaml.")
            if not block.enabled:
                continue
            # a ladder may use a subset of the sampler's temperatures, never more (galfor has none: the psd move's)
            tempering = getattr(block, "tempering", None)
            ntemps = None if tempering is None else tempering.ntemps
            if ntemps is not None and ntemps > self.sampler.ntemps:
                raise ValueError(
                    f"block {name!r}: tempering.ntemps ({ntemps}) is above sampler.ntemps ({self.sampler.ntemps})."
                )
            if ntemps is not None and block.FULL_LADDER and ntemps != self.sampler.ntemps:
                raise ValueError(
                    f"block {name!r}: its ladder spans every temperature; tempering.ntemps ({ntemps}) "
                    f"must equal sampler.ntemps ({self.sampler.ntemps}) or be left out."
                )
            if block.NEEDS_REFERENCE_TIME and data.reference_time is None:
                raise ValueError(f"block {name!r} needs data.reference_time (absolute SSB time, in seconds).")
            if isinstance(block, SourceBlock):
                if block.flow is not None and self.compute.backend != "cpu" and self.compute.flow_gpu is None:
                    raise ValueError(f"block {name!r} has a flow: set compute.flow_gpu, the GPU its trainer runs on.")
                key = block.SOURCE_IDS_KEY
                if key not in source_types:
                    raise ValueError(f"block {name!r} needs {key!r} in data.source_types.")
                if block.nleaves == "auto" and not data.source_ids.get(key):
                    raise ValueError(f"block {name!r}: nleaves: auto needs data.source_ids[{key!r}].")
        return self

    @model_validator(mode="after")
    def _galfor_rides_on_psd(self):
        """galfor is sampled by the psd move, through the sensitivity's galactic grid."""
        galfor = self.blocks.get("galfor")
        if galfor is None or not galfor.enabled:
            return self
        psd = self.blocks.get("psd")
        if psd is None or not psd.enabled:
            raise ValueError("the galfor block is sampled by the psd move; add an enabled psd block.")
        accepted = _init_params(_sensitivity_class(self.sensitivity.class_name))
        if accepted is not None and "galactic_grid_kwargs" not in accepted:
            raise ValueError(f"the galfor block needs a sensitivity with a galactic grid; {self.sensitivity.class_name} has none.")
        if not self.sensitivity.kwargs.get("galactic_grid_kwargs"):
            raise ValueError(
                "the galfor block needs sensitivity.kwargs.galactic_grid_kwargs (R_d, z_d, ...); "
                "without the grid the sensitivity has no galaxy and galfor would sample nothing."
            )
        return self


def load_run(path: str | Path) -> RunConfig:
    """Load a run folder (``global.yaml`` + ``blocks/*.yaml``) or a single resolved config file.

    Relative paths inside are relative to that folder (for a file: the folder it sits in).
    """
    path = Path(path).absolute()
    if path.is_dir():
        raw = read_yaml(path / "global.yaml")
        if "blocks" in raw:
            raise ValueError("global.yaml must not define 'blocks'; use one file per block in blocks/.")
        raw["blocks"] = {p.stem: read_yaml(p) for p in sorted((path / "blocks").glob("*.yaml"))}
        run_dir = path
    else:
        raw = read_yaml(path)
        run_dir = path.parent
    return RunConfig.model_validate(raw, context={"run_dir": run_dir})


def dump_run(cfg: RunConfig, path: str | Path) -> Path:
    """Write ``cfg`` fully resolved (defaults filled, paths absolute) as one reloadable YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.model_dump(mode="json", by_alias=True), f, sort_keys=False)
    return path