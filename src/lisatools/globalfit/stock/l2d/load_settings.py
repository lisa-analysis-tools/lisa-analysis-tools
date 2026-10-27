"""Turn a validated :class:`RunConfig` into the engine's ``GeneralSettings`` / ``GeneralSetup``.

YAML keys are user-facing; each is mapped here onto its GeneralSettings field. Values that
need building (domain factory, loader and preprocessing kwargs, sensitivity class) are
assembled here too. What GeneralSetup derives (window_alpha, start/end_freq, final Tobs,
file paths) is read from it, never recomputed.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

from lisatools.detector import EqualArmlengthOrbits, ESAOrbits, L1Orbits
from lisatools.domains import FDSettings, STFTSettings
from lisatools.globalfit.preprocessing import L1ProcessingStep
from lisatools.globalfit.engine import GeneralSettings, GeneralSetup, GlobalFitSettings, RankInfo

from .blocks import BUILDERS, BlockBuild, BuildContext
from .io import FDDomain, RunConfig, STFTDomain, _sensitivity_class, dump_run

SNAPSHOT_NAME = "run_config.yaml"  # the resolved config, next to the run's other artifacts

# YAML tags -> engine classes.
LOADERS = {"mojito_l1": L1ProcessingStep}
ORBITS = {"l1": L1Orbits, "equal_armlength": EqualArmlengthOrbits, "esa": ESAOrbits}

def _domain_factory(domain):
    if isinstance(domain, FDDomain):
        return FDSettings.make_factory(min_freq=domain.min_freq, max_freq=domain.max_freq)
    if isinstance(domain, STFTDomain):
        return STFTSettings.make_factory(
            big_dt=domain.segment_duration, min_freq=domain.min_freq, max_freq=domain.max_freq
        )
    raise TypeError(f"no domain factory for {type(domain).__name__}")

def _processor_init_kwargs(cfg: RunConfig) -> dict:
    orbits = cfg.data.orbits
    orbits_kwargs = orbits.model_dump(exclude={"type"}, exclude_none=True)
    orbits_kwargs["force_backend"] = cfg.compute.backend
    return dict(
        L1_folder=str(cfg.data.path),
        source_types=list(cfg.data.source_types),
        source_ids=deepcopy(cfg.data.source_ids),  # {} not None: the loader calls .get() on it
        orbits_class=ORBITS[orbits.type],
        orbits_kwargs=orbits_kwargs,
    )

def _preprocess_kwargs(cfg: RunConfig) -> dict:
    """process() kwargs; absent steps are an explicit None so GeneralSetup's defaults don't apply."""
    data = cfg.data

    def dump(step):
        return None if step is None else step.model_dump()

    downsample = None
    if data.downsample is not None:
        downsample = dict(target_fs=1.0 / data.dt, window=tuple(data.downsample.window))
    return dict(
        highpass_kwargs=dump(data.highpass),
        lowpass_kwargs=dump(data.lowpass),
        downsample_kwargs=downsample,
        trim_kwargs=dump(data.trim),
        Tobs=data.Tobs,
    )

def make_general_settings(cfg: RunConfig) -> GeneralSettings:
    """Map ``cfg`` onto :class:`GeneralSettings`. Cheap: nothing is loaded."""
    run, window, meta = cfg.run, cfg.window, cfg.metadata
    taper = 1.0 / cfg.domain.min_freq if window.taper_duration == "auto" else window.taper_duration

    return GeneralSettings(
        # run
        base_file_name=run.name,
        file_store_dir=os.path.join(str(run.output_dir), ""),  # engine concatenates paths
        submission_parent_folder=None if run.submission_dir is None else str(run.submission_dir),
        random_seed=run.seed,
        num_iterations=run.num_iterations,
        backup_iter=run.backup_iter,
        # sampler / compute (head_rank / main_rank go to RankInfo later)
        nwalkers=cfg.sampler.nwalkers,
        ntemps=cfg.sampler.ntemps,
        gpus=cfg.compute.gpus,
        gpu_backend=None if cfg.compute.backend == "cpu" else cfg.compute.backend,
        # data / domain / window
        Tobs=cfg.data.Tobs,
        dt=cfg.data.dt,
        data_processor_class=LOADERS[cfg.data.loader],
        processor_init_kwargs=_processor_init_kwargs(cfg),
        preprocess_kwargs=_preprocess_kwargs(cfg),
        domain_settings=_domain_factory(cfg.domain),
        window_type=window.type,
        window_taper_duration=taper,
        normalize_window=window.normalize,
        # sensitivity
        sensitivity_backend_class=_sensitivity_class(cfg.sensitivity.class_name),
        sensitivity_init_kwargs=deepcopy(cfg.sensitivity.kwargs),  # GeneralSetup mutates nested dicts
        fixed_psd_kwargs=deepcopy(cfg.sensitivity.fixed_psd_kwargs),
        # metadata
        global_fit_codename=meta.codename,
        global_fit_version=meta.version,
        global_fit_contact=meta.contact,
        global_fit_code_link=meta.code_link,
        input_data_link=meta.input_data_link,
        input_reference=meta.input_reference,
        noise_model=meta.noise_model,
        noise_model_code_link=meta.noise_model_code_link,
        comment=meta.comment,
    )

def load_general_setup(cfg: RunConfig) -> GeneralSetup:
    """Build the GeneralSetup: loads and preprocesses the data, builds domain, orbits, sensitivity."""
    return GeneralSetup(make_general_settings(cfg))

def make_rank_info(cfg: RunConfig) -> RankInfo:
    return RankInfo(head_rank=cfg.compute.head_rank, main_rank=cfg.compute.main_rank)


def load_blocks(cfg: RunConfig, general: GeneralSetup) -> dict[str, BlockBuild]:
    """Build every enabled block against the built GeneralSetup (derived values are read from it)."""
    ctx = BuildContext(cfg, general)
    return {name: BUILDERS[block.type](block, ctx) for name, block in cfg.blocks.items() if block.enabled}


def build_curr_info(cfg: RunConfig):
    """The engine's ``CurrentInfoGlobalFit`` for a run: general setup, every enabled block, the recipe."""
    from lisatools.globalfit.run import CurrentInfoGlobalFit  # local: run.py imports mpi4py

    from .build_recipe import make_setup_function

    general = load_general_setup(cfg)
    builds = load_blocks(cfg, general)
    settings = GlobalFitSettings(
        source_info={name: build.setup for name, build in builds.items()},
        general_info=general,
        rank_info=make_rank_info(cfg),
        setup_function=make_setup_function(cfg, builds),
        source_metadata={name: build.metadata for name, build in builds.items()},
    )
    return CurrentInfoGlobalFit(settings)


def write_snapshot(cfg: RunConfig, general: GeneralSetup) -> Path:
    """Write the resolved config into the run's artifacts folder (call it on one rank only)."""
    return dump_run(cfg, Path(general.artifacts_file_dir) / SNAPSHOT_NAME)