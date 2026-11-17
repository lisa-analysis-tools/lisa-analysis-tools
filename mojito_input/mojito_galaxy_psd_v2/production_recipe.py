"""Recipe of mojito_galaxy_psd_v2, from the Spider runs from 8 mHz up: the psd search, the exact stretch in the burn-in
only, the GB moves taking turns.

Named in global.yaml as ``recipe: {file: production_recipe.py}``. It does five things:

1. Fixed noise. The sensitivity splits ``psd_params`` with ``psd_params[..., :n]``, which a YAML list cannot do, so
   the lists of ``sensitivity.fixed_psd_kwargs`` become arrays when this file is imported (``noise_lists_to_arrays``).
   It does nothing once the configuration loader makes the arrays itself, and nothing in a run that samples the noise.
2. Fixed sources. The search candidates (or true sources) that lie outside the sampled range but whose template
   reaches the domain are subtracted once from every walker's residual with the sampler's own DFFR templates
   (``subtract_fixed_sources``). The setup function runs at every start and restart, and the engine rebuilds the
   residuals from the data and the state at each, so the subtraction is repeated the same way and survives a restart.
   Off when ``FIXED_SOURCES_FILE`` is None, as in this run, whose sampled range holds every start candidate.
3. The in-model knobs of the user's decision of 2026-11-05: in the burn-in the exact group stretch with probability
   0.5 and a new Fisher matrix at every block, in the kept steps no stretch and the matrices held for 50 blocks.
4. The steps of the stock recipe (``StockRecipe.add_steps``): the psd search, then the stages ``burn-in`` and ``pe``.
   A branch whose PE moves take turns gets one combined move per choice (``JointPEMoves``), picked by the shares of
   each stage.
5. The split and merge's search part. The step of the kept iterations fixes it, so that the search memory takes no
   new fit from then on, also after a restart inside the step.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np

from lisatools.globalfit.moves.gbspecialmove import GBSpecialBase
from lisatools.globalfit.recipe import IterationCountRecipeStep, PERecipeStep, SearchRecipeStep
from lisatools.globalfit.stock.l2d import load_settings
from lisatools.globalfit.stock.l2d.build_recipe import JointPEMoves

# Under "lisatools", whose level and run-log handler the global fit sets.
logger = logging.getLogger("lisatools.globalfit.production_recipe")

# * A file of the sources outside the sampled range whose template reaches the domain, which the recipe subtracts once.
# * None: no such source. This run samples the whole f0 range of the GB prior, which holds every start candidate.
FIXED_SOURCES_FILE: Path | None = None
# Rows of one template-fill call: this many sources times the walkers. The fill adds no big buffer, so it only bounds
# the index arrays.
FIXED_SOURCES_PER_CALL = 8
# Pixels of the STFT grid above the first interior edge in which the residual's power is read, next to the edge bands
POWER_PIXELS_ABOVE_EDGE_BAND = 16

# The burn-in runs this many iterations: the recipe stores the iteration each step started at in the backend file, so a
# restart inside the burn-in runs the iterations that are left, not the whole length again.
BURN_IN_ITERATIONS = 50
BURN_IN_KNOBS = {"stretch_probability": 0.5, "num_stationary_fim_steps": 1}
KEPT_KNOBS = {"stretch_probability": 0.0, "num_stationary_fim_steps": 50}


# --- 1. the fixed noise --- #

def noise_lists_to_arrays(make_general_settings):
    """``make_general_settings`` with the list values of ``fixed_psd_kwargs`` made into float arrays."""

    def make_general_settings_with_arrays(cfg):
        settings = make_general_settings(cfg)
        fixed = settings.fixed_psd_kwargs
        if fixed:
            settings.fixed_psd_kwargs = {
                name: np.asarray(value, dtype=float) if isinstance(value, (list, tuple)) else value
                for name, value in fixed.items()
            }
        return settings

    make_general_settings_with_arrays.converts_noise_lists = True
    return make_general_settings_with_arrays


# * This file is imported at validation and again when the setup function is built; the second time changes nothing.
if not getattr(load_settings.make_general_settings, "converts_noise_lists", False):
    load_settings.make_general_settings = noise_lists_to_arrays(load_settings.make_general_settings)


def report_noise_in_sensitivity(curr, acs) -> None:
    """Log that walker 0's sensitivity holds the fixed noise values, and the XX element it holds at two frequencies.

    The sensitivity does not store the amplitudes it was set with, so the matrix is compared with one computed from the
    fixed values (it must agree to rounding) and with one computed from the old defaults, psd_params [1.5e-11, 3e-15]
    and no foreground (it must not), which shows the comparison can tell the two apart.
    """
    xp = acs.xp
    backend = curr.general_info.sensitivity_backend
    fixed = curr.general_info.fixed_psd_kwargs
    held = acs.flatten()[0].sens_mat.sens_mat
    psd_params, galfor_params = fixed["psd_params"], fixed["galfor_params"]
    fixed_matrix = backend.compute_sensitivity_matrix(
        Soms_d_in=float(psd_params[0]), Sa_a_in=float(psd_params[1]), Amp=float(galfor_params[0]),
        alpha=float(galfor_params[1]), f_1=float(galfor_params[2]), kn=float(galfor_params[3]),
        f_2=float(galfor_params[4]), smooth=True, convolve_window=backend.convolve_window,
    )
    default_matrix = backend.compute_sensitivity_matrix(smooth=True, convolve_window=backend.convolve_window)
    scale = float(xp.abs(fixed_matrix).max())
    same = float(xp.abs(held - fixed_matrix).max()) / scale
    other = float(xp.abs(held - default_matrix).max()) / scale
    settings = acs.settings
    # * the largest element sits where the noise is highest, so the per-pixel ratio of the XX elements tells the two apart
    ratio_to_default = float(xp.median(xp.abs(held[0, 0]) / xp.abs(default_matrix[0, 0])))
    lines = [f"[FIXED_NOISE] walker 0 sensitivity against the matrix of the fixed values: largest difference {same:.3e} of "
             f"the largest element; against the old defaults: {other:.3e}; median over pixels of XX held / XX old defaults "
             f"{ratio_to_default:.4f}"]
    for frequency_hz in (10e-3, 20e-3):
        pixel = int(np.clip(round(frequency_hz / settings.df) - settings.ind_min, 0, settings.NF_active - 1))
        element = float(held[0, 0, 0, pixel].real)
        lines.append(f"[FIXED_NOISE] XX element (time segment 0) at {frequency_hz * 1e3:.3f} mHz "
                     f"(pixel {pixel}): {element:.6e}")
    for line in lines:
        logger.info(line)
    del fixed_matrix, default_matrix
    acs.free_gpu_memory()


# --- 2. the fixed sources --- #

def residual_power_in_windows(acs, windows: dict) -> dict:
    """Power of walker 0's residual, summed over time and channels, in pixel windows of the STFT grid.

    ``windows`` maps a label to ``(first pixel, end pixel)`` of the active grid. Two sums are returned per window: the raw
    power |d|^2 and the noise-weighted power d^dagger C^-1 d with the full complex 3 x 3 covariance of the walker's
    sensitivity (None when the covariance has an unexpected shape).
    """
    xp = acs.xp
    settings = acs.settings
    num_times, num_pixels, num_channels = settings.NT, settings.NF_active, acs.nchannels
    residual = acs.linear_data_arr[0].reshape(-1, num_channels, num_times, num_pixels)[0]
    inverse_covariance = acs.flatten()[0].sens_mat.invC
    raw_per_pixel = (residual.real**2 + residual.imag**2).sum(axis=(0, 1))
    weighted_per_pixel = None
    if tuple(inverse_covariance.shape) == (num_channels, num_channels, num_times, num_pixels):
        weighted_per_pixel = xp.einsum("itf,ijtf,jtf->f", residual.conj(), inverse_covariance, residual).real
    to_host = xp.asnumpy if hasattr(xp, "asnumpy") else np.asarray
    raw_per_pixel = to_host(raw_per_pixel)
    weighted_per_pixel = None if weighted_per_pixel is None else to_host(weighted_per_pixel)
    power = {}
    for label, (first, end) in windows.items():
        power[label] = dict(
            raw=float(raw_per_pixel[first:end].sum()),
            weighted=None if weighted_per_pixel is None else float(weighted_per_pixel[first:end].sum()),
            pixels=int(end - first),
        )
    return power


def power_windows(acs, band_edges_hz) -> dict:
    """Pixel windows of the first edge band, the pixels above it, the last edge band and the whole grid."""
    settings = acs.settings

    def to_pixel(frequency_hz: float) -> int:
        return int(np.clip(np.ceil(frequency_hz / settings.df) - settings.ind_min, 0, settings.NF_active))

    first_interior = to_pixel(float(band_edges_hz[1]))
    last_interior_end = to_pixel(float(band_edges_hz[-2]))
    return {
        "lower edge band": (0, first_interior),
        f"{POWER_PIXELS_ABOVE_EDGE_BAND} pixels above it": (first_interior, first_interior + POWER_PIXELS_ABOVE_EDGE_BAND),
        "upper edge band": (last_interior_end, settings.NF_active),
        "whole grid": (0, settings.NF_active),
    }


def subtract_fixed_sources(curr, acs, fixed_sources_file: Path) -> int:
    """Subtract the sources of ``fixed_sources_file`` from every walker's residual, with the sampler's DFFR templates.

    The rows are mapped as the seeding maps a catalogue row (the GB search interface, parameters at the catalogue's
    reference time) and filled through the band likelihood engine with factor -1, as ``prepare_gb_moves`` subtracts
    the start sources. Returns the number of sources subtracted.
    """
    from gbgpu.gb_likelihood import make_band_likelihood_engine

    gb_info = curr.source_info["gb"]
    interface = copy.copy(gb_info.search_interface)
    interface.search_path = str(fixed_sources_file)
    sampling_rows = interface.get_catalog_values(curr)
    num_sources = len(sampling_rows)
    if num_sources == 0:
        logger.info(f"[FIXED_SOURCES] {fixed_sources_file.name}: no source to subtract")
        return 0

    physical_rows = np.asarray(gb_info.transform.both_transforms(sampling_rows))
    num_walkers = int(curr.general_info.nwalkers)
    xp = gb_info.gb_stft_comp.xp
    engine = make_band_likelihood_engine(
        curr.general_info.domain_settings, gb_stft_comp=gb_info.gb_stft_comp, nchannels=acs.nchannels,
        tdi_channel_setup=gb_info.waveform.tdi_channel_setup,
    )
    windows = power_windows(acs, gb_info.band_edges)
    before = residual_power_in_windows(acs, windows)
    for first in range(0, num_sources, FIXED_SOURCES_PER_CALL):
        chunk = physical_rows[first:first + FIXED_SOURCES_PER_CALL]
        rows = np.repeat(chunk, num_walkers, axis=0)
        walker_index = np.tile(np.arange(num_walkers), len(chunk))
        engine.fill_template(
            acs, xp.asarray(rows), xp.asarray(walker_index).astype(xp.int32), None, factor=-1,
            waveform_kwargs=gb_info.waveform.waveform_kwargs,
        )
        acs.free_gpu_memory()
    after = residual_power_in_windows(acs, windows)

    f0_mhz = np.sort(sampling_rows[:, 1])
    logger.info(
        f"[FIXED_SOURCES] subtracted {num_sources} sources of {fixed_sources_file.name} (f0 {f0_mhz[0]:.4f} to "
        f"{f0_mhz[-1]:.4f} mHz) from {num_walkers} walkers"
    )
    for label in windows:
        weighted_before, weighted_after = before[label]["weighted"], after[label]["weighted"]
        # * per pixel, time segment and channel: a region with no source reads the noise level of the covariance
        values_per_pixel = before[label]["pixels"] * acs.settings.NT * acs.nchannels
        weighted_text = "" if weighted_before is None else (
            f"; weighted {weighted_before:.6e} -> {weighted_after:.6e} (x{weighted_after / weighted_before:.4f}), per "
            f"pixel, segment and channel {weighted_before / values_per_pixel:.5e} -> {weighted_after / values_per_pixel:.5e}"
        )
        logger.info(
            f"[FIXED_SOURCES] walker 0 residual power in the {label} ({before[label]['pixels']} pixels): raw "
            f"{before[label]['raw']:.6e} -> {after[label]['raw']:.6e}{weighted_text}"
        )
    return num_sources


# --- 3. the knobs and 4. the steps --- #

def set_gb_knobs(gb_moves, knobs, step_name):
    """Set the stretch probability and the Fisher refresh interval on every GB move."""
    for move in gb_moves:
        # ! A tree without the port of rule B (2026-11-05) would run the earlier, biased stretch.
        if knobs["stretch_probability"] > 0.0 and not hasattr(move, "_run_in_model_halves"):
            raise AttributeError(
                f"{type(move).__name__} has no exact stretch: this lisa-analysis-tools predates the port of rule B."
            )
        for name, value in knobs.items():
            setattr(move, name, value)
    logger.info(
        f"{step_name}: GB stretch probability {knobs['stretch_probability']}, Fisher matrices held for "
        f"{knobs['num_stationary_fim_steps']} blocks"
    )


class GBKnobsIterationStep(IterationCountRecipeStep):
    """IterationCountRecipeStep that sets the GB moves' knobs when it starts."""

    def __init__(self, moves, gb_moves, knobs, num_iters, weights=None, label="burn-in"):
        super().__init__(moves, weights=weights, num_iters=num_iters)
        self.gb_moves, self.knobs, self.label = list(gb_moves), dict(knobs), label

    def setup_run(self, iteration, last_sample, sampler):
        super().setup_run(iteration, last_sample, sampler)
        set_gb_knobs(self.gb_moves, self.knobs, self.label)


class GBKnobsPEStep(PERecipeStep):
    """PERecipeStep that sets the GB moves' knobs when it starts, also after a restart inside it."""

    def __init__(self, moves, gb_moves, knobs, weights=None):
        super().__init__(moves=moves, weights=weights)
        self.gb_moves, self.knobs = list(gb_moves), dict(knobs)

    def setup_run(self, iteration, last_sample, sampler):
        super().setup_run(iteration, last_sample, sampler)
        set_gb_knobs(self.gb_moves, self.knobs, "pe")
        # The split and merge reads new fits of the search memory in the burn-in only: this call fixes its search part
        # when the kept steps start, and after a restart inside them it finds the memory fixed already.
        for move in self.gb_moves:
            if hasattr(move, "apply_settings"):
                move.apply_settings(fix_search_part=True)
                logger.info(f"pe: the search part of {move.name} is fixed")


def add_steps(recipe, engine_info, curr, acs, priors, state, l2d, burn_in_iterations):
    # * A diagnostic of a fixed-noise run only: a failure is logged as a warning and must not stop the run. The run's
    # * config is read, since the engine fills default values when the config names no fixed noise.
    if l2d.cfg.sensitivity.fixed_psd_kwargs:
        try:
            report_noise_in_sensitivity(curr, acs)
        except Exception as error:
            logger.warning(f"[FIXED_NOISE] the check of the sensitivity failed: {error!r}")
    if FIXED_SOURCES_FILE is not None:
        subtract_fixed_sources(curr, acs, FIXED_SOURCES_FILE)

    moves = l2d.block_moves(engine_info, curr, acs, priors, state)
    search = [move for branch in moves.values() for move in branch.search]
    gb_moves = [move for branch in moves.values() for move in branch.pe if isinstance(move, GBSpecialBase)]
    for branch in moves.values():
        for alternative in branch.alternatives or []:
            if isinstance(alternative.move, GBSpecialBase) and alternative.move not in gb_moves:
                gb_moves.append(alternative.move)
    if not gb_moves:
        raise ValueError("this recipe sets the GB moves' stretch, but no block has a GB PE move.")
    # Step names are recorded by the backend; a restart skips the completed ones.
    if search:
        recipe.add_recipe_component(SearchRecipeStep(moves=search), name="psd search")

    # * One combined move per choice of the GB block (its pe_move, then each extra move), with every other block's PE
    # * moves. The weights are the shares of the stage, whose names are the step names below.
    joint = JointPEMoves(moves)
    burn_in_moves, burn_in_weights = joint.moves("burn-in")
    pe_moves, pe_weights = joint.moves("pe")
    logger.info(f"burn-in shares {burn_in_weights}, kept-step shares {pe_weights}: the pe_move, then the extra moves")
    recipe.add_recipe_component(
        GBKnobsIterationStep(burn_in_moves, gb_moves, BURN_IN_KNOBS, burn_in_iterations, weights=burn_in_weights),
        name="burn-in",
    )
    recipe.add_recipe_component(GBKnobsPEStep(pe_moves, gb_moves, KEPT_KNOBS, weights=pe_weights), name="pe")


def setup_recipe(recipe, engine_info, curr, acs, priors, state, l2d):
    add_steps(recipe, engine_info, curr, acs, priors, state, l2d, BURN_IN_ITERATIONS)
