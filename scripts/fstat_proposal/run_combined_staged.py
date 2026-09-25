#!/usr/bin/env python
"""Staged combined run: noise+foreground search -> GB search -> GB PE, with VGBs.

The run this builds toward: mojito data, ``T_obs`` = 3 months, one composition
carrying **gb + vgb + psd + galfor**, driven by a THREE-stage recipe rather
than the single combined PE stage ``all_sources`` ships with.

    1. ``noise_search``  kind="search"  noise_joint_search (psd_pe+galfor_pe)
       The stage is a "search" in the sense of being convergence-gated, but
       its sub-moves are the PE ones: psd and galfor are fixed-dimensional,
       so there is nothing to search FOR and a maximizing proposal would be
       the wrong tool. ``JointMaxLogLSearch`` supplies the criterion --
       ``run_move_max_likelihood`` loops internally until the cold-chain max
       lnL plateaus ACROSS ALL its sub-moves jointly, and ``SearchRecipeStep``
       is done on its first check because the criterion lives INSIDE the
       move. So this stage converges the noise model before a single GB is
       subtracted.

    1b. ``noise_vgb_search`` kind="search" noise_vgb_joint_search
       The same, plus the 55 VGBs. They are KNOWN sources -- seeded from the
       catalogue, fixed-dimensional, no RJ -- so they too are fitted rather
       than searched for, and their power leaves the residual before the GB
       search starts.

    2. ``gb_search``     kind="rj"      psd_pe + galfor_pe + GB search moves
       ``RJRecipeStep`` watches the cold-chain leaf count on ``plateau_branch
       ="gb"`` and advances when it plateaus -- "GB PE when the leaves of
       search converge". Noise keeps sampling underneath (PE mode) so the GB
       search sees a live PSD rather than a frozen one.

    3. ``gb_pe``         kind="pe"      everything, VGBs included
       ``PERecipeStep`` never stops on its own.

Composition is ``all_sources`` MINUS mbh/emri/sobbh -- it already carries vgb,
and ``remove_branch`` drops the moves that sample a removed branch (stock
``bd02f94``), so nothing dangles.

WHY NOT gb_no_fg (default composition): it loads ``source_types = ("GB",)``
with a FIXED PSD, i.e. there is no instrument-noise realization in the data at
all. Sampling a PSD against it would be fitting noise that is not there. This
script defaults ``source_types`` to NOISE + GB + VGB so every sampled branch
has something to fit.

GB_ONLY=1 (2026-08-24, user ruling: "just do GBs, no PSD/foreground, or VGB
modules ... use the injection psd") flips that reasoning around ON PURPOSE:
the fit becomes ``erebor.gb_no_fg`` -- the stock variant DESIGNED for
GB-only running -- but with ``source_types`` kept at NOISE+GB+VGB, so the
DATA is the same full mojito injection (noise brick + GB galaxy + VGBs)
while only the gb branch is SAMPLED. The likelihood sensitivity is
gb_no_fg's fixed-PSD path: ``adjust_general`` fits the analytic 2-parameter
instrument model [Soms_d, Sa_a] to the mojito NOISE brick's tabulated
estimates (``resolve_noise_file_psd_params``; pin PSD_FROM_NOISE_FILE=1 to
make a missing brick a hard error instead of a silent fallback to the stock
levels) -> ``fixed_psd_kwargs`` -> run.py setup_acs's no-psd-branch path
builds every walker AC from it, and the ACA's ``linear_psd_arr`` (built from
each AC's invC) is the ONLY sensitivity the GB band engine, the sig-het
in-model scorer, the in-move F-stat grid fit, and the SNR rejection gates
read. Consequences, accepted by the ruling: the unmodeled confusion
foreground and the VGBs simply sit in the residual (unwhitened -- the fixed
PSD carries no galaxy term), and the reported lnL is source-only
(-1/2 <r|r>, gb_no_fg's ``likelihood_source_only=True``). Stages become
gb_search -> full_pe directly: no noise/vgb stages, no waiting on a PSD fit.
The GB machinery is bit-identical to the all_sources composition --
all_sources' GB branch IS the gb_no_fg stack (``AllSourcesGBSettings``
subclasses ``GBNoFgGBSettings``; ``prepare_gb_branch``/``setup_gb_moves``
shared) -- so every band/cap/sig-het/F-stat env knob applies unchanged.

Run (one GPU)::

    MOJITO_DATA_PATH=/path/to/L1 USE_GPU=1 GPU_BACKEND=cuda12x GPUS=0 \
    NITER=100 NWALKERS=16 \
    python scripts/fstat_proposal/run_combined_staged.py

Smoke mode (COMBINED_SMOKE=1) shrinks every axis and turns the GB/VGB debug
verifications on -- see the knob table at the bottom of this docstring.

Key env knobs
-------------
    COMBINED_SMOKE=1     small band / few iterations / debug on
    GB_ONLY=1            erebor.gb_no_fg composition: gb is the ONLY sampled
                         branch (fixed injection PSD, source-only lnL); the
                         data still carries SOURCE_TYPES in full. Stages:
                         gb_search -> full_pe. Incompatible with the
                         STAGE_NOISE_* flags below.
    SOURCE_TYPES         comma list (default "NOISE,GB,VGB")
    NITER, NWALKERS      sampler shape
    GB_NTEMPS, VGB_NTEMPS, PSD_NTEMPS     per-branch ladders
    GB_DEBUG=1, VGB_DEBUG=1               residual round-trip verification
    GB_DEBUG_PLOT_BAND   ONE band index -- unset means EVERY band, which at
                         ~1150 cells renders thousands of figures per proposal
    GB_WARM_START_COMPONENTS  warm-start components npz (workstream B):
                         when set, the gb_search stage runs the
                         ``rj_warm_search`` birth move IMMEDIATELY BEFORE
                         ``rj_fstat_search`` (user ruling 2026-08-24);
                         unset = stage lists unchanged, bit-identical run.
                         full_pe is NOT touched (the ruling names search;
                         the warm move is search-configured).
    MBHB_IDS / EMRI_IDS / SOBHB_IDS
                         comma id lists arming the mbh/emri/sobbh branches
                         (campaign gate S6, docs/6mo-campaign.md; user
                         ruling 2026-09-02 "we are adding MBHB EMRI
                         SOBHB"). Absent/empty = branch dropped (today's
                         4-branch behavior, bit-identical). Armed: a
                         ``source_search`` stage (joint max-lnL over the
                         armed source PE moves) runs FIRST -- sources
                         converge and subtract before the noise stages
                         fit the PSD ("MBH search first, full_year
                         pattern") -- and the armed PE moves join
                         gb_search + full_pe in the sobbh -> mbh -> emri
                         banking order. Ids land on
                         ``general.mojito_source_ids``; SOURCE_TYPES
                         defaults gain the armed classes (explicit env
                         still wins). Incompatible with GB_ONLY=1.
    GB_SEARCH_RJ_REPLACE gb_search F-stat REPLACEMENT move (default 1,
                         2026-08-24 exact-MH reinstatement): ``rj_replace``
                         runs IMMEDIATELY AFTER ``rj_fstat_search`` in the
                         gb_search cycle -- full 9-column candidates from
                         the fitted F-stat grids + epoch center table,
                         scored at their EXACT likelihood (never
                         phase-maximized), so a higher-SNR table draw can
                         relocate a mis-seated leaf in one accepted move.
                         =0 removes it (stage lists bit-identical to the
                         pre-replace runs). full_pe is NOT touched.
    STAGE_SKIP_NOISE=1   start at stage 2 (noise already converged)
    GB_SEARCH_SOURCE_EVERY   sobbh/mbh/emri gb_search cadence (default 5):
                         they propose on every Nth gb_search iteration,
                         staying subtracted in between; full_pe is NEVER
                         cadenced (2026-09-18 ruling; 09-15 was mbh/emri
                         only, every 10)
    STAGE_SKIP_SOURCE_SEARCH=1  no source_search stage: armed sources
                         start (and stay subtracted) at their seeded
                         coords; source proposals only in gb_search +
                         full_pe (exact-truth-start flow, 2026-09-14)
    STAGE_NOISE_ONLY=1   run only the two noise search stages, then stop
    STAGE_NOISE_VGB_PE=1 searches, then PE-sample psd+galfor+vgb (no GB);
                         bounded by NUM_ITERATIONS
    REMOVE_BRANCHES      comma list of whole branches to drop: gb, galfor,
                         vgb, psd. Removed branches leave the fit, every
                         stage move list, AND the DEFAULT injection stream
                         list ("we will not inject them"). Removing psd
                         (2026-09-14, the TRUTH-INJECTION NULL TEST) also
                         drops both noise stages and the NOISE stream, and
                         switches the likelihood to the FIXED-sensitivity
                         path -- it REQUIRES gb + galfor removed,
                         ADD_INSTRUMENT_NOISE=0 and UNEQUAL_ARM=0, all
                         enforced here.
    PSD_FIXED_PARAMS     comma floats [Soms_d, Sa_a], PHYSICAL (linear)
    GALFOR_FIXED_PARAMS  comma floats (amp, fk, alpha, f_1, f_2), PHYSICAL
                         -> general.fixed_psd_kwargs, the only sensitivity
                         the no-psd-branch path reads. See the basis note at
                         the wiring below before setting them.
"""
from __future__ import annotations

import logging
import os
import sys

# GB stage scoping -- MUST be seeded before ANY lisatools.globalfit.stock
# import: ``erebor``'s module-level default instances snapshot every
# env-backed field at import time and ``erebor.all_sources(...)`` CLONES
# that snapshot, so a setdefault placed after the import is invisible.
# GB_MODE=search arms the SEARCH-stage GB moves (leaf caps from 1, birth
# phase-max, flip 1.0, zero-leaf start -- injection/SNR seeding skipped);
# GB_PE_MOVES_STRICT keeps that scoped: the pe-NAMED instances (rj_prior /
# rj_fstat_mcmc / rj_refit, the full_pe stage) stay strictly PE regardless.
os.environ.setdefault("GB_MODE", "search")
os.environ.setdefault("GB_PE_MOVES_STRICT", "1")
# The gb_search stage lists Move("rj_prior_removal") -- build_gb_moves only
# CONSTRUCTS that stock move when gb.search_prior_removal is on (and mode is
# search), so the knob must default on here or recipe materialization fails
# with "no stock move under this name". Search-stage-only either way: the
# move is search-gated and full_pe never references it.
os.environ.setdefault("GB_SEARCH_PRIOR_REMOVAL", "1")
# GB_RIDGE_GIBBS (default 1): zero-likelihood fiber resample of the exact
# Mc-ratio-distance degeneracy after the RJ cycle in both GB stages -- the
# (Mc, r, dist) marginals are frozen along the curved ridge without it
# (measured 2026-08-20). Registered by build_gb_moves; =0 removes it.
# NITER is the name this script's docs (and muscle memory) use; the stock
# field is general.num_iterations -> env NUM_ITERATIONS (rule 0). Map it
# HERE, before any stock.erebor import snapshots the env -- a bare NITER
# was silently ignored (the smoke "NITER=12" never applied; runs ended at
# whatever NUM_ITERATIONS resolved to, looking like silent deaths).
if os.environ.get("NITER") and not os.environ.get("NUM_ITERATIONS"):
    os.environ["NUM_ITERATIONS"] = os.environ["NITER"]

logger = logging.getLogger("combined_staged")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip() in ("1", "true", "True")


def _pe_combine_kwargs() -> dict:
    """Combine kwargs shared by every PE stage.

    PE cycle style (user ruling 2026-08-26): the SEARCH-style single
    propose per iteration -- the backend stores ONE row per iteration,
    fixing the 1-row-per-move PE storage the random_choice mode produced
    -- but the cycle composition is DRAWN each iteration: ``len(moves)``
    weighted draws WITH replacement, all proposals equal weight by
    default. FULL_PE_WEIGHTED_CYCLE=0 + FULL_PE_RANDOM_CHOICE=1 restores
    the legacy one-move-per-step mode; search stages pass neither flag
    and are untouched.
    """
    return dict(
        share_temperature_control=False,
        weighted_cycle=_env_flag("FULL_PE_WEIGHTED_CYCLE", "1"),
        random_choice=_env_flag("FULL_PE_RANDOM_CHOICE", "0"),
    )


def _apply_smoke_defaults() -> None:
    """Shrink every cost axis and arm the debug verifications.

    ``setdefault`` throughout: an explicitly-set knob always wins, so the
    same script serves the smoke and the production run and the diff between
    them is exactly this function.
    """
    smoke = {
        # --- shape ---
        # 12, not 3: the chunked noise search (MAXLOGL_ITERS_PER_STEP) now
        # spends real engine iterations per stage -- this is the run-wide
        # total -- and the budget must reach the GB stages. NUM_ITERATIONS
        # is the REAL knob (NITER was a dead name; the module-top shim maps
        # it now).
        "NUM_ITERATIONS": "12",
        "NWALKERS": "8",
        "GB_NTEMPS": "6",
        "VGB_NTEMPS": "4",
        "PSD_NTEMPS": "4",
        # --- band: 5 WDM layers instead of 13 ---
        # _band_klohi snaps INWARD and requires >= 3 whole layers, so the
        # span must clear ~4*layer_df (1.3889e-4 Hz) after snapping. 7.0-7.3
        # mHz gives k_lo=51, k_hi=52 -> ONE layer, and raises.
        # 7.0-7.8 mHz gives k_lo=51, k_hi=56 -> 5 layers, 3 interior.
        "GB_MIN_FREQ": "7.0e-3",
        "GB_MAX_FREQ": "7.8e-3",
        "GB_NUM_REPEAT_PROPOSALS": "20",
        "GB_N_SUBBANDS": "256",
        # --- F-stat fit: every stage still RUNS, each does less work ---
        "FSTAT_N_PER_AXIS": "8",
        "FSTAT_COMB_NSKY_MAX": "32",
        "FSTAT_COMB_NSKY_MIN": "8",
        "FSTAT_PEAKS_PER_BAND": "5",
        "FSTAT_CKPT_SECS": "15",   # so checkpointing actually fires
        # --- verification ---
        "GB_DEBUG": "1",
        "VGB_DEBUG": "1",
        # ONE band: unset renders a WDM tile per source per block.
        "GB_DEBUG_PLOT_BAND": "1",
        "GB_DEBUG_PLOT_WALKER": "0",
    }
    for k, v in smoke.items():
        os.environ.setdefault(k, v)


from lisatools.globalfit.moves.globalfitmove import Move  # noqa: E402


class JointMaxLogLSearch(Move):
    """A set of stock moves converging under ONE max-logl criterion.

    ``PSDMove(max_logl_mode=True)`` converges each noise branch separately,
    to its own plateau. psd and galfor are two parameterizations of the same
    noise model -- galfor moves change the residual the psd move is fitting
    and vice versa -- so maximizing them independently can stall each where
    the pair could still climb together. This promotes the criterion to span
    both.

    Wraps the ``*_pe`` moves DELIBERATELY: ``max_logl_mode`` is consulted in
    exactly one place (``psdmove.py:918``), choosing between the plateau loop
    and a single ``run_move_for_loop``. So the pe move IS the search move
    minus its private loop -- the right inner behaviour here. Wrapping the
    ``*_search`` moves would nest two plateau loops.

    This is also what makes ``Stage(kind="search")`` correct for these
    stages: ``SearchRecipeStep`` reports done on its FIRST check, because the
    stopping criterion is supposed to live INSIDE the move. A stage that
    listed the moves side by side would run one pass and advance -- the
    criterion has to span the whole stage, which is what this object is.

    Module level, not a closure: the pre-build fit must pickle/deepcopy
    (LISA Analysis Tools-wide rule), so a local class would break it.
    """

    def __init__(self, name, inner_names, iters_per_step=None,
                 num_checks=None, **kwargs):
        super().__init__(name, **kwargs)
        self.inner_names = list(inner_names)
        # Per-propose inner-iteration cap and plateau length handed to
        # MaxLogLCombineMove. None = the global knobs (MAXLOGL_ITERS_PER_STEP
        # / NOISE_SEARCH_CHECKS, the standalone noise stages); the gb_search
        # rider overrides both (2026-09-11).
        self.iters_per_step = iters_per_step
        self.num_checks = num_checks

    def stock_dependencies(self):
        """The stock moves this wraps -- without this they are never BUILT.

        Variant setup functions construct only the stock moves the recipe
        asks for (``recipe.stock_names()``). This move resolves by its own
        ``setup``, so its name tells the builder nothing about the moves it
        composes; under STAGE_NOISE_ONLY, where these are the only moves in
        the recipe, that left ``ctx.stock_moves`` completely empty.
        """
        return list(self.inner_names)

    def setup(self, ctx):
        from lisatools.globalfit.moves.globalfitmove import MaxLogLCombineMove

        missing = [n for n in self.inner_names if n not in ctx.stock_moves]
        if missing:
            raise ValueError(
                f"{self.name}: no stock move(s) {missing} (available: "
                f"{sorted(ctx.stock_moves)})."
            )
        mv = MaxLogLCombineMove(
            [ctx.stock_moves[n] for n in self.inner_names],
            num_checks=(
                int(os.environ.get("NOISE_SEARCH_CHECKS", "5"))
                if self.num_checks is None else int(self.num_checks)
            ),
            share_temperature_control=False,
            iters_per_step=self.iters_per_step,
        )
        mv.gf_move_name = self.name
        return mv


# mbh/emri/sobbh arming (campaign S6): branch -> (env, mojito class), in the
# sobbh -> mbh -> emri BANKING order (user ruling 2026-08-27: the cheap
# branch banks its progress before the minutes-per-leaf MBH/EMRI proposals
# put the iteration at risk; move order is not a sampling statement).
_SOURCE_BRANCH_ENVS = (
    ("sobbh", "SOBHB_IDS", "SOBHB"),
    ("mbh", "MBHB_IDS", "MBHB"),
    ("emri", "EMRI_IDS", "EMRI"),
)


def _source_ids_from_env() -> dict:
    """Armed source branches: {branch: [ids]} from MBHB_IDS/EMRI_IDS/SOBHB_IDS.

    Absent or empty env = branch NOT armed (it is removed, today's
    behavior). Mirrors source_runtime.default_source_ids' parsing.
    """
    armed = {}
    for branch, env, _cls in _SOURCE_BRANCH_ENVS:
        raw = os.environ.get(env, "")
        ids = [int(x) for x in raw.split(",") if x.strip() != ""]
        if ids:
            armed[branch] = ids
    return armed


#: THE v9 SEARCH STAGE TABLE -- ``(stage name, profile, samples noise)``.
#:
#: User spec 2026-09-24. One ``gb_search`` stage becomes three that differ in
#: exactly four things, and this tuple is the single place all four are
#: stated, so the GB-only composition and the full one cannot drift:
#:
#:   * ``phase_maximize`` -- the RJ BIRTH phase-maximization heuristic.
#:     On in stage 1 (hunting: take the credit), off afterwards.
#:   * ``opt_snr`` -- the GB SNR prior boundary. 8 in stage 1 ("SNR 5 = noise,
#:     keep 8" -- a boundary of 5 measurably ballooned the hot ladder with
#:     noise births), 5 once the loud population is assembled and the faint
#:     tail must become reachable.
#:   * ``peak_min_snr`` -- the F-stat peak-selection floor. Moves with the
#:     opt-SNR boundary; 6.25 is the landed 2026-09-23 knee value.
#:   * samples noise -- whether the psd/galfor moves are in the stage at all.
#:
#: ⚠ ``peak_min_snr`` is consumed when the F-stat grid is FITTED and stamped
#: into the stage-B cache, whose loader refuses a mismatch. The stage entry
#: therefore sets an in-code override AND forces a fresh-epoch refit; see
#: ``recipe.SearchStageProfileStep``. Nothing on disk is deleted.
#: ``full_pe``'s F-stat peak floor, declared for the same reason the search
#: stages declare theirs: the override the search stages install is
#: process-global, so an undeclared full_pe would inherit gb_search_3's value
#: silently. 6.25 is what the design wants here (an assembled model's faint
#: tail must stay reachable) -- this makes choosing it explicit. ``None``
#: would clear the override and hand the stage back to FSTAT_PEAK_MIN_SNR.
_PE_PEAK_MIN_SNR = 6.25

V9_SEARCH_STAGE_PROFILES = (
    ("gb_search_1",
     dict(phase_maximize=True, opt_snr=8.0, peak_min_snr=8.0), False),
    # ⚠ STAGE 2 DOES NOT PHASE-MAXIMIZE, and the reason is the row itself.
    # It is the stage that DROPS the floors -- opt SNR 8 -> 5, F-stat peak
    # 8 -> 6.25 -- so it is already reaching for weaker, more marginal
    # sources. Maximizing the phase on top of that would add a second
    # optimistic bias to exactly the population least able to afford one:
    # a maximized delta is an upper bound on what the source can pay, and
    # near the floor that is the difference between a real detection and
    # a noise peak (user ruling 2026-09-25, correcting the same day's
    # "1 and 2" to stage 1 only). Only stage 1 -- the high-floor, high-
    # confidence pass -- maximizes.
    ("gb_search_2",
     dict(phase_maximize=False, opt_snr=5.0, peak_min_snr=6.25), False),
    ("gb_search_3",
     dict(phase_maximize=False, opt_snr=5.0, peak_min_snr=6.25), True),
)


def _v9_search_enabled() -> bool:
    """Is the v9 three-stage GB search on? ``STAGE_V9_SEARCH=0`` restores the
    single legacy ``gb_search`` stage, bit-identically."""
    return os.environ.get("STAGE_V9_SEARCH", "1").strip() not in (
        "0", "false", "False", "off")


def build_fit():
    from lisatools.globalfit.recipe import Move, Recipe, Stage
    from lisatools.globalfit.stock import erebor

    nwalkers = int(os.environ.get("NWALKERS", "16"))
    gb_only = _env_flag("GB_ONLY")
    armed_sources = _source_ids_from_env()
    if gb_only and armed_sources:
        raise ValueError(
            f"GB_ONLY=1 is the gb-only composition; source id envs "
            f"{sorted(armed_sources)} cannot be armed with it."
        )
    # GB_ONLY: the gb_no_fg stock variant IS the GB-only design (no psd /
    # galfor / vgb branch anywhere in it; fixed injection PSD via its
    # adjust_general -> fixed_psd_kwargs -> setup_acs no-psd-branch path).
    # all_sources minus psd would NOT be equivalent: it never fills
    # fixed_psd_params, so the engine would silently fall back to the
    # hardcoded stock levels (engine.py fixed_psd_kwargs default).
    fit = erebor.gb_no_fg(nwalkers=nwalkers) if gb_only \
        else erebor.all_sources(nwalkers=nwalkers)

    def warm():
        # rj_warm_search (workstream B; user ruling 2026-08-24: "BEFORE the
        # fstat proposal"): warm-start births from the previous run's
        # clustered posterior components (GB_WARM_START_COMPONENTS npz;
        # build_gb_moves constructs the stock move only when the knob is
        # set, so the descriptor must be knob-conditional too). Fresh Move
        # descriptor per call (never share one instance across stages).
        return ([Move("rj_warm_search", branch="gb")]
                if os.environ.get("GB_WARM_START_COMPONENTS", "").strip()
                else [])

    def warm_pe():
        # rj_warm_pe (user ruling 2026-09-07: the PE twin, mirroring the
        # fstat search <-> pe differences): warm-start births in the PE
        # cycle, IMMEDIATELY BEFORE rj_fstat_pe -- the same relative
        # position the search twin takes vs rj_fstat_search. Same knob
        # arms both twins; fresh Move descriptor per call.
        return ([Move("rj_warm_pe", branch="gb")]
                if os.environ.get("GB_WARM_START_COMPONENTS", "").strip()
                else [])

    def replace():
        # rj_replace (2026-08-24 exact-MH reinstatement, user directive
        # "after the rj move"): the F-stat REPLACEMENT move IMMEDIATELY
        # AFTER rj_fstat_search in the gb_search cycle. Knob-conditional
        # like warm()/ridge() -- build_gb_moves only constructs the stock
        # move when gb.search_rj_replace is on (same env, same default 1),
        # and a listed-but-unbuilt move fails recipe materialization.
        # Listing it here (rather than relying on the setup_gb_moves
        # auto-insertion, which would add it anyway) keeps the composed
        # stage printout truthful. Fresh Move descriptor per call.
        return ([Move("rj_replace", branch="gb")]
                if os.environ.get("GB_SEARCH_RJ_REPLACE", "1").strip()
                in ("1", "true", "True", "yes", "on")
                else [])

    # TOBS_TARGET honored (2026-08-13): all_sources pins a FIXED WDM grid
    # (legacy nf/nt override, mojito-adjusted to 1440x2160 = 90 d) which by
    # the settings contract BEATS general.tobs_target -- so an explicit
    # TOBS_TARGET was silently ignored (the 23-mo shakedown ran 90 d). When
    # the env asks for a Tobs, clear the fixed grid so the build derives
    # (Nf, Nt) from tobs_target + the wavelet-duration bounds -- the same
    # machinery that yields 1440x2160 at the 90-d default. Unset env ->
    # behavior unchanged.
    if os.environ.get("TOBS_TARGET", "").strip():
        fit.general.nf = None
        fit.general.nt = None
        print(
            f"[combined] TOBS_TARGET={fit.general.tobs_target:.6g} s: "
            "cleared any fixed-grid (nf, nt) override (all_sources pins "
            "one; gb_no_fg's is already None -- no-op there).",
            flush=True,
        )

    # REMOVE_BRANCHES (user ask 2026-09-11: "run for everything except the
    # GBs and galfor. We will not inject them."): comma list of whole
    # branches to drop -- from the fit, from every stage's move list, and
    # from the DEFAULT injection streams below. Only the branches the stage
    # lists know how to drop are accepted. With "gb" removed there is no
    # F-stat machinery and no RJ: the gb stages collapse to ONE full_pe over
    # what remains.
    #
    # "psd" became removable 2026-09-14 evening (the TRUTH-INJECTION NULL
    # TEST ruling: "remove the psd fitting, use best fit values for psd and
    # galfor from the 3mo run"). Removing it drops the joint noise criteria
    # (JointMaxLogLSearch branch="psd") and both noise stages, and hands the
    # likelihood to run.py setup_acs's FIXED-sensitivity path
    # (``sensitivity_backend(..., **general.fixed_psd_kwargs)``) -- see the
    # PSD_FIXED_PARAMS / GALFOR_FIXED_PARAMS block below, which is the only
    # way the 3mo best fit reaches that path. Guarded hard: gb and galfor
    # machinery both assume a SAMPLED psd (the GB band engine reads the
    # per-walker linear_psd_arr the psd branch drives; galfor's foreground is
    # a component OF the psd branch's sensitivity), so psd can only go once
    # those are already gone.
    _removable = ("gb", "galfor", "vgb", "psd")
    remove_branches = tuple(
        b.strip().lower()
        for b in os.environ.get("REMOVE_BRANCHES", "").split(",")
        if b.strip()
    )
    for _b in remove_branches:
        if _b not in _removable:
            raise ValueError(
                f"REMOVE_BRANCHES entry {_b!r} is not removable here "
                f"(supported: {', '.join(_removable)})."
            )
    if gb_only and remove_branches:
        raise ValueError(
            "GB_ONLY=1 is already the gb-only composition; "
            "REMOVE_BRANCHES makes no sense with it."
        )
    if "psd" in remove_branches:
        # Both of these are silent-wrongness traps rather than crashes, so
        # refuse at COMPOSITION time (seconds) instead of discovering it
        # from a null that never reaches zero after hours of allocation.
        for _need in ("galfor", "gb"):
            if _need not in remove_branches:
                raise ValueError(
                    f"REMOVE_BRANCHES removes 'psd' but keeps {_need!r}: "
                    f"the {_need} machinery samples against the psd "
                    "branch's sensitivity (galfor's foreground is a "
                    "component of it; the GB band engine reads the "
                    "per-walker linear_psd_arr it drives). Remove "
                    f"{_need!r} too, or keep psd."
                )
        # "NO INJECTED NOISE" (ruling 2026-09-14): with no psd branch there
        # is nothing to fit a noise realization with, so an injected one
        # sits unmodelled in the residual and the truth null cannot reach
        # zero. all_sources defaults add_instrument_noise=True (auto), so
        # this is the easy way to get a quietly-wrong null.
        if fit.general.add_instrument_noise:
            raise ValueError(
                "REMOVE_BRANCHES removes 'psd' but "
                f"add_instrument_noise={fit.general.add_instrument_noise!r}: "
                "an injected noise realization would sit unmodelled in the "
                "residual (no psd branch fits it), so a truth-injection "
                "null could not reach zero. Export ADD_INSTRUMENT_NOISE=0."
            )
        # The noise-stage flags have no stage to act on any more (same
        # reasoning as the GB_ONLY block below).
        for _flag in ("STAGE_NOISE_ONLY", "STAGE_NOISE_VGB_PE",
                      "STAGE_SKIP_NOISE"):
            if _env_flag(_flag):
                raise ValueError(
                    f"REMOVE_BRANCHES removes 'psd', so there are no noise "
                    f"stages; {_flag}=1 makes no sense here."
                )
        # UNEQUAL_ARM swaps the PSD BRANCH's instrument component
        # (all_sources._wire_unequal_arm raises "unequal_arm=1 requires the
        # psd branch"), so the fixed-sensitivity path runs the plain
        # analytic equal-arm model. Caught here rather than at build.
        if fit.general.unequal_arm:
            raise ValueError(
                "REMOVE_BRANCHES removes 'psd' but UNEQUAL_ARM=1: the "
                "unequal-arm model is installed ON the psd branch's "
                "instrument component, which no longer exists. Export "
                "UNEQUAL_ARM=0 (at the null point the residual cancels, so "
                "the PSD weighting only scales small deviations)."
            )

    # Every sampled branch needs a stream: NOISE for psd/galfor, GB, VGB --
    # plus the armed source classes' streams (their data must contain the
    # signals the branches fit). Explicit SOURCE_TYPES always wins.
    # GB_ONLY keeps the same default DELIBERATELY: the data is the SAME full
    # injection (noise + GB galaxy + VGBs) as the production runs -- only the
    # SAMPLED branch set shrinks to gb. Unmodeled content stays in the
    # residual; that is the accepted trade for not waiting on a noise fit.
    # REMOVE_BRANCHES is the opposite contract ("we will not inject them"):
    # a removed gb/vgb also leaves the DEFAULT stream list. A removed PSD
    # takes the NOISE stream with it -- nothing models a noise realization
    # any more, so injecting one would leave it in the residual and the
    # truth-injection null could never reach zero ("NO INJECTED NOISE").
    _default_src = ",".join(
        (["NOISE"] if "psd" not in remove_branches else [])
        + (["GB"] if "gb" not in remove_branches else [])
        + (["VGB"] if "vgb" not in remove_branches else [])
    ) + "".join(
        f",{cls}" for br, _env, cls in _SOURCE_BRANCH_ENVS
        if br in armed_sources
    )
    src = os.environ.get("SOURCE_TYPES", _default_src)
    fit.general.source_types = tuple(
        s.strip().upper() for s in src.split(",") if s.strip()
    )

    # gb_no_fg carries ONLY the gb branch; remove_branch raises on absent
    # names, so guard. Unarmed source branches drop (the pre-S6 4-branch
    # behavior); armed ones STAY and get their mojito catalogue id lists
    # (mojito_source_ids gates inclusion; GB/VGB entries preserved).
    for branch in ("mbh", "emri", "sobbh"):
        if branch in fit.branches and branch not in armed_sources:
            fit.remove_branch(branch)
    if armed_sources:
        _ids = dict(fit.general.mojito_source_ids)
        for branch, _env, cls in _SOURCE_BRANCH_ENVS:
            if branch in armed_sources:
                _ids[cls] = list(armed_sources[branch])
        fit.general.mojito_source_ids = _ids
        print(
            f"[combined] source branches armed: "
            + ", ".join(f"{br}={armed_sources[br]}"
                        for br, _e, _c in _SOURCE_BRANCH_ENVS
                        if br in armed_sources),
            flush=True,
        )

    for _b in remove_branches:
        if _b in fit.branches:
            fit.remove_branch(_b)
    if remove_branches:
        print(f"[combined] branches REMOVED (not sampled, not in the "
              f"default injection): {list(remove_branches)}", flush=True)

    # FIXED SENSITIVITY (psd removed): PSD_FIXED_PARAMS / GALFOR_FIXED_PARAMS
    # -> general.fixed_psd_kwargs, the ONE thing run.py setup_acs's
    # no-psd-branch path reads (``sensitivity_backend(f"walker_{w}",
    # **general_info.fixed_psd_kwargs)``, run.py:1515-1517).
    #
    # BASIS -- READ THIS BEFORE SETTING THEM. The fixed path applies NO
    # transform: whatever is in the dict goes straight to the backend as
    # ``psd_params`` / ``galfor_params``, so both must already be in the
    # PHYSICAL basis the backend documents -- psd = [Soms_d, Sa_a] as LINEAR
    # (square-root) values, galfor = the 5-vector (amp, fk, alpha, f_1, f_2)
    # linear. (The SAMPLED path is the one that has to call
    # ``both_transforms`` first, run.py:1472-1486; every stock psd/galfor
    # transform is None at the default PSD_LOG_SAMPLING=0 /
    # GALFOR_LOG_SAMPLING=0, which is why the 3mo store's raw chain
    # coordinates ARE physical and can be used verbatim.)
    #
    # The submit script fills these from the 3mo store's best-logL cold
    # walker via ``lisatools.globalfit.warmstart.opt_snr.best_logl_noise``,
    # which returns exactly those raw chain rows (no transform) -- so the
    # two ends agree by construction. If a future run is launched with
    # PSD_LOG_SAMPLING=1 / GALFOR_LOG_SAMPLING=1, that extraction returns
    # LOG values and this wiring would silently under-weight the noise.
    _psd_fixed = os.environ.get("PSD_FIXED_PARAMS", "").strip()
    _gal_fixed = os.environ.get("GALFOR_FIXED_PARAMS", "").strip()
    if _psd_fixed or _gal_fixed:
        import numpy as np

        def _floats(raw, name):
            try:
                return np.array([float(x) for x in raw.split(",")
                                 if x.strip()])
            except ValueError as exc:
                raise ValueError(
                    f"{name} must be a comma list of floats, got {raw!r}."
                ) from exc

        _kw = dict(fit.general.fixed_psd_kwargs or {})
        if _psd_fixed:
            _kw["psd_params"] = _floats(_psd_fixed, "PSD_FIXED_PARAMS")
        if _gal_fixed:
            _kw["galfor_params"] = _floats(_gal_fixed, "GALFOR_FIXED_PARAMS")
        _kw.setdefault("galfor_params", None)
        fit.general.fixed_psd_kwargs = _kw
        print(f"[combined] FIXED sensitivity (physical basis): "
              f"psd_params={_kw.get('psd_params')} "
              f"galfor_params={_kw.get('galfor_params')}", flush=True)
        # Which of the two actually reaches the likelihood depends on which
        # BRANCHES are sampled, and the two answers differ -- say both out
        # loud rather than leaving a pin to look live when it is not.
        if "psd" in fit.branches and _psd_fixed:
            print("[combined] WARNING: PSD_FIXED_PARAMS set while the psd "
                  "branch is SAMPLED -- setup_acs takes the sampled "
                  "coordinates and IGNORES psd_params.", flush=True)
        if _gal_fixed:
            if "galfor" in fit.branches:
                print("[combined] WARNING: GALFOR_FIXED_PARAMS set while the "
                      "galfor branch is SAMPLED -- setup_acs takes the "
                      "sampled coordinates and IGNORES galfor_params. Add "
                      "galfor to REMOVE_BRANCHES to pin it.", flush=True)
            else:
                print("[combined] galfor branch REMOVED and galfor_params "
                      "PINNED -- the foreground is held at the value above "
                      "while psd stays sampled (run.py setup_acs). This is "
                      "the loop-breaker configuration: galfor can no longer "
                      "absorb unresolved GB and raise the search's own noise "
                      "floor in the band those sources live in.", flush=True)

    if gb_only:
        # Branch-set sanity: gb only, or the composition is not what the
        # header promises (and setup_acs would sample a PSD we said is fixed).
        _extra = [b for b in fit.branches if b != "gb"]
        if _extra:
            raise ValueError(
                f"GB_ONLY=1 but the fit carries extra branches {_extra}."
            )
        if _env_flag("STAGE_NOISE_ONLY") or _env_flag("STAGE_NOISE_VGB_PE"):
            raise ValueError(
                "GB_ONLY=1 has no noise/vgb stages; STAGE_NOISE_ONLY / "
                "STAGE_NOISE_VGB_PE make no sense here."
            )
        # Same stage names / kinds / GB move stacks as the full composition
        # below (monitor + digests + resume statuses stay compatible), minus
        # every psd/galfor/vgb move. Fresh store starts directly at
        # gb_search; nothing in the package gates on the noise stage names.
        def ridge():
            # Fresh Move descriptor per stage (never share one instance).
            return ([Move("gb_ridge_gibbs", branch="gb")]
                    if os.environ.get("GB_RIDGE_GIBBS", "1") == "1" else [])

        def gb_only_in_model(slot):
            return ([Move(slot, branch="gb")]
                    if _env_flag("GB_SEARCH_IN_MODEL") else [])

        if _v9_search_enabled():
            # Same three-stage v9 cycle as the full composition below, minus
            # every psd/galfor/vgb move. Keeping the two in step matters: the
            # GB-only variant is what the probe scripts and the search-test
            # runbook drive, so a divergence here would mean the probes stop
            # testing the production cycle.
            _pe_stage = Stage(
                name="full_pe", kind="pe",
                moves=warm_pe() + [
                    Move("rj_fstat_pe", branch="gb"),
                    Move("rj_prior_pe", branch="gb"),
                ] + ridge(),
                step_kwargs=dict(peak_min_snr=_PE_PEAK_MIN_SNR,
                                 stage_name="full_pe"),
                combine_kwargs=_pe_combine_kwargs(),
            )
            _warm3 = int(os.environ.get("GB_SEARCH_3_WARM_EVERY", "5"))
            if _warm3 < 1:
                raise ValueError(
                    f"GB_SEARCH_3_WARM_EVERY={_warm3} must be >= 1.")
            _gb_only_stages = []
            for _name, _prof, _sampled in V9_SEARCH_STAGE_PROFILES:
                _every = _warm3 if _sampled else 1
                _warm = ([Move("rj_warm_search", branch="gb", every=_every)]
                         if warm() else [])
                _gb_only_stages.append(Stage(
                    name=_name, kind="gb_search",
                    moves=(_warm + gb_only_in_model("in_model")
                           + [Move("rj_fstat_search", branch="gb")]
                           + gb_only_in_model("in_model_fstat")
                           + replace()
                           + gb_only_in_model("in_model_replace")
                           + [Move("rj_prior_removal", branch="gb")]
                           + ridge()),
                    step_kwargs=dict(
                        plateau_branch="gb",
                        convergence_iter=int(
                            os.environ.get("GB_PLATEAU_ITERS", "5")),
                        stage_name=_name, profile=dict(_prof),
                    ),
                    combine_kwargs=dict(share_temperature_control=False),
                ))
            fit.recipe = Recipe(_gb_only_stages + [_pe_stage])
            return fit

        fit.recipe = Recipe([
            Stage(
                name="gb_search", kind="rj",
                # rj_warm_search (when armed) runs IMMEDIATELY BEFORE
                # rj_fstat_search -- warm-start births first, then the
                # F-stat grid births (user ruling 2026-08-24); rj_replace
                # (default on) IMMEDIATELY AFTER them -- the exact-MH
                # F-stat replacement pass before the removal judge.
                moves=warm() + [
                    Move("rj_fstat_search", branch="gb"),
                ] + replace() + [
                    Move("rj_prior_removal", branch="gb"),
                ] + ridge(),
                step_kwargs=dict(
                    plateau_branch="gb",
                    convergence_iter=int(
                        os.environ.get("GB_PLATEAU_ITERS", "5")
                    ),
                ),
                combine_kwargs=dict(share_temperature_control=False),
            ),
            Stage(
                name="full_pe", kind="pe",
                # rj_warm_pe (when armed) runs IMMEDIATELY BEFORE
                # rj_fstat_pe -- the PE mirror of the search-stage order.
                moves=warm_pe() + [
                    Move("rj_fstat_pe", branch="gb"),
                    Move("rj_prior_pe", branch="gb"),
                ] + ridge(),
                combine_kwargs=_pe_combine_kwargs(),
            ),
        ])
        return fit

    # GB move names (2026-08-12 rename, user ruling):
    #   rj_fstat_search  = F-stat grid births, search config (cap updater)
    #   rj_prior_removal = removal-only prior pruning (search cycle)
    #   rj_fstat_pe      = F-stat grid births, strict-PE config
    #   rj_prior_pe      = pure prior births, strict-PE config
    # NOTE ON NAMING (2026-09-08). The noise/VGB STAGES are called "*_search"
    # but their sub-moves are the "*_pe" ones, and that is deliberate, not a
    # leftover: psd (2 params), galfor (5) and vgb (55 KNOWN sources) are all
    # FIXED-DIMENSIONAL with nothing to search FOR. In this codebase a
    # "_search" proposal means maximize-and-skip-detailed-balance, which is
    # the right tool for finding unknown GBs and the wrong one for a
    # 2-parameter noise level. "search" in the STAGE name refers to the
    # stage's role in the ladder -- a convergence-gated burn-in -- and that
    # is what JointMaxLogLSearch supplies: a joint max-logL criterion
    # spanning all its sub-moves, so the stage advances when they have
    # JOINTLY plateaued. Hence `[GF_TIMING] stage=..._search move=psd_pe`,
    # which reads oddly but is correct.
    #
    # (A `noise_search = [Move("psd_search"), Move("galfor_search")]` list
    # used to sit here and was referenced by no stage -- removed 2026-09-08.
    # `setup_recipe` still BUILDS psd_search/galfor_search stock moves for
    # every present noise branch; this driver simply never requests them.)
    # Branch-aware lists (REMOVE_BRANCHES): a removed galfor/vgb drops out
    # of every move list and joint criterion below with no other change.
    _has_galfor = "galfor" in fit.branches
    _has_vgb = "vgb" in fit.branches
    # psd too (2026-09-14): with it removed there is no noise move at all and
    # no noise STAGE -- the sensitivity is fixed, so there is nothing to
    # converge. Same branch-aware pattern as galfor/vgb.
    _has_psd = "psd" in fit.branches
    noise_pe = (
        ([Move("psd_pe", branch="psd")] if _has_psd else [])
        + ([Move("galfor_pe", branch="galfor")] if _has_galfor else []))
    # VGBs are KNOWN sources: fixed-dimensional, no RJ, nothing to search
    # for. They sample from the first stage onward so their power is being
    # fitted while the noise converges, rather than sitting in the residual
    # and biasing the PSD.
    vgb = [Move("vgb_pe", branch="vgb")] if _has_vgb else []
    # VGB ridge-Gibbs, the GB twin (user ruling 2026-09-16, "VGBs get the
    # ridge-gibbs fiber move too"). ``build_vgb_moves`` registers
    # ``vgb_ridge_gibbs`` ONLY when the vgb basis carries dist/Mc/
    # fdot_astro_ratio -- i.e. under the 6-column VGB_CHIRP_MASS_BASIS=1
    # chirp basis -- so the stage must request it only there, or Move.setup
    # raises on the missing stock name. Same GB_RIDGE_GIBBS kill switch as
    # the GB one, and the same stage placement gb_ridge_gibbs has: BOTH
    # gb_search and the PE stages. It rides gb_search as its own Move (not
    # inside the noise_vgb joint-search CRITERION) because it is exactly
    # likelihood-invariant -- zero likelihood calls, prior x fiber-measure
    # MH only -- so it cannot perturb a max-logL plateau test, while
    # WITHOUT it the VGB chirp masses would sit frozen along the fiber for
    # the whole search stage, which is the disease the move exists to cure.
    _vgb_ridge_on = (
        _has_vgb
        and _env_flag("VGB_CHIRP_MASS_BASIS")
        and _env_flag("GB_RIDGE_GIBBS", "1")
    )

    def vgb_ridge():
        """Fresh Move descriptor per stage (never share one instance)."""
        return [Move("vgb_ridge_gibbs", branch="vgb")] if _vgb_ridge_on else []

    _noise_names = (["psd_pe"] if _has_psd else []) + (
        ["galfor_pe"] if _has_galfor else [])

    # Stage 1: noise alone. Stage 2 and the GB search: noise + VGBs, with the
    # max-logl criterion spanning ALL of them -- one object per stage, so the
    # convergence is joint rather than each move plateauing separately.

    # TODO: CLEAN
    noise_only = [JointMaxLogLSearch(
        "noise_joint_search", list(_noise_names), branch="psd")]
    noise_only_1 = [JointMaxLogLSearch(
        "noise_joint_search_1", list(_noise_names), branch="psd")]
    noise_only_2 = [JointMaxLogLSearch(
        "noise_joint_search_2", list(_noise_names), branch="psd")]
    noise_only_3 = [JointMaxLogLSearch(
        "noise_joint_search_3", list(_noise_names), branch="psd")]
    noise_vgb = [JointMaxLogLSearch(
        "noise_vgb_joint_search",
        _noise_names + (["vgb_pe"] if _has_vgb else []),
        branch="psd")]
    # The SAME joint move riding inside gb_search, with its OWN plateau rule.
    # It never plateaus for good there -- the GB residual moves every
    # iteration -- so under NOISE_SEARCH_CHECKS=5 it took ~6 rounds of 16 s
    # per GB iteration (93 s, 15% of the iteration, 3mo job 473). The
    # [MAXLOGL] trace shows the re-tracking happens in ROUND 1 (the
    # IMPROVED jumps of hundreds of lnL land there); rounds 2+ add ~5 lnL
    # per round, the tol-level wobble of a stretch ensemble near the mode.
    # GB_SEARCH_NOISE_CHECKS (default 1): keep taking rounds while a round
    # still improves by more than tol, stop at the first flat one -- so
    # after a big residual change the noise gets as many rounds as it
    # needs, and otherwise ~2. GB_SEARCH_NOISE_ITERS_PER_STEP (default 0 =
    # the global MAXLOGL_ITERS_PER_STEP ceiling) is only a hard cap. The
    # standalone noise stages above are untouched.
    _gb_noise_checks = int(os.environ.get("GB_SEARCH_NOISE_CHECKS", "1"))
    _gb_noise_cap = int(os.environ.get("GB_SEARCH_NOISE_ITERS_PER_STEP", "0"))
    noise_vgb_gb = [JointMaxLogLSearch(
        "noise_vgb_joint_search",
        _noise_names + (["vgb_pe"] if _has_vgb else []),
        branch="psd", num_checks=(_gb_noise_checks or None),
        iters_per_step=(_gb_noise_cap or None))]

    # gb_search source cadence (user ruling 2026-09-15: "run them in
    # gb_search as before, but make them run every 10 iterations"): the
    # mbh/emri dense rows cost minutes per pass at near-zero GPU, so in
    # gb_search they propose on every Nth stage iteration (stage-local —
    # full_pe runs them every iteration). Between cadence hits they stay
    # SUBTRACTED at their current coords exactly as before.
    #
    # SOBBH JOINED THE CADENCE (user ruling 2026-09-18), superseding the
    # 09-15 "sobbh's cheap chunked-het rows ride every iteration". They are
    # not cheap: the 4-GPU run measured sobbh_pe at 250 s per propose over
    # 47 calls (min 246, max 337) against mbh 147 s and emri 164 s on ~1 in
    # 4 iterations, i.e. sobbh alone was 43% of a 9.2-min iteration and the
    # single largest cost in gb_search. Its chunked-het scorer bills a FLAT
    # 1.73 s per CALL whatever the batch shape (see
    # sobbhspecialmove._LL_SPAN_KEYS' header), so the only lever on it is
    # how many calls run: num_repeats (20 -> 10 the same day) and this
    # cadence. full_pe is untouched — all three run every iteration there.
    _gb_search_src_every = int(os.environ.get("GB_SEARCH_SOURCE_EVERY", "5"))
    if _gb_search_src_every < 1:
        raise ValueError(
            f"GB_SEARCH_SOURCE_EVERY={_gb_search_src_every} must be >= 1.")

    #: Branches the gb_search cadence applies to. All three armed source
    #: branches as of the 2026-09-18 ruling; kept as a named set so the
    #: 09-15 behaviour (mbh/emri only) is one edit away.
    _GB_SEARCH_CADENCED = ("sobbh", "mbh", "emri")

    def source_pe(gb_search_cadence=False):
        # Fresh Move descriptors per stage (never share one instance):
        # the armed source PE moves, sobbh -> mbh -> emri (banking order).
        # ``gb_search_cadence`` puts them on the 1-in-N schedule.
        def _every(br):
            return (_gb_search_src_every
                    if gb_search_cadence and br in _GB_SEARCH_CADENCED else 1)
        return [Move(f"{br}_pe", branch=br, every=_every(br))
                for br, _env, _cls in _SOURCE_BRANCH_ENVS
                if br in armed_sources]

    stages = []
    _skip_src_search = _env_flag("STAGE_SKIP_SOURCE_SEARCH")
    if _skip_src_search and not armed_sources:
        raise ValueError(
            "STAGE_SKIP_SOURCE_SEARCH=1 but no source branches are armed "
            "(MBHB_IDS/EMRI_IDS/SOBHB_IDS empty) -- there is no "
            "source_search stage to skip."
        )
    if armed_sources and _skip_src_search:
        # User ruling 2026-09-14 late: with exact-truth starts
        # (*_START_FACTOR=0, valid now that the source inner moves are
        # eigen rather than stretch and need no ensemble spread) there is
        # nothing for the joint source search to converge. The sources
        # stay SUBTRACTED the whole time exactly as before -- their
        # templates enter the residual at setup_acs from the state
        # coords -- and their PE proposals run only where they already
        # ride: gb_search and full_pe.
        print("[combined] STAGE_SKIP_SOURCE_SEARCH=1: no source_search "
              "stage; sources subtracted at their start coords through "
              "the noise stages, proposals armed in gb_search + full_pe.",
              flush=True)
    if armed_sources and not _skip_src_search:
        # SOURCE SEARCH FIRST (campaign S6: "MBH search first, full_year
        # pattern"): the armed source branches converge under ONE joint
        # max-lnL criterion before anything else runs, so their (loud)
        # power is out of the residual before the noise stages fit the
        # PSD. They start near truth (*_START_FACTOR seeding), so this is
        # a refinement loop, not a blind search. Afterwards they keep
        # PE-sampling through gb_search and full_pe (live residual); they
        # sit converged-and-subtracted during the noise stages.
        _src_branch = next(br for br, _e, _c in _SOURCE_BRANCH_ENVS
                           if br in armed_sources)
        stages.append(Stage(
            name="source_search", kind="search",
            moves=[JointMaxLogLSearch(
                "source_joint_search",
                [f"{br}_pe" for br, _e, _c in _SOURCE_BRANCH_ENVS
                 if br in armed_sources],
                branch=_src_branch)],
            combine_kwargs=dict(share_temperature_control=False),
        ))
    # CONDITIONAL NOISE STAGES (v9, user spec 2026-09-24): "noise_search /
    # noise_vgb_search run ONLY when there is no psd/foreground estimate from
    # a previous run." The estimate IS the start pin -- {PSD,GALFOR}_START_
    # PARAMS, which the submit script fills from the previous run's maxlogL
    # point, the same folder the warm start comes from. With every sampled
    # noise branch pinned there is nothing for a convergence-gated noise
    # burn-in to find; without one they run exactly as before.
    #
    # ⚠ The gate requires a pin for EVERY sampled noise branch. A half-pinned
    # start (psd seeded, galfor prior-drawn) with the noise stages skipped
    # would enter gb_search_1 -- which does not sample noise at all -- with a
    # random foreground, and stay there for two whole stages.
    _noise_pinned = bool(os.environ.get("PSD_START_PARAMS", "").strip()) and (
        not _has_galfor
        or bool(os.environ.get("GALFOR_START_PARAMS", "").strip()))
    _force_noise = _env_flag("STAGE_FORCE_NOISE_SEARCH")
    if _v9_search_enabled() and _noise_pinned and not _force_noise:
        print(
            "[combined] v9: SKIPPING noise_search / noise_vgb_search -- the "
            "noise model is PINNED at a previous run's estimate "
            "(PSD_START_PARAMS"
            + (" + GALFOR_START_PARAMS" if _has_galfor else "")
            + "). Stages gb_search_1/2 hold it there; gb_search_3 releases "
              "it. Export STAGE_FORCE_NOISE_SEARCH=1 to run them anyway.",
            flush=True,
        )
    elif _has_psd and not _env_flag("STAGE_SKIP_NOISE"):
        if _v9_search_enabled() and not _noise_pinned:
            print(
                "[combined] v9: noise_search / noise_vgb_search KEPT -- no "
                "previous-run noise estimate was supplied "
                "(PSD_START_PARAMS / GALFOR_START_PARAMS unset), so the psd "
                "and foreground start from a PRIOR DRAW and must be fitted "
                "before the fixed-noise search stages freeze them.",
                flush=True,
            )
        stages.append(Stage(
            name="noise_search", kind="search", moves=noise_only,
            combine_kwargs=dict(share_temperature_control=False),
        ))
        if _has_vgb:
            # without a vgb branch this stage would duplicate noise_search
            stages.append(Stage(
                name="noise_vgb_search", kind="search", moves=noise_vgb,
                combine_kwargs=dict(share_temperature_control=False),
            ))
    if _env_flag("STAGE_NOISE_ONLY"):
        # Stages 1-2 only: watch the joint psd+galfor search converge without
        # paying for the F-stat grid fit (which lives in the gb_search RJ
        # birth move's setup) or any GB work. The noise stages run FIRST in
        # the full recipe too, so nothing here changes their behaviour -- it
        # just stops afterwards.
        if not stages:
            raise ValueError(
                "STAGE_NOISE_ONLY=1 with STAGE_SKIP_NOISE=1 leaves no stages."
            )
        fit.recipe = Recipe(stages)
        return fit

    if _env_flag("STAGE_NOISE_VGB_PE"):
        # Searches, then PE-sample psd+galfor+vgb — the gate between "the
        # noise search converged" and "turn on the GB machinery": posterior
        # sampling of every non-GB branch, no F-stat fit, no RJ. PE never
        # stops on its own, so NUM_ITERATIONS bounds the run.
        stages.append(Stage(
            name="noise_vgb_pe", kind="pe",
            moves=noise_pe + vgb + vgb_ridge(),
            combine_kwargs=_pe_combine_kwargs(),
        ))
        fit.recipe = Recipe(stages)
        return fit

    if "gb" not in fit.branches:
        # REMOVE_BRANCHES took gb: no F-stat machinery, no RJ, nothing to
        # search for. Everything that remains PE-samples together in one
        # full_pe -- the same composition as the stage below minus every
        # gb move (name kept so monitor/digests/resume statuses read the
        # same). PE never stops on its own; NUM_ITERATIONS bounds the run.
        stages.append(Stage(
            name="full_pe", kind="pe",
            moves=noise_pe + source_pe() + vgb + vgb_ridge(),
            combine_kwargs=_pe_combine_kwargs(),
        ))
        fit.recipe = Recipe(stages)
        return fit

    # ======================================================================
    # v9: THREE GB SEARCH STAGES (user spec 2026-09-24)
    # ======================================================================
    # Design: docs/superpowers/specs/2026-09-24-v9-search-stage-restructure-
    # design.md. One ``gb_search`` stage becomes three, differing in exactly
    # four things:
    #
    #   stage           noise     phase max   opt SNR   F-stat peak
    #   gb_search_1     FIXED     on          8         8
    #   gb_search_2     FIXED     off         5         6.25
    #   gb_search_3     sampled   off         5         6.25
    #
    # and each runs the same per-iteration move cycle:
    #
    #   1  rj_warm_search      (stage 3: every GB_SEARCH_3_WARM_EVERY iters)
    #   2  in_model            (adaptive passes -- see below)
    #   3  rj_fstat_search
    #   4  in_model_fstat
    #   5  rj_replace          (two internal passes: warm, then F-stat)
    #   6  in_model_replace
    #   7  rj_prior_removal    (removal only) -- THE CYCLE ENDS HERE
    #
    # The cycle ENDS on the removal judge (user amendment 2026-09-24), so
    # every birth and every swap has had a full in-model refinement pass
    # before it is judged for death. A source still sitting at its birth or
    # swap coordinates looks far more deletable than the same source after it
    # has walked onto its peak, and the three in-model slots are each named
    # for the RJ move they polish.
    #
    # THE IN-MODEL SLOTS ARE NOT FIXED-LENGTH. Each is ONE ``propose()`` that
    # internally repeats its whole pass -- GB_NUM_REPEAT_PROPOSALS repeats per
    # source per pass, a FIXED number -- until every occupied (walker, band)
    # sub-band's cold-chain logL has plateaued, or GB_INMODEL_GROUP_MAX_PASSES
    # hits. Fixed repeats, adaptive passes (user confirmation 2026-09-24);
    # per-source totals are consequently not uniform across the proposal,
    # which is accepted in search.
    #
    # "FIXED noise" is the ABSENCE of the psd/galfor moves, not a pin: an
    # unsampled branch does not move, and setup_acs rebuilds each walker's
    # sensitivity from the state coords every pass. The VALUE it is fixed at
    # is PSD_START_PARAMS / GALFOR_START_PARAMS (run.py's noise start pin),
    # which the submit script fills from the SAME previous-run folder the warm
    # start comes from, at that run's maxlogL point.
    #
    # STAGE_V9_SEARCH=0 restores the single legacy ``gb_search`` stage.
    _v9_stages = _v9_search_enabled()

    def in_model(slot):
        """One pure in-model slot, when the move exists to resolve.

        ``build_gb_moves`` registers the three slots only under
        GB_MODE=search + GB_SEARCH_IN_MODEL=1, and a listed-but-unbuilt move
        fails recipe materialization -- so the descriptor has to be
        knob-conditional exactly like warm()/replace(). Fresh Move descriptor
        per call (never share one instance across stages).
        """
        return ([Move(slot, branch="gb")]
                if _env_flag("GB_SEARCH_IN_MODEL") else [])

    def _search_stage(name, *, sample_noise, phase_maximize, opt_snr,
                      peak_min_snr, warm_every=1):
        # SAMPLED noise: the legacy gb_search composition verbatim -- the
        # leading joint psd+galfor+vgb search plus the two extra re-tracking
        # rounds that bracket the F-stat birth move, so the grid is always
        # fitted against a current noise level. FIXED noise: the vgb branch
        # keeps sampling (it is 55 KNOWN sources, nothing to do with the
        # noise model) and the psd/galfor moves are simply absent.
        _noise = (noise_vgb_gb if sample_noise
                  else ([Move("vgb_pe", branch="vgb")] if _has_vgb else []))
        _noise_pre = noise_only_1 if sample_noise else []
        _noise_post = noise_only_2 if sample_noise else []
        _warm = ([Move("rj_warm_search", branch="gb", every=warm_every)]
                 if warm() else [])
        return Stage(
            name=name, kind="gb_search",
            moves=(
                _noise
                + source_pe(gb_search_cadence=True)
                + _warm + in_model("in_model") + _noise_pre
                + [Move("rj_fstat_search", branch="gb")]
                + _noise_post
                + in_model("in_model_fstat")
                + replace()
                + in_model("in_model_replace")
                + [Move("rj_prior_removal", branch="gb")]
                + ([Move("gb_ridge_gibbs", branch="gb")]
                   if os.environ.get("GB_RIDGE_GIBBS", "1") == "1" else [])
                + vgb_ridge()
            ),
            step_kwargs=dict(
                plateau_branch="gb",
                convergence_iter=int(os.environ.get("GB_PLATEAU_ITERS", "5")),
                stage_name=name,
                profile=dict(phase_maximize=phase_maximize, opt_snr=opt_snr,
                             peak_min_snr=peak_min_snr),
            ),
            combine_kwargs=dict(share_temperature_control=False),
        )

    if _v9_stages:
        # Warm start every Nth iteration in stage 3 only (user ruling
        # 2026-09-24). By then the previous run's posterior has been mined;
        # what remains is expensive per hit and rarely productive, so it
        # rides a cadence instead of the every-iteration schedule stages 1-2
        # give it.
        _warm3 = int(os.environ.get("GB_SEARCH_3_WARM_EVERY", "5"))
        if _warm3 < 1:
            raise ValueError(
                f"GB_SEARCH_3_WARM_EVERY={_warm3} must be >= 1.")
        stages += [
            _search_stage(_name, sample_noise=_sampled,
                          warm_every=(_warm3 if _sampled else 1), **_prof)
            for _name, _prof, _sampled in V9_SEARCH_STAGE_PROFILES
        ] + [
            Stage(
                name="full_pe", kind="pe",
                # Move list unchanged from the legacy composition below. The
                # peak floor is DECLARED rather than inherited: the search
                # stages install a process-global override of it, and
                # full_pe wants the loose 6.25 (an assembled model's faint
                # tail must stay reachable) -- but getting it by accident
                # from gb_search_3 is not the same as choosing it.
                moves=noise_pe + source_pe() + warm_pe() + [
                    Move("rj_fstat_pe", branch="gb"),
                    Move("rj_prior_pe", branch="gb"),
                ] + ([Move("gb_ridge_gibbs", branch="gb")]
                     if os.environ.get("GB_RIDGE_GIBBS", "1") == "1" else [])
                + vgb + vgb_ridge(),
                step_kwargs=dict(peak_min_snr=_PE_PEAK_MIN_SNR,
                                 stage_name="full_pe"),
                combine_kwargs=_pe_combine_kwargs(),
            ),
        ]
        fit.recipe = Recipe(stages)
        return fit

    stages += [
        Stage(
            name="gb_search", kind="rj",
            # Noise stays in SEARCH (joint max-logl) mode through the GB
            # search. Per-stage move-name uniqueness (recipe.py ebd8612) is
            # what lets these recur across stages.
            # rj_fstat_search (ex rj_prior_search), NOT rj_fstat_mcmc_search (2026-08-12): the
            # serial-MCMC move scores gb.get_fstat_ll -- an FD kernel --
            # against the parent ACA, which is WDM here (wrong domain), and
            # carries leaf_cap_update=False. rj_prior_search is the
            # GPU-verified WDM birth engine (sig-het F-stat grids via
            # route_sighet_fstat, epoch refits, D/2 caps as the DESIGNATED
            # updater, at-cap skip) with use_prior_removal built in; the
            # removal-only move follows it per the search-cycle order.
            # Re-wiring the serial MCMC onto the sig-het scorer is a
            # post-run item (its batches are not f0-ordered, so the
            # reference-block stash would thrash every step).
            # rj_warm_search (when armed) runs IMMEDIATELY BEFORE
            # rj_fstat_search -- warm-start births first, then the F-stat
            # grid births (user ruling 2026-08-24); rj_replace (default
            # on) IMMEDIATELY AFTER them -- the exact-MH F-stat
            # replacement pass before the removal judge.
            # Armed source PE moves ride between the noise joint search
            # and the GB RJ cycle (sobbh -> mbh -> emri banking order).
            # mbh/emri ride at the 1-in-N gb_search cadence (block above);
            # sobbh every iteration.
            moves=noise_vgb_gb + source_pe(gb_search_cadence=True) + warm() + noise_only_1 + [
                Move("rj_fstat_search", branch="gb"),
            ] + noise_only_2 + replace() + [
                Move("rj_prior_removal", branch="gb"),
            ] + ([Move("gb_ridge_gibbs", branch="gb")]
                 if os.environ.get("GB_RIDGE_GIBBS", "1") == "1" else [])
            + vgb_ridge(),
            step_kwargs=dict(
                plateau_branch="gb",
                convergence_iter=int(os.environ.get("GB_PLATEAU_ITERS", "5")),
            ),
            combine_kwargs=dict(share_temperature_control=False),
        ),
        Stage(
            name="full_pe", kind="pe",
            # No rj_fstat_mcmc: the pe-named serial MCMC has the same
            # FD-kernel-on-WDM-data fstat scoring as its search twin.
            # rj_fstat_pe + rj_prior_pe are the GB PE moves. Armed source
            # PE moves follow the noise moves (the full_year composition:
            # noise, then sobbh -> mbh -> emri, then gb).
            # rj_warm_pe (when armed) runs IMMEDIATELY BEFORE rj_fstat_pe
            # -- the PE mirror of the search-stage order.
            moves=noise_pe + source_pe() + warm_pe() + [
                Move("rj_fstat_pe", branch="gb"),
                Move("rj_prior_pe", branch="gb"),
            ] + ([Move("gb_ridge_gibbs", branch="gb")]
                 if os.environ.get("GB_RIDGE_GIBBS", "1") == "1" else [])
            + vgb + vgb_ridge(),
            combine_kwargs=_pe_combine_kwargs(),
        ),
    ]
    fit.recipe = Recipe(stages)
    return fit


def main() -> int:
    # Only configure logging if nothing else has: the global-fit framework
    # installs its own handler, and adding a second one duplicates EVERY
    # line (which is what the first full-band run did).
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    if _env_flag("COMBINED_SMOKE"):
        _apply_smoke_defaults()
        print("[combined] SMOKE mode: shrunk axes, GB_DEBUG + VGB_DEBUG on.",
              flush=True)

    fit = build_fit()
    print(f"[combined] branches: {list(fit.branches)}", flush=True)
    for st in fit.recipe.stages:
        print(f"[combined] stage {st.name:>13s} ({st.kind:>6s}): "
              f"{[m.name for m in st.moves]}", flush=True)
    print(f"[combined] source_types: {fit.general.source_types}", flush=True)

    if _env_flag("COMBINED_DRY_RUN"):
        print("[combined] COMBINED_DRY_RUN=1 -- composition only, not built.",
              flush=True)
        return 0

    print("[combined] fit.build() ...", flush=True)
    from mpi4py import MPI
    from lisatools.globalfit.communication.ranks import layout_dry_run, prepare_rank, RankRole

    layout = prepare_rank(fit, MPI.COMM_WORLD)
    # GF_LAYOUT_DRY_RUN=1: print every rank's placement and stop before the
    # build allocates anything (a bad layout still raises inside prepare_rank).
    if layout_dry_run(layout, MPI.COMM_WORLD):
        print("[combined] GF_LAYOUT_DRY_RUN=1 -- layout only, not built.", flush=True)
        return 0
    fit.build()
    print("[combined] running", flush=True)
    fit.run()
    # LOUD completion marker on stdout: "Residuals saved" goes to the
    # logger FILE only, so a finished run used to look exactly like a
    # silent death on the console.
    #
    # RANK-GUARDED (2026-08-15): under `mpiexec -n 3` (dedicated saver
    # rank) ONLY the main rank runs the MCMC -- run_global_fit gates it on
    # `rank == main_rank` and the spare/saver ranks return immediately.
    # Printed unconditionally, those ranks announced "RUN COMPLETE:
    # num_iterations=2000 reached" seconds after launch, while the real
    # run was still starting its first stage. Alarming and completely
    # false, so say which rank is talking and only claim completion from
    # the rank that actually sampled.
    _rank = MPI.COMM_WORLD.Get_rank()
    role = layout.role_of(_rank)
    if role == RankRole.HEAD:
        if _env_flag("NULL_CHECK_ONLY"):
            # NULL_CHECK_ONLY stopped the run at the initial-lnL print
            # (run.py's null_check_only), so claiming "num_iterations
            # reached; residuals saved" would be flatly false.
            print(
                "[combined] NULL CHECK COMPLETE: initial lnL measured, no "
                "iterations run (NULL_CHECK_ONLY=1).",
                flush=True,
            )
        else:
            print(
                f"[combined] RUN COMPLETE: num_iterations="
                f"{fit.general.num_iterations} reached; residuals saved.",
                flush=True,
            )
    elif role == RankRole.COMPUTE:
        # under the walker-block layout a COMPUTE rank's fit.run() only
        # returns once the head has sent the fan-out STOP (review N-6) --
        # i.e. the run is over, not "still continuing on the head".
        print(
            f"[combined] rank {_rank} ({role.value}) released (head sent "
            f"STOP); run complete on the head.",
            flush=True,
        )
    elif role == RankRole.SAVER:
        # the saver's fit.run() returns only after it has handled the
        # head's {"finish_run": True} (review N-6).
        print(f"[combined] rank {_rank} ({role.value}) finished.", flush=True)
    else:  # legacy SPARE
        print(
            f"[combined] rank {_rank} ({role.value}) exiting; the run "
            f"continues on the head.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    from mpi4py import MPI
    from lisatools.globalfit.communication.ranks import install_mpi_abort_on_error

    _comm = install_mpi_abort_on_error(MPI.COMM_WORLD)
    try:
        _rc = main()
    except SystemExit:
        raise
    except BaseException:
        # The hook above handles the reporting + Abort; this only exists so
        # an exception escaping main() cannot fall through to a clean exit.
        sys.excepthook(*sys.exc_info())
        raise
    # A NONZERO return is also a failure: abort so no rank is left waiting.
    if _rc and _comm is not None:
        print(f"[MPI-ABORT] rank {_comm.Get_rank()} exiting with code {_rc}; "
              f"aborting all ranks.", file=sys.stderr, flush=True)
        _comm.Abort(_rc)
    sys.exit(_rc)
