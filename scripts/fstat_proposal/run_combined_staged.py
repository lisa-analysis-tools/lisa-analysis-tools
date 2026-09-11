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
    STAGE_NOISE_ONLY=1   run only the two noise search stages, then stop
    STAGE_NOISE_VGB_PE=1 searches, then PE-sample psd+galfor+vgb (no GB);
                         bounded by NUM_ITERATIONS
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import traceback

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
    # lists know how to drop are accepted; psd anchors the joint noise
    # criteria (JointMaxLogLSearch branch="psd") and the sensitivity model,
    # so it cannot go. With "gb" removed there is no F-stat machinery and
    # no RJ: the gb stages collapse to ONE full_pe over what remains.
    _removable = ("gb", "galfor", "vgb")
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

    # Every sampled branch needs a stream: NOISE for psd/galfor, GB, VGB --
    # plus the armed source classes' streams (their data must contain the
    # signals the branches fit). Explicit SOURCE_TYPES always wins.
    # GB_ONLY keeps the same default DELIBERATELY: the data is the SAME full
    # injection (noise + GB galaxy + VGBs) as the production runs -- only the
    # SAMPLED branch set shrinks to gb. Unmodeled content stays in the
    # residual; that is the accepted trade for not waiting on a noise fit.
    # REMOVE_BRANCHES is the opposite contract ("we will not inject them"):
    # a removed gb/vgb also leaves the DEFAULT stream list.
    _default_src = ",".join(
        ["NOISE"]
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
    noise_pe = [Move("psd_pe", branch="psd")] + (
        [Move("galfor_pe", branch="galfor")] if _has_galfor else [])
    # VGBs are KNOWN sources: fixed-dimensional, no RJ, nothing to search
    # for. They sample from the first stage onward so their power is being
    # fitted while the noise converges, rather than sitting in the residual
    # and biasing the PSD.
    vgb = [Move("vgb_pe", branch="vgb")] if _has_vgb else []
    _noise_names = ["psd_pe"] + (["galfor_pe"] if _has_galfor else [])

    # Stage 1: noise alone. Stage 2 and the GB search: noise + VGBs, with the
    # max-logl criterion spanning ALL of them -- one object per stage, so the
    # convergence is joint rather than each move plateauing separately.
    noise_only = [JointMaxLogLSearch(
        "noise_joint_search", list(_noise_names), branch="psd")]
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

    def source_pe():
        # Fresh Move descriptors per stage (never share one instance):
        # the armed source PE moves, sobbh -> mbh -> emri (banking order).
        return [Move(f"{br}_pe", branch=br)
                for br, _env, _cls in _SOURCE_BRANCH_ENVS
                if br in armed_sources]

    stages = []
    if armed_sources:
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
    if not _env_flag("STAGE_SKIP_NOISE"):
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
            moves=noise_pe + vgb,
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
            moves=noise_pe + source_pe() + vgb,
            combine_kwargs=_pe_combine_kwargs(),
        ))
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
            moves=noise_vgb_gb + source_pe() + warm() + [
                Move("rj_fstat_search", branch="gb"),
            ] + replace() + [
                Move("rj_prior_removal", branch="gb"),
            ] + ([Move("gb_ridge_gibbs", branch="gb")]
                 if os.environ.get("GB_RIDGE_GIBBS", "1") == "1" else []),
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
                 if os.environ.get("GB_RIDGE_GIBBS", "1") == "1" else []) + vgb,
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
    _rank, _main = 0, 0
    try:
        from mpi4py import MPI

        _rank = MPI.COMM_WORLD.Get_rank()
        _main = int(getattr(fit.settings_dict.rank_info, "main_rank", 0))
    except Exception:
        pass
    if _rank == _main:
        print(
            f"[combined] RUN COMPLETE: num_iterations="
            f"{fit.general.num_iterations} reached; residuals saved.",
            flush=True,
        )
    else:
        print(
            f"[combined] rank {_rank} (non-sampling helper) exiting; "
            f"the run continues on rank {_main}.",
            flush=True,
        )
    return 0


def _install_mpi_abort_on_error():
    """Make ANY rank's uncaught exception tear down the WHOLE job, loudly.

    Motivation (2026-08-15 forensics): job 210's main rank died on a corrupt
    HDF5 read at startup, but the run kept its Slurm allocation for ELEVEN
    HOURS at 0% GPU. Under ``mpiexec -n 3`` the dedicated saver rank sits in
    a blocking async save/plot loop, so when rank 0 exits nothing tells it to
    stop -- a crash silently becomes a resource-burning hang, and the only
    trace is a traceback in the sbatch stdout that nobody is watching.

    MPI gives no automatic teardown here: ``mpiexec`` waits on the surviving
    ranks. ``comm.Abort()`` is the sanctioned way to kill every rank at once,
    so route every uncaught exception through it AFTER printing the
    traceback (tagged with the rank, since otherwise it is guesswork which
    process failed).

    Returns the communicator when MPI is live, else None (a single-process
    run needs none of this and must keep normal Python exception behaviour).
    """
    try:
        from mpi4py import MPI
    except Exception:
        return None
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        return None  # single process: a plain traceback + exit is correct

    rank = comm.Get_rank()
    _prev_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        # KeyboardInterrupt stays interactive-friendly: still abort (the
        # other ranks would hang otherwise) but do not dump a scary trace.
        try:
            print(
                f"\n[MPI-ABORT] rank {rank} of {comm.Get_size()} raised "
                f"{exc_type.__name__}: {exc}\n"
                f"[MPI-ABORT] aborting ALL ranks so the job fails fast "
                f"instead of hanging on the surviving ones.",
                file=sys.stderr, flush=True)
            if exc_type is not KeyboardInterrupt:
                traceback.print_exception(exc_type, exc, tb, file=sys.stderr)
            sys.stderr.flush()
            sys.stdout.flush()
        except Exception:
            pass
        finally:
            try:
                comm.Abort(1)
            except Exception:
                os._exit(1)

    sys.excepthook = _hook

    # sys.excepthook is NOT used for exceptions raised in threads; the run
    # dispatches shard work on threads, so cover them too (3.8+).
    if hasattr(threading, "excepthook"):
        def _thread_hook(args):
            _hook(args.exc_type, args.exc_value, args.exc_traceback)
        threading.excepthook = _thread_hook

    return comm


if __name__ == "__main__":
    _comm = _install_mpi_abort_on_error()
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
