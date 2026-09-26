"""The 6mo campaign submit scripts size the rank count from the GPU count.

Both ``submit_gf_6mo_v8.sh`` (main) and ``submit_gf_6mo_v8_nogb_null.sh``
(null test) self-dispatch via ``exec sbatch ...`` before any heavy work
(environment activation, data preflight, ...) whenever ``SLURM_JOB_ID`` is
unset. That makes the pre-submit dispatch block cheap to exercise directly:
shadow ``sbatch`` with a stub that prints its argv and exits 0, run the
script with ``bash``, and inspect what the (never-actually-submitted) job
would have looked like.

Covers the layout invariants (user ruling 2026-09-16, after the WP7
transport gates passed on the cluster): at the ``NGPUS=2`` default the
scripts pin the walker-block layout (``GF_LEGACY_RANK_LAYOUT=0``: head +
1 compute + saver, still ``--ntasks=3``); ``GF_LEGACY_RANK_LAYOUT=1`` remains
the rollback knob and must still launch today's single-compute-rank shape;
``NGPUS=4`` must force the walker-block layout across 2 nodes of 2 GPUs each,
since the legacy layout cannot span nodes.
"""

import os
import re
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPTS = [
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_6mo_v8.sh"),
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_6mo_v8_nogb_null.sh"),
    # the 3-month twin of the 6mo campaign script: same layout machinery,
    # Tobs-derived settings reverted, source branches and warm start off
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_3mo_v8_4gpu.sh"),
    # the v9 campaign script: same dispatch machinery as 6mo_v8, three
    # deliberate deltas (see SixMonthV9DeltaTest)
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_6mo_v9_4gpu.sh"),
    # the 3-month twin OF THE V9 SCRIPT (2026-09-26): psd-only first stage,
    # inflated foreground reference, no warm start, 2 GPUs. See
    # ThreeMonthV9TwinTest for the declared delta list.
    os.path.join(ROOT, "scripts", "fstat_proposal", "submit_gf_3mo_v9_2gpu.sh"),
]

THREE_MO = SCRIPTS[2]
SIX_MO = SCRIPTS[0]
SIX_MO_V9 = SCRIPTS[3]
THREE_MO_V9 = SCRIPTS[4]

# Env knobs the dispatch block reads; stripped from the inherited environment
# before each scenario applies its own overrides, so a stray value in the
# test runner's shell can never leak into a scenario that expects it unset.
_DISPATCH_ENV_KEYS = (
    "SLURM_JOB_ID",
    "NGPUS",
    "NODES",
    "GPUS_PER_RANK",
    "RANKS_PER_GPU",
    "GF_LEGACY_RANK_LAYOUT",
)

_STUB_SBATCH = """#!/usr/bin/env bash
for a in "$@"; do printf '%s\\n' "$a"; done
exit 0
"""


def _exports(path):
    """``{KNOB: value}`` as bash resolves the script's export lines in order.

    Resolving them rather than regexing the file is the point: the values
    are ``${K:-default}`` forms and several are overridden further down, so
    a grep answers what the file SAYS and this answers what the run GETS.
    """
    src = open(path).read()
    lines = [l for l in src.split("\n") if re.match(r"^export [A-Z0-9_]+=", l)]
    env = dict(os.environ)
    for k in list(env):
        if k.isupper():
            env.pop(k, None)
    out = subprocess.run(["bash", "-c", "\n".join(lines) + "\nenv | sort\n"],
                         capture_output=True, text=True, env=env).stdout
    return dict(l.split("=", 1) for l in out.split("\n") if "=" in l)


class SixMonthV9DeltaTest(unittest.TestCase):
    """``submit_gf_6mo_v9_4gpu.sh`` is ``submit_gf_6mo_v8.sh`` plus exactly
    three families of change (user spec 2026-09-24).

    Same silent-failure shape as the 3-month twin below: the two files are
    ~99% the same text, and every block that MUST differ looks like an
    ordinary knob. A v9 that quietly kept v8's coarse surrogate, or ran with
    the convergence mode off, would produce entirely plausible output and
    waste the campaign -- so each delta is pinned here, and so is the
    v8 side of it, because a test that only checks v9 would still pass if
    someone "fixed" v8 to match.
    """

    def setUp(self):
        self.v8 = _exports(SIX_MO)
        self.v9 = _exports(SIX_MO_V9)

    # -- V9-2: noise likelihood back to exact-fine ----------------------
    def test_the_noise_model_is_normal_not_coarse(self):
        self.assertEqual(self.v8["COARSE_GPU_MODE"], "delayed_acceptance")
        self.assertEqual(self.v9["COARSE_GPU_MODE"], "off")

    def test_the_other_coarse_knobs_still_match_v8(self):
        """``COARSE_Q`` deliberately does NOT match v8 any more -- see
        :meth:`test_v9_runs_the_EXACT_FINE_noise_likelihood` and job 618.
        This test used to include it, on the same false premise the script
        comment carried ("they select nothing at mode=off"), which is how
        an illegal pair shipped. The two that really are inert stay pinned:
        dropping them would let the stamped noise identity and the
        preflight's "wanted" values disagree."""
        for k in ("COARSE_USE_WS", "COARSE_FIDUCIAL"):
            self.assertEqual(self.v9[k], self.v8[k], k)

    # -- V9-1: per-source in-model convergence --------------------------
    def test_the_convergence_mode_is_armed(self):
        self.assertNotIn("GB_INMODEL_CONVERGE", self.v8)
        self.assertIn(self.v9["GB_INMODEL_CONVERGE"], ("on", "observe"))

    def test_the_convergence_defaults_are_the_ruled_values(self):
        # 100 -> 250 (user, 2026-09-25, "just to be safe" for births).
        self.assertEqual(self.v9["GB_INMODEL_CONVERGE_ITERS"], "250")
        self.assertEqual(self.v9["GB_INMODEL_CONVERGE_DLL"], "4.0")
        # BOTH. "RJ moves [should] have their sources run to convergence
        # (in the search) whether they are birthed or survived" (user,
        # 2026-09-25). This is the per-ROW rule in the RJ pool; the pure
        # in-model slots converge per SUB-BAND via the GROUP rule and are
        # a separate scope, untouched by this knob.
        self.assertEqual(
            self.v9["GB_INMODEL_CONVERGE_CLASSES"], "newborn,mature")

    def test_the_row_ceiling_leaves_room_ABOVE_the_patience_floor(self):
        """⚠ The floor is W+1: a row cannot retire before the window is
        full. So the ceiling has to sit well above W or it, not the
        evidence, decides when a birth stops polishing.

        This is why raising ITERS alone is wrong: at W=250 the default
        ceiling (0 = 4x the 100-repeat class budget = 400) would leave a
        row just 149 repeats in which to plateau, against a measured p50 of
        260 at W=100. Pinned as a RATIO so the two cannot drift apart."""
        w = int(self.v9["GB_INMODEL_CONVERGE_ITERS"])
        ceil = int(self.v9["GB_INMODEL_CONVERGE_MAX"])
        self.assertGreater(ceil, 0, "0 resolves to 4x the CLASS budget, "
                                    "which does not track the window")
        self.assertGreaterEqual(
            ceil, 3 * w,
            f"ceiling {ceil} is too tight for a {w}-repeat window")

    def test_the_ladder_gate_is_on(self):
        """1.0 would make every rung vote, and the hot rungs never retire."""
        self.assertLess(
            float(self.v9["GB_INMODEL_CONVERGE_GATE_FRAC"]), 1.0)

    def test_the_refill_is_armed_and_can_actually_fire(self):
        """stop_frac 1.0 silently disables the refill: every column would
        finish together, nothing would carry, and the loop degenerates into
        the fixed chunk loop. Arming REFILL without lowering it is the
        no-op combination this pins against."""
        self.assertEqual(self.v9["GB_INMODEL_CONVERGE_REFILL"], "1")
        self.assertLess(
            float(self.v9["GB_INMODEL_CONVERGE_STOP_FRAC"]), 1.0)

    # -- V9-5 / V9-6: flip fraction + the in-GROUP convergence ----------
    def test_the_search_flip_fraction_is_one(self):
        self.assertEqual(self.v8["GB_SEARCH_RJ_FLIP_FRACTION"], "0.5")
        self.assertEqual(self.v9["GB_SEARCH_RJ_FLIP_FRACTION"], "1.0")
        # PE is untouched
        self.assertEqual(self.v9["GB_PE_RJ_FLIP_FRACTION"],
                         self.v8["GB_PE_RJ_FLIP_FRACTION"])

    def test_the_in_group_convergence_is_armed_and_distinct(self):
        """V9-6 is a DIFFERENT scope from V9-1 -- (walker, band) on a
        pass clock inside one proposal, vs row on a repeat clock. Pinned
        together so nobody "tidies" one family into the other."""
        self.assertEqual(self.v9["GB_INMODEL_GROUP"], "1")
        self.assertEqual(self.v9["GB_INMODEL_GROUP_SCALE"], "flat")
        self.assertEqual(self.v9["GB_INMODEL_GROUP_DLL"], "4.0")
        # the group window is in PASSES and must stay far smaller than the
        # row window, which is in repeats -- swapping them would make the
        # group rule unfireable (and the row rule fire on noise)
        self.assertLess(int(self.v9["GB_INMODEL_GROUP_ITERS"]),
                        int(self.v9["GB_INMODEL_CONVERGE_ITERS"]))

    def test_the_group_has_a_hard_pass_ceiling(self):
        """Unbounded by construction: one pass is a full sweep of every
        source, so the ceiling is the only bound."""
        self.assertGreater(int(self.v9["GB_INMODEL_GROUP_MAX_PASSES"]), 0)

    # -- V9-7: permuted ("fancy") swaps off, vertical KEPT ---------------
    def test_fancy_tempering_is_off_but_vertical_survives(self):
        """⚠ Killing run_tempering must NOT take the vertical rung swaps
        with it. They are additive to the permuted swaps, not part of
        them, and they are the transport the in-model convergence depends
        on -- AND, since _adapt_band_temps is only ever called from
        run_tempering, they are now the only thing keeping the temperature
        ladder from freezing for the whole run."""
        self.assertEqual(self.v9["GB_RUN_FANCY_TEMPERING"], "0")
        self.assertEqual(self.v9["GB_TEMPER_VERTICAL"], "1")

    # -- V9-8: the standalone in-model move is actually BUILT -------------
    def test_the_pure_in_model_move_is_armed(self):
        """Without this the group-convergence knobs are exported, ignored,
        and nothing ever logs [GB_IMGROUP] -- the move is not built and not
        scheduled. This pair is what makes V9-6 reachable at all."""
        self.assertEqual(self.v9["GB_SEARCH_IN_MODEL"], "1")
        # Briefly 100 on 2026-09-25, reverted the same day: the group
        # window is 3 PASSES and a pass is this many repeats, so raising it
        # tightens the enforced rate 4x rather than coarsening granularity.
        self.assertEqual(self.v9["GB_NUM_REPEAT_PROPOSALS"], "25")
        self.assertNotIn("GB_SEARCH_IN_MODEL", self.v8)

    # -- V9-9: the stage-convergence valve --------------------------------
    def test_the_stage_shutoff_valve_is_on_and_the_band_schedule_is_not(self):
        """SNR limits move per recipe STAGE in v9, never per band, so the
        per-band stage schedule must stay out of the way. Independent
        flags; pinned together because arming both would have the two
        fighting over the same thresholds."""
        self.assertEqual(self.v9["GB_SEARCH_BAND_SHUTOFF_PER_WALKER"], "1")
        self.assertEqual(self.v9["GB_SEARCH_STAGE_PER_WALKER"], "0")

    # -- V9-10: refit cadence ---------------------------------------------
    def test_the_pe_refit_cadence_is_separate_and_longer(self):
        """full_pe is a random_choice stage; a search-tuned cadence there
        refits far more often in wall-clock terms than the number says."""
        self.assertEqual(self.v9["GB_FSTAT_REFIT_EVERY_PE"], "250")
        # SEARCH stages refit 4x more often (user, 2026-09-25): the grid
        # goes stale as the search subtracts what it finds, so an old grid
        # aims births at peaks already claimed. PE is deliberately NOT
        # changed -- the two cadences are separate knobs for that reason.
        # 10 -> 1 (2026-09-25): refit every search iteration, because
        # the residual moves fast enough that the grid goes stale --
        # 628's rj_fstat_search cold yield fell 135 -> 96 -> 54.
        self.assertEqual(self.v9["GB_FSTAT_REFIT_EVERY"], "1")
        self.assertLess(int(self.v9["GB_FSTAT_REFIT_EVERY"]),
                        int(self.v9["GB_FSTAT_REFIT_EVERY_PE"]))

    # -- V9-4: cap cells off --------------------------------------------
    def test_the_leaf_cap_is_off_and_its_companions_with_it(self):
        """User ruling 2026-09-24: "no caps. Keep all those options
        available but turn them off."

        The cap is armed ONLY by a non-empty GB_LEAF_CAP_START, so empty is
        the off switch and every other knob stays present but inert. The
        companions are pinned separately because bash resolves exports in
        order -- an early `=0` followed by a later `=1` silently re-arms
        them, which is exactly the bug this test caught during authoring.
        """
        self.assertEqual(self.v9["GB_LEAF_CAP_START"], "")
        self.assertEqual(self.v9["GB_CAP_DRIFT_GATE"], "0")
        self.assertEqual(self.v9["GB_CAP_DRIFT_GATE_EDGE_LEAK"], "0")
        # the two stage-convergence vetoes the cap needed
        self.assertEqual(self.v9["GB_SEARCH_CAP_QUIESCENT"], "0")
        self.assertEqual(self.v9.get("GB_SEARCH_CAP_HEADROOM", "0"), "0")

    def test_the_cap_knobs_are_still_present_not_deleted(self):
        """"Keep all those options available" -- re-arming must be a
        one-line change, not an archaeology exercise."""
        src = open(SIX_MO_V9).read()
        for knob in ("GB_CAP_DIVISOR", "GB_LEAF_CAP_MIN_ITERS",
                     "GB_LEAF_CAP_REQUIRE_IMPROVEMENT"):
            self.assertIn(f"export {knob}=", src, knob)

    # -- V9-3: naming / a fresh store -----------------------------------
    def test_the_store_is_not_v8s(self):
        """The coarse mode is part of noise_model_identity, so a v8 store
        cannot be resumed under V9-2 -- run.py refuses it. The default
        STORE_DIR must therefore be a new path, not v8's."""
        src = open(SIX_MO_V9).read()
        self.assertIn("gf_prod_6mo_v9_4gpu/", src)
        self.assertNotIn(
            "STORE_DIR:-/shared/data/global_fit_output/gf_prod_6mo_v8/", src)

    def test_it_does_not_write_over_v8s_log(self):
        src = open(SIX_MO_V9).read()
        self.assertIn("--job-name=gf6mo_v9_4gpu", src)
        self.assertNotIn("gf6mo_v8_%j.log", src)

    # -- V9-11: the three-stage GB search --------------------------------
    def test_the_three_stage_search_is_armed(self):
        """One gb_search stage becomes gb_search_1/2/3. Without this the
        driver composes the single legacy stage and EVERY per-stage profile
        -- phase max, opt SNR, the F-stat peak floor -- silently never
        applies, while the script's own header claims all three."""
        self.assertEqual(self.v9["STAGE_V9_SEARCH"], "1")
        self.assertNotIn("STAGE_V9_SEARCH", self.v8)

    def test_the_stage1_values_are_what_the_moves_are_BUILT_with(self):
        """phase max / opt SNR / peak floor are per-stage PROFILE values
        now, applied at stage entry. These exports are what is in force
        BEFORE the first profile applies, so they must be stage 1's: if a
        profile ever failed to apply, the run would hold stage 1's
        configuration rather than silently run stage 1 as stage 2."""
        self.assertEqual(self.v9["GB_RJ_PHASE_MAXIMIZE"], "1")
        self.assertEqual(self.v9["GB_OPT_SNR_LIMIT_SEARCH"], "8.0")
        self.assertEqual(self.v9["FSTAT_PEAK_MIN_SNR"], "8.0")

    def test_stage_3_warm_start_rides_a_cadence(self):
        """User ruling 2026-09-24: warm start every 5th iteration in
        gb_search_3 only."""
        self.assertEqual(self.v9["GB_SEARCH_3_WARM_EVERY"], "5")

    def test_rj_replace_is_back_and_two_pass(self):
        """Reinstated 2026-09-24, then switched OFF again 2026-09-25 on
        yield: six consecutive zero-yield proposes in job 628, 8 swaps out
        of ~707k proposals and ZERO at the cold chain, while costing 19.4%
        of the cycle. The two-pass WIRING stays configured so reinstating
        it for gb_search_3 is a one-knob change."""
        self.assertEqual(self.v9["GB_SEARCH_RJ_REPLACE"], "0")
        self.assertEqual(self.v9["GB_REPLACE_WARM_PASS"], "1")
        self.assertEqual(self.v8["GB_SEARCH_RJ_REPLACE"], "0")
        # PE is explicitly NOT part of the reinstatement.
        self.assertEqual(self.v9["GB_PE_RJ_REPLACE"], "0")

    # -- V9-12: one seed store for the warm start AND the noise pin -------
    def test_the_warm_start_and_the_noise_pin_share_one_store(self):
        """User ruling 2026-09-24: "make sure the warmstart and PSD/GB
        frozen start point come from the same folder ... It should use
        maxlogL for PSD/GB." One variable, so overriding one without the
        other is not something a launch can do by accident."""
        src = open(SIX_MO_V9).read()
        self.assertIn("export GF_SEED_STORE=", src)
        self.assertIn("GB_WARM_START_SOURCE_STORE-${GF_SEED_STORE}", src)
        self.assertIn("lisatools.globalfit.warmstart.noise_pin", src)

    def test_the_seed_is_the_3mo_10_WALKER_run_that_actually_EXISTS(self):
        """REGRESSION, job 616 (2026-09-24 19:52). The default used to name
        a `_noisefix` sibling that does not exist on the cluster, and the
        first v9 launch died on it before reaching python -- the script's
        own candidate scan listed eight gf_prod_3mo* runs with no _noisefix
        among them. User ruling on the forensics, 2026-09-25: "yea it is
        just gf_prod_3mo_v8_10walkers".

        A path this test cannot verify (no filesystem here) is exactly the
        kind that has to be pinned by NAME once it has been confirmed by a
        launch, so a plausible-looking edit cannot silently reintroduce it.
        """
        self.assertEqual(
            self.v9["GF_SEED_STORE"],
            "/shared/data/global_fit_output/gf_prod_3mo_v8_10walkers/"
            "gf_prod_3mo_testing.h5",
        )
        self.assertNotIn("noisefix", self.v9["GF_SEED_STORE"])

    def test_the_seed_never_falls_back_to_a_DIFFERENT_run(self):
        """The resolver may pick the single .h5 inside a given directory,
        but it must never substitute another run: a seed is a provenance
        statement, and quietly swapping it is the exact "warm start and
        noise pin from different folders" split the ruling forbids.

        Note this is about the RESOLVER, not the default. Changing which
        store the default names is a user ruling; substituting one at run
        time, after the operator named a different one, is the defect.
        """
        src = open(SIX_MO_V9).read()
        self.assertIn("does NOT fall back to a", src)
        # The only substitution allowed is directory -> the ONE .h5 in it.
        self.assertIn('_n_h5=$(find "${_seed_dir}" -maxdepth 1 -name \'*.h5\'',
                      src)
        self.assertIn('if [ "${_n_h5}" = "1" ]; then', src)

    def test_the_shipped_COARSE_pair_PASSES_the_real_validator(self):
        """REGRESSION, job 618 (2026-09-25): all 5 ranks aborted at build
        with ``coarse_Q > 1 in an all-source run requires an explicit
        COARSE_GPU_MODE``.

        The script shipped COARSE_Q=8 with COARSE_GPU_MODE=off on the
        written-down belief that "with mode=off they select nothing". They
        do not: ``run.py`` reads ``coarse_Q`` on its own and builds the
        coarse basis for any Q > 1 without consulting the mode.

        Asserting the two VALUES would only re-state the fix. This runs the
        actual function that rejected the job, so the pair is checked by
        the rule itself and a future edit to either knob is caught here
        rather than on the cluster.
        """
        from lisatools.globalfit.stock.erebor.noise import (
            validate_coarse_settings)

        class _GS:
            pass

        gs = _GS()
        gs.coarse_Q = int(self.v9["COARSE_Q"])
        gs.coarse_gpu_mode = self.v9["COARSE_GPU_MODE"]
        gs.coarse_fiducial = self.v9["COARSE_FIDUCIAL"]
        gs.gpus = [0, 1]
        validate_coarse_settings(gs, all_source=True)   # must not raise

    def test_v9_runs_the_EXACT_FINE_noise_likelihood(self):
        """V9-2 ruling, restated 2026-09-25: "We want the general psd
        computations for galfor and psd. Not the coarse version." Q=1 IS
        that configuration -- there is nothing to coarsen."""
        self.assertEqual(self.v9["COARSE_Q"], "1")
        self.assertEqual(self.v9["COARSE_GPU_MODE"], "off")
        # ...and v8, which this file is rebased on, ran the OTHER legal
        # pair. Pinned so the delta stays deliberate and visible.
        self.assertEqual(self.v8["COARSE_Q"], "8")
        self.assertEqual(self.v8["COARSE_GPU_MODE"], "delayed_acceptance")

    def test_the_noise_pin_output_is_PARSED_never_evald(self):
        """REGRESSION, job 617 (2026-09-25):

            slurm_script: eval: line 3572: syntax error near unexpected
            token `('

        The CLI is clean -- three diagnostics to STDERR, two
        ``export NAME=<digits>`` lines to stdout, verified against a real
        store both bare and under this script's full export environment.
        Something else in the cluster process wrote to stdout (a backend
        banner is the likely culprit; it cannot appear on a laptop with no
        CUDA, which is why local testing could not have caught it).

        ``eval`` of another process's stdout is the defect whoever the
        polluter is: ANY stray line becomes shell. The values must be
        parsed out with an anchored, number-only pattern instead.
        """
        src = open(SIX_MO_V9).read()
        self.assertNotIn('eval "${_pin}"', src)
        self.assertIn("NEVER `eval` this output", src)
        for var in ("PSD", "GALFOR"):
            self.assertIn(
                rf"sed -n 's/^export {var}_START_PARAMS="
                rf"\([-+0-9.eE,]*\)$/\1/p'", src)

    def test_the_pin_satisfies_each_variable_INDEPENDENTLY(self):
        """User ruling 2026-09-25: "the galfor pin should come from our
        estimate. The PSD estimate can come from the 3mo run." Since
        GALFOR_START_PARAMS ships hand-set, requiring BOTH lines to parse
        would discard a good instrument-noise pin over a galfor value this
        run was never going to use."""
        src = open(SIX_MO_V9).read()
        self.assertIn(
            '{ [ -n "${_psd_was}" ] || [ -n "${_psd_pin}" ]; }', src)
        self.assertIn(
            '{ [ -n "${_gal_was}" ] || [ -n "${_gal_pin}" ]; }', src)
        # ...and galfor still ships hand-set, so the store never supplies it.
        self.assertTrue(self.v9["GALFOR_START_PARAMS"])

    def test_unexpected_pin_stdout_is_reported_not_silently_dropped(self):
        """Dropping the junk fixes the crash but loses the evidence. The
        next run has to NAME the polluter, or this comment stays a guess."""
        src = open(SIX_MO_V9).read()
        self.assertIn("unexpected stdout", src)
        self.assertIn("_pin_junk", src)

    def test_a_missing_seed_says_it_will_ALSO_kill_the_warm_start_gate(self):
        """Job 616's log was hard to read because one cause tripped two
        gates that never referenced each other: the [V9-SEED] miss printed
        a SOFT message ("otherwise the noise stages will fit it"), then the
        [WARMSTART] gate hard-exited 10 lines later on the same inherited
        path with no traceback. The seed gate must forward-declare the
        fatality, and the fatal gate must name where its path came from."""
        src = open(SIX_MO_V9).read()
        self.assertIn("AND THIS WILL BE FATAL BELOW", src)
        self.assertIn("there is one problem here, not two", src)
        self.assertIn("that path came from GF_SEED_STORE", src)

    def test_mid_iteration_checkpoints_are_pinned_on(self):
        """User ask 2026-09-24: "checkpoints between all the GB moves for
        resume". The HOOKS are unconditional (GFCombineMove writes after
        every sub-move); what this pins is that the feature is ARMED and
        that its throttle -- the thing that actually decides how much a spot
        preemption costs -- is on the run record rather than a silent
        default."""
        self.assertEqual(self.v9["MIDIT_CHECKPOINT"], "1")
        self.assertIn("MIDIT_CHECKPOINT_MIN_INTERVAL", self.v9)
        self.assertGreaterEqual(
            int(self.v9["MIDIT_CHECKPOINT_MIN_INTERVAL"]), 0)

    def test_the_galfor_start_is_the_offline_3mo_estimate(self):
        """User ruling 2026-09-24, chosen explicitly over the 6mo
        alternative. PHYSICAL/LINEAR, amp MODULATION-CORRECTED (the 90 d
        window sits on a dim stretch, <M_XX> = 0.808, so the correction
        RAISES amp -- using the raw number understates by 19%).

        Pinned as numbers against the file they came from, because the
        vector is only meaningful as a whole: two of its five parameters
        are unidentified, and cherry-picking one would silently change the
        curve.
        """
        import json
        import os as _os

        raw = self.v9["GALFOR_START_PARAMS"]
        got = [float(x) for x in raw.split(",")]
        self.assertEqual(len(got), 5)
        src = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "galfor_6mo_figs", "addback_fixedpoint_90d.json")
        if not _os.path.exists(src):
            self.skipTest("addback_fixedpoint_90d.json is untracked scratch")
        fp = json.load(open(src))["history"][-1]["prior_box_refit"]
        want = [fp["amp_modulation_corrected"], fp["params"]["fk"],
                fp["params"]["alpha"], fp["params"]["f_1"],
                fp["params"]["f_2"]]
        for g, w, n in zip(got, want, ("amp", "fk", "alpha", "f_1", "f_2")):
            self.assertAlmostEqual(g / w, 1.0, places=9, msg=n)

    def test_the_galfor_start_is_inside_the_run_prior(self):
        """⚠ The one check that matters: outside the support, every walker
        and every rung prices at log_prior = -inf on iteration 0. alpha 5.0
        is INTERIOR only because GALFOR_ALPHA_MAX=20.0 is also exported --
        the two knobs are load-bearing together."""
        from lisatools.globalfit.stock.erebor.noise import (
            GALFOR_BASIS, GALFOR_PRIOR_RANGE)

        got = [float(x) for x in self.v9["GALFOR_START_PARAMS"].split(",")]
        box = [list(map(float, r)) for r in GALFOR_PRIOR_RANGE]
        box[GALFOR_BASIS.index("alpha")][1] = float(
            self.v9["GALFOR_ALPHA_MAX"])
        for n, v, (lo, hi) in zip(GALFOR_BASIS, got, box):
            self.assertTrue(lo <= v <= hi, f"{n}={v:g} outside [{lo:g},{hi:g}]")

    def test_a_hand_set_galfor_start_survives_the_seed_pin(self):
        """The script hardcodes GALFOR_START_PARAMS and ALSO reads a pin
        from GF_SEED_STORE. The pin block must not clobber it -- that is
        what the per-variable guard is for, and it is the whole reason an
        offline estimate can be supplied at all."""
        src = open(SIX_MO_V9).read()
        self.assertIn("_gal_was=${GALFOR_START_PARAMS+set}", src)
        self.assertIn("was set by hand -- keeping it, not the store's", src)
        self.assertIn("${GALFOR_START_PARAMS:-", src)

    def test_the_noise_pin_is_read_at_MAXLOGL(self):
        """noise_pin reads best_logl_noise, i.e. the best-logL cold walker
        of the last valid row -- not a posterior mean."""
        src = open(SIX_MO_V9).read()
        self.assertIn("maxlogL", src)

    def test_everything_else_is_still_v8(self):
        """The whole point of a copy-with-deltas: any OTHER knob that
        drifted is either an unrecorded change or a bad merge."""
        allowed = {
            # V9-2. COARSE_Q joined this list on 2026-09-25: it is not an
            # extra change but the OTHER HALF of the same one -- mode=off
            # with Q=8 is the illegal pair that aborted job 618.
            "COARSE_GPU_MODE", "COARSE_Q",
            "STORE_DIR", "FILE_STORE_DIR",          # V9-3
            "GB_LEAF_CAP_START", "GB_CAP_DRIFT_GATE",
            "GB_CAP_DRIFT_GATE_EDGE_LEAK", "GB_SEARCH_CAP_QUIESCENT",
            "GB_SEARCH_RJ_FLIP_FRACTION",           # V9-5
            "GB_RUN_FANCY_TEMPERING",               # V9-7
            "GB_SEARCH_IN_MODEL",                   # V9-8
            "GB_NUM_REPEAT_PROPOSALS",              # V9-8
            "GB_SEARCH_BAND_SHUTOFF_PER_WALKER",    # V9-9
            "GB_SEARCH_BAND_SHUTOFF_CONV_ITER",     # V9-9
            "GB_SEARCH_STAGE_PER_WALKER",           # V9-9
            "GB_FSTAT_REFIT_EVERY_PE",              # V9-10
            "GB_FSTAT_REFIT_EVERY",                 # V9-14 (40 -> 10)
            # V9-13 (2026-09-25): the RJ schedule itself. The rigid path
            # ran ALL RJ rounds before any polish; the staged scheduler
            # interleaves RJ round -> in-model to convergence -> refill.
            "GB_RJ_DIRECT_BATCH",
            # V9-11, the three-stage search restructure
            "STAGE_V9_SEARCH", "GB_SEARCH_3_WARM_EVERY",
            "GB_SEARCH_RJ_REPLACE", "GB_REPLACE_WARM_PASS",
            "GB_RJ_PHASE_MAXIMIZE", "GB_OPT_SNR_LIMIT_SEARCH",
            "FSTAT_PEAK_MIN_SNR",
            # V9-12, the shared seed store + noise pin
            "GF_SEED_STORE", "GB_WARM_START_SOURCE_STORE",
            # mid-iteration checkpoints, pinned explicitly for v9
            "MIDIT_CHECKPOINT", "MIDIT_CHECKPOINT_MIN_INTERVAL",
            # the offline 3mo galfor start point
            "GALFOR_START_PARAMS",
            # f0-adaptive stage-B sky grid (the F-stat fix, 2026-09-24)
            "FSTAT_STAGEB_SKY_ADAPT", "FSTAT_STAGEB_NSKY_MIN",
            "FSTAT_STAGEB_NSKY_MAX", "FSTAT_STAGEB_GROUP_MAX_GB",
            # V9-15 (2026-09-25, off job 628): sig-het reference refresh
            # 25 -> 50. It was the largest single line item inside the
            # in-model polish (167-186 s per RJ move). Trades reference
            # staleness for wall time; [GB_TRUST] and the end-of-block
            # "ll AUDIT vs exact" COLD median are the instruments.
            "GB_SIGHET_REFRESH_EVERY",
            # V9-16 (2026-09-26): gb_search_seed -- a new FIRST search
            # stage, warm-start RJ + in-model only, fixed 5 iterations,
            # before any F-stat grid is fitted. Named rather than
            # renumbered, so gb_search_1/2/3 are untouched.
            "GB_SEARCH_SEED_ITERS",
        }
        drift = {
            k: (self.v8.get(k), self.v9.get(k))
            for k in set(self.v8) | set(self.v9)
            if self.v8.get(k) != self.v9.get(k)
            and not k.startswith("GB_INMODEL_CONVERGE")
            and not k.startswith("GB_INMODEL_GROUP")
            and k not in allowed
        }
        self.assertEqual(drift, {}, f"undeclared v8 -> v9 drift: {drift}")


class MpiPlacementTest(unittest.TestCase):
    """⚠ EVERY script that launches `-ppn 1` must also export
    I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0, or the placement silently reverts
    to SLURM's per-node block split.

    This is a LAUNCH BLOCKER, not a tuning knob. At 5 tasks over 2 nodes --
    the v9 4-GPU shape, 4 compute + 1 saver -- SLURM's 3/2 split puts three
    compute ranks on a 2-GPU node and build_layout refuses outright. At 3
    tasks the two placements coincide, which is exactly why it went unnoticed
    until a 5-task launch, and why it is pinned here rather than left to the
    next person to rediscover.

    Found 2026-09-24: the fix lived on an UNMERGED branch, so neither v9 nor
    v8-on-dev carried it.
    """

    def test_every_ppn_launcher_sets_placement_respect_off(self):
        import glob

        checked = 0
        for path in glob.glob(os.path.join(
                ROOT, "scripts", "fstat_proposal", "submit_gf_*.sh")):
            src = open(path).read()
            if "-ppn 1" not in src:
                continue
            checked += 1
            self.assertIn(
                "export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0", src,
                f"{os.path.basename(path)} launches `-ppn 1` without the "
                f"placement export; a 5-task 2-node run would be refused by "
                f"build_layout")
        self.assertGreater(checked, 0, "no -ppn launcher found to check")

    def test_the_v9_launcher_is_one_of_them(self):
        src = open(SIX_MO_V9).read()
        self.assertIn("-ppn 1", src)
        self.assertIn("export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0", src)


class V9RankLayoutTest(unittest.TestCase):
    """What (NGPUS, NWALKERS) actually resolve to, run against the SCRIPT's
    own arithmetic rather than a reimplementation of it.

    Written 2026-09-24 while the user was testing the NWALKERS/NGPU shape.
    The hazard is not a crash: a NWALKERS that does not divide N_COMPUTE is
    ROUNDED UP rather than refused, and the store then LOCKS to the rounded
    value for its whole life (a resume refuses any change). So testing at one
    value and launching at another silently produces a different run.
    """

    #: The two formulas this pins, quoted from the script so a drift in
    #: either place fails here rather than at launch.
    N_COMPUTE_SRC = ("N_COMPUTE_EFF=$(( ${SLURM_NNODES:-1} * _NGPUS_EFF "
                     "* RANKS_PER_GPU / _k ))")
    ROUND_SRC = "export NWALKERS=$(( (NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF ))"

    def setUp(self):
        self.src = open(SIX_MO_V9).read()

    def test_the_formulas_are_still_the_ones_modelled_here(self):
        """If either line changes, the expectations below are stale and must
        be re-derived -- fail loudly instead of asserting against fiction."""
        self.assertIn(self.N_COMPUTE_SRC, self.src)
        self.assertIn(self.ROUND_SRC, self.src)

    @staticmethod
    def _resolve(ngpus, nwalkers, ranks_per_gpu=1, gpus_per_rank=1):
        nodes, gpus_per_node = (2, 2) if ngpus == 4 else (1, 2)
        n_compute = nodes * gpus_per_node * ranks_per_gpu // gpus_per_rank
        if n_compute > 1 and nwalkers == 1:
            return n_compute, 1, "replica"
        if n_compute > 0 and nwalkers % n_compute:
            nwalkers = (nwalkers // n_compute + 1) * n_compute
            return n_compute, nwalkers, "rounded"
        return n_compute, nwalkers, "exact"

    def test_the_intended_4gpu_shape_needs_no_rounding(self):
        """NGPUS=4, NWALKERS=4 -> N_COMPUTE 4, one walker per rank, exact."""
        n_compute, nw, how = self._resolve(4, int(self.v9_nwalkers()))
        self.assertEqual((n_compute, nw, how), (4, 4, "exact"))

    def v9_nwalkers(self):
        return _exports(SIX_MO_V9)["NWALKERS"]

    def test_the_script_ships_the_shape_that_divides(self):
        self.assertEqual(int(self.v9_nwalkers()), 4)

    def test_non_multiples_are_ROUNDED_not_refused(self):
        """⚠ The quiet one. Only a [SUBMIT] line marks it, and the store then
        locks to the rounded number."""
        for asked, got in ((2, 4), (3, 4), (5, 8), (6, 8), (10, 12)):
            n_compute, nw, how = self._resolve(4, asked)
            self.assertEqual((nw, how), (got, "rounded"),
                             f"NGPUS=4 NWALKERS={asked}")

    def test_one_walker_is_replica_mode_not_a_rounding(self):
        self.assertEqual(self._resolve(4, 1)[2], "replica")
        self.assertEqual(self._resolve(2, 1)[2], "replica")

    def test_ranks_per_gpu_2_gives_EIGHT_compute_ranks_not_five(self):
        """The script's old comment claimed 'NGPUS=4 needs 5 compute ranks
        with RANKS_PER_GPU=2'. It does not -- that gives 8, and 5 is not
        reachable from 4 GPUs at all. Pinned so the corrected comment cannot
        quietly revert."""
        self.assertEqual(self._resolve(4, 8, ranks_per_gpu=2)[0], 8)
        # The script still QUOTES the old claim in order to correct it, so
        # test for the correction rather than the absence of the string.
        self.assertIn("that is wrong", self.src)
        self.assertIn("RANKS_PER_GPU=2 gives N_COMPUTE=8, not 5", self.src)


class ThreeMonthTwinTest(unittest.TestCase):
    """``submit_gf_3mo_v8_4gpu.sh`` is the 6mo script at 3 months.

    Written 2026-09-18 to the spec "use all the updates and base it on the
    6mo run, but make sure the high level 3mo things are there (Tobs, no
    emris/sobhbs/mbhbs)" plus "(no warmstart)".

    The failure this guards is a silent one. The two files are 97% the
    same text, so the obvious way to carry a 6mo fix across is to copy the
    block -- and the blocks that must NOT be copied are exactly the ones
    that look like ordinary knobs (``TOBS_TARGET``, ``GB_N_SUBBANDS``). A
    3-month run that quietly analysed 6 months of data, or armed the
    source branches, would produce plausible output and waste the
    campaign.
    """

    def setUp(self):
        self.three = _exports(THREE_MO)
        self.six = _exports(SIX_MO)

    def test_tobs_and_its_derived_settings_are_the_3mo_values(self):
        self.assertEqual(self.three["TOBS_TARGET"], "7776000")
        self.assertEqual(self.three["GB_NLEAVES_MAX"], "10000")
        self.assertEqual(self.three["GB_N_SUBBANDS"], "32768")
        self.assertEqual(self.three["GB_RJ_INMODEL_CHUNK"], "65536")
        self.assertEqual(self.three["COARSE_Q"], "1")
        self.assertEqual(self.three["COARSE_GPU_MODE"], "off")
        self.assertEqual(self.three["BASE_FILE_NAME"], "gf_prod_3mo")
        # 6mo-only; the 3mo arm takes the defaults
        self.assertNotIn("SIGHET_NT_LAYER", self.three)
        self.assertNotIn("EDGE_CROP_WAVELETS", self.three)

    def test_source_id_vars_are_UNSET_not_set_empty(self):
        """Set-empty looks equivalent and breaks the import (2026-09-18).

        Two readers disagree about what "empty" means:

        * ``source_runtime.default_source_ids`` seeds the SETTINGS and
          tests ``os.environ.get(f"{cls}_IDS") is not None`` -- so a
          set-but-empty var REPLACES the class default
          ``{"MBHB": [], "EMRI": [1], "SOBHB": []}`` with three empty
          lists;
        * ``run_combined_staged._source_ids_from_env`` ARMS the branches
          and reads ``os.environ.get(env, "")`` -- unset and set-empty are
          identical to it.

        With all three set-empty, importing
        ``lisatools.globalfit.stock.erebor`` FAILS outright: the module
        builds its stock registry eagerly, and ``FullYearCombinedGlobalFit``
        -- a variant this run never uses -- raises "mojito_source_ids must
        inject at least 1 source total". The run died before it built
        anything.
        """
        for knob in ("MBHB_IDS", "EMRI_IDS", "SOBHB_IDS"):
            self.assertNotIn(
                knob, self.three,
                f"{knob} must be UNSET, not set-empty: a set-empty value "
                f"overrides the settings default and makes `import erebor` "
                f"raise from an unrelated variant's constructor")

    def test_unset_ids_still_leave_every_source_branch_unarmed(self):
        """The other half: unset must not accidentally ARM anything."""
        import sys
        sys.path.insert(0, os.path.join(ROOT, "scripts", "fstat_proposal"))
        from run_combined_staged import _source_ids_from_env

        from lisatools.globalfit.stock.erebor.source_runtime import (
            default_source_ids,
        )

        saved = {k: os.environ.pop(k, None)
                 for k in ("MBHB_IDS", "EMRI_IDS", "SOBHB_IDS")}
        try:
            self.assertEqual(_source_ids_from_env(), {},
                             "no source branch may be armed")
            # ... while the SETTINGS default still injects something, which
            # is what keeps the eager registry constructible.
            self.assertGreaterEqual(
                sum(len(v) for v in default_source_ids().values()), 1,
                "the settings default must stay non-empty or `import "
                "erebor` raises again")
        finally:
            for k, v in saved.items():
                if v is not None:
                    os.environ[k] = v

    def test_the_data_carries_no_source_streams(self):
        self.assertEqual(self.three["SOURCE_TYPES"], "NOISE,GB,VGB")

    def test_the_source_search_skip_is_not_set(self):
        """It is an ERROR, not a no-op, with no armed sources.

        ``run_combined_staged`` raises "STAGE_SKIP_SOURCE_SEARCH=1 but no
        source branches are armed -- there is no source_search stage to
        skip." The 6mo script sets it; carrying it across would kill this
        run at startup, which is how it was found.
        """
        self.assertNotIn("STAGE_SKIP_SOURCE_SEARCH", self.three)

    def test_no_warm_start(self):
        self.assertEqual(
            self.three.get("GB_WARM_START_COMPONENTS", ""), "",
            "a 3-month run is the SOURCE of warm-start components, not a "
            "consumer; empty is the documented off switch")

    def test_every_other_knob_matches_the_6mo_script(self):
        """The whole point of the merge: only the listed knobs differ."""
        allowed = {
            "TOBS_TARGET", "GB_NLEAVES_MAX", "GB_N_SUBBANDS",
            "GB_RJ_INMODEL_CHUNK", "SIGHET_NT_LAYER", "EDGE_CROP_WAVELETS",
            "COARSE_Q", "COARSE_GPU_MODE", "BASE_FILE_NAME",
            "SOURCE_TYPES", "MBHB_IDS", "EMRI_IDS", "SOBHB_IDS",
            "GB_WARM_START_COMPONENTS", "GB_WARM_START_SOURCE_STORE",
            "STAGE_SKIP_SOURCE_SEARCH",
            # sig-het memory sizing, split 2026-09-18. The 6-month run OOM'd
            # in gb_search inside bin_fold_real (the fold chunker sizes each
            # chunk to FILL the byte cap, so an 8 GiB cap builds an ~8 GiB
            # transient) and took the revert its own comment block prescribes
            # for the full_pe handoff. The 3-month run's slots are ~0.25 MB
            # against the 6mo ~0.5 MB and it has been healthy through
            # iteration 210, so it keeps the aggressive sizing. All four are
            # transient knobs -- no stored number depends on them -- and both
            # scripts make them env-overridable, so the split is a default,
            # not a fork.
            "GB_INMODEL_SETUP_BATCH", "GB_SIGHET_FOLD_MAX_BYTES",
            "GB_INFOMAT_MEMPOOL_FREE", "GB_INMODEL_BATCH_MEMPOOL_FREE",
        }
        keys = (set(self.three) | set(self.six)) - {"_", "SHLVL", "PWD"}
        diff = {k for k in keys
                if self.three.get(k, "<unset>") != self.six.get(k, "<unset>")}
        unexpected = diff - allowed
        self.assertEqual(
            unexpected, set(),
            f"these knobs drifted apart and are not on the 3mo reversion "
            f"list: {sorted(unexpected)}")

    def test_the_multirank_and_correctness_updates_came_across(self):
        """The reason to derive from the 6mo file rather than the old 3mo one."""
        for knob, value in (("VGB_CHIRP_MASS_BASIS", "1"),
                            ("VGB_SIGHET_INMODEL", "1"),
                            ("VGB_INMODEL_PROPOSAL", "observable"),
                            ("GB_INMODEL_OBSERVABLE_EIGEN", "full"),
                            ("GB_LEAF_CAP_MIN_ITERS", "3"),
                            ("NWALKERS", "4"),
                            ("SIGHET_TUKEY_ALPHA", "0.01"),
                            ("SOBBH_EIGEN_SCOPE", "walker_max")):
            self.assertEqual(self.three.get(knob), value, knob)


class ThreeMonthV9TwinTest(unittest.TestCase):
    """``submit_gf_3mo_v9_2gpu.sh`` is ``submit_gf_6mo_v9_4gpu.sh`` at 3 months.

    Same silent-failure shape the v8 twin above guards, and the same remedy:
    the two files are ~99% the same text, so the obvious way to carry a fix
    across is to copy the block -- and the blocks that must NOT be copied are
    exactly the ones that look like ordinary knobs. A 3-month run that quietly
    analysed 6 months of data, armed the source branches, or fitted the
    foreground in its first stage would produce entirely plausible output and
    waste the allocation.

    Design + the provenance of every number:
    ``docs/superpowers/specs/2026-09-26-3mo-v9-2gpu-design.md``.
    """

    # the 90-day add-back fixed point, before inflation
    GALFOR_BASE_AMP = 1.436180605904e-44
    GALFOR_BASE_FK = 2.533915614978e-03

    def setUp(self):
        self.three = _exports(THREE_MO_V9)
        self.six = _exports(SIX_MO_V9)

    # -- 3MO-1: Tobs and only its derived settings ----------------------
    def test_tobs_and_its_derived_settings_are_the_3mo_values(self):
        self.assertEqual(self.three["TOBS_TARGET"], "7776000")
        self.assertEqual(self.three["GB_NLEAVES_MAX"], "10000")
        self.assertEqual(self.three["GB_N_SUBBANDS"], "32768")
        self.assertEqual(self.three["GB_RJ_INMODEL_CHUNK"], "65536")
        self.assertEqual(self.three["BASE_FILE_NAME"], "gf_prod_3mo")
        # 6mo-only; at Nt=2160 the CODE DEFAULTS are the validated values
        # (nt_layer 64 -> snaps to 60, stride 36 = 36 h; taper 11 + margin 8
        # = 19 <= the default crop 20).
        self.assertNotIn("SIGHET_NT_LAYER", self.three)
        self.assertNotIn("EDGE_CROP_WAVELETS", self.three)

    def test_the_sighet_STAGING_knobs_do_NOT_revert(self):
        """The v8 3mo arm's aggressive sizing is deliberately NOT taken.

        ``GB_N_SUBBANDS=32768`` is only safe because these four keep the 6mo
        post-OOM values: the fold chunker sizes each chunk to FILL
        ``GB_SIGHET_FOLD_MAX_BYTES``, so an 8 GiB cap builds an ~8 GiB
        transient by design, and that plus one-block staging is what killed
        the 6mo run in ``bin_fold_real``. Pinned as a PAIR with the slot
        count above, because raising one without the other is the failure.
        """
        for knob, want in (("GB_INMODEL_SETUP_BATCH", "2048"),
                           ("GB_SIGHET_FOLD_MAX_BYTES", "1073741824"),
                           ("GB_INFOMAT_MEMPOOL_FREE", "1"),
                           ("GB_INMODEL_BATCH_MEMPOOL_FREE", "1")):
            self.assertEqual(self.three[knob], want, knob)
            self.assertEqual(self.three[knob], self.six[knob], knob)

    # -- 3MO-2: no warm start, no seed stage, no seed store -------------
    def test_no_warm_start(self):
        self.assertEqual(
            self.three.get("GB_WARM_START_COMPONENTS", "<unset>"), "",
            "a 3-month run is the SOURCE of warm-start components, not a "
            "consumer; EMPTY (not unset) is the documented off switch, and "
            "the 6mo `${VAR-default}` form would re-arm it if the line went")

    def test_no_seed_stage_and_no_cadence_knob_left_dangling(self):
        self.assertEqual(self.three["GB_SEARCH_SEED_ITERS"], "0")
        self.assertNotIn(
            "GB_SEARCH_3_WARM_EVERY", self.three,
            "it cadences rj_warm_search, and there is no warm move to "
            "cadence -- a knob that resolves and reaches nothing")

    def test_the_seed_store_is_gone_entirely(self):
        for knob in ("GF_SEED_STORE", "GB_WARM_START_SOURCE_STORE",
                     "GB_WARM_START_SOURCE_TOBS", "GB_WARM_START_LAST_K",
                     "GB_WARM_START_FLOOR_EPS", "GB_WARM_START_CIRC_IMAGES"):
            self.assertNotIn(knob, self.three, knob)

    def test_replace_cannot_run_a_warm_pass_it_has_no_container_for(self):
        self.assertEqual(self.three["GB_REPLACE_WARM_PASS"], "0")

    # -- 3MO-3: psd-only first stage, galfor frozen at the reference -----
    def test_the_first_stage_samples_the_psd_alone(self):
        self.assertEqual(self.three["STAGE_NOISE_PSD_ONLY"], "1")
        self.assertEqual(self.three["STAGE_SKIP_NOISE_VGB"], "1")
        self.assertEqual(self.three["NOISE_SEARCH_CHECKS"], "5")
        # global to JointMaxLogLSearch, so it governs this stage too
        self.assertEqual(self.three["MAXLOGL_TOL"], "20")

    def test_PSD_START_PARAMS_is_unset_or_the_first_stage_disappears(self):
        """Pinning it flips ``_noise_pinned`` and the noise stages are SKIPPED.

        That would delete this run's entire first stage while every log line
        still read as though the psd had been fitted.
        """
        self.assertNotIn("PSD_START_PARAMS", self.three)

    def test_the_foreground_reference_is_the_INFLATED_3mo_fixed_point(self):
        vals = [float(x) for x in
                self.three["GALFOR_START_PARAMS"].split(",")]
        self.assertEqual(len(vals), 5)
        amp, fk, alpha, f_1, f_2 = vals
        self.assertAlmostEqual(amp / self.GALFOR_BASE_AMP, 1.25, places=9,
                               msg="amp inflation (user ruling 2026-09-26)")
        self.assertAlmostEqual(fk / self.GALFOR_BASE_FK, 1.10, places=9,
                               msg="knee inflation: x1.06 is the "
                                   "self-consistent G^(3/11) partner of "
                                   "x1.25 and x1.20 reproduces the 6mo "
                                   "k~2.0 pathology, so 1.10 is the point")
        self.assertEqual(alpha, 5.0)
        self.assertAlmostEqual(f_2, 1.405721657329e-03, places=15)
        # every value strictly INSIDE its prior box -- f_1 in particular is
        # nudged off the 1e-2 rail the v9 launch audit flagged.
        for name, v, lo, hi in (("amp", amp, 1e-47, 1e-41),
                                ("fk", fk, 0.8e-3, 1e-2),
                                ("f_1", f_1, 1e-5, 1e-2),
                                ("f_2", f_2, 1e-5, 1e-2)):
            self.assertGreater(v, lo, name)
            self.assertLess(v, hi, f"{name} is ON or past its prior rail")

    # -- 3MO-4: data + source branches -----------------------------------
    def test_the_data_is_the_COMBINED_stream_with_no_source_branches(self):
        self.assertEqual(self.three["SOURCE_TYPES"], "COMBINED,GB,VGB")

    def test_source_id_vars_are_UNSET_not_set_empty(self):
        """Set-empty looks equivalent and breaks ``import erebor`` outright.

        ``source_runtime.default_source_ids`` tests ``... is not None``, so a
        set-but-empty value replaces the class default with three empty lists
        and ``FullYearCombinedGlobalFit`` -- a variant this run never uses --
        raises from the eagerly-built stock registry.
        """
        for knob in ("MBHB_IDS", "EMRI_IDS", "SOBHB_IDS"):
            self.assertNotIn(knob, self.three, knob)

    def test_the_source_search_skip_is_not_set(self):
        """It is an ERROR, not a no-op, with no armed sources."""
        self.assertNotIn("STAGE_SKIP_SOURCE_SEARCH", self.three)

    # -- 3MO-6: the fresh store takes the correct reference fit ----------
    def test_the_unequal_arm_psd_reference_fit_is_taken(self):
        self.assertEqual(
            self.six["MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM"], "0",
            "the 6mo pin exists to keep its v8-lineage store resumable")
        self.assertNotIn(
            "MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM", self.three,
            "this store is FRESH, so it takes the default (=1, unequal arm) "
            "-- general.psd_injection is the truth line every monitor "
            "compares against, and the estimator was fitting EQUAL-arm while "
            "the run ran unequal")

    def test_every_other_knob_matches_the_6mo_v9_script(self):
        """The whole point of deriving from 6mo: only the listed knobs differ."""
        allowed = {
            # 3MO-1 Tobs + derived
            "TOBS_TARGET", "GB_NLEAVES_MAX", "GB_N_SUBBANDS",
            "GB_RJ_INMODEL_CHUNK", "SIGHET_NT_LAYER", "EDGE_CROP_WAVELETS",
            "BASE_FILE_NAME", "STORE_DIR", "SLURM_LOG",
            # 3MO-2 no warm start / no seed stage / no seed store
            "GB_WARM_START_COMPONENTS", "GB_WARM_START_SOURCE_STORE",
            "GB_WARM_START_SOURCE_TOBS", "GB_WARM_START_LAST_K",
            "GB_WARM_START_FLOOR_EPS", "GB_WARM_START_CIRC_IMAGES",
            "GF_SEED_STORE", "GB_SEARCH_SEED_ITERS", "GB_SEARCH_3_WARM_EVERY",
            "GB_REPLACE_WARM_PASS",
            # 3MO-3 psd-only first stage + the inflated reference
            "STAGE_NOISE_PSD_ONLY", "STAGE_SKIP_NOISE_VGB",
            "NOISE_SEARCH_CHECKS", "GALFOR_START_PARAMS",
            # 3MO-4 data + source branches
            "SOURCE_TYPES", "MBHB_IDS", "EMRI_IDS", "SOBHB_IDS",
            "STAGE_SKIP_SOURCE_SEARCH",
            # 3MO-6 the unequal-arm reference fit
            "MOJITO_PSD_REFERENCE_FIT_UNEQUAL_ARM",
        }
        keys = (set(self.three) | set(self.six)) - {"_", "SHLVL", "PWD"}
        diff = {k for k in keys
                if self.three.get(k, "<unset>") != self.six.get(k, "<unset>")}
        unexpected = diff - allowed
        self.assertEqual(
            unexpected, set(),
            f"these knobs drifted apart and are not on the 3mo v9 delta "
            f"list: {sorted(unexpected)}")

    def test_the_v9_search_machinery_came_across_intact(self):
        """The reason to derive from the 6mo v9 file rather than 3mo_v9.sh."""
        for knob in ("GB_INMODEL_CONVERGE", "GB_INMODEL_CONVERGE_ITERS",
                     "GB_INMODEL_CONVERGE_ITERS_SURVIVOR",
                     "GB_INMODEL_CONVERGE_CLASSES", "GB_INMODEL_GROUP",
                     "GB_NUM_REPEAT_PROPOSALS", "GB_RJ_DIRECT_BATCH",
                     "GB_RUN_FANCY_TEMPERING", "GB_TEMPER_VERTICAL",
                     "GB_SEARCH_BAND_SHUTOFF_PER_WALKER",
                     "GB_SEARCH_BAND_SHUTOFF_CONV_ITER",
                     "GB_FSTAT_REFIT_EVERY", "FSTAT_PEAK_MIN_SNR",
                     "GB_OPT_SNR_LIMIT_SEARCH", "GB_RJ_PHASE_MAXIMIZE",
                     "GB_LEAF_CAP_START", "STAGE_V9_SEARCH",
                     "COARSE_Q", "COARSE_GPU_MODE",
                     # sig-het accuracy: Tobs-independent, and staleness is
                     # LESS harmful at shorter Tobs, so 50 is safer here
                     "GB_SIGHET_REFRESH_EVERY", "SIGHET_N_CP",
                     "SIGHET_TUKEY_ALPHA", "GB_SIGHET_TRUST_PHASE_C",
                     "NWALKERS", "GB_NTEMPS"):
            self.assertEqual(self.three.get(knob), self.six.get(knob), knob)


class WarmStartPathSeparatorTest(unittest.TestCase):
    """The warm-start npz must land INSIDE the run store (2026-09-18).

    The default was written ``${STORE_DIR}warmstart/...`` with no
    separator, which relies on ``STORE_DIR`` ending in a slash. The
    built-in default does; a command-line override
    (``STORE_DIR=/.../gf_prod_6mo_v8_4gpu ./submit...``) does not, so the
    npz went to a SIBLING directory ``gf_prod_6mo_v8_4gpuwarmstart/`` --
    outside the snapshot zips, and not rebuilt when the store is reset.
    """

    def test_store_dir_and_warmstart_are_separated(self):
        for path in SCRIPTS:
            with open(path) as fh:
                src = fh.read()
            for m in re.finditer(r"\$\{STORE_DIR\}(?!/)(\S{0,24})", src):
                self.assertNotIn(
                    "warmstart", m.group(1),
                    f"{os.path.basename(path)}: "
                    f"${{STORE_DIR}}{m.group(1)} has no path separator, so "
                    f"an override without a trailing slash puts the warm "
                    f"start outside the store")


class SobbhKnobsSingleExportTest(unittest.TestCase):
    """One effective export per SOBBH knob (2026-09-17).

    ``SOBBH_EIGEN_SCOPE=walker_max`` (user ruling 2026-09-16) was exported
    once and then silently overridden by a second, older
    ``export SOBBH_EIGEN_SCOPE=per_walker`` further down the same file, so
    the ruling never reached a run. ``SOBBH_NTEMPS`` must be overridable
    from the environment (a store born at 8 rungs is resumed with
    ``SOBBH_NTEMPS=8``; the stored count wins either way, but the knob
    should not lie in run_settings.log).
    """

    def _exports(self, path, name):
        with open(path) as fh:
            return [
                line.strip() for line in fh
                if line.lstrip().startswith(f"export {name}=")
            ]

    def test_sobbh_eigen_scope_exported_once(self):
        for path in SCRIPTS:
            lines = self._exports(path, "SOBBH_EIGEN_SCOPE")
            self.assertLessEqual(
                len(lines), 1,
                f"{path}: SOBBH_EIGEN_SCOPE exported {len(lines)} times: {lines}",
            )

    def test_campaign_sobbh_eigen_scope_is_walker_max(self):
        # The ruling applies to the campaign script; the null-test script
        # keeps its own (per-walker) setting and is only held to one export.
        lines = self._exports(SCRIPTS[0], "SOBBH_EIGEN_SCOPE")
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("walker_max", lines[0])
        self.assertNotIn("per_walker", lines[0])

    def test_sobbh_repeats_is_env_overridable_and_defaults_to_10(self):
        # User ruling 2026-09-18. Repeats are the ONLY knob that moves the
        # SOBBH cost: [SOBBH_LL_TIMING] measured a flat 1.73 s per scoring
        # call regardless of rows, and calls come from repeats, not walkers
        # or rungs. 20 -> 10 halves the dominant per-iteration cost.
        lines = self._exports(SCRIPTS[0], "SOBBH_NUM_PROP_REPEATS")
        self.assertEqual(len(lines), 1, lines)
        self.assertEqual(
            lines[0], "export SOBBH_NUM_PROP_REPEATS=${SOBBH_NUM_PROP_REPEATS:-10}")

    def test_sobbh_ntemps_is_env_overridable_and_defaults_to_8(self):
        # User ruling 2026-09-17: 8 in the scripts (the 4-GPU store is an
        # 8-rung store; resumed 12-rung stores keep 12 via the store-wins
        # rule). Both campaign scripts.
        for path in SCRIPTS:
            lines = self._exports(path, "SOBBH_NTEMPS")
            self.assertEqual(len(lines), 1, f"{path}: {lines}")
            self.assertEqual(
                lines[0], "export SOBBH_NTEMPS=${SOBBH_NTEMPS:-8}", path)


class SubmitScriptsSyntaxTest(unittest.TestCase):
    def test_bash_syntax(self):
        for path in SCRIPTS:
            result = subprocess.run(
                ["bash", "-n", path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(
                result.returncode, 0, f"{path}: bash -n failed: {result.stderr}"
            )


class SubmitScriptsInJobBlockTest(unittest.TestCase):
    """The IN-JOB rank-layout block (after the `exec sbatch` self-dispatch).

    The stub-sbatch scenarios below can only observe the PRE-submit dispatch
    block -- everything after `exec sbatch` never runs without a real job --
    so these invariants are checked as text. They are the ones that only bite
    a launch that bypassed the dispatch block entirely (a manual
    `sbatch --nodes=2 --ntasks=5 <script>`), which is exactly the path no
    other test covers.
    """

    def _text(self, path):
        with open(path) as fh:
            return fh.read()

    def test_multi_node_forces_walker_block_layout_in_job(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn(
                    'if [ "${SLURM_NNODES:-1}" -gt 1 ] && '
                    '[ "${GF_LEGACY_RANK_LAYOUT}" = "1" ]; then',
                    text,
                )
                self.assertIn("FORCING GF_LEGACY_RANK_LAYOUT=0", text)

    def test_in_job_exports_all_three_layout_knobs(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn("export GPUS_PER_RANK RANKS_PER_GPU", text)
                self.assertIn("export GF_LEGACY_RANK_LAYOUT", text)

    def test_nwalkers_modulo_is_guarded_against_zero_compute_ranks(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn('[ "${N_COMPUTE_EFF}" -gt 0 ]', text)
                # the division itself must not be reachable unguarded
                self.assertNotIn(
                    'if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && '
                    "[ $(( NWALKERS % N_COMPUTE_EFF )) -ne 0 ]; then",
                    text,
                )

    def test_one_walker_is_exempt_from_the_divisibility_fix(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn('[ "${NWALKERS}" -eq 1 ]', text)
                self.assertIn("one-walker replica mode", text)
                # The sampler-shape export must honour the submitting shell's
                # NWALKERS (carried by --export=ALL), or the branch below it is
                # unreachable: a hard `export NWALKERS=<n>` above the `-eq 1`
                # test would silently run n walkers for `NWALKERS=1 ./submit`.
                # The DEFAULT VALUE is deliberately not pinned here -- it is a
                # campaign choice that moves (10 at the 2026-09-11 rebase, 4
                # for the walker-block store on 2026-09-18); what this test
                # protects is the overridable FORM and its position.
                m = re.search(r"^export NWALKERS=\$\{NWALKERS:-\d+\}",
                              text, re.M)
                self.assertIsNotNone(
                    m, "NWALKERS must be exported as ${NWALKERS:-<default>}")
                self.assertNotRegex(text, r"\nexport NWALKERS=\d+\s")
                self.assertLess(
                    m.start(), text.index('[ "${NWALKERS}" -eq 1 ]'))
                # the rounding branch survives for NWALKERS > 1
                self.assertIn(
                    "(NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF", text
                )
                # the exemption is tested BEFORE the rounding branch
                self.assertLess(
                    text.index('[ "${NWALKERS}" -eq 1 ]'),
                    text.index(
                        "(NWALKERS / N_COMPUTE_EFF + 1) * N_COMPUTE_EFF"
                    ),
                )

    def test_multinode_launch_line_is_hydra_round_robin(self):
        # WP7 Step 0 (2026-09-16): on this cluster `srun --mpi=pmix` bootstraps
        # Intel MPI but its OFI address exchange fails, and a bare `srun` gives
        # three size-1 worlds; the launcher that works is hydra with the SLURM
        # bootstrap, `-ppn 1` (round-robin over hosts = cyclic placement) and
        # the tcp fabric provider. Pin all of it so a refactor cannot drift
        # back to `srun`.
        for path in SCRIPTS:
            with self.subTest(script=path):
                text = self._text(path)
                self.assertIn(
                    'mpiexec -n "${SLURM_NTASKS:-3}" -ppn 1 python', text,
                )
                self.assertIn(
                    "export I_MPI_HYDRA_BOOTSTRAP=slurm I_MPI_FABRICS=shm:ofi "
                    "FI_PROVIDER=tcp",
                    text,
                )
                # ... and the knob that makes `-ppn 1` bind at all (2026-09-18):
                # with the SLURM bootstrap hydra otherwise honours the
                # scheduler's per-node task counts, so 5 tasks over 2 nodes
                # land 3/2 (block) instead of A,B,A,B,A (cyclic) and
                # build_layout refuses -- 3 compute ranks on a 2-GPU node.
                self.assertIn(
                    "export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=0", text,
                )
                self.assertNotIn('srun --ntasks="${SLURM_NTASKS', text)


class SubmitScriptsNwalkersBlockTest(unittest.TestCase):
    """Execute the in-job NWALKERS divisibility/exemption block with bash.

    The text checks in ``SubmitScriptsInJobBlockTest`` confirm the exemption
    line and message exist and come first; this class actually runs the
    extracted `if ... fi` block under bash to confirm the one-walker branch
    leaves NWALKERS untouched and the rounding branch still fires (and still
    rounds correctly) for every other NWALKERS value.
    """

    _START_MARKER = (
        'if [ "${GF_LEGACY_RANK_LAYOUT}" = "0" ] && '
        '[ "${N_COMPUTE_EFF}" -gt 1 ] && [ "${NWALKERS}" -eq 1 ]; then'
    )

    def _text(self, path):
        with open(path) as fh:
            return fh.read()

    def _extract_block(self, path):
        text = self._text(path)
        start = text.index(self._START_MARKER)
        end = text.index("\nfi", start)
        return text[start : end + len("\nfi")]

    def _run_block(self, path, overrides):
        block = self._extract_block(path)
        env = {"PATH": os.environ.get("PATH", "")}
        env.update(overrides)
        result = subprocess.run(
            ["bash", "-c", block + "\necho NW=${NWALKERS}"],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"{path} {overrides}: exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result.stdout

    def test_nwalkers_1_is_one_walker_replica_mode_and_unchanged(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                stdout = self._run_block(
                    path,
                    {
                        "GF_LEGACY_RANK_LAYOUT": "0",
                        "N_COMPUTE_EFF": "4",
                        "NWALKERS": "1",
                    },
                )
                self.assertIn("one-walker replica mode", stdout)
                self.assertIn("NW=1", stdout.splitlines())

    def test_nwalkers_6_still_rounds_up_to_8(self):
        for path in SCRIPTS:
            with self.subTest(script=path):
                stdout = self._run_block(
                    path,
                    {
                        "GF_LEGACY_RANK_LAYOUT": "0",
                        "N_COMPUTE_EFF": "4",
                        "NWALKERS": "6",
                    },
                )
                self.assertIn("is not a multiple of N_COMPUTE", stdout)
                self.assertIn("NW=8", stdout.splitlines())


class SubmitScriptsDispatchTest(unittest.TestCase):
    """Exercise the pre-submit `exec sbatch ...` dispatch block via a stub."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="submit_scripts_stub_")
        cls.stub_dir = cls._tmpdir.name
        stub_path = os.path.join(cls.stub_dir, "sbatch")
        with open(stub_path, "w") as fh:
            fh.write(_STUB_SBATCH)
        os.chmod(stub_path, 0o755)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def _run_dispatch(self, script, overrides):
        env = dict(os.environ)
        for key in _DISPATCH_ENV_KEYS:
            env.pop(key, None)
        env["PATH"] = self.stub_dir + os.pathsep + env.get("PATH", "")
        env.update(overrides)
        result = subprocess.run(
            ["bash", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"{script} {overrides}: exited {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        return result.stdout.splitlines()

    def _assert_export_contains(self, lines, needle):
        export_lines = [l for l in lines if l.startswith("--export=")]
        self.assertTrue(
            export_lines, f"no --export= argv line found; stdout lines: {lines}"
        )
        self.assertTrue(
            any(needle in l for l in export_lines),
            f"{needle!r} not found in --export= line(s): {export_lines}",
        )

    def test_ngpus_2_explicit_legacy_still_launches_today_shape(self):
        # GF_LEGACY_RANK_LAYOUT=1 is the rollback knob: same resource request,
        # --ntasks=3, mpiexec -n 3, the single-compute-rank layout.
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(
                    script, {"NGPUS": "2", "GF_LEGACY_RANK_LAYOUT": "1"}
                )
                self.assertIn("--ntasks=3", lines)
                self.assertIn("--nodes=1", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=1")

    def test_ngpus_2_default_is_walker_block(self):
        # user ruling 2026-09-16 (WP7 Steps 0-2/4 green): the walker-block
        # layout is the default at NGPUS=2 -- head + 1 compute + saver, still
        # --ntasks=3, but GF_LEGACY_RANK_LAYOUT=0 is what the job sees.
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(script, {"NGPUS": "2"})
                self.assertIn("--ntasks=3", lines)
                self.assertIn("--nodes=1", lines)
                self.assertIn("--partition=gpu-80-spot", lines)
                self.assertIn("--gres=gpu:2", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")

    def test_ngpus_4_forces_walker_block_two_nodes(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(script, {"NGPUS": "4"})
                self.assertIn("--ntasks=5", lines)
                self.assertIn("--nodes=2", lines)
                self.assertIn("--gres=gpu:2", lines)
                # NGPUS=4 is 2 nodes, so a spot preemption of EITHER node
                # kills the whole MPI world -- twice the exposure of the
                # 1-node flow. It goes to ON-DEMAND (user 2026-09-18);
                # NGPUS=2 above stays on spot deliberately.
                self.assertIn("--partition=gpu-80-ondemand", lines)
                self.assertNotIn("--partition=gpu-80-spot", lines)
                self.assertIn("--distribution=cyclic", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")

    def test_the_4gpu_partition_is_overridable_without_editing(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(
                    script, {"NGPUS": "4", "PARTITION": "gpu-80-spot"})
                self.assertIn("--partition=gpu-80-spot", lines)

    def test_nodes_knob_spreads_the_gpus_one_per_node(self):
        # one-walker replica gates (user ruling 2026-09-16: test ACROSS nodes):
        # NGPUS=2 NODES=2 -> 2 nodes x gpu:1, head + 1 compute + saver, cyclic
        for script in SCRIPTS:
            with self.subTest(script=script, nodes=2):
                lines = self._run_dispatch(script, {"NGPUS": "2", "NODES": "2"})
                self.assertIn("--nodes=2", lines)
                self.assertIn("--gres=gpu:1", lines)
                self.assertIn("--ntasks=3", lines)
                self.assertIn("--distribution=cyclic", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")
            with self.subTest(script=script, nodes=4):
                lines = self._run_dispatch(script, {"NGPUS": "4", "NODES": "4"})
                self.assertIn("--nodes=4", lines)
                self.assertIn("--gres=gpu:1", lines)
                self.assertIn("--ntasks=5", lines)
                self.assertIn("--distribution=cyclic", lines)
            with self.subTest(script=script, nodes="unset"):
                # unset NODES = the NGPUS table, unchanged
                lines = self._run_dispatch(script, {"NGPUS": "2"})
                self.assertIn("--nodes=1", lines)
                self.assertIn("--gres=gpu:2", lines)

    def test_nodes_knob_must_divide_ngpus(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                env = {k: v for k, v in os.environ.items() if k not in _DISPATCH_ENV_KEYS}
                env["PATH"] = self.stub_dir + os.pathsep + env.get("PATH", "")
                env.update({"NGPUS": "2", "NODES": "3"})
                result = subprocess.run(
                    ["bash", script], env=env, capture_output=True, text=True, timeout=60
                )
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("must be >= 1 and divide NGPUS=2", result.stdout)

    def test_ngpus_2_explicit_walker_block(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(
                    script, {"NGPUS": "2", "GF_LEGACY_RANK_LAYOUT": "0"}
                )
                self.assertIn("--ntasks=3", lines)
                self._assert_export_contains(lines, "GF_LEGACY_RANK_LAYOUT=0")

    def test_ngpus_2_gpus_per_rank_2_walker_block(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                lines = self._run_dispatch(
                    script,
                    {
                        "NGPUS": "2",
                        "GPUS_PER_RANK": "2",
                        "GF_LEGACY_RANK_LAYOUT": "0",
                    },
                )
                self.assertIn("--ntasks=2", lines)


if __name__ == "__main__":
    unittest.main()
