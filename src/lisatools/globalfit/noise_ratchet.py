"""The galfor RATCHET: nudge the foreground down, hold, release, repeat.

WHY (user design 2026-09-30). In the 6-month v9 run the fitted foreground at
3-5 mHz sits 2-5x above the add-back estimate in TOTAL noise, because the
psd/galfor maximizer absorbs the unresolved-but-resolvable GBs and then hides
them: a true SNR-8 source at 4 mHz scores as SNR 3.6 against that curve, below
every floor. Released, the fit re-converges to the same state within two
iterations, so it is a stable trap. The ratchet breaks it by force:

  * NUDGE   -- shift the galfor coordinates DOWN on every rung and walker
               (a deterministic step in the sampled log10 basis), publish the
               new noise through the noise move's own accept path, arm a HARD
               F-stat refit (the old grid selected peaks against the old
               curve), then run one GB in-model pass so the sources settle to
               the new noise before any RJ move sees the residual;
  * HOLD    -- the noise moves do not run; births eat the exposed excess;
  * RELEASE -- the ordinary joint max-logL noise search runs again, and the
               same in-model pass follows it (the noise changed).

The release IS the measurement (user correction, same day: "we do not know
the 1-2 mHz side is fine -- we should see that the model brings that side of
the tanh back up"): where the released fit climbs back the model was honest,
where it stays down the fit had been absorbing resolvable power. So the
default nudge is BROAD (amplitude too), not aimed at 3-5 mHz alone.

Composition (``run_combined_staged._search_stage`` under GALFOR_RATCHET=1):
gb_search_3 carries ONE gated noise proposal at the head of the iteration,
:class:`NoiseRatchetGate`, in place of the leading rider plus the four
interleaved ``noise_joint_search_*`` slots. The recipe step
(``SearchStageProfileStep``) drives the gate's mode from the STAGE-LOCAL
iteration, so a resume lands in the right phase instead of re-nudging.

Knobs (all read at composition / stage time, default = ratchet OFF):

  GALFOR_RATCHET=1              arm
  GALFOR_RATCHET_HOLD=3         iterations held per cycle, INCLUDING the nudge one
  GALFOR_RATCHET_RELEASE=2      iterations of free noise search per cycle
  GALFOR_RATCHET_CYCLES=2       number of nudges; afterwards permanent release
  GALFOR_RATCHET_DLOG10_AMP=-0.05   nudge, sampled (log10) basis
  GALFOR_RATCHET_DLOG10_FK=-0.10
  GALFOR_RATCHET_DLOG10_F2=-0.15

STEP SIZE (user instruction 2026-09-30: "make sure your step size in the
forcing downward is sufficient"). Sized against the 6mo row-47 fit and the
add-back estimate with the stock sangria A-channel instrument PSD: the total
noise at 3 / 3.5 / 4 / 4.5 / 5 mHz is 1.9 / 3.1 / 5.0 / 4.7 / 2.9 times the
estimate, so the F-stat peak floor (SNR 6.25 in model units) admits only
true SNR 8.6 / 11 / 14 / 13.5 / 10.6 there. One default nudge brings the
total to 0.75 / 0.97 / 1.4 / 1.46 / 1.2 times the estimate: the peak floor
becomes true 5.4 / 6.2 / 7.4 / 7.5 / 6.8 and a true SNR-8 source scores as
9.2 / 8.1 / 6.8 / 6.6 / 7.3 -- visible to the grid everywhere in the band.
The smaller step first considered (fk -0.05, f_2 -0.10) left SNR-8 at 5.3
at 4 mHz, still under the peak floor, i.e. insufficient. CYCLES defaults to
2 (user ruling 2026-09-30: "do at least 2 cycles for now"). ⚠ The second
nudge applies the SAME delta to whatever the release left: on a fit that
climbed back it repeats the first experiment; on a fit that stayed down it
lands at 0.2-0.6 of the estimate below 4.5 mHz (a model floor of 5 = true
2.3-3.9), the noise-birth regime -- so read the second cycle's release with
the junk indicators (birth truth-partner fraction, power share per band).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from logging import getLogger

import numpy as np

from .moves.globalfitmove import GFCombineMove

logger = getLogger(__name__)

__all__ = [
    "GALFOR_RATCHET_DEFAULT_DELTA",
    "GALFOR_LOG10_COLUMNS",
    "NoiseRatchetGate",
    "RatchetSchedule",
    "galfor_curve_ratio",
    "is_noise_ratchet_gate",
    "nudge_delta_from_env",
    "ratchet_from_env",
]

#: The sampled galfor basis under GALFOR_LOG_SAMPLING=1 (the production
#: runs): ``[log10 amp, log10 fk, alpha, log10 f_1, log10 f_2]``.
GALFOR_LOG10_COLUMNS = (0, 1, 3, 4)

#: The broad nudge: amplitude -0.05 dex, knee -0.10 dex, transition width
#: -0.15 dex; alpha and the exponential roll-off untouched. See STEP SIZE in
#: the module docstring for why these and not smaller.
GALFOR_RATCHET_DEFAULT_DELTA = np.array([-0.05, -0.10, 0.0, 0.0, -0.15])

_ACTIONS = ("nudge", "hold", "release")
_TRUE = ("1", "true", "True", "yes", "on")


@dataclass(frozen=True)
class RatchetSchedule:
    """Stage-local iteration ``k`` -> ``"nudge" | "hold" | "release"``.

    A cycle is ``hold`` iterations (the first of which is the nudge) followed
    by ``release`` iterations; after ``cycles`` cycles every iteration is a
    release.
    """

    hold: int
    release: int
    cycles: int
    #: User design 2026-10-02: start with a RELEASE (the noise search to
    #: convergence) before the first nudge. On a store whose noise was pinned
    #: through the earlier stages the first release IS the first fit of the
    #: noise to the data; every nudge after it is then measured against a
    #: converged baseline rather than against the pinned start vector.
    release_first: bool = False

    def __post_init__(self):
        for name in ("hold", "release", "cycles"):
            v = getattr(self, name)
            if int(v) != v or int(v) < 1:
                raise ValueError(
                    f"RatchetSchedule.{name}={v!r} must be an integer >= 1 "
                    f"(hold counts the nudge iteration itself).")
            object.__setattr__(self, name, int(v))
        object.__setattr__(self, "release_first", bool(self.release_first))

    @property
    def cycle_length(self) -> int:
        return self.hold + self.release

    @property
    def total_iterations(self) -> int:
        """Stage-local iterations the schedule spans before "release forever"."""
        return self.cycles * self.cycle_length + (1 if self.release_first else 0)

    def _shift(self, k: int) -> int:
        k = int(k)
        if k < 0:
            raise ValueError(f"stage-local iteration {k} < 0")
        return k - 1 if self.release_first else k

    def cycle_of(self, k: int) -> int:
        """1-based cycle number of stage-local iteration ``k`` (0 = the
        leading release of a release-first schedule)."""
        ks = self._shift(k)
        return 0 if ks < 0 else ks // self.cycle_length + 1

    def action(self, k: int) -> str:
        ks = self._shift(k)
        if ks < 0:
            return "release"
        if ks // self.cycle_length >= self.cycles:
            return "release"
        p = ks % self.cycle_length
        if p == 0:
            return "nudge"
        return "hold" if p < self.hold else "release"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer.") from None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a number.") from None


def ratchet_from_env():
    """``RatchetSchedule`` when ``GALFOR_RATCHET`` arms it, else ``None``.

    A malformed knob RAISES rather than quietly running unratcheted: the
    silent knob-reaches-nothing failure is the documented one in this tree.
    """
    if os.environ.get("GALFOR_RATCHET", "0").strip() not in _TRUE:
        return None
    return RatchetSchedule(
        hold=_env_int("GALFOR_RATCHET_HOLD", 3),
        release=_env_int("GALFOR_RATCHET_RELEASE", 2),
        # TWO nudges by default (user ruling 2026-09-30, "do at least 2
        # cycles for now"); see STEP SIZE in the module docstring for what the
        # second one does to a fit that did not climb back. With
        # GALFOR_RATCHET_MIN_GAIN set this is a CEILING: the data-driven stop
        # (min_gain_from_env) ends the ratcheting earlier.
        cycles=_env_int("GALFOR_RATCHET_CYCLES", 2),
        release_first=os.environ.get("GALFOR_RATCHET_RELEASE_FIRST", "0").strip() in _TRUE,
    )


def min_gain_from_env() -> float:
    """``GALFOR_RATCHET_MIN_GAIN``: the data-driven stop of the ratchet.

    User design 2026-10-02: "do two step cycles (ratchet/hold, release) until
    the max logL does not change by some threshold. Something that is a real
    step. This first ratchet went up by ~1000s. If get a step that causes
    less than ~200 logL, no more ratcheting." After every RELEASE iteration
    the step records the maximum cold log-likelihood over the walkers; when
    the gain over the previous release's maximum is below this many nats the
    schedule stops nudging: the gate stays released (rider mode) and the
    stage ends on its ordinary per-band shut-off rule. 0 (the default) keeps
    the schedule alone in charge.
    """
    v = _env_float("GALFOR_RATCHET_MIN_GAIN", 0.0)
    if v < 0:
        raise ValueError(f"GALFOR_RATCHET_MIN_GAIN={v} must be >= 0 (0 = off).")
    return v


def nudge_delta_from_env() -> np.ndarray:
    """The per-nudge shift of the galfor coordinates, sampled (log10) basis."""
    d = np.array(GALFOR_RATCHET_DEFAULT_DELTA, copy=True)
    d[0] = _env_float("GALFOR_RATCHET_DLOG10_AMP", d[0])
    d[1] = _env_float("GALFOR_RATCHET_DLOG10_FK", d[1])
    d[4] = _env_float("GALFOR_RATCHET_DLOG10_F2", d[4])
    return d


def _to_physical(v) -> tuple:
    v = np.asarray(v, dtype=float)
    out = list(v)
    for c in GALFOR_LOG10_COLUMNS:
        out[c] = 10.0 ** v[c]
    return tuple(out)


def galfor_curve_ratio(v_new, v_old, f_hz) -> np.ndarray:
    """``S_gal(f; v_new) / S_gal(f; v_old)`` for two sampled-basis vectors.

    The readout for the [GALFOR_RATCHET] log lines: normalization-free, so it
    reads the same whatever the instrument PSD does.
    """
    from lisatools.stochastic import HyperbolicTangentGalacticForeground as HT

    f = np.asarray(f_hz, dtype=float)
    return np.asarray(HT.specific_Sh_function(f, *_to_physical(v_new))
                      / HT.specific_Sh_function(f, *_to_physical(v_old)))


def galfor_curves(v_rows, f_hz) -> np.ndarray:
    """``(n_rows, n_f)`` of ``S_gal(f; v)`` for a stack of sampled-basis vectors.

    The readout's per-WALKER form. The curve of the walkers' MEAN parameters
    is not the mean of their curves once ``alpha``/``f1`` split across
    walkers (6mo job 695, row 81: alpha 2.3/3.0/8.6/9.8 -- the curve of the
    mean read +2..5 % at 3.5-5 mHz where the mean of the curves read -6..7 %),
    so the log compares mean-of-curves to mean-of-curves.
    """
    from lisatools.stochastic import HyperbolicTangentGalacticForeground as HT

    f = np.asarray(f_hz, dtype=float)
    rows = np.atleast_2d(np.asarray(v_rows, dtype=float))
    return np.asarray([HT.specific_Sh_function(f, *_to_physical(v)) for v in rows])


def is_noise_ratchet_gate(obj) -> bool:
    return bool(getattr(obj, "is_noise_ratchet_gate", False))


def reset_maxlogl_search(move) -> bool:
    """Forget a :class:`MaxLogLCombineMove`'s plateau bookkeeping.

    The chunked plateau loop keeps its per-walker baselines, flat counters
    and the ``maxlogl_plateau_done`` verdict ON THE INSTANCE across propose
    calls, so that a killed stage resumes mid-search. Inside a held stage
    that persistence has a second effect: once the loop has declared a
    plateau, every later propose takes ONE inner round and returns ("a
    plateaued instance keeps taking one inner iteration per call"). 6mo job
    675's first release ran 17 rounds and declared done; its second release
    ran one round in 2 s against a residual that had changed by a whole GB
    cycle. User ruling 2026-10-02: the psd and galfor branches run in
    SEARCH mode until the log-likelihood converges -- on every release.

    Returns True when there was state to clear. Safe on any object.
    """
    had = hasattr(move, "_ml_state") or bool(getattr(move, "maxlogl_plateau_done", False))
    if hasattr(move, "_ml_state"):
        try:
            del move._ml_state
        except AttributeError:
            pass
    try:
        move.maxlogl_plateau_done = False
    except AttributeError:
        pass
    return had


class NoiseRatchetGate(GFCombineMove):
    """The one noise proposal of a ratcheted search stage.

    Wraps the built joint max-logL noise search (``inner``) so the move tree
    stays walkable (``.moves == [inner]``), and holds the galfor
    :class:`~lisatools.globalfit.moves.psdmove.PSDMove` that performs the
    forced step plus the GB in-model move that follows every noise change.

    ``mode`` is set per iteration by the recipe step
    (:meth:`~lisatools.globalfit.recipe.SearchStageProfileStep._drive_ratchet`);
    a gate nobody drives is a plain release, i.e. the ordinary noise search.
    """

    is_noise_ratchet_gate = True
    #: Search legs: True right after a propose that CHANGED the noise (nudge
    #: or release, followed by the in-model pass), so the stage combine ends
    #: the leg here and a row is saved after this move (user ruling
    #: 2026-09-30: "a save after the in-model noise step whenever it runs").
    #: Consumed by the combine; False after a hold.
    gf_leg_end_now = False

    def __init__(self, inner, galfor_move, delta, in_model_move=None,
                 release_to_convergence=True, **kwargs):
        super().__init__([inner], share_temperature_control=False, **kwargs)
        self.inner = inner
        self.galfor_move = galfor_move
        self.delta = np.asarray(delta, dtype=float)
        if self.delta.shape != (5,):
            raise ValueError(
                f"galfor nudge delta must have 5 entries (sampled basis); got "
                f"shape {self.delta.shape}.")
        self.in_model_move = in_model_move
        #: Every RELEASE starts the inner max-logL search afresh
        #: (:func:`reset_maxlogl_search`), so it runs to ITS plateau rule
        #: against the current residual instead of inheriting the verdict
        #: of an earlier release (user ruling 2026-10-02). False restores
        #: the rider behaviour: one round per call once plateaued.
        self.release_to_convergence = bool(release_to_convergence)
        self.mode = "release"
        self.nudges = 0
        self.releases = 0

    def set_mode(self, mode: str) -> None:
        if mode not in _ACTIONS:
            raise ValueError(f"ratchet mode {mode!r} not in {_ACTIONS}")
        self.mode = mode

    #: Set by :meth:`finish_ratchet`: no more nudges; the gate stays a plain
    #: release in RIDER mode (one search round per iteration, tracking the
    #: residual as sources are added and removed) while the stage runs to
    #: its ordinary per-band shut-off.
    ratchet_finished = False

    def finish_ratchet(self) -> None:
        """The data-driven stop (user design 2026-10-02): the last release
        gained less than the threshold, so the ratcheting is over."""
        self.mode = "release"
        self.release_to_convergence = False
        self.ratchet_finished = True

    def _propose_moves(self, model, state):
        self.gf_leg_end_now = False
        if self.mode == "hold":
            return state, np.zeros(np.shape(state.log_like), dtype=bool)
        if self.mode == "nudge":
            self._gf_precondition(self.galfor_move, model)
            state, accepted = self.galfor_move.forced_noise_step(
                model, state, {"galfor": self.delta})
            self.nudges += 1
            # a nudge iteration is the first HELD iteration of its cycle;
            # the step re-drives the mode before the next iteration anyway.
            self.mode = "hold"
        else:
            if self.release_to_convergence:
                # search mode, from scratch, every release: the plateau rule
                # (num_checks flat rounds within tol) decides when the noise
                # has converged on THIS iteration's residual
                reset_maxlogl_search(self.inner)
            self._gf_precondition(self.inner, model)
            state, accepted = self.inner.propose(model, state)
            self.releases += 1
        if self.in_model_move is not None:
            # the noise changed (forced or free): let the GB sources settle
            # to it before any RJ move scores against the new residual.
            self._gf_precondition(self.in_model_move, model)
            state, _ = self.in_model_move.propose(model, state)
        # the noise changed: under search legs this ends the leg here
        self.gf_leg_end_now = True
        return state, accepted
