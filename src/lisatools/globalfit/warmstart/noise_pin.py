"""Extract a previous run's maxlogL NOISE point as a start pin for the next.

USER RULING 2026-09-24, for the v9 three-stage GB search:

    *"make sure the warmstart and PSD/GB frozen start point come from the
    same folder (3mo 10 walker noise fix). It should use maxlogL for
    PSD/GB."*

The v9 recipe's first two search stages do not sample the noise model at
all -- "fixed noise" is the ABSENCE of the psd/galfor moves. That is only
correct if the run STARTS at a sensible noise point, and until this existed
it could not: psd and galfor are the only sampled branches with no
start-coordinate path in ``run.py`` (they are in
``_LOAD_INFO_NAMED_BRANCHES``, so the generic injection seeder skips them),
so their chains began at a PRIOR DRAW. Two whole stages of GB search against
a random noise level is the failure this module exists to prevent.

WHAT IT DOES. Reads the best-logL cold walker of the last valid row of a
previous run's store (:func:`~lisatools.globalfit.warmstart.opt_snr.best_logl_noise`
-- the same maxlogL point, from the same folder, that the warm-start mixture
is fitted from), converts it to the PHYSICAL/linear basis, and prints

    PSD_START_PARAMS=<Soms_d>,<Sa_a>
    GALFOR_START_PARAMS=<amp>,<fk>,<alpha>,<f_1>,<f_2>

for a submit script to ``eval``/export. ``run.py`` converts those physical
values back into whatever basis THIS run samples in, so the two runs' log
sampling flags are allowed to differ -- which is the whole reason the
interchange format is physical.

BASIS DETECTION. A stored chain row is in the SOURCE run's sampling basis,
which may be log. Detection is not a heuristic: every log-sampled column is
strictly positive under its physical prior (psd's two levels, galfor's
``amp``/``fk``/``f_1``/``f_2``), so a value ``<= 0`` in one of those columns
can only be a log. ``--psd-basis`` / ``--galfor-basis`` force it, and the
decision is always printed.

Usage::

    python -m lisatools.globalfit.warmstart.noise_pin --store <prev_run.h5>
    python -m lisatools.globalfit.warmstart.noise_pin --store <h5> --export
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["noise_pin_from_store", "main"]

def _windows(kind):
    """The PHYSICAL support the run's own prior defines, for one branch.

    Read from the module that DEFINES the prior rather than duplicated here
    (a hand-rolled window drifts, and this one has to be exactly the box the
    chain is allowed to live in). ``GALFOR_ALPHA_MAX`` widens alpha the same
    way ``galfor_prior_dict`` does.

    ⚠ WHY THE PRIOR AND NOT A LOOSER "SANITY" WINDOW. A pin outside the
    branch's support puts EVERY walker and EVERY rung at ``log_prior =
    -inf`` on iteration 0. A window that merely catches order-of-magnitude
    basis mistakes would pass such a point and the run would die -- or
    worse, sit there. The prior is the real constraint, so it is the test.
    """
    from ...globalfit.stock.erebor.noise import (
        GALFOR_BASIS, GALFOR_PRIOR_RANGE, PSD_PRIOR_RANGE)

    if kind == "psd":
        return tuple(tuple(map(float, r)) for r in PSD_PRIOR_RANGE)
    rngs = [tuple(map(float, r)) for r in GALFOR_PRIOR_RANGE]
    _am = os.environ.get("GALFOR_ALPHA_MAX", "").strip()
    if _am:
        ia = GALFOR_BASIS.index("alpha")
        rngs[ia] = (rngs[ia][0], float(_am))
    return tuple(rngs)


def _looks_log(vals, log_cols) -> bool:
    """True when any strictly-positive-by-prior column is ``<= 0``."""
    return bool(any(float(vals[c]) <= 0.0 for c in log_cols))


def _to_physical(vals, kind: str, basis: str):
    """``(physical values, resolved basis)`` for one noise branch's row."""
    from ..stock.erebor.noise import GALFOR_BASIS, GALFOR_LOG_PARAMS

    vals = np.asarray(vals, dtype=float)
    log_cols = ([0, 1] if kind == "psd"
                else [GALFOR_BASIS.index(n) for n in GALFOR_LOG_PARAMS])
    if basis == "auto":
        basis = "log" if _looks_log(vals, log_cols) else "linear"
    if basis == "linear":
        return vals.copy(), basis
    out = vals.copy()
    if kind == "psd":
        # prepare_psd_branch uses ln for BOTH columns.
        out = np.exp(vals)
    else:
        # prepare_galfor_branch uses log10, and alpha stays linear.
        out[log_cols] = np.power(10.0, vals[log_cols])
    return out, basis


#: ``fk``/``f_1``/``f_2`` ceilings were tightened on 2026-08-13 by ea7790f3,
#: "galfor priors: bring the fk / f_1 / f_2 ceilings into the analysis band".
#: The OLD ceilings (fk 1e-1, f_1 1e7, f_2 1e4) were, in that commit's words,
#: "the old slope-unit numbers" carried over from the pre-abf52571
#: parameterization and never updated -- so any store written before it can
#: hold a perfectly legitimate sample that today's box excludes.
_PRIOR_TIGHTENED = "2026-08-13 (ea7790f3)"


def _check(vals, window, name) -> None:
    for i, (lo, hi) in enumerate(window):
        if not (lo <= float(vals[i]) <= hi):
            raise SystemExit(
                f"[NOISE-PIN] FATAL: {name}[{i}] = {float(vals[i]):.6g} is "
                f"outside the run's PRIOR support [{lo:.3g}, {hi:.3g}] "
                f"AFTER basis conversion.\n"
                f"[NOISE-PIN] A pin outside the prior puts every walker and "
                f"every rung at log_prior = -inf on iteration 0, so this is "
                f"refused rather than started from.\n"
                f"[NOISE-PIN] TWO likely causes:\n"
                f"[NOISE-PIN]  (a) the source store predates "
                f"{_PRIOR_TIGHTENED}, which pulled the galfor fk / f_1 / f_2 "
                f"ceilings in from the old slope-unit values (1e-1 / 1e7 / "
                f"1e4) to 1e-2. Such a point is a legitimate sample of the "
                f"OLD box that sits on the degeneracy that commit removed "
                f"(both shape factors flat across the band), and it is NOT "
                f"reachable inside today's box -- measured on a v7 store, "
                f"the closest in-box fit rails all four shape parameters "
                f"and is still 1.3x off across the GB band. Use a store "
                f"written after that date.\n"
                f"[NOISE-PIN]  (b) the basis was resolved wrongly -- pass "
                f"--psd-basis / --galfor-basis explicitly."
            )


def noise_pin_from_store(store: str, *, psd_basis: str = "auto",
                         galfor_basis: str = "auto") -> dict:
    """``{psd, galfor, iteration, walker, log_like, psd_basis, galfor_basis}``.

    ``psd`` / ``galfor`` are PHYSICAL (linear) lists ready for
    ``PSD_START_PARAMS`` / ``GALFOR_START_PARAMS``.
    """
    from .opt_snr import best_logl_noise

    raw = best_logl_noise(store)
    psd, pb = _to_physical(raw["psd_params"], "psd", psd_basis)
    gal, gb = _to_physical(raw["galfor_params"], "galfor", galfor_basis)
    _check(psd, _windows("psd"), "PSD_START_PARAMS")
    _check(gal, _windows("galfor"), "GALFOR_START_PARAMS")
    return dict(psd=psd.tolist(), galfor=gal.tolist(),
                iteration=int(raw["iteration"]), walker=int(raw["walker"]),
                log_like=float(raw["log_like"]),
                psd_basis=pb, galfor_basis=gb)


def _fmt(vals) -> str:
    return ",".join(f"{float(v):.10g}" for v in vals)


#: Ratio band, pin / scirdv1 nominal, outside which the psd pin is called
#: out. NOT a guess -- measured against the converged 6mo v8 4-GPU run
#: (`gf_prod_6mo_v8_4gpu`, cold chain, last 50 stored rows to iteration
#: 746):
#:
#:     Soms_d  1.49989e-11   p16/p84 [1.49931e-11, 1.50052e-11]   0.99993x
#:     Sa_a    3.04401e-15   p16/p84 [3.02744e-15, 3.05933e-15]   1.01467x
#:
#: which reproduces that run's own monitor page ("medians sit -0.0% and
#: +1.5% from injection") and so CONFIRMS the injection is scirdv1,
#: 1.5e-11 / 3.0e-15 in sqrt units. The recovered posterior is startlingly
#: tight -- ±0.004% on Soms_d -- so a band of a few tens of percent is
#: already extremely generous for a pin; it is set here at ±33% only
#: because the pin is a single maxlogL sample from a possibly
#: unconverged source run, not a posterior median.
#:
#: ⚠ A sangria-injection run would sit at ~0.53x Soms_d and trip this. That
#: is a warning, never an error, and the message says as much -- the v9
#: launcher is mojito-only.
_PSD_PLAUSIBLE = (0.75, 1.33)


def psd_plausibility(psd) -> list:
    """``[(name, value, nominal, ratio, ok), ...]`` against scirdv1.

    ⚠ WHY THIS EXISTS ON TOP OF ``_check``. ``_check`` tests the PRIOR
    support, which is the right test for "will every walker be at
    ``-inf`` on iteration 0" and the WRONG test for "is this the
    instrument". ``PSD_PRIOR_RANGE`` spans a factor of 33 in ``Soms_d``
    and 200 in ``Sa_a``: a pin ten times the real noise level sails
    through it, and a global fit seeded at a tenfold noise floor does
    not crash -- it quietly finds nothing, which is far more expensive
    than crashing. So report the ratio every time and flag the outliers.

    The branch samples ``(Soms_d, Sa_a)`` in SQRT units while
    ``lisatools.detector`` models carry the squared PSD levels, hence
    the ``sqrt`` here: scirdv1 ``2.25e-22 -> 1.5e-11``, ``9e-30 -> 3e-15``.
    """
    from math import sqrt

    from ...detector import scirdv1

    nominal = (sqrt(float(scirdv1.Soms_d)), sqrt(float(scirdv1.Sa_a)))
    lo, hi = _PSD_PLAUSIBLE
    out = []
    for name, val, nom in zip(("Soms_d", "Sa_a"), psd, nominal):
        ratio = float(val) / nom if nom else float("inf")
        out.append((name, float(val), nom, ratio, lo <= ratio <= hi))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m lisatools.globalfit.warmstart.noise_pin",
        description=__doc__.split("\n\n")[0],
    )
    ap.add_argument("--store", required=True,
                    help="previous run's FULL FINAL h5 (never a snapshot tar "
                         "-- its chain slabs are keep-window extracts)")
    ap.add_argument("--psd-basis", default="auto",
                    choices=("auto", "linear", "log"),
                    help="basis of the SOURCE store's psd chain (default: "
                         "auto -- a non-positive level can only be a log)")
    ap.add_argument("--galfor-basis", default="auto",
                    choices=("auto", "linear", "log"),
                    help="basis of the SOURCE store's galfor chain")
    ap.add_argument("--export", action="store_true",
                    help="prefix each line with 'export ' so the output can "
                         "be sourced directly")
    args = ap.parse_args(argv)

    pin = noise_pin_from_store(args.store, psd_basis=args.psd_basis,
                               galfor_basis=args.galfor_basis)
    pre = "export " if args.export else ""
    print(f"# [NOISE-PIN] {args.store}", file=sys.stderr)
    print(f"# [NOISE-PIN] maxlogL cold walker {pin['walker']} at stored "
          f"iteration {pin['iteration']} (lnL {pin['log_like']:.3f})",
          file=sys.stderr)
    print(f"# [NOISE-PIN] source basis: psd={pin['psd_basis']} "
          f"galfor={pin['galfor_basis']}; emitted values are PHYSICAL/linear "
          f"and run.py converts them into THIS run's sampling basis.",
          file=sys.stderr)
    rows = psd_plausibility(pin["psd"])
    for name, val, nom, ratio, ok in rows:
        print(f"# [NOISE-PIN] {'' if ok else '⚠ '}{name} = {val:.6e} = "
              f"{ratio:.3f}x scirdv1 nominal ({nom:.6e})", file=sys.stderr)
    if not all(ok for *_, ok in rows):
        lo, hi = _PSD_PLAUSIBLE
        print(f"# [NOISE-PIN] ⚠ outside the plausible band [{lo}, {hi}]x. "
              f"This is NOT fatal -- it is inside the prior, so the run will "
              f"start -- but a pin far from the instrument level seeds the "
              f"search at the wrong noise floor and it will quietly find "
              f"less. Check the store before committing an allocation.",
              file=sys.stderr)
    print(f"{pre}PSD_START_PARAMS={_fmt(pin['psd'])}")
    print(f"{pre}GALFOR_START_PARAMS={_fmt(pin['galfor'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
