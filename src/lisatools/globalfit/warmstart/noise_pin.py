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
import sys

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["noise_pin_from_store", "main"]

#: Physical sanity windows, mirroring ``run.py``'s ``_NOISE_PIN_WINDOWS``.
#: Deliberately wide: they catch a BASIS mistake (orders out), not physics.
_PSD_WINDOW = ((1e-13, 1e-10), (1e-16, 1e-13))
_GALFOR_WINDOW = ((1e-50, 1e-30), (1e-6, 1e-1), (0.0, 30.0),
                  (1e-6, 1e-1), (1e-6, 1e-1))


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


def _check(vals, window, name) -> None:
    for i, (lo, hi) in enumerate(window):
        if not (lo <= float(vals[i]) <= hi):
            raise SystemExit(
                f"[NOISE-PIN] FATAL: {name}[{i}] = {float(vals[i]):.6g} is "
                f"outside the physical window [{lo:.3g}, {hi:.3g}] AFTER "
                f"basis conversion. Either the source store is not what it "
                f"is claimed to be, or the basis was resolved wrongly -- "
                f"pass --psd-basis / --galfor-basis explicitly. Refusing to "
                f"emit a pin that would start the run somewhere absurd."
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
    _check(psd, _PSD_WINDOW, "PSD_START_PARAMS")
    _check(gal, _GALFOR_WINDOW, "GALFOR_START_PARAMS")
    return dict(psd=psd.tolist(), galfor=gal.tolist(),
                iteration=int(raw["iteration"]), walker=int(raw["walker"]),
                log_like=float(raw["log_like"]),
                psd_basis=pb, galfor_basis=gb)


def _fmt(vals) -> str:
    return ",".join(f"{float(v):.10g}" for v in vals)


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
    print(f"{pre}PSD_START_PARAMS={_fmt(pin['psd'])}")
    print(f"{pre}GALFOR_START_PARAMS={_fmt(pin['galfor'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
