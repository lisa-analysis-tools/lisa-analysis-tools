"""Shared conventions of the MBH speed / accuracy / data harness (scripts/mbh/mbh_speed_durations.sh).

The EMRI harness's source-agnostic helpers are IMPORTED from
``scripts/emri/emri_batch_speed.py`` (origin/dev >= ddaad46f), never copied:

* ``RunBox`` -- the fit's run box (BAND x [EDGE, Nt - EDGE) layers) and its noise,
  ``XYZ2SensitivityMatrix(dom, model="scirdv1", stochastic_params=(Tobs,))``: SciRD v1 XYZ
  (TDI-2) plus, with ``foreground=True``, the fitted hyperbolic-tangent galactic confusion
  foreground (``stochastic_params=(Tobs,)`` alone selects
  :class:`~lisatools.stochastic.FittedHyperbolicTangentGalacticForeground`) at the WINDOW's
  ``Tobs = Nf * Nt * dt``, the stock erebor convention. Every noise-weighted MBH quantity uses
  ``RunBox(...).sens`` (:func:`harness_box`) and records ``RunBox.describe()``. MBH builds the
  box on its own run grid's parameters (same Nf, Nt, dt, band, edge: identical active box,
  checked) and scores through the containers (``AnalysisContainer._slice_to_template``),
  because its batched templates live on a SUB-box of the grid and its run grid carries the
  data start as ``t0`` (RunBox's domain has ``t0 = 0``; the PSD does not depend on it).
* ``check_window`` -- refuses a window whose end plus the production wrapper's 4e4 s runs
  past the orbits' light-travel-time table (packaged equal-arm: REF + 697.9 d; the 731-d
  mojito bricks: REF + 730.5 d). Every MBH step calls it (:func:`check_window`).

Light on purpose (no jax / phentax import): the scripts and tests/test_mbh_harness_noise.py
import it.
"""
from __future__ import annotations

import os
import sys

DAY = 86400.0
#: ``--foreground`` choices (the driver's FOREGROUND), as the EMRI / SOBBH drivers
FOREGROUNDS = ("on", "off")
#: the model ``stochastic_params=(Tobs,)`` alone selects in ``Sensitivity.get_stochastic_contribution``
FOREGROUND_MODEL = "FittedHyperbolicTangentGalacticForeground"
#: the EMRI helpers this harness imports (scripts/emri/emri_batch_speed.py, origin/dev >= ddaad46f)
EMRI_HELPERS = ("RunBox", "check_window", "load_l1_data")

#: ``differs_from_emri`` of every MBH JSON line: the deliberate MBH-specific choices
DIFFERS_FROM_EMRI = (
    "window: the batched template lives on a merger-centred per-leaf window (90 d before / 10 d after "
    "the merger, 1 d margins, clamped to the active box) transformed on a segment 4 d wider each side "
    "and cropped back -- a SUB-box of the WDM grid, scored through AnalysisContainer._slice_to_template",
    "references: two stock templates -- production (T 30.44 d, unsnapped epoch, order 8) and the "
    "90-d lattice-snapped stock (the move's cross-check); the acceptance (|dlogL| <= 0.5, mm < 1e-6) "
    "gates on the 90-d one",
    "extras: band residual SNRs, kept-box geometry, the reference's power outside the kept box, the "
    "pairwise prod90 comparisons",
    "grid: ONE merger-centred grid (WINDOW_DAYS 120 d, the merger MERGER_AT_DAYS 100 d in; shifted into "
    "the brick near its ends) instead of EMRI's 180 / 360 / 720-d sweep from START_OFFSET_S: an MBH "
    "template is always the 90 d before / 10 d after its merger, whatever the data length",
    "templates: speed / accuracy score injected noiseless stock templates; the mojito stream is scored "
    "in the separate data step (mbh_cd1l_campaign.py --window centered)",
)


def emri_helpers():
    """``scripts/emri/emri_batch_speed.py`` as a module (its directory APPENDED to sys.path, so a
    PYTHONPATH entry can stand in a newer copy); loud when it predates the shared helpers."""
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "emri")
    if d not in sys.path:
        sys.path.append(d)
    import emri_batch_speed as E

    missing = [n for n in EMRI_HELPERS if not hasattr(E, n)]
    if missing:
        raise ImportError(
            f"{E.__file__} has no {missing}: this checkout predates origin/dev ddaad46f (the EMRI "
            "harness helpers the MBH harness imports); merge origin/dev")
    return E


def _backend_name(settings) -> str:
    b = getattr(settings, "backend", None)
    name = getattr(b, "name", b)
    return str(name).split("_")[-1] if name is not None else "cpu"


def harness_box(settings, edge: int, foreground: str = "on"):
    """``RunBox`` on the run grid ``settings`` (its Nf, Nt, dt, band; ``edge`` = the crop the
    grid was built with). Refuses a box whose active layers differ from the grid's."""
    if foreground not in FOREGROUNDS:
        raise ValueError(f"foreground={foreground!r}: choose one of {FOREGROUNDS}")
    E = emri_helpers()
    box = E.RunBox(int(settings.Nf), int(settings.Nt), float(settings.data_dt), _backend_name(settings),
                   edge=int(edge), min_freq=float(settings.min_freq), max_freq=float(settings.max_freq),
                   foreground=foreground == "on")
    got = tuple(int(getattr(box.dom, k)) for k in ("ind_min_f", "ind_max_f", "ind_min_t", "ind_max_t"))
    want = tuple(int(getattr(settings, k)) for k in ("ind_min_f", "ind_max_f", "ind_min_t", "ind_max_t"))
    if got != want:
        raise ValueError(f"RunBox(edge={edge}) active layers (f_lo, f_hi, t_lo, t_hi) {got} != the run grid's "
                         f"{want}: pass the edge crop the grid was built with")
    return box


def harness_sensitivity(settings, edge: int, foreground: str = "on"):
    """The harness noise on ``settings``: ``harness_box(...).sens`` (one fresh matrix per call)."""
    return harness_box(settings, edge, foreground).sens


def noise_record(box) -> dict:
    """The EMRI-named JSON fields of a RunBox: ``tobs_s``, ``foreground`` (bool), ``noise``."""
    return dict(tobs_s=float(box.tobs), foreground=bool(box.foreground), noise=box.describe())


def check_window(orb, data_t0, span, kind):
    """``emri_batch_speed.check_window`` as a loud refusal (SystemExit with its numbers)."""
    try:
        emri_helpers().check_window(orb, data_t0, span, kind)
    except ValueError as exc:
        raise SystemExit(f"REFUSED: {exc}") from None
