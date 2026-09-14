"""GB warm start: the whole pipeline as installed package code.

User ruling 2026-09-14: "All the warmstart code should be part of the
globalfit/lisatools package" -- the former ``scripts/gb/warmstart_*.py``
stage scripts and ``lisatools.sampling.warmstart_proposal`` now live here
(the old locations remain as thin shims). One home, five modules:

* :mod:`.fit_from_store` -- stage 1: cluster a finished run's cold-chain
  leaf table into Gaussian components (CLI ``main(argv=None)``).
* :mod:`.match_referee` -- stage 2: judge the components with real
  waveform matches (CLI ``main(argv=None)``).
* :mod:`.referee_apply` -- stage 2.5: apply the referee verdicts and
  write the REFEREED npz production arms (CLI ``main(argv=None)``).
* :mod:`.proposal` -- :class:`WarmStartComponents`, the eryn RJ birth
  distribution that loads the refereed npz at recipe build.
* :mod:`.build` -- :func:`ensure_warm_start_components`, the automatic
  check-then-build orchestrator (fit -> referee -> apply, in-process,
  MPI-safe lock) recipe build calls when the npz is missing.

Each stage module is runnable directly too:
``python -m lisatools.globalfit.warmstart.fit_from_store --help``.
"""

from .build import ensure_warm_start_components
from .proposal import WarmStartComponents

__all__ = ["WarmStartComponents", "ensure_warm_start_components"]
