#!/usr/bin/env python
"""One-walker gate driver: a stock fit under the CAMPAIGN's sig-het settings.

Why this exists (2026-09-17). The cluster gates in
``docs/one-walker-testing-campaign.md`` ran ``scripts/run_global.py --stock
gb_no_fg`` with the STOCK sig-het defaults -- ``SIGHET_NT_LAYER=64`` (snapped
to 60, the 36 h stride the sig-het v4 notes flagged as 2.2x too coarse), no
reference refresh (``GB_SIGHET_REFRESH_EVERY=0``), ``SIGHET_N_CP`` AUTO --
while the production run pins a finer setup in
``scripts/fstat_proposal/submit_gf_6mo_v8.sh``. A gate that reads accuracy
must run what production runs, and the two must not drift apart by hand:
this driver reads the sig-het ``export`` lines from the campaign script at
launch (single source of truth), applies them to the environment unless the
shell already set a value (shell wins, so a knob A/B is still one ``VAR=...``
prefix away), prints what it applied, and then builds the stock fit.

The stock synthetic ``gb_no_fg`` data has no GB sources, so by default one
loud in-band binary is injected (the laptop parity gate's, also the WP7
runbook's ``gate_gb.py``); without it the F-stat epoch finds no peaks and the
replicas trivially agree. ``--no-injection`` turns that off.

Launch it exactly like ``run_global.py``::

    FILE_STORE_DIR=$G/T1/ NWALKERS=1 NUM_ITERATIONS=4 GPUS=0 \\
      mpiexec -n 3 -ppn 1 python scripts/diagnostics/gate_run.py --stock gb_no_fg

``--print-only`` shows the pins and exits (no MPI, no build).
"""

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

#: The production script whose sig-het exports the gates mirror.
CAMPAIGN_SCRIPT = os.path.join(REPO, "scripts", "fstat_proposal", "submit_gf_6mo_v8.sh")

#: Sig-het knob families to mirror ...
_KEEP = re.compile(r"^(SIGHET_|GB_SIGHET_)")
#: ... minus the in-run diagnostics (engine sweeps / dissections / tier scans),
#: which are empty in production anyway and would only add cost to a gate.
_DROP = re.compile(r"DISSECT|SWEEP|TIER_SCAN")
#: Non-prefixed knobs the sig-het setup depends on (the in-model setup batch
#: mode and the orthogonality-check safety net the campaign keeps on).
_EXTRA = ("GB_ORTHO_LL_CHECK", "GB_INMODEL_SETUP_BATCH")

_EXPORT = re.compile(r"""^export\s+([A-Z][A-Z0-9_]*)=("[^"]*"|'[^']*'|[^\s#]*)""")

#: amplitude, f0 [Hz], fdot, fddot, phi0, iota, psi, lambda, beta (GBGPU order)
LOUD_INJECTION = [[1e-21, 7.5e-3, 1e-16, 0.0, 1.2, 0.9, 1.0, 4.0, -0.6]]


def campaign_sighet_pins(path=CAMPAIGN_SCRIPT):
    """``{NAME: value}`` for every sig-het ``export`` line of the campaign script.

    Only top-level ``export NAME=value`` lines count (the script's in-job
    block); a later export of the same name wins, like it does in the shell.
    Quotes are stripped; a trailing ``# comment`` is not part of the value.
    """
    pins = {}
    with open(path) as fh:
        for line in fh:
            m = _EXPORT.match(line)
            if m is None:
                continue
            name, value = m.group(1), m.group(2)
            if not ((_KEEP.match(name) and not _DROP.search(name)) or name in _EXTRA):
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            pins[name] = value
    return pins


def apply_pins(pins, environ=None):
    """Set every pin the shell did not already set; return (applied, kept)."""
    environ = os.environ if environ is None else environ
    applied, kept = {}, {}
    for name, value in pins.items():
        if name in environ:
            kept[name] = environ[name]
        else:
            environ[name] = value
            applied[name] = value
    return applied, kept


def _fmt(d):
    return " ".join(f"{k}={v}" for k, v in sorted(d.items())) or "(none)"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stock", default="gb_no_fg", help="stock global-fit name (default gb_no_fg)")
    parser.add_argument("--no-injection", action="store_true",
                        help="do not inject the loud in-band GB (gb_no_fg synthetic then has no GB sources)")
    parser.add_argument("--campaign", default=CAMPAIGN_SCRIPT,
                        help="submit script whose sig-het exports are mirrored")
    parser.add_argument("--print-only", action="store_true", help="print the pins and exit")
    args = parser.parse_args(argv)

    pins = campaign_sighet_pins(args.campaign)
    applied, kept = apply_pins(pins)
    print(f"[GATE] sig-het pins from {os.path.relpath(args.campaign, REPO)}: "
          f"applied {_fmt(applied)}; shell overrides kept {_fmt(kept)}", flush=True)
    if args.print_only:
        return 0

    # import AFTER the environment is set: the stock fits read their env knobs
    # at construction (env_default), and nothing below may see the defaults
    from lisatools.globalfit.stock import erebor  # noqa: E402

    fit = erebor.get_stock(args.stock)
    if not args.no_injection and getattr(fit, "gb", None) is not None:
        fit.general.gb_injection_params = LOUD_INJECTION
        print(f"[GATE] loud GB injection: {LOUD_INJECTION[0]}", flush=True)
    fit.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
