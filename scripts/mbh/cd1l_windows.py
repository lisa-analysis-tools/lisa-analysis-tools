"""Observation taper defined in physical time, independent of allocated grid."""
import numpy as np


def observation_window(n_keep, n_total, dt, entry_seconds, exit_seconds=1000.):
    if not 0 <= n_keep <= n_total or dt <= 0 or entry_seconds < 0 or exit_seconds <= 0:
        raise ValueError("Invalid observation window")
    t = np.arange(n_total) * dt
    entry = (0.5 * (1 - np.cos(np.pi * np.minimum(t / entry_seconds, 1.)))
             if entry_seconds else np.ones(n_total))
    # Preserve the existing discrete trailing ramp at the native cadence.
    remaining = np.maximum((n_keep - 1) * dt - t, 0.)
    tail = 0.5 * (1 - np.cos(np.pi * np.minimum(remaining / exit_seconds, 1.)))
    return entry * tail * (np.arange(n_total) < n_keep)
