"""Basic example, sampler way: register the sine as a branch, then fit.run().

The engine owns the residual: at run start it subtracts the branch's template
(built by ``signal_gen`` from each walker's coords) from every walker's copy
of the data, and the walkers start at ``injection`` (scattered by
``SINE_START_FACTOR``, default 1e-5).
"""
import shutil

import numpy as np
from eryn.prior import uniform_dist

from lisatools.domains import TDSettings, TDSignal
from lisatools.globalfit.hdfbackend import GFHDFBackend
from lisatools.globalfit.stock import erebor

shutil.rmtree("./gf_output_blank_run/", ignore_errors=True)
fit = erebor.blank(
    nwalkers=4, ntemps=2, num_iterations=3, make_diagnostic_plots=False,
    file_store_dir="./gf_output_blank_run/",
)

truth = np.array([1e-23, 2e-3, 3.0980983423])  # A, f0, phi0


def sine_template(A, f0, phi0):
    """signal_gen: sampled params -> template in the data's domain (called after build)."""
    gi = fit.general_info
    Nobs = int(gi.Tobs / gi.dt)
    t_arr = np.arange(Nobs) * gi.dt + gi.data_t0
    X = A * np.sin(2 * np.pi * f0 * t_arr + phi0)
    td_sig = TDSignal(np.asarray([X, X.copy(), X.copy()]), TDSettings(Nobs, gi.dt))
    return td_sig.transform(gi.input_data_residual_array.settings)


def sine_move(model, state):
    """Your proposal: read model.analysis_container_arr (the residuals), adjust, write back."""
    return state, None


fit.add_branch(
    "sine",
    ndim=3,
    priors={0: uniform_dist(1e-24, 1e-22), 1: uniform_dist(1e-3, 4e-3), 2: uniform_dist(0.0, 2 * np.pi)},
    injection=truth,
    signal_gen=sine_template,
    moves=[sine_move],
)
fit.build()

# inject the signal into the data; the run subtracts the branch template from every residual
fit.general_info.input_data_residual_array += sine_template(*truth)
fit.run()

reader = GFHDFBackend(fit.general_info.main_file_path)
print("stored iterations:", reader.iteration, "| last log-like:", reader.get_log_like()[-1])
