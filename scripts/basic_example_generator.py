"""Basic example, generator way: no branch, no interaction with the sampler.

You own the signal: inject it into the data, subtract your starting template,
then adjust the residual in place inside the loop. Every walker's container is
a copy of ``fit.general_info.input_data_residual_array`` (the blank fit has no
template of its own to subtract), so the residual is exactly what you put there.
"""
import numpy as np

from lisatools.domains import TDSettings, TDSignal
from lisatools.globalfit.stock import erebor

fit = erebor.blank(
    nwalkers=1, ntemps=1, make_diagnostic_plots=False, file_store_dir="./gf_output_blank_gen/"
)
fit.build()
gi = fit.general_info


def sine_template(A, f0, phi0):
    Nobs = int(gi.Tobs / gi.dt)
    t_arr = np.arange(Nobs) * gi.dt + gi.data_t0
    X = A * np.sin(2 * np.pi * f0 * t_arr + phi0)
    td_sig = TDSignal(np.asarray([X, X.copy(), X.copy()]), TDSettings(Nobs, gi.dt))
    return td_sig.transform(gi.input_data_residual_array.settings)


truth = (1e-23, 2e-3, 3.0980983423)
current = (1.001e-23, 2.000001e-3, 3.0980983423 + 0.001)  # starting point

# before the loop: inject the signal, then subtract the starting template
gi.input_data_residual_array += sine_template(*truth)
gi.input_data_residual_array -= sine_template(*current)

for iteration, (model, state) in enumerate(fit.sample(iterations=3, store=False, progress=False)):
    r = model.analysis_container_arr[0]  # the residual: data - template(current)
    ll_cur = float(r.likelihood())       # -1/2 <r|r>

    proposal = truth
    h_cur, h_new = sine_template(*current), sine_template(*proposal)
    ll_new = float(r.template_likelihood(h_new - h_cur))  # -1/2 <r + h_cur - h_new | ...>
    print(f"it {iteration}: log-like {ll_cur:.4f} -> proposal {ll_new:.4f}")

    if ll_new > ll_cur:
        r.add_signal_to_data(h_cur)          # residual update, in place
        r.subtract_signal_from_data(h_new)
        current = proposal
