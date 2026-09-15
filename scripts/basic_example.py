import numpy as np
from lisatools.globalfit.stock import erebor
from lisatools.domains import *
fit = erebor.blank(nwalkers=1, ntemps=1, make_diagnostic_plots=False)
# adjust the branches like above -- or keep it with nothing
fit.build()

# sine wave 
A = 1e-23
f0 = 2e-3
phi0 = 3.0980983423
Nobs = int(fit.general_info.Tobs / fit.general_info.dt)
t_arr = np.arange(Nobs) * fit.general_info.dt + fit.general_info.data_t0
#breakpoint()

X = A * np.sin(2 * np.pi * f0 * t_arr + phi0)
Y, Z = X.copy(), X.copy()
td_set = TDSettings(Nobs, fit.general_info.dt)
td_sig = TDSignal(np.asarray([X, Y, Z]), td_set)
for iteration, (model, state) in enumerate(fit.sample(iterations=3, store=False, progress=False)):
    aca = model.analysis_container_arr    # read the residuals; adjust them in place
    # customized operation
    if iteration == 0:
        # inject into data
        inj_sig = td_sig.transform(aca[0].data_res_arr.settings)
        aca[0].data_res_arr += inj_sig
        ll_base = aca[0].template_likelihood(inj_sig)
        snr_base = aca[0].template_snr(inj_sig)
        print("ll_base:", ll_base, "snr_base:", snr_base)

    # new parameters
    # recommend looking at the analysis containers. You can move this under the hood.
    A1 = 1.001e-23
    f01 = 2.000001e-3
    phi01 = phi0 + 0.001
 
    X1 = A1 * np.sin(2 * np.pi * f01 * t_arr + phi01)
    Y1, Z1 = X1.copy(), X1.copy()

    wdm_sig = TDSignal(np.asarray([X1, Y1, Z1]), td_set).transform(aca[0].data_res_arr.settings)
    ll_new = aca[0].template_likelihood(wdm_sig)
    snr_new = aca[0].template_snr(wdm_sig)
    print("ll new:", ll_new, "snr_new:", snr_new)

