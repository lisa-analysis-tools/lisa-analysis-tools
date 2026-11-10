"""

Waveform/likelihood/inner-product generator using the gwtf package (https://github.com/cchapmanbird/gwtf).

The waveform used here is TaylorT3, same as what is used in waveform.py.
The fresnel approximation is applied in a box-car window scenario to model the time-frequency waveform.
Using a leading order time-frequency local response function.

"""
import cupy as cp
import numpy as np
from pygwtf.models import TaylorT3Spin
from pygwtf.generator import AnalyticTimeFrequencyWaveform

from ...utils.constants import C_SI
from ...utils.utility import asnumpy
from ..utils import SOBHB_GWTF_PHASE_OFFSET
from ...detector import Orbits
from ...domains import STFTSettings, STFTSignal

# XYZ -> AET, orthonormal (so AET -> XYZ is the transpose). Rows are A, E, T.
M_AET = np.array([[-1, 0, 1] / np.sqrt(2),
                  [1, -2, 1] / np.sqrt(6),
                  [1, 1, 1] / np.sqrt(3)])

# Same bands as SC_Search's DEFAULT_PSD_CLIP_BANDS (TDI transfer-function nulls).
DEFAULT_PSD_DROP_BANDS = [(0.029, 0.031), (0.059, 0.061), (0.089, 0.091)]



class GWTF_generator:

    def __init__(self,
                 settings: STFTSettings,
                 orbits: Orbits,
                 T_obs,
                 fresnel_kernel_width = 10,
                 use_GPU = True,
                 psd_drop_bands = DEFAULT_PSD_DROP_BANDS,
                 tukey_alpha = None):
        """
        Assumptions:
        - PSD and data are already pre-processed and in the correct shape corresponding to the tf grid provided.
          Use to_gwtf_layout to map lisatools STFT arrays onto the gwtf grid.
        - PSD is clipped around the zeros as expected.
        - tukey_alpha: None -> box-car Fresnel kernel. Otherwise the Tukey-windowed Fresnel kernel with this taper
          fraction, which should match the per-segment STFT window of the data (window_taper_duration / dT).

        Frequency grid: gwtf bin k is centred on (k+1)*dF, i.e. the grid always starts at dF (no DC bin).
        The gwtf grid therefore covers lisatools STFT bins 1..settings.ind_max, and bins below settings.ind_min
        (outside the active band) are zero-filled in the data and given infinite PSD by to_gwtf_layout.
        """

        # Constructor arguments, used by MultiGPUResidualAddRemoveMove to build one replica per GPU
        # (it rebuilds the generator as GWTF_generator(**self.kwargs) with "orbits" swapped for that device's orbits).

        self.psd_drop_bands = psd_drop_bands
        self.tukey_alpha = tukey_alpha
        self.use_tukey = tukey_alpha is not None

        self.kwargs = dict(settings=settings,
                           orbits=orbits,
                           T_obs=T_obs,
                           fresnel_kernel_width=fresnel_kernel_width,
                           use_GPU=use_GPU,
                           psd_drop_bands=self.psd_drop_bands,
                           tukey_alpha=self.tukey_alpha)


        self.settings = settings

        self.dF = settings.df # Frequency bin width
        self.dT = settings.dt # Time bin width

        self.nT = settings.NT # Number of time segments
        # nF here is not the same as settings.NF_active, as the gwtf grid always starts at dF (no DC bin) and goes up to settings.ind_max.
        # LISAtools uses a more general (and better) setup where the STFT grid starts at settings.ind_min and goes up to settings.ind_max, and the active band is settings.ind_min..settings.ind_max.
        self.nF = settings.NF_active # Number of gwtf frequency bins, STFT bins ind_min..ind_max (must match the data/PSD width: gwtf does not check this at call time)

        # gwtf t=0 is the start of the first segment, i.e. settings.t0 (= t_init + 850.5 + pad_trim)
        self.t_grid = settings.t0 + np.arange(self.nT + 1) * self.dT # (nT+1,) segment edges for GWTF
        self.f_grid = settings.f_arr # (nF,) central frequency grid for GWTF

        self.T_obs = T_obs # Observation time in seconds.

        # midpoints of the time segments, used for the spacecraft positions and LTTs.
        t_midpoints = orbits.xp.asarray(self.t_grid[:-1] + self.dT/2) # (nT,) on the orbits backend

        # Compute orbital quantities for response from the orbits object.
        # NOTE: Spacecraft positions are computed in equitorial frame, so the response function needs (ra,dec) instead of (lon,lat) for the sky location.
        spacecraft_positions = orbits.xp.stack([orbits.get_pos(t_midpoints, sc) for sc in (1, 2, 3)], axis=1) # (nT, 3 (spacecrafts), 3 (x,y,z)) for GWTF

        # 12, 23, 31 only needed for gwtf for now
        spacecraft_LTTs = orbits.xp.stack([orbits.get_light_travel_times(t_midpoints, link) for link in (12, 23, 31)], axis=1) * C_SI # (nT, 3), seconds -> metres, as gwtf expects

        # Setup config that waveform generator needs
        config = {'nT':self.nT,
                  'nF':self.nF,
                  'dT':self.dT,
                  'dF':self.dF,
                  'fmin': float(settings.f_arr[0]),        # centre of active bin 0 = ind_min*dF
                  'kernel_width':fresnel_kernel_width}
        if self.use_tukey:
            config['alpha'] = float(tukey_alpha)

        if use_GPU:
            backend = 'gpu'
            self.xp = cp
        else:
            backend = 'cpu'
            self.xp = np
        self.config = config
        self.backend = backend

        self.spacecraft_positions = self.xp.asarray(spacecraft_positions)
        self.spacecraft_LTTs = self.xp.asarray(spacecraft_LTTs)

        # Waveform generator object which computes raw <h|h> and <d|d> statistics per waveform generator.
        self.inner_product_statistics_object = AnalyticTimeFrequencyWaveform(model_class=TaylorT3Spin,
                                                                config=config,
                                                                tdi_type=2,
                                                                backend=backend,
                                                                spacecraft_orbits=self.spacecraft_positions,
                                                                spacecraft_ltts=self.spacecraft_LTTs,
                                                                block_vectorised_gpu = use_GPU, # Block vectorised mode (GPU only), better for small batches -> global fit
                                                                gf_mode = True, # Use the GF mode, allowing for one PSD and DATA array per walker.
                                                                use_tukey = self.use_tukey,
                                                                )

        # Waveform-only generator (non-gf_mode), built lazily in compute_time_frequency_waveform.
        # Only used for the residuals at the last step.
        self.waveform_gen_object = None

    def to_gwtf_layout(self, arr):
        """
        Map a lisatools STFT array onto the gwtf grid. 
        Used for both the data and the PSD.
        
        - Reshape from ([#nWalkers,] #nChannels, #nT, #NF_active) to ([#nWalkers,] #nT, #nF, #nChannels)

        Parameters:
            arr: ([#nWalkers,] #nChannels, #nT, #NF_active) array on the active band settings.ind_min..settings.ind_max.
                 The walker axis is optional.
        Returns:
            ([#nWalkers,] #nT, #nF, #nChannels) array on the gwtf grid (STFT bins 1..settings.ind_max).
        """
        out = self.xp.asarray(arr)          # ([nW,] 3, nT, NF_active)

        # (#nChannels, #nT, #nF) -> (#nT, #nF, #nChannels)
        if out.ndim == 3:
            return out.transpose(1, 2, 0)
        # (#nWalkers, #nChannels, #nT, #nF) -> (#nWalkers, #nT, #nF, #nChannels)
        return out.transpose(0, 2, 3, 1)

    def to_gwtf_layout_from_xyz(self, data_xyz):
        """
        Map lisatools XYZ STFT data (e.g. the global-fit residuals) onto the gwtf grid in AET.

        Parameters:
            data_xyz: ([#nWalkers,] 3 XYZ, #nT, #NF_active) complex array.
        Returns:
            ([#nWalkers,] #nT, #nF, 3 AET) contiguous complex array on self.xp.
        """
        data_xyz = self.xp.asarray(data_xyz)
        M = self.xp.asarray(M_AET)
        data_aet = self.xp.einsum("ci,...itf->...ctf", M, data_xyz) # ([nW,] 3 AET, nT, NF_active)
        return self.xp.ascontiguousarray(self.to_gwtf_layout(data_aet))

    def psd_aet_from_invC(self, invC):
        """
        Diagonal AET PSD on the gwtf grid from the lisatools XYZ inverse covariance.

        Inverts invC to the XYZ covariance per (segment, bin), rotates it to AET and keeps the
        (real) diagonal. Exact only if the noise is diagonal in AET (equal arms); the off-diagonal
        AET terms are dropped. Bins with f <= 0 are not invertible and never used, so they get an
        infinite PSD. 

        Parameters:
            invC: ([#nWalkers,] 3, 3, #nT, #NF_active) XYZ inverse covariance.
        Returns:
            ([#nWalkers,] #nT, #nF, 3 AET) contiguous real array on self.xp.
        """
        invC = self.xp.asarray(invC)
        batched = invC.ndim == 5
        if not batched:
            invC = invC[None]

        M = self.xp.asarray(M_AET)
        # Isolate the positive frequency bins, as the negative frequencies are not invertible and never used.
        fpos = self.xp.asarray(self.settings.f_arr) > 0

        S_aet = self.xp.full((invC.shape[0], 3) + invC.shape[-2:], self.xp.inf)  # (nW, 3 AET, nT, NF_active)

        # One walker at a time (PSD is unique per walker): the batched 3x3 inverse needs a full (nT, NF_active, 3, 3) temporary.
        for w in range(invC.shape[0]):
            # Only taking the diagonal of the AET covariance. 
            C_xyz = self.xp.linalg.inv(self.xp.moveaxis(invC[w][:, :, :, fpos], (0, 1), (-2, -1)))  # (nT, nF_pos, 3, 3)

            S_aet[w][..., fpos] = self.xp.einsum("ci,klij,cj->ckl", M, C_xyz, M).real
            del C_xyz

        f = self.xp.asarray(self.settings.f_arr)
        drop = self.xp.zeros(f.shape, dtype=bool)
        for f_lo, f_hi in (self.psd_drop_bands or []):
            drop |= (f >= f_lo) & (f <= f_hi)
        S_aet[..., drop] = self.xp.inf      # zero information: the bins drop out of <d|h> and <h|h>

        out = self.xp.ascontiguousarray(self.to_gwtf_layout(S_aet))  # (nW, nT, nF, 3 AET)
        return out if batched else out[0]

    def _to_gwtf_parameters(self, params):
        """
        Convert the lisatools parameters to the gwtf (TaylorT3Spin) parameters.

        params: (#nSources, 11) array of parameters (the SOBBHTaylorT3TDIWaveform basis), where the columns are:
        0: m1 (Msun), 1: m2 (Msun), 2: s1, 3: s2 (aligned spins), 4: dist (Gpc), 5: inc (rad),
        6: f0 (Hz, GW frequency at the data start settings.t0, i.e. gwtf t=0),
        7: phi_coal (rad, coalescence orbital phase, Phi(x) = phi_coal - P(x), in the mojito /
           SOBBHTaylorT3TDIWaveform(phase_at_coalescence=True) strain convention), 8: psi (rad),
        9: ra, 10: dec (rad, in the frame of the orbits)

        Returns Mc, eta, cosinc, dist_pc, f0, s1, s2, phi_coal, psi, ra, dec (distance in pc), with
        phi_coal shifted by SOBHB_GWTF_PHASE_OFFSET: the pygwtf strain has the opposite sign
        (tasks/sobbh_gwtf_td_discrepancies.md), so gwtf(phi_coal + pi/2) == TD(phi_coal).
        """
        m1, m2, s1, s2, dist, inc, f0, phi_coal, psi, ra, dec = params.T

        Mc = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        eta = (m1 * m2) / (m1 + m2)**2
        cosinc = self.xp.cos(inc)
        dist_pc = dist * 1.0e9 # Convert distance from Gpc to pc
        phi_coal = phi_coal + SOBHB_GWTF_PHASE_OFFSET  # pygwtf strain sign -> mojito / TD convention

        return Mc, eta, cosinc, dist_pc, f0, s1, s2, phi_coal, psi, ra, dec

    def _split_parameters(self, params):
        """
        Split the lisatools parameters into the gwtf waveform and response parameter arrays.

        params: (#nSources, 11) array of parameters, see _to_gwtf_parameters.

        Returns the gwtf waveform parameters (M, eta, cosinc, D (pc), f0, s1, s2, phi_coal) and
        response parameters (cosinc, psi, ra, dec).
        """
        params = self.xp.asarray(params)

        Mc, eta, cosinc, dist_pc, f0, s1, s2, phi_coal, psi, ra, dec = self._to_gwtf_parameters(params)

        M = Mc * (eta)**(-3/5)

        wf_params = self.xp.column_stack((M, eta, cosinc, dist_pc, f0, s1, s2, phi_coal))
        resp_params = self.xp.column_stack((cosinc, psi, ra, dec))

        return wf_params, resp_params

    def compute_time_frequency_waveform(self, params, channel_basis="AET"):
        """
        Compute the time-frequency waveform for each source.

        params: (#nSources, 11) array of parameters, see _to_gwtf_parameters.

        channel_basis: "AET" (gwtf's native output) or "XYZ" (the basis of the lisatools data/residuals).

        Returns: stft_waveform_filled (list)
        1. Each element of the list corresponds to a source.
        2. Each element is a STFTSignal object, which contains the time-frequency waveform
        """
        wf_params, resp_params = self._split_parameters(params)

        # gf_mode only supports the inner-product kernels, so waveforms come from a separate
        # non-gf_mode generator on the same grid/orbits. Built on first use, no data/PSD needed.
        if self.waveform_gen_object is None:
            self.waveform_gen_object = AnalyticTimeFrequencyWaveform(model_class=TaylorT3Spin,
                                                                    config=self.config,
                                                                    tdi_type=2,
                                                                    backend=self.backend,
                                                                    spacecraft_orbits=self.spacecraft_positions,
                                                                    spacecraft_ltts=self.spacecraft_LTTs,
                                                                    use_tukey=self.use_tukey,
                                                                    )

        waveform_array = self.waveform_gen_object(parameters=wf_params,
                                                  parameters_response=resp_params,
                                                  out=None,
                                                  compute_statistic=False)

        # gwtf layout -> lisatools layout: (nSources, nTimes, nF, nChannels) -> (nSources, nChannels, nTimes, nF)
        arr = waveform_array.transpose(0, 3, 1, 2)
        if channel_basis == "XYZ":
            # M_AET is orthonormal, so AET -> XYZ is its transpose (inverse of a orthonormal matrix is its transpose).
            arr = self.xp.einsum("ci,sctf->sitf", self.xp.asarray(M_AET), arr)
        elif channel_basis != "AET":
            raise ValueError(f"channel_basis must be 'AET' or 'XYZ', got {channel_basis!r}.")
        # move to whichever backend the settings use (np.asarray refuses cupy input, so go through asnumpy)
        arr = self.settings.xp.asarray(arr) if self.settings.backend.uses_cupy else asnumpy(arr)

        # Create a list of STFTSignal objects, one for each source.
        stft_waveform_filled = [STFTSignal(arr[source_num], self.settings) for source_num in range(arr.shape[0])]
        
        return stft_waveform_filled

    def get_signals_for_residuals(self, *params):
        """
        Waveform for residual add/remove in the global fit (the move's waveform_gen_method).
        params: the 11 parameters in _to_gwtf_parameters order, as separate arguments. Either scalars
                (one source) or equal-length 1-D arrays (a batch,
                the move's get_waveforms_here path).

        Returns a single STFTSignal for scalar input, else a list of STFTSignal, one per source.
        Signals are in XYZ (the basis of the lisatools residuals) on the settings backend.
        """
        if len(params) != 11:
            raise ValueError(f"Expected 11 parameters, got {len(params)}.")

        # Check if its batched or single 
        single = all(np.ndim(p) == 0 for p in params)

        # Rewire into the (#nSources, 11) array of parameters, where the columns are:
        # 0: m1, 1: m2, 2: s1, 3: s2, 4: dist (Gpc), 5: inc, 6: f0, 7: phi_coal, 8: psi, 9: ra, 10: dec
        params_arr = self.xp.stack([self.xp.atleast_1d(self.xp.asarray(p, dtype=self.xp.float64)) for p in params], axis=1) # (nSources, 11)

        # Compute the time-frequency waveform in XYZ (the basis of the lisatools residuals) on the settings backend.
        signals = self.compute_time_frequency_waveform(params_arr, channel_basis="XYZ")

        return signals[0] if single else signals

    def compute_inner_products_per_segment(self, params, data_indices, data, psd):
        """
        Per time-segment inner products.

        params: (#nSources, 11) array of parameters, see _to_gwtf_parameters.

        data_indices: (#nSources,) array of indices corresponding to walker for the data and psd arrays for each source.

        data, psd: (#nWalkers, #nT, #nF, #nChannels) on the gwtf grid.

        Returns d_h, h_h each of shape (#nSources, #nT).
        """
        wf_params, resp_params = self._split_parameters(params)

        # NOTE:  nWalkers <= nSources, as we have nTemps. Where nWalkers x nTemps = nSources.

        # Contains the inner product statistics for each source, shape (nSources, nT, 2) -> (d_h, h_h) per segment.
        statistic_array = self.inner_product_statistics_object(parameters=wf_params,#(#nSources)
                                        channels=self.xp.asarray(data), #(#nWalkers, #nT, #nF, #nChannels)
                                        psds=self.xp.asarray(psd),#(#nWalkers, #nT, #nF, #nChannels)
                                        parameters_response=resp_params,
                                        out = None,
                                        compute_statistic=True,
                                        data_indices = data_indices) # (nSources)

        return statistic_array[:,:,0], statistic_array[:,:,1]


    def compute_inner_products(self, params, data_indices, data, psd):
        """
        Compute inner products <d|h> and <h|h> for each source, summed over time segments.

        Args:
            params: (#nSources, 11) array of parameters, see _to_gwtf_parameters.
            data_indices: (#nSources,) array of indices corresponding to walker for the data and psd arrays for each source.
            data, psd: (#nWalkers, #nT, #nF, #nChannels) on the gwtf grid.

        Returns d_h_per_source, h_h_per_source each of shape (#nSources,).


        """
        data_indices = self.xp.asarray(data_indices)

        d_h, h_h = self.compute_inner_products_per_segment(params, data_indices, data, psd)

        d_h_per_source = self.xp.sum(d_h, axis=1)
        h_h_per_source = self.xp.sum(h_h, axis=1)
        # log_likelihoods = -0.5 * (h_h_per_source.real - 2*d_h_per_source.real)

        return d_h_per_source, h_h_per_source
