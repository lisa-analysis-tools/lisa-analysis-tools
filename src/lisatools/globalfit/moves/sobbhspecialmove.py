"""SOBBH multi-GPU add/remove move: likelihood via gwtf in-kernel inner products."""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from ...utils.utility import asnumpy
from ...utils.constants import MTSUN_SI
from .addremovemove import MultiGPUResidualAddRemoveMove

if TYPE_CHECKING:
    from eryn.prior import ProbDistContainer
    from eryn.utils.transform import TransformContainer
    from typing import Any
    from ...domaincomputation import DomainComputationGroupArray
    from ...sources.sobbh import GWTF_generator


def update_fn(i, last_sample, sampler):
    """Search-mode update hook that copies the cold chain into the hottest chain."""
    print("max logl:", last_sample.log_like.max())
    last_sample.branches_coords["mbh"][-1] = last_sample.branches_coords["mbh"][0]
    last_sample.log_like[-1] = last_sample.log_like[0]
    last_sample.log_prior[-1] = last_sample.log_prior[0]


class SOBBHSpecialMove(MultiGPUResidualAddRemoveMove):
    def __init__(
        self, 
        dcga: DomainComputationGroupArray,
        waveform_gen: GWTF_generator,
        branch_name: str,
        coords_shape: tuple,
        waveform_gen_kwargs: dict,
        waveform_like_kwargs: dict,
        num_repeats: int,
        transform_fn: TransformContainer,
        priors: ProbDistContainer,
        inner_moves: list,
        Tmax: float = np.inf,
        betas_all: np.ndarray = None,
        permute_every: int = 20,
        pad_out_of_prior: bool = False,
        run_async: bool = False,
        run_threaded: bool = False,
        tc_min_days: float = None,
        **kwargs
    ):
        # Reject proposals merging less than tc_min_days after data start (None = off).
        self.tc_min_days = tc_min_days

        waveform_gen_method: str = "get_signal_for_residuals"
        waveform_like_method: str = "compute_inner_products"

        if not hasattr(waveform_gen, waveform_gen_method):
            raise ValueError(f"GWTF generator must have method {waveform_gen_method} for SOBBHSpecialMove.")
        if not hasattr(waveform_gen, waveform_like_method):
            raise ValueError(f"GWTF generator must have method {waveform_like_method} for SOBBHSpecialMove.")


        super().__init__(
            dcga,
            waveform_gen,
            branch_name,
            coords_shape,
            waveform_gen_method,
            waveform_gen_kwargs,
            waveform_like_kwargs,
            num_repeats,
            transform_fn,
            priors,
            inner_moves,
            Tmax,
            betas_all,
            permute_every,
            pad_out_of_prior,
            run_async,
            run_threaded,
            waveform_like_method,
            **kwargs
        )
                
    def setup_likelihood_here(self, coords):
        ''''
        Setup the likelihood for the sobbh move.
        - One split per GPU. 
        - Aribitrary number of walkers per split, on one GPU. 
        '''

        super().setup_likelihood_here(coords)            # <d|d> per split -> self.acs.cpp_split(i).d_d
        # Residuals and PSDs per split. 
        self._data_gwtf, self._psd_gwtf = [], []

        for i, device in enumerate(self.acs.gpus or [None] * self.acs.num_splits):

            # Number of walkers in split i 
            n = len(self.acs.gpu_splits[i])

            # Waveform generator on that device. 
            gen = self.waveform_generators[i]

            # Inside that GPU
            with self.acs.device_context(device):

                # Reshape data from flat to (n_in_split, 3 XYZ, nT, NF_active)
                data = self.acs.linear_data_arr[i].reshape(n, self.acs.nchannels, *self.acs.end_shape)
                # Reshape invC from flat to (n_in_split, 3, 3, nT, NF_active)
                invC = self.acs.linear_psd_arr[i].reshape(n, *self.acs.shape_sens, *self.acs.end_shape)

                self._data_gwtf.append(gen.to_gwtf_layout_from_xyz(data))
                self._psd_gwtf.append(gen.psd_aet_from_invC(invC))

    def _compute_like_chunk(self, coords_in, data_index):
        '''
        data_index is the walker index in the flattened data array, tiled over temepratures. 
        One residual per walker. 

        coords_in (nwalkers x ntemps, ndim) shape.
        data_index (nwalkers x ntemps, ) shape. 
        
        '''
        # Split batch into splits, one per GPU.
        positions, intra, _ = self.acs.unpack_indices(data_index)
        # positions[i] which rows of the batch belong to split i.
        # intra[i] walkers index within split i.
        # _ noise index, so not needed here. 

        # Operation happening on one split, i.e. on one GPU. 
        def op(i):
            # Grab the waveform generator stored on this gpu.
            gen = self.waveform_generators[i]

            # Intra split walker indices, as a cupy array.
            idx = gen.xp.asarray(intra[i], dtype=gen.xp.int32)

            # Grab the inner-product terms for each walker in this split, and compute the log-likelihood.
            d_h,h_h = gen.compute_inner_products(coords_in[positions[i]], idx, self._data_gwtf[i], self._psd_gwtf[i])
            d_d = self.acs.cpp_split(i).d_d[idx]

            return asnumpy(-0.5 * (d_d + h_h.real - 2 * d_h.real))

        # Run the operation on each split.
        out = self.acs._loop_operation(op, [(i,) for i in range(self.acs.num_splits)],
                                    positions_per_split=positions, run_threaded=self.run_threaded)

        # Create a full-length log-likelihood array.        
        ll = np.full(len(data_index), -1e300)

        for i, pos in enumerate(positions):
            # I.e. if there were any walkers assigned to this split, fill in their log-likelihoods.
            if len(pos):
                ll[pos] = out[i]

        # Reject templates merging within tc_min_days (Newtonian tc from Mc, f0).
        if self.tc_min_days is not None:
            Mc, f0 = asnumpy(coords_in[:, 0]), asnumpy(coords_in[:, 4])
            # 0PN merger time in seconds.
            tc = 5 / 256 * (Mc * MTSUN_SI) ** (-5 / 3) * (np.pi * f0) ** (-8 / 3)
            ll[tc < self.tc_min_days * 86400] = -1e300

        # Guard against any NaN or inf values in the log-likelihood array.
        return np.where(np.isfinite(ll), ll, -1e300)