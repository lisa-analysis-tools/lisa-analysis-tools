from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Optional
import numpy as np
from numpy.typing import ArrayLike

from eryn.prior import ProbDistContainer
from eryn.priors.joint import JointPrior
from eryn.utils.transform import TransformContainer

from ...utils.typing import NDArrayLike
from .gfpriors import BaseSourcePrior


class GalacticBinaryPrior(BaseSourcePrior):
    r"""Galactic Binary specific prior infrastructure.
    
    If left as None, the initialisation defaults to a standard 8-parameter 
    sampling basis mapped to a 9-parameter physical basis (injecting \ddot{f} = 0).
    """
    
    def __init__(
        self,
        param_priors: Dict[str | Tuple[str, ...], Callable[..., Any] | Any] | ProbDistContainer,
        param_prior_inputs: Optional[Dict[str | Tuple[str, ...], List[Any]] | List[List[Any]]] = None,
        source_name: Optional[str] = None,
        sampling_params: Optional[List[str]] = None,
        physical_params: Optional[List[str]] = None,
        fill_dict: Optional[Dict[str, float]] = None,
        param_transforms: Optional[Dict[str | Tuple[str, ...], Callable[..., Any]] | TransformContainer] = None,
        use_cupy: bool = False,
        return_gpu: bool = False,
        verbose: bool = False
    ):  
        
        source_name = source_name if source_name is not None else "gb"
        
        if sampling_params is None:
            sampling_params = [
                "A", "f0", "fdot", "phi0",
                "cos_iota", "psi", "alpha", "sin_delta"
            ]

        if physical_params is None:
            # Names persist through the value transforms (exp / arccos /
            # arcsin), matching GBSetup.init_sampling_info and
            # make_gb_transform_container -- TransformContainer requires
            # every sampling name to appear in the output basis.
            physical_params = [
                "A", "f0", "fdot", "fddot",
                "phi0", "cos_iota", "psi", "alpha", "sin_delta"
            ]

        if fill_dict is None:
            fill_dict = {"fddot": 0.0}

        if param_transforms is None:
            param_transforms = {
                "A": np.exp,
                "cos_iota": np.arccos,
                "sin_delta": np.arcsin,
            }
            
        super().__init__(
            source_name=source_name,
            sampling_params=sampling_params,
            param_priors=param_priors,
            param_prior_inputs=param_prior_inputs,
            physical_params=physical_params,
            fill_dict=fill_dict,
            param_transforms=param_transforms,
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            verbose=verbose
        )


def f_Hz_to_mHz(f_Hz):
    """Convert frequency from Hz to mHz."""
    return f_Hz * 1e3

def get_fdot_mojito(f, sign="+"):
    """Get fdot of GW according to predetermined lisa priors for mojito    

    Args:
        f (xp.ndarray): Frequency of gravitational wave in Hz.
        sign (str): Either positive or negative fdot bound.

    Returns:
        xp.ndarray: fdot.

    Raises:
        ValueError: Inputs are incorrect.
    """
    assert f is not None
    
    if sign == "+":
        fdot = 3e-21 * (f / 1e-4) ** (11/3)
    elif sign == "-":
        fdot = -2e-20 * (f / 4e-4) ** (16/3)
    else:
        raise ValueError("sign must be either positive '+' or negative '-'.")

    return fdot


def f_mHz_to_Hz_and_fdot(f0_mHz: NDArrayLike, fdot: NDArrayLike) -> Tuple[NDArrayLike, NDArrayLike]:
    """Map f0 from mHz to Hz; fdot is already in Hz/s and passes through."""
    f0_Hz = f0_mHz * 1e-3
    return f0_Hz, fdot


class MojitoF0mHzFdotPrior(JointPrior):
    """Joint prior for f0 in mHz and fdot in Hz/s.

    f0 is uniform between ``f0_min_mHz`` and ``f0_max_mHz``. fdot is uniform between the
    mojito bounds at that f0 (``get_fdot_mojito``), so its density depends on f0.
    """

    def __init__(
        self,
        f0_min_mHz: float = 0.1,
        f0_max_mHz: float = 21.0,
        use_cupy: bool = False,
        return_gpu: bool = False,
        **kwargs
    ):
        super().__init__(
            names=("f0", "fdot"),
            names_phys=("f0", "fdot"), # Maps to the physical Hz basis
            latex_labels=(r"f_0", r"\dot{f}"),
            units=("mHz", "Hz/s"),
            use_cupy=use_cupy,
            return_gpu=return_gpu,
            inverse_transform_nd=f_mHz_to_Hz_and_fdot,
            **kwargs
        )
        self.f0_min = f0_min_mHz
        self.f0_max = f0_max_mHz

    def _get_fdot_bounds(self, f0_mHz: NDArrayLike) -> tuple[NDArrayLike, NDArrayLike]:
        """fdot bounds at f0: the mojito power laws take f0 in Hz."""
        f0_Hz = f0_mHz * 1e-3
        return get_fdot_mojito(f0_Hz, sign="-"), get_fdot_mojito(f0_Hz, sign="+")

    def rvs(self, size: int | tuple[int, ...] = (1,), **kwargs) -> NDArrayLike:
        if isinstance(size, int):
            size = (size,)

        # 1. Sample f0 in mHz
        f0_mHz_samples = self.xp.random.uniform(self.f0_min, self.f0_max, size=size)

        # 2. Get bounds and sample fdot
        fdot_min, fdot_max = self._get_fdot_bounds(f0_mHz_samples)
        fdot_samples = self.xp.random.uniform(fdot_min, fdot_max, size=size)

        samples = self.xp.stack([f0_mHz_samples, fdot_samples], axis=-1)
        return self._to_device(samples)

    def logpdf(self, x: ArrayLike, **kwargs) -> NDArrayLike:
        x_arr = self.xp.asarray(x)

        if x_arr.shape[-1] != 2:
            raise ValueError("Expected last dimension of x to be 2.")

        f0_mHz = x_arr[..., 0]
        fdot = x_arr[..., 1]

        out = self.xp.full_like(f0_mHz, -self.xp.inf, dtype=self.xp.float64)

        mask_f0 = (f0_mHz >= self.f0_min) & (f0_mHz <= self.f0_max)
        fdot_min, fdot_max = self._get_fdot_bounds(f0_mHz)
        mask_fdot = (fdot >= fdot_min) & (fdot <= fdot_max)

        valid = mask_f0 & mask_fdot
        valid_fdot_diff = self.xp.where(valid, fdot_max - fdot_min, 1.0)

        # p(f0, fdot) = p(f0) * p(fdot|f0)
        f0_log_norm = -self.xp.log(self.f0_max - self.f0_min)
        out[valid] = f0_log_norm - self.xp.log(valid_fdot_diff[valid])

        return self._to_device(out)
