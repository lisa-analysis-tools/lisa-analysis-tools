"""MBH block: the shared source build (``SourceBlock``) plus the ICRS <-> LISA-frame SkyMove wiring."""
from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np

from .base import BuildContext, SourceBlock


def sky_ind_map(basis: list[str], names: dict[str, str]) -> dict[str, int]:
    """``SkyMove`` ``ind_map`` (move name -> column) from the prior's input basis."""
    missing = sorted(set(names.values()) - set(basis))
    if missing:
        raise ValueError(f"the prior basis {basis} lacks {missing}, which SkyMove needs.")
    return {move_name: basis.index(basis_name) for move_name, basis_name in names.items()}


def make_sky_wrappers(orbits, t_ref: float, ind_map: dict[str, int]):
    """``(to_lisa_frame, from_lisa_frame)`` acting on the sky columns of an ``(n, ndim)`` coords array.

    The sampling basis carries ICRS ``(t, alpha, sin_delta, psi)``; SkyMove hops in the LISA frame.
    """
    from functools import partial

    from lisatools.sources.utils import from_lisa_frame, to_lisa_frame

    to_lisa = partial(to_lisa_frame, orbits=orbits, t_ref=t_ref)
    from_lisa = partial(from_lisa_frame, orbits=orbits, t_ref=t_ref)
    t_col, lam_col, sinbeta_col, psi_col = (ind_map[k] for k in ("t_ref", "lam", "sinbeta", "psi"))

    def to_lisa_frame_wrap(coords: np.ndarray) -> np.ndarray:
        out = coords.copy()
        dec = np.arcsin(np.clip(out[:, sinbeta_col], -1.0, 1.0))
        t_L, lam_L, beta_L, psi_L = to_lisa(out[:, t_col], out[:, lam_col], dec, out[:, psi_col])
        out[:, t_col], out[:, lam_col], out[:, sinbeta_col], out[:, psi_col] = (
            t_L, lam_L, np.sin(beta_L), psi_L,
        )
        return out

    def from_lisa_frame_wrap(coords: np.ndarray) -> np.ndarray:
        out = coords.copy()
        beta_L = np.arcsin(np.clip(out[:, sinbeta_col], -1.0, 1.0))
        t_ref_, ra, dec, psi = from_lisa(out[:, t_col], out[:, lam_col], beta_L, out[:, psi_col])
        out[:, t_col], out[:, lam_col], out[:, sinbeta_col], out[:, psi_col] = (
            t_ref_, ra, np.sin(dec), psi,
        )
        return out

    return to_lisa_frame_wrap, from_lisa_frame_wrap


class MBHBlock(SourceBlock):
    """Block for massive black-hole binary inference."""

    type: Literal["mbh"]
    CATALOGUE_KEY: ClassVar[str] = "MBHB"
    SOURCE_IDS_KEY: ClassVar[str] = "mbhb"
    WAVEFORM_MODULES: ClassVar[tuple[str, ...]] = ("lisatools.sources.bbh.waveform",)
    SOURCE_TYPE: ClassVar[str] = "MBHB"
    SETTINGS_NAME: ClassVar[str] = "MBHSettings"
    SETUP_NAME: ClassVar[str] = "MBHSetup"
    SEARCH_INTERFACE: ClassVar[str] = "MBHBSearchInterface"

    # SkyMove ind_map names -> names in the prior's input basis (SkyMove still calls the
    # coalescence-time column "t_ref")
    SKY_BASIS: ClassVar[dict[str, str]] = dict(
        cosinc="cos_iota", psi="psi", lam="alpha", sinbeta="sin_delta", t_ref="t_c"
    )

    def filled_move_kwargs(self, ctx: BuildContext, info: dict[str, Any]) -> dict[str, dict]:
        if not any(move.class_name == "SkyMove" for move in self.inner_moves or []):
            return {}
        ind_map = sky_ind_map(info["basis"], self.SKY_BASIS)
        to_lisa, from_lisa = make_sky_wrappers(ctx.general.orbits, ctx.cfg.data.reference_time, ind_map)
        return {
            "SkyMove": dict(
                ind_map=ind_map, coord_frame="icrs", to_lisa_frame=to_lisa, from_lisa_frame=from_lisa
            )
        }
