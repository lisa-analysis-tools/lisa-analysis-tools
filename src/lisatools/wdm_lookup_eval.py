# src/lisatools/wdm_lookup_eval.py
"""Vectorized evaluation of an ``n_ref`` WDM lookup table on numpy or cupy arrays.

The table (:class:`lisatools.domains.WDMLookupTable`, build kinds ``n_ref_only`` /
``n_ref_complex``) stores the WDM response at one reference pixel ``(m_ref, n_ref)`` of a unit
linear chirp as a function of the carrier's offset inside its layer (``f_vals_norm``, uniform,
support ``[-k, k+1)`` layers) and its chirp rate (``fdot_vals``, uniform). This module evaluates
that table for arrays of pixels in one shot -- the vectorized twin of
``WDMLookupTable.get_wdm_coeffs`` (whose per-layer Python loop and scipy interpolators are the
EMRI reference path, docs/emri-direct-wdm.md) -- with the same evaluation rule
(``BASIS_CYCLE = "quarter_turn"``):

1. ``delta = f - m * layer_df``; read ``c`` (cos response) and ``s`` (sin response) at
   ``(fdot, delta)``, ``s`` from the seam-continuous table (the build's ``(-1)^block`` bake
   undone at the NODES, never per pixel);
2. if ``(m_ref + n_ref)`` is odd: ``c -> -c``;
3. if the absolute pixel parity ``(m + n)`` is odd: ``(c, s) -> (s, -c)``;
4. ``w[m, n] = amp * (c cos(phi) - s sin(phi))``, ``phi`` the carrier phase at the pixel centre
   ``t_n = n * layer_dt`` (signal ``amp * cos(phi)``, ``phi`` increasing in time).

Interpolation on the uniform ``(fdot, f_norm)`` grid is bilinear (``"linear"``) or Keys cubic
convolution (``"cubic"``, a = -1/2; interpolating; 16 gathers), both written with array-module
gathers so the same code runs on numpy and cupy. Cubic is the default: linear interpolation
across the table's peaked response left a 2.5e-4 amplitude deficit on the EMRI. Entries outside
the table support are zero (and flagged).
"""

from __future__ import annotations

import numpy as np

from .utils.utility import asnumpy


def _keys_weights(t):
    """Catmull-Rom (Keys, a = -1/2) weights for the nodes at offsets -1, 0, 1, 2; ``t`` in
    [0, 1)."""
    t2 = t * t
    t3 = t2 * t
    return (
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    )


class WDMLookupEvaluator:
    """Vectorized ``n_ref`` lookup-table evaluation (see the module docstring for the rule).

    Args:
        table: a built or loaded :class:`~lisatools.domains.WDMLookupTable` of build kind
            ``n_ref_only`` or ``n_ref_complex``.
        interp: ``"cubic"`` (default, Keys cubic convolution) or ``"linear"`` (bilinear).
        force_backend: backend NAME the table arrays are placed on (``"cpu"``, ``"cuda12x"``,
            ...); ``None`` keeps the table's own backend. Chosen here, never per call.
    """

    INTERPS = ("linear", "cubic")
    #: ``"no_parity_turn"`` skips step 3 of the rule. NEGATIVE CONTROL ONLY (tests).
    BASIS_CYCLES = ("quarter_turn", "no_parity_turn")

    def __init__(self, table, interp="cubic", force_backend=None):
        kind = getattr(table, "build_kind", None)
        if kind not in ("n_ref_only", "n_ref_complex"):
            raise ValueError(
                f"WDMLookupEvaluator needs an n_ref_only / n_ref_complex table, got {kind!r}"
            )
        if interp not in self.INTERPS:
            raise ValueError(f"interp must be one of {self.INTERPS}, got {interp!r}")
        self.interp = interp
        self.basis_cycle = "quarter_turn"
        if force_backend is None:
            self._backend_name = table.backend.name.split("_")[-1]
        elif isinstance(force_backend, str):
            self._backend_name = force_backend
        else:
            self._backend_name = force_backend.name.split("_")[-1]

        f_norm = np.asarray(asnumpy(table.f_vals_norm), dtype=float).ravel()
        fdot = np.asarray(asnumpy(table.fdot_vals), dtype=float).ravel()
        if kind == "n_ref_complex":
            cx = np.asarray(asnumpy(table.table_cx))
            tab_cos, tab_sin = np.real(cx), np.imag(cx)
        else:
            tab_cos = np.asarray(asnumpy(table.table_cos), dtype=float)
            tab_sin = np.asarray(asnumpy(table.table_sin), dtype=float)
        tab_cos = np.ascontiguousarray(tab_cos.reshape(fdot.size, f_norm.size), dtype=float)
        tab_sin = np.ascontiguousarray(tab_sin.reshape(fdot.size, f_norm.size), dtype=float)

        self.layer_df = float(table.layer_df)
        self.layer_dt = float(table.layer_dt)
        self.m_ref = int(table.m_ref)
        self.n_ref = int(table.n_ref)
        # undo the build's (-1)^block bake on the chirp (sin) term at the NODES so the stored
        # function is continuous across block seams (WDMLookupTable._sin_unbaked_coeffs)
        block = np.floor(f_norm / self.layer_df + 1e-9).astype(int)
        tab_sin = tab_sin * np.where(block % 2 != 0, -1.0, 1.0)[None, :]

        df = float(f_norm[1] - f_norm[0])
        if not np.allclose(np.diff(f_norm), df, rtol=1e-6, atol=0.0):
            raise ValueError("table f_vals_norm axis is not uniform")
        self.nf = int(f_norm.size)
        self.f_min, self.f_max, self.df = float(f_norm[0]), float(f_norm[-1]), df
        self.nfdot = int(fdot.size)
        if self.nfdot > 1:
            dfd = float(fdot[1] - fdot[0])
            if not np.allclose(np.diff(fdot), dfd, rtol=1e-6, atol=0.0):
                raise ValueError("table fdot_vals axis is not uniform")
        else:
            dfd = 1.0
        self.fdot_min, self.fdot_max, self.dfdot = float(fdot[0]), float(fdot[-1]), dfd

        xp = self.xp
        self.tab_cos = xp.asarray(tab_cos)
        self.tab_sin = xp.asarray(tab_sin)

    @property
    def backend(self):
        import lisatools

        return lisatools.get_backend(self._backend_name)

    @property
    def xp(self):
        return self.backend.xp

    def layers_for(self, f_ref, num_m_layers):
        """Target layers ``floor(f_ref / layer_df) + [-k .. k]`` -> int array ``(..., 2k + 1)``.

        Purely geometric: for ``f_ref < k * layer_df`` this can return negative layers. The
        caller owns the valid layer-range mask for its own grid (this evaluator is built from
        one table and has no notion of any particular caller's ``Nf``).
        """
        xp = self.xp
        k = int(num_m_layers)
        m0 = xp.floor(xp.asarray(f_ref, dtype=float) / self.layer_df).astype(xp.int64)
        return m0[..., None] + xp.arange(-k, k + 1, dtype=xp.int64)

    def _interp(self, tab, u, v):
        """Interpolate ``tab[j, i]`` at fractional indices ``(v, u)`` (fdot, f_norm)."""
        xp = self.xp
        nf, nfd = self.nf, self.nfdot
        i0 = xp.floor(u).astype(xp.int64)
        j0 = xp.floor(v).astype(xp.int64)
        if self.interp == "linear":
            i0c = xp.clip(i0, 0, max(nf - 2, 0))
            j0c = xp.clip(j0, 0, max(nfd - 2, 0))
            tu = u - i0c
            tv = v - j0c
            i1 = xp.minimum(i0c + 1, nf - 1)
            j1 = xp.minimum(j0c + 1, nfd - 1)
            z00 = tab[j0c, i0c]
            z01 = tab[j0c, i1]
            z10 = tab[j1, i0c]
            z11 = tab[j1, i1]
            return (1.0 - tv) * ((1.0 - tu) * z00 + tu * z01) + tv * ((1.0 - tu) * z10 + tu * z11)
        wu = _keys_weights(u - i0)
        wv = _keys_weights(v - j0)
        out = None
        for a in range(4):
            ja = xp.clip(j0 + (a - 1), 0, nfd - 1)
            row = None
            for b in range(4):
                ib = xp.clip(i0 + (b - 1), 0, nf - 1)
                term = wu[b] * tab[ja, ib]
                row = term if row is None else row + term
            term = wv[a] * row
            out = term if out is None else out + term
        return out

    def coeffs(self, amp, phi, f, fdot, n, m):
        """WDM coefficients of carriers ``amp cos(phi)`` at pixels ``(m, n)``.

        All arguments broadcast together (``m`` and ``n`` integer-valued). Returns ``(w, ok)``:
        ``w`` the coefficient (zero where ``ok`` is False) and ``ok`` the in-support mask
        (offset ``f - m * layer_df`` inside the table's offset axis and, for a table with an
        fdot axis, ``fdot`` inside that axis). ``ok`` does not check that ``m`` itself is a
        valid layer on any particular grid (e.g. ``0 <= m < Nf``) -- this evaluator does not
        know the calling grid's ``Nf``, so that mask is the caller's responsibility.
        """
        xp = self.xp
        amp, phi, f, fdot, n, m = xp.broadcast_arrays(
            *(xp.asarray(a) for a in (amp, phi, f, fdot, n, m))
        )
        m = m.astype(xp.int64)
        n = n.astype(xp.int64)
        delta = f - m * self.layer_df
        ok = (delta >= self.f_min) & (delta <= self.f_max)
        u = (delta - self.f_min) / self.df
        if self.nfdot > 1:
            ok = ok & (fdot >= self.fdot_min) & (fdot <= self.fdot_max)
            v = (fdot - self.fdot_min) / self.dfdot
        else:
            v = xp.zeros_like(u)
        c = self._interp(self.tab_cos, u, v)
        s = self._interp(self.tab_sin, u, v)
        if (self.m_ref + self.n_ref) % 2:
            c = -c
        if self.basis_cycle == "quarter_turn":
            odd = ((m + n) % 2) != 0  # ABSOLUTE pixel parity
            cc = xp.where(odd, s, c)
            ss = xp.where(odd, -c, s)
        else:  # negative control: no quarter turn
            cc, ss = c, s
        w = amp * (cc * xp.cos(phi) - ss * xp.sin(phi))
        return xp.where(ok, w, 0.0), ok


__all__ = ["WDMLookupEvaluator"]
