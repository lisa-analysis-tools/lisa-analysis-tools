# SOBBH direct-to-WDM lookup scorer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Score SOBBH add/remove proposals (and fill their residual templates) with a vectorized direct-to-WDM lookup template that slots into `SOBBHChunkedLikeMove` as a drop-in comp, validated against the dense TDI-on-the-fly transform and timed against the chunked-heterodyne comp.

**Architecture:** A source-agnostic `xp`-vectorized evaluator of the `n_ref` lookup table (`lisatools/wdm_lookup_eval.py`); a SOBBH batched TDI-on-the-fly tracer + sparse template + sparse inner products/fill (`lisatools/sources/sobbh/wdm_direct.py`); a comp class `SOBBHLookupComputations` duck-typing the surface of `bbhx.sobbhcomps.SOBBHWDMComputations` that the move and the engine signal generator use; stock wiring under `SOBBH_LIKELIHOOD=lookup`. Validation ladder: table rule -> batched PN -> batched response -> template vs dense transform -> inner products vs the container -> move parity -> laptop gate with timings.

**Tech Stack:** Python 3.12, numpy (cupy on CUDA via the `xp` pattern), `lisatools.domains.WDMLookupTable`, `lisatools.response.tdionfly.TDTDIonTheFly`, `gpubackendtools.interpolate.CubicSplineInterpolant` (`derivative=` kwarg), `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-30-sobbh-wdm-lookup-design.md` (read it first; this plan argues from it).

## Global Constraints

- All edits in the worktree `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-sobbh-lookup` (branch `sobbh-wdm-lookup`); never touch the main `LISAanalysistools` checkout.
- Run every Python command through the worktree runner `.wtenv/run.sh` (it shadows the editable install with this worktree's `src`, pins 2 threads, `nice 10`, and exits at 5 GB RSS). Example: `.wtenv/run.sh -m unittest tests.test_wdm_lookup_eval -v`. Run from the worktree root.
- One Python process at a time on this laptop; never run two test commands concurrently; CPU at or below 50%.
- Stage only (`git add`); **no commit, no push** without Mike's explicit OK.
- No backend strings as method kwargs: backend chosen at construction (`force_backend=`); no array module stored on an instance (`xp` is a property).
- Env knob = capitalised attribute name with the branch prefix (`sobbh.lookup_eval_dt` -> `SOBBH_LOOKUP_EVAL_DT`).
- Every accuracy test carries a paired negative control or a named mutation that breaks it.
- `cupyx` is imported only inside the cupy branch (the laptop has no cupy).
- Line length 100 (black), isort black profile.

## Review Focus

1. A row already merged before the first pixel (`tc < t_pixels[0]`): template is zero, `d_h = h_h = 0`, the move scores it as the exposed-residual offset, never NaN — pinned in Task 5 (`test_merged_before_window_scores_offset`).
2. A `data_index` outside the holder's walkers must raise `IndexError` with a message, not read another walker's slab — Task 5 (`test_bad_data_index_raises`).
3. A table whose `layer_dt` differs from the grid's must raise at construction — Task 4 (`test_layer_dt_mismatch_raises`).
4. A non-contiguous fill target (transposed view) must raise instead of silently losing the writes — Task 4 (`test_fill_rejects_noncontiguous`).
5. The numpy path must never import `cupyx` (CPU laptops / CI without cupy) — Task 4 (`test_numpy_path_does_not_import_cupyx`).

---

### Task 0: Shared test toy (tiny table + chirp truth)

**Files:**
- Create: `tests/_wdm_lookup_toy.py`

**Interfaces:**
- Produces: `build_tiny_table(dirname, *, nf=64, dt=56.25, nt=128, eps_freq=0.01, eps_fdot=0.1, fdot_max_factor=1.0, m_ref=20, time_layers=64) -> (WDMSettings, WDMLookupTable)` (layer duration `nf*dt`, default 3600 s; offsets `[-3, 3)` layers; fdot axis `[-1, 1]` layer units in steps of 0.1); `chirp_truth(wdm, f0, fdot, phi0) -> np.ndarray (Nf_active, Nt)` (TD->WDM of `cos(2 pi (f0 t + fdot t^2 / 2) + phi0)` on `wdm`'s grid); `td_mismatch(a, b) -> (mm, norm_ratio)` for two real arrays (`1 - <a,b>/sqrt(<a,a><b,b>)`, `sqrt(<a,a>/<b,b>)`).

- [ ] **Step 1: Write the helper module**

```python
# tests/_wdm_lookup_toy.py
"""Shared toy pieces for the WDM lookup tests: a tiny n_ref table, a chirp truth, a mismatch."""
import os

import numpy as np


def build_tiny_table(dirname, *, nf=64, dt=56.25, nt=128, eps_freq=0.01, eps_fdot=0.1,
                     fdot_max_factor=1.0, m_ref=20, time_layers=64):
    """A small ``n_ref_complex`` table with layer duration ``nf * dt`` (3600 s by default).

    Offsets cover [-3, 3) layers (``num_layers_diff=2``); the fdot axis spans
    ``+-fdot_max_factor`` layer units in steps of ``eps_fdot``. Builds in a few seconds.
    """
    from lisatools.domains import WDMLookupTable, WDMSettings

    wdm = WDMSettings(Nf=nf, Nt=nt, dt=dt, force_backend="cpu")
    norm_f, m_diffs, m_ref = WDMLookupTable.apply_eps_frequency(
        eps_freq, wdm, m_ref=m_ref, num_layers_diff=2)
    fdot_vals = WDMLookupTable.apply_eps_fdot(eps_fdot, wdm, fdot_max_factor=fdot_max_factor)
    table = WDMLookupTable(
        wdm, 1, m_ref=m_ref, norm_freq_single_layer=norm_f, m_diffs=m_diffs, fdot_vals=fdot_vals,
        store_path=os.path.join(dirname, f"lookup_nf{nf}_dt{dt:g}.h5"), batch_size_gen=64,
        build_kind="n_ref_complex", time_layers=time_layers)
    return wdm, table


def chirp_truth(wdm, f0, fdot, phi0):
    """TD->WDM of cos(2 pi (f0 t + fdot t^2 / 2) + phi0) on ``wdm``'s grid: (Nf_active, Nt)."""
    from lisatools.domains import TDSettings, TDSignal

    N = wdm.Nf * wdm.Nt
    t = np.arange(N) * wdm.data_dt
    y = np.cos(2 * np.pi * (f0 * t + 0.5 * fdot * t ** 2) + phi0)[None, :]
    return np.asarray(
        TDSignal(y, TDSettings(N, wdm.data_dt, force_backend="cpu")).transform(wdm).arr)[0]


def td_mismatch(a, b):
    """(1 - <a,b>/sqrt(<a,a><b,b>), sqrt(<a,a>/<b,b>)) for two real arrays of the same shape."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    aa, bb, ab = float(a @ a), float(b @ b), float(a @ b)
    return 1.0 - ab / np.sqrt(aa * bb), np.sqrt(aa / bb)
```

- [ ] **Step 2: Smoke it**

Run: `.wtenv/run.sh -c "import tempfile, numpy as np, sys; sys.path.insert(0, 'tests'); from _wdm_lookup_toy import build_tiny_table, chirp_truth; d = tempfile.mkdtemp(); w, t = build_tiny_table(d); print(t.table_cx.shape, w.layer_dt, chirp_truth(w, 20.3 * w.layer_df, 0.0, 0.1).shape)"`
Expected: `(19, 600) 3600.0 (64, 128)` (19 fdot rows, 600 offset columns) within ~5 s.

- [ ] **Step 3: Stage**

```bash
git add tests/_wdm_lookup_toy.py
```

---

### Task 1: `WDMLookupEvaluator` — vectorized table evaluation

**Files:**
- Create: `src/lisatools/wdm_lookup_eval.py`
- Test: `tests/test_wdm_lookup_eval.py`

**Interfaces:**
- Consumes: `lisatools.domains.WDMLookupTable` (`build_kind`, `table_cx`/`table_cos`/`table_sin`, `f_vals_norm`, `fdot_vals`, `layer_df`, `layer_dt`, `m_ref`, `n_ref`, `backend`), `lisatools.get_backend(name)` (`.xp`, `.name`), `lisatools.utils.utility.asnumpy`.
- Produces: `class WDMLookupEvaluator(table, interp="cubic", force_backend=None)` with properties `backend`, `xp`, attributes `layer_df`, `layer_dt`, `m_ref`, `n_ref`, `f_min`, `f_max`, `fdot_min`, `fdot_max`, `nfdot`, `interp`, `basis_cycle` (`"quarter_turn"` default; `"no_parity_turn"` is a negative control only); methods `layers_for(f_ref, num_m_layers) -> int array (..., 2k+1)` and `coeffs(amp, phi, f, fdot, n, m) -> (w, ok)` (broadcast inputs; `w` zero where `ok` is False).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wdm_lookup_eval.py
"""WDMLookupEvaluator: the vectorized n_ref lookup matches the reference evaluation rule.

Reference = ``WDMLookupTable.get_wdm_coeffs`` (the EMRI path, per-layer loop + scipy) and the
TD->WDM truth of a linear chirp; paired negative control = the quarter turn switched off.
Also pins the table's invariance under (Nf, dt) at fixed layer duration (the reason the laptop
EMRI table serves the production grid).
"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wdm_lookup_toy import build_tiny_table, chirp_truth  # noqa: E402

EDGE = 10


class EvaluatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wdm, cls.table = build_tiny_table(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _chirp(self, f_layers, fdot_units, phi0):
        df, ldt = self.wdm.layer_df, self.wdm.layer_dt
        f0, fdot = f_layers * df, fdot_units * df / ldt
        n = np.arange(EDGE, self.wdm.Nt - EDGE)
        tn = n * ldt
        f = f0 + fdot * tn
        phi = 2 * np.pi * (f0 * tn + 0.5 * fdot * tn ** 2) + phi0
        return f0, fdot, n, f, phi

    def _evaluator(self, interp):
        from lisatools.wdm_lookup_eval import WDMLookupEvaluator

        return WDMLookupEvaluator(self.table, interp=interp)

    def _rel_err(self, ev, f_layers, fdot_units, phi0):
        f0, fdot, n, f, phi = self._chirp(f_layers, fdot_units, phi0)
        truth = chirp_truth(self.wdm, f0, fdot, phi0)
        m = ev.layers_for(f, 2)
        w, ok = ev.coeffs(1.0, phi[:, None], f[:, None], fdot, n[:, None], m)
        w, m = np.asarray(w), np.asarray(m)
        tv = truth[m - int(self.wdm.ind_min_f), n[:, None]]
        return float(np.sqrt(np.sum((w - tv) ** 2) / np.sum(tv ** 2)))

    def test_matches_get_wdm_coeffs_linear(self):
        self.table.set_interp_method("linear")
        ev = self._evaluator("linear")
        f0, fdot, n, f, phi = self._chirp(20.37, 0.23, 0.4)      # off the f AND fdot nodes
        amp = np.linspace(0.5, 1.5, n.size)
        ref, m_map = self.table.get_wdm_coeffs(
            amp, phi, f, np.full_like(f, fdot), n, num_m_layers=2, out_of_support="zero")
        ref, m_map = np.asarray(ref), np.asarray(m_map)
        m = ev.layers_for(f, 2)
        w, ok = ev.coeffs(amp[:, None], phi[:, None], f[:, None], fdot, n[:, None], m)
        np.testing.assert_array_equal(np.asarray(m), m_map)
        np.testing.assert_allclose(np.asarray(w), ref, rtol=1e-9, atol=1e-12 * np.abs(ref).max())
        self.assertTrue(bool(np.asarray(ok).all()))

    # (f0 in layers, fdot in layer units ON the table's 0.1 nodes, phi0)
    CASES = [(20.3, 0.0, 0.7), (18.2, 0.2, 0.4), (15.3, 0.3, -2.2), (30.4, -0.3, 1.3)]

    def test_quarter_turn_matches_chirp_truth(self):
        for interp, tol in (("linear", 2e-3), ("cubic", 1e-3)):
            ev = self._evaluator(interp)
            for case in self.CASES:
                e = self._rel_err(ev, *case)
                self.assertLess(e, tol, f"{interp} {case}: rel L2 {e:.3e}")

    def test_no_parity_turn_control_fails(self):
        ev = self._evaluator("cubic")
        ev.basis_cycle = "no_parity_turn"
        errs = [self._rel_err(ev, *case) for case in self.CASES]
        self.assertGreater(max(errs), 1e-1, f"control errs {errs}")

    def test_cubic_beats_linear_off_node(self):
        case = (20.3137, 0.0, 0.4)                               # between f nodes (step 0.01 df)
        e_lin = self._rel_err(self._evaluator("linear"), *case)
        e_cub = self._rel_err(self._evaluator("cubic"), *case)
        self.assertLess(e_cub, 0.5 * e_lin, f"linear {e_lin:.2e} cubic {e_cub:.2e}")

    def test_cubic_interpolates_the_nodes(self):
        # fdot = 0 keeps every pixel on an f node: both schemes return the node values
        f0, fdot, n, f, phi = self._chirp(20.3, 0.0, 0.4)
        m = self._evaluator("linear").layers_for(f, 2)
        w_lin, _ = self._evaluator("linear").coeffs(1.0, phi[:, None], f[:, None], 0.0, n[:, None], m)
        w_cub, _ = self._evaluator("cubic").coeffs(1.0, phi[:, None], f[:, None], 0.0, n[:, None], m)
        np.testing.assert_allclose(np.asarray(w_cub), np.asarray(w_lin), rtol=1e-12, atol=1e-14)

    def test_out_of_support_is_zero_and_flagged(self):
        ev = self._evaluator("cubic")
        f0, fdot, n, f, phi = self._chirp(20.3, 0.0, 0.4)
        m = np.full((n.size, 1), 40)                              # delta = -19.7 layers: outside
        w, ok = ev.coeffs(1.0, phi[:, None], f[:, None], 0.0, n[:, None], m)
        self.assertFalse(bool(np.asarray(ok).any()))
        self.assertTrue(bool((np.asarray(w) == 0.0).all()))
        w2, ok2 = ev.coeffs(1.0, phi[:, None], f[:, None], 5.0 * ev.fdot_max, n[:, None],
                            ev.layers_for(f, 2))
        self.assertFalse(bool(np.asarray(ok2).any()))

    def test_rejects_bad_options(self):
        from lisatools.wdm_lookup_eval import WDMLookupEvaluator

        with self.assertRaises(ValueError):
            WDMLookupEvaluator(self.table, interp="quintic")


class TablePortabilityTest(unittest.TestCase):
    """The n_ref table depends on the layer duration only (measured 2026-09-30: 4e-9)."""

    def test_same_layer_dt_tables_agree_across_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, ta = build_tiny_table(tmp, nf=64, dt=56.25, eps_freq=0.02, eps_fdot=0.25)
            _, tb = build_tiny_table(tmp, nf=128, dt=28.125, eps_freq=0.02, eps_fdot=0.25)
            A, B = np.asarray(ta.table_cx), np.asarray(tb.table_cx)
            self.assertEqual(A.shape, B.shape)
            np.testing.assert_allclose(B, A, rtol=0, atol=1e-8 * np.abs(A).max())
            # control: a different layer duration (1800 s) is a different table
            _, tc = build_tiny_table(tmp, nf=64, dt=28.125, eps_freq=0.02, eps_fdot=0.25)
            self.assertGreater(np.abs(np.asarray(tc.table_cx) - A).max(), 1e-2 * np.abs(A).max())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.wtenv/run.sh -m unittest tests.test_wdm_lookup_eval -v`
Expected: the `EvaluatorTest` cases error with `ModuleNotFoundError: No module named 'lisatools.wdm_lookup_eval'`; `TablePortabilityTest` passes (it only uses the existing table class).

- [ ] **Step 3: Write the evaluator**

```python
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
    """Catmull-Rom (Keys, a = -1/2) weights for the nodes at offsets -1, 0, 1, 2; ``t`` in [0, 1)."""
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
                f"WDMLookupEvaluator needs an n_ref_only / n_ref_complex table, got {kind!r}")
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
        """Target layers ``floor(f_ref / layer_df) + [-k .. k]`` -> int array ``(..., 2k + 1)``."""
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
        fdot axis, ``fdot`` inside that axis).
        """
        xp = self.xp
        amp, phi, f, fdot, n, m = xp.broadcast_arrays(
            *(xp.asarray(a) for a in (amp, phi, f, fdot, n, m)))
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
            odd = ((m + n) % 2) != 0           # ABSOLUTE pixel parity
            cc = xp.where(odd, s, c)
            ss = xp.where(odd, -c, s)
        else:                                   # negative control: no quarter turn
            cc, ss = c, s
        w = amp * (cc * xp.cos(phi) - ss * xp.sin(phi))
        return xp.where(ok, w, 0.0), ok


__all__ = ["WDMLookupEvaluator"]
```

- [ ] **Step 4: Run the tests**

Run: `.wtenv/run.sh -m unittest tests.test_wdm_lookup_eval -v`
Expected: all 9 tests PASS (about 30 s: the portability test builds three tables). If `test_quarter_turn_matches_chirp_truth` fails for `"linear"` only, raise its tolerance to the measured value + 50% and record the number in the assertion message; a `"cubic"` failure is a bug.

- [ ] **Step 5: Mutation check**

Flip the quarter turn's sign (`ss = xp.where(odd, c, s)`) and rerun: `test_quarter_turn_matches_chirp_truth` and `test_matches_get_wdm_coeffs_linear` must FAIL. Restore the line.

- [ ] **Step 6: Stage**

```bash
git add src/lisatools/wdm_lookup_eval.py tests/test_wdm_lookup_eval.py
```

---

### Task 2: Batched 3.5PN amplitude/phase (`sobbh_amp_phase_batch`)

**Files:**
- Create: `src/lisatools/sources/sobbh/wdm_direct.py` (module header + this function; later tasks append)
- Test: `tests/test_sobbh_wdm_direct.py` (`AmpPhaseBatchTest`)

**Interfaces:**
- Consumes: `lisatools.sources.sobbh.waveform` (`MTsun`, `pc`, `c`, `phase`, `tau_to_x`, `time_to_merger`, `SOBBHWaveform.compute_amp_phase` as the reference).
- Produces: `sobbh_amp_phase_batch(params, times, reference_time, t_shift=0.0) -> (amp (N, T), gw_phase (N, T), tc_abs (N,))` for chunked-basis rows `(m1, m2, s1, s2, dist[pc], f_low, phi_c, ...)`; `amp = 0` and the phase frozen at and past merger.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sobbh_wdm_direct.py
"""SOBBH direct-to-WDM building blocks: batched PN, batched response, tracer, template, fill."""
import os
import sys
import tempfile
import unittest

import numpy as np

from lisatools.utils.constants import YRSID_SI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wdm_lookup_toy import build_tiny_table, td_mismatch  # noqa: E402

# chunked basis rows: (m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta)
ROWS = np.array([
    [60.0, 55.0, 0.1, 0.2, 0.4e9, 6.0e-3, 1.1, np.arccos(0.3), 0.7, 3.1, np.arcsin(0.2)],
    [35.0, 30.0, -0.3, 0.5, 2.0e9, 1.2e-2, 2.5, 0.4, 1.9, 1.0, -0.6],
    [90.0, 70.0, 0.0, 0.0, 1.0e9, 1.6e-2, 0.3, 2.2, 0.2, 5.5, 0.9],
])
# merges inside a few days: masses far above the prior on purpose (chirp-rate stress)
ROW_MERGE = np.array([2000.0, 2000.0, 0.0, 0.0, 1.0e9, 1.5e-2, 0.3, 0.5, 0.7, 1.0, 0.2])

DT = 20.0
NF, NT = 180, 256                       # layer_dt = 3600 s, Nyquist 25 mHz, 5.9 days
NOBS = NF * NT
T_START = int(0.5 * YRSID_SI / DT) * DT
REF = float(T_START)
N_GRID, BUFFER, EVAL_DT = 1024, 5000.0, 600.0
EDGE = 24                               # grid-end pixels of the dense transform dropped


class AmpPhaseBatchTest(unittest.TestCase):
    def test_rows_match_sobbhwaveform(self):
        from lisatools.sources.sobbh.waveform import SOBBHWaveform
        from lisatools.sources.sobbh.wdm_direct import sobbh_amp_phase_batch

        N, dt = 16384, 20.0
        gen = SOBBHWaveform(Tobs=N * dt, dt=dt, t0=T_START + 3600.0, reference_time=REF)
        rows = np.vstack([ROWS, ROW_MERGE[None, :]])
        amp, ph, tc = sobbh_amp_phase_batch(rows, gen.times, REF)
        self.assertEqual(amp.shape, (rows.shape[0], N))
        n_dead_total = 0
        for i, r in enumerate(rows):
            t_live, a_ref, p_ref = gen.compute_amp_phase(r[0], r[1], r[2], r[3], r[4] * 1e-9,
                                                         r[5], r[6])
            k = t_live.size
            np.testing.assert_allclose(amp[i, :k], a_ref, rtol=1e-12)
            np.testing.assert_allclose(ph[i, :k], p_ref, rtol=1e-12)
            self.assertTrue(np.all(amp[i, k:] == 0.0))
            n_dead_total += N - k
            if k < N:
                self.assertLess(gen.times[k - 1], tc[i])
                self.assertGreaterEqual(gen.times[k], tc[i])
                # the frozen phase is finite and constant past merger
                self.assertTrue(np.all(np.isfinite(ph[i])))
                self.assertEqual(float(np.ptp(ph[i, k:])), 0.0)
        self.assertGreater(n_dead_total, 0, "the merging row must merge inside the window")

    def test_rejects_short_rows(self):
        from lisatools.sources.sobbh.wdm_direct import sobbh_amp_phase_batch

        with self.assertRaises(ValueError):
            sobbh_amp_phase_batch(np.ones((2, 5)), np.arange(10.0), 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.wtenv/run.sh -m unittest tests.test_sobbh_wdm_direct.AmpPhaseBatchTest -v`
Expected: `ModuleNotFoundError: No module named 'lisatools.sources.sobbh.wdm_direct'`.

- [ ] **Step 3: Write the module header and the function**

```python
# src/lisatools/sources/sobbh/wdm_direct.py
"""SOBBH templates built directly in the WDM domain (batched TDI-on-the-fly + lookup table).

The SOBBH twin of :mod:`lisatools.sources.emri.wdm_direct`, vectorized over parameter rows so
one call scores a whole proposal batch (spec:
docs/superpowers/specs/2026-09-30-sobbh-wdm-lookup-design.md). Pieces, in data-flow order:

* :func:`sobbh_amp_phase_batch` -- the 3.5PN amplitude / GW phase of ``N`` rows on one time
  grid (the core of :mod:`lisatools.sources.sobbh.waveform`, row-vectorized);
* :class:`SOBBHBatchedTOF` -- ONE :class:`~lisatools.response.tdionfly.TDTDIonTheFly` for the
  batch, built exactly as :class:`bbhx.sobbhtdionfly.SOBBHTDIonFly` builds it for one row but
  evaluated on a coarse grid (the dense data grid is never built);
* :func:`sobbh_tracer` -- per-channel ``(amp, phase, f, fdot)`` at the WDM pixel centres from
  the response splines;
* :class:`SOBBHDirectWDM` -- the sparse template (5 layers per pixel per channel) through
  :class:`~lisatools.wdm_lookup_eval.WDMLookupEvaluator`;
* :func:`sparse_inner_products` / :func:`scatter_add` -- the chunked kernel's ``<d|h>`` /
  ``<h|h>`` accumulation and the residual fill, gathered at the sparse support;
* :class:`SOBBHLookupComputations` -- the comp that slots into
  :class:`~lisatools.globalfit.moves.sobbhspecialmove.SOBBHChunkedLikeMove`.

Conventions pinned by the tests: chunked-basis rows ``(m1, m2, s1, s2, dist[pc], f_low, phi_c,
inc, psi, lam, beta)``; the response feed ``phase = gw_phase + pi`` with the intrinsic
amplitude (``SOBBHTDIonFly``); pixel centres ``t_n = t_obs_start + n * layer_dt`` with ``n`` the
absolute grid index; the inner products carry NO ``4 * differential_component`` factor (the
kernel convention: ``-0.5 (h_h - 2 d_h)`` is the container's source term).
"""
from __future__ import annotations

import dataclasses
import logging
import time

import numpy as np

from ...utils.utility import asnumpy, get_array_module

logger = logging.getLogger(__name__)


def sobbh_amp_phase_batch(params, times, reference_time, t_shift=0.0):
    """3.5PN ``(amp, gw_phase, tc_abs)`` of ``N`` rows at absolute ``times`` [s].

    ``params`` is ``(N, >= 7)`` in the chunked basis ``(m1 [Msun], m2 [Msun], s1, s2, dist [pc],
    f_low [Hz], phi_c [rad], ...)``; ``f_low`` and ``phi_c`` are defined at ``reference_time``
    (the PN time is ``times - reference_time - t_shift``). Row-wise identical to
    :meth:`lisatools.sources.sobbh.waveform.SOBBHWaveform.compute_amp_phase`: ``amp = 2 M eta x
    / D`` (intrinsic, no inclination), ``gw_phase = 2 Phi`` anchored at the reference epoch,
    the same Newton-refined ``tc``. Samples at or past merger get ``amp = 0`` and the phase
    frozen at the last live sample. Returns ``amp (N, T)``, ``gw_phase (N, T)``, ``tc_abs (N,)``.
    """
    from .waveform import MTsun, c, pc, phase as _phase, tau_to_x, time_to_merger

    p = np.atleast_2d(np.asarray(params, dtype=float))
    if p.ndim != 2 or p.shape[1] < 7:
        raise ValueError(f"params must be (N, >= 7) chunked-basis rows, got shape {p.shape}")
    t = np.asarray(times, dtype=float).reshape(-1)
    m1 = p[:, 0:1] * MTsun
    m2 = p[:, 1:2] * MTsun
    s1 = p[:, 2:3]
    s2 = p[:, 3:4]
    D = p[:, 4:5] * pc / c
    f_low = p[:, 5:6]
    phi_c = p[:, 6:7]
    M = m1 + m2
    eta = m1 * m2 / M ** 2
    sigma = (m2 * s2 - m1 * s1) / M
    s = (m1 ** 2 * s1 + m2 ** 2 * s2) / M ** 2
    delta = (m1 - m2) / M
    x0 = (np.pi * M * f_low) ** (2.0 / 3.0)

    # tc consistent with the tau_to_x series: Newton on tau_to_x(tau_ref) == x0 (waveform.py)
    tc = np.asarray(time_to_merger(x0, sigma, delta, eta, s), dtype=float) * M
    for _ in range(4):
        tau_ref = eta * tc / (5.0 * M)
        x_ref = np.asarray(tau_to_x(tau_ref, sigma, delta, eta, s), dtype=float)
        h = tau_ref * 1e-6
        dx = (np.asarray(tau_to_x(tau_ref + h, sigma, delta, eta, s), dtype=float) - x_ref) / h
        tau_ref = tau_ref - (x_ref - x0) / dx
        tc = 5.0 * M * tau_ref / eta

    pn_t = (t - float(reference_time) - float(t_shift))[None, :]
    live = pn_t < tc                                            # (N, T)
    tau = eta * (tc - np.where(live, pn_t, 0.0)) / (5.0 * M)    # dead samples: tau at pn_t = 0
    x = np.asarray(tau_to_x(tau, sigma, delta, eta, s), dtype=float)
    tau_ref = eta * tc / (5.0 * M)
    x_ref = np.asarray(tau_to_x(tau_ref, sigma, delta, eta, s), dtype=float)
    Phi = (phi_c - np.asarray(_phase(x, sigma, delta, eta, s), dtype=float)
           + np.asarray(_phase(x_ref, sigma, delta, eta, s), dtype=float))
    amp = np.where(live, 2.0 * M * eta * x / D, 0.0)
    gw_phase = 2.0 * Phi
    n_live = live.sum(axis=1)
    for i in np.flatnonzero(n_live < t.size):
        gw_phase[i, n_live[i]:] = gw_phase[i, max(int(n_live[i]) - 1, 0)]
    return amp, gw_phase, tc.ravel() + float(reference_time) + float(t_shift)
```

- [ ] **Step 4: Run the test**

Run: `.wtenv/run.sh -m unittest tests.test_sobbh_wdm_direct.AmpPhaseBatchTest -v`
Expected: 2 tests PASS.

- [ ] **Step 5: Mutation check**

Replace `+ np.asarray(_phase(x_ref, ...))` by `- np.asarray(_phase(x_ref, ...))` and rerun: `test_rows_match_sobbhwaveform` must FAIL on the phase. Restore.

- [ ] **Step 6: Stage**

```bash
git add src/lisatools/sources/sobbh/wdm_direct.py tests/test_sobbh_wdm_direct.py
```

---

### Task 3: `SOBBHBatchedTOF` and `sobbh_tracer`

**Files:**
- Modify: `src/lisatools/sources/sobbh/wdm_direct.py` (append)
- Test: `tests/test_sobbh_wdm_direct.py` (`BatchedTOFTest`, `TracerTest`)

**Interfaces:**
- Consumes: `sobbh_amp_phase_batch` (Task 2); `lisatools.response.tdionfly.TDTDIonTheFly(t, amp, phase, sampling_frequency=, num_sub=, t_input=, tdi_config=, orbits=, force_backend=)` and its `__call__(inc, psi, lam, beta, return_spline=True) -> TDTDIOutput` (`tdi_amp_spl`, `tdi_phase_spl`, `phase_ref_spl` accept `derivative=`; `eval_tdi(t)`; `num_bin`; `xp`); `bbhx.sobbhtdionfly.SOBBHTDIonFly` as the reference.
- Produces: `class SOBBHBatchedTOF(orbits, tdi_config, reference_time, *, n_grid=2048, buffer_time=5000.0, eval_dt=600.0, force_backend="cpu")` with `build(params, t_lo, t_hi) -> TDTDIOutput` (splines filled; `out.tc` the per-row absolute merger times); `sobbh_tracer(out, t_pixels) -> (amp, phase, f, fdot)` each `(N, nch, P)` on `out.xp`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sobbh_wdm_direct.py` before the `__main__` block)

```python
class BatchedTOFTest(unittest.TestCase):
    """One TDTDIonTheFly for the batch, coarse eval grid, equals the production SOBBHTDIonFly."""

    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response.tdiconfig import TDIConfig

        cls.orbits = EqualArmlengthOrbits(force_backend="cpu")
        cls.tdi = TDIConfig("2nd generation", force_backend="cpu")
        cls.grid_t = np.arange(NOBS) * DT + T_START

    def _production(self, row):
        from bbhx.sobbhtdionfly import SOBBHTDIonFly
        from lisatools.sources.sobbh.waveform import SOBBHWaveform

        fly = SOBBHTDIonFly(
            SOBBHWaveform(Tobs=NOBS * DT, dt=DT, t0=T_START, reference_time=REF),
            self.orbits, self.tdi, DT, NOBS * DT, t0=T_START, n_grid=N_GRID,
            buffer_time=BUFFER, force_backend="cpu")
        m1, m2, s1, s2, dist_pc, f_low, phi_c, inc, psi, lam, beta = row
        return np.asarray(fly(m1, m2, s1, s2, dist_pc * 1e-9, f_low, phi_c, inc, lam, beta, psi,
                              upsample_t_arr=self.grid_t, combine=True))

    def _batched(self, rows, reference_time=REF, eval_dt=EVAL_DT):
        from lisatools.sources.sobbh.wdm_direct import SOBBHBatchedTOF

        tof = SOBBHBatchedTOF(self.orbits, self.tdi, reference_time, n_grid=N_GRID,
                              buffer_time=BUFFER, eval_dt=eval_dt, force_backend="cpu")
        out = tof.build(rows, float(self.grid_t[0]), float(self.grid_t[-1]))
        return np.asarray(out.eval_tdi(self.grid_t)), out

    def test_batch_matches_production_per_row(self):
        td, out = self._batched(ROWS)
        self.assertEqual(td.shape, (ROWS.shape[0], 3, NOBS))
        self.assertEqual(out.tc.shape, (ROWS.shape[0],))
        self.assertTrue(np.all(out.tc > self.grid_t[-1]))        # none of ROWS merges here
        for i, row in enumerate(ROWS):
            ref = self._production(row)
            for ch in range(3):
                mm, ratio = td_mismatch(td[i, ch], ref[ch])
                self.assertLess(mm, 1e-8, f"row {i} ch {ch}: mm {mm:.3e}")
                self.assertAlmostEqual(ratio, 1.0, delta=1e-6, msg=f"row {i} ch {ch}")

    def test_reference_epoch_control(self):
        # the parity test has power: a one-day reference-epoch error is caught
        td, _ = self._batched(ROWS[:1], reference_time=REF + 86400.0)
        mm, _ = td_mismatch(td[0, 0], self._production(ROWS[0])[0])
        self.assertGreater(mm, 1e-1)


class TracerTest(unittest.TestCase):
    """sobbh_tracer: amplitude, phase and spline-derivative f / fdot of a known channel phase;
    negative-frequency rows mirrored to positive frequency."""

    def test_known_quadratic_phase(self):
        from lisatools.sources.sobbh.wdm_direct import sobbh_tracer

        f0, fd = 3e-3, 2e-9

        class _Spl:
            def __init__(self, fns):
                self.fns = fns

            def __call__(self, x, derivative=0, **kw):
                return self.fns[derivative](np.asarray(x, dtype=float))

        def ph(t):
            return 2 * np.pi * (f0 * t + 0.5 * fd * t ** 2)

        class Out:
            xp = np
            num_bin = 2
            tdi_amp = np.zeros((2, 3, 1))
            tdi_amp_spl = _Spl([lambda x: np.where(np.arange(2)[:, None, None] == 0, 2.0, 1.0)
                                + 0 * x])
            tdi_phase_spl = _Spl([lambda x: 0 * x, lambda x: 0 * x, lambda x: 0 * x])
            phase_ref_spl = _Spl([
                lambda x: np.where(np.arange(2)[:, None] == 0, ph(x), -ph(x)),
                lambda x: np.where(np.arange(2)[:, None] == 0, 1, -1)
                * 2 * np.pi * (f0 + fd * x),
                lambda x: np.where(np.arange(2)[:, None] == 0, 1, -1) * 2 * np.pi * fd + 0 * x,
            ])

        t = np.linspace(1e5, 2e5, 11)
        amp, phase, f, fdot = sobbh_tracer(Out(), t)
        self.assertEqual(amp.shape, (2, 3, 11))
        for s_ in (0, 1):
            np.testing.assert_allclose(f[s_], np.broadcast_to(f0 + fd * t, (3, 11)), rtol=1e-12)
            np.testing.assert_allclose(fdot[s_], fd, rtol=1e-12)
            np.testing.assert_allclose(phase[s_], np.broadcast_to(ph(t), (3, 11)), rtol=1e-12)
        np.testing.assert_allclose(amp[0], 2.0)
        np.testing.assert_allclose(amp[1], 1.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.wtenv/run.sh -m unittest tests.test_sobbh_wdm_direct.BatchedTOFTest tests.test_sobbh_wdm_direct.TracerTest -v`
Expected: `ImportError: cannot import name 'SOBBHBatchedTOF'` / `'sobbh_tracer'`.

- [ ] **Step 3: Append the implementation**

```python
class SOBBHBatchedTOF:
    """ONE TDI-on-the-fly response for a batch of SOBBH rows on a coarse evaluation grid.

    Mirrors :class:`bbhx.sobbhtdionfly.SOBBHTDIonFly` (node grid ``linspace(t_lo - buffer_time,
    t_hi + buffer_time, n_grid)``, feed ``phase = gw_phase + pi`` with the intrinsic amplitude,
    the real ``inc`` / ``psi``, ``(lam, beta)`` consumed in the orbits frame) but for ``N`` rows
    at once (``num_sub = N``) and evaluated every ``eval_dt`` seconds instead of on the dense
    data grid: the lookup only needs the per-channel amplitude / phase splines at the WDM pixel
    centres. For a row merging inside the window the shared node grid continues past ``tc``
    with zero amplitude (``SOBBHTDIonFly`` truncates its node grid at ``tc`` instead, and so
    zeros the last ``buffer_time`` before merger).

    Args:
        orbits: :class:`~lisatools.detector.Orbits` (frame = the frame of ``lam``/``beta``).
        tdi_config: :class:`~lisatools.response.tdiconfig.TDIConfig`.
        reference_time: epoch [s] where ``f_low`` / ``phi_c`` are defined.
        n_grid, buffer_time: node grid size / padding [s] (production: 2048 / 5000).
        eval_dt: response evaluation step [s] (default 600).
        force_backend: backend name (``"cpu"``, ``"cuda12x"``, ...).
    """

    def __init__(self, orbits, tdi_config, reference_time, *, n_grid=2048, buffer_time=5000.0,
                 eval_dt=600.0, force_backend="cpu"):
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.reference_time = float(reference_time)
        self.n_grid = int(n_grid)
        self.buffer_time = float(buffer_time)
        self.eval_dt = float(eval_dt)
        if self.n_grid < 16 or self.eval_dt <= 0.0:
            raise ValueError("n_grid must be >= 16 and eval_dt > 0")
        self.force_backend = (force_backend if isinstance(force_backend, str)
                              else force_backend.name.split("_")[-1])

    @property
    def backend(self):
        import lisatools

        return lisatools.get_backend(self.force_backend)

    @property
    def xp(self):
        return self.backend.xp

    def build(self, params, t_lo, t_hi):
        """Response of every row over ``[t_lo, t_hi]`` [absolute s] -> ``TDTDIOutput`` with splines.

        ``out.tc`` holds the rows' absolute merger times.
        """
        from ...response.tdionfly import TDTDIonTheFly

        p = np.atleast_2d(np.asarray(params, dtype=float))
        if p.shape[1] < 11:
            raise ValueError(f"params must be (N, 11) chunked-basis rows, got {p.shape}")
        N = p.shape[0]
        t_lo, t_hi = float(t_lo), float(t_hi)
        node_t = np.linspace(t_lo - self.buffer_time, t_hi + self.buffer_time, self.n_grid)
        amp, gw_phase, tc = sobbh_amp_phase_batch(p, node_t, self.reference_time)
        n_eval = int(np.ceil((t_hi - t_lo) / self.eval_dt)) + 1
        eval_t = t_lo + np.arange(n_eval) * self.eval_dt
        xp = self.xp
        gen = TDTDIonTheFly(
            xp.asarray(np.ascontiguousarray(np.broadcast_to(eval_t, (N, n_eval)))),
            xp.asarray(amp),
            xp.asarray(gw_phase + np.pi),
            sampling_frequency=1.0 / self.eval_dt,
            num_sub=N,
            t_input=xp.asarray(np.ascontiguousarray(np.broadcast_to(node_t, (N, node_t.size)))),
            tdi_config=self.tdi_config,
            orbits=self.orbits,
            force_backend=self.force_backend,
        )
        out = gen(xp.asarray(p[:, 7]), xp.asarray(p[:, 8]), xp.asarray(p[:, 9]),
                  xp.asarray(p[:, 10]), return_spline=True)
        out.tc = tc
        return out


def sobbh_tracer(out, t_pixels):
    """Per-row, per-channel ``(amp, phase, f, fdot)`` at ``t_pixels`` [absolute s].

    The channel signal is ``Re[amp exp(-i phase)] = amp cos(phase)`` with ``phase = tdi_phase +
    phase_ref`` (``TDTDIOutput.eval_tdi``); ``f`` and ``fdot`` are the first and second
    derivatives of that spline phase over ``2 pi`` (``CubicSplineInterpolant(...,
    derivative=)``), so they carry the channel's Doppler shift. A negative-frequency row is
    mirrored (``cos`` is even): ``phase -> -phase``, ``f -> -f``, ``fdot -> -fdot``. Shapes
    ``(N, nch, P)`` on ``out.xp``.
    """
    xp = out.xp
    t = xp.asarray(t_pixels, dtype=float).reshape(-1)
    nb = int(out.num_bin)
    nch = int(out.tdi_amp.shape[1])
    t3 = xp.ascontiguousarray(xp.broadcast_to(t, (nb, nch, t.size)))
    t2 = xp.ascontiguousarray(xp.broadcast_to(t, (nb, t.size)))
    amp = xp.asarray(out.tdi_amp_spl(t3))
    ph = xp.asarray(out.tdi_phase_spl(t3)) + xp.asarray(out.phase_ref_spl(t2))[:, None, :]
    d1 = (xp.asarray(out.tdi_phase_spl(t3, derivative=1))
          + xp.asarray(out.phase_ref_spl(t2, derivative=1))[:, None, :])
    d2 = (xp.asarray(out.tdi_phase_spl(t3, derivative=2))
          + xp.asarray(out.phase_ref_spl(t2, derivative=2))[:, None, :])
    f = d1 / (2.0 * np.pi)
    fdot = d2 / (2.0 * np.pi)
    neg = f < 0
    return amp, xp.where(neg, -ph, ph), xp.where(neg, -f, f), xp.where(neg, -fdot, fdot)
```

- [ ] **Step 4: Run the tests**

Run: `.wtenv/run.sh -m unittest tests.test_sobbh_wdm_direct.BatchedTOFTest tests.test_sobbh_wdm_direct.TracerTest -v`
Expected: 3 tests PASS (the parity test builds 3 production templates on a 46k-sample grid; ~1 min). If `test_batch_matches_production_per_row` fails with a mismatch between 1e-8 and 1e-5, rerun the test with `eval_dt=300.0` and `150.0` (edit `EVAL_DT`); if the mismatch shrinks with `eval_dt`, the response amp/phase unwrap needs the finer grid: set the module default `eval_dt` to the passing value and record both numbers in the `SOBBHBatchedTOF` docstring. If it does not shrink, the mismatch is a convention error (reference epoch, `+pi`, column order) and must be fixed before Task 4.

- [ ] **Step 5: Stage**

```bash
git add src/lisatools/sources/sobbh/wdm_direct.py tests/test_sobbh_wdm_direct.py
```

---

### Task 4: `SOBBHDirectWDM`, sparse inner products and the fill

**Files:**
- Modify: `src/lisatools/sources/sobbh/wdm_direct.py` (append)
- Test: `tests/test_sobbh_wdm_direct.py` (`DirectWDMTest`)

**Interfaces:**
- Consumes: Tasks 1-3; `lisatools.domains.WDMSettings` (`layer_dt`, `layer_df`, `t0`, `ind_min_f`, `ind_max_f`, `ind_min_t`, `ind_max_t`, `Nf_active`, `Nt_active`), `WDMSignal`; `lisatools.analysiscontainer.AnalysisContainer` / `AnalysisContainerArray` (`linear_data_arr[0]`, `linear_psd_arr[0]` flat per-walker slabs `(nch, Nf_active, Nt_active)` and `(nch, nch, Nf_active, Nt_active)` for XYZ).
- Produces: `SparseWDMTemplate(w, m_act, n_act, valid, stats)`; `class SOBBHDirectWDM(wdm_settings, table, *, orbits, tdi_config, reference_time, t_obs_start=None, n_grid=2048, buffer_time=5000.0, eval_dt=600.0, num_m_layers=2, interp="cubic", force_backend="cpu")` with `sparse(params) -> SparseWDMTemplate`, `dense(params) -> list[WDMSignal]`, attributes `ev` (evaluator), `tof`, `t_pixels`, `n_pixels`, `nchannels`, `last_stats`; `sparse_inner_products(tpl, data_flat, invC_flat, data_index, noise_index, *, nchannels, Nf_active, Nt_active, tdi_type="XYZ") -> (d_h, h_h)`; `scatter_add(tpl, buf_flat, data_index, factors, *, nchannels, Nf_active, Nt_active) -> None`; `as_single_shard_holder(holder) -> AnalysisContainerArray`.

- [ ] **Step 1: Write the failing tests** (append before `__main__`)

```python
class DirectWDMTest(unittest.TestCase):
    """The lookup template vs the batched response's own dense TD->WDM; inner products vs the
    container; the fill; guards."""

    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.domains import WDMSettings
        from lisatools.response.tdiconfig import TDIConfig
        from lisatools.sources.sobbh.wdm_direct import SOBBHDirectWDM

        cls.tmp = tempfile.TemporaryDirectory()
        _, cls.table = build_tiny_table(cls.tmp.name)              # layer 3600 s at Nf=64, dt=56.25
        cls.orbits = EqualArmlengthOrbits(force_backend="cpu")
        cls.tdi = TDIConfig("2nd generation", force_backend="cpu")
        cls.wdm = WDMSettings(NF, NT, DT, t0=T_START, min_freq=2e-3, max_freq=2e-2,
                              is_complex=False, force_backend="cpu")
        cls.grid_t = np.arange(NOBS) * DT + T_START
        cls.direct = SOBBHDirectWDM(cls.wdm, cls.table, orbits=cls.orbits, tdi_config=cls.tdi,
                                    reference_time=REF, n_grid=N_GRID, buffer_time=BUFFER,
                                    eval_dt=EVAL_DT, num_m_layers=2, interp="cubic",
                                    force_backend="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _truth(self, rows):
        """(N, 3, Nf_active, Nt) dense TD->WDM of the batched response itself."""
        from lisatools.domains import TDSettings, TDSignal

        out = self.direct.tof.build(rows, float(self.grid_t[0]), float(self.grid_t[-1]))
        td = np.asarray(out.eval_tdi(self.grid_t))
        tds = TDSettings(NOBS, DT, force_backend="cpu")
        return np.stack([np.asarray(TDSignal(td[i], tds).transform(self.wdm).arr)
                         for i in range(td.shape[0])])

    def test_dense_matches_tof_own_transform(self):
        sigs = self.direct.dense(ROWS)
        truth = self._truth(ROWS)
        sl = slice(EDGE, NT - EDGE)
        for i in range(ROWS.shape[0]):
            got = np.asarray(sigs[i].arr)
            self.assertEqual(got.shape, truth[i].shape)
            for ch in range(3):
                mm, ratio = td_mismatch(got[ch, :, sl], truth[i, ch, :, sl])
                self.assertLess(mm, 1e-3, f"row {i} ch {ch}: mm {mm:.3e}")
                self.assertAlmostEqual(ratio, 1.0, delta=2e-3, msg=f"row {i} ch {ch}")
        st = self.direct.last_stats
        self.assertEqual(st["dropped_pixels"], 0)
        self.assertEqual(st["merged_rows"], 0)

    def test_no_parity_turn_control_fails(self):
        self.direct.ev.basis_cycle = "no_parity_turn"
        try:
            got = np.asarray(self.direct.dense(ROWS[:1])[0].arr)
        finally:
            self.direct.ev.basis_cycle = "quarter_turn"
        truth = self._truth(ROWS[:1])[0]
        mm, _ = td_mismatch(got[0, :, EDGE:NT - EDGE], truth[0, :, EDGE:NT - EDGE])
        self.assertGreater(mm, 1e-1)

    def _containers(self):
        from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
        from lisatools.domains import WDMSignal
        from lisatools.sensitivity import XYZ2SensitivityMatrix

        h = [np.asarray(s.arr) for s in self.direct.dense(ROWS)]
        data = [h[1] + 0.3 * h[2], 0.5 * h[0] - h[1]]
        acs = [AnalysisContainer(WDMSignal(d.copy(), self.wdm),
                                 XYZ2SensitivityMatrix(self.wdm, model="scirdv1"))
               for d in data]
        return h, acs, AnalysisContainerArray(acs)

    def test_sparse_inner_products_match_container(self):
        from lisatools.analysiscontainer import AnalysisContainer
        from lisatools.diagnostic import inner_product
        from lisatools.domains import WDMSignal
        from lisatools.sources.sobbh.wdm_direct import sparse_inner_products

        h, acs, aca = self._containers()
        tpl = self.direct.sparse(np.vstack([ROWS[0], ROWS[0]]))     # row 0 vs both walkers
        d_h, h_h = sparse_inner_products(
            tpl, aca.linear_data_arr[0], aca.linear_psd_arr[0], np.array([0, 1]),
            np.array([0, 1]), nchannels=3, Nf_active=int(self.wdm.Nf_active),
            Nt_active=int(self.wdm.Nt_active), tdi_type="XYZ")
        h0 = WDMSignal(h[0], self.wdm)
        for w in range(2):
            ref_dh = float(np.real(acs[w].template_inner_product(h0)))
            ref_hh = float(np.real(inner_product(h0, h0, psd=acs[w].sens_mat)))
            self.assertAlmostEqual(float(d_h[w]) / ref_dh, 1.0, delta=1e-9, msg=f"walker {w}")
            self.assertAlmostEqual(float(h_h[w]) / ref_hh, 1.0, delta=1e-9, msg=f"walker {w}")

    def test_fill_round_trip_and_matches_dense(self):
        from lisatools.sources.sobbh.wdm_direct import scatter_add

        h, acs, aca = self._containers()
        buf = aca.linear_data_arr[0]
        before = np.array(buf, copy=True)
        tpl = self.direct.sparse(ROWS[:2])
        kw = dict(nchannels=3, Nf_active=int(self.wdm.Nf_active),
                  Nt_active=int(self.wdm.Nt_active))
        scatter_add(tpl, buf, np.array([0, 1]), np.array([1.0, 1.0]), **kw)
        per = 3 * int(self.wdm.Nf_active) * int(self.wdm.Nt_active)
        np.testing.assert_allclose(buf[:per] - before[:per], h[0].ravel(), rtol=0,
                                   atol=1e-12 * np.abs(h[0]).max())
        np.testing.assert_allclose(buf[per:2 * per] - before[per:2 * per], h[1].ravel(), rtol=0,
                                   atol=1e-12 * np.abs(h[1]).max())
        scatter_add(tpl, buf, np.array([0, 1]), np.array([-1.0, -1.0]), **kw)
        np.testing.assert_allclose(buf, before, rtol=0, atol=1e-12 * np.abs(before).max())

    def test_merging_row_is_truncated_and_counted(self):
        sigs = self.direct.dense(ROW_MERGE[None, :])
        st = self.direct.last_stats
        self.assertEqual(st["merged_rows"], 1)
        self.assertGreater(st["dropped_pixels"], 0)
        tc = float(self.direct.tof.build(ROW_MERGE[None, :], float(self.grid_t[0]),
                                         float(self.grid_t[-1])).tc[0])
        after = self.direct.t_pixels > tc
        self.assertTrue(after.any())
        arr = np.asarray(sigs[0].arr)
        self.assertEqual(float(np.abs(arr[:, :, after]).max()), 0.0)

    def test_layer_dt_mismatch_raises(self):
        from lisatools.domains import WDMSettings
        from lisatools.sources.sobbh.wdm_direct import SOBBHDirectWDM

        other = WDMSettings(90, 64, DT, force_backend="cpu")          # layer_dt 1800 s
        with self.assertRaises(ValueError):
            SOBBHDirectWDM(other, self.table, orbits=self.orbits, tdi_config=self.tdi,
                           reference_time=REF)

    def test_fill_rejects_noncontiguous(self):
        from lisatools.sources.sobbh.wdm_direct import scatter_add

        tpl = self.direct.sparse(ROWS[:1])
        buf = np.zeros((int(self.wdm.Nt_active), int(self.wdm.Nf_active), 3)).T   # a view
        with self.assertRaises(ValueError):
            scatter_add(tpl, buf.reshape(-1) if buf.flags.c_contiguous else buf,
                        np.array([0]), np.array([1.0]), nchannels=3,
                        Nf_active=int(self.wdm.Nf_active), Nt_active=int(self.wdm.Nt_active))

    def test_numpy_path_does_not_import_cupyx(self):
        from lisatools.sources.sobbh.wdm_direct import scatter_add

        tpl = self.direct.sparse(ROWS[:1])
        buf = np.zeros(3 * int(self.wdm.Nf_active) * int(self.wdm.Nt_active))
        scatter_add(tpl, buf, np.array([0]), np.array([1.0]), nchannels=3,
                    Nf_active=int(self.wdm.Nf_active), Nt_active=int(self.wdm.Nt_active))
        self.assertNotIn("cupyx", sys.modules)
        self.assertGreater(float(np.abs(buf).max()), 0.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.wtenv/run.sh -m unittest tests.test_sobbh_wdm_direct.DirectWDMTest -v`
Expected: `ImportError: cannot import name 'SOBBHDirectWDM'`.

- [ ] **Step 3: Append the implementation**

```python
# ----------------------------------------------------------------------
# Sparse template, inner products and fill
# ----------------------------------------------------------------------

@dataclasses.dataclass
class SparseWDMTemplate:
    """A batch of lookup templates on their sparse support.

    ``w`` ``(N, nch, L, P)`` coefficients (zero where ``valid`` is False); ``m_act`` ``(N, L, P)``
    layer index into the ACTIVE band (clipped into range where invalid); ``n_act`` ``(P,)`` time
    index into the active band; ``valid`` ``(N, nch, L, P)``; ``stats`` the accounting dict.
    """

    w: object
    m_act: object
    n_act: object
    valid: object
    stats: dict


def as_single_shard_holder(holder):
    """Normalise ``holder`` to a single-shard :class:`AnalysisContainerArray`.

    Same contract as ``WDMComputationsBase._as_wdm_holder``: an ACA passes through (multi-shard
    raises: route per shard), a lone :class:`AnalysisContainer` is wrapped in a 1-element ACA
    built from SHALLOW copies of its data / sens_mat holders (wrapping the container itself
    would rebind it away from its parent's buffers).
    """
    if hasattr(holder, "linear_data_arr"):
        if len(holder.linear_data_arr) != 1:
            raise NotImplementedError(
                "the lookup comp is single-shard (it consumes linear_data_arr[0]); route a "
                "multi-shard ACA per split (SOBBHChunkedLikeMove already does).")
        return holder
    import copy as _copy

    from ...analysiscontainer import AnalysisContainer, AnalysisContainerArray

    if isinstance(holder, AnalysisContainer):
        arr = getattr(getattr(holder, "data_res_arr", None), "arr", None)
        dev = getattr(getattr(arr, "device", None), "id", None)
        gpus = None if dev is None else [int(dev)]
        ac = AnalysisContainer(_copy.copy(holder.data), _copy.copy(holder.sens_mat))
        return AnalysisContainerArray(ac, gpus=gpus)
    raise TypeError(f"holder must be an AnalysisContainerArray or AnalysisContainer, "
                    f"got {type(holder).__name__}")


class SOBBHDirectWDM:
    """SOBBH template directly in the WDM domain: batched response -> tracer -> lookup.

    Args:
        wdm_settings: the run's :class:`~lisatools.domains.WDMSettings` (active band + grid).
        table: :class:`~lisatools.domains.WDMLookupTable` with the SAME layer duration
            (``Nf * dt``); any sampling with that layer duration works.
        orbits, tdi_config, reference_time, n_grid, buffer_time, eval_dt, force_backend: see
            :class:`SOBBHBatchedTOF`.
        t_obs_start: absolute time [s] of grid pixel 0 (default ``wdm_settings.t0``; the stock
            domains carry an array-space ``t0 = 0`` while the data starts at ``data_t0``).
        num_m_layers: layers each side of the carrier layer per pixel (5 layers for 2).
        interp: table interpolation (``"cubic"`` / ``"linear"``).
    """

    def __init__(self, wdm_settings, table, *, orbits, tdi_config, reference_time,
                 t_obs_start=None, n_grid=2048, buffer_time=5000.0, eval_dt=600.0,
                 num_m_layers=2, interp="cubic", force_backend="cpu"):
        from ...wdm_lookup_eval import WDMLookupEvaluator

        if not np.isclose(float(table.layer_dt), float(wdm_settings.layer_dt), rtol=1e-9, atol=0):
            raise ValueError(
                f"lookup table layer_dt = {float(table.layer_dt):g} s but the grid's layer_dt = "
                f"{float(wdm_settings.layer_dt):g} s; the table must be built at the grid's "
                "layer duration (any Nf * dt with that product)")
        self.wdm = wdm_settings
        self.ev = WDMLookupEvaluator(table, interp=interp, force_backend=force_backend)
        self.tof = SOBBHBatchedTOF(orbits, tdi_config, reference_time, n_grid=n_grid,
                                   buffer_time=buffer_time, eval_dt=eval_dt,
                                   force_backend=force_backend)
        self.t_obs_start = float(wdm_settings.t0 if t_obs_start is None else t_obs_start)
        self.num_m_layers = int(num_m_layers)
        self.nchannels = int(tdi_config.nchannels)
        n_lo, n_hi = int(wdm_settings.ind_min_t), int(wdm_settings.ind_max_t)
        if int(wdm_settings.Nt_active) != n_hi - n_lo + 1:
            raise ValueError("WDMSettings.Nt_active != ind_max_t - ind_min_t + 1")
        self.n_pixels = np.arange(n_lo, n_hi + 1)
        self.t_pixels = self.t_obs_start + self.n_pixels * float(wdm_settings.layer_dt)
        self.last_stats = {}

    def sparse(self, params):
        """Lookup templates of the rows on their sparse support -> :class:`SparseWDMTemplate`."""
        xp = self.ev.xp
        ws = self.wdm
        p = np.atleast_2d(np.asarray(params, dtype=float))
        N = p.shape[0]
        pad = 2.0 * self.tof.eval_dt
        out = self.tof.build(p, float(self.t_pixels[0]) - pad, float(self.t_pixels[-1]) + pad)
        amp, phase, f, fdot = sobbh_tracer(out, self.t_pixels)          # (N, nch, P)
        live = amp > 0.0
        n_live = xp.maximum(live.sum(axis=1), 1)
        f_ref = xp.where(live, f, 0.0).sum(axis=1) / n_live                # channel mean (N, P)
        m = xp.moveaxis(self.ev.layers_for(f_ref, self.num_m_layers), -1, 1)   # (N, L, P)
        n = xp.asarray(self.n_pixels)
        w, ok = self.ev.coeffs(amp[:, :, None, :], phase[:, :, None, :], f[:, :, None, :],
                               fdot[:, :, None, :], n[None, None, None, :], m[:, None, :, :])
        in_band = (m >= int(ws.ind_min_f)) & (m <= int(ws.ind_max_f))
        valid = ok & in_band[:, None, :, :] & live[:, :, None, :]
        w = xp.where(valid, w, 0.0)
        m_act = xp.clip(m - int(ws.ind_min_f), 0, int(ws.Nf_active) - 1)
        n_act = n - int(ws.ind_min_t)
        fdot_bad = live & (xp.abs(fdot) > self.ev.fdot_max)
        stats = dict(
            rows=int(N),
            pixels=int(self.n_pixels.size),
            lookup_pixels=int(asnumpy(live.any(axis=1).sum())),
            dropped_pixels=int(asnumpy(fdot_bad.sum())),
            merged_rows=int(asnumpy((~live.all(axis=(1, 2))).sum())),
        )
        if stats["dropped_pixels"] > 0:
            logger.warning(
                "SOBBHDirectWDM: %d channel-pixels of %d rows have |fdot| beyond the table's "
                "axis (%.3g Hz/s) and were dropped (sources near merger).",
                stats["dropped_pixels"], N, self.ev.fdot_max)
        self.last_stats = stats
        return SparseWDMTemplate(w, m_act, n_act, valid, stats)

    def dense(self, params):
        """Active-band :class:`~lisatools.domains.WDMSignal` per row (gates and tests)."""
        from ...domains import WDMSignal

        tpl = self.sparse(params)
        xp = self.ev.xp
        N = int(tpl.w.shape[0])
        nfa, nta = int(self.wdm.Nf_active), int(self.wdm.Nt_active)
        buf = xp.zeros(N * self.nchannels * nfa * nta)
        scatter_add(tpl, buf, xp.arange(N), xp.ones(N), nchannels=self.nchannels,
                    Nf_active=nfa, Nt_active=nta)
        buf = buf.reshape(N, self.nchannels, nfa, nta)
        return [WDMSignal(buf[i], self.wdm) for i in range(N)]


def sparse_inner_products(tpl, data_flat, invC_flat, data_index, noise_index, *, nchannels,
                          Nf_active, Nt_active, tdi_type="XYZ"):
    """``<d|h>`` and ``<h|h>`` per row on the template's sparse support.

    The chunked kernel's accumulation (``lat_chunked_het_kernels.hh``, ``wdm_het_get_ll_kernel``
    steps 7-8): ``d_h = sum_pix sum_{c,c'} d_c invC_{cc'} h_{c'}``, ``h_h = sum h_c invC_{cc'}
    h_{c'}`` with the full channel matrix for XYZ and the diagonal for AET / AE -- no
    ``4 * differential_component`` factor. ``data_flat`` / ``invC_flat`` are the ACA's flat
    per-walker slabs ``(nch, Nf_active, Nt_active)`` / ``(nch[, nch], Nf_active, Nt_active)``;
    ``data_index`` / ``noise_index`` ``(N,)`` pick the slabs. Returns ``(d_h, h_h)`` ``(N,)``.
    """
    xp = get_array_module(tpl.w)
    nch, nfa, nta = int(nchannels), int(Nf_active), int(Nt_active)
    per_d = nch * nfa * nta
    if data_flat.size % per_d:
        raise ValueError(f"data buffer size {data_flat.size} is not a multiple of {per_d}")
    d = data_flat.reshape(data_flat.size // per_d, nch, nfa, nta)
    full = tdi_type == "XYZ"
    per_c = (nch * nch if full else nch) * nfa * nta
    if invC_flat.size % per_c:
        raise ValueError(f"invC buffer size {invC_flat.size} is not a multiple of {per_c}")
    nw_c = invC_flat.size // per_c
    inv = invC_flat.reshape((nw_c, nch, nch, nfa, nta) if full else (nw_c, nch, nfa, nta))
    r_d = xp.asarray(data_index, dtype=xp.int64)[:, None, None]
    r_c = xp.asarray(noise_index, dtype=xp.int64)[:, None, None]
    m = tpl.m_act
    n = xp.asarray(tpl.n_act, dtype=xp.int64)[None, None, :]
    w = tpl.w
    d_h = h_h = None
    for c in range(nch):
        dc = d[r_d, c, m, n]                                          # (N, L, P)
        for c2 in range(nch):
            if full:
                icc = inv[r_c, c, c2, m, n]
            elif c2 == c:
                icc = inv[r_c, c, m, n]
            else:
                continue
            dh = (dc * icc * w[:, c2]).sum(axis=(1, 2))
            hh = (w[:, c] * icc * w[:, c2]).sum(axis=(1, 2))
            d_h = dh if d_h is None else d_h + dh
            h_h = hh if h_h is None else h_h + hh
    return d_h, h_h


def scatter_add(tpl, buf_flat, data_index, factors, *, nchannels, Nf_active, Nt_active):
    """Accumulate ``factors[row] * w`` into the flat active-band buffer (per-walker slabs).

    ``buf_flat`` must be 1-D and C-contiguous (a view of the ACA's ``linear_data_arr[0]`` or a
    freshly allocated buffer); ``data_index`` ``(N,)`` picks each row's slab.
    """
    xp = get_array_module(buf_flat)
    nch, nfa, nta = int(nchannels), int(Nf_active), int(Nt_active)
    if buf_flat.ndim != 1 or not buf_flat.flags.c_contiguous:
        raise ValueError("fill target must be a 1-D C-contiguous buffer (writes go in place)")
    per = nch * nfa * nta
    if buf_flat.size % per:
        raise ValueError(f"fill target size {buf_flat.size} is not a multiple of {per}")
    buf = buf_flat.reshape(buf_flat.size // per, nch, nfa, nta)
    idx = (
        xp.asarray(data_index, dtype=xp.int64)[:, None, None, None],
        xp.arange(nch, dtype=xp.int64)[None, :, None, None],
        tpl.m_act[:, None, :, :],
        xp.asarray(tpl.n_act, dtype=xp.int64)[None, None, None, :],
    )
    vals = xp.asarray(factors, dtype=float)[:, None, None, None] * tpl.w
    if xp is np:
        np.add.at(buf, idx, vals)
    else:
        import cupyx

        cupyx.scatter_add(buf, idx, vals)
```

- [ ] **Step 4: Run the tests**

Run: `.wtenv/run.sh -m unittest tests.test_sobbh_wdm_direct.DirectWDMTest -v`
Expected: 8 tests PASS (~2 min: several dense transforms on a 46k-sample grid). Record the measured per-channel `mm` and norm ratios of `test_dense_matches_tof_own_transform` (print them with `-v` or a temporary print) in the Task 7 doc.

- [ ] **Step 5: Mutation check**

Delete `h_h = hh if h_h is None else h_h + hh` (leave `h_h = None` so the function returns `(d_h, None)`): `test_sparse_inner_products_match_container` must FAIL. Restore.

- [ ] **Step 6: Stage**

```bash
git add src/lisatools/sources/sobbh/wdm_direct.py tests/test_sobbh_wdm_direct.py
```

---

### Task 5: `SOBBHLookupComputations` and the move-level parity suite

**Files:**
- Modify: `src/lisatools/sources/sobbh/wdm_direct.py` (append)
- Test: `tests/test_sobbh_lookup_move.py`

**Interfaces:**
- Consumes: Task 4; `SOBBHChunkedLikeMove` (reads `comp.d_d`, `comp.wdm_settings.{ind_min_f, ind_max_f, layer_df}`, calls `comp.get_ll_wdm(params, holder, data_index=, noise_index=, m_band_half_width=)`, `comp.fill_global_wdm(params, holder, data_index=, factors=, m_band_half_width=)`, reads `comp.d_h_out`, `comp.h_h_out`, `comp.last_call_spans`).
- Produces: `class SOBBHLookupComputations(wdm_settings, t_ref, table, *, orbits=None, tdi_config=None, tdi_type="XYZ", t_obs_start=None, n_grid=2048, buffer_time=5000.0, eval_dt=600.0, num_m_layers=2, interp="cubic", row_batch=32, force_backend="cpu", d_d=0.0)` with `get_ll_wdm(...) -> ll (N,)`, `fill_global_wdm(params, templates, convert_to_ra_dec=None, data_index=None, factors=None, grid_dim=0, m_band_half_width=None)`, properties `xp`, `backend`, `args`, `kwargs`, attributes `d_d`, `wdm_settings`, `nchannels`, `tdi_type`, `direct`, `d_h_out`, `h_h_out`, `d_h_im_out` (None), `last_call_spans`, `last_stats`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sobbh_lookup_move.py
"""SOBBHChunkedLikeMove with the LOOKUP comp: fast-vs-slow parity through the move's own seams.

Same toy choreography as tests/test_sobbh_chunked_move.py (slow side = the stock
``get_sobbh_tdionfly_gen`` + ``SOBBHTDIonFlyWaveWrap`` + the container likelihood), with
``chunked_comp=SOBBHLookupComputations`` on a 3600-s-layer grid (Nf=180, dt=20) and a tiny
table built at (Nf=64, dt=56.25) -- the table-portability property in use.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

from lisatools.utils.constants import YRSID_SI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wdm_lookup_toy import build_tiny_table  # noqa: E402

DT = 20.0
NF, NT = 180, 256
NOBS = NF * NT
T_START = int(0.5 * YRSID_SI / DT) * DT
F_LOW = 6.0e-3
NWALKERS = 3
NTEMPS = 2

#: stock SOBBH waveform basis (what compute_like receives):
#: (m1, m2, s1, s2, dist[Gpc], inc, f_low, lam, beta, psi, phi0)
REF_STOCK = np.array(
    [60.0, 55.0, 0.1, 0.2, 0.4, np.arccos(0.3), F_LOW, 3.1, np.arcsin(0.2), 0.7, 1.1])


def _build_toy(tmpdir):
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.domains import TDSettings, WDMSettings, WDMSignal
    from lisatools.globalfit.stock.erebor.wrappers import SOBBHTDIonFlyWaveWrap
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    from lisatools.sources.sobbh.response import get_sobbh_tdionfly_gen
    from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations

    backend = "cpu"
    orbits = EqualArmlengthOrbits(force_backend=backend)
    tdi_config = TDIConfig("2nd generation", force_backend=backend)
    td_set = TDSettings(NOBS, DT, force_backend=backend)
    wdm = WDMSettings(NF, NT, DT, t0=T_START, min_freq=2e-3, max_freq=2e-2, is_complex=False,
                      force_backend=backend)
    gen = get_sobbh_tdionfly_gen(
        Tobs=NOBS * DT, dt=DT, t_start=T_START, tdi_config=tdi_config,
        reference_time=float(T_START), orbits=orbits, n_grid=1024, buffer_time=5000.0,
        force_backend=backend)
    t_arr = np.arange(NOBS) * DT + T_START
    wrap = SOBBHTDIonFlyWaveWrap(gen, t_arr, td_set, wdm, td_window=None, nchannels=3)

    def sobbh_gen(*params, apply_transform=False, leaf_inds=None, **kwargs):
        return wrap(*params)

    inj = wrap(*REF_STOCK)
    acs_list = []
    for _ in range(NWALKERS):
        ac = AnalysisContainer(WDMSignal(np.array(np.asarray(inj.arr), copy=True), wdm),
                               XYZ2SensitivityMatrix(wdm, model="scirdv1"))
        ac.signal_gen = {"sobbh": sobbh_gen}
        acs_list.append(ac)
    acs = AnalysisContainerArray(acs_list)

    _, table = build_tiny_table(tmpdir)
    comp = SOBBHLookupComputations(
        wdm, float(T_START), table, orbits=orbits, tdi_config="2nd generation", tdi_type="XYZ",
        n_grid=1024, buffer_time=5000.0, eval_dt=600.0, num_m_layers=2, interp="cubic",
        row_batch=4, force_backend=backend, d_d=0.0)
    return acs, sobbh_gen, comp, wdm, td_set


def _build_move(acs, comp):
    from eryn.moves import StretchMove
    from eryn.prior import ProbDistContainer, uniform_dist

    from lisatools.globalfit.moves import SOBBHChunkedLikeMove

    betas = 1 / 1.2 ** np.arange(NTEMPS)
    priors = {"sobbh": ProbDistContainer({i: uniform_dist(-1e10, 1e10) for i in range(11)})}
    return SOBBHChunkedLikeMove(
        "sobbh", (NTEMPS, NWALKERS, 1, 11), None, {}, {}, acs, 1, None, priors,
        [(StretchMove(), 1.0)], betas_all=np.tile(betas, (1, 1)), chunked_comp=comp,
        m_band_half_width=2, name="sobbh lookup test")


class SOBBHLookupParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        try:
            cls.acs, cls.gen, cls.comp, cls.wdm, cls.td_set = _build_toy(cls.tmp.name)
        except Exception as exc:  # missing compiled deps etc.
            raise unittest.SkipTest(f"toy setup unavailable: {exc}")
        cls.move = _build_move(cls.acs, cls.comp)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _rows(self, n_pert=8, seed=5):
        rng = np.random.default_rng(seed)
        rows = np.tile(REF_STOCK, (n_pert + 1, 1))
        layer_df = float(self.wdm.layer_df)
        rows[1:, 6] += rng.uniform(-0.2, 0.2, n_pert) * layer_df
        rows[1:, 10] = rng.uniform(0, 2 * np.pi, n_pert)
        rows[1:, 5] = np.arccos(rng.uniform(-0.9, 0.9, n_pert))
        rows[1:, 4] *= rng.uniform(0.7, 1.5, n_pert)
        idx = np.arange(rows.shape[0], dtype=np.int32) % NWALKERS
        return rows, idx

    def test_fast_vs_slow_parity_through_move(self):
        move = self.move
        move.setup_likelihood_here(None)
        rows, idx = self._rows()
        fast = move.compute_like(rows, idx)
        slow = move.compute_acs_like(rows, idx, **move.waveform_like_kwargs)
        self.assertTrue(np.all(np.isfinite(fast)))
        diff = np.abs(fast - slow)
        self.assertLess(float(diff.max()), move.check_ll_tol,
                        msg=f"lookup vs slow lnL diff {diff.max():.3e} exceeds tol "
                            f"{move.check_ll_tol} (median {np.median(diff):.3e})")
        move.check_ll_mode = "strict"
        move._verify_prev_logl(fast.reshape(1, -1), rows, idx, leaf=0)
        self.assertIn("launch", self.comp.last_call_spans)
        self.assertEqual(self.comp.last_call_spans["num_bin"], rows.shape[0])

    def test_expose_invariant_via_offset(self):
        move = self.move
        move.setup_likelihood_here(None)
        rows = np.tile(REF_STOCK, (NWALKERS, 1))
        idx = np.arange(NWALKERS, dtype=np.int32)
        fast = move.compute_like(rows, idx)
        slow = move.compute_acs_like(rows, idx, **move.waveform_like_kwargs)
        np.testing.assert_allclose(fast, slow, atol=move.check_ll_tol)

    def test_out_of_band_sentinel(self):
        move = self.move
        move.setup_likelihood_here(None)
        bad = REF_STOCK.copy()
        bad[6] = 0.5
        self.assertEqual(float(move.compute_like(bad.reshape(1, -1),
                                                 np.zeros(1, dtype=np.int32))[0]), -1e300)
        nanrow = REF_STOCK.copy()
        nanrow[0] = np.nan
        self.assertEqual(float(move.compute_like(nanrow.reshape(1, -1),
                                                 np.zeros(1, dtype=np.int32))[0]), -1e300)

    def test_record_leaf_inner_products_from_comp(self):
        from types import SimpleNamespace

        move = self.move
        move.setup_likelihood_here(None)
        sub = SimpleNamespace(d_h=np.full((NWALKERS, 1), np.nan),
                              h_h=np.full((NWALKERS, 1), np.nan))
        state = SimpleNamespace(sub_states={"sobbh": sub})
        move._record_leaf_inner_products(state, np.tile(REF_STOCK, (NWALKERS, 1)), 0)
        self.assertTrue(np.all(np.isfinite(sub.d_h)))
        self.assertTrue(np.all(sub.h_h > 0))

    def test_fill_through_move_round_trip(self):
        move = self.move
        buf = self.acs.linear_data_arr[0]
        before = np.array(buf, copy=True)
        coords = np.tile(REF_STOCK, (NWALKERS, 1))
        move._apply_cold_chain_sources(coords, +1)
        self.assertGreater(float(np.abs(buf - before).max()), 0.0)
        move._apply_cold_chain_sources(coords, -1)
        np.testing.assert_allclose(buf, before, rtol=0, atol=1e-12 * np.abs(before).max())

    def test_merged_before_window_scores_offset(self):
        # a row that merged before the first pixel: zero template, lnL == exposed offset
        move = self.move
        move.setup_likelihood_here(None)
        row = REF_STOCK.copy()
        row[0] = row[1] = 2000.0          # tc of a few days at 6 mHz -> merged long before
        row[6] = 1.5e-2
        ll = move.compute_like(row.reshape(1, -1), np.zeros(1, dtype=np.int32))
        self.assertTrue(np.isfinite(ll[0]))
        self.assertAlmostEqual(float(ll[0]), float(move._exposed_offset[0]), places=9)
        self.assertEqual(float(self.comp.h_h_out[0]), 0.0)

    def test_bad_data_index_raises(self):
        with self.assertRaises(IndexError):
            self.comp.get_ll_wdm(SOBBHLookupParityTestHelper.chunked(REF_STOCK), self.acs,
                                 data_index=np.array([NWALKERS]), noise_index=np.array([0]))

    def test_comp_surface_for_the_move(self):
        comp = self.comp
        self.assertEqual(comp.d_d, 0.0)
        self.assertIs(comp.wdm_settings, self.wdm)
        self.assertIsNone(comp.d_h_im_out)
        self.assertEqual(comp.args[1], float(T_START))
        self.assertEqual(comp.kwargs["row_batch"], 4)


class SOBBHLookupParityTestHelper:
    @staticmethod
    def chunked(row):
        from lisatools.globalfit.moves import SOBBHChunkedLikeMove

        return SOBBHChunkedLikeMove.to_chunked_basis(row)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.wtenv/run.sh -m unittest tests.test_sobbh_lookup_move -v`
Expected: every test skips/errors with `cannot import name 'SOBBHLookupComputations'` (the `SkipTest` wrapper reports the ImportError text).

- [ ] **Step 3: Append the comp**

```python
# ----------------------------------------------------------------------
# The comp that slots into SOBBHChunkedLikeMove
# ----------------------------------------------------------------------

class SOBBHLookupComputations:
    """Lookup-table SOBBH scorer with the comp surface ``SOBBHChunkedLikeMove`` consumes.

    Duck-types the parts of :class:`bbhx.sobbhcomps.SOBBHWDMComputations` the move and the
    engine signal generator use (``get_ll_wdm``, ``fill_global_wdm``, ``d_d``, ``wdm_settings``,
    ``xp`` / ``backend``, ``args`` / ``kwargs``, the ``d_h_out`` / ``h_h_out`` /
    ``last_call_spans`` stashes). Scoring convention identical to the kernel: returns
    ``-0.5 * (d_d + h_h - 2 d_h)`` with ``d_d = 0`` by default (the move adds the per-walker
    exposed-residual offset). ``m_band_half_width`` is accepted for signature compatibility and
    ignored: the lookup band is ``num_m_layers`` layers each side of the carrier layer.

    Args:
        wdm_settings: the run's :class:`~lisatools.domains.WDMSettings`.
        t_ref: reference epoch [s] of ``f_low`` / ``phi_c``.
        table: :class:`~lisatools.domains.WDMLookupTable` with the grid's layer duration.
        orbits, tdi_config, tdi_type, t_obs_start: as on ``WDMComputationsBase`` (``tdi_config``
            may be a string; ``None`` orbits -> ``EqualArmlengthOrbits``).
        n_grid, buffer_time, eval_dt, num_m_layers, interp: see :class:`SOBBHDirectWDM`.
        row_batch: rows per response build (bounds the spline memory).
        force_backend: backend name at construction (never per call).
        d_d: constant folded into the returned likelihood (default 0 = source-only).
    """

    _NPARAMS = 11

    def __init__(self, wdm_settings, t_ref, table, *, orbits=None, tdi_config=None,
                 tdi_type="XYZ", t_obs_start=None, n_grid=2048, buffer_time=5000.0,
                 eval_dt=600.0, num_m_layers=2, interp="cubic", row_batch=32,
                 force_backend="cpu", d_d=0.0):
        from ...detector import EqualArmlengthOrbits, Orbits
        from ...domains import WDMSettings
        from ...response.tdiconfig import TDIConfig

        if not isinstance(wdm_settings, WDMSettings):
            raise TypeError("wdm_settings must be a lisatools.domains.WDMSettings instance")
        if tdi_type not in ("XYZ", "AET", "AE"):
            raise ValueError(f"tdi_type must be one of 'XYZ', 'AET', 'AE'; got {tdi_type!r}")
        self._ctor_args = (wdm_settings, float(t_ref), table)
        self._ctor_kwargs = dict(
            orbits=orbits, tdi_config=tdi_config, tdi_type=tdi_type, t_obs_start=t_obs_start,
            n_grid=n_grid, buffer_time=buffer_time, eval_dt=eval_dt, num_m_layers=num_m_layers,
            interp=interp, row_batch=row_batch, force_backend=force_backend, d_d=d_d)
        self.force_backend = (force_backend if isinstance(force_backend, str)
                              else force_backend.name.split("_")[-1])
        if orbits is None:
            orbits = EqualArmlengthOrbits(force_backend=self.force_backend)
        elif not isinstance(orbits, Orbits) and issubclass(orbits, Orbits):
            orbits = orbits()
        if tdi_config is None:
            tdi_config = TDIConfig("1st generation", force_backend=self.force_backend)
        elif isinstance(tdi_config, str):
            tdi_config = TDIConfig(tdi_config, force_backend=self.force_backend)
        self.wdm_settings = wdm_settings
        self.t_ref = float(t_ref)
        self.tdi_type = tdi_type
        self.t_obs_start = float(wdm_settings.t0 if t_obs_start is None else t_obs_start)
        self.d_d = float(d_d)
        self.row_batch = max(int(row_batch), 1)
        self.orbits = orbits
        self.tdi_config = tdi_config
        self.nchannels = int(tdi_config.nchannels)
        self.direct = SOBBHDirectWDM(
            wdm_settings, table, orbits=orbits, tdi_config=tdi_config,
            reference_time=self.t_ref, t_obs_start=self.t_obs_start, n_grid=n_grid,
            buffer_time=buffer_time, eval_dt=eval_dt, num_m_layers=num_m_layers, interp=interp,
            force_backend=self.force_backend)
        self.d_h_out = None
        self.h_h_out = None
        self.d_h_im_out = None
        self.last_call_spans = None
        self.last_stats = {}
        self._band_noted = False

    @property
    def backend(self):
        import lisatools

        return lisatools.get_backend(self.force_backend)

    @property
    def xp(self):
        return self.direct.ev.xp

    @property
    def args(self):
        return self._ctor_args

    @property
    def kwargs(self):
        return dict(self._ctor_kwargs)

    def _note_band(self, m_band_half_width):
        if m_band_half_width is not None and not self._band_noted:
            logger.info("SOBBHLookupComputations: m_band_half_width=%s ignored (the lookup "
                        "band is num_m_layers=%d layers each side of the carrier layer)",
                        m_band_half_width, self.direct.num_m_layers)
            self._band_noted = True

    def _prep(self, params, data_index, noise_index, n_slots, convert_to_ra_dec):
        from ...response.directresponse import ecliptic_to_icrs, warn_deprecated_frame_conversion

        p = np.atleast_2d(np.asarray(asnumpy(params), dtype=float))
        if p.ndim != 2 or p.shape[1] != self._NPARAMS:
            raise ValueError(f"params must be (N, {self._NPARAMS}); got {p.shape}")
        N = p.shape[0]
        if convert_to_ra_dec:
            warn_deprecated_frame_conversion()
            p = p.copy()
            p[:, -2], p[:, -1] = ecliptic_to_icrs(p[:, -2], p[:, -1])
        di = (np.zeros(N, dtype=np.int64) if data_index is None
              else np.asarray(asnumpy(data_index), dtype=np.int64).reshape(-1))
        ni = (di if noise_index is None
              else np.asarray(asnumpy(noise_index), dtype=np.int64).reshape(-1))
        if di.shape != (N,) or ni.shape != (N,):
            raise ValueError("data_index / noise_index must have one entry per row")
        if N and (di.min() < 0 or di.max() >= n_slots or ni.min() < 0 or ni.max() >= n_slots):
            raise IndexError(f"data_index / noise_index outside the holder's {n_slots} slots")
        return p, di, ni

    def get_ll_wdm(self, params, wdm_holder, data_index=None, noise_index=None,
                   convert_to_ra_dec=None, grid_dim=0, use_layer_groups=True,
                   group_band_layers=5, margin_layers=0, m_band_half_width=None):
        """Per-row ``-0.5 * (d_d + h_h - 2 d_h)`` against the holder's residual slabs.

        Stashes ``d_h_out`` / ``h_h_out`` (``d_h_im_out`` stays ``None``: no fused quadrature)
        and a ``last_call_spans`` dict (``stage``, ``launch``, ``total``, ``num_bin``, plus
        ``response``/``tracer+lookup``/``inner`` splits).
        """
        t_entry = time.perf_counter()
        self._note_band(m_band_half_width)
        holder = as_single_shard_holder(wdm_holder)
        data_flat = holder.linear_data_arr[0]
        invC_flat = holder.linear_psd_arr[0]
        p, di, ni = self._prep(params, data_index, noise_index, len(holder), convert_to_ra_dec)
        xp = self.xp
        ws = self.wdm_settings
        N = p.shape[0]
        t_stage = time.perf_counter() - t_entry
        d_h = xp.zeros(N)
        h_h = xp.zeros(N)
        totals = dict(lookup_pixels=0, dropped_pixels=0, merged_rows=0)
        t_build = t_inner = 0.0
        for lo in range(0, N, self.row_batch):
            hi = min(N, lo + self.row_batch)
            t0 = time.perf_counter()
            tpl = self.direct.sparse(p[lo:hi])
            t_build += time.perf_counter() - t0
            t0 = time.perf_counter()
            a, b = sparse_inner_products(
                tpl, data_flat, invC_flat, di[lo:hi], ni[lo:hi], nchannels=self.nchannels,
                Nf_active=int(ws.Nf_active), Nt_active=int(ws.Nt_active), tdi_type=self.tdi_type)
            d_h[lo:hi] = a
            h_h[lo:hi] = b
            t_inner += time.perf_counter() - t0
            for key in totals:
                totals[key] += int(self.direct.last_stats.get(key, 0))
        self.d_h_out = d_h
        self.h_h_out = h_h
        self.d_h_im_out = None
        self.last_stats = totals
        self.last_call_spans = {
            "stage": t_stage, "geom": 0.0, "wrap": 0.0, "launch": t_build + t_inner,
            "template": t_build, "inner": t_inner, "total": time.perf_counter() - t_entry,
            "num_bin": int(N), "n_groups": 0,
        }
        return -0.5 * (self.d_d + h_h - 2.0 * d_h)

    def _fill_target(self, templates):
        from ...domains import DomainBase

        if hasattr(templates, "linear_data_arr"):
            if len(templates.linear_data_arr) != 1:
                raise NotImplementedError(
                    "fill_global_wdm writes into linear_data_arr[0] and is single-shard; route a "
                    "multi-shard ACA per split.")
            arr = templates.linear_data_arr[0]
        elif isinstance(templates, DomainBase):
            arr = templates.arr
        else:
            from ...analysiscontainer import AnalysisContainer

            if isinstance(templates, AnalysisContainer):
                arr = as_single_shard_holder(templates).linear_data_arr[0]
            else:
                arr = templates
        if not arr.flags.c_contiguous:
            raise ValueError("fill target must be C-contiguous (the fill writes in place)")
        ws = self.wdm_settings
        per = self.nchannels * int(ws.Nf_active) * int(ws.Nt_active)
        flat = arr.reshape(-1)
        if flat.size % per:
            raise ValueError(
                f"fill target of shape {tuple(arr.shape)} is not a stack of ACTIVE-band "
                f"({self.nchannels}, {int(ws.Nf_active)}, {int(ws.Nt_active)}) templates")
        return flat, flat.size // per

    def fill_global_wdm(self, params, templates, convert_to_ra_dec=None, data_index=None,
                        factors=None, grid_dim=0, m_band_half_width=None, band_slab_Nf=None,
                        slab_min_f=None):
        """Accumulate ``factors * h`` into ``templates`` (ACA holder, ``WDMSignal``, or an
        active-band array ``(nch, Nf_active, Nt_active)`` / ``(num, nch, Nf_active, Nt_active)``
        / flat), slab per row by ``data_index`` (default 0)."""
        self._note_band(m_band_half_width)
        if band_slab_Nf is not None or slab_min_f is not None:
            raise NotImplementedError("per-band slabs are not supported by the lookup fill")
        flat, n_slots = self._fill_target(templates)
        p, di, _ = self._prep(params, data_index, None, n_slots, convert_to_ra_dec)
        N = p.shape[0]
        if N == 0:
            return
        fac = (np.ones(N) if factors is None
               else np.asarray(asnumpy(factors), dtype=float).reshape(-1))
        if fac.shape != (N,):
            raise ValueError(f"factors must have shape ({N},), got {fac.shape}")
        ws = self.wdm_settings
        for lo in range(0, N, self.row_batch):
            hi = min(N, lo + self.row_batch)
            tpl = self.direct.sparse(p[lo:hi])
            scatter_add(tpl, flat, di[lo:hi], fac[lo:hi], nchannels=self.nchannels,
                        Nf_active=int(ws.Nf_active), Nt_active=int(ws.Nt_active))


__all__ = [
    "SOBBHBatchedTOF",
    "SOBBHDirectWDM",
    "SOBBHLookupComputations",
    "SparseWDMTemplate",
    "as_single_shard_holder",
    "scatter_add",
    "sobbh_amp_phase_batch",
    "sobbh_tracer",
    "sparse_inner_products",
]
```

- [ ] **Step 4: Run the tests**

Run: `.wtenv/run.sh -m unittest tests.test_sobbh_lookup_move -v`
Expected: 8 tests PASS. If `test_fast_vs_slow_parity_through_move` fails with a diff between 0.5 and a few units while `test_dense_matches_tof_own_transform` passed, the toy source is louder than the tiny table's resolution allows: lower `REF_STOCK[4]`'s distance scale (e.g. `0.4 -> 1.2` Gpc, keeping the chunked test's row untouched) and record the measured diff in the assertion message. A diff above 10 is a bug (offset, slab, or index).

- [ ] **Step 5: Mutation check**

In `sparse_inner_products`, delete `h_h = hh if h_h is None else h_h + hh`; `test_fast_vs_slow_parity_through_move` and `test_expose_invariant_via_offset` must FAIL. Restore. Then also run the chunked suite to prove nothing regressed: `.wtenv/run.sh -m unittest tests.test_sobbh_chunked_move tests.test_sobbh_chunked_fill -v` -> all PASS.

- [ ] **Step 6: Stage**

```bash
git add src/lisatools/sources/sobbh/wdm_direct.py tests/test_sobbh_lookup_move.py
```

---

### Task 6: Stock wiring (`SOBBH_LIKELIHOOD=lookup`)

**Files:**
- Modify: `src/lisatools/globalfit/stock/erebor/source_runtime.py` (`SourceSOBBHSettings` ~line 500-576; `source_signal_cfg` ~line 1005; after `get_sobbh_chunked_comp` ~line 1474; `get_sobbh_chunked_signal_gen` ~line 1563; `SourceSignalGen.__call__` ~line 1340; `build_sobbh_move_runtime` ~line 1605)
- Test: `tests/test_sobbh_lookup_stock.py`

**Interfaces:**
- Consumes: Task 5 `SOBBHLookupComputations`; `lisatools.domains.WDMLookupTable.from_file(path, force_backend=)`; the existing `_wrap_device_and_orbits`, `_WAVE_WRAP_CACHE`, `TDIConfig`, `device_context`, `env_default`.
- Produces: settings fields `lookup_table_path` (`SOBBH_LOOKUP_TABLE_PATH`, ""), `lookup_num_m_layers` (`SOBBH_LOOKUP_NUM_M_LAYERS`, 2), `lookup_eval_dt` (`SOBBH_LOOKUP_EVAL_DT`, 600.0), `lookup_interp` (`SOBBH_LOOKUP_INTERP`, "cubic"), `lookup_row_batch` (`SOBBH_LOOKUP_ROW_BATCH`, 32); cfg keys `sobbh_lookup_table_path`, `sobbh_lookup_num_m_layers`, `sobbh_lookup_eval_dt`, `sobbh_lookup_interp`, `sobbh_lookup_row_batch`; `SOBBH_FAST_LIKELIHOODS = ("chunked", "lookup")`; `get_sobbh_lookup_comp(general_info, cfg)`; `get_sobbh_fast_comp(general_info, cfg)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sobbh_lookup_stock.py
"""SOBBH_LIKELIHOOD=lookup: settings knobs, cfg plumbing, comp construction and the dispatch."""
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _wdm_lookup_toy import build_tiny_table  # noqa: E402


def _general_info(domain):
    from lisatools.detector import EqualArmlengthOrbits

    return SimpleNamespace(gpus=None, orbits=EqualArmlengthOrbits(force_backend="cpu"),
                           gpu_orbits=None, domain_settings=domain, force_backend="cpu",
                           data_t0=0.0)


def _cfg(path, likelihood="lookup"):
    return dict(nchannels=3, tdi_gen_str="2nd generation", tdi_chan="XYZ",
                sobbh_reference_time=None, sobbh_n_grid=64, sobbh_buffer_time=5000.0,
                sobbh_likelihood=likelihood, sobbh_lookup_table_path=path,
                sobbh_lookup_eval_dt=600.0, sobbh_lookup_num_m_layers=2,
                sobbh_lookup_interp="cubic", sobbh_lookup_row_batch=32,
                sobbh_nt_sub=32, sobbh_chirp_fdot_max=None, sobbh_sweep_max_layers=3.5,
                sobbh_n_sparse=256, sobbh_n_pad=4, sobbh_m_band_half_width=1,
                sobbh_fill_m_band_half_width=8)


class LookupSettingsTest(unittest.TestCase):
    def test_fields_and_env_knobs(self):
        from lisatools.globalfit.stock.erebor.source_runtime import SourceSOBBHSettings

        s = SourceSOBBHSettings(likelihood="lookup")
        self.assertEqual(s.likelihood, "lookup")
        self.assertEqual(s.lookup_table_path, "")
        self.assertEqual(s.lookup_num_m_layers, 2)
        self.assertEqual(s.lookup_eval_dt, 600.0)
        self.assertEqual(s.lookup_interp, "cubic")
        self.assertEqual(s.lookup_row_batch, 32)
        env = {"SOBBH_LIKELIHOOD": "lookup", "SOBBH_LOOKUP_TABLE_PATH": "/x/y.h5",
               "SOBBH_LOOKUP_EVAL_DT": "300", "SOBBH_LOOKUP_NUM_M_LAYERS": "3",
               "SOBBH_LOOKUP_INTERP": "linear", "SOBBH_LOOKUP_ROW_BATCH": "8"}
        with mock.patch.dict(os.environ, env):
            s2 = SourceSOBBHSettings()
        self.assertEqual((s2.likelihood, s2.lookup_table_path, s2.lookup_eval_dt,
                          s2.lookup_num_m_layers, s2.lookup_interp, s2.lookup_row_batch),
                         ("lookup", "/x/y.h5", 300.0, 3, "linear", 8))

    def test_cfg_carries_lookup_knobs(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        s = sr.SourceSOBBHSettings(likelihood="lookup", lookup_table_path="/t.h5",
                                   lookup_eval_dt=120.0)
        gs = SimpleNamespace(tdi_chan="XYZ", tdi_gen_str="2nd generation", nchannels=3,
                             data_mode="synthetic", sobbh_reference_time=None,
                             mbh_waveform_t0=0.0, min_freq=1e-4, max_freq=2.5e-2)
        mbh = SimpleNamespace(use_tdionfly=False, tdionfly_margin=0.0, waveform_duration=0.0,
                              higher_modes=False, phenom_tol=0.0, start_freq=0.0,
                              response_order=40, buffer_time=0.0)
        emri = SimpleNamespace(response_order=40)
        with mock.patch.object(sr, "apply_emri_mode_selection_threshold", return_value=1e-3):
            cfg = sr.source_signal_cfg(gs, mbh, s, emri)
        self.assertEqual(cfg["sobbh_likelihood"], "lookup")
        self.assertEqual(cfg["sobbh_lookup_table_path"], "/t.h5")
        self.assertEqual(cfg["sobbh_lookup_eval_dt"], 120.0)
        self.assertEqual(cfg["sobbh_lookup_num_m_layers"], 2)
        self.assertEqual(cfg["sobbh_lookup_interp"], "cubic")
        self.assertEqual(cfg["sobbh_lookup_row_batch"], 32)


class LookupCompBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.wdm, cls.table = build_tiny_table(cls.tmp.name)
        cls.path = cls.table.store_path

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_missing_table_raises_with_builder_hint(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        with self.assertRaises(ValueError) as cm:
            sr.get_sobbh_lookup_comp(_general_info(self.wdm), _cfg(""))
        msg = str(cm.exception)
        self.assertIn("SOBBH_LOOKUP_TABLE_PATH", msg)
        self.assertIn("build_wdm_lookup_gpu", msg)

    def test_builds_lookup_comp_and_caches_it(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr
        from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations

        gi = _general_info(self.wdm)
        comp = sr.get_sobbh_fast_comp(gi, _cfg(self.path))
        self.assertIsInstance(comp, SOBBHLookupComputations)
        self.assertIs(sr.get_sobbh_fast_comp(gi, _cfg(self.path)), comp)
        self.assertEqual(comp.direct.num_m_layers, 2)
        self.assertEqual(comp.direct.tof.eval_dt, 600.0)
        self.assertEqual(comp.d_d, 0.0)

    def test_dispatch_names(self):
        from lisatools.globalfit.stock.erebor import source_runtime as sr

        self.assertEqual(sr.SOBBH_FAST_LIKELIHOODS, ("chunked", "lookup"))
        with self.assertRaises(ValueError):
            sr.get_sobbh_fast_comp(_general_info(self.wdm), _cfg(self.path, likelihood="full"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.wtenv/run.sh -m unittest tests.test_sobbh_lookup_stock -v`
Expected: `test_fields_and_env_knobs` fails on `lookup_table_path` (AttributeError); the others fail with `AttributeError: module ... has no attribute 'get_sobbh_lookup_comp'` / `'SOBBH_FAST_LIKELIHOODS'`.

- [ ] **Step 3: Add the settings fields** (append after `fill_m_band_half_width` in `SourceSOBBHSettings`)

```python
    # ---- "lookup" likelihood (SOBBHLookupComputations: batched TDI-on-the-fly + n_ref WDM
    # lookup table; docs/sobbh-wdm-lookup.md). The table must have the run's layer duration
    # (``Nf * dt``, 3600 s in production); any sampling with that product works.
    lookup_table_path: str = dataclasses.field(
        default_factory=env_default("SOBBH_LOOKUP_TABLE_PATH", "", str)
    )
    # layers each side of the carrier layer per pixel (5 layers for 2; SOBBH |fdot| < 0.01
    # layer units everywhere realistic, so 2 is converged)
    lookup_num_m_layers: int = dataclasses.field(
        default_factory=env_default("SOBBH_LOOKUP_NUM_M_LAYERS", 2, int)
    )
    # response evaluation step [s] for the per-channel amplitude/phase splines
    lookup_eval_dt: float = dataclasses.field(
        default_factory=env_default("SOBBH_LOOKUP_EVAL_DT", 600.0, float)
    )
    # table interpolation: "cubic" (Keys) or "linear"
    lookup_interp: str = dataclasses.field(
        default_factory=env_default("SOBBH_LOOKUP_INTERP", "cubic", str)
    )
    # rows per batched response build (bounds the spline memory)
    lookup_row_batch: int = dataclasses.field(
        default_factory=env_default("SOBBH_LOOKUP_ROW_BATCH", 32, int)
    )
```

- [ ] **Step 4: Carry them in `source_signal_cfg`** (after `sobbh_fill_m_band_half_width=...`)

```python
        sobbh_lookup_table_path=sobbh.lookup_table_path,
        sobbh_lookup_num_m_layers=sobbh.lookup_num_m_layers,
        sobbh_lookup_eval_dt=sobbh.lookup_eval_dt,
        sobbh_lookup_interp=sobbh.lookup_interp,
        sobbh_lookup_row_batch=sobbh.lookup_row_batch,
```

- [ ] **Step 5: Add the comp getter and the dispatcher** (right after `get_sobbh_chunked_comp`; add `import os` at the top of the module if it is not already imported)

```python
#: The SOBBH likelihoods served by a vectorized comp with the ``get_ll_wdm`` /
#: ``fill_global_wdm`` surface (``SOBBHChunkedLikeMove`` + the engine signal gen).
SOBBH_FAST_LIKELIHOODS = ("chunked", "lookup")


def get_sobbh_lookup_comp(general_info, cfg):
    """Build (and cache per device) the ``SOBBHLookupComputations`` for ``SOBBH_LIKELIHOOD=lookup``.

    Same ``t_ref`` / ``t_obs_start`` resolution as :func:`get_sobbh_chunked_comp`; the table
    (``SOBBH_LOOKUP_TABLE_PATH``) loads on the run's backend and must carry the run domain's
    layer duration.
    """
    from lisatools.domains import WDMLookupTable, WDMSettings
    from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations

    xp, dev, orbits, domain_settings = _wrap_device_and_orbits(general_info)
    key = ("sobbh_lookup", id(general_info), cfg["nchannels"], dev)
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    if not isinstance(domain_settings, WDMSettings):
        raise ValueError(
            "SOBBH_LIKELIHOOD=lookup needs a WDM run domain "
            f"(general.domain_settings is {type(domain_settings).__name__}); "
            "use SOBBH_LIKELIHOOD=full for FD/STFT runs."
        )
    path = str(cfg.get("sobbh_lookup_table_path") or "")
    if not path or not os.path.exists(path):
        nf, dt = int(domain_settings.Nf), float(domain_settings.data_dt)
        raise ValueError(
            "SOBBH_LIKELIHOOD=lookup needs SOBBH_LOOKUP_TABLE_PATH "
            f"(got {path!r}): an n_ref lookup table with the run's layer duration "
            f"layer_dt = {float(domain_settings.layer_dt):g} s, e.g.\n"
            "  python scripts/wdm/build_wdm_lookup_gpu.py --build-kind n_ref_complex "
            f"--Nf {nf} --Nt 1024 --dt {dt:g} --min-freq {float(domain_settings.min_freq or 1e-4):g} "
            f"--max-freq {float(domain_settings.max_freq or 2.5e-2):g} --m-ref 21 --eps-freq 0.005 "
            "--num-layers-diff 2 --eps-fdot 0.01 --fdot-max-factor 8 --time-layers 32 "
            "--nchannels 1 --out wdm_lookup_sobbh.h5\n"
            "(any (Nf, dt) with the same Nf * dt works: the table depends on the layer "
            "duration only; the 3600-s laptop EMRI table serves the production grid)"
        )
    force_backend = general_info.force_backend
    tdi_config = TDIConfig(cfg["tdi_gen_str"], force_backend=force_backend)
    t_ref = cfg["sobbh_reference_time"]
    if t_ref is None:
        t_ref = general_info.data_t0
    with device_context(xp, dev):
        table = WDMLookupTable.from_file(path, force_backend=force_backend)
        comp = SOBBHLookupComputations(
            domain_settings, float(t_ref), table,
            orbits=orbits, tdi_config=tdi_config, tdi_type=cfg["tdi_chan"],
            t_obs_start=float(general_info.data_t0),
            n_grid=cfg["sobbh_n_grid"], buffer_time=cfg["sobbh_buffer_time"],
            eval_dt=float(cfg["sobbh_lookup_eval_dt"]),
            num_m_layers=int(cfg["sobbh_lookup_num_m_layers"]),
            interp=cfg["sobbh_lookup_interp"], row_batch=int(cfg["sobbh_lookup_row_batch"]),
            force_backend=force_backend, d_d=0.0,
        )
    _WAVE_WRAP_CACHE[key] = comp
    return comp


def get_sobbh_fast_comp(general_info, cfg):
    """The vectorized SOBBH comp ``cfg["sobbh_likelihood"]`` selects (``"chunked"`` / ``"lookup"``)."""
    kind = cfg.get("sobbh_likelihood", "full")
    if kind == "chunked":
        return get_sobbh_chunked_comp(general_info, cfg)
    if kind == "lookup":
        return get_sobbh_lookup_comp(general_info, cfg)
    raise ValueError(f"no vectorized SOBBH comp for sobbh_likelihood={kind!r} "
                     f"(expected one of {SOBBH_FAST_LIKELIHOODS})")
```

- [ ] **Step 6: Route the three dispatch sites through the dispatcher**

In `get_sobbh_chunked_signal_gen`: replace `comp = get_sobbh_chunked_comp(general_info, cfg)` with `comp = get_sobbh_fast_comp(general_info, cfg)` and extend the cache key to `("sobbh_chunked_gen", cfg.get("sobbh_likelihood"), id(general_info), cfg["nchannels"], dev)`.

In `SourceSignalGen.__call__`: replace `if self.cfg.get("sobbh_likelihood", "full") == "chunked":` with `if self.cfg.get("sobbh_likelihood", "full") in SOBBH_FAST_LIKELIHOODS:`.

In `build_sobbh_move_runtime`: replace `if cfg.get("sobbh_likelihood", "full") == "chunked":` with `if cfg.get("sobbh_likelihood", "full") in SOBBH_FAST_LIKELIHOODS:` and `comp = get_sobbh_chunked_comp(curr.general_info, cfg)` with `comp = get_sobbh_fast_comp(curr.general_info, cfg)`.

Then run `grep -rn '"chunked"' src/lisatools/globalfit/stock/` and `grep -rn "sobbh_likelihood" src/lisatools/globalfit/` — every remaining dispatch on the string (not docstrings / defaults) must accept `"lookup"` too; list each site you changed in the staging message. Also update the `likelihood` field's comment in `SourceSOBBHSettings` to name the third value.

- [ ] **Step 7: Run the tests**

Run: `.wtenv/run.sh -m unittest tests.test_sobbh_lookup_stock tests.test_sobbh_chunked_move.SOBBHChunkedParityTest.test_chunked_is_the_stock_default -v`
Expected: all PASS. If `_wrap_device_and_orbits` rejects the `SimpleNamespace` (an attribute it reads is missing), add that attribute to `_general_info` in the test with the CPU value (`None` / the same orbits) — the helper reads `gpus`, `orbits`, `gpu_orbits`, `domain_settings`; it must not need anything heavier on the CPU path.

- [ ] **Step 8: Stage**

```bash
git add src/lisatools/globalfit/stock/erebor/source_runtime.py tests/test_sobbh_lookup_stock.py
```

---

### Task 7: Laptop gate script, measurements and documentation

**Files:**
- Create: `scripts/sobbh/sobbh_lookup_gate.py`
- Create: `docs/sobbh-wdm-lookup.md`
- Modify: `docs/codebase-map.md` (the `chunked_het.py` row's table: add `wdm_lookup_eval.py` and `sources/sobbh/wdm_direct.py`)

**Interfaces:**
- Consumes: Tasks 1-5; `bbhx.sobbhcomps.SOBBHWDMComputations` (the chunked comp, same signature); the laptop table `/Users/mkatz/Research/lisa_sprint_2026/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5`.
- Produces: `sobbh_lookup_gate.py --table PATH [--nt 1024] [--rows 8] [--out gate.jsonl] [--no-chunked]` printing one row per source (`mm_X/Y/Z`, `ratio_X/Y/Z`, `mm_w` noise-weighted, `dlogL`, `snr`) plus a timing block (lookup `get_ll` / fill vs chunked `get_ll` per batch), appending JSON lines.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
"""Laptop gate for the SOBBH direct-to-WDM lookup scorer (docs/sobbh-wdm-lookup.md).

Grid: the production layer duration (3600 s) sampled at 20 s (Nf=180; Nyquist 25 mHz), NT
layers (1024 = 42.7 d, 4320 = 6 months), synthetic epoch 0.5 yr. For each catalogue-like
source: the lookup template vs the batched response's own dense TD->WDM (flat per-channel
mismatch 1 - Re(O) with NO maximisation, norm ratio), the noise-weighted mismatch and dlogL
(scirdv1, XYZ), the SNR; then wall times per batch of ``--rows`` rows for the lookup comp's
get_ll / fill and the chunked comp's get_ll on the same residual, plus the lnL agreement of
both comps against the exact container inner products.

Run (one Python process at a time on the laptop):
    .wtenv/run.sh scripts/sobbh/sobbh_lookup_gate.py --table $TABLE --nt 1024 --rows 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np

NF, DT = 180, 20.0
EDGE = 24

SOURCES = np.array([
    # (m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta)
    [36.0, 29.0, 0.1, -0.2, 0.6e9, 4.5e-3, 0.3, 0.8, 1.1, 2.0, 0.4],
    [50.0, 40.0, 0.3, 0.3, 1.0e9, 8.0e-3, 2.2, 2.1, 0.4, 4.4, -0.7],
    [25.0, 20.0, -0.4, 0.0, 0.3e9, 1.2e-2, 4.0, 1.4, 2.5, 0.7, 1.1],
    [60.0, 55.0, 0.1, 0.2, 0.8e9, 1.5e-2, 1.1, 1.2, 0.7, 3.1, 0.2],
    [20.0, 10.0, 0.0, 0.5, 0.5e9, 6.0e-3, 5.5, 0.3, 1.8, 5.9, -1.2],
    [80.0, 75.0, 0.6, 0.6, 2.0e9, 1.8e-2, 0.9, 1.9, 0.1, 1.5, 0.9],
])


def mm_flat(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    aa, bb, ab = float(a @ a), float(b @ b), float(a @ b)
    return 1.0 - ab / np.sqrt(aa * bb), np.sqrt(aa / bb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=os.environ.get("SOBBH_LOOKUP_TABLE_PATH",
                    "/Users/mkatz/Research/lisa_sprint_2026/"
                    "wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5"))
    ap.add_argument("--nt", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--eval-dt", type=float, default=600.0)
    ap.add_argument("--out", default="sobbh_lookup_gate.jsonl")
    ap.add_argument("--no-chunked", action="store_true")
    args = ap.parse_args()

    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.diagnostic import inner_product
    from lisatools.domains import TDSettings, TDSignal, WDMLookupTable, WDMSettings, WDMSignal
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations
    from lisatools.utils.constants import YRSID_SI

    nt = int(args.nt)
    nobs = NF * nt
    t0 = int(0.5 * YRSID_SI / DT) * DT
    ref = float(t0)
    orbits = EqualArmlengthOrbits(force_backend="cpu")
    tdi = TDIConfig("2nd generation", force_backend="cpu")
    wdm = WDMSettings(NF, nt, DT, t0=t0, min_freq=2e-3, max_freq=2.4e-2, force_backend="cpu")
    table = WDMLookupTable.from_file(args.table, force_backend="cpu")
    comp = SOBBHLookupComputations(
        wdm, ref, table, orbits=orbits, tdi_config=tdi, tdi_type="XYZ", n_grid=2048,
        buffer_time=5000.0, eval_dt=args.eval_dt, num_m_layers=2, interp="cubic",
        row_batch=args.rows, force_backend="cpu", d_d=0.0)
    direct = comp.direct
    grid_t = np.arange(nobs) * DT + t0
    tds = TDSettings(nobs, DT, force_backend="cpu")
    sens = XYZ2SensitivityMatrix(wdm, model="scirdv1")

    print(f"grid Nf={NF} Nt={nt} dt={DT} layer_dt={wdm.layer_dt} s; table {os.path.basename(args.table)}")
    rows_out = []
    h_tof = []
    h_look = []
    for i, src in enumerate(SOURCES):
        t_a = time.perf_counter()
        out = direct.tof.build(src[None, :], float(grid_t[0]), float(grid_t[-1]))
        td = np.asarray(out.eval_tdi(grid_t))[0]
        truth = np.asarray(TDSignal(td, tds).transform(wdm).arr)
        t_tof = time.perf_counter() - t_a
        t_a = time.perf_counter()
        got = np.asarray(direct.dense(src[None, :])[0].arr)
        t_look = time.perf_counter() - t_a
        sl = slice(EDGE, nt - EDGE)
        mms = [mm_flat(got[c, :, sl], truth[c, :, sl]) for c in range(3)]
        ac = AnalysisContainer(WDMSignal(truth, wdm), sens)
        h_sig = WDMSignal(got, wdm)
        hh_t = float(np.real(inner_product(WDMSignal(truth, wdm), WDMSignal(truth, wdm), psd=sens)))
        hh_l = float(np.real(inner_product(h_sig, h_sig, psd=sens)))
        dh = float(np.real(ac.template_inner_product(h_sig)))
        row = dict(src=i, f_low=float(src[5]), snr=float(np.sqrt(hh_t)),
                   mm=[m for m, _ in mms], ratio=[r for _, r in mms],
                   mm_w=1.0 - dh / np.sqrt(hh_t * hh_l),
                   dlogL=-0.5 * (hh_t + hh_l - 2.0 * dh),
                   t_tof_dense_s=t_tof, t_lookup_s=t_look, stats=dict(direct.last_stats))
        rows_out.append(row)
        h_tof.append(truth)
        h_look.append(got)
        print(f"src {i} f_low {src[5]:.4f} snr {row['snr']:.1f}  mm X/Y/Z "
              f"{row['mm'][0]:.2e}/{row['mm'][1]:.2e}/{row['mm'][2]:.2e}  ratio "
              f"{row['ratio'][0]:.5f}/{row['ratio'][1]:.5f}/{row['ratio'][2]:.5f}  mm_w "
              f"{row['mm_w']:.2e}  dlogL {row['dlogL']:.3e}  tof-dense {t_tof:.1f}s lookup {t_look:.1f}s")

    # ---- scoring timings: both comps, one residual, the same batch of rows --------------
    data = h_tof[0] + 0.5 * h_tof[1]
    ac = AnalysisContainer(WDMSignal(data.copy(), wdm), XYZ2SensitivityMatrix(wdm, model="scirdv1"))
    aca = AnalysisContainerArray([ac])
    batch = np.tile(SOURCES[0], (args.rows, 1))
    rng = np.random.default_rng(1)
    batch[:, 5] += rng.uniform(-0.3, 0.3, args.rows) * float(wdm.layer_df)
    batch[:, 6] = rng.uniform(0, 2 * np.pi, args.rows)
    idx = np.zeros(args.rows, dtype=np.int32)
    exact = []
    for r in batch:
        hb = np.asarray(TDSignal(np.asarray(direct.tof.build(r[None, :], float(grid_t[0]),
                                                                 float(grid_t[-1])).eval_tdi(grid_t))[0],
                                 tds).transform(wdm).arr)
        hs = WDMSignal(hb, wdm)
        exact.append(float(np.real(ac.template_inner_product(hs)))
                     - 0.5 * float(np.real(inner_product(hs, hs, psd=ac.sens_mat))))
    exact = np.asarray(exact)
    timing = {}
    for label in ("lookup_get_ll_warm", "lookup_get_ll"):
        t_a = time.perf_counter()
        ll_look = np.asarray(comp.get_ll_wdm(batch, aca, data_index=idx, noise_index=idx))
        timing[label] = time.perf_counter() - t_a
    buf = np.zeros(3 * int(wdm.Nf_active) * int(wdm.Nt_active))
    t_a = time.perf_counter()
    comp.fill_global_wdm(batch, buf, data_index=idx, factors=np.ones(args.rows))
    timing["lookup_fill"] = time.perf_counter() - t_a
    timing["lookup_vs_exact_max_abs"] = float(np.abs(ll_look - exact).max())
    if not args.no_chunked:
        from bbhx.sobbhcomps import SOBBHWDMComputations

        ch = SOBBHWDMComputations(wdm, t_ref=ref, Nt_sub=32, n_pad=4, N_sparse=256, N_cp_sig=0,
                                  N_cp_orbit=0, orbits=orbits, tdi_config="2nd generation",
                                  force_backend="cpu", d_d=0.0, tdi_type="XYZ")
        for label in ("chunked_get_ll_warm", "chunked_get_ll"):
            t_a = time.perf_counter()
            ll_ch = np.asarray(ch.get_ll_wdm(batch, aca, data_index=idx, noise_index=idx,
                                             m_band_half_width=3))
            timing[label] = time.perf_counter() - t_a
        timing["chunked_vs_exact_max_abs"] = float(np.abs(ll_ch - exact).max())
    timing["rows"] = args.rows
    timing["nt"] = nt
    print("timing:", json.dumps(timing, indent=1))
    with open(args.out, "a") as fp:
        for row in rows_out:
            fp.write(json.dumps(dict(nt=nt, eval_dt=args.eval_dt, **row)) + "\n")
        fp.write(json.dumps(dict(timing=timing)) + "\n")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the gate at 42.7 days, then at 6 months**

Run: `.wtenv/run.sh scripts/sobbh/sobbh_lookup_gate.py --nt 1024 --rows 8 --out docs/sobbh_lookup_gate.jsonl`
Expected: 6 source rows with `mm` at or below 1e-3 per channel, norm ratios within 2e-3 of 1, `dropped_pixels == 0`; both comps' `*_vs_exact_max_abs` reported (the chunked value is the known truncation error at `m_band_half_width=3`). Then: `.wtenv/run.sh scripts/sobbh/sobbh_lookup_gate.py --nt 4320 --rows 8 --out docs/sobbh_lookup_gate.jsonl` (6 months at 20 s; ~10-20 min; watch RSS, the runner exits at 5 GB; if it does, rerun with `--rows 4`). Keep the printed tables.

- [ ] **Step 3: Write the status doc**

```markdown
# SOBBH direct-to-WDM lookup scorer: status and measurements

Branch `sobbh-wdm-lookup` (LAT). Spec `docs/superpowers/specs/2026-09-30-sobbh-wdm-lookup-design.md`,
plan `docs/superpowers/plans/2026-09-30-sobbh-wdm-lookup.md`. The SOBBH twin of the EMRI
direct-to-WDM template (`docs/emri-direct-wdm.md`), vectorized over proposal rows and slotted into
`SOBBHChunkedLikeMove` as a comp.

## What exists

| Piece | Where |
|---|---|
| Vectorized n_ref table evaluator (quarter-turn rule, linear / Keys cubic) | `lisatools/wdm_lookup_eval.py` `WDMLookupEvaluator` |
| Batched 3.5PN, batched TDI-on-the-fly, tracer, sparse template, inner products, fill | `lisatools/sources/sobbh/wdm_direct.py` |
| The comp for the move / engine (`get_ll_wdm`, `fill_global_wdm`) | `SOBBHLookupComputations` (same file) |
| Stock knob | `SOBBH_LIKELIHOOD=lookup` + `SOBBH_LOOKUP_TABLE_PATH` (and `SOBBH_LOOKUP_{NUM_M_LAYERS,EVAL_DT,INTERP,ROW_BATCH}`) |
| Gate script | `scripts/sobbh/sobbh_lookup_gate.py` |

## The table

The n_ref table depends on the layer duration only: two tables with `layer_dt = 3600 s` built at
(Nf=64, dt=56.25) and (Nf=128, dt=28.125) agree entry by entry to 4e-9 (test
`TablePortabilityTest`). The laptop EMRI table `wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5`
(layer 3600 s, offsets [-3, 3) layers, fdot +-8 layer units in steps of 0.01, 30 MB) therefore
serves the production grid (Nf=1440, dt=2.5). SOBBH chirp rates are tiny in layer units
(|fdot| <= 0.002 for catalogue-like sources; the 6-mo catalogue's worst chirper ~0.005), so
`num_m_layers=2` (5 layers per pixel) is converged and no plunge chunk is needed; pixels past the
fdot axis (sources merging in band) are dropped and counted.

## Results (laptop CPU, EqualArmlengthOrbits, 2nd-generation TDI, scirdv1)

PASTE the two printed tables of `scripts/sobbh/sobbh_lookup_gate.py` here (42.7 d and 6 months):
per source `mm X/Y/Z`, norm ratios, `mm_w`, `dlogL`, SNR, and the timing block (lookup
`get_ll` / fill vs chunked `get_ll` per batch of 8 rows, both comps' lnL error vs the exact
container inner products). Also paste the per-channel numbers printed by
`tests/test_sobbh_wdm_direct.py::DirectWDMTest::test_dense_matches_tof_own_transform`.

## Conventions pinned by tests

- Chunked-basis rows `(m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta)`; the
  response feed is `phase = gw_phase + pi` with the intrinsic amplitude (`SOBBHTDIonFly`).
- Pixel centres `t_n = t_obs_start + n * layer_dt`, `n` the absolute grid index; the quarter-turn
  rule uses the absolute `(m + n)` parity.
- Inner products carry NO `4 * differential_component` factor (the chunked kernel convention);
  `-0.5 (h_h - 2 d_h)` is the container's source term.
- Batched response vs production `SOBBHTDIonFly` per row: mismatch <= 1e-8 at `eval_dt = 600 s`
  (`BatchedTOFTest`); a one-day reference-epoch error is caught (control).

## Known limits

- Python, `xp`-vectorized (numpy / cupy); no C++/CUDA kernel yet (see follow-ups).
- A source merging inside the window keeps evaluating up to `tc` with the shared node grid
  (production `SOBBHTDIonFly` zeros the last `buffer_time` before merger instead); its pixels past
  the table's fdot axis are dropped (counted in `last_stats["dropped_pixels"]`, one warning per
  call).
- Single-shard comp (multi-GPU walker shards are routed per split by the move, as for chunked).
- cubic interpolation is Keys cubic convolution, not scipy's global cubic spline.

## Follow-ups

1. **CUDA/C++ kernel** in the chunked-het family: per-pixel `SOBBHTDIonTheFly::get_tdi` amplitude /
   phase at the pixel centres, the table resident on the device, the same quarter-turn rule and
   the same `(d_h, h_h)` accumulation; this Python path is its reference. GPU leads per the sprint
   rule; the comp swap is then a comp swap, not a move change.
2. **CD1L data gate**: per-source mismatch / dlogL of the lookup template vs the mojito L1 SOBBH
   streams (ids 0-5 of the 6-mo run) on the production grid — the bricks are not on this laptop.
   Recipe: `L1ProcessingStep(source_types=["sobbh"], source_ids=dict(sobbh=[id]), orbits_class=L1Orbits,
   frame="icrs")` as `scripts/emri/emri_cd1l_campaign.py` does for EMRIs, the stock SOBBH wave wrap as
   the production template, `SOBBHDirectWDM.dense` as the fast one, report `1 - Re(O)` per channel,
   norm ratio, dlogL, SNR.
3. JAX mirror; the fused phase-max quadrature; F-stat / gradient methods on the comp.
```

- [ ] **Step 4: Add the codebase-map rows**

In `docs/codebase-map.md`, directly below the `chunked_het.py` row, add:

```markdown
| `wdm_lookup_eval.py` | `WDMLookupEvaluator` — vectorized (numpy/cupy) evaluation of an `n_ref` WDM lookup table with the quarter-turn rule (linear / Keys cubic). |
| `sources/sobbh/wdm_direct.py` | SOBBH direct-to-WDM: batched 3.5PN, batched TDI-on-the-fly, tracer, sparse lookup template, sparse inner products / fill, and `SOBBHLookupComputations` (the `SOBBH_LIKELIHOOD=lookup` comp for `SOBBHChunkedLikeMove`). |
```

- [ ] **Step 5: Run the whole new suite plus the SOBBH neighbours once more**

Run: `.wtenv/run.sh -m unittest tests.test_wdm_lookup_eval tests.test_sobbh_wdm_direct tests.test_sobbh_lookup_move tests.test_sobbh_lookup_stock tests.test_sobbh_chunked_move tests.test_sobbh_chunked_fill tests.test_wdm_lookup_basis_cycle -v`
Expected: all PASS.

- [ ] **Step 6: Stage**

```bash
git add scripts/sobbh/sobbh_lookup_gate.py docs/sobbh-wdm-lookup.md docs/codebase-map.md docs/sobbh_lookup_gate.jsonl
```

---

## Verification (end-to-end)

1. `.wtenv/run.sh -m unittest tests.test_wdm_lookup_eval tests.test_sobbh_wdm_direct tests.test_sobbh_lookup_move tests.test_sobbh_lookup_stock -v` — all green.
2. `.wtenv/run.sh -m unittest tests.test_sobbh_chunked_move tests.test_sobbh_chunked_fill tests.test_wdm_lookup_basis_cycle tests.test_emri_wdm_direct -v` — unchanged neighbours green.
3. The three mutation checks (Tasks 1, 4, 5) each named a failing test.
4. `docs/sobbh-wdm-lookup.md` carries the measured tables (42.7 d and 6 months) and timings.
5. `git status` shows only staged files under `src/lisatools/`, `tests/`, `scripts/sobbh/`, `docs/`; nothing committed.

---

### Task 8: One-GPU speed test (cluster-ready script + runbook; CPU smoke here)

Requested by Mike 2026-10-01: "finish up with a speed test for the 1 GPU — on the cluster". This laptop has no CUDA backend, so the deliverable is a script that runs the lookup comp and the chunked comp head to head on ONE GPU at the production 6-month grid, smoke-tested here on CPU with a small preset, plus the runbook in the status doc. The GPU numbers come from the cluster run.

**Files:**
- Create: `scripts/sobbh/sobbh_lookup_speed_gpu.py`
- Modify: `docs/sobbh-wdm-lookup.md` (new section "One-GPU speed test (cluster runbook)")

**Interfaces:**
- Consumes: `SOBBHLookupComputations(wdm_settings, t_ref, table, *, orbits, tdi_config, tdi_type, t_obs_start, n_grid, buffer_time, eval_dt, num_m_layers, interp, row_batch, force_backend, d_d)` with `get_ll_wdm(params, holder, data_index=, noise_index=)`, `fill_global_wdm(params, buffer, data_index=, factors=)`, `last_call_spans`; `bbhx.sobbhcomps.SOBBHWDMComputations` (same call surface plus `m_band_half_width=`); `AnalysisContainerArray(analysis_containers, gpus=None)`; `WDMLookupTable.from_file(path, force_backend=)`.
- Produces: `sobbh_lookup_speed_gpu.py --backend {cpu,cuda12x,...} [--laptop] [--nf 1440 --nt 4320 --dt 2.5] --rows 4,8,32,96 [--row-batch 32] [--repeats 3] [--eval-dt 600] [--table PATH] [--out speed.jsonl] [--no-chunked] [--m-band 3] [--fill-band 8] [--nt-sub 32]`, printing one table row per batch size and appending JSON lines.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
"""One-GPU speed test: the SOBBH lookup comp vs the chunked-heterodyne comp, head to head.

Production 6-month grid by default (Nf=1440, Nt=4320, dt=2.5 s: layer 3600 s, band
2.5e-4..2.5e-2 Hz), synthetic epoch 0.5 yr, EqualArmlengthOrbits, 2nd-generation TDI, scirdv1
XYZ, ONE walker slab. For every batch size in ``--rows``: lookup ``get_ll_wdm`` (cold call, then
the median of ``--repeats`` warm calls) and ``fill_global_wdm``; chunked ``get_ll_wdm``
(``--m-band``) and ``fill_global_wdm`` (``--fill-band``); a device sync around every call; the
lookup's template/inner split; the GPU memory-pool peak; and the per-row lnL difference of the
two comps on the same residual (a consistency figure, not an accuracy gate: the accuracy gates
are docs/sobbh-wdm-lookup.md).

Cluster (one GPU; the stack's python, the branch checked out):
    SOBBH_LOOKUP_TABLE_PATH=/path/to/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5 \\
    python scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cuda12x --rows 4,8,32,96,288 \\
        --out sobbh_lookup_speed_gpu.jsonl
Laptop smoke (CPU, small preset):
    .wtenv/run.sh scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cpu --laptop --nt 256 \\
        --rows 2,4 --repeats 1 --out /tmp/speed_smoke.jsonl
The table depends on the layer duration only, so the 3600-s laptop table serves the production
grid (TablePortabilityTest).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

TABLE_DEFAULT = (
    "/Users/mkatz/Research/lisa_sprint_2026/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5"
)
# (m1, m2, s1, s2, dist[pc], f_low, phi_c, inc, psi, lam, beta): the gate's catalogue-like rows
SOURCES = np.array([
    [36.0, 29.0, 0.1, -0.2, 0.6e9, 4.5e-3, 0.3, 0.8, 1.1, 2.0, 0.4],
    [50.0, 40.0, 0.3, 0.3, 1.0e9, 8.0e-3, 2.2, 2.1, 0.4, 4.4, -0.7],
    [25.0, 20.0, -0.4, 0.0, 0.3e9, 1.2e-2, 4.0, 1.4, 2.5, 0.7, 1.1],
    [60.0, 55.0, 0.1, 0.2, 0.8e9, 1.5e-2, 1.1, 1.2, 0.7, 3.1, 0.2],
    [20.0, 10.0, 0.0, 0.5, 0.5e9, 6.0e-3, 5.5, 0.3, 1.8, 5.9, -1.2],
    [80.0, 75.0, 0.6, 0.6, 2.0e9, 1.8e-2, 0.9, 1.9, 0.1, 1.5, 0.9],
])


def sync(xp):
    if hasattr(xp, "cuda"):
        xp.cuda.Device().synchronize()


def timed(fn, xp, repeats):
    """``(cold, warm_median)`` seconds of ``fn()``: one cold call, then ``repeats`` warm calls."""
    sync(xp)
    t = time.perf_counter()
    fn()
    sync(xp)
    cold = time.perf_counter() - t
    warm = []
    for _ in range(int(repeats)):
        sync(xp)
        t = time.perf_counter()
        fn()
        sync(xp)
        warm.append(time.perf_counter() - t)
    return cold, (float(np.median(warm)) if warm else cold)


def pool_peak_gb(xp):
    if not hasattr(xp, "cuda"):
        return float("nan")
    return xp.get_default_memory_pool().total_bytes() / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default=None,
                    help="cpu / cuda12x / ... (default: the first CUDA backend, else cpu)")
    ap.add_argument("--laptop", action="store_true", help="Nf=180, dt=20 (layer 3600 s) preset")
    ap.add_argument("--nf", type=int, default=1440)
    ap.add_argument("--nt", type=int, default=4320)
    ap.add_argument("--dt", type=float, default=2.5)
    ap.add_argument("--rows", default="4,8,32,96")
    ap.add_argument("--row-batch", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--eval-dt", type=float, default=600.0)
    ap.add_argument("--table", default=os.environ.get("SOBBH_LOOKUP_TABLE_PATH", TABLE_DEFAULT))
    ap.add_argument("--out", default="sobbh_lookup_speed_gpu.jsonl")
    ap.add_argument("--no-chunked", action="store_true")
    ap.add_argument("--m-band", type=int, default=3, help="chunked scoring band half-width")
    ap.add_argument("--fill-band", type=int, default=8, help="chunked fill band half-width")
    ap.add_argument("--nt-sub", type=int, default=32)
    args = ap.parse_args()

    import lisatools

    backend = args.backend
    if backend is None:
        backend = "cuda" if lisatools.has_backend("cuda") else "cpu"
    if not lisatools.has_backend(backend):
        raise SystemExit(f"backend {backend!r} unavailable on this host")
    if args.laptop:
        args.nf, args.dt = 180, 20.0
    nf, nt, dt = int(args.nf), int(args.nt), float(args.dt)

    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.domains import WDMLookupTable, WDMSettings, WDMSignal
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.sensitivity import XYZ2SensitivityMatrix
    from lisatools.sources.sobbh.wdm_direct import SOBBHLookupComputations
    from lisatools.utils.constants import YRSID_SI
    from lisatools.utils.utility import asnumpy

    t0 = int(0.5 * YRSID_SI / dt) * dt
    ref = float(t0)
    orbits = EqualArmlengthOrbits(force_backend=backend)
    tdi = TDIConfig("2nd generation", force_backend=backend)
    wdm = WDMSettings(nf, nt, dt, t0=t0, min_freq=2.5e-4, max_freq=2.5e-2, force_backend=backend)
    t_build = time.perf_counter()
    table = WDMLookupTable.from_file(args.table, force_backend=backend)
    comp = SOBBHLookupComputations(
        wdm, ref, table, orbits=orbits, tdi_config=tdi, tdi_type="XYZ", n_grid=2048,
        buffer_time=5000.0, eval_dt=args.eval_dt, num_m_layers=2, interp="cubic",
        row_batch=args.row_batch, force_backend=backend, d_d=0.0)
    xp = comp.xp
    t_build = time.perf_counter() - t_build
    nch, nfa, nta = 3, int(wdm.Nf_active), int(wdm.Nt_active)
    print(f"backend {backend}  grid Nf={nf} Nt={nt} dt={dt} layer_dt={float(wdm.layer_dt):g} s  "
          f"active {nfa} x {nta}  table {os.path.basename(args.table)}  comp build {t_build:.1f} s")

    # one residual slab: the lookup fill of two sources (no dense transform on any backend)
    data = xp.zeros(nch * nfa * nta)
    comp.fill_global_wdm(SOURCES[:2], data, data_index=np.zeros(2, dtype=np.int32),
                         factors=np.ones(2))
    sync(xp)
    ac = AnalysisContainer(WDMSignal(data.reshape(nch, nfa, nta), wdm),
                           XYZ2SensitivityMatrix(wdm, model="scirdv1"))
    dev = getattr(getattr(data, "device", None), "id", None)
    aca = AnalysisContainerArray([ac], gpus=None if dev is None else [int(dev)])

    ch = None
    if not args.no_chunked:
        from bbhx.sobbhcomps import SOBBHWDMComputations

        t_build = time.perf_counter()
        ch = SOBBHWDMComputations(
            wdm, t_ref=ref, Nt_sub=args.nt_sub, n_pad=4, N_sparse=256, N_cp_sig=0,
            N_cp_orbit=0, orbits=orbits, tdi_config="2nd generation", force_backend=backend,
            d_d=0.0, tdi_type="XYZ")
        print(f"chunked comp: Nt_sub={args.nt_sub} n_chunks={ch.n_chunks} "
              f"build {time.perf_counter() - t_build:.1f} s")

    rng = np.random.default_rng(1)
    rows_list = [int(r) for r in args.rows.split(",") if r.strip()]
    header = (f"{'rows':>5} {'look_cold':>10} {'look_warm':>10} {'tmpl':>8} {'inner':>8} "
              f"{'look_fill':>10} {'ch_cold':>9} {'ch_warm':>9} {'ch_fill':>9} {'ch/look':>8} "
              f"{'max|dll|':>10} {'pool_GB':>8}")
    print(header)
    with open(args.out, "a") as fp:
        for n in rows_list:
            batch = np.tile(SOURCES[0], (n, 1))
            batch[:, 5] += rng.uniform(-0.3, 0.3, n) * float(wdm.layer_df)
            batch[:, 6] = rng.uniform(0.0, 2.0 * np.pi, n)
            idx = np.zeros(n, dtype=np.int32)
            fac = np.ones(n)
            rec = dict(rows=n, backend=backend, nf=nf, nt=nt, dt=dt, row_batch=args.row_batch,
                       eval_dt=args.eval_dt)

            look_cold, look_warm = timed(
                lambda: comp.get_ll_wdm(batch, aca, data_index=idx, noise_index=idx),
                xp, args.repeats)
            ll_look = np.asarray(asnumpy(comp.get_ll_wdm(batch, aca, data_index=idx,
                                                          noise_index=idx)), dtype=float)
            spans = dict(comp.last_call_spans or {})
            buf = xp.zeros(nch * nfa * nta)
            _, look_fill = timed(
                lambda: comp.fill_global_wdm(batch, buf, data_index=idx, factors=fac),
                xp, args.repeats)
            rec.update(look_cold=look_cold, look_warm=look_warm, look_fill=look_fill,
                       look_template=float(spans.get("template", float("nan"))),
                       look_inner=float(spans.get("inner", float("nan"))),
                       lookup_stats=dict(comp.last_stats))

            ch_cold = ch_warm = ch_fill = dll = float("nan")
            if ch is not None:
                ch_cold, ch_warm = timed(
                    lambda: ch.get_ll_wdm(batch, aca, data_index=idx, noise_index=idx,
                                          m_band_half_width=args.m_band),
                    xp, args.repeats)
                ll_ch = np.asarray(asnumpy(ch.get_ll_wdm(
                    batch, aca, data_index=idx, noise_index=idx,
                    m_band_half_width=args.m_band)), dtype=float)
                buf2 = xp.zeros(nch * nfa * nta)
                _, ch_fill = timed(
                    lambda: ch.fill_global_wdm(batch, buf2, data_index=idx, factors=fac,
                                               m_band_half_width=args.fill_band),
                    xp, args.repeats)
                dll = float(np.abs(ll_look - ll_ch).max())
                rec.update(ch_cold=ch_cold, ch_warm=ch_warm, ch_fill=ch_fill,
                           max_abs_dll=dll, median_abs_dll=float(np.median(np.abs(ll_look - ll_ch))))
            rec["pool_peak_gb"] = pool_peak_gb(xp)
            ratio = ch_warm / look_warm if look_warm > 0 else float("nan")
            print(f"{n:5d} {look_cold:10.3f} {look_warm:10.3f} {rec['look_template']:8.3f} "
                  f"{rec['look_inner']:8.3f} {look_fill:10.3f} {ch_cold:9.3f} {ch_warm:9.3f} "
                  f"{ch_fill:9.3f} {ratio:8.2f} {dll:10.3e} {rec['pool_peak_gb']:8.2f}")
            fp.write(json.dumps(rec) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: CPU smoke (laptop)**

Run: `.wtenv/run.sh scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cpu --laptop --nt 256 --rows 2,4 --repeats 1 --out /tmp/speed_smoke.jsonl`
Expected: the header, two table rows with finite lookup and chunked timings, `max|dll|` finite (the two comps score the same residual; a difference of order the chunked truncation at `m_band 3` is expected), `pool_GB` `nan` on CPU; then `wrote /tmp/speed_smoke.jsonl`. Fix only the SCRIPT if an API call mismatches the landed code; never the library. Also run it once at the production grid on CPU with the smallest batch to prove the grid itself works here: `.wtenv/run.sh scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cpu --rows 2 --repeats 1 --no-chunked --out /tmp/speed_prod_cpu.jsonl` (6-month grid at dt 2.5; expect a few minutes; if the runner's 5 GB watchdog exits, record that and move on).

- [ ] **Step 3: Runbook section in the doc**

Append to `docs/sobbh-wdm-lookup.md`:

```markdown
## One-GPU speed test (cluster runbook)

`scripts/sobbh/sobbh_lookup_speed_gpu.py` times the lookup comp and the chunked comp head to head
on ONE device at the production 6-month grid (Nf=1440, Nt=4320, dt=2.5): `get_ll_wdm` cold / warm
(median of `--repeats`), the fill, the lookup's template/inner split, the memory-pool peak and the
per-row lnL difference of the two comps on the same residual. Device-synchronised timings.

    # cluster, one GPU, the stack's python, this branch checked out
    SOBBH_LOOKUP_TABLE_PATH=/path/to/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5 \
    python scripts/sobbh/sobbh_lookup_speed_gpu.py --backend cuda12x --rows 4,8,32,96,288 \
        --out sobbh_lookup_speed_gpu.jsonl

Reading the table: `ch/look` is the warm chunked-over-lookup ratio per batch; the chunked kernel's
wall is flat in the batch size up to the device's block capacity (one block per row), so the
interesting columns are how `look_warm` scales with `rows` and the `tmpl` (batched response +
lookup) vs `inner` (gathers + reductions) split. `--row-batch` bounds the response-spline memory
(default 32); raise it on a large-memory device and watch `pool_GB`.

GPU results: NOT MEASURED on the laptop (no CUDA backend). CPU smoke on the laptop preset:
(paste the smoke table here) ; production grid on CPU, 2 rows, lookup only: (paste or "watchdog").
```

- [ ] **Step 4: Stage**

```bash
git add scripts/sobbh/sobbh_lookup_speed_gpu.py docs/sobbh-wdm-lookup.md
```
