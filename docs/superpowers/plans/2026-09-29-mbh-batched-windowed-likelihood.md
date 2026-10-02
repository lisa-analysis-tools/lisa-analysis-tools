# MBH Batched Windowed Likelihood Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the global fit an MBH add/remove move that generates a chunk of walker rows in one batched response launch on a per-leaf 90-day-before / 10-day-after-merger lattice, transforms each row on a segment WDM grid, and scores every row against its own container's residual and PSD, with the stock path untouched and the knob OFF by default.

**Architecture:** Four library additions (WDM sub-box slicing/adding in `domains.py` and `analysiscontainer.py`; a windowed grid-aligned phentax generator in `sources/bbh/gridaligned.py`; a segment-transform adapter in `sources/batching.py`) feed a new `MBHBatchedLikeMove` in `globalfit/moves/mbhbatchedmove.py` that mirrors `SOBBHChunkedLikeMove` (per-walker exposed-residual offset, shard routing, batched fill, built-in fast-vs-slow check). Stock erebor wiring selects it on `MBH_LIKELIHOOD=batched`. A script validates match and logL against the mojito MBHB files on this laptop and doubles as the GPU probe.

**Tech Stack:** Python 3.12, numpy/cupy `xp` pattern, JAX + phentax (MBH strain), `pyResponseTDI` (LAT native response), `unittest`.

**Spec:** `docs/superpowers/specs/2026-09-29-mbh-batched-windowed-likelihood-design.md` (read it first; this plan argues from it).

## Global Constraints

- Branch/worktree: `/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-cd1l` (branch `cd1l-merge`, PR #82/#81 merged and staged, NOT committed). Eryn worktree: `/Users/mkatz/Research/lisa_sprint_2026/Eryn-cd1l`.
- **Commits: Mike's standing rule is no `git commit` / `git push` unless he says so in the session.** Each task's last step STAGES (`git add`) and stops. If Mike has said "commit" in the executing session, commit with the message shown.
- **Laptop: ONE python process at a time, CPU at or below 50%, 8 GB RAM.** Never run two test files concurrently. Reviewers read; they do not run tests while an implementer's tests run.
- Tests are `unittest`. Run them ONLY through the worktree-shadowed environment below (the `deving` env has LAT editable-installed from the MAIN tree; without the shadow you test the wrong code):

```bash
export LAT_WORKTREE_SRC=/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-cd1l/src
export PYTHONPATH=/Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-noise-merge/.wtenv:/Users/mkatz/Research/lisa_sprint_2026/Eryn-cd1l/src
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu
PY=/Users/mkatz/miniconda3/envs/deving/bin/python
cd /Users/mkatz/Research/lisa_sprint_2026/LISAanalysistools-cd1l
nice -n 10 $PY -m unittest tests.<module> -v
```

- Rule 0 knob naming: env var = capitalised field name (`likelihood` -> `MBH_LIKELIHOOD`). No new settings files. No backend strings as method kwargs.
- Settings objects must survive `pickle.loads(pickle.dumps(copy.deepcopy(obj)))`.
- Never store `self.xp = cp`; derive `xp` from arrays or settings.
- Waveform-basis MBH row (what every scoring call receives): columns `0 m1, 1 m2, 2 s1z, 3 s2z, 4 dist[Mpc], 5 phi_ref, 6 iota, 7 psi, 8 alpha, 9 delta, 10 t_plunge` (seconds relative to `waveform_t0`).
- Spec rulings: `MBH_RESPONSE_ORDER` default 8; `MBH_BATCH_MAX_SIZE` default 16; window 90 d before / 10 d after; pad 4 d; margin 1 d; defaults leave `MBH_LIKELIHOOD=full`.

## Review Focus

1. A template box that reaches outside the data's active box (merger within 90 days of the data start, or the window clamped at the data end): `mbh_window_layers` must clamp so the segment fits, and `_apply_wdm_add` / `_slice_wdm_to_template` must raise, never silently wrap. Pinned in Task 1 (`test_box_outside_data_raises`) and Task 4 (`test_window_clamps_at_grid_edges`).
2. A batch chunk of exactly one row (remainder when `batch_max_size` does not divide the walker count): the adapter must still return a batched template with a leading axis and the move must score it. Pinned in Task 2 (`test_single_row_batch_keeps_leading_axis`) and Task 4 (`test_chunk_size_does_not_change_results`).
3. A leaf whose cold-chain median merger time drifts between visits: the window must stay put inside the margin and rebuild (logged) outside it, never mid-visit. Pinned in Task 4 (`test_window_hysteresis`).
4. Two walkers with DIFFERENT PSDs: the batched path must weight each row by its own container's PSD. Pinned in Task 4 (`test_batched_matches_container_path_per_walker_psd`).
5. `MBH_LIKELIHOOD=batched` with `MBH_WAVEFORM_DURATION` set to something other than the window, or with `USE_TDIONFLY=1`: the build must refuse loudly. Pinned in Task 5 (`test_duration_conflict_raises`, `test_tdionfly_conflict_raises`).

---

### Task 1: WDM sub-box support in the domain and container layers

**Files:**
- Modify: `src/lisatools/domains.py` (add `WDMSettings.get_slice` next to `eq_without_inds` at ~line 2389; replace `_apply_wdm_add` at ~line 641)
- Modify: `src/lisatools/analysiscontainer.py` (`_slice_to_template` at ~line 708; add `_slice_wdm_to_template` after `_slice_stft_to_template`)
- Test: `tests/test_wdm_subbox.py`

**Interfaces:**
- Produces: `WDMSettings.get_slice(index: tuple[slice, slice]) -> WDMSettings` (index RELATIVE to the active box, contiguous slices only); `DomainBase.add_signal(template, sign)` accepting a WDM template whose active box is a sub-box of the target's; `AnalysisContainer._slice_wdm_to_template(template) -> (data_box, template, sens_box)` used by `template_likelihood`, `template_inner_product`, `template_snr`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_wdm_subbox.py
"""WDM sub-box templates: settings slicing, sub-box add/subtract, and
container slicing to a template's box (MBH batched windowed likelihood,
2026-09-29). CPU, toy grid."""
from __future__ import annotations

import copy
import pickle
import unittest

import numpy as np

from lisatools.analysiscontainer import AnalysisContainer
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix

NF, NT, DT = 32, 128, 10.0
N = NF * NT


def _chirp(lo_layer, hi_layer):
    """Three-channel chirp EXACTLY zero outside layers [lo_layer, hi_layer)."""
    from scipy.signal.windows import tukey

    t = np.arange(N) * DT
    x = np.zeros(N)
    lo, hi = lo_layer * NF, hi_layer * NF
    tt = t[lo:hi]
    x[lo:hi] = tukey(hi - lo, alpha=0.3) * np.sin(
        2 * np.pi * (4e-3 * tt + 0.5 * 1.6e-2 * (tt - tt[0]) ** 2 / (tt[-1] - tt[0]))
    )
    return np.stack([x, 0.5 * x, 0.25 * x])


def _wdm(**kw):
    return WDMSettings(NF, NT, DT, force_backend="cpu", **kw)


class WDMSettingsGetSliceTest(unittest.TestCase):
    def test_slice_narrows_box_and_round_trips(self):
        s = _wdm(min_freq=2e-3, max_freq=2e-2)
        sub = s.get_slice((slice(2, 6), slice(40, 80)))
        self.assertEqual((sub.ind_min_f, sub.ind_max_f), (s.ind_min_f + 2, s.ind_min_f + 5))
        self.assertEqual((sub.ind_min_t, sub.ind_max_t), (40, 79))
        self.assertTrue(sub.eq_without_inds(s))
        # every WDMSignal rebuilds its settings from (args, kwargs): the box must survive
        rebuilt = WDMSettings(*sub.args, **sub.kwargs)
        self.assertEqual(rebuilt, sub)
        again = pickle.loads(pickle.dumps(copy.deepcopy(sub)))
        self.assertEqual(again, sub)

    def test_slice_from_zero_is_allowed(self):
        sub = _wdm().get_slice((slice(0, NF), slice(0, 10)))
        self.assertEqual((sub.ind_min_f, sub.ind_min_t, sub.ind_max_t), (0, 0, 9))

    def test_bad_index_raises(self):
        s = _wdm()
        with self.assertRaises(ValueError):
            s.get_slice((slice(0, 4, 2), slice(0, 4)))
        with self.assertRaises(ValueError):
            s.get_slice((slice(3, 3), slice(0, 4)))
        with self.assertRaises(ValueError):
            s.get_slice(slice(0, 4))


class WDMSubBoxAddTest(unittest.TestCase):
    def test_add_then_subtract_sub_box_restores_residual(self):
        full = _wdm()
        rng = np.random.default_rng(1)
        data = WDMSignal(rng.normal(size=(3, NF, NT)), full)
        before = data.arr.copy()
        box = full.get_slice((slice(0, NF), slice(40, 80)))
        tmpl = WDMSignal(rng.normal(size=(3, NF, 40)), box)
        data.add_signal(tmpl, sign=+1)
        self.assertFalse(np.allclose(data.arr[..., 40:80], before[..., 40:80]))
        np.testing.assert_array_equal(data.arr[..., :40], before[..., :40])
        np.testing.assert_array_equal(data.arr[..., 80:], before[..., 80:])
        data.add_signal(tmpl, sign=-1)
        np.testing.assert_allclose(data.arr, before, rtol=0, atol=1e-12)

    def test_same_box_add_is_unchanged(self):
        full = _wdm()
        data = WDMSignal(np.zeros((3, NF, NT)), full)
        data.add_signal(WDMSignal(np.ones((3, NF, NT)), full), sign=+1)
        np.testing.assert_array_equal(data.arr, np.ones((3, NF, NT)))

    def test_box_outside_data_raises(self):
        full = _wdm(min_time=50 * NF * DT)  # data box starts at layer 50
        data = WDMSignal(np.zeros((3, NF, full.Nt_active)), full)
        box = _wdm().get_slice((slice(0, NF), slice(40, 80)))  # starts at 40 < 50
        with self.assertRaises(ValueError):
            data.add_signal(WDMSignal(np.zeros((3, NF, 40)), box))

    def test_shifted_grid_raises(self):
        data = WDMSignal(np.zeros((3, NF, NT)), _wdm())
        other = WDMSettings(NF, NT, DT, t0=12345.0, force_backend="cpu").get_slice(
            (slice(0, NF), slice(40, 80))
        )
        with self.assertRaises(ValueError):
            data.add_signal(WDMSignal(np.zeros((3, NF, 40)), other))


class SliceWDMToTemplateTest(unittest.TestCase):
    def test_sub_box_likelihood_equals_full_grid(self):
        full = _wdm(min_freq=2e-3)
        h_full = TDSignal(_chirp(45, 75), TDSettings(N, DT, force_backend="cpu")).transform(full)
        # template restricted to layers [40, 80): zero the rest on the full grid
        h_arr = np.array(h_full.arr, copy=True)
        h_arr[..., :40] = 0.0
        h_arr[..., 80:] = 0.0
        h_trunc = WDMSignal(h_arr, full)
        rng = np.random.default_rng(2)
        d = WDMSignal(h_arr + 1e-1 * rng.normal(size=h_arr.shape), full)
        ac = AnalysisContainer(d, XYZ2SensitivityMatrix(full, model="scirdv1"))
        ref_dh = complex(ac.template_inner_product(h_trunc))
        ref_ll = float(np.real(ac.template_likelihood(h_trunc)))

        box = full.get_slice((slice(0, full.Nf_active), slice(40, 80)))
        h_box = WDMSignal(h_arr[..., 40:80], box)
        d_box, t_box, s_box = ac._slice_to_template(h_box)
        self.assertEqual(d_box.arr.shape, h_box.arr.shape)
        self.assertEqual(tuple(s_box.invC.shape[-2:]), tuple(h_box.arr.shape[-2:]))
        self.assertIs(t_box, h_box)
        np.testing.assert_allclose(
            complex(ac.template_inner_product(h_box)), ref_dh, rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(
            float(np.real(ac.template_likelihood(h_box))), ref_ll, rtol=1e-12, atol=0
        )

    def test_box_outside_data_raises(self):
        full = _wdm(min_time=50 * NF * DT)
        d = WDMSignal(np.zeros((3, NF, full.Nt_active)), full)
        ac = AnalysisContainer(d, XYZ2SensitivityMatrix(full, model="scirdv1"))
        box = _wdm().get_slice((slice(0, NF), slice(40, 80)))
        with self.assertRaises(ValueError):
            ac._slice_to_template(WDMSignal(np.zeros((3, NF, 40)), box))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `nice -n 10 $PY -m unittest tests.test_wdm_subbox -v`
Expected: FAIL / ERROR with `AttributeError: 'WDMSettings' object has no attribute 'get_slice'` (or `NotImplementedError` from `get_slice` base) and `NotImplementedError: Automatic region slicing not yet implemented for WDMSettings`.

- [ ] **Step 3: Implement `WDMSettings.get_slice`**

In `src/lisatools/domains.py`, inside `class WDMSettings`, directly after `eq_without_inds`:

```python
    def get_slice(self, index: tuple) -> "WDMSettings":
        """Settings for a SUB-BOX of this settings' active box.

        ``index = (f_slice, t_slice)`` is RELATIVE to the active box -- the
        same index :meth:`DomainBase.get_array_slice` applies to the stored
        ``(Nf_active, Nt_active)`` array, so array and settings stay in step.
        The grid (``Nf``, ``Nt``, ``dt``, ``t0``, window) is unchanged; only
        ``ind_min/max_{f,t}`` narrow.

        The result is rebuilt through the PHYSICAL ``min/max_*`` setters,
        because every ``WDMSignal`` reconstructs its settings from
        ``(args, kwargs)`` and ``kwargs`` carries those inputs, not the
        indices: setting ``ind_*`` directly would be undone by the next
        reconstruction. Half-bin offsets make the setters' ceil/floor land
        exactly on the requested integers; the result is checked.
        """
        if not isinstance(index, tuple) or len(index) != 2:
            raise ValueError("WDMSettings.get_slice expects (f_slice, t_slice).")
        f_sl, t_sl = index
        for sl in (f_sl, t_sl):
            if not isinstance(sl, slice) or sl.step not in (None, 1):
                raise ValueError(
                    "WDMSettings.get_slice: slices must be contiguous (step 1)."
                )
        f_lo, f_hi, _ = f_sl.indices(int(self.Nf_active))
        t_lo, t_hi, _ = t_sl.indices(int(self.Nt_active))
        if f_hi <= f_lo or t_hi <= t_lo:
            raise ValueError(f"WDMSettings.get_slice: empty slice {index!r}.")
        want = (
            int(self.ind_min_f) + f_lo,
            int(self.ind_min_f) + f_hi - 1,
            int(self.ind_min_t) + t_lo,
            int(self.ind_min_t) + t_hi - 1,
        )
        kw = dict(self.kwargs)
        kw.update(
            min_freq=None if want[0] == 0 else (want[0] - 0.5) * self.layer_df,
            max_freq=(want[1] + 0.5) * self.layer_df,
            min_time=None if want[2] == 0 else (want[2] - 0.5) * self.layer_dt,
            max_time=(want[3] + 0.5) * self.layer_dt,
        )
        new = WDMSettings(*self.args, **kw)
        got = (int(new.ind_min_f), int(new.ind_max_f), int(new.ind_min_t), int(new.ind_max_t))
        if got != want:
            raise RuntimeError(
                f"WDMSettings.get_slice: rebuilt box {got} != requested {want}"
            )
        return new
```

- [ ] **Step 4: Implement the sub-box `_apply_wdm_add`**

Replace the existing `_apply_wdm_add` in `src/lisatools/domains.py` (module scope, ~line 641):

```python
def _apply_wdm_add(target, sign, template_arr, template_settings):
    """Add ``sign * template_arr`` (WDM) to ``target.arr``.

    Same active box: plain in-place add (unchanged behaviour). A template whose
    active box is a SUB-BOX of the target's -- same grid (``Nf``, ``Nt``,
    ``dt``, ``t0``), ``ind_min/max`` inside -- is added into that box (the MBH
    batched windowed templates, 2026-09-29). Anything else raises.
    """
    ts = template_settings
    same_origin = (
        int(ts.ind_min_f) == int(target.ind_min_f)
        and int(ts.ind_min_t) == int(target.ind_min_t)
    )
    if target.arr.shape[-2:] == template_arr.shape[-2:] and same_origin:
        target.arr[...] += sign * template_arr
        return
    if not target.eq_without_inds(ts) or abs(float(ts.t0) - float(target.t0)) > 1e-6 * float(target.data_dt):
        raise ValueError(
            "WDM add_signal: template and data must share the wavelet grid "
            f"(Nf, Nt, dt, t0); got template Nf={ts.Nf} Nt={ts.Nt} dt={ts.data_dt} "
            f"t0={ts.t0} vs data Nf={target.Nf} Nt={target.Nt} dt={target.data_dt} "
            f"t0={target.t0}."
        )
    f0 = int(ts.ind_min_f) - int(target.ind_min_f)
    f1 = int(ts.ind_max_f) - int(target.ind_min_f) + 1
    t0 = int(ts.ind_min_t) - int(target.ind_min_t)
    t1 = int(ts.ind_max_t) - int(target.ind_min_t) + 1
    if f0 < 0 or t0 < 0 or f1 > int(target.Nf_active) or t1 > int(target.Nt_active):
        raise ValueError(
            f"WDM add_signal: template box f[{ts.ind_min_f}:{ts.ind_max_f}] "
            f"t[{ts.ind_min_t}:{ts.ind_max_t}] is not inside the data box "
            f"f[{target.ind_min_f}:{target.ind_max_f}] "
            f"t[{target.ind_min_t}:{target.ind_max_t}]."
        )
    if tuple(template_arr.shape[-2:]) != (f1 - f0, t1 - t0):
        raise ValueError(
            f"WDM add_signal: template array {tuple(template_arr.shape[-2:])} does "
            f"not match its box {(f1 - f0, t1 - t0)}."
        )
    target.arr[..., f0:f1, t0:t1] += sign * template_arr
```

- [ ] **Step 5: Implement `_slice_wdm_to_template` and wire the dispatch**

In `src/lisatools/analysiscontainer.py`, in `_slice_to_template`, replace the final `else: raise NotImplementedError(...)` with:

```python
        elif isinstance(data_settings, domains.WDMSettings):
            return self._slice_wdm_to_template(template)
        else:
            raise NotImplementedError(
                f"Automatic region slicing not yet implemented for "
                f"{type(data_settings).__name__}. Ensure template and data "
                f"have the same shape, or use STFT domain."
            )
```

and add, after `_slice_stft_to_template`:

```python
    def _slice_wdm_to_template(
        self, template: DomainBase
    ) -> Tuple[DomainBase, DomainBase, SensitivityMatrixBase]:
        """WDM slice helper used by :meth:`_slice_to_template`.

        The template's active box must be a sub-box of the data's on the SAME
        wavelet grid (the MBH batched windowed templates). Returns the data and
        the sensitivity matrix restricted to that box (views, not copies) and
        the template itself, so ``inner_product`` sees matching shapes.
        """
        data_settings = self._data.settings
        templ_settings = template.settings
        if not data_settings.eq_without_inds(templ_settings) or abs(
            float(templ_settings.t0) - float(data_settings.t0)
        ) > 1e-6 * float(data_settings.data_dt):
            raise ValueError(
                "WDM template and data must share the wavelet grid (Nf, Nt, dt, "
                f"t0); got template Nf={templ_settings.Nf} Nt={templ_settings.Nt} "
                f"t0={templ_settings.t0} vs data Nf={data_settings.Nf} "
                f"Nt={data_settings.Nt} t0={data_settings.t0}."
            )
        f0 = int(templ_settings.ind_min_f) - int(data_settings.ind_min_f)
        f1 = int(templ_settings.ind_max_f) - int(data_settings.ind_min_f) + 1
        t0 = int(templ_settings.ind_min_t) - int(data_settings.ind_min_t)
        t1 = int(templ_settings.ind_max_t) - int(data_settings.ind_min_t) + 1
        if (
            f0 < 0 or t0 < 0
            or f1 > int(data_settings.Nf_active) or t1 > int(data_settings.Nt_active)
        ):
            raise ValueError(
                f"WDM template box f[{templ_settings.ind_min_f}:{templ_settings.ind_max_f}] "
                f"t[{templ_settings.ind_min_t}:{templ_settings.ind_max_t}] is not inside "
                f"the data box f[{data_settings.ind_min_f}:{data_settings.ind_max_f}] "
                f"t[{data_settings.ind_min_t}:{data_settings.ind_max_t}]."
            )
        box = (slice(f0, f1), slice(t0, t1))
        return self._data.get_array_slice(box), template, self.sens_mat.get_slice(box)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `nice -n 10 $PY -m unittest tests.test_wdm_subbox -v`
Expected: all 8 tests PASS. Then run the neighbours that touch the same code: `nice -n 10 $PY -m unittest tests.test_coarse_wdm tests.test_batched_likelihood` (49 + 19 OK).

- [ ] **Step 7: Stage**

```bash
git add src/lisatools/domains.py src/lisatools/analysiscontainer.py tests/test_wdm_subbox.py
# commit ONLY if Mike said "commit" this session:
# git commit -m "feat(wdm): sub-box templates -- WDMSettings.get_slice, sub-box add_signal, container slicing"
```

---

### Task 2: Segment sub-transform adapter `MBHWindowedWDMSignalGen`

**Files:**
- Modify: `src/lisatools/sources/batching.py` (append the class; add imports)
- Test: `tests/test_mbh_windowed_signal_gen.py`

**Interfaces:**
- Consumes: `WDMSettings.get_slice` (Task 1); `BatchedDomainSignalGen._stack`; `place_td_signal_on_grid`; `TDSignal.transform`; `lisatools.utils.utility.tukey(N, alpha, xp=None)`.
- Produces: `MBHWindowedWDMSignalGen(wave_gen, wdm_settings, nchannels=3, tukey_alpha=0.0)` with `set_window(n_start: int, Nt_keep: int, n_pad: int) -> None`, `window_key -> tuple | None`, `geometry -> dict | None`, and `__call__(*params, **kwargs) -> WDMSignal` (batched leading axis when given arrays; kept box `[n_start, n_start + Nt_keep)` on the run's grid). If `wave_gen` has `set_window(t_seg_abs, n_seg)` the adapter calls it with the SEGMENT start and sample count (Task 3's generator).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mbh_windowed_signal_gen.py
"""MBH windowed sub-transform adapter: kept layers equal the full-grid
transform to a pinned tolerance, the pad is load-bearing, rows stack.
Stub generator (no phentax) so the transform is what is under test."""
from __future__ import annotations

import unittest

import numpy as np

from lisatools.domains import TDSettings, TDSignal, WDMSettings

NF, NT, DT = 32, 128, 10.0
N = NF * NT


def _chirp(lo_layer, hi_layer):
    from scipy.signal.windows import tukey

    t = np.arange(N) * DT
    x = np.zeros(N)
    lo, hi = lo_layer * NF, hi_layer * NF
    tt = t[lo:hi]
    x[lo:hi] = tukey(hi - lo, alpha=0.3) * np.sin(
        2 * np.pi * (4e-3 * tt + 0.5 * 1.6e-2 * (tt - tt[0]) ** 2 / (tt[-1] - tt[0]))
    )
    return np.stack([x, 0.5 * x, 0.25 * x])


class _SegmentGen:
    """Stub generator returning (times, channels) of a TD signal on the data
    lattice: scalar ``amp`` -> ``(N,)`` / ``(3, N)``; array -> ``(B, N)`` /
    ``(B, 3, N)``, the ``compute_tdi_channels`` contract."""

    supports_batch = True

    def __init__(self, td_full):
        self.td_full = td_full
        self.n_calls = 0

    def compute_tdi_channels(self, amp, **kwargs):
        self.n_calls += 1
        t = np.arange(N) * DT
        if np.ndim(amp) == 0:
            return t, float(amp) * self.td_full
        a = np.asarray(amp, dtype=float)
        return np.broadcast_to(t, (a.size, N)).copy(), a[:, None, None] * self.td_full[None]


def _wdm():
    return WDMSettings(NF, NT, DT, force_backend="cpu")


class WindowedSignalGenTest(unittest.TestCase):
    def setUp(self):
        from lisatools.sources.batching import MBHWindowedWDMSignalGen

        self.full = _wdm()
        self.h_td = _chirp(45, 75)
        self.gen = _SegmentGen(self.h_td)
        self.sg = MBHWindowedWDMSignalGen(self.gen, self.full, nchannels=3, tukey_alpha=0.0)
        self.ref = TDSignal(self.h_td, TDSettings(N, DT, force_backend="cpu")).transform(self.full).arr
        self.scale = float(np.abs(self.ref).max())

    def _err(self, out):
        return float(np.abs(np.asarray(out.arr) - self.ref[..., 40:80]).max() / self.scale)

    def test_kept_layers_match_full_transform(self):
        self.sg.set_window(n_start=40, Nt_keep=40, n_pad=8)
        out = self.sg(1.0)
        self.assertEqual(out.arr.shape, (3, NF, 40))
        self.assertEqual((out.ind_min_t, out.ind_max_t), (40, 79))
        self.assertEqual(self.sg.window_key, (40, 40, 8))
        # measured 2026-09-29 spike: 1.2e-5 at 8 pad layers on this grid
        self.assertLess(self._err(out), 5e-5)

    def test_pad_is_load_bearing(self):
        self.sg.set_window(n_start=40, Nt_keep=40, n_pad=0)
        # measured: 2e-2 with no pad -- the discarded edge layers carry the error
        self.assertGreater(self._err(self.sg(1.0)), 1e-3)

    def test_batched_rows_stack_and_scale(self):
        self.sg.set_window(n_start=40, Nt_keep=40, n_pad=8)
        out = self.sg(np.array([1.0, 2.0, 0.5]))
        self.assertTrue(out.is_batched)
        self.assertEqual(out.arr.shape, (3, 3, NF, 40))
        np.testing.assert_allclose(out.arr[1], 2.0 * out.arr[0], rtol=1e-12, atol=0)
        self.assertEqual(self.gen.n_calls, 1)

    def test_single_row_batch_keeps_leading_axis(self):
        self.sg.set_window(n_start=40, Nt_keep=40, n_pad=8)
        out = self.sg(np.array([1.0]))
        self.assertTrue(out.is_batched)
        self.assertEqual(out.arr.shape, (1, 3, NF, 40))

    def test_window_must_be_set(self):
        with self.assertRaises(RuntimeError):
            self.sg(1.0)

    def test_window_outside_grid_raises(self):
        with self.assertRaises(ValueError):
            self.sg.set_window(n_start=2, Nt_keep=40, n_pad=8)
        with self.assertRaises(ValueError):
            self.sg.set_window(n_start=100, Nt_keep=40, n_pad=8)

    def test_odd_segment_is_made_even(self):
        self.sg.set_window(n_start=40, Nt_keep=41, n_pad=8)
        g = self.sg.geometry
        self.assertEqual(g["Nt_seg"] % 2, 0)
        self.assertEqual(g["Nt_keep"], 41)
        self.assertEqual(self.sg(1.0).arr.shape, (3, NF, 41))

    def test_forwards_set_window_to_generator(self):
        calls = []

        class _Gen(_SegmentGen):
            def set_window(self, t_seg_abs, n_seg):
                calls.append((t_seg_abs, n_seg))

        from lisatools.sources.batching import MBHWindowedWDMSignalGen

        sg = MBHWindowedWDMSignalGen(_Gen(self.h_td), self.full, nchannels=3)
        sg.set_window(n_start=40, Nt_keep=40, n_pad=8)
        self.assertEqual(calls, [(32 * NF * DT, NF * 56)])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `nice -n 10 $PY -m unittest tests.test_mbh_windowed_signal_gen -v`
Expected: `ImportError: cannot import name 'MBHWindowedWDMSignalGen'`.

- [ ] **Step 3: Implement the adapter**

Append to `src/lisatools/sources/batching.py` (and extend the imports at the top: `from ..domains import DomainBase, DomainBaseArray, TDSettings, WDMSettings, WDMSignal, place_td_signal_on_grid` and `from ..utils.utility import get_array_module, tukey`; add the class name to `__all__`):

```python
class MBHWindowedWDMSignalGen(BatchedDomainSignalGen):
    """Batched ``signal_gen`` whose per-row domain step is a SEGMENT transform.

    Each row's TD channels are placed on a segment of the data lattice that
    is ``n_pad`` WDM layers wider than the kept box on each side, transformed
    on a segment ``WDMSettings`` (same ``Nf``, ``dt``; ``Nt_seg`` layers), and
    the ``n_pad`` edge layers are discarded. The result is a ``WDMSignal`` on
    the RUN's settings with ``active_slice_t = [n_start, n_start + Nt_keep)``,
    so containers slice their residual and PSD to it (``_slice_to_template``)
    and fills add it into the full residual (``add_signal``). The template is
    zero at both segment ends (onset ramp inside the box; ringdown dead), so
    the periodic segment transform's wrap-around touches only the discarded
    pad layers -- measured on a toy grid: kept-layer relative error 1.2e-5 at
    8 pad layers, 9e-7 at 16 (tests/test_mbh_windowed_signal_gen.py).

    Args:
        wave_gen: generator exposing ``compute_tdi_channels`` (times on the
            data lattice, ABSOLUTE seconds). If it also exposes
            ``set_window(t_seg_abs, n_seg)`` it is told the segment.
        wdm_settings: the run's :class:`WDMSettings` (device-local).
        nchannels: TDI channels the data carries (leading channels kept).
        tukey_alpha: the run's full-length data window alpha; the matching
            slice of that window multiplies the segment (0 -> none).
    """

    def __init__(self, wave_gen, wdm_settings, nchannels: int = 3, tukey_alpha: float = 0.0):
        super().__init__(wave_gen)
        self.wdm = wdm_settings
        self.nchannels = int(nchannels)
        self.tukey_alpha = float(tukey_alpha or 0.0)
        self.geometry = None
        self._seg_td = None
        self._seg_wdm = None
        self._box = None
        self._win_seg = None

    @property
    def window_key(self):
        g = self.geometry
        return None if g is None else (g["n_start"], g["Nt_keep"], g["n_pad"])

    def set_window(self, n_start: int, Nt_keep: int, n_pad: int) -> None:
        n_start, Nt_keep, n_pad = int(n_start), int(Nt_keep), int(n_pad)
        if Nt_keep < 1 or n_pad < 0:
            raise ValueError(f"set_window: Nt_keep={Nt_keep} n_pad={n_pad}")
        Nt_seg = Nt_keep + 2 * n_pad
        n_pad_hi = n_pad
        if Nt_seg % 2:  # WDMSettings needs an even layer count
            Nt_seg += 1
            n_pad_hi += 1
        s0 = n_start - n_pad
        Nt = int(self.wdm.Nt)
        if s0 < 0 or s0 + Nt_seg > Nt:
            raise ValueError(
                f"segment layers [{s0}, {s0 + Nt_seg}) fall outside the WDM grid [0, {Nt})"
            )
        rel_t0 = n_start - int(self.wdm.ind_min_t)
        if rel_t0 < 0 or rel_t0 + Nt_keep > int(self.wdm.Nt_active):
            raise ValueError(
                f"kept layers [{n_start}, {n_start + Nt_keep}) fall outside the data's "
                f"active box [{self.wdm.ind_min_t}, {self.wdm.ind_max_t + 1})"
            )
        Nf = int(self.wdm.Nf)
        dt = float(self.wdm.data_dt)
        layer_dt = float(self.wdm.layer_dt)
        t_seg = float(self.wdm.t0) + s0 * layer_dt
        backend = self.wdm.backend
        self._seg_td = TDSettings(Nf * Nt_seg, dt, t0=t_seg, force_backend=backend)
        self._seg_wdm = WDMSettings(
            Nf, Nt_seg, dt, t0=t_seg, oversample=self.wdm.oversample,
            min_freq=self.wdm.min_freq, max_freq=self.wdm.max_freq,
            is_complex=self.wdm.is_complex, force_backend=backend,
        )
        if (
            int(self._seg_wdm.Nf_active) != int(self.wdm.Nf_active)
            or int(self._seg_wdm.ind_min_f) != int(self.wdm.ind_min_f)
        ):
            raise RuntimeError(
                "segment WDM settings do not reproduce the run's active frequency layers"
            )
        self._box = self.wdm.get_slice(
            (slice(0, int(self.wdm.Nf_active)), slice(rel_t0, rel_t0 + Nt_keep))
        )
        if self.tukey_alpha > 0.0:
            w = tukey(int(self.wdm.N), self.tukey_alpha, xp=np)
            self._win_seg = self.wdm.xp.asarray(w[s0 * Nf:(s0 + Nt_seg) * Nf])
        else:
            self._win_seg = None
        self.geometry = dict(
            n_start=n_start, Nt_keep=Nt_keep, n_pad=n_pad, n_pad_hi=n_pad_hi,
            s0=s0, Nt_seg=Nt_seg, t_seg=t_seg,
        )
        if hasattr(self.wave_gen, "set_window"):
            self.wave_gen.set_window(t_seg, Nf * Nt_seg)

    def _to_domain(self, times, channels):
        if self.geometry is None:
            raise RuntimeError(
                "MBHWindowedWDMSignalGen: call set_window() before generating"
            )
        g = self.geometry
        placed = place_td_signal_on_grid(
            channels[: self.nchannels], self._seg_td, times=times
        )
        seg = placed.transform(self._seg_wdm, window=self._win_seg)
        kept = seg.arr[..., g["n_pad"]: g["n_pad"] + g["Nt_keep"]]
        return WDMSignal(kept, self._box)

    def __call__(self, *params, **kwargs):
        if self.geometry is None:
            raise RuntimeError(
                "MBHWindowedWDMSignalGen: call set_window() before generating"
            )
        times, channels = self.wave_gen.compute_tdi_channels(*params, **kwargs)
        if getattr(times, "ndim", 1) == 1:
            return self._to_domain(times, channels)
        n_src = int(times.shape[0])
        squeezed = channels.ndim == times.ndim
        return self._stack([
            self._to_domain(times[i], channels if squeezed else channels[i])
            for i in range(n_src)
        ])
```

Note for the implementer: `self.wdm.backend` is what `WDMSettings.kwargs` itself passes as `force_backend`, so it is the right object to forward. If `WDMSettings(...)` rejects it, pass `self.wdm.kwargs["force_backend"]` instead.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `nice -n 10 $PY -m unittest tests.test_mbh_windowed_signal_gen -v`
Expected: 8 PASS. If `test_kept_layers_match_full_transform` reports an error above 5e-5, the segment `WDMSettings` differ from the full ones in `min_freq/max_freq/oversample` handling; compare `self._seg_wdm.window` against `WDMSettings(NF, 56, DT).window` before changing the tolerance.

- [ ] **Step 5: Stage**

```bash
git add src/lisatools/sources/batching.py tests/test_mbh_windowed_signal_gen.py
# git commit -m "feat(mbh): segment sub-transform adapter for windowed batched MBH templates"
```

---

### Task 3: `WindowedGridAlignedMBHWaveform` (per-leaf lattice)

**Files:**
- Modify: `src/lisatools/sources/bbh/gridaligned.py` (append class; extend `__all__`)
- Modify: `src/lisatools/sources/bbh/__init__.py` (export)
- Test: append `WindowedGridAlignedPhentaxTest` to `tests/test_mbh_windowed_signal_gen.py`

**Interfaces:**
- Consumes: `GridAlignedPhenomTHMTDIWaveform._common_grid_spec(T) -> (k0, n_grid)`, `self.dt`, `self.waveform_t0`, `self.tdi_buffer_time`.
- Produces: `WindowedGridAlignedMBHWaveform.set_window(t_seg_abs: float, n_seg: int) -> None`, `window_spec -> (k0, n_grid) | None`; `_common_grid_spec` raises `RuntimeError` until `set_window` is called.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mbh_windowed_signal_gen.py`:

```python
def _phentax_available():
    try:
        import jax  # noqa: F401
        import phentax  # noqa: F401
        from lisatools.sources.bbh import waveform as _w
        return _w.jax is not None
    except Exception:
        return False


@unittest.skipUnless(_phentax_available(), "needs jax + phentax")
class WindowedGridAlignedPhentaxTest(unittest.TestCase):
    """Real phentax on a tiny CPU grid: 2 days at dt = 10 s, a 6e6 Msun binary
    merging at day 1, generation window T = 12 h, kept box ~19 h."""

    DT = 10.0
    NF, NT = 96, 180          # N = 17280 samples = 2 days
    T_GEN = 43200.0           # phentax T: 12 h before merger
    # waveform basis: m1 m2 s1z s2z dist[Mpc] phi_ref iota psi alpha delta t_plunge
    ROW = np.array([4.0e6, 2.0e6, 0.3, 0.3, 5000.0, 0.3, 0.9, 0.4, 1.0, 0.2, 86400.0])

    @classmethod
    def setUpClass(cls):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform
        from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
        from lisatools.sources.batching import MBHWindowedWDMSignalGen

        cls.wdm = WDMSettings(cls.NF, cls.NT, cls.DT, t0=0.0, min_freq=1e-4, max_freq=2.5e-2, force_backend="cpu")
        common = dict(
            waveform_kwargs=dict(higher_modes=[21, 33, 44], include_negative_modes=True,
                                 t_low_fit=True, coarse_grain=False, atol=1e-12, rtol=1e-12),
            Tobs=cls.T_GEN, start_freq=7e-5, use_reference_time=True,
            waveform_t0=0.0,
            data_td_settings=TDSettings(cls.NF * cls.NT, cls.DT, t0=0.0, force_backend="cpu"),
            tdi_generation="2nd generation", tdi_channels="XYZ",
            sampling_frequency=1.0 / cls.DT,
            orbits=EqualArmlengthOrbits(force_backend="cpu"),
            order=8, tukey_alpha=0.0, stft_dt=None, freq_min=1e-4, freq_max=2.5e-2,
            fft_batch_size=1, buffer_time=15000.0,
            output_domain_settings=cls.wdm, force_backend="cpu",
        )
        cls.windowed = WindowedGridAlignedMBHWaveform(**common)
        cls.stock = PhenomTHMTDIWaveform(**common)
        cls.adapter = MBHWindowedWDMSignalGen(cls.windowed, cls.wdm, nchannels=3, tukey_alpha=0.0)
        # kept box: 12 h before + 4 h after merger + 1 h margins; pad 1 h (4 layers of 960 s)
        layer = cls.NF * cls.DT
        n_start = int(np.floor((86400.0 - cls.T_GEN - 3600.0) / layer))
        Nt_keep = int(np.ceil((cls.T_GEN + 4 * 3600.0 + 2 * 3600.0) / layer)) + 1
        if (Nt_keep + 8) % 2:
            Nt_keep += 1
        cls.adapter.set_window(n_start=n_start, Nt_keep=Nt_keep, n_pad=4)
        cls.n_start, cls.Nt_keep = n_start, Nt_keep

    def test_refuses_without_window(self):
        from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform

        gen = WindowedGridAlignedMBHWaveform.__new__(WindowedGridAlignedMBHWaveform)
        with self.assertRaises(RuntimeError):
            WindowedGridAlignedMBHWaveform._common_grid_spec(gen, self.T_GEN)

    def test_batched_rows_share_the_lattice_and_scale_with_distance(self):
        rows = np.stack([self.ROW, self.ROW])
        rows[1, 4] *= 2.0
        times, ch = self.windowed.compute_tdi_channels(*rows.T)
        self.assertEqual(times.ndim, 2)
        np.testing.assert_array_equal(times[0], times[1])
        np.testing.assert_allclose(np.diff(times[0]), self.DT, rtol=0, atol=1e-9)
        self.assertEqual(ch.shape[:2], (2, 3))
        m = np.abs(ch[0]).max()
        np.testing.assert_allclose(ch[1], 0.5 * ch[0], rtol=0, atol=1e-6 * m)

    def test_single_equals_batched_row(self):
        t_b, ch_b = self.windowed.compute_tdi_channels(*np.stack([self.ROW, self.ROW]).T)
        t_s, ch_s = self.windowed.compute_tdi_channels(*self.ROW)
        np.testing.assert_array_equal(t_s, t_b[0])
        np.testing.assert_allclose(ch_s, ch_b[0], rtol=1e-10, atol=0)

    def test_windowed_template_tracks_the_stock_template(self):
        out = self.adapter(*np.stack([self.ROW]).T)
        self.assertEqual(out.arr.shape, (1, 3, self.wdm.Nf_active, self.Nt_keep))
        ref = self.stock.get_signals_for_residuals(*self.ROW)
        ref_box = np.asarray(ref.arr)[..., self.n_start: self.n_start + self.Nt_keep]
        scale = float(np.abs(np.asarray(ref.arr)).max())
        rel = float(np.abs(np.asarray(out.arr[0]) - ref_box).max() / scale)
        outside = float(np.abs(np.delete(np.asarray(ref.arr), np.s_[self.n_start: self.n_start + self.Nt_keep], axis=-1)).max() / scale)
        print(f"[windowed vs stock] kept-box max rel diff {rel:.3e}; stock power outside box {outside:.3e}")
        self.assertLess(rel, 1e-2)
        self.assertLess(outside, 1e-2)
```

- [ ] **Step 2: Run to verify failure**

Run: `nice -n 10 $PY -m unittest tests.test_mbh_windowed_signal_gen.WindowedGridAlignedPhentaxTest -v`
Expected: `ImportError: cannot import name 'WindowedGridAlignedMBHWaveform'` (or skip if phentax is missing: then confirm `$PY -c "import phentax"` works in `deving`; it did on 2026-09-29).

- [ ] **Step 3: Implement the class**

Append to `src/lisatools/sources/bbh/gridaligned.py`:

```python
class WindowedGridAlignedMBHWaveform(GridAlignedPhenomTHMTDIWaveform):
    """Grid-aligned generation on a PER-LEAF window instead of the analysis window.

    The parent's shared lattice spans the whole analysis window
    (``domain_settings.N``) -- six months of samples for a source that lives
    for 90 days. The global fit sets a window around each leaf's merger
    (user ruling 2026-09-29: 90 days before to 10 days after, plus pads) and
    this class evaluates the batch on THAT lattice. Everything else -- the
    exact integer lattice, the split merger time, ``merger_time = 0`` handed
    to ``_apply_response`` -- is the parent's.

    ``set_window`` takes the ABSOLUTE segment start (a data-lattice time that
    is also a WDM layer boundary) and the segment's sample count; the lattice
    is prepended with ``n_lead`` samples so the response's retarded reads
    (up to ``tdi_buffer_time`` before the segment) fall on generated samples
    and the invalid head lands outside the segment, where the placement
    clips it. Generating without a window is refused: silently falling back
    to the analysis-window lattice is exactly the 6-month generation this
    class exists to avoid.
    """

    _window_spec = None

    def set_window(self, t_seg_abs: float, n_seg: int) -> None:
        dt = float(self.dt)
        rel = (float(t_seg_abs) - float(self.waveform_t0)) / dt
        k_seg = int(np.rint(rel))
        if abs(rel - k_seg) > 1e-6:
            raise ValueError(
                f"segment start {t_seg_abs!r} is not on the waveform lattice "
                f"(waveform_t0 + k*dt): residual {(rel - k_seg) * dt:.3e} s"
            )
        n_lead = int(np.ceil(self.tdi_buffer_time / dt)) + 1
        self._window_spec = (k_seg - n_lead, n_lead + int(n_seg))

    @property
    def window_spec(self):
        return self._window_spec

    def _common_grid_spec(self, T):
        if self._window_spec is None:
            raise RuntimeError(
                "WindowedGridAlignedMBHWaveform.set_window() was never called; "
                "refusing to fall back to the full analysis-window lattice."
            )
        return self._window_spec
```

Update `__all__` in the module to `["GridAlignedPhenomTHMTDIWaveform", "WindowedGridAlignedMBHWaveform"]` and add the export to `src/lisatools/sources/bbh/__init__.py` next to the existing `GridAlignedPhenomTHMTDIWaveform` import.

- [ ] **Step 4: Run the tests**

Run: `nice -n 10 $PY -m unittest tests.test_mbh_windowed_signal_gen -v`
Expected: all PASS (the phentax class runs in well under a minute on CPU). If phentax refuses the 12 h window for this mass ("t_min" / domain errors), raise `T_GEN` to 86400 and `NT` to 360 (4-day grid) and adjust `n_start`; keep the assertions. If `test_windowed_template_tracks_the_stock_template` exceeds 1e-2, print both arrays' argmax columns: an off-by-one-layer window (`n_start`) or an unsnapped `waveform_t0` are the two known ways to fail it.

- [ ] **Step 5: Stage**

```bash
git add src/lisatools/sources/bbh/gridaligned.py src/lisatools/sources/bbh/__init__.py tests/test_mbh_windowed_signal_gen.py
# git commit -m "feat(mbh): WindowedGridAlignedMBHWaveform -- grid-aligned batch on a per-leaf lattice"
```

---

### Task 4: `MBHBatchedLikeMove`

**Files:**
- Create: `src/lisatools/globalfit/moves/mbhbatchedmove.py`
- Modify: `src/lisatools/globalfit/moves/__init__.py` (export next to `SOBBHChunkedLikeMove`)
- Test: `tests/test_mbh_batched_move.py`

**Interfaces:**
- Consumes: `ResidualAddOneRemoveOneMove` (base positionals `branch_name, coords_shape, waveform_gen, waveform_gen_kwargs, waveform_like_kwargs, acs, num_repeats, transform_fn, priors, inner_moves`, kwargs `betas_all`, `name`, `dcga`), its `compute_acs_like`, `_resolve_signal_gen_override`, `_branch_waveform_kwargs`, `_dbg_prefix`, `check_ll_mode`, `_current_leaf`, `compute_check_like`; `AnalysisContainerArray.likelihood()`, `.signal_operation(sign, templates, data_index)`, `.apply_signal_from_params(...)`, `.linear_data_arr`, `.gpus`, `.xp`, `.acs`; `shard_lookup_maps(aca)`; `device_context(xp, device)`; `MBHWindowedWDMSignalGen` (Task 2); `AnalysisContainer._slice_to_template` (Task 1); `inner_product`.
- Produces: `mbh_window_layers(wdm, t_merge_abs, window_before, window_after, window_pad, window_margin) -> dict(n_start, Nt_keep, n_pad)`; `MBHBatchedLikeMove(*base_args, batched_gen=..., batch_max_size=16, window_before=..., window_after=..., window_pad=..., window_margin=..., **base_kwargs)`. `batched_gen` must expose `set_window(n_start, Nt_keep, n_pad)`, `window_key`, `waveform_t0`, `t_plunge_snap`, `__call__(*cols, **gen_kwargs) -> WDMSignal`; a `DeviceLocalWaveGen` resolving to an adapter per device also works (attribute access resolves per device).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mbh_batched_move.py
"""MBHBatchedLikeMove: batched windowed scoring against per-walker residuals
AND per-walker PSDs equals the base container path; batched expose/fold
restores the residual; chunking, hysteresis, routing, fallback."""
from __future__ import annotations

import unittest

import numpy as np

from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal
from lisatools.sensitivity import XYZ2SensitivityMatrix

NF, NT, DT = 32, 128, 10.0
N = NF * NT
LAYER = NF * DT
NWALKERS, NTEMPS = 3, 2
T0 = 0.0  # data start == waveform_t0 (already on the lattice)
# waveform basis: m1 m2 s1z s2z dist phi_ref iota psi alpha delta t_plunge
BASE_ROW = np.array([1.0e6, 5.0e5, 0.1, 0.2, 2.0e3, 0.3, 0.9, 0.4, 1.0, 0.2, 64 * LAYER])
WINDOW = dict(window_before=6 * LAYER, window_after=2 * LAYER, window_pad=2 * LAYER, window_margin=1 * LAYER)


def _model_td(row):
    """Toy 'MBH': three channels of a Gaussian-enveloped chirp centred at
    t = waveform_t0 + t_plunge, amplitude m1/dist, inclination in channel 1."""
    m1, m2, s1z, s2z, dist, phi_ref, inc, psi, alpha, delta, t_plunge = [float(v) for v in row]
    t = np.arange(N) * DT + T0
    tc = T0 + t_plunge
    env = np.exp(-0.5 * ((t - tc) / (1.5 * LAYER)) ** 2)
    amp = 3.0e-3 * (m1 / 1.0e6) * (2.0e3 / dist)
    ph = 2 * np.pi * (4e-3 * (t - tc) + 0.5 * 2e-6 * (t - tc) ** 2) + phi_ref
    x = amp * env * np.cos(ph)
    return np.stack([x, (0.5 + 0.1 * inc) * x, 0.25 * np.cos(psi) * x])


class _FastGen:
    """compute_tdi_channels contract on the full lattice; the adapter clips."""

    supports_batch = True
    waveform_t0 = T0
    t_plunge_snap = 0.0

    def __init__(self):
        self.n_calls = 0
        self.fail_with = None

    def compute_tdi_channels(self, *cols, **kwargs):
        self.n_calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        rows = np.stack([np.atleast_1d(np.asarray(c, dtype=float)) for c in cols], axis=1)
        t = np.arange(N) * DT + T0
        ch = np.stack([_model_td(r) for r in rows])
        if np.ndim(cols[0]) == 0:
            return t, ch[0]
        return np.broadcast_to(t, (rows.shape[0], N)).copy(), ch


def _slow_gen(*params, apply_transform=False, leaf_inds=None, **kwargs):
    """The containers' installed (slow, full-grid) generator: same model."""
    return TDSignal(_model_td(params), TDSettings(N, DT, t0=T0, force_backend="cpu")).transform(_slow_gen.wdm)


def _build():
    from lisatools.sources.batching import MBHWindowedWDMSignalGen

    wdm = WDMSettings(NF, NT, DT, t0=T0, min_freq=2e-3, max_freq=2e-2, force_backend="cpu")
    _slow_gen.wdm = wdm
    rng = np.random.default_rng(7)
    acs_list = []
    models = ["scirdv1", "mrdv1", "scirdv1"]      # walker 1 has a DIFFERENT PSD
    for w in range(NWALKERS):
        noise = 2e-4 * rng.normal(size=(3, wdm.Nf_active, NT))
        ac = AnalysisContainer(WDMSignal(noise, wdm), XYZ2SensitivityMatrix(wdm, model=models[w]))
        ac.signal_gen = {"mbh": _slow_gen}
        acs_list.append(ac)
    acs = AnalysisContainerArray(acs_list)
    fast = _FastGen()
    adapter = MBHWindowedWDMSignalGen(fast, wdm, nchannels=3, tukey_alpha=0.0)
    return acs, adapter, fast, wdm


def _build_move(acs, adapter, batch_max_size=2, **window):
    from eryn.moves import StretchMove
    from eryn.prior import ProbDistContainer, uniform_dist
    from lisatools.globalfit.moves import MBHBatchedLikeMove

    betas = 1 / 1.2 ** np.arange(NTEMPS)
    priors = {"mbh": ProbDistContainer({i: uniform_dist(-1e10, 1e10) for i in range(11)})}
    kw = dict(WINDOW)
    kw.update(window)
    move = MBHBatchedLikeMove(
        "mbh", (NTEMPS, NWALKERS, 1, 11), None, {}, {}, acs, 1, None, priors,
        [(StretchMove(), 1.0)], betas_all=np.tile(betas, (1, 1)),
        batched_gen=adapter, batch_max_size=batch_max_size, name="mbh batched test", **kw,
    )
    move._current_leaf = 0
    return move


def _cold_rows(seed=3):
    rng = np.random.default_rng(seed)
    rows = np.tile(BASE_ROW, (NWALKERS, 1))
    rows[:, 4] *= rng.uniform(0.8, 1.2, NWALKERS)          # dist
    rows[:, 10] += rng.uniform(-0.2, 0.2, NWALKERS) * LAYER  # t_plunge
    return rows


class WindowLayersTest(unittest.TestCase):
    def test_geometry_is_a_function_of_durations_only(self):
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        wdm = WDMSettings(NF, NT, DT, t0=T0, force_backend="cpu")
        a = mbh_window_layers(wdm, T0 + 64 * LAYER, **WINDOW)
        b = mbh_window_layers(wdm, T0 + 70 * LAYER, **WINDOW)
        self.assertEqual(a["Nt_keep"], b["Nt_keep"])
        self.assertEqual(a["n_pad"], 2)
        self.assertEqual(b["n_start"] - a["n_start"], 6)
        self.assertEqual((a["Nt_keep"] + 2 * a["n_pad"]) % 2, 0)
        # kept box covers [t - before - margin, t + after + margin]
        self.assertLessEqual(a["n_start"] * LAYER, 64 * LAYER - 7 * LAYER)
        self.assertGreaterEqual((a["n_start"] + a["Nt_keep"]) * LAYER, 64 * LAYER + 3 * LAYER)

    def test_window_clamps_at_grid_edges(self):
        from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers

        wdm = WDMSettings(NF, NT, DT, t0=T0, force_backend="cpu")
        lo = mbh_window_layers(wdm, T0 + 1 * LAYER, **WINDOW)
        self.assertEqual(lo["n_start"], lo["n_pad"])
        hi = mbh_window_layers(wdm, T0 + 127 * LAYER, **WINDOW)
        self.assertEqual(hi["n_start"] + hi["Nt_keep"] + hi["n_pad"], NT)
        with self.assertRaises(ValueError):
            mbh_window_layers(wdm, T0 + 64 * LAYER, window_before=100 * LAYER, window_after=100 * LAYER, window_pad=LAYER, window_margin=0.0)


class MBHBatchedParityTest(unittest.TestCase):
    def setUp(self):
        self.acs, self.adapter, self.fast, self.wdm = _build()
        self.move = _build_move(self.acs, self.adapter)
        self.cold = _cold_rows()
        self.move.remove_cold_chain_sources(self.cold)     # expose (sets the window)
        self.move.setup_likelihood_here(self.cold)

    def _rows(self, n, seed=11):
        rng = np.random.default_rng(seed)
        rows = np.tile(BASE_ROW, (n, 1))
        rows[:, 4] *= rng.uniform(0.7, 1.3, n)
        rows[:, 10] += rng.uniform(-0.3, 0.3, n) * LAYER
        rows[:, 5] += rng.uniform(-1, 1, n)
        return rows

    def test_batched_matches_container_path_per_walker_psd(self):
        rows = self._rows(6)
        idx = np.array([0, 1, 2, 1, 0, 1])
        fast = self.move.compute_like(rows, idx)
        slow = self.move.compute_acs_like(rows, idx)
        print(f"[mbh batched parity] max |fast-slow| = {np.abs(fast - slow).max():.3e} on lnL ~ {np.abs(slow).max():.3e}")
        np.testing.assert_allclose(fast, slow, rtol=0, atol=5e-2)
        self.assertEqual(self.move.n_batch_fallbacks, 0)
        self.assertTrue(np.all(np.isfinite(self.move._last_d_h)))
        # walker 1's PSD differs: scoring the same row against walker 0 vs 1 must differ
        same = np.tile(rows[:1], (2, 1))
        vals = self.move.compute_like(same, np.array([0, 1]))
        self.assertNotAlmostEqual(vals[0] - self.move._exposed_offset[0], vals[1] - self.move._exposed_offset[1], places=3)

    def test_chunk_size_does_not_change_results(self):
        rows = self._rows(5)
        idx = np.array([0, 1, 2, 0, 1])
        a = self.move.compute_like(rows, idx)
        big = _build_move(self.acs, self.adapter, batch_max_size=16)
        big._exposed_offset = self.move._exposed_offset
        big._leaf_windows = self.move._leaf_windows
        b = big.compute_like(rows, idx)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)

    def test_non_finite_row_gets_sentinel(self):
        rows = self._rows(2)
        rows[1, 0] = np.nan
        out = self.move.compute_like(rows, np.array([0, 1]))
        self.assertTrue(np.isfinite(out[0]))
        self.assertEqual(out[1], -1e300)

    def test_compute_like_requires_armed_offset(self):
        move = _build_move(self.acs, self.adapter)
        with self.assertRaises(RuntimeError):
            move.compute_like(self._rows(1), np.array([0]))

    def test_verify_prev_logl_passes_at_default_tolerance(self):
        rows = self._rows(NWALKERS * NTEMPS)
        idx = np.tile(np.arange(NWALKERS), NTEMPS)
        prev = self.move.compute_like(rows, idx).reshape(NTEMPS, NWALKERS)
        self.move._verify_prev_logl(prev, rows, idx, 0)   # must not raise/warn

    def test_batch_refusal_falls_back_to_container_path(self):
        from lisatools.utils.exceptions import BatchNotLaunchable

        rows = self._rows(3)
        idx = np.array([0, 1, 2])
        slow = self.move.compute_acs_like(rows, idx)
        self.fast.fail_with = BatchNotLaunchable("test refusal")
        out = self.move.compute_like(rows, idx)
        np.testing.assert_allclose(out, slow, rtol=0, atol=0)
        self.assertEqual(self.move.n_batch_fallbacks, 2)   # one per chunk of 2
        self.fast.fail_with = None


class MBHBatchedFillTest(unittest.TestCase):
    def test_expose_then_fold_restores_residual(self):
        acs, adapter, fast, wdm = _build()
        move = _build_move(acs, adapter)
        before = [np.array(ac.data.arr, copy=True) for ac in acs.acs.flatten()]
        cold = _cold_rows()
        move.remove_cold_chain_sources(cold)
        changed = [not np.allclose(ac.data.arr, b) for ac, b in zip(acs.acs.flatten(), before)]
        self.assertTrue(all(changed))
        move.add_back_in_cold_chain_sources(cold)
        for ac, b in zip(acs.acs.flatten(), before):
            np.testing.assert_allclose(ac.data.arr, b, rtol=0, atol=1e-12 * np.abs(b).max())
        self.assertEqual(fast.n_calls, 4)   # 3 walkers at batch 2 -> 2 chunks per pass

    def test_window_hysteresis(self):
        acs, adapter, fast, wdm = _build()
        move = _build_move(acs, adapter)
        cold = _cold_rows()
        move.remove_cold_chain_sources(cold)
        first = dict(move._leaf_windows[0])
        moved = cold.copy()
        moved[:, 10] += 0.5 * LAYER              # inside the 1-layer margin
        move.add_back_in_cold_chain_sources(moved)
        self.assertEqual(move._leaf_windows[0]["n_start"], first["n_start"])
        far = cold.copy()
        far[:, 10] += 5 * LAYER                  # outside the margin
        move.add_back_in_cold_chain_sources(far)  # fold never rebuilds
        self.assertEqual(move._leaf_windows[0]["n_start"], first["n_start"])
        move.remove_cold_chain_sources(far)       # expose rebuilds
        self.assertEqual(move._leaf_windows[0]["n_start"], first["n_start"] + 5)

    def test_dense_fill_knob(self):
        import os

        acs, adapter, fast, wdm = _build()
        move = _build_move(acs, adapter)
        os.environ["MBH_BATCHED_FILL"] = "0"
        try:
            move.remove_cold_chain_sources(_cold_rows())
        finally:
            del os.environ["MBH_BATCHED_FILL"]
        self.assertEqual(fast.n_calls, 0)


class MBHBatchedRoutingTest(unittest.TestCase):
    def test_split_rows_by_shard(self):
        from lisatools.globalfit.moves.mbhbatchedmove import MBHBatchedLikeMove

        class _Holder:
            linear_data_arr = [object(), object()]
            gpus = [0, 1]
            gpu_splits = [np.array([0, 2]), np.array([1])]
            split_map = np.array([0, 1, 0])
            acs_total_entries = 3

        groups = MBHBatchedLikeMove._split_rows_static(_Holder(), np.array([2, 1, 0, 1]))
        self.assertEqual([g[0] for g in groups], [0, 1])
        np.testing.assert_array_equal(groups[0][1], [0, 2])
        np.testing.assert_array_equal(groups[1][1], [1, 3])

    def test_ctor_guards(self):
        acs, adapter, fast, wdm = _build()
        with self.assertRaises(ValueError):
            _build_move(acs, None)
        from eryn.moves import StretchMove
        from eryn.prior import ProbDistContainer, uniform_dist
        from lisatools.globalfit.moves import MBHBatchedLikeMove

        priors = {"mbh": ProbDistContainer({i: uniform_dist(-1e10, 1e10) for i in range(11)})}
        with self.assertRaises(ValueError):
            MBHBatchedLikeMove("mbh", (NTEMPS, NWALKERS, 1, 11), None, {}, {}, acs, 1, None, priors,
                               [(StretchMove(), 1.0)], batched_gen=adapter, dcga=object(), **WINDOW)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `nice -n 10 $PY -m unittest tests.test_mbh_batched_move -v`
Expected: `ImportError: cannot import name 'MBHBatchedLikeMove'`.

- [ ] **Step 3: Implement the move**

Create `src/lisatools/globalfit/moves/mbhbatchedmove.py`:

```python
"""MBH add/remove move scored by the batched, windowed grid-aligned likelihood.

:class:`MBHBatchedLikeMove` keeps ALL of :class:`ResidualAddOneRemoveOneMove`'s
choreography (per-leaf expose/fold, in-model repeats, per-leaf tempering,
cold-chain bookkeeping) and swaps the scoring and fill paths, exactly as
:class:`SOBBHChunkedLikeMove` does for SOBBH:

* every chunk of ``batch_max_size`` rows is ONE batched ``compute_tdi_channels``
  launch on the leaf's shared lattice (the ``WindowedGridAlignedMBHWaveform``
  behind a ``MBHWindowedWDMSignalGen``), then a per-row segment WDM transform;
* every row is scored against ITS OWN container: the residual and the PSD are
  sliced to the template's box (``AnalysisContainer._slice_to_template``), so
  per-walker PSD samples are honoured;
* the convention bridge is the SOBBH one: ``compute_like`` returns
  ``offset[walker] + <r|h> - 1/2 <h|h>`` with ``offset = acs.likelihood()`` on
  the freshly exposed residual, which reproduces the container path's
  ``-1/2 <r-h|r-h>`` + noise term on every scoring site;
* the leaf window (kept WDM layers around the leaf's median cold-chain merger
  time) is fixed on expose/setup and locked for the visit; it is rebuilt only
  when the median leaves the margin band, and never on fold-back.

A batch the response refuses (``BatchNotLaunchable``) or a domain error in the
chunk falls back to the per-row container path for that chunk, LOUDLY, and is
counted in ``n_batch_fallbacks``. ``MBH_BATCHED_FILL=0`` restores the dense
per-row expose/fold. The built-in fast-vs-slow check (``_verify_prev_logl``)
recomputes through the stock generator at matching convention with
``MBH_CHECK_LL_TOL`` (default 0.5 nats).
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from ...analysiscontainer import shard_lookup_maps
from ...diagnostic import inner_product
from ...domains import WDMSettings, WDMSignal
from ...utils.device import device_context
from ...utils.exceptions import BatchNotLaunchable, WaveformDomainError
from ...utils.utility import asnumpy
from .addremovemove import ResidualAddOneRemoveOneMove

logger = logging.getLogger(__name__)

__all__ = ["MBHBatchedLikeMove", "mbh_window_layers"]

#: waveform-basis column of the merger time (seconds relative to waveform_t0)
_T_PLUNGE_COL = 10
#: the only generator kwargs the grid-aligned generator accepts
_ACCEPTED_GEN_KWARGS = ("start_freq", "ref_freq", "T")


def mbh_window_layers(wdm, t_merge_abs, window_before, window_after, window_pad, window_margin):
    """Kept-box and pad geometry, in WDM layers of ``wdm``, for a merger at ``t_merge_abs``.

    Returns ``dict(n_start, Nt_keep, n_pad)``: kept layers
    ``[n_start, n_start + Nt_keep)`` cover ``[t - before - margin, t + after +
    margin]`` snapped OUTWARD to layer boundaries, clamped so ``n_pad`` layers
    of segment fit on each side of the grid. ``Nt_keep`` depends on the
    durations only (never on ``t_merge_abs``), so the shared lattice length --
    and phentax's jit cache -- is constant across leaves. ``Nt_keep + 2 n_pad``
    is made even (``WDMSettings`` needs an even layer count).
    """
    layer_dt = float(wdm.layer_dt)
    t0 = float(wdm.t0)
    Nt = int(wdm.Nt)
    n_pad = int(np.ceil(float(window_pad) / layer_dt))
    span = float(window_before) + float(window_after) + 2.0 * float(window_margin)
    Nt_keep = int(np.ceil(span / layer_dt)) + 1
    if (Nt_keep + 2 * n_pad) % 2:
        Nt_keep += 1
    if Nt_keep + 2 * n_pad > Nt:
        raise ValueError(
            f"MBH window ({span / 86400:.1f} d kept + 2 x {n_pad} pad layers = "
            f"{Nt_keep + 2 * n_pad} layers) does not fit the WDM grid of {Nt} layers "
            f"x {layer_dt:.0f} s; shorten MBH_WINDOW_BEFORE_DAYS for this data span."
        )
    lo_abs = float(t_merge_abs) - float(window_before) - float(window_margin)
    n_start = int(np.floor((lo_abs - t0) / layer_dt))
    n_start = max(n_start, n_pad)
    n_start = min(n_start, Nt - n_pad - Nt_keep)
    return dict(n_start=int(n_start), Nt_keep=int(Nt_keep), n_pad=int(n_pad))


class MBHBatchedLikeMove(ResidualAddOneRemoveOneMove):
    """Add/remove move for the MBH branch scored through the batched windowed path.

    Args:
        *args: Positional arguments of :class:`ResidualAddOneRemoveOneMove`.
            ``waveform_gen`` stays the SLOW exact generator (or ``None`` to use
            the containers' installed one): it still owns the cross-check.
        batched_gen: the windowed sub-transform adapter
            (:class:`~lisatools.sources.batching.MBHWindowedWDMSignalGen`), or a
            ``DeviceLocalWaveGen`` resolving to one per device. Must expose
            ``set_window``, ``window_key``, ``waveform_t0``, ``t_plunge_snap``.
        batch_max_size: rows per generator launch (``MBH_BATCH_MAX_SIZE``).
        window_before, window_after, window_pad, window_margin: seconds.
        **kwargs: Keyword arguments of the base. ``dcga`` must be ``None``.
    """

    _record_dh_default = "1"

    def __init__(
        self, *args, batched_gen=None, batch_max_size=16,
        window_before=90 * 86400.0, window_after=10 * 86400.0,
        window_pad=4 * 86400.0, window_margin=86400.0, **kwargs,
    ):
        if kwargs.get("dcga") is not None:
            raise ValueError(
                "MBHBatchedLikeMove has no DCGA (replica) path: the batched "
                "generator scores against the ACA containers directly, and "
                "multi-GPU walker shards are served by per-shard routing inside "
                "compute_like_local. Build it without dcga= (use_dcga=False)."
            )
        if batched_gen is None:
            raise ValueError("MBHBatchedLikeMove requires batched_gen= (the windowed adapter).")
        super().__init__(*args, **kwargs)
        self.batched_gen = batched_gen
        self.batch_max_size = max(1, int(batch_max_size))
        self.window_before = float(window_before)
        self.window_after = float(window_after)
        self.window_pad = float(window_pad)
        self.window_margin = float(window_margin)
        self._wdm = self.acs.acs.flatten()[0].data.settings
        if not isinstance(self._wdm, WDMSettings):
            raise ValueError(
                "MBH_LIKELIHOOD=batched needs a WDM run domain; the containers "
                f"carry {type(self._wdm).__name__}."
            )
        self._exposed_offset = None
        self._leaf_windows = {}
        self.n_batch_fallbacks = 0
        self.last_batch_error = None
        self._warned_fallback_leaf = None
        self.check_ll_tol = float(
            os.environ.get(f"{self._dbg_prefix}_CHECK_LL_TOL", "0.5")
        )
        gen_kwargs = dict(self.waveform_gen_kwargs or {})
        dropped = sorted(set(gen_kwargs) - set(_ACCEPTED_GEN_KWARGS))
        self._gen_kwargs = {k: v for k, v in gen_kwargs.items() if k in _ACCEPTED_GEN_KWARGS}
        if dropped:
            logger.info(
                "[MBH_BATCH] waveform kwargs not forwarded to the grid-aligned "
                "generator (it takes only %s): %s", _ACCEPTED_GEN_KWARGS, dropped,
            )
        self._stats = dict(rows=0, seconds=0.0, chunks=0, fallbacks=0)

    # ------------------------------------------------------------------
    # window management
    # ------------------------------------------------------------------

    def _adapter(self):
        resolve = getattr(self.batched_gen, "_resolve", None)
        return resolve() if callable(resolve) else self.batched_gen

    def _stock_waveform_t0(self):
        """The UNSNAPPED epoch the rows' ``t_plunge`` is relative to."""
        return float(self.batched_gen.waveform_t0) - float(
            getattr(self.batched_gen, "t_plunge_snap", 0.0)
        )

    def _leaf_window(self, leaf, coords_ws, allow_rebuild):
        t_ref = self._stock_waveform_t0() + float(np.median(coords_ws[:, _T_PLUNGE_COL]))
        win = self._leaf_windows.get(leaf)
        if win is not None and (
            not allow_rebuild or abs(t_ref - win["t_ref"]) <= self.window_margin
        ):
            return win
        if win is None and not allow_rebuild:
            raise RuntimeError(
                f"MBHBatchedLikeMove: leaf {leaf} fold-back reached before any "
                "expose set its window (propose choreography violated)."
            )
        geom = mbh_window_layers(
            self._wdm, t_ref, self.window_before, self.window_after,
            self.window_pad, self.window_margin,
        )
        geom["t_ref"] = t_ref
        if win is not None:
            logger.info(
                "[MBH_BATCH] leaf %d window rebuilt: median merger moved %.2f h "
                "(layers %d -> %d)", leaf, (t_ref - win["t_ref"]) / 3600.0,
                win["n_start"], geom["n_start"],
            )
        self._leaf_windows[leaf] = geom
        return geom

    @staticmethod
    def _apply_window(adapter, geom):
        key = (geom["n_start"], geom["Nt_keep"], geom["n_pad"])
        if getattr(adapter, "window_key", None) != key:
            adapter.set_window(*key)

    # ------------------------------------------------------------------
    # shard routing
    # ------------------------------------------------------------------

    @staticmethod
    def _split_rows_static(holder, idx):
        """``[(device, positions)]`` grouping ``idx`` rows by owning walker shard."""
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        n_shards = len(holder.linear_data_arr)
        if holder.gpus is None:
            return [(None, np.arange(idx.size))]
        if n_shards == 1:
            return [(int(holder.gpus[0]), np.arange(idx.size))]
        split_map, _ = shard_lookup_maps(holder)
        owner = np.asarray(split_map)[idx]
        return [
            (int(holder.gpus[s]), np.where(owner == s)[0])
            for s in range(n_shards) if np.any(owner == s)
        ]

    def _split_rows(self, idx):
        return self._split_rows_static(self.acs, idx)

    # ------------------------------------------------------------------
    # likelihood
    # ------------------------------------------------------------------

    def setup_likelihood_here(self, coords):
        """Arm the per-walker exposed-residual offset and the leaf window."""
        self._flush_stats()
        self._exposed_offset = np.asarray(asnumpy(self.acs.likelihood()), dtype=float)
        coords_np = np.atleast_2d(np.asarray(asnumpy(coords), dtype=np.float64))
        self._leaf_window(int(self._current_leaf), coords_np, allow_rebuild=True)
        super().setup_likelihood_here(coords)

    def compute_like_local(self, coords_in, data_index):
        if self._dcga is not None:  # unreachable (ctor guard); keep loud
            raise NotImplementedError("MBHBatchedLikeMove has no DCGA path.")
        if self._exposed_offset is None:
            raise RuntimeError(
                "compute_like_local called before setup_likelihood_here armed the "
                "exposed-residual offset (propose() choreography violated)."
            )
        t_start = time.perf_counter()
        coords = np.atleast_2d(np.asarray(asnumpy(coords_in), dtype=np.float64))
        idx = np.asarray(asnumpy(data_index)).astype(np.int64).reshape(-1)
        n = int(coords.shape[0])
        out = np.full(n, -1e300, dtype=float)
        d_h = np.full(n, np.nan)
        h_h = np.full(n, np.nan)
        self._last_d_h = d_h
        self._last_h_h = h_h
        valid = np.all(np.isfinite(coords), axis=1)
        if not np.any(valid):
            return out
        leaf = int(self._current_leaf)
        geom = self._leaf_windows.get(leaf)
        if geom is None:
            raise RuntimeError(f"MBHBatchedLikeMove: leaf {leaf} window not set.")
        valid_pos = np.where(valid)[0]
        for device, pos in self._split_rows(idx[valid_pos]):
            rows = valid_pos[pos]
            with device_context(self.acs.xp, device):
                adapter = self._adapter()
                self._apply_window(adapter, geom)
                for lo in range(0, rows.size, self.batch_max_size):
                    sel = rows[lo: lo + self.batch_max_size]
                    ll_c, dh_c, hh_c = self._score_chunk(adapter, coords[sel], idx[sel], leaf)
                    out[sel] = ll_c
                    d_h[sel] = dh_c
                    h_h[sel] = hh_c
                    self._stats["chunks"] += 1
        self._stats["rows"] += n
        self._stats["seconds"] += time.perf_counter() - t_start
        return out

    def _generate(self, adapter, coords):
        params = np.array(coords, dtype=np.float64, copy=True)
        params[:, _T_PLUNGE_COL] -= float(getattr(adapter, "t_plunge_snap", 0.0))
        return adapter(*params.T, **self._gen_kwargs)

    def _score_chunk(self, adapter, coords, idx, leaf):
        try:
            tmpl = self._generate(adapter, coords)
        except (BatchNotLaunchable, WaveformDomainError) as exc:
            self.n_batch_fallbacks += 1
            self._stats["fallbacks"] += 1
            self.last_batch_error = exc
            if self._warned_fallback_leaf != leaf:
                self._warned_fallback_leaf = leaf
                logger.warning(
                    "[MBH_BATCH] leaf %d: batch refused (%s); scoring %d rows through "
                    "the per-row container path. Watch n_batch_fallbacks.",
                    leaf, exc, int(coords.shape[0]),
                )
            ll = np.real(np.asarray(
                self.compute_acs_like(coords, idx, **self.waveform_like_kwargs), dtype=float
            )).reshape(-1)
            return ll, np.full(ll.shape, np.nan), np.full(ll.shape, np.nan)
        return self._score_templates(tmpl, idx)

    def _score_templates(self, tmpl, idx):
        like_kw = {
            k: v for k, v in (self.waveform_like_kwargs or {}).items()
            if k not in ("psd", "complex", "include_psd_info")
        }
        box = tmpl.settings
        acs_flat = self.acs.acs.flatten()
        arr = tmpl.arr if tmpl.is_batched else tmpl.arr[None]
        n = int(idx.size)
        ll = np.empty(n)
        dh = np.empty(n)
        hh = np.empty(n)
        cache = {}
        for i in range(n):
            g = int(idx[i])
            h_i = WDMSignal(arr[i], box)
            if g not in cache:
                r_box, _, s_box = acs_flat[g]._slice_to_template(h_i)
                cache[g] = (r_box, s_box)
            r_box, s_box = cache[g]
            d = float(np.real(asnumpy(inner_product(r_box, h_i, psd=s_box, **like_kw))))
            h = float(np.real(asnumpy(inner_product(h_i, h_i, psd=s_box, **like_kw))))
            dh[i] = d
            hh[i] = h
            ll[i] = float(self._exposed_offset[g]) + d - 0.5 * h
        return ll, dh, hh

    # ------------------------------------------------------------------
    # expose / fold
    # ------------------------------------------------------------------

    def _apply_cold_chain_sources(self, coords, sign):
        what = "EXPOSE (r += h)" if sign > 0 else "FOLD-BACK (r -= h)"
        if os.environ.get("MBH_BATCHED_FILL", "1").strip() != "1":
            logger.info(
                "[MBH_FILL] %s via the DENSE per-row path (MBH_BATCHED_FILL=0): %d rows",
                what, int(np.shape(coords)[0]),
            )
            return super()._apply_cold_chain_sources(coords, sign)
        coords_np = np.atleast_2d(np.asarray(asnumpy(coords), dtype=np.float64))
        leaf = int(self._current_leaf)
        geom = self._leaf_window(leaf, coords_np, allow_rebuild=(sign > 0))
        idx_all = np.arange(coords_np.shape[0])
        valid = np.all(np.isfinite(coords_np), axis=1)
        if not np.any(valid):
            return
        t_start = time.perf_counter()
        n_fallback = 0
        valid_idx = idx_all[valid]
        for device, pos in self._split_rows(valid_idx):
            rows = valid_idx[pos]
            with device_context(self.acs.xp, device):
                adapter = self._adapter()
                self._apply_window(adapter, geom)
                for lo in range(0, rows.size, self.batch_max_size):
                    sel = rows[lo: lo + self.batch_max_size]
                    try:
                        tmpl = self._generate(adapter, coords_np[sel])
                    except (BatchNotLaunchable, WaveformDomainError) as exc:
                        n_fallback += 1
                        self.last_batch_error = exc
                        self.acs.apply_signal_from_params(
                            sign, {self.branch_name: coords_np[sel]}, index=sel,
                            waveform_kwargs=self._branch_waveform_kwargs(),
                            signal_gen_resolver=self._resolve_signal_gen_override,
                            apply_transform=False, domain_error="skip",
                        )
                        continue
                    self.acs.signal_operation(sign, tmpl, data_index=sel)
        logger.info(
            "[MBH_FILL] %s via batched windowed templates: %d rows (%d skipped, "
            "%d chunk fallbacks) in %.2f s", what, int(valid.sum()),
            int((~valid).sum()), n_fallback, time.perf_counter() - t_start,
        )

    # ------------------------------------------------------------------
    # records, checks, telemetry
    # ------------------------------------------------------------------

    def _record_leaf_inner_products(self, new_state, add_coords_in, leaf):
        """Record cold-chain ``<d|h>``, ``<h|h>`` from the batched scorer (free)."""
        if not getattr(self, "record_inner_products", False):
            return
        _sub = (getattr(new_state, "sub_states", None) or {}).get(self.branch_name)
        if _sub is None or getattr(_sub, "d_h", None) is None:
            return
        walker_idx = np.arange(self.nwalkers, dtype=np.int32)
        self.compute_like(add_coords_in, walker_idx)
        _sub.d_h[:, leaf] = self._last_d_h[: self.nwalkers]
        _sub.h_h[:, leaf] = self._last_h_h[: self.nwalkers]

    def _verify_prev_logl(self, prev_logl, old_coords_in, data_index_in, leaf):
        """Built-in fast-vs-slow A/B at MATCHING convention (see the SOBBH move)."""
        acs_like = (
            self.compute_check_like(old_coords_in, data_index_in)
            .reshape(prev_logl.shape)
            .real
        )
        both = (
            np.isfinite(prev_logl) & np.isfinite(acs_like)
            & (prev_logl > -1e299) & (acs_like > -1e299)
        )
        if not np.any(both):
            return
        diff = prev_logl[both] - acs_like[both]
        max_abs = float(np.abs(diff).max())
        if max_abs <= self.check_ll_tol:
            return
        spread = float(diff.max() - diff.min())
        msg = (
            f"{self.branch_name} leaf {leaf}: batched windowed fast path vs slow "
            f"container path disagree beyond tol={self.check_ll_tol}: "
            f"max|diff|={max_abs:.6e}, spread={spread:.6e} over {int(both.sum())} "
            "points. A small spread at high SNR is sub-transform truncation "
            "(widen MBH_WINDOW_PAD_DAYS / raise MBH_CHECK_LL_TOL); a large spread "
            "means a window, snap or residual bug."
        )
        if self.check_ll_mode == "strict":
            raise ValueError(msg)
        logger.warning(msg)

    def _flush_stats(self):
        st = self._stats
        if st["rows"]:
            logger.info(
                "[MBH_BATCH] leaf %s: %d rows in %d chunks, %.1f s (%.3f s/row), "
                "%d fallbacks", self._current_leaf, st["rows"], st["chunks"],
                st["seconds"], st["seconds"] / st["rows"], st["fallbacks"],
            )
        self._stats = dict(rows=0, seconds=0.0, chunks=0, fallbacks=0)
```

Add to `src/lisatools/globalfit/moves/__init__.py`, next to the SOBBH import: `from .mbhbatchedmove import MBHBatchedLikeMove` (and to `__all__` if the module defines one).

- [ ] **Step 4: Run the tests**

Run: `nice -n 10 $PY -m unittest tests.test_mbh_batched_move -v`
Expected: all PASS. Known places to look if not: (a) `acs.acs.flatten()[0].data` -- if `AnalysisContainerArray` has no `.acs` attribute use `self.acs.flatten()` (there is a `flatten` at ~line 2605); (b) `record_inner_products` may not exist on the base until `setup()` runs -- the `getattr(..., False)` guard covers it; (c) the parity tolerance: print the max diff; anything above 5e-2 on this toy means the window or the offset is wrong, not truncation.

- [ ] **Step 5: Run the SOBBH neighbour to prove nothing shared regressed**

Run: `nice -n 10 $PY -m unittest tests.test_sobbh_chunked_move tests.test_sobbh_chunked_fill`
Expected: OK.

- [ ] **Step 6: Stage**

```bash
git add src/lisatools/globalfit/moves/mbhbatchedmove.py src/lisatools/globalfit/moves/__init__.py tests/test_mbh_batched_move.py
# git commit -m "feat(mbh): MBHBatchedLikeMove -- batched windowed scoring and fill against per-walker containers"
```

---

### Task 5: Stock erebor wiring, builder, knobs, docs

**Files:**
- Modify: `src/lisatools/globalfit/stock/erebor/source_runtime.py` (`SourceMBHSettings` ~line 453; `source_signal_cfg` ~line 1005; add `resolve_mbh_batched_cfg`, `get_mbh_windowed_gen`; `build_mbh_move_runtime` ~line 1625)
- Modify: `src/lisatools/globalfit/recipe.py` (add `MBHBatchedMoveBuilder` after `MBHMoveBuilder` ~line 5362; import `MBHBatchedLikeMove` where `SOBBHChunkedLikeMove` is imported)
- Modify: `docs/codebase-map.md` (`globalfit/moves` row: add the move)
- Test: `tests/test_mbh_batched_wiring.py`

**Interfaces:**
- Consumes: Task 3's `WindowedGridAlignedMBHWaveform`, Task 2's `MBHWindowedWDMSignalGen`, Task 4's `MBHBatchedLikeMove`; `DeviceLocalWaveGen`, `_general_info_xp`, `_primary_device`, `_device_local_orbits`, `_device_local_domain_settings`, `_WAVE_WRAP_CACHE`, `get_mbh_phenom_gen`, `env_default`, `SingleSourcePEBuilder`.
- Produces: `resolve_mbh_batched_cfg(mbh) -> dict` (keys `mbh_likelihood`, `mbh_batch_max_size`, `mbh_window_before`, `mbh_window_after`, `mbh_window_pad`, `mbh_window_margin`, `mbh_waveform_duration`); `get_mbh_windowed_gen(general_info, cfg) -> MBHWindowedWDMSignalGen` (per device, cached; adapter carries `waveform_t0` and `t_plunge_snap`); `MBHBatchedMoveBuilder`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mbh_batched_wiring.py
"""Knobs, cfg resolution and builder selection for MBH_LIKELIHOOD=batched."""
from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np


class MBHBatchedKnobsTest(unittest.TestCase):
    def test_defaults_leave_the_stock_path(self):
        from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

        with mock.patch.dict(os.environ, {}, clear=False):
            for k in ("MBH_LIKELIHOOD", "MBH_BATCH_MAX_SIZE", "MBH_RESPONSE_ORDER",
                      "MBH_WINDOW_BEFORE_DAYS", "MBH_WINDOW_AFTER_DAYS",
                      "MBH_WINDOW_PAD_DAYS", "MBH_WINDOW_MARGIN_DAYS"):
                os.environ.pop(k, None)
            s = SourceMBHSettings()
        self.assertEqual(s.likelihood, "full")
        self.assertEqual(s.batch_max_size, 16)
        self.assertEqual(s.response_order, 8)
        self.assertEqual((s.window_before_days, s.window_after_days), (90.0, 10.0))
        self.assertEqual((s.window_pad_days, s.window_margin_days), (4.0, 1.0))

    def test_env_knobs(self):
        from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

        with mock.patch.dict(os.environ, {"MBH_LIKELIHOOD": "batched", "MBH_BATCH_MAX_SIZE": "8",
                                          "MBH_RESPONSE_ORDER": "30", "MBH_WINDOW_BEFORE_DAYS": "60"}):
            s = SourceMBHSettings()
        self.assertEqual((s.likelihood, s.batch_max_size, s.response_order, s.window_before_days),
                         ("batched", 8, 30, 60.0))


class ResolveBatchedCfgTest(unittest.TestCase):
    def _mbh(self, **kw):
        from lisatools.globalfit.stock.erebor.source_runtime import SourceMBHSettings

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MBH_WAVEFORM_DURATION", None)
            s = SourceMBHSettings()
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    def test_full_path_passes_through(self):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_mbh_batched_cfg

        cfg = resolve_mbh_batched_cfg(self._mbh(likelihood="full"))
        self.assertEqual(cfg["mbh_likelihood"], "full")
        self.assertEqual(cfg["mbh_waveform_duration"], self._mbh().waveform_duration)

    def test_batched_pins_duration_to_the_window(self):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_mbh_batched_cfg

        cfg = resolve_mbh_batched_cfg(self._mbh(likelihood="batched"))
        self.assertEqual(cfg["mbh_waveform_duration"], 90 * 86400.0)
        self.assertEqual(cfg["mbh_window_before"], 90 * 86400.0)
        self.assertEqual(cfg["mbh_window_after"], 10 * 86400.0)
        self.assertEqual(cfg["mbh_window_pad"], 4 * 86400.0)
        self.assertEqual(cfg["mbh_window_margin"], 86400.0)
        self.assertEqual(cfg["mbh_batch_max_size"], 16)

    def test_duration_conflict_raises(self):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_mbh_batched_cfg

        with mock.patch.dict(os.environ, {"MBH_WAVEFORM_DURATION": "2592000"}):
            with self.assertRaises(ValueError):
                resolve_mbh_batched_cfg(self._mbh(likelihood="batched", waveform_duration=2592000.0))

    def test_tdionfly_conflict_raises(self):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_mbh_batched_cfg

        with self.assertRaises(ValueError):
            resolve_mbh_batched_cfg(self._mbh(likelihood="batched", use_tdionfly=True))

    def test_unknown_value_raises(self):
        from lisatools.globalfit.stock.erebor.source_runtime import resolve_mbh_batched_cfg

        with self.assertRaises(ValueError):
            resolve_mbh_batched_cfg(self._mbh(likelihood="fast"))


class BuilderTest(unittest.TestCase):
    def test_builder_class_attrs(self):
        from lisatools.globalfit.moves import MBHBatchedLikeMove
        from lisatools.globalfit.recipe import MBHBatchedMoveBuilder, MBHMoveBuilder

        self.assertTrue(issubclass(MBHBatchedMoveBuilder, MBHMoveBuilder))
        self.assertIs(MBHBatchedMoveBuilder.move_class, MBHBatchedLikeMove)
        self.assertFalse(MBHBatchedMoveBuilder.use_dcga)

    def test_snap_helper(self):
        from lisatools.globalfit.stock.erebor.source_runtime import snap_waveform_t0_to_lattice

        # offset 0.327664 s = 0.13 samples -> NEAREST lattice point is k = 0
        t0, snap = snap_waveform_t0_to_lattice(97729089.327664, 97729089.0, 2.5)
        self.assertAlmostEqual(t0, 97729089.0, places=6)
        self.assertAlmostEqual(snap, -0.327664, places=6)
        self.assertAlmostEqual((t0 - 97729089.0) / 2.5, round((t0 - 97729089.0) / 2.5), places=9)
        # offset 1.5 s = 0.6 samples -> k = 1, snapped 2.5 s after data_t0
        t0c, snapc = snap_waveform_t0_to_lattice(97729090.5, 97729089.0, 2.5)
        self.assertAlmostEqual(t0c, 97729091.5, places=6)
        self.assertAlmostEqual(snapc, 1.0, places=6)
        t0b, snapb = snap_waveform_t0_to_lattice(100.0, 0.0, 2.5)
        self.assertEqual((t0b, snapb), (100.0, 0.0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `nice -n 10 $PY -m unittest tests.test_mbh_batched_wiring -v`
Expected: `AttributeError: 'SourceMBHSettings' object has no attribute 'likelihood'`, `ImportError` for `resolve_mbh_batched_cfg`, `MBHBatchedMoveBuilder`, `snap_waveform_t0_to_lattice`.

- [ ] **Step 3: Settings fields**

In `SourceMBHSettings` (source_runtime.py), replace `response_order: int = 30` with, and add after it:

```python
    # Response Lagrange order. User ruling 2026-09-29: default 8 (PR #82
    # evidence: mismatch flat from order 30 down to 4; ~1.6x cheaper).
    response_order: int = dataclasses.field(
        default_factory=env_default("MBH_RESPONSE_ORDER", 8, int)
    )
    # Scoring path: "full" (stock per-row container path) or "batched" (the
    # MBHBatchedLikeMove: one grid-aligned response launch per chunk of rows
    # on a per-leaf window, segment WDM transform, per-walker residual+PSD).
    likelihood: str = dataclasses.field(
        default_factory=env_default("MBH_LIKELIHOOD", "full", str)
    )
    batch_max_size: int = dataclasses.field(
        default_factory=env_default("MBH_BATCH_MAX_SIZE", 16, int)
    )
    # Per-leaf window around the median cold-chain merger (days): kept box =
    # [t - before - margin, t + after + margin]; pad = discarded segment edge.
    window_before_days: float = dataclasses.field(
        default_factory=env_default("MBH_WINDOW_BEFORE_DAYS", 90.0, float)
    )
    window_after_days: float = dataclasses.field(
        default_factory=env_default("MBH_WINDOW_AFTER_DAYS", 10.0, float)
    )
    window_pad_days: float = dataclasses.field(
        default_factory=env_default("MBH_WINDOW_PAD_DAYS", 4.0, float)
    )
    window_margin_days: float = dataclasses.field(
        default_factory=env_default("MBH_WINDOW_MARGIN_DAYS", 1.0, float)
    )
```

- [ ] **Step 4: cfg resolution, snap helper, getter, selection**

Add to `source_runtime.py` (module scope, before `source_signal_cfg`):

```python
def resolve_mbh_batched_cfg(mbh) -> dict:
    """Plain-value MBH scoring-path config, with the batched-mode consistency rules.

    ``batched`` pins BOTH generators (the stock one the engine installs for
    residual rebuilds and the cross-check, and the windowed one) to
    ``waveform_duration = window_before`` so the two agree on the inspiral
    length; an explicit ``MBH_WAVEFORM_DURATION`` that disagrees is refused,
    and so is ``use_tdionfly`` (the windowed generator is the legacy-response
    family). ``full`` passes the stock values through untouched.
    """
    mode = str(mbh.likelihood)
    if mode not in ("full", "batched"):
        raise ValueError(f"MBH_LIKELIHOOD must be 'full' or 'batched'; got {mode!r}")
    before = float(mbh.window_before_days) * 86400.0
    after = float(mbh.window_after_days) * 86400.0
    pad = float(mbh.window_pad_days) * 86400.0
    margin = float(mbh.window_margin_days) * 86400.0
    duration = mbh.waveform_duration
    if mode == "batched":
        if bool(mbh.use_tdionfly):
            raise ValueError(
                "MBH_LIKELIHOOD=batched uses the legacy-response grid-aligned "
                "generator; it cannot be combined with USE_TDIONFLY=1."
            )
        explicit = os.environ.get("MBH_WAVEFORM_DURATION")
        if explicit is not None and (duration is None or abs(float(duration) - before) > 1.0):
            raise ValueError(
                f"MBH_LIKELIHOOD=batched generates {mbh.window_before_days} days before "
                f"the merger (MBH_WINDOW_BEFORE_DAYS) for BOTH the stock and the "
                f"windowed generator; MBH_WAVEFORM_DURATION={explicit} disagrees. "
                "Unset it or make the two equal."
            )
        duration = before
    return dict(
        mbh_likelihood=mode,
        mbh_batch_max_size=int(mbh.batch_max_size),
        mbh_window_before=before,
        mbh_window_after=after,
        mbh_window_pad=pad,
        mbh_window_margin=margin,
        mbh_waveform_duration=duration,
    )


def snap_waveform_t0_to_lattice(waveform_t0: float, data_t0: float, dt: float):
    """``(waveform_t0_snapped, snap)`` with the snapped epoch on ``data_t0 + k*dt``.

    ``snap = waveform_t0_snapped - waveform_t0``; callers subtract it from
    every ``t_plunge`` so absolute merger times are unchanged (PR #82 remedy
    for ``GridAlignedPhenomTHMTDIWaveform._check_alignable``).
    """
    offset = float(waveform_t0) - float(data_t0)
    k = int(np.rint(offset / float(dt)))
    snapped = float(data_t0) + k * float(dt)
    return snapped, snapped - float(waveform_t0)
```

In `source_signal_cfg`, replace `waveform_duration=mbh.waveform_duration,` inside `mbh_phenom_kwargs` with `waveform_duration=_mbh_batched["mbh_waveform_duration"],`, add `_mbh_batched = resolve_mbh_batched_cfg(mbh)` as the first statement of the function, and add `**_mbh_batched,` to the returned dict.

Add the getter after `get_mbh_phenom_gen`:

```python
def get_mbh_windowed_gen(general_info, cfg):
    """Per-device (cached) windowed grid-aligned MBH generator in its sub-transform adapter.

    Built like :func:`get_mbh_phenom_gen` (device-local orbits and domain
    settings) but from :class:`WindowedGridAlignedMBHWaveform`, with
    ``waveform_t0`` snapped onto the data lattice; the adapter carries
    ``waveform_t0`` (snapped) and ``t_plunge_snap`` for the move.
    """
    from lisatools.domains import WDMSettings
    from lisatools.sources.batching import MBHWindowedWDMSignalGen
    from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform

    xp = _general_info_xp(general_info)
    primary = _primary_device(general_info)
    base_orbits = (
        general_info.gpu_orbits if general_info.gpus is not None else general_info.orbits
    )
    orbits = _device_local_orbits(base_orbits, xp, primary)
    wdm = _device_local_domain_settings(general_info.domain_settings, xp, primary)
    key = ("mbh_windowed", id(orbits), id(wdm))
    if key in _WAVE_WRAP_CACHE:
        return _WAVE_WRAP_CACHE[key]
    if not isinstance(wdm, WDMSettings):
        raise ValueError(
            "MBH_LIKELIHOOD=batched needs a WDM run domain "
            f"(general.domain_settings is {type(wdm).__name__})."
        )
    pk = cfg["mbh_phenom_kwargs"]
    t0_snapped, snap = snap_waveform_t0_to_lattice(
        cfg["mbh_waveform_t0"], general_info.data_t0, general_info.dt
    )
    if snap != 0.0:
        logger.info(
            "[MBH_BATCH] waveform_t0 snapped onto the data lattice by %+.6f s "
            "(t_plunge rows are shifted by the same amount inside the move)", snap,
        )
    gen = WindowedGridAlignedMBHWaveform(
        waveform_kwargs=dict(
            higher_modes=list(pk["higher_modes"]), include_negative_modes=True,
            t_low_fit=True, coarse_grain=False, atol=pk["phenom_tol"], rtol=pk["phenom_tol"],
        ),
        Tobs=float(pk["waveform_duration"]),
        start_freq=pk["start_freq"],
        use_reference_time=True,
        waveform_t0=t0_snapped,
        data_td_settings=general_info.data_td_settings,
        tdi_generation=cfg["tdi_gen_str"],
        tdi_channels=cfg["tdi_chan"],
        sampling_frequency=1.0 / general_info.dt,
        orbits=orbits,
        order=pk["response_order"],
        tukey_alpha=general_info.window_alpha,
        stft_dt=None,
        freq_min=pk["min_freq"],
        freq_max=pk["max_freq"],
        fft_batch_size=1,
        buffer_time=pk["buffer_time"],
        output_domain_settings=wdm,
        force_backend=general_info.force_backend,
    )
    adapter = MBHWindowedWDMSignalGen(
        gen, wdm, nchannels=cfg["nchannels"], tukey_alpha=general_info.window_alpha
    )
    adapter.waveform_t0 = t0_snapped
    adapter.t_plunge_snap = snap
    _WAVE_WRAP_CACHE[key] = adapter
    return adapter
```

Check `get_mbh_phenom_wave_gen`'s key list in `wrappers.py` for the exact kwarg names the stock generator takes (`min_freq`, `max_freq`, `fft_batch_size` are in `mbh_phenom_kwargs` / the cache key); mirror them.

Replace `build_mbh_move_runtime`'s legacy branch:

```python
def build_mbh_move_runtime(curr, acs, priors, state, cfg):
    """MBH PE move: batched windowed (MBH_LIKELIHOOD=batched), stretch RJ move on
    the tdionfly wrap, or the stock ``build_mbh_moves_phenom`` builder."""
    mbh_info = curr.source_info["mbh"]
    if cfg.get("mbh_likelihood", "full") == "batched":
        slow = DeviceLocalWaveGen(get_mbh_phenom_gen, curr.general_info, cfg)
        batched = DeviceLocalWaveGen(get_mbh_windowed_gen, curr.general_info, cfg)
        _, moves = MBHBatchedMoveBuilder(
            wave_gen=slow.get_signals_for_residuals,
            batched_gen=batched,
            batch_max_size=cfg["mbh_batch_max_size"],
            window_before=cfg["mbh_window_before"],
            window_after=cfg["mbh_window_after"],
            window_pad=cfg["mbh_window_pad"],
            window_margin=cfg["mbh_window_margin"],
        ).build(None, curr, acs, priors, state)
        return moves[0]
    if not cfg["mbh_use_tdionfly"]:
        wave_gen = DeviceLocalWaveGen(get_mbh_phenom_gen, curr.general_info, cfg)
        _, move = build_mbh_moves_phenom(
            curr, acs, priors, state, wave_gen=wave_gen, subtract_initial=False
        )
        return move
    wave_gen = DeviceLocalWaveGen(get_mbh_tdionfly_wave_wrap, curr.general_info, cfg)
    _, moves = MBHMoveBuilder(
        wave_gen=wave_gen, waveform_like_kwargs=mbh_info.waveform_kwargs
    ).build(None, curr, acs, priors, state)
    return moves[0]
```

Import `MBHBatchedMoveBuilder` at the top of `source_runtime.py` where `SOBBHChunkedMoveBuilder` is imported (line ~49).

In `recipe.py`, after `class MBHMoveBuilder`:

```python
class MBHBatchedMoveBuilder(MBHMoveBuilder):
    """:class:`MBHMoveBuilder` constructing :class:`MBHBatchedLikeMove`.

    ``wave_gen`` stays the SLOW exact generator (residual parity with the
    engine + the fast-vs-slow cross-check); the windowed adapter and its knobs
    pass through ``move_kwargs`` (``batched_gen=``, ``batch_max_size=``,
    ``window_*=``). The DCGA branch is skipped: the move raises if handed one.
    """

    move_class = MBHBatchedLikeMove
    use_dcga = False
```

and add `MBHBatchedLikeMove` to the `from .moves import (...)` list that brings in `SOBBHChunkedLikeMove`.

Docs: in `docs/codebase-map.md` add to the `globalfit/` row (or the moves description) one sentence: "`moves/mbhbatchedmove.py` (`MBHBatchedLikeMove`): MBH add/remove move scored through one batched grid-aligned response launch per chunk on a per-leaf 90 d / 10 d window with a segment WDM transform, per-walker residual+PSD; `MBH_LIKELIHOOD=batched`, default off (2026-09-29)."

- [ ] **Step 5: Run the tests**

Run: `nice -n 10 $PY -m unittest tests.test_mbh_batched_wiring -v`
Expected: all PASS. If `SourceMBHSettings()` cannot be constructed without arguments (a required field on the `Settings` base), obtain the default block the way the stock fits do instead: `from lisatools.globalfit.stock import erebor; s = erebor.all_sources_lite().mbh` (construction is validation-only by rule 2, no data is loaded), and apply the same env patches around that call.

Then the stock-fit cheapness and alignment suites that read these settings: `nice -n 10 $PY -m unittest tests.test_stock_globalfit tests.test_stock_waveform_alignment` (the slow class is skipped without `LAT_SLOW_TESTS=1`).

- [ ] **Step 6: Stage**

```bash
git add src/lisatools/globalfit/stock/erebor/source_runtime.py src/lisatools/globalfit/recipe.py docs/codebase-map.md tests/test_mbh_batched_wiring.py
# git commit -m "feat(stock): MBH_LIKELIHOOD=batched wiring, MBHBatchedMoveBuilder, window/order knobs"
```

---

### Task 6: Validation on this laptop against the mojito MBHB files (and the GPU probe)

**Files:**
- Create: `scripts/mbh/mbh_batched_mojito_check.py`
- Create (results): `scripts/mbh/mbh_batched_mojito_check_results.md`

**Interfaces:**
- Consumes: `lisatools.globalfit.preprocessing.L1ProcessingStep`, `find_file`; `lisatools.detector.L1Orbits`; `mbh_catalogue_to_sampling_basis`; `make_mbh_transform_container`; `PhenomTHMTDIWaveform`; Tasks 2-4 classes; `mbh_window_layers`.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
"""Batched windowed MBH likelihood vs the stock path on REAL mojito MBHB data.

Laptop (CPU, one id) or cluster GPU (batch sweep + memory). For the catalogue
truth plus jittered rows it reports mismatch between the two templates,
<d|h>, <h|h>, logL of each against the mojito stream, delta logL, the stock
template's power outside the kept box, the kept-layer relative error, and
s/row. User ruling 2026-09-29: "make sure our match/logL will be okay".

    MBHB_ID=16 MBH_BACKEND=cpu python scripts/mbh/mbh_batched_mojito_check.py
    MBHB_ID=16 MBH_BACKEND=cuda12x MBH_BATCH_SIZES=1,4,8,16,24 python scripts/mbh/mbh_batched_mojito_check.py
"""
import gc, json, os, resource, sys, threading, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

MEM_CAP_GB = float(os.environ.get("MBH_MEM_CAP_GB", "7.0"))
_IS_MAC = sys.platform == "darwin"


def rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if _IS_MAC else r / 1e6


def _watchdog():
    while True:
        if rss_gb() > MEM_CAP_GB:
            print(f"[watchdog] RSS {rss_gb():.2f} GB > cap {MEM_CAP_GB} GB -> exit", flush=True)
            os._exit(42)
        time.sleep(0.3)


def mark(m):
    print(f"[RSS {rss_gb():5.2f} GB] {m}", flush=True)


threading.Thread(target=_watchdog, daemon=True).start()

from lisatools.detector import L1Orbits
from lisatools.globalfit.preprocessing import L1ProcessingStep, find_file
from lisatools.globalfit.recipe import mbh_catalogue_to_sampling_basis
from lisatools.globalfit.stock.erebor import make_mbh_transform_container
from lisatools.domains import TDSettings, WDMSettings, WDMSignal
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.sources.bbh.waveform import PhenomTHMTDIWaveform
from lisatools.sources.bbh.gridaligned import WindowedGridAlignedMBHWaveform
from lisatools.sources.batching import MBHWindowedWDMSignalGen
from lisatools.globalfit.moves.mbhbatchedmove import mbh_window_layers
from lisatools.globalfit.stock.erebor.source_runtime import snap_waveform_t0_to_lattice
from lisatools.utils.utility import asnumpy

REF = 97729089.327664                       # MOJITO_REFERENCE_TIME = waveform_t0
PATH = os.environ.get("MOJITO_ROOT", os.path.expanduser("~/.mojito_cache/brickmarket/mojito_light_v1_0_0/"))
MBHB_L1 = os.path.join(PATH, "data", "MBHB", "L1")
MBHB_ID = int(os.environ.get("MBHB_ID", "16"))
BACKEND = os.environ.get("MBH_BACKEND", "cpu")
DT = 2.5
WINDOW_DAYS = float(os.environ.get("MBH_CHECK_WINDOW_DAYS", "120"))
MERGER_AT_DAYS = float(os.environ.get("MBH_CHECK_MERGER_AT_DAYS", "100"))
BEFORE, AFTER, PAD, MARGIN = 90 * 86400.0, 10 * 86400.0, 4 * 86400.0, 86400.0
ORDER = int(os.environ.get("MBH_RESPONSE_ORDER", "8"))
BATCH_SIZES = [int(b) for b in os.environ.get("MBH_BATCH_SIZES", "1,4").split(",")]
N_JITTER = int(os.environ.get("MBH_CHECK_N_ROWS", "4"))
OUT_DIR = os.environ.get("MBH_CHECK_OUT", os.path.join(os.path.dirname(__file__), "mbh_batched_check_out"))
os.makedirs(OUT_DIR, exist_ok=True)
CACHE = os.path.join(OUT_DIR, f"mbh_mojito_window_id{MBHB_ID}_{int(WINDOW_DAYS)}d.npz")
TRANSFORM = make_mbh_transform_container()


def load_window():
    """(data_td (3, N_WIN), window_t0, cat) -- cached; merger MERGER_AT_DAYS in."""
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["data_td"], float(z["window_t0"]), z["cat"].item()
    mark("reading the mojito MBHB L1 stream (one time)")
    loader = L1ProcessingStep(
        L1_folder=PATH, source_types=["mbhb"], source_ids={"mbhb": MBHB_ID},
        orbits_class=L1Orbits, orbits_kwargs=dict(force_backend="cpu", frame="icrs"), verbose=True,
    )
    times = np.asarray(loader.times)
    data_full = np.asarray(loader.data)
    dt_native = float(loader.dt)
    assert abs(dt_native - DT) < 1e-9, dt_native
    data_t0 = float(times[0])
    cat = {k: float(np.asarray(v)) for k, v in loader.catalogue["MBHB"][MBHB_ID].items()
           if np.asarray(v).dtype.kind in "fi"}
    abs_merger = REF + cat["TimeCoalescencePhenomTPHMSSBFrame"]
    n_win = int(round(WINDOW_DAYS * 86400.0 / DT))
    start = int(round((abs_merger - MERGER_AT_DAYS * 86400.0 - data_t0) / DT))
    start = max(0, min(start, data_full.shape[1] - n_win))
    data_td = data_full[:, start:start + n_win].copy()
    window_t0 = data_t0 + start * DT
    np.savez(CACHE, data_td=data_td, window_t0=window_t0, cat=cat)
    del data_full, times, loader
    gc.collect()
    return data_td, window_t0, cat


def orbits_for(window_t0, tobs):
    orb = L1Orbits(find_file(MBHB_L1, "MBHB", MBHB_ID), force_backend=BACKEND, frame="icrs")
    pad = 1.0e5
    lo = max(window_t0 - pad, float(orb.sc_t0))
    hi = min(window_t0 + tobs + pad, float(orb._sc_t_base[-1]))
    ltt_t = np.asarray(orb.ltt_t)
    m = (ltt_t >= lo) & (ltt_t <= hi)
    orb.ltt = np.asarray(orb.ltt)[m].copy()
    orb.ltt_t = ltt_t[m].copy()
    orb.ltt_t0 = float(orb.ltt_t[0])
    orb.configure(linear_interp_setup=True, dt=300.0)
    return orb


def gen_kwargs(window_t0, n_win, orb, wdm, waveform_t0):
    return dict(
        waveform_kwargs=dict(higher_modes=[21, 33, 44], include_negative_modes=True,
                             t_low_fit=True, coarse_grain=False, atol=1e-12, rtol=1e-12),
        Tobs=BEFORE, start_freq=7e-5, use_reference_time=True, waveform_t0=waveform_t0,
        data_td_settings=TDSettings(n_win, DT, t0=window_t0, force_backend=BACKEND),
        tdi_generation="2nd generation", tdi_channels="XYZ", sampling_frequency=1.0 / DT,
        orbits=orb, order=ORDER, tukey_alpha=0.0, stft_dt=None, freq_min=1e-4, freq_max=2.5e-2,
        fft_batch_size=1, buffer_time=15000.0, output_domain_settings=wdm, force_backend=BACKEND,
    )


def device_mem():
    if BACKEND == "cpu":
        return rss_gb()
    import cupy as cp
    return cp.get_default_memory_pool().used_bytes() / 1e9


def main():
    data_td, window_t0, cat = load_window()
    n_win = data_td.shape[1]
    NF, NT, _ = WDMSettings.adjust_to_even_bins(0.5 * 86400.0, 0.75 * 86400.0, DT, n_win * DT)
    n_win = NF * NT
    data_td = data_td[:, :n_win]
    tobs = n_win * DT
    wdm = WDMSettings(NF, NT, DT, t0=window_t0, min_freq=1e-4, max_freq=2.5e-2, force_backend=BACKEND)
    mark(f"window {tobs / 86400:.1f} d, Nf={NF} Nt={NT} layer={NF * DT / 3600:.2f} h")
    xp = wdm.xp
    # the data is the mojito MBHB stream itself (noiseless), transformed on the run grid
    d_wdm = TDSignal(
        xp.asarray(data_td), TDSettings(n_win, DT, t0=window_t0, force_backend=BACKEND)
    ).transform(wdm)
    ac = AnalysisContainer(d_wdm, XYZ2SensitivityMatrix(wdm, model="scirdv1"))
    mark("container ready")

    truth = np.asarray(TRANSFORM.both_transforms(np.asarray(mbh_catalogue_to_sampling_basis(cat), float)), float)
    rng = np.random.default_rng(0)
    rows = np.tile(truth, (N_JITTER, 1))
    rows[1:, 4] *= rng.uniform(0.97, 1.03, N_JITTER - 1)          # dist
    rows[1:, 10] += rng.uniform(-20.0, 20.0, N_JITTER - 1)          # t_plunge, seconds
    rows[1:, 5] += rng.uniform(-0.2, 0.2, N_JITTER - 1)             # phi_ref

    orb = orbits_for(window_t0, tobs)
    stock = PhenomTHMTDIWaveform(**gen_kwargs(window_t0, n_win, orb, wdm, REF))
    t0s, snap = snap_waveform_t0_to_lattice(REF, window_t0, DT)
    windowed = WindowedGridAlignedMBHWaveform(**gen_kwargs(window_t0, n_win, orb, wdm, t0s))
    adapter = MBHWindowedWDMSignalGen(windowed, wdm, nchannels=3, tukey_alpha=0.0)
    geom = mbh_window_layers(wdm, REF + truth[10], BEFORE, AFTER, PAD, MARGIN)
    adapter.set_window(geom["n_start"], geom["Nt_keep"], geom["n_pad"])
    box = slice(geom["n_start"], geom["n_start"] + geom["Nt_keep"])
    mark(f"generators ready; snap {snap:+.6f} s; kept layers {box.start}..{box.stop} of {NT}")

    results = dict(id=MBHB_ID, backend=BACKEND, order=ORDER, window_days=tobs / 86400, Nf=NF, Nt=NT, rows=[])
    # ---- stock, one row at a time ------------------------------------------
    stock_tmpl, stock_t = [], []
    for r in rows:
        t = time.perf_counter()
        h = stock.get_signals_for_residuals(*r)
        stock_t.append(time.perf_counter() - t)
        stock_tmpl.append(h)
    mark(f"stock: {np.mean(stock_t):.2f} s/row")
    # ---- batched windowed, batch sweep --------------------------------------
    batched_tmpl = None
    for B in BATCH_SIZES:
        mem0 = device_mem()
        t = time.perf_counter()
        outs = []
        for lo in range(0, N_JITTER, B):
            p = rows[lo:lo + B].copy()
            p[:, 10] -= snap
            outs.append(adapter(*p.T))
        el = time.perf_counter() - t
        peak = device_mem()
        print(f"[batched B={B}] {el / N_JITTER:.3f} s/row, device mem {mem0:.2f} -> {peak:.2f} GB", flush=True)
        results[f"batched_B{B}_s_per_row"] = el / N_JITTER
        results[f"batched_B{B}_mem_gb"] = peak
        if batched_tmpl is None:
            batched_tmpl = [WDMSignal(o.arr[i], o.settings) for o in outs for i in range(o.arr.shape[0])]
    results["stock_s_per_row"] = float(np.mean(stock_t))
    # ---- accuracy -----------------------------------------------------------
    for i, (hs, hb) in enumerate(zip(stock_tmpl, batched_tmpl)):
        hs_arr = np.asarray(asnumpy(hs.arr))
        hb_arr = np.asarray(asnumpy(hb.arr))
        scale = float(np.abs(hs_arr).max())
        kept_err = float(np.abs(hs_arr[..., box] - hb_arr).max() / scale)
        outside = float(np.sqrt((np.delete(hs_arr, np.s_[box], axis=-1) ** 2).sum() / (hs_arr ** 2).sum()))
        hh_s = float(ac.template_snr(hs)[0]) ** 2       # optimal SNR^2 = <h|h>
        hh_b = float(ac.template_snr(hb)[0]) ** 2
        dh_s = float(np.real(asnumpy(ac.template_inner_product(hs))))
        dh_b = float(np.real(asnumpy(ac.template_inner_product(hb))))
        ll_s = float(np.real(asnumpy(ac.template_likelihood(hs))))
        ll_b = float(np.real(asnumpy(ac.template_likelihood(hb))))
        # mismatch between the two templates: stock restricted to the box vs batched
        hs_box = WDMSignal(hs_arr[..., box], hb.settings)
        ac_b = AnalysisContainer(hs_box, XYZ2SensitivityMatrix(hb.settings, model="scirdv1"))
        O = complex(ac_b.template_inner_product(hb, normalize=True, complex=True))
        row = dict(row=i, dh_stock=dh_s, dh_batched=dh_b, hh_stock=hh_s, hh_batched=hh_b,
                   logL_stock=ll_s, logL_batched=ll_b, dlogL=ll_b - ll_s,
                   mismatch=1.0 - abs(O), kept_layer_rel_err=kept_err, stock_power_outside_box=outside ** 2)
        results["rows"].append(row)
        print(f"row {i}: logL stock {ll_s:.3f} batched {ll_b:.3f} dlogL {ll_b - ll_s:+.4f} | "
              f"mm {1 - abs(O):.3e} | kept err {kept_err:.2e} | power outside box {outside ** 2:.2e} | "
              f"SNRopt {np.sqrt(hh_s):.1f}/{np.sqrt(hh_b):.1f}", flush=True)
    with open(os.path.join(OUT_DIR, f"results_id{MBHB_ID}_{BACKEND}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
```

The script imports `TDSignal` with the other domain names: make the domain import line `from lisatools.domains import TDSettings, TDSignal, WDMSettings, WDMSignal`.

- [ ] **Step 2: Dry-run the script's imports and window geometry without data**

Run: `nice -n 10 $PY -c "import runpy, sys; sys.argv=['x']; import scripts.mbh.mbh_batched_mojito_check as m; print(m.BEFORE, m.MBHB_L1)"` (or `python -c "import ast; ast.parse(open('scripts/mbh/mbh_batched_mojito_check.py').read())"`).
Expected: no import errors.

- [ ] **Step 3: Run on the laptop (CPU) for id 16**

Close other heavy processes first (8 GB box). Run:

```bash
MBHB_ID=16 MBH_BACKEND=cpu MBH_BATCH_SIZES=1,4 nice -n 10 $PY scripts/mbh/mbh_batched_mojito_check.py 2>&1 | tee scripts/mbh/mbh_batched_check_out/log_id16_cpu.txt
```

Expected: the L1 read (minutes, ~1.5 GB RSS transient), then per-row lines. Acceptance: every row has `|dlogL| <= 0.5` and `mismatch < 1e-6`; `stock_power_outside_box` well below `1e-4` (it is what the sub-box drops). If `kept_layer_rel_err` is above `1e-4`, raise `PAD` to 8 days and re-run before touching anything else. If phentax on CPU takes more than ~2 min per stock row, set `MBH_CHECK_N_ROWS=2`.

- [ ] **Step 4: Repeat for one more id (17) and record**

Run the same command with `MBHB_ID=17`. Write `scripts/mbh/mbh_batched_mojito_check_results.md` with the two per-row tables (copy the printed lines), the s/row numbers, the window geometry line, and the machine (laptop CPU, deving). State plainly whether the acceptance line held.

- [ ] **Step 5: Stage**

```bash
git add scripts/mbh/mbh_batched_mojito_check.py scripts/mbh/mbh_batched_mojito_check_results.md
# git commit -m "scripts(mbh): batched windowed MBH likelihood vs stock on mojito MBHB data (laptop check + GPU probe)"
```

(The `mbh_batched_check_out/` cache directory is scratch; do not stage it.)

---

### Task 7: Regression pass and handoff

**Files:** none new.

- [ ] **Step 1: Run the suites that touch the changed code, one process at a time**

```bash
for t in test_wdm_subbox test_mbh_windowed_signal_gen test_mbh_batched_move test_mbh_batched_wiring \
         test_batched_likelihood test_batching_isolation test_coarse_wdm test_batched_response \
         test_sobbh_chunked_move test_sobbh_chunked_fill test_psd_move_batched test_aca_vectorized_dispatch \
         test_stock_globalfit test_sensitivity test_wdm_domain_cpp; do
  echo "=== $t"; nice -n 10 $PY -m unittest tests.$t 2>&1 | tail -3
done
```

Expected: every block ends in `OK` (skips allowed).

- [ ] **Step 2: Update the memory note and report**

Append to `/Users/mkatz/.claude/projects/-Users-mkatz-Research-lisa-sprint-2026/memory/project_pr82_cd1l_merge_state_0929.md` a "IMPLEMENTED" section: files added, the mojito check numbers (dlogL, mismatch, s/row per id), the pad actually needed, and what is still uncommitted. Report to Mike: the numbers, the knob to flip (`MBH_LIKELIHOOD=batched`, plus `MBH_CHECK_LL_EVERY=10` for the first segment), and that the cluster probe is the same script with `MBH_BACKEND=cuda12x MBH_BATCH_SIZES=1,4,8,16,24`.
