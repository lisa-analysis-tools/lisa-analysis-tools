"""Cross terms of residual and template rows: ``STFTComputationGroup.compute_cross_terms``.

The kernel of ``cutils/stft_cross_terms.cu`` takes tables of residual rows (the group's data rows,
each with its noise row and window start) and template rows (rows of a second array with their own
width and window starts). For every table ``b``, residual ``i`` and template ``j`` it returns
``(r_i|t_j)``, ``(t_j|t_j)`` and ``(r_i|r_i)``, all under the noise row of residual ``i`` and summed
over the table's bin window. The GB tempering reads the change of ln L of a swap from these terms.

The tests build random STFT cells with store windows in a small grid, in two layouts of the same
numbers: noise windows held by each cell (the copied layout) and full-grid noise rows read through a
``NoiseRowSource`` (the zero-copy band buffer). They check:

* both layouts against a NumPy reference, for windows equal to the cells, narrower than the cells,
  and reaching past them and past the grid; XYZ with the full 3 x 3 inverse CSD and AET with the
  diagonal; residual and template counts that differ and need several pair tiles;
* the two layouts give bit-identical outputs;
* tables split over calls, or given in another order, give bit-identical outputs, also when the
  calls size their launches for different window widths;
* malformed inputs are refused.

The CPU always runs; CUDA when a GPU is usable.
"""

from __future__ import annotations

import unittest

import numpy as np

NUM_CHANNELS = 3
# A cell then spans two of the kernel's 2048-pixel ranges, and the widest table four.
NUM_SEGMENTS = 120
SEGMENT_DURATION = 86400.0
SAMPLING_INTERVAL = 5.0
GRID_MIN_FREQ = 1.0e-3
GRID_BINS_WANTED = 72
CELL_WIDTH = 24
TEMPLATE_WIDTH = 19
NUM_WALKERS = 4
NUM_CELLS = 12
NUM_TEMPLATE_ROWS = 10
# Window starts of the cells cluster around these active bins, as the cells of one band share a start.
BAND_STARTS = (3, 21, 44)
SEED = 20261029
# Relative tolerance against the reference, measured on the scale of each term (see `_scale`).
RTOL = 1e-12


def _cuda_usable() -> bool:
    """True when lisatools has a built CUDA backend and a device can allocate."""
    try:
        import cupy

        import lisatools

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False
        # has_backend knows the backend names only, not the "gpu" alias of get_backend.
        if not any(lisatools.has_backend(name) for name in ("cuda13x", "cuda12x", "cuda11x")):
            return False
        cupy.zeros(1)
        return True
    except Exception:
        return False


def to_host(array):
    """NumPy copy of a NumPy or CuPy array."""
    if hasattr(array, "get"):
        return array.get()
    return np.array(array, copy=True)


def grid_and_cell_settings(backend):
    """The grid's settings and those of a ``CELL_WIDTH``-bin cell, as ``gbbands.py`` builds them."""
    from lisatools.domains import STFTSettings

    df = 1.0 / SEGMENT_DURATION
    grid = STFTSettings(
        t0=0.0,
        dt=SEGMENT_DURATION,
        df=df,
        NT=NUM_SEGMENTS,
        NF=int(SEGMENT_DURATION / SAMPLING_INTERVAL) // 2 + 1,
        min_freq=GRID_MIN_FREQ,
        max_freq=GRID_MIN_FREQ + (GRID_BINS_WANTED - 1) * df,
        force_backend=backend,
    )
    cell = STFTSettings(
        *grid.args,
        min_freq=max(grid.ind_min - 0.5, 0.0) * df,
        max_freq=(grid.ind_min + CELL_WIDTH - 0.5) * df,
        force_backend=backend,
    )
    if cell.NF_active != CELL_WIDTH or cell.min_freq != grid.min_freq:
        raise RuntimeError("cell settings do not match the store-window layout of gbbands.py")
    return grid, cell


def draw_noise_rows(rng, num_rows, num_bins, diagonal):
    """Full-grid inverse-CSD rows: Hermitian and positive definite per pixel (XYZ), or positive."""
    pixel_shape = (NUM_SEGMENTS, num_bins)
    if diagonal:
        rows = 1.0 + 0.5 * rng.random((num_rows, NUM_CHANNELS) + pixel_shape)
        return rows.astype(np.complex128)
    rows = np.zeros((num_rows, NUM_CHANNELS, NUM_CHANNELS) + pixel_shape, np.complex128)
    for first in range(NUM_CHANNELS):
        rows[:, first, first] = 1.0 + 0.5 * rng.random((num_rows,) + pixel_shape)
        for second in range(first + 1, NUM_CHANNELS):
            off_diagonal = 0.3 * (
                rng.random((num_rows,) + pixel_shape) - 0.5
                + 1j * (rng.random((num_rows,) + pixel_shape) - 0.5)
            )
            rows[:, first, second] = off_diagonal
            rows[:, second, first] = np.conj(off_diagonal)
    return rows


def complex_normal(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


class CellSet:
    """Random cells, their full-grid noise rows and a template array, on host and on ``backend``.

    Cell ``c`` belongs to walker ``c % NUM_WALKERS``, whose noise row it reads; its window starts
    near one of ``BAND_STARTS``.
    """

    def __init__(self, backend, diagonal, seed=SEED):
        import lisatools

        self.backend = backend
        self.diagonal = diagonal
        self.tdi_type = "AET" if diagonal else "XYZ"
        self.xp = lisatools.get_backend(backend).xp
        self.grid, self.cell_settings = grid_and_cell_settings(backend)
        self.grid_bins = int(self.grid.NF_active)
        rng = np.random.default_rng(seed)
        self.walkers = np.arange(NUM_CELLS) % NUM_WALKERS
        jitter = rng.integers(-2, 3, NUM_CELLS)
        self.starts = np.clip(
            np.asarray(BAND_STARTS)[np.arange(NUM_CELLS) % len(BAND_STARTS)] + jitter,
            0, self.grid_bins - CELL_WIDTH,
        ).astype(np.int32)
        self.data = complex_normal(rng, (NUM_CELLS, NUM_CHANNELS, NUM_SEGMENTS, CELL_WIDTH))
        self.noise_rows = draw_noise_rows(rng, NUM_WALKERS, self.grid_bins, diagonal)
        self.templates_host = 0.5 * complex_normal(
            rng, (NUM_TEMPLATE_ROWS, NUM_CHANNELS, NUM_SEGMENTS, TEMPLATE_WIDTH)
        )
        self.templates = self.xp.ascontiguousarray(self.xp.asarray(self.templates_host))
        self.rng = rng

    def noise_window(self, cell):
        start = int(self.starts[cell])
        return self.noise_rows[self.walkers[cell]][..., start:start + CELL_WIDTH]

    def container_array(self, full_grid):
        """The cells in one array: own noise windows, or full-grid rows through a noise source."""
        from lisatools.analysiscontainer import (
            AnalysisContainer, AnalysisContainerArray, NoiseRowSource,
        )
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.sensitivity import XYZSensitivityBackend

        xp = self.xp
        settings = self.cell_settings
        orbits = EqualArmlengthOrbits(force_backend=self.backend)
        containers = []
        for cell in range(NUM_CELLS):
            window = self.noise_window(cell)
            sensitivity = XYZSensitivityBackend(
                orbits=orbits, settings=settings, force_backend=self.backend
            )
            sensitivity.sens_mat = xp.zeros(window.shape, dtype=np.complex128)
            sensitivity.invC = (
                xp.zeros(window.shape, dtype=np.complex128) if full_grid
                else xp.asarray(np.ascontiguousarray(window))
            )
            sensitivity.channel_shape = window.shape[:-2]
            containers.append(AnalysisContainer(
                settings.associated_class(xp.asarray(self.data[cell].copy()), settings),
                sensitivity,
            ))
        source = None
        if full_grid:
            source = NoiseRowSource(
                arrays=[xp.ascontiguousarray(xp.asarray(self.noise_rows.reshape(-1)))],
                row_shape=self.noise_rows.shape[-2:],
                row_of_ac=self.walkers,
            )
        return AnalysisContainerArray(
            containers,
            gpus=None if self.backend == "cpu" else [0],
            domain_group_kwargs=dict(tdi_type=self.tdi_type),
            noise_source=source,
        )

    def draw_tables(self, num_tables, num_residuals, num_templates, window_kind):
        """Random tables: cells of mixed starts, templates anywhere near the table's band."""
        rng = self.rng
        bands = rng.integers(0, len(BAND_STARTS), num_tables)
        base = np.asarray(BAND_STARTS)[bands]
        residual_cells = rng.integers(0, NUM_CELLS, (num_tables, num_residuals))
        template_rows = rng.integers(0, NUM_TEMPLATE_ROWS, (num_tables, num_templates))
        template_starts = base[:, None] + rng.integers(-6, 10, (num_tables, num_templates))
        if window_kind == "cell":
            window_lo, window_hi = base, base + CELL_WIDTH
        elif window_kind == "band":
            window_lo, window_hi = base + 4, base + CELL_WIDTH - 6
        elif window_kind == "wide":
            # Past the cells on both sides, and below the grid's first bin for the first band.
            window_lo, window_hi = base - 7, base + CELL_WIDTH + 9
        else:
            raise ValueError(window_kind)
        return dict(
            residual_cells=residual_cells.astype(np.int32),
            template_rows=template_rows.astype(np.int32),
            template_starts=template_starts.astype(np.int32),
            window_lo=window_lo.astype(np.int32),
            window_hi=window_hi.astype(np.int32),
        )

    def reference(self, tables):
        """NumPy sums of the three terms, with the same window rules as the kernel."""
        df = float(self.grid.df)
        cells = tables["residual_cells"]
        num_tables, num_residuals = cells.shape
        num_templates = tables["template_rows"].shape[1]
        cross = np.zeros((num_tables, num_residuals, num_templates))
        template = np.zeros((num_tables, num_residuals, num_templates))
        residual = np.zeros((num_tables, num_residuals))
        for table in range(num_tables):
            lo, hi = int(tables["window_lo"][table]), int(tables["window_hi"][table])
            for slot, cell in enumerate(cells[table]):
                start = int(self.starts[cell])
                bins = np.arange(max(lo, start, 0), min(hi, start + CELL_WIDTH, self.grid_bins))
                if bins.size == 0:
                    continue
                res = self.data[cell][..., bins - start]
                inv_cov = self.noise_rows[self.walkers[cell]][..., bins]
                tpl = np.zeros((num_templates, NUM_CHANNELS, NUM_SEGMENTS, bins.size), complex)
                for column in range(num_templates):
                    local = bins - int(tables["template_starts"][table, column])
                    inside = (local >= 0) & (local < TEMPLATE_WIDTH)
                    row = self.templates_host[tables["template_rows"][table, column]]
                    tpl[column][..., inside] = row[..., local[inside]]
                if self.diagonal:
                    weighted = inv_cov[None] * tpl
                    residual[table, slot] = np.einsum("atf,atf,atf->", res.conj(), inv_cov, res).real
                else:
                    weighted = np.einsum("abtf,jbtf->jatf", inv_cov, tpl)
                    residual[table, slot] = np.einsum(
                        "atf,abtf,btf->", res.conj(), inv_cov, res
                    ).real
                cross[table, slot] = np.einsum("atf,jatf->j", res.conj(), weighted).real
                template[table, slot] = np.einsum("jatf,jatf->j", tpl.conj(), weighted).real
        factor = 4.0 * df
        return factor * cross, factor * template, factor * residual


def call(group, cell_set, tables, noise_rows, subset=None):
    """``compute_cross_terms`` on ``tables`` (or the tables ``subset``), outputs on the host."""
    pick = slice(None) if subset is None else subset
    cells = tables["residual_cells"][pick]
    starts = cell_set.starts[cells]
    outputs = group.compute_cross_terms(
        cells, noise_rows[cells], starts, cell_set.templates,
        tables["template_rows"][pick], tables["template_starts"][pick],
        tables["window_lo"][pick], tables["window_hi"][pick],
    )
    return tuple(to_host(output) for output in outputs)


def _scale(template_ref, residual_ref):
    """Size of a cross or template term: |(t|t)| + sqrt(|(r|r)(t|t)|), the bound of |(r|t)|."""
    return np.abs(template_ref) + np.sqrt(np.abs(residual_ref[..., None] * template_ref))


class _CrossTermCases:
    """Bodies shared by the CPU and CUDA cases; ``BACKEND`` is set by the subclasses."""

    BACKEND = "cpu"

    def layouts(self, diagonal):
        """The cell set and its two groups, with the noise row each cell reads in each."""
        cell_set = CellSet(self.BACKEND, diagonal)
        windowed = cell_set.container_array(full_grid=False)
        full_grid = cell_set.container_array(full_grid=True)
        self.assertTrue(full_grid.external_noise_rows)
        return cell_set, {
            "windowed": (windowed.cpp_splits[0], np.arange(NUM_CELLS, dtype=np.int32)),
            "full grid": (full_grid.cpp_splits[0], cell_set.walkers.astype(np.int32)),
        }

    def assert_matches_reference(self, outputs, reference, label):
        cross, template, residual = outputs
        cross_ref, template_ref, residual_ref = reference
        scale = _scale(template_ref, residual_ref)
        np.testing.assert_array_less(np.abs(residual - residual_ref),
                                     RTOL * np.abs(residual_ref) + 1e-300, err_msg=label)
        for name, value, ref in (("cross", cross, cross_ref), ("template", template, template_ref)):
            np.testing.assert_array_less(np.abs(value - ref), RTOL * scale + 1e-300,
                                         err_msg=f"{label}: {name}")
        # * Each kind of window must leave some pixels in every table, or the test proves nothing.
        self.assertTrue(np.all(np.abs(residual_ref).max(axis=1) > 0), msg=label)

    def test_matches_reference_in_both_layouts(self):
        for diagonal in (False, True):
            cell_set, layouts = self.layouts(diagonal)
            for window_kind in ("cell", "band", "wide"):
                tables = cell_set.draw_tables(6, 5, 7, window_kind)
                reference = cell_set.reference(tables)
                outputs = {}
                for name, (group, noise_rows) in layouts.items():
                    label = f"{cell_set.tdi_type}, {window_kind} window, {name} noise"
                    with self.subTest(label):
                        outputs[name] = call(group, cell_set, tables, noise_rows)
                        self.assert_matches_reference(outputs[name], reference, label)
                for value, other in zip(outputs["windowed"], outputs["full grid"]):
                    np.testing.assert_array_equal(
                        value, other, err_msg=f"{cell_set.tdi_type}, {window_kind}: layouts differ"
                    )

    def test_several_pair_tiles(self):
        # * More than 32 residuals and templates: the kernel splits the pairs into tiles.
        cell_set, layouts = self.layouts(diagonal=False)
        group, noise_rows = layouts["full grid"]
        tables = cell_set.draw_tables(3, 37, 34, "cell")
        self.assert_matches_reference(
            call(group, cell_set, tables, noise_rows), cell_set.reference(tables), "37 x 34"
        )

    def test_tables_split_or_reordered_give_identical_results(self):
        cell_set, layouts = self.layouts(diagonal=False)
        group, noise_rows = layouts["full grid"]
        # Wide windows give the tables different widths, so the calls size their launches differently.
        tables = cell_set.draw_tables(7, 6, 5, "wide")
        tables["window_hi"][3] += 20
        together = call(group, cell_set, tables, noise_rows)
        parts = [call(group, cell_set, tables, noise_rows, subset=part)
                 for part in (slice(0, 2), slice(2, 7))]
        order = np.arange(7)[::-1]
        reordered = call(group, cell_set, tables, noise_rows, subset=order)
        for index, value in enumerate(together):
            np.testing.assert_array_equal(np.concatenate([part[index] for part in parts]), value)
            np.testing.assert_array_equal(reordered[index][order.argsort()], value)

    def test_malformed_inputs_are_refused(self):
        cell_set, layouts = self.layouts(diagonal=False)
        group, noise_rows = layouts["full grid"]
        tables = cell_set.draw_tables(2, 3, 4, "cell")
        cells = tables["residual_cells"]
        arguments = dict(
            residual_index=cells, noise_index=noise_rows[cells],
            residual_starts=cell_set.starts[cells], templates=cell_set.templates,
            template_index=tables["template_rows"], template_starts=tables["template_starts"],
            window_lo=tables["window_lo"], window_hi=tables["window_hi"],
        )
        xp = cell_set.xp
        changes = {
            "noise rows of another shape": dict(noise_index=noise_rows[cells][:, :2]),
            "template starts of another shape": dict(template_starts=tables["template_starts"][:1]),
            "one window per table missing": dict(window_hi=tables["window_hi"][:1]),
            "strided templates": dict(templates=cell_set.templates[:, :, :, ::2]),
            "single-precision templates": dict(templates=cell_set.templates.astype(xp.complex64)),
            "templates without channels": dict(templates=cell_set.templates[:, 0]),
        }
        for label, change in changes.items():
            with self.subTest(label), self.assertRaises(ValueError):
                group.compute_cross_terms(**{**arguments, **change})


class CrossTermsCPUTest(_CrossTermCases, unittest.TestCase):
    BACKEND = "cpu"


@unittest.skipUnless(_cuda_usable(), "requires a usable CUDA backend")
class CrossTermsCudaTest(_CrossTermCases, unittest.TestCase):
    BACKEND = "gpu"


if __name__ == "__main__":
    unittest.main()
