"""Noise rows wider than the data rows: the full-grid noise layout of the STFT domain.

The STFT GB kernels can read each cell's inverse CSD from a full-grid noise row while the data rows
hold store windows (``STFTDomainWrap(..., num_freqs_noise=, noise_full_grid=True)``): a pixel then
reads its data at its window bin and its noise at its active bin. The tests take the production
fixture of ``scripts/dffr/stft_fixtures.py`` (two cells with 64-bin store windows at bins 250 and
1287 of a 2497-bin band, sources next to both window edges) and build three layouts of the same
numbers:

* windowed: the fixture's group, whose noise rows are the cells' windows (the existing layout);
* full-grid noise: the same data rows with full-grid noise rows whose windows equal the cells'
  noise windows; every other bin holds other values, so a read at a wrong bin changes the result;
* full grid: a group whose data and noise rows both hold the full grid, with the same noise rows.

Every output of the golden evaluation except the Fisher matrices must be bit-identical between the
first two (``fill_global_stft`` reads no noise and is left out, because the GPU adds it atomically).
The Fisher matrix contracts the noise rows only, so with full-grid noise it covers the whole grid
and must be bit-identical to the third layout. The case of a fixed PSD, where every cell reads the
same noise row, is one full-grid row holding the windows of both cells and read by every source
through noise index 0; its likelihood, swap and gradient outputs must equal the windowed layout's. Both responses run on the XYZ configuration, TDI on
the fly also on the diagonal AET one; the CPU always, CUDA when a GPU is usable. The default layout
itself is covered by the golden captures of ``scripts/dffr/run_dffr.sh``.

The same full-grid rows also feed an ``AnalysisContainerArray`` built on a ``NoiseRowSource`` (the
layout of the zero-copy band buffer, whose cells read their walkers' parent rows): its computation
group must give the windowed outputs bit for bit and the full grid's Fisher matrix, its containers
must read views of the rows, a repack must leave the rows alone, ``set_noise_rows`` must repoint in
place, a malformed source must be refused, and with two CPU splits each split's group must read the
row its container names.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# * The fixtures of the STFT golden captures live with the DFFR scripts of this repository.
SCRIPTS_DFFR = Path(__file__).resolve().parents[1] / "scripts" / "dffr"
# A seed apart from the fixture's, for the noise bins outside the cells' windows.
OUTSIDE_SEED = 20261027
CASES = (
    ("esa_tdi2_envelope", "tdi_on_the_fly"),
    ("esa_aet_diag", "tdi_on_the_fly"),
    ("esa_tdi2_envelope", "dffr"),
)
FISHER_VARIANTS = {
    "fisher": dict(),
    "fisher_easy_cd": dict(easy_central_difference=True),
    "fisher_envelope": dict(linear_envelope=True),
}


def _fixture_modules():
    """The fixture and golden-output modules, or None when gbgpu's STFT computations are missing."""
    try:
        from gbgpu.gbcomps import STFTGBComputations  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return None
    if str(SCRIPTS_DFFR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DFFR))
    import stft_fixtures
    import stft_golden_outputs

    return stft_fixtures, stft_golden_outputs


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


class _FullGridNoiseGroup:
    """The fixture's group with a C++ domain that reads full-grid noise rows.

    Everything but the domain and the noise count comes from the fixture's group, so the Fresnel
    evaluator, its envelope switch, ``d_d`` and the DFFR kinematics are the ones the windowed
    layout uses.
    """

    def __init__(self, base_group, cpp_domain, num_noise, keepalive):
        self._base_group = base_group
        self.cpp_domain = cpp_domain
        self.num_noise = num_noise
        # ! The domain wrap holds raw pointers into these arrays.
        self._keepalive = keepalive

    def __getattr__(self, name):
        return getattr(self._base_group, name)


def full_grid_noise_rows(fixtures, fixture, group):
    """Host noise rows over the whole band whose store windows equal the cells' noise windows."""
    description = fixture.description
    num_cells = fixture.num_templates
    num_segments = description["num_segments"]
    width = description["cell_window_width"]
    full_width = description["band_num_freqs"]
    starts = description["cell_window_starts"]
    diagonal = description["tdi_type"] != "XYZ"
    channel_shape = (fixtures.NUM_CHANNELS,) if diagonal else (fixtures.NUM_CHANNELS,) * 2
    windows = fixtures.to_host(group.invC_arr).reshape(
        (num_cells,) + channel_shape + (num_segments, width)
    )
    rng = np.random.default_rng(OUTSIDE_SEED)
    rows = np.empty((num_cells,) + channel_shape + (num_segments, full_width), dtype=np.complex128)
    for cell in range(num_cells):
        _, outside = fixtures.draw_cell_arrays(rng, num_segments, full_width, diagonal)
        outside[..., starts[cell]:starts[cell] + width] = windows[cell]
        rows[cell] = outside
    return rows


def full_grid_noise_group(fixture, group, rows):
    """The fixture's data rows with ``rows`` as full-grid noise rows."""
    xp = fixture.comp.xp
    num_cells, full_width = rows.shape[0], rows.shape[-1]
    with group.group_device_context():
        noise = xp.ascontiguousarray(xp.asarray(rows.reshape(-1)))
        arguments = list(group.domain_args)
        # domain_args: [..., data (8), invC (9), num_data (10), num_noise (11), tdi type (12)]
        arguments[9] = noise
        arguments[11] = num_cells
        domain = group.backend.STFTDomainWrap(
            *arguments, num_freqs_noise=full_width, noise_full_grid=True
        )
    return _FullGridNoiseGroup(group, domain, num_cells, keepalive=[noise])


def full_grid_group(fixtures, fixture, rows, backend):
    """A real group whose data and noise rows hold the whole band, with ``rows`` as noise."""
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.sensitivity import XYZSensitivityBackend

    description = fixture.description
    band_settings, _ = fixtures.production_settings(
        description["data_start"], description["num_segments"], backend
    )
    xp = fixtures.array_module(backend)
    # * The GB kernels read the comp's own orbits, never the sensitivity backend's.
    sensitivity_orbits = EqualArmlengthOrbits(force_backend=backend)
    containers = []
    for row in rows:
        # The Fisher matrix reads no data, so the data rows stay zero.
        data = xp.zeros((fixtures.NUM_CHANNELS,) + row.shape[-2:], dtype=xp.complex128)
        sensitivity = XYZSensitivityBackend(
            orbits=sensitivity_orbits, settings=band_settings, force_backend=backend
        )
        sensitivity.sens_mat = xp.zeros(row.shape, dtype=np.complex128)
        sensitivity.invC = xp.asarray(row)
        sensitivity.channel_shape = row.shape[: -len(band_settings.basis_shape_active)]
        containers.append(
            AnalysisContainer(band_settings.associated_class(data, band_settings), sensitivity)
        )
    container_array = AnalysisContainerArray(
        containers,
        gpus=None if backend == "cpu" else [0],
        domain_group_kwargs=dict(
            tdi_type=description["tdi_type"],
            window_alpha=description["window_alpha"],
            use_midpoint=description["use_midpoint"],
            linear_envelope=description["linear_envelope"],
        ),
    )
    return container_array, container_array.cpp_splits[0]


def noise_source_array(fixtures, fixture, group, rows, backend, n_splits=None, **source_changes):
    """The fixture's windowed data rows in an array that reads ``rows`` through a noise source.

    This is the layout of the zero-copy band buffer: each container holds its data window and reads
    its noise from a full-grid row it does not own. ``n_splits`` (CPU only) gives every split the
    same host array of all rows. ``source_changes`` replace entries of the source, to test its checks.
    """
    from lisatools.analysiscontainer import (
        AnalysisContainer, AnalysisContainerArray, NoiseRowSource,
    )
    from lisatools.detector import EqualArmlengthOrbits
    from lisatools.sensitivity import XYZSensitivityBackend

    description = fixture.description
    num_cells, num_segments = rows.shape[0], description["num_segments"]
    width = description["cell_window_width"]
    channel_shape = rows.shape[1:-2]
    _, cell_settings = fixtures.production_settings(
        description["data_start"], num_segments, backend
    )
    xp = fixtures.array_module(backend)
    data_rows = fixtures.to_host(group.data_arr).reshape(
        (num_cells, fixtures.NUM_CHANNELS, num_segments, width)
    )
    sensitivity_orbits = EqualArmlengthOrbits(force_backend=backend)
    containers = []
    for cell in range(num_cells):
        window_shape = channel_shape + (num_segments, width)
        sensitivity = XYZSensitivityBackend(
            orbits=sensitivity_orbits, settings=cell_settings, force_backend=backend
        )
        sensitivity.sens_mat = xp.zeros(window_shape, dtype=np.complex128)
        sensitivity.invC = xp.zeros(window_shape, dtype=np.complex128)
        sensitivity.channel_shape = channel_shape
        containers.append(AnalysisContainer(
            cell_settings.associated_class(xp.asarray(data_rows[cell].copy()), cell_settings),
            sensitivity,
        ))
    rows_flat = xp.ascontiguousarray(xp.asarray(rows.reshape(-1)))
    source = dict(
        arrays=[rows_flat] * (1 if n_splits is None else n_splits),
        row_shape=rows.shape[-2:],
        row_of_ac=np.arange(num_cells),
    )
    source.update(source_changes)
    return AnalysisContainerArray(
        containers,
        gpus=None if backend == "cpu" else [0],
        n_splits=n_splits,
        domain_group_kwargs=dict(
            tdi_type=description["tdi_type"],
            window_alpha=description["window_alpha"],
            use_midpoint=description["use_midpoint"],
            linear_envelope=description["linear_envelope"],
        ),
        noise_source=NoiseRowSource(**source),
    )


def address_range(array):
    """First byte address and byte count of a host or device array."""
    if hasattr(array, "data") and hasattr(array.data, "ptr"):
        return int(array.data.ptr), int(array.nbytes)
    return int(array.__array_interface__["data"][0]), int(array.nbytes)


class _NoiseLayoutCases:
    """Bodies shared by the CPU and CUDA test cases; ``BACKEND`` is set by the subclasses."""

    BACKEND = "cpu"

    @classmethod
    def setUpClass(cls):
        modules = _fixture_modules()
        if modules is None:
            raise unittest.SkipTest("gbgpu with STFTGBComputations is not importable")
        cls.fixtures, cls.golden = modules
        cls.orbits_cache = {}

    def build(self, config_name, response):
        fixture = self.fixtures.build_fixture(
            config_name, self.BACKEND, orbits_cache=self.orbits_cache, response=response
        )
        fixture.description["backend"] = self.BACKEND
        return fixture

    def test_full_grid_noise_matches_windowed_noise(self):
        for config_name, response in CASES:
            with self.subTest(config=config_name, response=response):
                fixture = self.build(config_name, response)
                comp = fixture.comp
                windowed = comp.stft_comps
                rows = full_grid_noise_rows(self.fixtures, fixture, windowed)
                full_noise = full_grid_noise_group(fixture, windowed, rows)

                reference = self.golden.evaluate(fixture)
                comp.stft_comps = full_noise
                try:
                    candidate = self.golden.evaluate(fixture)
                finally:
                    comp.stft_comps = windowed
                compared = [
                    key for key in reference if not key.startswith("fisher") and key != "fill"
                ]
                self.assertIn("sw_ar", compared)
                self.assertIn("gswap_a", compared)
                for key in compared:
                    np.testing.assert_array_equal(
                        candidate[key], reference[key],
                        err_msg=f"{key} differs between full-grid and windowed noise rows",
                    )

    def test_fisher_with_full_grid_noise_matches_full_grid(self):
        for config_name, response in CASES:
            with self.subTest(config=config_name, response=response):
                fixture = self.build(config_name, response)
                comp = fixture.comp
                xp = comp.xp
                windowed = comp.stft_comps
                rows = full_grid_noise_rows(self.fixtures, fixture, windowed)
                full_noise = full_grid_noise_group(fixture, windowed, rows)
                container_array, full_grid = full_grid_group(
                    self.fixtures, fixture, rows, self.BACKEND
                )
                sources = self.golden.first_sources(
                    fixture.cells, self.golden.NUM_FISHER_SOURCES
                )
                params = fixture.params[sources]
                cells = xp.asarray(fixture.cells[sources])
                for key, options in FISHER_VARIANTS.items():
                    # * The window starts must not matter: full-grid noise rows start at bin 0.
                    with_full_noise = comp.information_matrix(
                        params, holder=full_noise, noise_index=cells,
                        start_freq_inds=fixture.window_starts, **options,
                    )
                    on_full_grid = comp.information_matrix(
                        params, holder=full_grid, noise_index=cells, **options
                    )
                    self.assertTrue(bool(xp.isfinite(with_full_noise).all()), msg=key)
                    np.testing.assert_array_equal(
                        self.fixtures.to_host(with_full_noise),
                        self.fixtures.to_host(on_full_grid),
                        err_msg=f"{key}: full-grid noise rows differ from the full-grid group",
                    )
                del container_array

    def test_one_noise_row_serves_every_cell(self):
        # * The fixed-PSD case: many cells, one noise row, so the noise index repeats across cells.
        for config_name, response in CASES:
            with self.subTest(config=config_name, response=response):
                fixture = self.build(config_name, response)
                comp = fixture.comp
                xp = comp.xp
                windowed = comp.stft_comps
                rows = full_grid_noise_rows(self.fixtures, fixture, windowed)
                # The store windows do not overlap, so one row can hold the windows of both cells.
                width = fixture.description["cell_window_width"]
                shared = rows[:1].copy()
                for cell, start in enumerate(fixture.description["cell_window_starts"]):
                    shared[0, ..., start:start + width] = rows[cell, ..., start:start + width]
                one_row = full_grid_noise_group(fixture, windowed, shared)

                params = fixture.params
                cells = xp.asarray(fixture.cells)
                row_zero = xp.zeros_like(cells)
                partners = params[self.golden.swap_partners(fixture.cells)]
                starts = fixture.window_starts

                def outputs(noise_index):
                    log_like = comp.get_ll_stft(
                        params, data_index=cells, noise_index=noise_index, start_freq_inds=starts
                    )
                    values = dict(ll=log_like, d_h=comp.d_h_out, h_h=comp.h_h_out)
                    swap = comp.get_swap_ll_stft(
                        params, partners, data_index=cells, noise_index=noise_index,
                        start_freq_inds=starts,
                    )
                    values.update(("swap_" + str(index), term) for index, term in enumerate(swap))
                    values["grad"] = comp.get_ll_grad_stft(
                        params, data_index=cells, noise_index=noise_index, start_freq_inds=starts
                    )
                    return {key: self.fixtures.to_host(value) for key, value in values.items()}

                reference = outputs(cells)
                comp.stft_comps = one_row
                try:
                    candidate = outputs(row_zero)
                finally:
                    comp.stft_comps = windowed
                for key, value in reference.items():
                    np.testing.assert_array_equal(
                        candidate[key], value,
                        err_msg=f"{key} differs between one shared noise row and per-cell windows",
                    )

    def test_narrow_full_grid_rows_are_rejected(self):
        fixture = self.build(*CASES[0])
        group = fixture.comp.stft_comps
        width = fixture.description["cell_window_width"]
        with self.assertRaises((ValueError, RuntimeError)):
            group.backend.STFTDomainWrap(
                *group.domain_args, num_freqs_noise=width - 1, noise_full_grid=True
            )

    def test_array_on_noise_source_matches_windowed_noise(self):
        for config_name, response in CASES:
            with self.subTest(config=config_name, response=response):
                fixture = self.build(config_name, response)
                comp = fixture.comp
                windowed = comp.stft_comps
                rows = full_grid_noise_rows(self.fixtures, fixture, windowed)
                array = noise_source_array(self.fixtures, fixture, windowed, rows, self.BACKEND)
                group = array.cpp_splits[0]
                self.assertIsNone(group.d_d, msg="an array on external rows took a (d|d) snapshot")
                # The data rows are the fixture's, so its (d|d) applies.
                group.d_d = windowed.d_d

                reference = self.golden.evaluate(fixture)
                comp.stft_comps = group
                try:
                    candidate = self.golden.evaluate(fixture)
                finally:
                    comp.stft_comps = windowed
                compared = [
                    key for key in reference if not key.startswith("fisher") and key != "fill"
                ]
                self.assertIn("sw_ar", compared)
                for key in compared:
                    np.testing.assert_array_equal(
                        candidate[key], reference[key],
                        err_msg=f"{key} differs between the noise-source array and windowed noise",
                    )

                # * The Fisher matrix contracts the noise rows only: it must equal the full grid's.
                _, full_grid = full_grid_group(self.fixtures, fixture, rows, self.BACKEND)
                sources = self.golden.first_sources(fixture.cells, self.golden.NUM_FISHER_SOURCES)
                cells = comp.xp.asarray(fixture.cells[sources])
                for key, options in FISHER_VARIANTS.items():
                    np.testing.assert_array_equal(
                        self.fixtures.to_host(comp.information_matrix(
                            fixture.params[sources], holder=array, noise_index=cells, **options)),
                        self.fixtures.to_host(comp.information_matrix(
                            fixture.params[sources], holder=full_grid, noise_index=cells, **options)),
                        err_msg=f"{key}: the noise-source array differs from the full-grid group",
                    )

    def test_noise_source_layout_and_repointing(self):
        fixture = self.build(*CASES[0])
        windowed = fixture.comp.stft_comps
        rows = full_grid_noise_rows(self.fixtures, fixture, windowed)
        num_cells = rows.shape[0]
        array = noise_source_array(self.fixtures, fixture, windowed, rows, self.BACKEND)
        source = array.linear_psd_arr[0]
        source_before = self.fixtures.to_host(source).copy()
        source_start, source_bytes = address_range(source)

        self.assertTrue(array.external_noise_rows)
        self.assertEqual(array.num_noise_rows(0), num_cells)
        self.assertEqual(tuple(array.psd_shaped[0].shape), rows.shape)
        for cell in range(num_cells):
            invc = array.acs[cell].sens_mat.invC
            view_start, view_bytes = address_range(invc)
            self.assertTrue(
                source_start <= view_start and view_start + view_bytes <= source_start + source_bytes,
                msg=f"container {cell} does not read a view of the source",
            )
            np.testing.assert_array_equal(self.fixtures.to_host(invc), rows[cell])

        group = array.cpp_splits[0]
        self.assertEqual(group.num_noise, num_cells)
        self.assertEqual(len(group.domain_args), 15)
        self.assertEqual(list(group.domain_args[-2:]), [rows.shape[-1], True])

        # ! A repack must leave the rows alone: they belong to the source.
        array.reset_linear_psd_arr()
        np.testing.assert_array_equal(self.fixtures.to_host(source), source_before)

        tables = list(array.noise_row_by_split)
        array.set_noise_rows([0, 1], [1, 0])
        self.assertEqual([id(table) for table in array.noise_row_by_split],
                         [id(table) for table in tables],
                         msg="set_noise_rows rebound a device table instead of writing it in place")
        np.testing.assert_array_equal(self.fixtures.to_host(array.noise_row_by_split[0]), [1, 0])
        np.testing.assert_array_equal(array.noise_row_of_ac, [1, 0])
        np.testing.assert_array_equal(self.fixtures.to_host(array.acs[0].sens_mat.invC), rows[1])
        with self.assertRaises(ValueError):
            array.set_noise_rows([0], [num_cells])
        array.set_noise_rows([0, 1], [0, 1])
        np.testing.assert_array_equal(self.fixtures.to_host(array.acs[0].sens_mat.invC), rows[0])

    def test_noise_source_is_checked(self):
        fixture = self.build(*CASES[0])
        windowed = fixture.comp.stft_comps
        rows = full_grid_noise_rows(self.fixtures, fixture, windowed)
        xp = self.fixtures.array_module(self.BACKEND)
        rows_flat = xp.ascontiguousarray(xp.asarray(rows.reshape(-1)))
        bad_sources = dict(
            two_arrays_for_one_split=dict(arrays=[rows_flat, rows_flat]),
            real_rows=dict(arrays=[rows_flat.real.copy()]),
            partial_row=dict(arrays=[rows_flat[:-1].copy()]),
            row_out_of_range=dict(row_of_ac=np.asarray([0, rows.shape[0]])),
            rows_for_one_container=dict(row_of_ac=np.asarray([0])),
        )
        for name, changes in bad_sources.items():
            with self.subTest(source=name):
                with self.assertRaises(ValueError):
                    noise_source_array(
                        self.fixtures, fixture, windowed, rows, self.BACKEND, **changes
                    )


class NoiseLayoutCPUTest(_NoiseLayoutCases, unittest.TestCase):
    BACKEND = "cpu"

    def test_noise_source_with_cpu_splits(self):
        # * One container per CPU split: each split's group must read the row its container names.
        fixture = self.build(*CASES[0])
        comp = fixture.comp
        windowed = comp.stft_comps
        rows = full_grid_noise_rows(self.fixtures, fixture, windowed)
        array = noise_source_array(self.fixtures, fixture, windowed, rows, "cpu", n_splits=2)
        self.assertEqual(len(array.cpp_splits), 2)
        _, data_intra, noise_intra = array.unpack_indices(np.asarray([0, 1]), np.asarray([0, 1]))
        self.assertEqual([list(rows_here) for rows_here in noise_intra], [[0], [1]])
        self.assertEqual([list(rows_here) for rows_here in data_intra], [[0], [0]])

        params, cells, starts = fixture.params, fixture.cells, fixture.window_starts
        comp.get_ll_stft(params, data_index=cells, noise_index=cells, start_freq_inds=starts)
        reference = dict(d_h=comp.d_h_out.copy(), h_h=comp.h_h_out.copy())
        reference["grad"] = comp.get_ll_grad_stft(
            params, data_index=cells, noise_index=cells, start_freq_inds=starts
        )
        for split in (0, 1):
            sources = np.flatnonzero(cells == split)
            data_index = np.zeros(sources.size, dtype=np.int32)
            noise_index = np.full(sources.size, split, dtype=np.int32)
            split_starts = starts[split:split + 1]
            comp.stft_comps = array.cpp_splits[split]
            try:
                comp.get_ll_stft(params[sources], data_index=data_index, noise_index=noise_index,
                                 start_freq_inds=split_starts)
                candidate = dict(d_h=comp.d_h_out, h_h=comp.h_h_out)
                candidate["grad"] = comp.get_ll_grad_stft(
                    params[sources], data_index=data_index, noise_index=noise_index,
                    start_freq_inds=split_starts,
                )
            finally:
                comp.stft_comps = windowed
            for key, value in candidate.items():
                np.testing.assert_array_equal(
                    value, reference[key][sources],
                    err_msg=f"split {split}: {key} differs from the windowed group",
                )


@unittest.skipUnless(_cuda_usable(), "requires a usable CUDA backend")
class NoiseLayoutCudaTest(_NoiseLayoutCases, unittest.TestCase):
    BACKEND = "gpu"


if __name__ == "__main__":
    unittest.main()
