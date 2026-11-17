"""The GB setup of the full-band DFFR run, as its settings file builds it today.

``get_gb_settings`` of ``mojito_input/galaxy_psd_dffr_global_fit_settings.py`` runs with a stand-in
general setup on the run's STFT grid (713 one-day segments, 0.1 to 30 mHz, 5 s sampling, Tukey taper
of 1e4 s), built from a synthetic time grid: no data is loaded, and the test runs on the CPU in a
few seconds. The recorded values are the reference for the clean-up of ``GBSettings`` and
``GBSetup``. The prior is the erebor-priors stock prior, with its (f0, fdot) entry over the bands
the moves propose in.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

SETTINGS_FILE = (
    Path(__file__).resolve().parents[1] / "mojito_input" / "galaxy_psd_dffr_global_fit_settings.py"
)

# * The run's STFT grid. The data start is the Mojito catalogue epoch plus the 720850.5 s lag of the
# * 714-day resolved-source run.
SEGMENT_DURATION_S = 86400.0
NUM_SEGMENTS = 713
SAMPLING_INTERVAL_S = 5.0
MIN_FREQ_HZ = 1.0e-4
MAX_FREQ_HZ = 30.0e-3
TAPER_DURATION_S = 1.0e4
DATA_START_S = 97729089.327664 + 720850.5
NUM_TEMPERATURES = 24

# * Recorded from the code of 2026-11-03.
NUM_BANDS = 1733
FIRST_EDGE_HZ = 0.00010416666666666666
SECOND_EDGE_HZ = 0.00010564221618036097
SECOND_LAST_EDGE_HZ = 0.026346158882316065
LAST_EDGE_HZ = 0.2179904150724717
OBSERVATION_TIME_S = 61603200.0
WINDOW_ALPHA = 0.11574074074074074
WAVEFORM_KWARGS = dict(
    dt=5.0,
    T=61603200.0,
    use_c_implementation=True,
    start_freq_ind=6417,
    tdi_channel_setup="XYZ",
    tdi2=True,
    oversample=4,
    window="tukey",
    window_alpha=0.11574074074074074,
)
STFT_COMPUTATION_KWARGS = dict(
    n_side_bins=16,
    freq_from_tdi_phase=True,
    use_midpoint=True,
    linear_envelope=True,
    response="dffr",
    first_order_delays=True,
)


def load_settings_module():
    """Import the DFFR settings file as a module (it is a script, not part of the package)."""
    spec = importlib.util.spec_from_file_location(
        "galaxy_psd_dffr_global_fit_settings", SETTINGS_FILE
    )
    settings_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings_module)
    return settings_module


def make_general_setup(run_folder: str) -> types.SimpleNamespace:
    """A stand-in for the run's GeneralSetup, with the attributes the GB settings read."""
    from lisatools.domains import STFTSettings

    num_samples = int(NUM_SEGMENTS * SEGMENT_DURATION_S / SAMPLING_INTERVAL_S)
    times = DATA_START_S + np.arange(num_samples) * SAMPLING_INTERVAL_S
    make_domain = STFTSettings.make_factory(
        big_dt=SEGMENT_DURATION_S, min_freq=MIN_FREQ_HZ, max_freq=MAX_FREQ_HZ
    )
    domain_settings = make_domain(times=times, dt=SAMPLING_INTERVAL_S, force_backend="cpu")
    # * The taper as a fraction of one segment, as GeneralSetup derives it.
    window_alpha = TAPER_DURATION_S / (
        domain_settings.get_nperseg(SAMPLING_INTERVAL_S) * SAMPLING_INTERVAL_S
    )
    return types.SimpleNamespace(
        domain_settings=domain_settings,
        Tobs=num_samples * SAMPLING_INTERVAL_S,
        dt=SAMPLING_INTERVAL_S,
        data_t0=DATA_START_S,
        window_type="tukey",
        window_alpha=window_alpha,
        # ? Placeholders: the GB settings pass the orbits on and never read them.
        orbits=object(),
        gpu_orbits=object(),
        # * A CPU run: no GPUs, so the GBGPU arguments take the host orbits.
        gpus=None,
        gpu_backend="cpu",
        force_backend="cpu",
        ntemps=NUM_TEMPERATURES,
        file_store_dir=os.path.join(run_folder, ""),
        artifacts_file_dir=os.path.join(run_folder, "artifacts", ""),
    )


# ? Where each checked value lives on the setup: its waveform, STFT and moves groups.
def gbgpu_arguments(gb_setup) -> tuple[dict, dict]:
    """The GBGPU constructor arguments and the waveform call arguments."""
    return gb_setup.waveform.initialize_kwargs, gb_setup.waveform.waveform_kwargs


def stft_arguments(gb_setup) -> dict:
    """What the band structure and the STFT computation are built from."""
    return dict(
        domain_settings=gb_setup.stft.domain_settings,
        band_alpha=gb_setup.stft.band_alpha,
        band_kwargs=gb_setup.stft.band_kwargs,
        computation_kwargs=gb_setup.stft.computation_kwargs,
    )


def move_settings(gb_setup) -> dict:
    """The settings that build_gb_moves passes to every GB move."""
    move_group = gb_setup.moves
    return dict(
        num_repeat_proposals=move_group.num_repeat_proposals,
        stretch_probability=move_group.stretch_probability,
        num_stationary_fim_steps=move_group.num_stationary_fim_steps,
        num_band_preload=move_group.num_band_preload,
        num_bands_preload_temp=move_group.num_bands_preload_temp,
        debug_settings=move_group.debug_settings,
        leaf_cap_settings=move_group.leaf_cap_settings,
        sighet_settings=move_group.sighet_settings,
    )


class DFFRGBSetupReferenceTest(unittest.TestCase):
    """The DFFR GB setup keeps its bands, arguments, moves, ladder, transform and metadata."""

    @classmethod
    def setUpClass(cls):
        cls.run_folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.run_folder.cleanup)
        cls.general_setup = make_general_setup(cls.run_folder.name)
        cls.settings_module = load_settings_module()
        cls.gb_setup, cls.gb_metadata = cls.build(mem_probe_file=None)

    @classmethod
    def build(cls, mem_probe_file: str | None):
        return cls.settings_module.get_gb_settings(cls.general_setup, mem_probe_file=mem_probe_file)

    def test_band_structure(self):
        from lisatools.globalfit.moves.gbbandstructure import stft_band_structure

        band_edges = np.asarray(self.gb_setup.band_edges)
        self.assertEqual(len(band_edges) - 1, NUM_BANDS)
        self.assertEqual(
            [band_edges[0], band_edges[1], band_edges[-2], band_edges[-1]],
            [FIRST_EDGE_HZ, SECOND_EDGE_HZ, SECOND_LAST_EDGE_HZ, LAST_EDGE_HZ],
        )
        # * Every edge is the band-structure recipe on the run's grid and taper.
        recipe = stft_band_structure(self.general_setup.domain_settings, WINDOW_ALPHA)
        np.testing.assert_array_equal(band_edges, np.asarray(recipe.edges_hz))
        # ? The STFT engine sizes its buffers from the domain: no per-band sample counts.
        self.assertIsNone(self.gb_setup.band_N_vals)

    def test_sampled_f0_range_is_the_inner_bands(self):
        # * The moves never propose in the first and the last band.
        self.assertEqual(
            [float(limit) for limit in self.gb_setup.f0_lims],
            [SECOND_EDGE_HZ, SECOND_LAST_EDGE_HZ],
        )

    def test_prior_is_the_stock_prior_over_the_proposed_bands(self):
        from ereborpriors.priors import F0FdotPrior
        from ereborpriors.stock.mojitolight import GBPrior

        gb_prior = self.gb_setup.priors["gb"]
        self.assertIsInstance(gb_prior, GBPrior)
        joint_prior = gb_prior.priors_in[("f0", "fdot")]
        self.assertIsInstance(joint_prior, F0FdotPrior)
        # * f0 in mHz over the inner bands, where the moves propose.
        self.assertEqual(
            (joint_prior.f0_min, joint_prior.f0_max),
            (SECOND_EDGE_HZ * 1e3, SECOND_LAST_EDGE_HZ * 1e3),
        )
        # * Every other entry keeps the stock bounds.
        stock_entries = GBPrior.prior_factory().priors_in
        for name in ("logA", "phi0", "cos_iota", "psi", "alpha", "sin_delta"):
            self.assertEqual(
                (gb_prior.priors_in[name].minimum, gb_prior.priors_in[name].maximum),
                (stock_entries[name].minimum, stock_entries[name].maximum),
            )

    def test_gbgpu_arguments(self):
        initialize_kwargs, waveform_kwargs = gbgpu_arguments(self.gb_setup)
        self.assertEqual(self.gb_setup.Tobs, OBSERVATION_TIME_S)
        self.assertEqual(self.gb_setup.dt, SAMPLING_INTERVAL_S)
        self.assertEqual(waveform_kwargs, WAVEFORM_KWARGS)
        self.assertEqual(sorted(initialize_kwargs), ["force_backend", "orbits", "t0"])
        self.assertIs(initialize_kwargs["orbits"], self.general_setup.orbits)
        self.assertEqual(initialize_kwargs["t0"], DATA_START_S)
        self.assertEqual(initialize_kwargs["force_backend"], "cpu")

    def test_stft_arguments(self):
        arguments = stft_arguments(self.gb_setup)
        self.assertIs(arguments["domain_settings"], self.general_setup.domain_settings)
        self.assertEqual(arguments["band_alpha"], WINDOW_ALPHA)
        self.assertIsNone(arguments["band_kwargs"])
        self.assertEqual(arguments["computation_kwargs"], STFT_COMPUTATION_KWARGS)

    def test_move_names(self):
        move_group = self.gb_setup.moves
        self.assertEqual(list(move_group.pe_move_names), ["rj_prior"])
        self.assertEqual(list(move_group.search_move_names), [])
        self.assertFalse(move_group.include_refit)
        self.assertEqual(move_group.Tmax, 1e6)

    def test_move_settings(self):
        from lisatools.globalfit.moves.gbdebug import GBDebugSettings

        self.assertEqual(
            move_settings(self.gb_setup),
            dict(
                num_repeat_proposals=50,
                stretch_probability=0.0,
                num_stationary_fim_steps=1,
                num_band_preload=None,
                num_bands_preload_temp=None,
                # ? The settings file passes the debug defaults, with the memory probe off.
                debug_settings=GBDebugSettings(),
                leaf_cap_settings=None,
                sighet_settings=None,
            ),
        )

    def test_memory_probe_switch(self):
        probe_file = os.path.join(self.run_folder.name, "mem_probe.jsonl")
        gb_setup, _ = self.build(mem_probe_file=probe_file)
        debug_settings = move_settings(gb_setup)["debug_settings"]
        self.assertTrue(debug_settings.mem_probe)
        self.assertEqual(debug_settings.mem_probe_file, probe_file)

    def test_leaves_and_ladder(self):
        self.assertEqual(
            (self.gb_setup.nleaves_max, self.gb_setup.nleaves_min, self.gb_setup.ndim), (8000, 0, 8)
        )
        # * Geometric ladder with ratio 1.2; the hottest rung is fixed at 1e-4.
        expected_betas = 1.0 / 1.2 ** np.arange(NUM_TEMPERATURES)
        expected_betas[-1] = 1.0e-4
        np.testing.assert_array_equal(self.gb_setup.betas, expected_betas)

    def test_transform_and_periodic_map(self):
        # * Sampling basis: ln A, f0 [mHz], fdot, phi0, cos iota, psi, alpha, sin delta.
        sampling_point = np.array([[-50.0, 2.5, 1.0e-17, 1.0, 0.3, 2.0, 4.0, -0.2]])
        # * Waveform basis: A, f0 [Hz], fdot, fddot (filled with 0), phi0, iota, psi, alpha, delta.
        expected_point = np.array(
            [[np.exp(-50.0), 2.5e-3, 1.0e-17, 0.0, 1.0, np.arccos(0.3), 2.0, 4.0, np.arcsin(-0.2)]]
        )
        np.testing.assert_allclose(
            self.gb_setup.transform.both_transforms(sampling_point),
            expected_point,
            rtol=1e-15,
            atol=0.0,
        )
        two_pi = 2 * np.pi
        self.assertEqual(
            self.gb_setup.periodic, {"gb": {"phi0": two_pi, "psi": two_pi, "alpha": two_pi}}
        )

    def test_source_metadata(self):
        band_edges = np.asarray(self.gb_setup.band_edges)
        self.assertEqual(self.gb_metadata.source_type, "gb")
        self.assertEqual(self.gb_metadata.waveform_model, "GBGPU")
        self.assertEqual(
            self.gb_metadata.frequency_ranges, list(zip(band_edges[:-1], band_edges[1:]))
        )
        self.assertEqual(self.gb_metadata.waveform_model_config["runtime_kwargs"], WAVEFORM_KWARGS)
        # ? The metadata is written to disk: no orbits, and the CPU backend.
        self.assertEqual(
            self.gb_metadata.waveform_model_config["init_kwargs"],
            dict(t0=DATA_START_S, force_backend="cpu"),
        )


class OtherDomainsTest(unittest.TestCase):
    """GBSetup builds STFT bands only for now and refuses other domains with a clear error."""

    def test_fd_domain_is_refused(self):
        from lisatools.domains import FDSettings
        from lisatools.globalfit.stock.erebor import GBSettings, GBSetup, GBSTFTSettings

        # * One day of 5 s samples is enough for an FD grid.
        times = DATA_START_S + np.arange(17280) * SAMPLING_INTERVAL_S
        make_domain = FDSettings.make_factory(min_freq=MIN_FREQ_HZ, max_freq=MAX_FREQ_HZ)
        fd_domain = make_domain(times=times, dt=SAMPLING_INTERVAL_S, force_backend="cpu")
        with tempfile.TemporaryDirectory() as run_folder:
            gb_settings = GBSettings(
                Tobs=86400.0,
                dt=SAMPLING_INTERVAL_S,
                log_dir=run_folder,
                stft=GBSTFTSettings(domain_settings=fd_domain),
            )
            with self.assertRaisesRegex(NotImplementedError, "come at a later stage"):
                GBSetup(gb_settings)


class GBSetupPriorChecksTest(unittest.TestCase):
    """GBSetup refuses a missing prior and a prior whose columns are in another order."""

    @classmethod
    def setUpClass(cls):
        from lisatools.domains import STFTSettings

        cls.run_folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.run_folder.cleanup)
        # * 30 one-day segments keep the band structure quick.
        num_samples = int(30 * SEGMENT_DURATION_S / SAMPLING_INTERVAL_S)
        times = DATA_START_S + np.arange(num_samples) * SAMPLING_INTERVAL_S
        make_domain = STFTSettings.make_factory(
            big_dt=SEGMENT_DURATION_S, min_freq=MIN_FREQ_HZ, max_freq=MAX_FREQ_HZ
        )
        cls.domain_settings = make_domain(times=times, dt=SAMPLING_INTERVAL_S, force_backend="cpu")
        cls.observation_time_s = num_samples * SAMPLING_INTERVAL_S

    def build(self, priors):
        from lisatools.globalfit.stock.erebor import GBSettings, GBSetup, GBSTFTSettings

        return GBSetup(
            GBSettings(
                Tobs=self.observation_time_s,
                dt=SAMPLING_INTERVAL_S,
                log_dir=self.run_folder.name,
                stft=GBSTFTSettings(domain_settings=self.domain_settings, band_alpha=WINDOW_ALPHA),
                priors=priors,
            )
        )

    def test_missing_prior_is_refused(self):
        with self.assertRaisesRegex(ValueError, "priors"):
            self.build(priors=None)

    def test_reordered_columns_are_refused(self):
        from ereborpriors.stock.mojitolight import GBPrior

        stock_entries = GBPrior.prior_factory().priors_in
        # * psi before phi0: the GB code would read the wrong columns.
        reordered_names = ("logA", ("f0", "fdot"), "psi", "cos_iota", "phi0", "alpha", "sin_delta")
        reordered_prior = GBPrior(priors_in={name: stock_entries[name] for name in reordered_names})
        with self.assertRaisesRegex(ValueError, "sampling basis"):
            self.build(priors={"gb": reordered_prior})


class GBBandsFollowThePriorTest(unittest.TestCase):
    """The bands are the ones the prior's f0 range reaches, plus a guard band on each side; the prior never widens."""

    @classmethod
    def setUpClass(cls):
        cls.run_folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.run_folder.cleanup)
        # * 30 one-day segments: narrow bands, so a prior can drop some at both ends.
        cls.num_samples = int(30 * SEGMENT_DURATION_S / SAMPLING_INTERVAL_S)

    def build(self, prior, max_freq_hz):
        from lisatools.domains import STFTSettings
        from lisatools.globalfit.moves.gbbandstructure import stft_band_structure
        from lisatools.globalfit.stock.erebor import GBSettings, GBSetup, GBSTFTSettings

        times = DATA_START_S + np.arange(self.num_samples) * SAMPLING_INTERVAL_S
        make_domain = STFTSettings.make_factory(
            big_dt=SEGMENT_DURATION_S, min_freq=MIN_FREQ_HZ, max_freq=max_freq_hz
        )
        domain_settings = make_domain(times=times, dt=SAMPLING_INTERVAL_S, force_backend="cpu")
        gb_setup = GBSetup(
            GBSettings(
                Tobs=self.num_samples * SAMPLING_INTERVAL_S,
                dt=SAMPLING_INTERVAL_S,
                log_dir=self.run_folder.name,
                stft=GBSTFTSettings(domain_settings=domain_settings, band_alpha=WINDOW_ALPHA),
                priors={"gb": prior},
            )
        )
        full_walk = np.asarray(stft_band_structure(domain_settings, WINDOW_ALPHA).edges_hz)
        return gb_setup, full_walk

    def assert_contiguous_part_of(self, band_edges, full_walk):
        first_edge = int(np.flatnonzero(full_walk == band_edges[0])[0])
        np.testing.assert_array_equal(band_edges, full_walk[first_edge:first_edge + len(band_edges)])

    def test_a_narrow_prior_keeps_its_bands_and_one_guard_band_each_side(self):
        from ereborpriors.priors import F0FdotPrior
        from ereborpriors.stock.mojitolight import GBPrior

        stock_prior = GBPrior.prior_factory()
        priors_in = dict(stock_prior.priors_in)
        priors_in[("f0", "fdot")] = F0FdotPrior(f0_min=2.0, f0_max=5.0, f0_units="mHz")
        narrow_prior = GBPrior(priors_in=priors_in, include_fddot=stock_prior.include_fddot)
        gb_setup, full_walk = self.build(narrow_prior, MAX_FREQ_HZ)

        band_edges = np.asarray(gb_setup.band_edges)
        self.assertLess(len(band_edges), len(full_walk))
        self.assert_contiguous_part_of(band_edges, full_walk)
        # * The prior's ends lie in the second and the second-last band: the guard bands hold none of it.
        self.assertTrue(band_edges[1] <= 2.0e-3 < band_edges[2])
        self.assertTrue(band_edges[-3] <= 5.0e-3 < band_edges[-2])
        self.assertEqual([float(limit) for limit in gb_setup.f0_lims], [2.0e-3, 5.0e-3])

    def test_the_stock_prior_keeps_its_top_on_a_wider_grid(self):
        from ereborpriors.stock.mojitolight import GBPrior

        gb_setup, full_walk = self.build(GBPrior.prior_factory(), 0.09999)
        band_edges = np.asarray(gb_setup.band_edges)
        self.assert_contiguous_part_of(band_edges, full_walk)
        # * The bands above the one holding 30 mHz and its guard band are dropped.
        self.assertTrue(band_edges[-3] <= 30.0e-3 < band_edges[-2])
        self.assertLess(band_edges[-1], full_walk[-1])
        self.assertEqual(float(gb_setup.f0_lims[1]), 30.0e-3)
        joint_prior = gb_setup.priors["gb"].priors_in[("f0", "fdot")]
        self.assertEqual(joint_prior.f0_max, 30.0)


if __name__ == "__main__":
    unittest.main()
