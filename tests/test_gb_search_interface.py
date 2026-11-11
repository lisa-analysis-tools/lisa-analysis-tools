"""GB walkers seed from a galaxy search candidates file: a ``point_estimate`` search over a catalogue table.

The file holds one dataset per catalogue column at its root, with the column names of the mojito GB
catalogue, and the search's own columns in a group, as ``data/gb_candidates.hdf5`` does. The values
are at the catalogue's reference epoch, which the STFT GB computations use, so the search does not
evolve them. The prior is the erebor-priors stock GB prior; the test runs on the CPU.

A catalogue table follows the shape rules of a ``.npy`` of physical values (``read_samples_from_file``): in a
single file the rows are sources, in a directory of per-source files they are samples of that source, and columns
of shape ``(num_sources, num_samples)`` are taken as they are.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import h5py
import numpy as np

MOJITO_REFERENCE_TIME_S = 97729089.327664
# * Two candidates, physical values at the catalogue's reference epoch.
CANDIDATE_COLUMNS = dict(
    Amplitude=[1.0e-22, 3.0e-23],
    GW22FrequencySSBFrame=[2.5e-3, 7.0e-3],
    GW22FrequencyDerivativeSourceFrame=[1.0e-17, -2.0e-18],
    TrueAnomaly=[1.0, 7.0],  # 7 rad wraps to 7 - 2 pi
    InclinationAngle=[0.4, 2.0],
    PolarisationAngle=[2.0, -1.0],  # -1 rad wraps to 2 pi - 1
    RightAscension=[4.0, 0.5],
    Declination=[-0.2, 1.1],
    TimeReferenceSSBFrame=[MOJITO_REFERENCE_TIME_S, MOJITO_REFERENCE_TIME_S],
    ID=[0, 1],
)
# ? Sampling basis: ln A, f0 [mHz], fdot, phi0, cos iota, psi, alpha, sin delta.
SPREAD = [1.0e-3, 1.0e-9, 1.0e-19, 1.0e-2, 1.0e-2, 1.0e-2, 1.0e-3, 1.0e-3]
NUM_SAMPLES = 3


def sampling_rows(columns: dict) -> np.ndarray:
    """The expected sampling-basis values of catalogue columns, over the columns' shape (from the GB prior basis)."""
    columns = {name: np.asarray(values) for name, values in columns.items()}
    two_pi = 2 * np.pi
    return np.stack(
        [
            np.log(columns["Amplitude"]),
            columns["GW22FrequencySSBFrame"] * 1e3,
            columns["GW22FrequencyDerivativeSourceFrame"],
            columns["TrueAnomaly"] % two_pi,
            np.cos(columns["InclinationAngle"]),
            columns["PolarisationAngle"] % two_pi,
            columns["RightAscension"] % two_pi,
            np.sin(columns["Declination"]),
        ],
        axis=-1,
    )


def samples_of(columns: dict) -> dict:
    """``NUM_SAMPLES`` samples of each candidate: columns of shape ``(num_sources, NUM_SAMPLES)``, f0 nudged."""
    out = {name: np.repeat(np.asarray(values)[:, None], NUM_SAMPLES, axis=1) for name, values in columns.items()}
    out["GW22FrequencySSBFrame"] = out["GW22FrequencySSBFrame"] * (1.0 + 1e-6 * np.arange(NUM_SAMPLES))
    return out


def write_table(path: str, columns: dict) -> str:
    with h5py.File(path, "w") as handle:
        for column_name, values in columns.items():
            handle.create_dataset(column_name, data=np.asarray(values))
        # ? The search's own columns sit in a group, which the catalogue reading skips.
        handle.create_dataset("Search/SignalToNoiseRatio", data=np.full(np.shape(columns["Amplitude"]), 8.0))
    return path


class GBCandidatesSeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ereborpriors.stock.mojitolight import GBPrior

        cls.folder = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.folder.cleanup)
        cls.candidates_file = write_table(os.path.join(cls.folder.name, "gb_candidates.hdf5"), CANDIDATE_COLUMNS)
        cls.prior = GBPrior.prior_factory()

    def make_interface(self, search_path, search_type="point_estimate"):
        from lisatools.globalfit.searches import GBSearchInterface

        return GBSearchInterface(
            search_type=search_type,
            search_path=search_path,
            spread=SPREAD,
            transform=self.prior.get_transforms(),
            periodic={"gb": self.prior.get_periodic()},
            evolve_to_data_start=False,
        )

    def read(self, search_path):
        return self.make_interface(search_path).read_samples_from_file(curr=None)

    def test_rows_of_a_file_are_sources(self):
        seed = self.read(self.candidates_file)
        self.assertEqual(seed.shape, (2, 1, 8))  # (num_leaves, num_samples, num_dim), as a 2-D .npy
        np.testing.assert_allclose(seed[:, 0], sampling_rows(CANDIDATE_COLUMNS), rtol=1e-12)
        self.assertTrue(np.isfinite(self.prior.logpdf(seed[:, 0])).all())

    def test_rows_of_a_per_source_file_are_samples(self):
        folder = os.path.join(self.folder.name, "per_source")
        os.makedirs(folder, exist_ok=True)
        samples = samples_of(CANDIDATE_COLUMNS)
        for source in range(2):
            write_table(os.path.join(folder, f"candidate_{source}.h5"),
                        {name: values[source] for name, values in samples.items()})
        seed = self.read(folder)
        self.assertEqual(seed.shape, (2, NUM_SAMPLES, 8))
        np.testing.assert_allclose(seed, sampling_rows(samples), rtol=1e-12)

    def test_two_dimensional_columns_are_samples_per_source(self):
        samples = samples_of(CANDIDATE_COLUMNS)
        seed = self.read(write_table(os.path.join(self.folder.name, "samples.h5"), samples))
        self.assertEqual(seed.shape, (2, NUM_SAMPLES, 8))  # as a 3-D .npy: taken as it is
        np.testing.assert_allclose(seed, sampling_rows(samples), rtol=1e-12)

    def test_a_single_row(self):
        one = {name: values[:1] for name, values in CANDIDATE_COLUMNS.items()}
        seed = self.read(write_table(os.path.join(self.folder.name, "one.h5"), one))
        self.assertEqual(seed.shape, (1, 1, 8))
        np.testing.assert_allclose(seed[0, 0], sampling_rows(one)[0], rtol=1e-12)

    def test_walkers_scatter_inside_the_prior(self):
        draws = self.make_interface(self.candidates_file).generate_initial_draws(
            curr=None, num_walkers=4, betas=np.array([1.0, 0.5]), priors={"gb": self.prior}
        )
        self.assertEqual(draws.shape, (2, 4, 2, 8))
        self.assertTrue(np.isfinite(self.prior.logpdf(draws.reshape(-1, 8))).all())

    def test_samples_need_several_points_per_source(self):
        interface = self.make_interface(self.candidates_file, search_type="samples")
        with self.assertRaisesRegex(ValueError, "one point per source.*use search_type point_estimate"):
            interface.generate_initial_draws(curr=None, num_walkers=4, betas=np.array([1.0]), priors={"gb": self.prior})

    def test_a_catalog_search_reads_no_file(self):
        # * catalog seeds from the injection catalogue: a search_path would be ignored, so it is refused
        with self.assertRaisesRegex(ValueError, "catalog seeds from the injection catalogue"):
            self.make_interface(self.candidates_file, search_type="catalog")

    def test_columns_must_share_one_shape(self):
        mixed = dict(CANDIDATE_COLUMNS, Amplitude=[1.0e-22, 3.0e-23, 2.0e-23])
        with self.assertRaisesRegex(ValueError, "must share one shape"):
            self.read(write_table(os.path.join(self.folder.name, "mixed.h5"), mixed))

    def test_file_without_root_columns_is_refused(self):
        # * The galaxy search writes its columns in a Binaries group; they must sit at the root.
        grouped_file = os.path.join(self.folder.name, "grouped_candidates.hdf5")
        with h5py.File(grouped_file, "w") as handle:
            handle.create_dataset("Binaries/Amplitude", data=[1.0e-22])
        with self.assertRaisesRegex(ValueError, "no catalogue columns at its root"):
            self.read(grouped_file)


if __name__ == "__main__":
    unittest.main()
