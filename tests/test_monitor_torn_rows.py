"""One unreadable chain row must not kill the monitor page.

2026-10-02, 6mo cluster, ``python -m lisatools.globalfit.monitor`` against
the LIVE run directory:

    File ".../monitor/_generator.py", line 4132, in _leaf_f0
        return (g["chain/gb"][it_, 0, 0, w, :, 1] * 1e-3)[al]
    OSError: Can't synchronously read data (filter returned failure during read)

The chunk was not torn on disk. The same build had already read EVERY
``chain/gb`` row cleanly at the leaf-count panel (the truth set is beside the
store, so that read runs), and minutes later one of those rows would not
decompress. The saver appended in between: a reader without SWMR is reading
a file that is changing under it. ``_store_extract`` met the same thing on
2026-08-22 and falls back per read to the run's ``*_running_backup_copy.h5``;
``_safe`` does the same for whole datasets. The GB panels' per-row reads went
straight to ``g``, outside any try, so one row cost the whole page.

The generator is a script (it raises SystemExit on import), so these tests
exec its torn-read block against a real store whose chunk is torn on purpose
-- the exact HDF5 error, not a mock -- and read the source for the call sites.
"""
import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_GEN = os.path.join(_HERE, os.pardir, "src", "lisatools", "globalfit", "monitor",
                    "_generator.py")

NIT, NW, NLEAF, NDIM = 4, 2, 8, 9
TORN = (2, 0, 0, 1, 0, 1)            # row 2, walker 1, the f0 column


def _read_src():
    with open(_GEN, encoding="utf-8") as fh:
        return fh.read()


def _make_store(path):
    rng = np.random.default_rng(3)
    chain = rng.uniform(1.0, 5.0, size=(NIT, 1, 1, NW, NLEAF, NDIM))
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = NIT
        # one row per chunk on the iteration axis, one column per chunk on
        # the parameter axis: the live store's layout on those two axes
        g.create_dataset("chain/gb", data=chain, chunks=(1, 1, 1, 1, NLEAF, 1),
                         compression="gzip", compression_opts=4)
        g.create_dataset("inds/gb", data=np.ones((NIT, 1, 1, NW, NLEAF), bool),
                         compression="gzip", compression_opts=4)
    return chain


def _tear(path, coord):
    """Overwrite one compressed chunk in place, the way a racing writer does."""
    with h5py.File(path, "r") as f:
        info = f["global_fit/chain/gb"].id.get_chunk_info_by_coord(coord)
    with open(path, "r+b") as fh:
        fh.seek(info.byte_offset)
        fh.write(b"\xa5" * info.size)


class TornRowReaderTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        src = _read_src()
        # the torn-read block: _backup_group, _safe and the per-row reader
        cls.block = src[src.index("_BACKUP_G = None"):
                        src.index("# Every GB panel stands on these two")]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = os.path.join(self.tmp, "gf_prod_testing.h5")
        self.chain = _make_store(self.store)
        self.backup = self.store[:-3] + "_running_backup_copy.h5"
        shutil.copyfile(self.store, self.backup)
        _tear(self.store, TORN)
        self.files = []

    def tearDown(self):
        for f in self.files:
            try:
                f.close()
            except Exception:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ns(self):
        f = h5py.File(self.store, "r")
        self.files.append(f)
        ns = {"os": os, "h5py": h5py, "np": np, "RUN_DIR": self.tmp,
              "g": f["global_fit"], "NIT": NIT, "MISSING": []}
        exec(self.block, ns)                       # noqa: S102
        return ns

    def _close_backup(self, ns):
        # now, not at tearDown: HDF5 refuses an r+ open of a file this
        # process still holds read-only
        bg = ns.get("_BACKUP_G")
        if bg:
            bg.file.close()

    def test_the_fixture_is_the_cluster_error(self):
        with h5py.File(self.store, "r") as f:
            with self.assertRaisesRegex(OSError, "filter returned failure"):
                f["global_fit/chain/gb"][2, 0, 0, 1, :, 1]

    def test_a_torn_row_is_read_from_the_backup_copy(self):
        ns = self._ns()
        got = ns["_row"]("chain/gb", (2, 0, 0, 1, slice(None), 1))
        self._close_backup(ns)
        np.testing.assert_array_equal(got, self.chain[2, 0, 0, 1, :, 1])
        self.assertEqual(len(ns["_TORN_ROWS"]), 1)
        self.assertIn("backup", ns["_TORN_ROWS"][0])

    def test_a_healthy_row_is_read_live_and_not_reported(self):
        ns = self._ns()
        got = ns["_row"]("chain/gb", (1, 0, 0))
        np.testing.assert_array_equal(got, self.chain[1, 0, 0])
        self.assertEqual(ns["_TORN_ROWS"], [])

    def test_without_a_backup_the_row_is_skipped_not_fatal(self):
        os.remove(self.backup)
        ns = self._ns()
        self.assertIsNone(ns["_row"]("chain/gb", (2, 0, 0)))
        self.assertIn("SKIPPED", ns["_TORN_ROWS"][0])

    def test_the_backup_never_supplies_a_row_it_has_not_written(self):
        # The backup is one save behind and PREALLOCATED: a row past its
        # iteration attr reads back as zeros, which every GB panel would
        # render as "the model lost all its sources".
        with h5py.File(self.backup, "r+") as f:
            f["global_fit"].attrs["iteration"] = 2
        ns = self._ns()
        got = ns["_row"]("chain/gb", (2, 0, 0, 1, slice(None), 1))
        self._close_backup(ns)
        self.assertIsNone(got)
        self.assertIn("SKIPPED", ns["_TORN_ROWS"][0])

    def test_a_row_range_needs_every_row_in_the_backup(self):
        ns = self._ns()
        got = ns["_row"]("chain/gb", (slice(0, NIT), 0, 0))
        self._close_backup(ns)
        np.testing.assert_array_equal(got, self.chain[:NIT, 0, 0])
        with h5py.File(self.backup, "r+") as f:
            f["global_fit"].attrs["iteration"] = NIT - 1
        ns = self._ns()
        got = ns["_row"]("chain/gb", (slice(0, NIT), 0, 0))
        self._close_backup(ns)
        self.assertIsNone(got)


class CallSitesTest(unittest.TestCase):
    """Every unguarded GB row read goes through ``_row``."""

    @classmethod
    def setUpClass(cls):
        cls.src = _read_src()

    def test_no_bare_gb_row_reads_remain(self):
        # The ratchet panels keep ``_chg = g["chain/gb"]`` handles: they sit
        # inside a try that turns a failure into a MISSING line. A bare
        # ``g["chain/gb"][...]`` at module level is what killed the page.
        for pat in ('g["chain/gb"][', 'g["inds/gb"]['):
            hits = [i + 1 for i, line in enumerate(self.src.splitlines())
                    if pat in line and not line.lstrip().startswith("#")]
            self.assertEqual(hits, [], f"bare {pat}...] read on lines {hits}")

    def test_leaf_f0_reads_through_the_tolerant_reader(self):
        body = self.src[self.src.index("def _leaf_f0("):]
        body = body[:body.index("\n\n\n")]
        self.assertIn('_row("chain/gb"', body)
        self.assertIn("gb_inds[", body)

    def test_the_torn_rows_reach_the_page(self):
        i_sum = self.src.index("if _TORN_ROWS:")
        i_html = self.src.index('missing_html = "".join(')
        self.assertLess(i_sum, i_html)

    def test_a_skipped_progress_row_holds_the_previous_counts(self):
        loop = self.src[self.src.index("for _i in range(NIT):\n        _fv = _leaf_f0("):]
        loop = loop[:loop.index("_nz = np.nonzero(n_all > 0)")]
        self.assertRegex(loop, r"if _fv is None:")
        self.assertRegex(loop, r"n_all\[_i\] = n_all\[_i - 1\]")


if __name__ == "__main__":
    unittest.main()
