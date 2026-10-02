"""The WDM lookup-table store: canonical names/paths, build once and save, find on restart,
one builder under a lock while other callers wait, never rebuild over a wrong-grid table."""
from __future__ import annotations

import os
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

#: a tiny build (seconds): 4 offsets/layer x 6 layers x 3 fdot rows on an Nf=32 grid
TINY = dict(eps_freq=0.25, eps_fdot=0.5, fdot_max_factor=1.0, batch_size=16)
NF, DT = 32, 10.0


class NamesTest(unittest.TestCase):
    def test_canonical_names_are_the_existing_tables(self):
        from lisatools.wdm_lookup_store import lookup_table_name

        self.assertEqual(lookup_table_name(1440, 2.5), "wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5")
        self.assertEqual(lookup_table_name(180, 20.0), "wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5")
        self.assertEqual(lookup_table_name(720, 5.0), "wdm_lookup_emri_cx_NF720_DT5_TL32_fd8x0p01_nld2.h5")

    def test_path_pointer_wins_else_the_run_folder(self):
        from lisatools.wdm_lookup_store import lookup_table_path

        self.assertEqual(lookup_table_path("/x/t.h5", "/run", 1440, 2.5), "/x/t.h5")
        self.assertEqual(lookup_table_path(None, "/run", 1440, 2.5),
                         "/run/wdm_lookup_emri_cx_NF1440_DT2p5_TL32_fd8x0p01_nld2.h5")
        self.assertEqual(lookup_table_path("", "/run", 180, 20.0),
                         "/run/wdm_lookup_emri_cx_NF180_DT20_TL32_fd8x0p01_nld2.h5")
        with self.assertRaises(ValueError):
            lookup_table_path(None, None, 1440, 2.5)


class EnsureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lisatools.wdm_lookup_store import build_lookup_table

        cls._dir = tempfile.TemporaryDirectory()
        cls.prebuilt = os.path.join(cls._dir.name, "prebuilt.h5")
        build_lookup_table(cls.prebuilt, Nf=NF, dt=DT, recipe=TINY)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "run", "table.h5")

    def _ensure(self, **kw):
        from lisatools.wdm_lookup_store import ensure_lookup_table

        return ensure_lookup_table(self.path, Nf=kw.pop("Nf", NF), dt=DT, recipe=TINY, **kw)

    def test_missing_is_built_and_saved_then_found_on_restart(self):
        import lisatools.wdm_lookup_store as store
        from lisatools.domains import WDMLookupTable

        self.assertEqual(self._ensure(), "built")
        self.assertTrue(os.path.isfile(self.path))
        t = WDMLookupTable.from_file(self.path, force_backend="cpu")
        self.assertEqual((int(t.Nf), float(t.data_dt)), (NF, DT))
        self.assertEqual(sorted(os.listdir(os.path.dirname(self.path))), ["table.h5"])  # no lock/tmp left
        with mock.patch.object(store, "build_lookup_table",
                               side_effect=AssertionError("rebuilt on restart")):
            self.assertEqual(self._ensure(), "found")

    def test_a_table_on_another_grid_is_refused_never_overwritten(self):
        os.makedirs(os.path.dirname(self.path))
        shutil.copy(self.prebuilt, self.path)
        before = os.path.getmtime(self.path)
        with self.assertRaisesRegex(ValueError, "built for Nf=32"):
            self._ensure(Nf=64)
        self.assertEqual(os.path.getmtime(self.path), before)

    def test_waits_for_another_builder_instead_of_building(self):
        import lisatools.wdm_lookup_store as store

        os.makedirs(os.path.dirname(self.path))
        lock = self.path + ".lock"
        open(lock, "w").close()

        def other_builder():
            time.sleep(0.4)
            shutil.copy(self.prebuilt, self.path + ".tmp")
            os.replace(self.path + ".tmp", self.path)
            os.remove(lock)

        th = threading.Thread(target=other_builder)
        th.start()
        with mock.patch.object(store, "build_lookup_table",
                               side_effect=AssertionError("built twice")):
            status = self._ensure(poll=0.05)
        th.join()
        self.assertEqual(status, "waited")

    def test_a_failed_builder_lock_is_taken_over(self):
        os.makedirs(os.path.dirname(self.path))
        lock = self.path + ".lock"
        open(lock, "w").close()
        threading.Timer(0.3, os.remove, args=(lock,)).start()     # died without a table
        self.assertEqual(self._ensure(poll=0.05), "built")

    def test_a_stale_lock_is_taken_over(self):
        os.makedirs(os.path.dirname(self.path))
        lock = self.path + ".lock"
        open(lock, "w").close()
        os.utime(lock, (time.time() - 100.0, time.time() - 100.0))
        self.assertEqual(self._ensure(poll=0.05, wait_timeout=50.0), "built")

    def test_a_failed_build_leaves_no_lock_and_no_partial_table(self):
        import lisatools.domains as domains

        with mock.patch.object(domains, "WDMLookupTable") as T:
            T.apply_eps_frequency.return_value = (None, None, 21)
            T.apply_eps_fdot.return_value = None
            T.side_effect = RuntimeError("cuda OOM stand-in")
            with self.assertRaisesRegex(RuntimeError, "OOM"):
                self._ensure()
        self.assertEqual(os.listdir(os.path.dirname(self.path)), [])


if __name__ == "__main__":
    unittest.main()
