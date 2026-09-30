"""WDM chunk primitives for the EMRI plunge chunk (Task A7).

* A power-of-two chunk starting at an EVEN global pixel reproduces the full
  TD->WDM transform in its interior; an ODD start does not (paired control).
* The edge contamination is MEASURED: it decays algebraically from each chunk edge
  (Nt_sub=128: 0.2 at the edge, 5e-5 at 16 px, 6e-6 at 24 px, 2e-7 at the centre; a
  Tukey does not help), so the interior is ``[n_pad, Nt_sub - n_pad)`` with
  ``n_pad = Nt_sub // 4`` and a 1e-5-of-peak tolerance, not "exact".
* The plunge chunk splices onto the tail exactly; a chunk that would run into the
  production time crop raises instead of producing a partially tapered chunk.
"""

import unittest

import numpy as np

NF, NT, DT = 32, 256, 112.5


class ChunkSpliceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lisatools.domains import TDSettings, TDSignal, WDMSettings
        from lisatools.wdm_het import chunk_start_for_pixels, splice_chunk, wdm_chunk_of_td

        cls.start = staticmethod(chunk_start_for_pixels)
        cls.splice = staticmethod(splice_chunk)
        cls.chunk_of = staticmethod(wdm_chunk_of_td)
        N = NF * NT
        t = np.arange(N) * DT
        f0, fdot = 12.3 / (2 * NF * DT), 2e-10
        cls.td = np.cos(2 * np.pi * (f0 * t + 0.5 * fdot * t ** 2))[None, :]
        wdm = WDMSettings(Nf=NF, Nt=NT, dt=DT, force_backend="cpu")
        cls.full = np.asarray(TDSignal(cls.td, TDSettings(N, DT, force_backend="cpu")).transform(wdm).arr)

    def _chunk(self, n0, Nt_sub=128):
        return np.asarray(self.chunk_of(self.td, n0 * NF, NF, Nt_sub, DT))

    def _interior_err(self, n0, Nt_sub=128, n_pad=32):
        out = np.zeros_like(self.full)
        self.splice(out, self._chunk(n0, Nt_sub), n0, n_pad, Nt_sub - n_pad)
        sl = slice(n0 + n_pad, n0 + Nt_sub - n_pad)
        return np.max(np.abs(out[:, :, sl] - self.full[:, :, sl])) / np.max(np.abs(self.full))

    def test_even_start_matches_interior(self):
        self.assertLess(self._interior_err(64), 1e-5)

    def test_odd_start_fails(self):
        self.assertGreater(self._interior_err(65), 1e-2)

    def test_edge_contamination_length_is_covered(self):
        n0, Nt_sub, tol = 64, 128, 1e-5
        err = np.abs(self._chunk(n0, Nt_sub) - self.full[:, :, n0:n0 + Nt_sub]).max(axis=(0, 1))
        err = err / np.abs(self.full).max()
        self.assertTrue(np.any(err < tol), "no pixel of the chunk reaches the tolerance")
        k_lo = int(np.argmax(err < tol))                           # first clean pixel from the left
        k_hi = int(np.argmax(err[::-1] < tol))                     # ... from the right
        print(f"\n[edge contamination] {k_lo} px (left), {k_hi} px (right) above {tol:g}; edge err {err[0]:.2e}")
        self.assertGreater(err[0], 1e-3)                           # the raw edge IS contaminated
        self.assertGreaterEqual(Nt_sub // 4, max(k_lo, k_hi))      # n_pad = Nt_sub//4 covers it
        self.assertTrue(np.all(err[Nt_sub // 4: Nt_sub - Nt_sub // 4] < tol))

    def test_chunk_start_is_even_and_covers(self):
        n0 = self.start(150, 200, NT, 128, 32)
        self.assertEqual(n0 % 2, 0)
        self.assertLessEqual(n0 + 32, 150)
        self.assertGreaterEqual(n0 + 128 - 32, 200)
        self.assertLessEqual(n0 + 128, NT)

    def test_chunk_start_rejects_span_wider_than_interior(self):
        with self.assertRaises(ValueError):
            self.start(100, 170, NT, 128, 32)

    def test_plunge_chunk_splices_tail_exactly(self):
        from lisatools.sources.emri.wdm_direct import plunge_chunk_wdm

        # the tail [n_h, n_end) is interior to the chunk (the plunge ends before the
        # grid does; the grid's own end is cropped in production)
        n_h, n_end = 150, 200
        chunk, n0, klo, khi = plunge_chunk_wdm(lambda s, n: self.td[:, s:s + n], n_h, NT, NF, DT,
                                               Nt_sub=128, n_end=n_end)
        out = np.zeros_like(self.full)
        self.splice(out, np.asarray(chunk), n0, klo, khi)
        sl = slice(n_h, n_end)
        err = np.max(np.abs(out[:, :, sl] - self.full[:, :, sl])) / np.max(np.abs(self.full))
        self.assertLess(err, 1e-5)
        self.assertTrue(np.all(out[:, :, :n_h] == 0.0) and np.all(out[:, :, n_end:] == 0.0))

    def test_plunge_chunk_into_cropped_tail_raises(self):
        from lisatools.sources.emri.wdm_direct import plunge_chunk_wdm

        with self.assertRaises(ValueError):
            plunge_chunk_wdm(lambda s, n: self.td[:, s:s + n], 150, NT, NF, DT, Nt_sub=128, n_end=200, ind_max_t=220)


if __name__ == "__main__":
    unittest.main()
