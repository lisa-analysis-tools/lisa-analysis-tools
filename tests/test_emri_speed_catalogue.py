"""The EMRI speed scripts read a catalogue source in the FIT's basis (emri_catalogue_to_waveform_basis):
xI0 from the catalogue's InclinationAngle. CD1L EMRIs 0 and 3 are retrograde; the scripts once hard-coded
xI0 = +1, so a speed/accuracy test of those sources would have scored a prograde template."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "emri"))


class CatalogueParamsTest(unittest.TestCase):
    def test_retrograde_source_keeps_its_xI0(self):
        import h5py

        import emri_batch_speed as S
        from lisatools.sources.emri.waveform import emri_catalogue_to_waveform_basis

        fields = dict(PrimaryMassSSBFrame=[1e6, 2e6], SecondaryMassSSBFrame=[10.0, 20.0],
                      PrimarySpinParameter=[0.9, 0.5], SemiLatusRectum=[10.0, 12.0], Eccentricity=[0.2, 0.3],
                      InclinationAngle=[0.0, np.pi], LuminosityDistance=[1e3, 2e3], RightAscension=[1.0, 4.0],
                      Declination=[0.3, -0.5], PolarAnglePrimarySpin=[0.8, 1.2], AzimuthalAnglePrimarySpin=[0.5, 2.0],
                      AzimuthalPhase=[0.1, 0.2], PolarPhase=[0.0, 0.3], RadialPhase=[0.2, 0.4])
        with tempfile.TemporaryDirectory() as tmp:
            fp = os.path.join(tmp, "cat.h5")
            with h5py.File(fp, "w") as f:
                g = f.create_group("Binaries")
                for k, v in fields.items():
                    g.create_dataset(k, data=np.asarray(v))
            for src in (0, 1):
                got = S.catalogue_params(fp, src)
                want = emri_catalogue_to_waveform_basis({k: v[src] for k, v in fields.items()})
                np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
            self.assertEqual(S.catalogue_params(fp, 1)[5], -1.0)


if __name__ == "__main__":
    unittest.main()
