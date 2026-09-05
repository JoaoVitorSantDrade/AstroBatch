import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

import hdr_logic


class HDRPipelineTests(unittest.TestCase):
    def make(self, root, name, data, **cards):
        p = root / name
        fits.PrimaryHDU(np.asarray(data, np.float32), fits.Header(cards)).writeto(p)
        return p

    def execute(self, paths, out, **cfg):
        return hdr_logic.run_hdr_pipeline({"input_paths": paths, "output_path": out, **cfg}, lambda _: None, lambda *_: None, threading.Event())

    def test_same_exposure_success_uint16(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d); ps = [self.make(r, f"{i}.fits", np.ones((4, 4))*10, EXPTIME=2.) for i in range(2)]
            out = r / "hdr.fits"; result = self.execute(ps, out)
            self.assertEqual(result["status"], "success"); self.assertEqual(fits.getdata(out).dtype, np.uint16)
            self.assertEqual(len(result["exposure_provenance"]), 2)

    def test_calnorm_restored_expected_radiance(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d); h = {"EXPTIME": 1., "CALNORM": True, "CALMIN": 0., "CALMAX": 100.}
            ps = [self.make(r, f"{i}.fits", np.full((2, 2), 32768), **h) for i in range(2)]
            result = self.execute(ps, r / "o.fits"); self.assertEqual(result["status"], "success")
            self.assertAlmostEqual(result["radiance_min"], 50.00076, places=2)

    def test_input_valid_mask_black_border_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d); ps=[]
            for i in range(2):
                p=self.make(r, f"{i}.fits", np.ones((3,3))*10, EXPTIME=1.)
                with fits.open(p, mode="update") as hdul: hdul.append(fits.ImageHDU(np.pad(np.ones((1,1),np.uint8),((1,1),(1,1))), name="VALID_MASK"))
                ps.append(p)
            out=r/"o.fits"; self.assertEqual(self.execute(ps,out)["valid_pixels"],1)

    def test_output_collision_and_overwrite_false(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); a=self.make(r,"a.fits",np.ones((2,2)),EXPTIME=1.); b=self.make(r,"b.fits",np.ones((2,2)),EXPTIME=1.)
            out=r/"o.fits"; out.write_bytes(b"x"); self.assertEqual(self.execute([a,b],out,overwrite=False)["status"],"error")
            self.assertEqual(self.execute([a,b],a)["status"],"error")

    def test_incompatible_gain_and_filter(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); a=self.make(r,"a.fits",np.ones((2,2)),EXPTIME=1.,GAIN=1,FILTER="R"); b=self.make(r,"b.fits",np.ones((2,2)),EXPTIME=1.,GAIN=2,FILTER="R")
            self.assertEqual(self.execute([a,b],r/"o.fits")["status"],"error")

    def test_pre_and_mid_cancel(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); ps=[self.make(r,f"{i}.fits",np.ones((100,100)),EXPTIME=1.) for i in range(2)]; out=r/"o.fits"; e=threading.Event(); e.set()
            self.assertEqual(hdr_logic.run_hdr_pipeline({"input_paths":ps,"output_path":out},lambda _:None,lambda *_:None,e)["status"],"cancelled"); self.assertFalse(out.exists())

    def test_all_saturated_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); ps=[self.make(r,f"{i}.fits",np.ones((2,2))*100,EXPTIME=1.,SATURATE=10.) for i in range(2)]
            self.assertEqual(self.execute(ps,r/"o.fits")["status"],"error")

    def test_rgb_channels_three_and_2d_mask(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d); ps=[self.make(r,f"{i}.fits",np.ones((3,2,2))*10,EXPTIME=1.) for i in range(2)]
            out=r/"o.fits"; self.assertEqual(self.execute(ps,out)["status"],"success"); self.assertEqual(fits.getdata(out).shape,(3,2,2))


if __name__ == "__main__": unittest.main()
