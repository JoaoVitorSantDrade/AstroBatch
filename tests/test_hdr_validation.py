import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from astropy.io import fits
from hdr_logic import HDRConfig, build_hdr_config, validate_hdr_config, run_hdr_pipeline


class HDRValidationTests(unittest.TestCase):
    def test_config_parity_rejects_fractional_and_boolean_values(self):
        for invalid in (0,-1,1.5,True):
            with self.assertRaises((ValueError,TypeError)):
                build_hdr_config(dict(input_paths=['a','b'],output_path='out',row_band=invalid))
            with self.assertRaises((ValueError,TypeError)):
                validate_hdr_config(HDRConfig((Path('a'),Path('b')),Path('out'),row_band=invalid))

    def fuse(self,root,headers,data=None):
        paths=[]
        for i,header in enumerate(headers):
            path=root/f'{i}.fits'
            fits.PrimaryHDU(np.full((8,8),10.,np.float32) if data is None else data[i],fits.Header(header)).writeto(path)
            paths.append(path)
        return run_hdr_pipeline(dict(input_paths=paths,output_path=root/'out.fits'),lambda *a:None,lambda *a:None,None)

    def test_rate_input_is_not_divided_again_and_provenance_is_embedded(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); result=self.fuse(root,[dict(EXPTIME=5.,BUNIT='ADU/s')]*2)
            self.assertEqual(result['status'],'success',result)
            self.assertAlmostEqual(result['radiance_min'],10.)
            with fits.open(root/'out.fits',memmap=False) as hdul:
                meta=json.loads(hdul['HDR_META'].data.tobytes().decode('utf-8'))
                self.assertEqual(meta['frames'],2)
                self.assertEqual(hdul[0].header['EXPTIME'],1.)
                self.assertEqual(meta['exposure_groups'],[{'seconds':5.,'count':2}])
            shutil.copy2(root/'out.fits',root/'repeat.fits')
            result=run_hdr_pipeline(dict(input_paths=[root/'out.fits',root/'repeat.fits'],output_path=root/'again.fits'),lambda *a:None,lambda *a:None,None)
            self.assertEqual(result['status'],'success',result); self.assertAlmostEqual(result['radiance_min'],10.,places=3)

    def test_exposure_alias_and_missing_metadata_conflicts(self):
        for headers,expected in (
            ([dict(EXPOSURE=2.)]*2,'success'),
            ([dict(EXPTIME=2.,EXPOSURE=5.)]*2,'error'),
            ([dict(EXPTIME=2.,GAIN=100),dict(EXPTIME=2.)],'error'),
        ):
            with tempfile.TemporaryDirectory() as td:
                result=self.fuse(Path(td),headers)
                self.assertEqual(result['status'],expected,result)

    def test_mixed_exposure_radiance_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            result=self.fuse(Path(td),[dict(EXPTIME=2.),dict(EXPTIME=5.)],
                             [np.full((8,8),20.,np.float32),np.full((8,8),50.,np.float32)])
            self.assertEqual(result['status'],'success',result)
            self.assertAlmostEqual(result['radiance_min'],10.)


if __name__=='__main__': unittest.main()
