import tempfile
import unittest
from pathlib import Path
import numpy as np
from astropy.io import fits
import astroalign_logic as align
from calibration_logic import calibrate_single_frame
from hdr_logic import run_hdr_pipeline


class ScienceMaskChainTests(unittest.TestCase):
    def test_calibration_align_hdr_preserve_sensor_clipping(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); calibrated=root/'cal'; calibrated.mkdir(); output=root/'aligned'
            for i in range(2):
                path=root/f'{i}.fits'; data=np.full((20,20),100,np.uint16); data[10,10]=65535
                fits.PrimaryHDU(data,fits.Header({'EXPTIME':2.,'SATURATE':65000})).writeto(path)
                self.assertIsNone(calibrate_single_frame(path,calibrated,None,None,True,(0.,65535.)))
                cfg=align._build_align_config(root,output,dict(debayer_pattern='Nenhum',keep_header=False,rgb_registration=False))
                result=align._process_single_alignment(path.name,{'matrix':np.eye(3).tolist()},calibrated,output,np.eye(3),'nearest',cfg)
                self.assertIsNone(result[1],result)
                with fits.open(output/path.name,memmap=False) as hdul:
                    self.assertEqual(hdul['SAT_MASK'].data[10,10],1)
                    self.assertTrue(hdul[1].header['SATKNOWN'])
            result=run_hdr_pipeline(dict(input_paths=list(output.glob('*.fits')),output_path=root/'hdr.fits'),lambda *a:None,lambda *a:None,None)
            self.assertEqual(result['status'],'success',result)
            self.assertEqual(result['valid_pixels'],399)
            with fits.open(root/'hdr.fits',memmap=False) as hdul:
                self.assertEqual(hdul['VALID_MASK'].data[10,10],0)

    def test_rgb_mask_holes_and_corrected_border(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); y,x=np.mgrid[:64,:64]
            g=(3000*np.exp(-((x-30)**2+(y-30)**2)/10)).astype(np.float32)
            rgb=np.stack([np.roll(g,1,axis=1),g,g])
            mask=np.ones((64,64),np.uint8); mask[6,6]=0
            sat=np.zeros((64,64),np.uint8); sat[30,30]=1
            fits.HDUList([fits.PrimaryHDU(rgb),fits.ImageHDU(mask,name='VALID_MASK'),fits.ImageHDU(sat,name='SAT_MASK')]).writeto(root/'rgb.fits')
            cfg=align._build_align_config(root,root/'out',dict(engine_profile='Fast',debayer_pattern='Nenhum',rgb_registration=True))
            result=align._process_single_alignment('rgb.fits',{'matrix':np.eye(3).tolist()},root,root/'out',np.eye(3),'nearest',cfg)
            self.assertIsNone(result[1],result)
            with fits.open(root/'out'/'rgb.fits',memmap=False) as hdul:
                self.assertEqual(hdul['VALID_MASK'].data.ndim,2)
                self.assertEqual(hdul['VALID_MASK'].data[6,6],0)
                self.assertTrue(np.all(hdul['VALID_MASK'].data[:,-1]==0))
                self.assertGreaterEqual(int(hdul['SAT_MASK'].data.sum()),2)


if __name__=='__main__': unittest.main()
