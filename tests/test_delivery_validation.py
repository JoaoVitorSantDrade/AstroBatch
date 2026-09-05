"""Behavioral gates for resource scheduling, RGB correction and linear fusion."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from astropy.io import fits
import astroalign_logic as align
import astroflow_logic as flow
import hdr_logic as hdr
from app.engines.execution import ExecutionBudget, science_frame_bytes


class DeliveryValidation(unittest.TestCase):
    def test_compressed_header_budget(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'compressed.fits'
            fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(np.ones((3,32,48), np.uint16))]).writeto(path)
            self.assertEqual(science_frame_bytes(path), 3*32*48*4)
        budget = ExecutionBudget.for_frame_pipeline(8, 256, 32*1024**2)
        self.assertLessEqual((2+budget.max_in_flight)*32, 256)
        self.assertEqual(budget.worker_count, 3)

    def test_rgb_both_profiles_correct_sign_and_skip_large_shift(self):
        rng = np.random.default_rng(33)
        g = cv2.GaussianBlur(rng.normal(size=(128,128)).astype(np.float32), (0,0), 1)
        for profile in ('Stable','Fast'):
            for shift in (1,8):
                rgb = np.stack([np.roll(g,shift,axis=1),g,g],axis=-1)
                diag = {}
                warped = align.warp_frame(rgb,np.eye(3),'bilinear',True,profile,diagnostics=diag)
                if shift == 1:
                    self.assertLess(np.mean((warped[4:-4,4:-4,0]-g[4:-4,4:-4])**2), .001)
                    self.assertAlmostEqual(diag['rgb_shifts'][0][0],-1,delta=.15)
                else:
                    self.assertNotIn(0,diag.get('rgb_shifts',{}))
                    np.testing.assert_allclose(warped[:,:,0],rgb[:,:,0],atol=1e-6)

    def test_align_cancellation_during_write_preserves_destination(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td)/'out.fits'; output.write_bytes(b'existing')
            event = threading.Event()
            original = fits.HDUList.writeto
            def cancel_after_write(hdul,*args,**kwargs):
                original(hdul,*args,**kwargs); event.set()
            with patch.object(fits.HDUList,'writeto',cancel_after_write):
                with self.assertRaises(InterruptedError):
                    align.save_aligned_fits(np.ones((4,4)),np.ones((4,4)),fits.Header(),output,False,cancel_event=event)
            self.assertEqual(output.read_bytes(),b'existing')
            self.assertEqual(list(Path(td).glob('*.tmp')),[])

    def make_hdr(self, root, values, **cards):
        paths=[]
        for i, value in enumerate(values):
            path=root/f'{i}.fits'
            fits.PrimaryHDU(np.asarray(value,np.uint16),fits.Header({'EXPTIME':2.,**cards})).writeto(path)
            paths.append(path)
        return paths

    def test_hdr_raw_saturation_restoration_masks_and_unbiased_mean(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            paths=self.make_hdr(root, [[[65535,32768]],[[65535,32768]]], CALNORM=True,CALMIN=0.,CALMAX=100.,SATURATE=65500)
            out=root/'out.fits'
            result=hdr.run_hdr_pipeline({'input_paths':paths,'output_path':out},print,lambda *a:None,None)
            self.assertEqual(result['status'],'success',result)
            self.assertEqual(result['valid_pixels'],1)
            self.assertAlmostEqual(result['radiance_min'],25.00038,places=3)
            with fits.open(out,memmap=False) as hdul:
                np.testing.assert_array_equal(hdul['VALID_MASK'].data,[[0,1]])
                self.assertEqual(hdul[0].header['EXPTIME'],1.)
                self.assertEqual(hdul[0].header['HDRNFRM'],2)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); paths=self.make_hdr(root, [[[10]],[[30]]])
            result=hdr.run_hdr_pipeline({'input_paths':paths,'output_path':root/'mean.fits'},lambda *a:None,lambda *a:None,None)
            self.assertAlmostEqual(result['radiance_min'],10.)

    def test_hdr_last_band_cancel_and_invalid_config(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); paths=self.make_hdr(root,[np.ones((3,3)),np.ones((3,3))])
            out=root/'out.fits'; event=threading.Event()
            result=hdr.run_hdr_pipeline({'input_paths':paths,'output_path':out},lambda *a:None,lambda *a:event.set(),event)
            self.assertEqual(result['status'],'cancelled'); self.assertFalse(out.exists())
            for key in ('noise_floor','exposure_override','saturation'):
                result=hdr.run_hdr_pipeline({'input_paths':paths,'output_path':out,key:float('nan')},lambda *a:None,lambda *a:None,None)
                self.assertEqual(result['status'],'error')

    def test_flow_relaxed_phase_reuse_and_anchor_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name in ('01','02','03'):
                fits.PrimaryHDU(np.zeros((32,32),np.uint16)).writeto(root/f'{name}.fits')
            stars=np.array([[4,4],[24,4],[4,24],[24,24],[16,16]],np.float32)
            def prepare(p,*a):
                return p.name,dict(path=p,phase_data=np.ones((32,32),np.float32),stars=stars,fwhm=2.,metrics={},status='prepared')
            metrics=dict(matches=5,inliers=5,inlier_ratio=1.,rms=.1)
            identity=np.eye(3)[:2]
            # Frame02 normal succeeds; frame03 fails previous twice, anchor succeeds.
            outcomes=[(identity,metrics.copy()),(None,metrics.copy()),(None,metrics.copy()),(identity,metrics.copy())]
            with patch.object(flow,'_process_single_frame',side_effect=prepare), patch.object(flow,'_estimate_incremental_transform',side_effect=outcomes), patch.object(flow.cv2,'phaseCorrelate',return_value=((0,0),1.)) as phase:
                flow.process_local_flow(root,{'min_stars':4},lambda *a:None)
            report=json.loads((root/'flow_local.json').read_text())
            self.assertEqual(report['frames']['03.fits']['recovery_method'],'anchor')
            self.assertEqual(report['frames']['03.fits']['relative_to'],'01.fits')
            self.assertEqual(phase.call_count,3)


if __name__ == '__main__': unittest.main()
