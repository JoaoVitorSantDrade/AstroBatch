import unittest
import numpy as np
from astroflow_logic import make_homogeneous

class FlowCorpusTests(unittest.TestCase):
    def test_inverse_ground_truth_maps_current_to_anchor(self):
        forward=np.array([[1.,0.,4.],[0.,1.,-3.]])
        inv=np.linalg.inv(make_homogeneous(forward))
        probes=np.array([[0.,0.,1.],[10.,4.,1.],[80.,90.,1.]])
        np.testing.assert_allclose((inv@((make_homogeneous(forward)@probes.T))).T,probes)

    def test_horizontal_nine_pixel_trail_kernel_is_deterministic(self):
        image=np.zeros((32,32),np.float32); image[16,16]=1
        trail=np.ones((1,9),np.float32)/9
        out=__import__('cv2').filter2D(image,-1,trail)
        self.assertEqual(int(np.count_nonzero(out)),9)
        self.assertAlmostEqual(float(out[16,16]),1/9,places=6)

if __name__=='__main__': unittest.main()
