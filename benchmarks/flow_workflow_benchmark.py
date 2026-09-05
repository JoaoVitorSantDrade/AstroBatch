"""Known-transform FITS corpus exercising actual detection and matching."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import time
import cv2
import numpy as np
from astropy.io import fits
from astroflow_logic import process_local_flow


def render(seed=42):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:256, :256]
    image = np.zeros((256, 256), np.float32)
    for sx, sy, flux in zip(rng.uniform(20,236,50), rng.uniform(20,236,50), rng.uniform(100,1000,50)):
        image += (flux * np.exp(-((x-sx)**2+(y-sy)**2)/5)).astype(np.float32)
    return image


def main(save=None):
    rng = np.random.default_rng(42)
    signal = render(42)
    base = signal + rng.normal(0,1,signal.shape).astype(np.float32)
    translation = np.array([[1.,0.,3.],[0.,1.,-2.]])
    cases = [('translation',translation,1.), ('rotation',cv2.getRotationMatrix2D((128,128),2.,1.),1.),
             ('low_snr',translation,20.), ('trailing',translation,2.), ('unrelated',None,1.)]
    rows = []
    for label, matrix, noise in cases:
        image = render(84) if matrix is None else cv2.warpAffine(signal,matrix,(256,256))
        if label == 'trailing':
            image = cv2.filter2D(image,-1,np.ones((1,9),np.float32)/9)
        image = image + rng.normal(0,noise,image.shape).astype(np.float32)
        timings = []
        error = None
        info = {}
        for _ in range(3):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                fits.PrimaryHDU(base).writeto(root/'01.fits')
                fits.PrimaryHDU(image).writeto(root/'02.fits')
                start = time.perf_counter()
                process_local_flow(root,dict(min_stars=4,sigma=3,matching_radius=15),lambda *a:None)
                timings.append(time.perf_counter()-start)
                report = json.loads((root/'flow_local.json').read_text())
                info = report['frames']['02.fits']
                if matrix is not None and info['status'] == 'accepted':
                    expected = np.linalg.inv(np.vstack((matrix,[0,0,1])))
                    probes = np.array([[20,20,1],[236,20,1],[20,236,1],[236,236,1],[128,128,1]])
                    error = float(np.max(np.linalg.norm((np.asarray(info['matrix'])@probes.T-expected@probes.T)[:2],axis=0)))
        rows.append(dict(case=label,median_seconds=float(np.median(timings)),status=info['status'],
                         confidence=info.get('confidence'),max_probe_error_px=error,
                         false_accept=bool(matrix is None and info['status']=='accepted')))
    report = dict(seed=42,repeats=3,shape=[256,256],cases=rows)
    print(json.dumps(report,indent=2))
    if save:
        Path(save).write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv)>1 else None)
