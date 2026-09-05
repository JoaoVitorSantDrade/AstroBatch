"""Deterministic interpolation and HDR quality/throughput measurements."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import time
import numpy as np
from astropy.io import fits
from astroalign_logic import warp_frame
from hdr_logic import run_hdr_pipeline
from calibration_logic import _restore_normalized_values


def main():
    y, x = np.mgrid[:384,:384]
    field = (1000*np.exp(-((x-180)**2+(y-190)**2)/18)).astype(np.float32)
    dx, dy = .35, -.6
    truth = 1000*np.exp(-((x-180-dx)**2+(y-190-dy)**2)/18)
    matrix = np.array([[1,0,dx],[0,1,dy],[0,0,1.]])
    rows=[]
    for profile in ('Stable','Fast'):
        for mode in ('nearest','bilinear','bicubic','lanczos'):
            warp_frame(field,matrix,mode,engine_profile=profile)
            timings=[]
            for _ in range(5):
                start=time.perf_counter(); output=warp_frame(field,matrix,mode,engine_profile=profile)
                timings.append(time.perf_counter()-start)
            rows.append(dict(profile=profile,mode=mode,median_seconds=float(np.median(timings)),
                             rms=float(np.sqrt(np.mean((output-truth)**2))),
                             relative_flux_error=float(output.sum()/truth.sum()-1)))
    rng=np.random.default_rng(42); signal=np.full((256,256),1000.,np.float32)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); paths=[]; noises=[]
        for i in range(8):
            data=signal+rng.normal(0,10,signal.shape).astype(np.float32)
            path=root/f'{i}.fits'; fits.PrimaryHDU(data,fits.Header({'EXPTIME':2.})).writeto(path)
            paths.append(path); noises.append(float(np.std(data-signal)))
        start=time.perf_counter()
        result=run_hdr_pipeline(dict(input_paths=paths,output_path=root/'fused.fits',noise_floor=10,row_band=64),lambda *a:None,lambda *a:None,None)
        seconds=time.perf_counter()-start
        if result['status']!='success': raise RuntimeError(result)
        with fits.open(root/'fused.fits',memmap=False) as hdul:
            recovered=_restore_normalized_values(hdul[0].data,hdul[0].header)*2
        fusion=dict(seconds=seconds,frames=8,noise_improvement=float(np.mean(noises)/np.std(recovered-signal)),
                    expected_independent_noise_improvement=float(np.sqrt(8)),quantization_step=result['quantization_step'])
    report=dict(seed=42,interpolation=rows,hdr=fusion)
    print(json.dumps(report,indent=2))
    if len(sys.argv)>1: Path(sys.argv[1]).write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__': main()
