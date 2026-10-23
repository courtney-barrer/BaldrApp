"""Small reproducible full-Fresnel benchmark; run from repository root.

    BALDR_FFT_WORKERS=1 python tests/benchmark_fresnel.py
    BALDR_FFT_WORKERS=4 python tests/benchmark_fresnel.py

The environment variable must be set before importing BaldrApp modules.
This uses a synthetic instrument, not the live SHM loop.
"""
import argparse
import json
from time import perf_counter
import numpy as np
from test_real_optics import EXAMPLE, load_real_optics, make_instrument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=int, default=256)
    parser.add_argument('--wavelengths', type=int, default=7)
    parser.add_argument('--repeats', type=int, default=7)
    args = parser.parse_args()
    cfg = json.loads(EXAMPLE.read_text())
    core = load_real_optics()
    inst, amp = make_instrument(cfg, args.wavelengths, size=args.grid,
                                pupil_diameter=int(args.grid * .625),
                                fresnel_enabled=True)
    yy, xx = np.indices(amp.shape)
    opd = 8e-8 * np.sin(2*np.pi*xx/33)*np.cos(2*np.pi*yy/39)
    kwargs = dict(opd_input=opd, amp_input=amp, opd_internal=np.zeros_like(amp),
                  zwfs_ns=inst, detector=None, include_shotnoise=False)
    for _ in range(2):
        core['get_frame_fresnel_polychromatic'](**kwargs)
    times = []
    for _ in range(args.repeats):
        t0 = perf_counter()
        frame = core['get_frame_fresnel_polychromatic'](**kwargs)
        times.append((perf_counter()-t0)*1000)
    print(f'grid={args.grid} wavelengths={args.wavelengths} '
          f'median_ms={np.median(times):.2f} photons_per_s={frame.sum():.12g}')


if __name__ == '__main__':
    main()
