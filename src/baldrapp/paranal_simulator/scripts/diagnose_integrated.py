#!/usr/bin/env python3
"""Stage 5 headless v2 integrated-physics diagnostics (no SHM or ZMQ).

The test support module imports numerical function bodies from the production
runtime *without executing* its live loop. A small synthetic zero-DM
instrument and a three-mode piston/tip/tilt AO basis isolate the integrated
atmosphere, source, ZWFS/Fresnel and detector physics.

From the repository root:

    python src/baldrapp/paranal_simulator/scripts/diagnose_integrated.py \
        --output-dir stage5_validation

This does not validate the full initialized HCIPy DM, hardware SHM formats,
CRED1 frame assembly, camera/control servers or GUI operation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tests'))
from integrated_support import run_scenarios  # noqa: E402


def write_plots(report, snapshots, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(report['scenarios'])
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, name in zip(axs.flat, names):
        image = snapshots[name]['first']
        display = np.log10(np.maximum(image, 0) + 1)
        im = ax.imshow(display, origin='lower')
        ax.set_title(name.replace('_', ' '))
        ax.set_xlabel('Detector pixel')
        ax.set_ylabel('Detector pixel')
        fig.colorbar(im, ax=ax, label=r'$\log_{10}(1 + e^- / pixel)$')
    fig.savefig(directory / 'stage5_detector_frames.png', dpi=140)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for name in names:
        scen = report['scenarios'][name]
        frames = np.arange(len(scen['detected_electrons']))
        axs[0].plot(frames, scen['optical_throughput'], '-o', label=name)
        axs[1].plot(frames, scen['residual_opd_rms_nm'], '-o', label=name)
        axs[2].plot(frames, np.asarray(scen['entrance_photons_per_s']) /
                    scen['source_photons_per_s'], '-o', label=name)
    axs[0].set_title('Optical throughput')
    axs[0].set_ylabel('Rate after optics / incident pupil rate')
    axs[1].set_title('Residual OPD RMS')
    axs[1].set_ylabel('nm')
    axs[2].set_title('Pupil flux variation')
    axs[2].set_ylabel('Incident pupil rate / nominal source rate')
    for ax in axs:
        ax.set_xlabel('Frame number')
        ax.grid(alpha=.3)
    axs[0].legend(fontsize=8, loc='best')
    fig.savefig(directory / 'stage5_temporal_metrics.png', dpi=140)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=Path('stage5_validation'))
    ap.add_argument('--n-frames', type=int, default=4)
    ap.add_argument('--n-wavelengths', type=int, default=3)
    ap.add_argument('--non-fresnel', action='store_true')
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args(argv)
    if args.n_frames < 1 or args.n_wavelengths < 1:
        ap.error('frame and wavelength counts must be positive')
    report, images = run_scenarios(n_frames=args.n_frames,
        fresnel_enabled=not args.non_fresnel,
        n_wavelengths=args.n_wavelengths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'stage5_metrics.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if not args.no_plots:
        write_plots(report, images, args.output_dir)
    print(f"Stage 5 diagnostics saved to {args.output_dir.resolve()}")
    print('Physics-only headless validation: NOT a live SHM/hardware simulation')
    print('Scenario                 throughput mean  OPD RMS mean [nm]   relative pupil rate')
    for name, s in report['scenarios'].items():
        print(f"{name:24s} {np.mean(s['optical_throughput']):.5f}"
              f"           {np.mean(s['residual_opd_rms_nm']):9.3f}"
              f"         {np.mean(s['entrance_photons_per_s'])/s['source_photons_per_s']:.6f}")


if __name__ == '__main__':
    main()
