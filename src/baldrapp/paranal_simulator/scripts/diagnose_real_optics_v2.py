#!/usr/bin/env python3
"""Reproducible numerical Stage 2.5 diagnostics (no SHM/camera required).

Uses the real optical function bodies through the test_real_optics_v2 support
harness. Writes a JSON numerical report and two PNG figures; does not change
any source configuration or instrument state on disk.

From repository root:
  python src/baldrapp/paranal_simulator/scripts/diagnose_real_optics_v2.py \
      --output-dir /tmp/baldr_stage25
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'tests'))
from test_real_optics_v2 import (  # noqa: E402
    RealOpticsV2Tests, make_instrument,
)


def norm_error(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def numeric_report():
    case = RealOpticsV2Tests('test_real_detector_scaling_and_single_application')
    case.setUpClass()
    cfg = case.cfg
    rate = cfg['source_profiles']['internal']['flux']['photons_per_second']
    yy, xx = np.indices((64, 64))
    opd = 1.1e-6 * np.sin(2*np.pi*xx/21) * np.cos(2*np.pi*yy/23)
    curves, images = {}, {}

    for name, fres in (('non_fresnel', False), ('fresnel', True)):
        inst, amp = make_instrument(cfg, 51, fresnel_enabled=fres)
        reference = case.simulate(inst, amp, opd=opd)
        errors, throughput = {}, {}
        for count in (1, 3, 7, 15, 31, 51):
            ns, a = make_instrument(cfg, count, fresnel_enabled=fres)
            result = case.simulate(ns, a, opd=opd)
            errors[str(count)] = norm_error(result, reference)
            throughput[str(count)] = float(result.sum()/rate)
            if count == 7:
                images[name] = result
        curves[name] = {'relative_l2_error_vs_51': errors,
                        'optical_throughput': throughput}

    ns, a = make_instrument(cfg, 1, fresnel_enabled=True)
    out = case.core['get_frame_fresnel'](
        opd_input=np.zeros_like(a), amp_input=a, opd_internal=np.zeros_like(a),
        zwfs_ns=ns, detector=None, include_shotnoise=False,
        wavelength=ns.optics.wvl0, return_intermediates=True)
    sq = lambda x: float(np.vdot(x, x).real)
    s0 = out['dx_phys']**2
    sc = out['dx_cold']*out['dy_cold']
    sd = out['dx_detector']*out['dy_detector']
    # Relative to the illuminated entrance rate, optical losses (phase-mask
    # field cropping, mirror and coldstop) are included, not renormalized out.
    ledger = {
        'entrance': float(np.sum(a*a)),
        'after_zwfs_crop': sq(out['psi_C']),
        'mirror_before_aperture': sq(out['psi_mirror']),
        'mirror_after_aperture': sq(out['psi_mirror']*out['mirror_mask_D']),
        'lens_before': sq(out['field_at_lens']),
        'lens_after': sq(out['field_after_lens']),
        'coldstop_before': sq(out['field_coldstop_plane'])*sc/s0,
        'coldstop_after': sq(out['field_after_coldstop'])*sc/s0,
        'detector_pupil': sq(out['field_detector_pupil'])*sd/s0,
        'detector_image_rate': float(out['intensity_pre_detector'].sum()),
    }
    ledger['relative_to_entrance'] = {
        k: v/ledger['entrance'] for k,v in ledger.items() if k != 'entrance'
    }
    return {
        'description': 'Real BaldrApp Fourier/Fresnel propagation; synthetic disc pupil and zero DM',
        'sample_geometry': {'array_pixels': 64, 'pupil_pixels_diameter': 40,
                            'fplane_pixels': 129},
        'bandpass_m': [1.5e-6, 1.8e-6],
        'source_flux_photons_per_s': rate,
        'curve_data': curves,
        'monochromatic_fresnel_energy_ledger_photons_per_s': ledger,
    }, images


def plot_report(report, images, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, name in zip(axes, ('non_fresnel', 'fresnel')):
        im = ax.imshow(images[name], origin='lower')
        ax.set_title(name.replace('_', ' ') + ' · 7 wavelength samples')
        ax.set_xlabel('Pixel')
        ax.set_ylabel('Pixel')
        fig.colorbar(im, ax=ax, label='photons/s/pixel')
    fig.tight_layout()
    fig.savefig(directory / 'stage25_real_optics_frames.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, section in report['curve_data'].items():
        counts = np.asarray([1, 3, 7, 15, 31], dtype=int)
        e = section['relative_l2_error_vs_51']
        t = section['optical_throughput']
        axes[0].semilogy(counts, [e[str(n)] for n in counts], '-o', label=name)
        axes[1].plot(counts, [t[str(n)] for n in counts], '-o', label=name)
    axes[0].set_title('Spectral convergence · reference = 51')
    axes[0].set_ylabel('Relative image L2 error')
    axes[1].set_title('Optical throughput (includes physical losses)')
    axes[1].set_ylabel('Detected-plane photons/s / entrance photons/s')
    for ax in axes:
        ax.set_xlabel('Number of wavelength samples')
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(directory / 'stage25_spectral_convergence.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('stage25_validation'))
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report, images = numeric_report()
    (args.output_dir / 'stage25_metrics.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if not args.no_plots:
        plot_report(report, images, args.output_dir)
    print(f'Report written to {args.output_dir.resolve()}')
    for name, d in report['curve_data'].items():
        print(f"{name}: throughput(7)={d['optical_throughput']['7']:.6f}; "
              f"relative error(7 vs 51)={d['relative_l2_error_vs_51']['7']:.6g}")


if __name__ == '__main__':
    main()
