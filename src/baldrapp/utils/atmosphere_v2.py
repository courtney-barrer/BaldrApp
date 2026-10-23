"""Multilayer phase-only atmosphere using BaldrApp's original infinite screens.

Every nonzero-Cn2 layer uses the existing ``phasescreens.PhaseScreenKolmogorov``
(Fried-style infinite phase screen), the same class used by the original
Paranal simulator. Wind speed determines how frequently its original
``add_row()`` method is called. A rotation of sampling coordinates aligns
that one extension direction with each layer's configured wind vector.

Outputs are optical-path difference in metres, using the 500 nm reference
wavelength at which the layer-specific r0 values are defined. No periodized
FFT tile, alternate screen generator, or new user-facing setting is used.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates

from baldrapp.utils import phasescreens

LAMBDA_REF_M = 500e-9


@dataclass
class FrozenFlowLayer:
    """One original infinite screen, moved along its wind direction.

    ``wind_x`` and ``wind_y`` are pattern velocities in pupil-array axes.
    Phase-screen extension happens in its native +row direction. Sampling
    coordinates are rotated so +row corresponds to the layer wind vector.

    ``sample_opd(t)`` is non-consuming, useful for diagnostics; ``commit=True``
    advances the screen for a frame. A committed infinite screen cannot be
    rewound, so non-consuming queries must not precede the committed history.
    """

    altitude_m: float
    cn2_fraction: float
    r0_500nm_m: float
    wind_x_m_per_s: float
    wind_y_m_per_s: float
    pixel_scale_m: float
    size: int
    pupil_diameter_pixels: int
    phase_screen: object | None
    _rows_added: int = field(default=0, init=False, repr=False)

    def sample_opd(self, time_s: float, *, commit: bool = False) -> np.ndarray:
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError('time_s must be finite and nonnegative')
        if self.phase_screen is None:  # exactly zero turbulence strength
            return np.zeros((self.size, self.size), dtype=float)

        wind_speed = float(np.hypot(self.wind_x_m_per_s, self.wind_y_m_per_s))
        displacement_px = wind_speed * time_s / self.pixel_scale_m
        # Roundoff must not turn an integer-row displacement into one fewer row.
        total_rows = int(np.floor(displacement_px + 1e-10))
        fractional_row = max(0.0, displacement_px - total_rows)
        if fractional_row >= 1.0 - 1e-10:
            total_rows += 1
            fractional_row = 0.0
        if total_rows < self._rows_added:
            raise ValueError('cannot sample an infinite screen before committed time')

        screen = self.phase_screen
        rows_needed = total_rows - self._rows_added
        if rows_needed and not commit:
            # A diagnostic lookahead must not alter the turbulence used by the
            # subsequent camera frames or their random-number sequence.
            saved_screen = screen._scrn.copy()
            saved_rng = deepcopy(screen._R.bit_generator.state)
        try:
            for _ in range(rows_needed):
                screen.add_row()

            # Rotate a square native row-growing screen into pupil coordinates.
            # Restrict the field to the entrance pupil: a rotated full square
            # would reach outside the screen's transverse width at its corners.
            coord = np.arange(self.size, dtype=float) - (self.size - 1) / 2
            y, x = np.meshgrid(coord, coord, indexing='ij')
            radius = self.pupil_diameter_pixels / 2
            support = x*x + y*y <= radius*radius
            if not np.any(support):
                return np.zeros((self.size, self.size), dtype=float)

            if wind_speed > 0:
                cos_a = self.wind_x_m_per_s / wind_speed
                sin_a = self.wind_y_m_per_s / wind_speed
            else:
                # Static wind: match the unrotated, original phase screen.
                cos_a, sin_a = 0.0, 1.0
            native_x = sin_a*x - cos_a*y
            native_y = cos_a*x + sin_a*y - fractional_row
            center = (self.size - 1) / 2
            src_x = center + native_x[support]
            src_y = center + native_y[support]
            # A telescope pupil with a normal zero-padded FFT grid fits within
            # the native screen at all wind angles. Fail clearly if it does not.
            if (np.any(src_x < 0) or np.any(src_x > self.size-1) or
                    np.any(src_y < 0) or np.any(src_y > self.size-1)):
                raise ValueError('grid.dim leaves insufficient margin around the pupil for wind rotation')
            phase = np.zeros((self.size, self.size), dtype=float)
            phase[support] = map_coordinates(
                screen.scrn, [src_y, src_x], order=1,
                mode='nearest', prefilter=False,
            )
            return phase * (LAMBDA_REF_M / (2*np.pi))
        finally:
            if rows_needed:
                if commit:
                    self._rows_added = total_rows
                else:
                    screen._scrn = saved_screen
                    screen._R.bit_generator.state = saved_rng


class MultilayerAtmosphere:
    """Sum seeded, independent original infinite-screen layers in OPD units."""

    def __init__(self, layers: list[FrozenFlowLayer], time_step_s: float):
        if not layers:
            raise ValueError('at least one layer is required')
        if not np.isfinite(time_step_s) or time_step_s <= 0:
            raise ValueError('time_step_s must be positive and finite')
        self.layers = layers
        self.time_step_s = float(time_step_s)
        self.time_s = 0.0
        self.output_shape = (layers[0].size, layers[0].size)
        if any(layer.size != layers[0].size for layer in layers):
            raise ValueError('all layers must use the same pupil sampling')

    @classmethod
    def from_resolved_config(cls, resolved: dict, *, grid_dim: int,
                             pupil_pixel_scale_m: float) -> 'MultilayerAtmosphere':
        if grid_dim < 2 or not np.isfinite(pupil_pixel_scale_m) or pupil_pixel_scale_m <= 0:
            raise ValueError('invalid pupil grid or physical sampling')
        cfg = resolved['configuration']['atmosphere']
        derived = resolved['derived']['atmosphere']
        if not cfg['enabled']:
            raise ValueError('atmosphere is disabled')
        if cfg['scintillation']['enabled']:
            raise NotImplementedError('v2 multilayer scintillation requires Stage 4 Fresnel propagation')

        pupil_diameter = int(resolved['configuration']['grid']['N'])
        if pupil_diameter > grid_dim - 3:
            raise ValueError('grid.dim needs at least three pixels of padding around grid.N for rotated wind sampling')
        seeds = np.random.SeedSequence(derived['random_seed']).spawn(len(derived['layers']))
        layers = []
        for entry, rng in zip(derived['layers'], seeds):
            frac = float(entry['cn2_fraction'])
            r0 = float(entry['r0_500nm_m']) if frac > 0 else np.inf
            if frac > 0:
                seed = int(rng.generate_state(1, dtype=np.uint32)[0])
                # This is *exactly* the phase-screen class used in the original
                # Paranal simulator: neither ft_phase_screen nor a second method.
                screen = phasescreens.PhaseScreenKolmogorov(
                    nx_size=int(grid_dim),
                    pixel_scale=float(pupil_pixel_scale_m),
                    r0=r0, L0=float(cfg['L0_m']), random_seed=seed,
                )
            else:
                screen = None
            layers.append(FrozenFlowLayer(
                altitude_m=float(entry['altitude_m']), cn2_fraction=frac,
                r0_500nm_m=r0,
                wind_x_m_per_s=float(entry['wind_x_m_per_s']),
                wind_y_m_per_s=float(entry['wind_y_m_per_s']),
                pixel_scale_m=float(pupil_pixel_scale_m), size=int(grid_dim),
                pupil_diameter_pixels=pupil_diameter, phase_screen=screen,
            ))
        return cls(layers, resolved['derived']['simulator_runtime']['time_step_s'])

    def opd_at(self, time_s: float) -> np.ndarray:
        """Inspect a future OPD without consuming any infinite-screen rows."""
        summed = np.zeros(self.output_shape, dtype=float)
        for layer in self.layers:
            summed += layer.sample_opd(time_s)
        return summed

    def next_opd(self) -> np.ndarray:
        """Advance each infinite screen as needed at the next simulated frame."""
        summed = np.zeros(self.output_shape, dtype=float)
        for layer in self.layers:
            summed += layer.sample_opd(self.time_s, commit=True)
        self.time_s += self.time_step_s
        return summed


def modal_ao_residual_opd(opd_m: np.ndarray, basis: np.ndarray,
                          pupil: np.ndarray, n_modes: int, *,
                          correction_gain: float = 1.0,
                          remove_piston: bool = True,
                          lag_frames: int = 0,
                          history: deque | None = None) -> np.ndarray:
    """Apply sampled least-squares modal correction with optional integer lag.

    ``basis`` is the existing HCIPy transformation matrix (n_pixels,n_terms)
    built by make_first_stage_ao_basis. The first mode is piston. The first
    stage corrects phase/OPD only, not amplitude; no Baldr DM term is included.
    ``history`` must be reused across frames when lag_frames>0.
    """
    opd = np.asarray(opd_m, dtype=float)
    support = np.asarray(pupil) > 0
    if opd.shape != support.shape or opd.ndim != 2:
        raise ValueError('OPD and pupil must be matching 2-D arrays')
    if not np.isfinite(correction_gain) or not 0 <= correction_gain <= 1:
        raise ValueError('correction_gain must be within [0, 1]')
    if lag_frames < 0 or not isinstance(lag_frames, int):
        raise ValueError('lag_frames must be a nonnegative integer')
    modes = np.asarray(basis, dtype=float)
    if modes.ndim == 2 and modes.shape[0] == opd.size:
        modes = modes.T.reshape((-1, *opd.shape))
    elif modes.ndim != 3 or modes.shape[1:] != opd.shape:
        raise ValueError('basis must have shape (n_pixels,n_terms) or (n_terms,ny,nx)')
    if n_modes > len(modes):
        raise ValueError('n_modes exceeds basis length')
    chosen = modes[0:n_modes]
    if not remove_piston:
        chosen = chosen[1:]
    if chosen.size == 0 or correction_gain == 0:
        return np.where(support, opd, 0.0)
    coeff, *_ = np.linalg.lstsq(chosen[:, support].T, opd[support], rcond=None)
    reconstruction = np.einsum('i,ijk->jk', coeff, chosen)
    if lag_frames:
        if history is None:
            raise ValueError('history deque required for nonzero lag_frames')
        history.append(reconstruction)
        if len(history) <= lag_frames:
            reconstruction = np.zeros_like(opd)
        else:
            reconstruction = history.popleft()
    return np.where(support, opd - correction_gain * reconstruction, 0.0)
