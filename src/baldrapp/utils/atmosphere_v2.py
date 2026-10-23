"""Phase-only multilayer von Karman atmosphere for BaldrApp configuration v2.

Atmospheric layers are sampled in optical-path difference (OPD, metres),
independent of the wavelength used later by the instrument propagation.
Each independent, frozen screen is translated using the configured 2-D wind
velocity and simulated time. Altitude is stored for subsequent Fresnel
propagation (Stage 4); it has no geometrical effect for an on-axis, phase-only
source.

Numerical limitation
--------------------
The FFT screens are *finite periodic tiles*. We generate tiles four times the
wave-space array width to suppress short-period artifacts. They repeat after
tile_width / |wind_component| and do not represent indefinitely nonrepeating
turbulence. This is an explicit Stage-3 approximation, not an infinite-screen
model. Bilinear interpolation also slightly suppresses fine-scale variance
for noninteger pixel shifts. For long simulations or precise turbulence PSD
studies, an extended/nonperiodic frozen-flow generator will be needed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from baldrapp.utils import phasescreens

LAMBDA_REF_M = 500e-9
TILE_FACTOR = 4


@dataclass
class FrozenFlowLayer:
    """One periodized turbulence screen, sampled at the telescope footprint.

    Positive wind_x moves features toward increasing array columns and
    positive wind_y toward increasing rows (directions are degrees from +x
    toward +y in the numerical pupil frame, *not* meteorological FROM
    directions). The tile contains phase radians at 500 nm.
    """

    altitude_m: float
    cn2_fraction: float
    r0_500nm_m: float
    wind_x_m_per_s: float
    wind_y_m_per_s: float
    pixel_scale_m: float
    size: int
    phase_tile_rad: np.ndarray

    def sample_opd(self, time_s: float) -> np.ndarray:
        """Return a single pupil-grid OPD (metres) at the given physical time."""
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError('time_s must be finite and nonnegative')
        tile = self.phase_tile_rad
        n = tile.shape[0]
        if tile.ndim != 2 or tile.shape[1] != n or n < self.size:
            raise ValueError('phase tile must be square and at least size pixels')
        # The window is initially centred in the periodic tile.
        origin = (n - self.size) // 2
        y = origin + np.arange(self.size, dtype=float) - (
            self.wind_y_m_per_s * time_s / self.pixel_scale_m
        )
        x = origin + np.arange(self.size, dtype=float) - (
            self.wind_x_m_per_s * time_s / self.pixel_scale_m
        )
        # Map only the telescope footprint, rather than shifting a full tile.
        yy, xx = np.broadcast_arrays(y[:, None], x[None, :])
        phase = map_coordinates(tile, [yy, xx], order=1, mode='grid-wrap', prefilter=False)
        return np.asarray(phase * (LAMBDA_REF_M / (2 * np.pi)), dtype=float)


class MultilayerAtmosphere:
    """A deterministic collection of frozen-flow phase screens.

    The first call to ``next_opd()`` samples t=0 and increments the clock by
    ``time_step_s``. The atmosphere clock advances only when the on-sky
    frame generator invokes it, not according to wall-clock sleep time.
    """

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
        """Instantiate independent layers from configuration.py's report."""
        if grid_dim < 2 or not np.isfinite(pupil_pixel_scale_m) or pupil_pixel_scale_m <= 0:
            raise ValueError('invalid pupil grid or physical sampling')
        cfg = resolved['configuration']['atmosphere']
        derived = resolved['derived']['atmosphere']
        if not cfg['enabled']:
            raise ValueError('atmosphere is disabled')
        if cfg['scintillation']['enabled']:
            raise NotImplementedError('v2 multilayer scintillation requires Stage 4 Fresnel propagation')

        # The periodic tile's FFT grid must be even for the existing generator.
        tile_size = 2 * int(np.ceil((TILE_FACTOR * grid_dim) / 2))
        seeds = np.random.SeedSequence(derived['random_seed']).spawn(len(derived['layers']))
        layers = []
        for entry, rng in zip(derived['layers'], seeds):
            frac = float(entry['cn2_fraction'])
            # Zero-Cn2 layers must contribute precisely zero OPD, not NaNs.
            if frac == 0:
                tile = np.zeros((tile_size, tile_size), dtype=float)
                r0 = np.inf
            else:
                r0 = float(entry['r0_500nm_m'])
                seed = int(rng.generate_state(1, dtype=np.uint32)[0])
                tile = phasescreens.ft_phase_screen(
                    r0=r0, N=tile_size, delta=float(pupil_pixel_scale_m),
                    L0=float(cfg['L0_m']), l0=1e-10, seed=seed,
                )
            layers.append(FrozenFlowLayer(
                altitude_m=float(entry['altitude_m']),
                cn2_fraction=frac,
                r0_500nm_m=r0,
                wind_x_m_per_s=float(entry['wind_x_m_per_s']),
                wind_y_m_per_s=float(entry['wind_y_m_per_s']),
                pixel_scale_m=float(pupil_pixel_scale_m),
                size=int(grid_dim),
                phase_tile_rad=np.asarray(tile, dtype=float),
            ))
        return cls(layers, resolved['derived']['simulator_runtime']['time_step_s'])

    def opd_at(self, time_s: float) -> np.ndarray:
        """Sum independent layer OPDs without a pupil mask or AO correction."""
        summed = np.zeros(self.output_shape, dtype=float)
        for layer in self.layers:
            summed += layer.sample_opd(time_s)
        return summed

    def next_opd(self) -> np.ndarray:
        opd = self.opd_at(self.time_s)
        self.time_s += self.time_step_s
        return opd


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
