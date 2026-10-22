"""Pupil-sampled NCPA modes shared by the simulator and preview (OPD metres)."""
import numpy as np

MODE_NAMES = (
    "Tip X (Z2)", "Tilt Y (Z3)", "Defocus (Z4)",
    "Oblique astigmatism (Z5)", "Vertical astigmatism (Z6)",
    "Coma Y (Z7)", "Coma X (Z8)", "Trefoil Y (Z9)",
    "Trefoil X (Z10)", "Spherical (Z11)",
)
CONVENTION = "Noll 2–11; X=columns, Y=increasing rows; piston removed and unit RMS on pupil>0"


def validate_coefficients(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(MODE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("Expected ten finite NCPA coefficients in nm RMS")
    if np.any(np.abs(values) > 10000):
        raise ValueError("NCPA coefficients must be between -10000 and 10000 nm")
    return values.copy()


def make_ncpa_basis(pupil, diameter):
    """Use the aperture's between-pixel centre and diameter, not padded width."""
    pupil = np.asarray(pupil)
    mask = pupil > 0
    if pupil.ndim != 2 or not np.isfinite(pupil).all() or mask.sum() < 12:
        raise ValueError("Invalid NCPA pupil")
    if not np.isfinite(diameter) or diameter <= 0:
        raise ValueError("Invalid pupil diameter")
    y, x = np.indices(pupil.shape, dtype=float)
    x = (x - (pupil.shape[1] - 1) / 2) / (diameter / 2)
    y = (y - (pupil.shape[0] - 1) / 2) / (diameter / 2)
    r2 = x*x + y*y
    modes = [x, y, 2*r2-1, 2*x*y, x*x-y*y,
             (3*r2-2)*y, (3*r2-2)*x, 3*x*x*y-y**3,
             x**3-3*x*y*y, 6*r2*r2-6*r2+1]
    basis = np.zeros((len(modes), *pupil.shape), dtype=float)
    for i, mode in enumerate(modes):
        samples = mode[mask] - mode[mask].mean()
        rms = np.sqrt(np.mean(samples**2))
        if rms <= 0:
            raise ValueError("Degenerate NCPA mode")
        basis[i, mask] = samples / rms
    return basis


def compose_ncpa(basis, coefficients_nm):
    return np.einsum("i,ijk->jk", validate_coefficients(coefficients_nm)*1e-9, basis)


class NcpaProfile:
    """One beam/source's baseline and cached adjustable OPD."""
    def __init__(self, pupil, diameter, baseline):
        self.pupil = np.asarray(pupil).copy()
        self.diameter = float(diameter)
        self.baseline = np.asarray(baseline).copy()
        self.basis = None
        self.coefficients = np.zeros(len(MODE_NAMES))
        self.revision = 0
        self.combined = self.baseline.copy()

    def apply(self, coefficients):
        coefficients = validate_coefficients(coefficients)
        if self.basis is None:
            self.basis = make_ncpa_basis(self.pupil, self.diameter)
        combined = self.baseline + compose_ncpa(self.basis, coefficients)
        self.coefficients = coefficients
        self.combined = combined
        self.revision += 1

    def metadata(self):
        return dict(coefficients_nm=self.coefficients.tolist(), revision=self.revision,
                    convention=CONVENTION)

    def preview_payload(self):
        return dict(self.metadata(), pupil=self.pupil.tolist(), diameter=self.diameter,
                    baseline=self.baseline.tolist())


def handle_ncpa_command(command, profiles):
    """Commands are 'ncpa get|set <JSON>'; no optical or SHM side effects."""
    import json
    _, action, raw = command.split(maxsplit=2)
    data = json.loads(raw)
    beam, source = int(data["beam"]), data["source"]
    profile = profiles[beam][source]
    if action == "get":
        result = profile.preview_payload()
    elif action == "set":
        profile.apply(data["coefficients_nm"])
        result = profile.metadata()
    else:
        raise ValueError("Unknown NCPA action")
    return json.dumps(dict(ok=True, ncpa=dict(result, beam=beam, source=source),
                           message="NCPA profile updated" if action == "set" else "NCPA profile"))
