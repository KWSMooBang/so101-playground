"""T012: overhead-camera pose delta (perception gap) for the SO-101 bench, as bench DomainRandomizationSpec fields.

The bench applies the overhead camera pose at every reset from ``episode_spec.domain_randomization``
(``so101_bench.mdp.resets._set_camera_domain_randomization``): position ``overhead_camera_position_{x,y,z}_m``
(env frame) and orientation ``quat_from_euler_xyz(roll, pitch, yaw)`` = Rz(yaw) Ry(pitch) Rx(roll) (extrinsic
XYZ, the same as the cfg's ``euler_angles_to_quat``), in the USD/OpenGL camera convention: the camera looks along
its local -Z, local +X is image right, local +Y is image up.

Bench default: position (0, -0.34, 0.457) m, rpy (51.361, -0.169, 0.603) deg. With roll alone the optical axis
is Rx(roll)(0, 0, -1) = (0, sin roll, -cos roll): the camera sits at y = -0.34 (the table edge opposite the robot,
whose base is at y = +0.18) and looks down, tilted 51.4 deg from vertical towards +y. So the bench field
``overhead_camera_roll_deg`` is the tilt from vertical.

The delta is defined in the camera's own terms, then converted back to the bench's (roll, pitch, yaw) fields:

  tilt  (deg) rotation about the camera's local +X (image horizontal axis): R' = R Rx(tilt).
              +tilt = the optical axis swings towards image-up = it looks further down the table (away from the
              camera, towards the robot); the scene moves DOWN in the image. At the bench pose this is almost
              exactly +overhead_camera_roll_deg.
  pan   (deg) rotation about the world vertical through the camera (a tripod pan head): R' = Rz(-pan) R.
              +pan = the camera turns to its right; the scene moves LEFT in the image. Exactly
              overhead_camera_yaw_deg - pan.
  roll  (deg) rotation about the optical axis: R' = R Rz(roll). +roll = the camera rotates counter-clockwise
              as seen from behind it; the scene rotates CLOCKWISE in the image.
  x, y, z (cm) added to the camera position, bench env frame (world axes; +z up).

The signs above are checked by projection in tests/test_t012.py and by renders in the T012 smoke contact sheet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

# Bench DomainRandomizationSpec defaults (so101_bench @ 470e09e, benchmark.py), copied so this file has no
# Isaac import; tests check them against the bench when it is importable.
BASE_POS_M = (0.0, -0.34, 0.45722)
BASE_RPY_DEG = (51.361, -0.169, 0.603)
HFOV_DEG = 74.0
IMAGE_WH = (640, 480)
FIELDS = ("tilt", "pan", "roll", "x", "y", "z")


@dataclass(frozen=True)
class CameraDelta:
    tilt: float = 0.0  # deg
    pan: float = 0.0  # deg
    roll: float = 0.0  # deg
    x: float = 0.0  # cm
    y: float = 0.0  # cm
    z: float = 0.0  # cm

    @classmethod
    def parse(cls, spec: str | dict | None) -> "CameraDelta":
        """'tilt=4,pan=-8' (or a dict, or None/'' for no delta)."""
        if not spec:
            return cls()
        if isinstance(spec, dict):
            items = spec.items()
        else:
            items = []
            for part in str(spec).split(","):
                part = part.strip()
                if not part:
                    continue
                if "=" not in part:
                    raise ValueError(f"camera delta part {part!r} is not key=value")
                k, v = part.split("=", 1)
                items.append((k.strip(), v))
        out = {}
        for k, v in items:
            if k not in FIELDS:
                raise ValueError(f"unknown camera delta field {k!r}; expected {FIELDS}")
            out[k] = float(v)
            if not math.isfinite(out[k]):
                raise ValueError(f"camera delta {k} must be finite")
        return cls(**out)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def is_zero(self) -> bool:
        return all(v == 0.0 for v in asdict(self).values())

    def label(self) -> str:
        nz = [f"{k}{v:+g}" for k, v in asdict(self).items() if v != 0.0]
        return "_".join(nz) if nz else "d0"


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def rpy_to_matrix(rpy_deg) -> np.ndarray:
    """Bench convention: quat_from_euler_xyz(roll, pitch, yaw) = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (math.radians(float(a)) for a in rpy_deg)
    return _rz(y) @ _ry(p) @ _rx(r)


def matrix_to_rpy(m: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rpy_to_matrix (pitch in (-90, 90) deg), degrees."""
    p = -math.asin(float(np.clip(m[2, 0], -1.0, 1.0)))
    r = math.atan2(m[2, 1], m[2, 2])
    y = math.atan2(m[1, 0], m[0, 0])
    return tuple(math.degrees(a) for a in (r, p, y))


def apply_delta(delta: CameraDelta, base_pos=BASE_POS_M, base_rpy=BASE_RPY_DEG):
    """(position m, rpy deg) of the overhead camera after the delta."""
    m = rpy_to_matrix(base_rpy)
    m = _rz(-math.radians(delta.pan)) @ m @ _rx(math.radians(delta.tilt)) @ _rz(math.radians(delta.roll))
    pos = tuple(float(b) + d / 100.0 for b, d in zip(base_pos, (delta.x, delta.y, delta.z)))
    return pos, matrix_to_rpy(m)


def bench_fields(delta: CameraDelta, base_pos=BASE_POS_M, base_rpy=BASE_RPY_DEG) -> dict[str, float]:
    """DomainRandomizationSpec fields for the overhead camera after the delta (merge into domain_randomization)."""
    pos, rpy = apply_delta(delta, base_pos, base_rpy)
    return {
        "overhead_camera_position_x_m": pos[0],
        "overhead_camera_position_y_m": pos[1],
        "overhead_camera_position_z_m": pos[2],
        "overhead_camera_roll_deg": rpy[0],
        "overhead_camera_pitch_deg": rpy[1],
        "overhead_camera_yaw_deg": rpy[2],
    }


def project(points_w, pos, rpy_deg, hfov_deg=HFOV_DEG, wh=IMAGE_WH) -> np.ndarray:
    """Pixel (u right, v down) of world points for a pinhole camera in the USD/OpenGL convention. [N, 2]."""
    pts = np.atleast_2d(np.asarray(points_w, dtype=float))
    m = rpy_to_matrix(rpy_deg)
    pc = (pts - np.asarray(pos, dtype=float)) @ m  # = (m^T (p - pos))^T
    f = wh[0] / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))
    depth = -pc[:, 2]
    u = wh[0] / 2.0 + f * pc[:, 0] / depth
    v = wh[1] / 2.0 - f * pc[:, 1] / depth
    return np.stack([u, v], axis=1)


def optical_axis_hit(pos, rpy_deg, z_plane: float = 0.0) -> np.ndarray:
    """Where the optical axis meets the plane z = z_plane (world xy)."""
    d = rpy_to_matrix(rpy_deg) @ np.array([0.0, 0.0, -1.0])
    t = (z_plane - pos[2]) / d[2]
    return np.asarray(pos[:2]) + t * d[:2]


def overhead_dr(dr: dict | None, delta: CameraDelta) -> dict:
    """A reset payload's ``domain_randomization`` with the overhead camera moved by ``delta``. The base pose is the
    row's own overhead pose (bench defaults for missing fields); the wrist camera and everything else are kept."""
    dr = dict(dr or {})
    if delta.is_zero():
        return dr
    pos = [float(dr.get(f"overhead_camera_position_{a}_m", b)) for a, b in zip("xyz", BASE_POS_M)]
    rpy = [float(dr.get(f"overhead_camera_{a}_deg", b)) for a, b in zip(("roll", "pitch", "yaw"), BASE_RPY_DEG)]
    dr.update(bench_fields(delta, pos, rpy))
    return dr
