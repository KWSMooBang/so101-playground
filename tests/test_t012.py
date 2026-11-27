"""T012 CPU checks: camera-delta plumbing, reach-error detector, static-probe isolation.

    PYTHONPATH=src <flux3 venv>/bin/python -m pytest tests/test_t012.py -q

The T010-trajectory and server tests skip when their inputs (runs/T010, torch/fastapi) are missing.
"""

from __future__ import annotations

import glob
import json
import math
import os
from pathlib import Path
import re
import types

import numpy as np
import pytest

from so101_playground import camera_delta as cd
from so101_playground import reach_error as reach
from so101_playground import t012_probe as probe

REPO = Path(__file__).resolve().parents[1]
BENCH = Path(os.environ.get("SO101_BENCH_ROOT", reach._DEFAULT_BENCH))
T010 = Path("/outputs/so101-playground/runs/T010")


# ---------------------------------------------------------------- camera delta
def _quat_xyz(r, p, y):
    """isaaclab.utils.math.quat_from_euler_xyz (wxyz), re-implemented: q = qz * qy * qx."""
    cr, sr, cp, sp, cy, sy = (f(a / 2) for a in (r, p, y) for f in (math.cos, math.sin))
    return np.array([cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp,
                     cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp])


def _quat_to_m(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def test_base_matches_bench():
    src = (BENCH / "source/so101_bench/so101_bench/benchmark.py")
    if not src.exists():
        pytest.skip("bench not found")
    text = src.read_text()
    vals = {k: float(v) for k, v in re.findall(r"^\s+(overhead_camera_\w+): float = ([-\d.]+)", text, re.M)}
    assert (vals["overhead_camera_position_x_m"], vals["overhead_camera_position_y_m"],
            vals["overhead_camera_position_z_m"]) == cd.BASE_POS_M
    assert (vals["overhead_camera_roll_deg"], vals["overhead_camera_pitch_deg"],
            vals["overhead_camera_yaw_deg"]) == cd.BASE_RPY_DEG
    assert vals["overhead_camera_horizontal_fov_deg"] == cd.HFOV_DEG


def test_rpy_matches_isaaclab_convention():
    for rpy in [cd.BASE_RPY_DEG, (10.0, -20.0, 30.0), (60.0, 5.0, -12.0)]:
        m = cd.rpy_to_matrix(rpy)
        q = _quat_xyz(*(math.radians(a) for a in rpy))
        np.testing.assert_allclose(m, _quat_to_m(q), atol=1e-12)
        np.testing.assert_allclose(cd.matrix_to_rpy(m), rpy, atol=1e-9)


def test_parse_and_zero():
    assert cd.CameraDelta.parse("").is_zero()
    d = cd.CameraDelta.parse("tilt=4, pan=-8")
    assert (d.tilt, d.pan, d.roll) == (4.0, -8.0, 0.0)
    assert d.label() == "tilt+4_pan-8"
    with pytest.raises(ValueError):
        cd.CameraDelta.parse("yaw=3")
    f = cd.bench_fields(cd.CameraDelta())
    assert f["overhead_camera_roll_deg"] == pytest.approx(51.361)
    assert f["overhead_camera_pitch_deg"] == pytest.approx(-0.169)
    assert f["overhead_camera_yaw_deg"] == pytest.approx(0.603)


def test_tilt_is_roll_field_pan_is_yaw_field():
    for a in (-8, -4, 4, 8, 12):
        f = cd.bench_fields(cd.CameraDelta(pan=a))
        assert f["overhead_camera_yaw_deg"] == pytest.approx(0.603 - a, abs=1e-9)
        assert f["overhead_camera_roll_deg"] == pytest.approx(51.361, abs=1e-9)
        assert f["overhead_camera_pitch_deg"] == pytest.approx(-0.169, abs=1e-9)
        f = cd.bench_fields(cd.CameraDelta(tilt=a))
        assert f["overhead_camera_roll_deg"] == pytest.approx(51.361 + a, abs=0.02)
        assert abs(f["overhead_camera_pitch_deg"] + 0.169) < 0.02 and abs(f["overhead_camera_yaw_deg"] - 0.603) < 0.02


def test_directions_by_projection():
    """+tilt: looks further down the table (optical axis hits the table further from the camera, the scene moves
    down in the image); +pan: turns right (scene moves left); +roll: scene rotates clockwise."""
    pts = np.array([[0.15, 0.0, 0.0], [0.05, 0.18, 0.0], [0.25, -0.1, 0.0]])  # table, robot base, a layout spot
    p0 = cd.project(pts, *cd.apply_delta(cd.CameraDelta()))
    assert np.all((p0 > 0) & (p0 < [640, 480]))  # all in view at the bench pose
    hit0 = cd.optical_axis_hit(*cd.apply_delta(cd.CameraDelta()))
    for a in (4, 8):
        pos, rpy = cd.apply_delta(cd.CameraDelta(tilt=a))
        assert cd.optical_axis_hit(pos, rpy)[1] > hit0[1] + 0.01  # further from the camera (camera at y=-0.34)
        assert np.all(cd.project(pts, pos, rpy)[:, 1] > p0[:, 1] + 5)  # scene moves down
        pos, rpy = cd.apply_delta(cd.CameraDelta(pan=a))
        assert np.all(cd.project(pts, pos, rpy)[:, 0] < p0[:, 0] - 5)  # scene moves left
    # +roll: a point right of centre moves down (clockwise in an image with v down)
    c = np.array([[0.30, 0.0, 0.0]])
    u0, v0 = cd.project(c, *cd.apply_delta(cd.CameraDelta()))[0]
    u1, v1 = cd.project(c, *cd.apply_delta(cd.CameraDelta(roll=5)))[0]
    centre = np.array([320, 240])
    ang = lambda u, v: math.atan2(v - centre[1], u - centre[0])  # noqa: E731
    assert ang(u1, v1) > ang(u0, v0)  # clockwise on screen


def test_image_shift_magnitude():
    """Sanity: 4 deg of pan moves the table centre by ~ f * tan(4 deg) px (f = 424 px at 74 deg HFOV)."""
    f = 640 / (2 * math.tan(math.radians(37)))
    pt = np.array([[0.15, 0.0, 0.0]])
    du = cd.project(pt, *cd.apply_delta(cd.CameraDelta(pan=4)))[0, 0] - cd.project(pt, *cd.apply_delta(cd.CameraDelta()))[0, 0]
    assert 0.5 * f * math.tan(math.radians(4)) < -du < 1.5 * f * math.tan(math.radians(4))


def test_overhead_dr_payload():
    base = {"background_mesh_index": 1.0, "wrist_camera_roll_deg": -45.0, "overhead_camera_roll_deg": 51.361,
            "overhead_camera_pitch_deg": -0.169, "overhead_camera_yaw_deg": 0.603,
            "overhead_camera_position_x_m": 0.0, "overhead_camera_position_y_m": -0.34,
            "overhead_camera_position_z_m": 0.45722}
    assert cd.overhead_dr(base, cd.CameraDelta()) == base
    out = cd.overhead_dr(base, cd.CameraDelta(pan=8, z=2))
    assert out["wrist_camera_roll_deg"] == -45.0 and out["background_mesh_index"] == 1.0
    assert out["overhead_camera_yaw_deg"] == pytest.approx(0.603 - 8)
    assert out["overhead_camera_position_z_m"] == pytest.approx(0.47722)
    assert base["overhead_camera_yaw_deg"] == 0.603  # input untouched
    assert cd.overhead_dr(None, cd.CameraDelta(tilt=4))["overhead_camera_roll_deg"] == pytest.approx(55.361, abs=0.02)


def test_bench_spec_accepts_fields():
    """The merged dict must pass DomainRandomizationSpec.from_json (only if the bench imports without Isaac)."""
    try:
        from so101_bench.benchmark import DomainRandomizationSpec  # noqa: F401
    except Exception:
        pytest.skip("so101_bench not importable here")
    from so101_bench.benchmark import DomainRandomizationSpec

    spec = DomainRandomizationSpec.from_json(cd.overhead_dr(DomainRandomizationSpec().to_json(), cd.CameraDelta(tilt=8)))
    assert spec.overhead_camera_roll_deg == pytest.approx(59.361, abs=0.03)


# ---------------------------------------------------------------- reach error
def test_close_rules_synthetic():
    closed, small_open = -11.7, 9.0
    thin = [closed] * 10 + [small_open] * 20 + list(np.linspace(small_open, -12, 8)) + [-12] * 10
    t = reach.first_close_tick(thin)
    assert thin[t] < -8 and thin[t - 1] >= -8
    assert reach.detect_close(thin) == (t, "abs")
    # starting closed is not a close; no opening -> no event
    assert reach.detect_close([closed] * 50) == (None, None)
    # thick object: opens to 38, closes to 7 -> relative rule
    thick = [closed] * 5 + [38.0] * 20 + [30.0, 20.0, 15.0, 7.0] + [7.0] * 10
    tt, kind = reach.detect_close(thick)
    assert kind == "rel" and thick[tt] <= 18.0 and thick[tt - 1] > 18.0
    # dithering pen (opens 11, drifts to ~0) -> no event
    assert reach.detect_close([closed] * 5 + list(np.linspace(11, 0.5, 200))) == (None, None)
    # online == offline
    det = reach.CloseDetector()
    hits = [i for i, j in enumerate(thick) if det(j)]
    assert hits == [tt]


def test_kinematics_base_is_root():
    """The kinematics' base body frame is the robot root (offline reach error uses the root pose)."""
    kin = reach.kinematics()
    dims = json.loads(reach.ASSET_DIMS.read_text())["robot_joints_root_frame"]
    np.testing.assert_allclose(kin._USD_JOINTS["Rotation"][0], dims["Rotation"]["pos_root_m"], atol=1e-6)


def test_object_centre_offset():
    off = reach.centre_offset_root("black pen")
    assert abs(off[0]) < 0.01 and abs(off[1]) < 0.01
    c = reach.object_centre("black pen", [0.1, 0.0, 0.0], yaw_rad=math.pi)
    np.testing.assert_allclose(c[:2], [0.1 - off[0], -off[1]], atol=1e-9)


@pytest.mark.skipif(not (T010 / "queue/done").exists(), reason="T010 run not present")
def test_reach_error_on_t010_trajectories():
    lay = [json.loads(l) for l in (REPO / "tasks/layouts/home_objects_layouts.jsonl").read_text().splitlines()][:6]
    eps = [json.loads(l) for l in (REPO / "tasks/home_objects.jsonl").read_text().splitlines()][:6]
    rows = []
    for f in sorted(glob.glob(str(T010 / "queue/done/*.json"))):
        r = json.loads(Path(f).read_text())
        o = reach.reach_error_offline(T010 / "episodes" / f"{r['id']}.npz", lay[r["row"]], eps[r["row"]]["objects"][0],
                                      r.get("jitter"))
        rows.append((r["row"], r["success"], o["close_kind"], o["reach_err_cm"]))
    assert len(rows) == 288
    # every success has a close event; the only no-close episodes are black-pen (row 0) failures
    assert all(k is not None for _, s, k, _ in rows if s)
    assert {row for row, s, k, _ in rows if k is None} <= {0}
    # the remote (row 1) closes by the relative rule, the thin objects by the brief's rule
    assert all(k == "rel" for row, s, k, _ in rows if row == 1 and s)
    assert all(k == "abs" for row, s, k, _ in rows if row in (0, 2, 3, 4) and k)
    # successful grasps are close to the object centre
    succ = [e for row, s, k, e in rows if s]
    assert np.median(succ) < 2.0 and np.percentile(succ, 90) < 5.0


# ---------------------------------------------------------------- static probe
def test_probe_schedule():
    s = probe.probe_schedule(3, 45, latents=2)
    assert s["queries"] == [45, 61, 77]
    assert s["scores"][61] == [(45, 1)] and s["scores"][77] == [(45, 2), (61, 1)] and s["scores"][109] == [(77, 2)]
    assert s["total_ticks"] == 109
    s1 = probe.probe_schedule(2, 45, latents=1)
    assert s1["total_ticks"] == 77 and all(k == 1 for v in s1["scores"].values() for _, k in v)
    # every frame a score needs is marked as needed
    for tick, items in s["scores"].items():
        for q, k in items:
            for o in probe.OFFSETS[k]:
                assert probe.frames_needed(q + o, s["queries"], 2)
    assert probe.score_due(77, [45, 61, 77], 2) == [(45, 2), (61, 1)]


def test_probe_keys_distinct():
    key = "abc123.t012-d0-home0-r0"
    assert probe.probe_dir("/r/episodes", key) != probe.rollout_dir("/r/episodes", key)
    assert probe.probe_attempt(key) != key
    # a probe key can never equal another job's rollout key (job ids never end in ".static")
    assert probe.probe_dir("/r", key).endswith(".static")


def _fake_server():
    torch = pytest.importorskip("torch")
    pytest.importorskip("fastapi")
    pytest.importorskip("torchvision")
    from so101_playground import internw0_probe_server as srv

    class Sess:
        made = []

        def __init__(self, *a):
            self.obs, self.pending_model_actions = [], []
            Sess.made.append(self)

        def update_obs(self, _):
            self.pending_model_actions.pop()

        def get_action(self, ob):
            self.obs.append(ob["observation"]["joint_action"]["vector"].copy())
            self.pending_model_actions = [0] * 3
            return np.zeros((16, 6), np.float32) + len(self.obs)  # chunk depends on the session's memory

    pol = srv.Policy.__new__(srv.Policy)
    pol.session_cls, pol.runtime, pol.session_cfg = Sess, None, None
    pol.sessions, pol.session_first, pol.queries = srv.OrderedDict(), {}, srv.OrderedDict()
    pol.seed_per_query, pol.save_latents_dir, pol.camera_hw, pol.n_future = True, None, (8, 8), 2
    pol._latents = lambda canv: torch.zeros(4, 1, 4, 4)
    pol._unpatch_delta = lambda d, hw: torch.zeros(4, 2, 4, 4)
    pol._last_delta = None
    return srv, pol, Sess


def _act(srv, dir_, ep, frame, state):
    import base64
    import io

    from PIL import Image

    b = io.BytesIO()
    Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(b, format="PNG")
    png = base64.b64encode(b.getvalue()).decode()
    return {"camera_names": ["overhead", "wrist"], "images_png_base64": [png, png], "state": state,
            "episode": ep, "instruction": "x", "trace": {"dir": dir_, "episode": ep, "frame": frame}}


def test_probe_session_isolation_on_server():
    """Static-probe queries (probe trace dir) never enter the rollout's session: the rollout's first chunk equals
    the probe's first chunk, and the rollout session only ever sees rollout states."""
    srv, pol, Sess = _fake_server()
    run, key, ep = "/r/episodes", "att.t012-d0-home0-r0", 5000
    pdir, rdir = probe.probe_dir(run, key), probe.rollout_dir(run, key)
    probe_chunks = [pol.predict(_act(srv, pdir, ep, f, [9.0] * 6)) for f in (45, 61, 77)]
    first = pol.predict(_act(srv, rdir, ep, 45, [1.0] * 6))
    np.testing.assert_array_equal(first, probe_chunks[0])
    pol.predict(_act(srv, rdir, ep, 61, [2.0] * 6))
    rs = pol.sessions[(rdir, ep)]
    ps = pol.sessions[(pdir, ep)]
    assert rs is not ps and len(Sess.made) == 2
    assert [o[0] for o in rs.obs] == [1.0, 2.0]
    assert [o[0] for o in ps.obs] == [9.0, 9.0, 9.0]
    # predictions are keyed apart too: a probe score can only find probe predictions
    assert (pdir, ep, 45) in pol.queries and (rdir, ep, 45) in pol.queries
    assert pol.queries[(pdir, ep, 45)] is not pol.queries[(rdir, ep, 45)]
