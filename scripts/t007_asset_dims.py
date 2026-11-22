"""T007: bounding boxes of the home-object bench assets, the plastic bin, and the SO-101 joint frames.

Reads the bench USDs (no physics). Writes JSON with, per asset, the axis-aligned box of the render meshes in
the asset's own root frame (USD root scale included, which is how the bench spawns them), and for the robot the
world-frame positions of the joint frames once the root is placed at the bench base pose.

    source <scaffold-env>/envs/isaac/env.sh
    CUDA_VISIBLE_DEVICES=1 python scripts/t007_asset_dims.py --out docs/home_objects/asset_dims.json
"""
import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
app = AppLauncher(headless=True).app

from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402
import os  # noqa: E402

ROOT = Path(os.environ["SO101_BENCH_ROOT"]) / "source/so101_bench/so101_bench/assets/usd"
OBJECTS = ["black pen", "red pen", "purple toothbrush", "razor", "black remote", "blue paintbrush"]


def bbox(stage, physics_only=True):
    """Box of the collision meshes (as the bench footprint generator uses) in the root frame, root scale kept.

    Render meshes are not used: some assets carry stray render geometry (black_remote's render box is 117 cm).
    """
    root = stage.GetDefaultPrim()
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    root_to_world = xc.GetLocalToWorldTransform(root)
    scale = Gf.Transform(root_to_world).GetScale()
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    coll = [p for p in meshes if p.HasAPI(UsdPhysics.CollisionAPI)] or \
        [p for p in meshes if any("physics" in s.lower() for s in p.GetPath().pathString.split("/")[:-1])]
    # Transform the mesh points themselves (authored extents can be stale: black_remote's say 117 cm).
    w2r = root_to_world.GetInverse()
    lo, hi = [1e9] * 3, [-1e9] * 3
    for p in (coll if physics_only and coll else meshes):
        m = xc.GetLocalToWorldTransform(p)
        for q in UsdGeom.Mesh(p).GetPointsAttr().Get() or []:
            v = w2r.Transform(m.Transform(Gf.Vec3d(q)))
            lo = [min(a, v[i]) for i, a in enumerate(lo)]
            hi = [max(a, v[i]) for i, a in enumerate(hi)]
    rng = Gf.Range3d(Gf.Vec3d(*lo), Gf.Vec3d(*hi))
    lo, hi = [rng.GetMin()[i] * scale[i] for i in range(3)], [rng.GetMax()[i] * scale[i] for i in range(3)]
    return {"n_collision_meshes": len(coll), "min_m": lo, "max_m": hi, "size_cm": [100 * (h - l) for l, h in zip(lo, hi)], "root_scale": list(scale)}


out = {"objects": {}}
for name in OBJECTS:
    stage = Usd.Stage.Open(str(ROOT / "objects" / (name.replace(" ", "_") + ".usdc")))
    out["objects"][name] = bbox(stage)
out["plastic_bin"] = bbox(Usd.Stage.Open(str(ROOT / "plastic_bin.usdc")))

# Robot: joint frame origins in the robot root frame (body0 frame * localPos0).
stage = Usd.Stage.Open(str(ROOT / "SO-ARM101-USD.usd"))
xc = UsdGeom.XformCache(Usd.TimeCode.Default())
root = stage.GetDefaultPrim()
w2r = xc.GetLocalToWorldTransform(root).GetInverse()
joints = {}
for prim in stage.Traverse():
    if prim.IsA(UsdPhysics.RevoluteJoint):
        j = UsdPhysics.Joint(prim)
        b0 = j.GetBody0Rel().GetTargets()
        lp = j.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0)
        if b0:
            m = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(b0[0]))
            p = w2r.Transform(m.Transform(Gf.Vec3d(lp)))
            axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
            joints[prim.GetName()] = {"pos_root_m": list(p), "body0": str(b0[0]), "axis": str(axis)}
out["robot_joints_root_frame"] = joints
out["robot_root_bbox"] = bbox(stage)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
app.close()
