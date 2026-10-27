"""Compare compiled MuJoCo, source USD and optional LIVE Isaac LL snapshot.

No Kit launch. pxr is only required for --refresh-reference; the checked-in
reference lets the normal server audit run with the MuJoCo dependencies alone.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "mujoco/deploy/deploy_mujoco/go2_piper"
ASSET = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/assets/go2_piper"
sys.path.insert(0, str(ENTRY))
from paint_loco import PaintBridge, JOINT_NAMES
from physics_audit import PROFILES, describe_model


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refresh_reference():
    from pxr import Usd, UsdPhysics, UsdGeom
    stage = Usd.Stage.Open(str(ASSET / "go2_piper.usd"))
    bodies, joints, contacts = {}, {}, []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI):
            api = UsdPhysics.MassAPI(prim)
            mass = api.GetMassAttr().Get()
            com = np.asarray(api.GetCenterOfMassAttr().Get(), dtype=float)
            inertia = np.asarray(api.GetDiagonalInertiaAttr().Get(), dtype=float)
            bodies[prim.GetName()] = {"mass_kg": float(mass or 0),
                "com_local": com.tolist() if np.isfinite(com).all() else None,
                "inertia_diagonal": inertia.tolist(),
                "principal_axes_wxyz": [api.GetPrincipalAxesAttr().Get().GetReal(),
                                         *api.GetPrincipalAxesAttr().Get().GetImaginary()]}
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            drive = UsdPhysics.DriveAPI(prim, "angular")
            joints[prim.GetName()] = {"axis_in_joint_frame": joint.GetAxisAttr().Get(),
                "limits_rad": np.deg2rad([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]).tolist(),
                "drive_max_force_Nm": drive.GetMaxForceAttr().Get()}
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            contacts.append(str(prim.GetPath()))
    # Read source actuator overrides without importing Isaac or running source.
    tree = ast.parse((ASSET / "go2_piper_articulation_cfg.py").read_text(encoding="utf-8"))
    actuator_dict = next(kw.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                         for kw in node.keywords if kw.arg == "actuators")
    actuators = {}
    for key, call in zip(actuator_dict.keys, actuator_dict.values):
        actuators[ast.literal_eval(key)] = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    source_files = [ASSET / "go2_piper_articulation_cfg.py", ASSET / "go2_piper.usd",
                    *sorted((ASSET / "configuration").glob("*.usd"))]
    return {"source_files_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in source_files},
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage), "bodies": bodies, "joints": joints,
            "actuator_cfg": actuators, "collision_prims": contacts,
            "authored_positive_mass_sum_kg": sum(v["mass_kg"] for v in bodies.values() if v["mass_kg"] > 0),
            "note": "Source-USD authoring only. Zero/auto mass and fixed-body merging require live PhysX verification; "
                    "body frame and principal axes must be checked before copying inertia."}


def read_policy(path):
    with Path(path).open("rb") as stream:
        return torch.jit.load(stream, map_location="cpu").eval()


def compare_policies(left, right):
    a, b = read_policy(left), read_policy(right)
    generator = torch.Generator().manual_seed(42)
    probe = torch.cat((torch.zeros(1, 247), torch.randn(32, 247, generator=generator)), dim=0)
    with torch.inference_mode():
        output_a, output_b = a(probe), b(probe)
    if output_a.shape != (33, 12) or output_b.shape != (33, 12):
        raise ValueError("Both policies must be 247->12")
    state_a, state_b = a.state_dict(), b.state_dict()
    keys_match = state_a.keys() == state_b.keys()
    state_equal = keys_match and all(torch.equal(state_a[k], state_b[k]) for k in state_a)
    return {"left_sha256": digest(left), "right_sha256": digest(right),
            "state_dict_exactly_equal": state_equal, "probe_count": 33,
            "max_abs_output_difference": float((output_a - output_b).abs().max()),
            "note": "Different serialization hashes can still contain identical actor tensors."}


def replay_snapshot(bridge, snapshot, policy_path):
    if snapshot["joint_names"] != JOINT_NAMES:
        raise ValueError("Snapshot joint order does not match the trained canonical interface")
    model, data = bridge.model, bridge.data
    free = model.body_jntadr[bridge.base]
    q, v = model.jnt_qposadr[free], model.jnt_dofadr[free]
    data.qpos[q:q + 7] = [*snapshot["root_pos_w"], *snapshot["root_quat_wxyz"]]
    data.qpos[bridge.qpos_ids] = snapshot["joint_pos"]
    data.qvel[bridge.qvel_ids] = snapshot["joint_vel"]
    mujoco.mj_forward(model, data)
    rotation = data.xmat[bridge.base].reshape(3, 3)
    omega = np.asarray(snapshot["root_ang_vel_b"])
    # Free-joint angular qvel is local; translation is world at the link origin.
    data.qvel[v:v + 3] = (rotation @ np.asarray(snapshot["root_lin_vel_b"]) -
                          np.cross(rotation @ omega, rotation @ model.body_ipos[bridge.base]))
    data.qvel[v + 3:v + 6] = omega
    source = np.asarray(snapshot["observation"], dtype=np.float32)
    bridge.previous_action = source[48:60].copy()
    mujoco.mj_forward(model, data)
    observation = bridge.observation(np.asarray(snapshot["command"]))
    slices = {"linear": (0, 3), "angular": (3, 6), "gravity": (6, 9), "command": (9, 12),
              "joint_position": (12, 30), "joint_velocity": (30, 48), "previous_action": (48, 60), "height_scan": (60, 247)}
    result = {"observation_block_max_abs_difference": {
        name: float(np.max(np.abs(observation[start:end] - source[start:end])))
        for name, (start, end) in slices.items()},
        "default_joint_pos_max_abs_difference": float(np.max(np.abs(bridge.default - snapshot["default_joint_pos"]))),
        "note": "State replay checks interface algebra, not trajectory/dynamics. For Flat snapshots only."}
    if policy_path:
        policy = read_policy(policy_path)
        with torch.inference_mode():
            on_source = policy(torch.from_numpy(source).unsqueeze(0)).numpy()[0]
            on_replay = policy(torch.from_numpy(observation).unsqueeze(0)).numpy()[0]
        result["export_vs_native_action_max_abs_difference"] = float(np.max(np.abs(on_source - snapshot["policy_action"])))
        result["replayed_vs_native_action_max_abs_difference"] = float(np.max(np.abs(on_replay - snapshot["policy_action"])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ENTRY / "paint_loco.yaml")
    parser.add_argument("--reference", type=Path, default=ENTRY / "isaac_asset_reference.json")
    parser.add_argument("--refresh-reference", action="store_true")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--compare-policy", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--physics-profile", choices=PROFILES, default="legacy")
    parser.add_argument("--report", type=Path, default=ROOT / "logs/mujoco/interface_audit.json")
    args = parser.parse_args()
    if args.refresh_reference:
        args.reference.write_text(json.dumps(refresh_reference(), indent=2), encoding="utf-8")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    stale = [name for name, sha in reference["source_files_sha256"].items()
             if not (ROOT / name).is_file() or digest(ROOT / name) != sha]
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["physics_profile"] = args.physics_profile
    bridge = PaintBridge(cfg)
    model = bridge.model
    result = {"stale_or_missing_reference_sources": stale, "mujoco": describe_model(bridge),
              "usd_authored_positive_mass_sum_kg": reference["authored_positive_mass_sum_kg"],
              "source_actuator_cfg": reference["actuator_cfg"], "joint_effort_comparison": {},
              "body_mass_comparison": {}, "inertia_comparison": {}, "joint_control_comparison": {}}
    for index, (name, effort) in enumerate(zip(JOINT_NAMES, result["mujoco"]["effort_limits_Nm"])):
        usd_effort = reference["joints"][name]["drive_max_force_Nm"]
        result["joint_effort_comparison"][name] = {"usd_drive": usd_effort, "mujoco_effective": effort}
        source_actuator = next(value for value in reference["actuator_cfg"].values()
                               if any(re.fullmatch(pattern, name) for pattern in value["joint_names_expr"]))
        result["joint_control_comparison"][name] = {
            "source_kp": source_actuator["stiffness"], "mujoco_kp": float(bridge.kp[index]),
            "source_kd": source_actuator["damping"], "mujoco_kd": float(bridge.kd[index]),
            "source_delay_range": [source_actuator["min_delay"], source_actuator["max_delay"]],
            "mujoco_delay": int(cfg["delay_steps"]),
            "source_limits_rad": reference["joints"][name]["limits_rad"],
            "mujoco_limits_rad": model.jnt_range[bridge.joint_ids[index]].tolist()}
    for index in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        source_name = {"base_link": "base", "Piper": "link0"}.get(name, name)
        source = reference["bodies"].get(source_name)
        if source:
            result["body_mass_comparison"][name] = {"usd_authored": source["mass_kg"], "mujoco": float(model.body_mass[index])}
            result["inertia_comparison"][name] = {"usd": source,
                "mujoco_diagonal": model.body_inertia[index].tolist(),
                "mujoco_principal_axes_wxyz": model.body_iquat[index].tolist(), "mujoco_com_local": model.body_ipos[index].tolist()}
    result["body_comparison_note"] = "Named bodies may merge different fixed links. Raw diagonal inertia axes differ; do not copy numbers without frame conversion."
    if args.compare_policy:
        if not args.policy:
            parser.error("--compare-policy requires --policy")
        result["policy_comparison"] = compare_policies(args.policy, args.compare_policy)
    if args.snapshot:
        snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
        result["snapshot_replay"] = replay_snapshot(bridge, snapshot, args.policy)
        result["live_isaac_actuators"] = snapshot["actuators"]
        result["live_isaac_delay_cfg"] = snapshot["actuator_delay_cfg"]
        result["live_isaac_mass_sum_kg"] = sum(snapshot["bodies"]["mass_kg"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.report), "reference_stale": stale,
        "usd_authored_mass_kg": result["usd_authored_positive_mass_sum_kg"],
        "mujoco_mass_kg": result["mujoco"]["total_model_mass_kg"],
        "effort_limits_Nm": result["mujoco"]["effort_limits_Nm"],
        **{key: result[key] for key in ("policy_comparison", "snapshot_replay") if key in result}}, indent=2))


if __name__ == "__main__":
    main()
