"""Capture the live LL interface after Isaac initialization, without a rollout."""

import json
from pathlib import Path

import torch

JOINT_NAMES = [f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL")
               for part in ("hip", "thigh", "calf")] + [f"joint{i}" for i in range(1, 7)]


def save_interface_snapshot(env, observations, policy, path, provenance=None):
    raw = env.unwrapped
    robot = raw.scene["robot"]
    data = robot.data
    ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
    if names != JOINT_NAMES:
        raise ValueError(f"Unexpected canonical joint names: {names}")
    def row(value):
        return value.detach().cpu()[0].tolist()
    with torch.inference_mode():
        output = policy(observations)
    obs = observations["policy"]
    if obs.shape[1] != 247 or output.shape[1] != 12:
        raise ValueError("Snapshot requires the trained LL 247->12 contract")
    result = {"joint_names": names, "observation": row(obs), "policy_action": row(output),
              "provenance": provenance,
              "root_pos_w": row(data.root_pos_w), "root_quat_wxyz": row(data.root_quat_w),
              "root_lin_vel_b": row(data.root_lin_vel_b), "root_ang_vel_b": row(data.root_ang_vel_b),
              "command": row(raw.command_manager.get_command("base_velocity")),
              "joint_pos": row(data.joint_pos[:, ids]), "joint_vel": row(data.joint_vel[:, ids]),
              "default_joint_pos": row(data.default_joint_pos[:, ids]),
              "default_joint_vel": row(data.default_joint_vel[:, ids]),
              "physics_dt": raw.physics_dt, "control_dt": raw.step_dt,
              "actuator_delay_cfg": {name: {"min": getattr(cfg, "min_delay", None),
                                             "max": getattr(cfg, "max_delay", None)}
                                      for name, cfg in raw.cfg.scene.robot.actuators.items()},
              "actuators": {}, "bodies": {},
              "note": "Live simulator values, first environment; raw PhysX inertia matrices are not automatically "
                      "reinterpreted as MuJoCo principal-axis inertia. Snapshot does not certify dynamics equivalence."}
    for name in ("joint_stiffness", "joint_damping", "joint_armature", "joint_friction_coeff",
                 "joint_effort_limits", "joint_vel_limits", "soft_joint_pos_limits"):
        value = getattr(data, name, None)
        if value is not None:
            result[name] = row(value[:, ids])
    for name, actuator in robot.actuators.items():
        values = {"joint_names": list(actuator.joint_names)}
        for field in ("stiffness", "damping", "effort_limit", "velocity_limit"):
            value = getattr(actuator, field, None)
            if value is not None:
                values[field] = row(value) if value.ndim > 1 else value.detach().cpu().tolist()
        result["actuators"][name] = values
    view = robot.root_physx_view
    result["bodies"] = {"names": list(robot.body_names), "mass_kg": row(view.get_masses()),
                        "inertia_raw": row(view.get_inertias())}
    try:
        result["bodies"]["materials_raw"] = row(view.get_material_properties())
    except (AttributeError, RuntimeError) as error:
        result["bodies"]["materials_unavailable"] = str(error)
    for name in ("body_com_pos_b", "body_com_quat_b"):
        value = getattr(data, name, None)
        if value is not None:
            result["bodies"][name] = row(value)
    scanner = raw.scene["height_scanner"].data
    result["scanner_pos_w"] = row(scanner.pos_w)
    result["scanner_hit_z"] = row(scanner.ray_hits_w[..., 2])
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[EVAL] Live LL interface snapshot saved: {target}", flush=True)
    return result
