"""Compiled-model diagnostics and explicit A/B options; legacy is unchanged."""

import mujoco
import numpy as np

PROFILES = ("legacy", "point-contact", "arm-effort-100", "point-contact-arm100")


def apply_profile(model, joint_ids, actuator_ids, profile="legacy", prefix=""):
    if profile not in PROFILES:
        raise ValueError(f"Unknown physics profile: {profile}")
    if profile in ("point-contact", "point-contact-arm100"):
        for name in ("FR", "FL", "RR", "RL"):
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + name)
            if index < 0:
                raise ValueError(f"Missing foot geom: {prefix + name}")
            # Keep tangential friction, size, position and contact solver intact.
            model.geom_condim[index] = 3
    if profile in ("arm-effort-100", "point-contact-arm100"):
        for index in actuator_ids[12:]:
            gear = model.actuator_gear[index, 0]
            model.actuator_ctrllimited[index] = True
            model.actuator_ctrlrange[index] = [-100. / abs(gear), 100. / abs(gear)]
            if model.actuator_forcelimited[index]:
                model.actuator_forcerange[index] = [-100. / abs(gear), 100. / abs(gear)]


def describe_model(bridge, prefix=""):
    model = bridge.model
    feet = {}
    for name in ("FR", "FL", "RR", "RL"):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, prefix + name)
        feet[name] = {"condim": int(model.geom_condim[index]), "priority": int(model.geom_priority[index]),
                      "friction_tangent_torsion_rolling": model.geom_friction[index].tolist(),
                      "position_local": model.geom_pos[index].tolist(), "size": model.geom_size[index].tolist()}
    ids = bridge.actuator_ids
    gears = np.abs(model.actuator_gear[ids, 0])
    effort = np.where(model.actuator_ctrllimited[ids],
                      np.max(np.abs(model.actuator_ctrlrange[ids]), axis=1) * gears, np.inf)
    force_limit = np.where(model.actuator_forcelimited[ids],
                           np.max(np.abs(model.actuator_forcerange[ids]), axis=1) * gears, np.inf)
    effort = np.minimum(effort, force_limit)
    return {"profile": bridge.cfg.get("physics_profile", "legacy"),
            "joint_names": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(i)) for i in bridge.joint_ids],
            "default_joint_pos": bridge.default.tolist(), "kp": bridge.kp.tolist(), "kd": bridge.kd.tolist(),
            "effort_limits_Nm": [float(v) if np.isfinite(v) else None for v in effort],
            "physics_dt": float(model.opt.timestep), "control_decimation": int(bridge.cfg["control_decimation"]),
            "delay_physics_steps": int(bridge.cfg["delay_steps"]), "feet": feet,
            "total_model_mass_kg": float(model.body_mass.sum()),
            "note": "condim=3 A/B disables torsional/rolling contact constraints, not an exact PhysX equivalence. "
                    "Arm-effort-100 is an opt-in source-USD limit test; verify live Isaac actuator limits first."}
