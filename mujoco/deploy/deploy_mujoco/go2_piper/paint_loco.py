"""PAINT Flat sim-to-sim: exported TorchScript, 247 observations, 12 leg actions.

Standalone: does not import Isaac Lab, RSL-RL or the project extension. Flat ground
only; 187 scan values are analytic intersections with the configured infinite
horizontal plane, not zero-filled sensor placeholders.
"""

import argparse
from collections import deque
from contextlib import nullcontext
from datetime import datetime
import json
import hashlib
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
import yaml
from physics_audit import PROFILES, apply_profile, describe_model

MUJOCO_ROOT = Path(__file__).resolve().parents[3]
LEG_NAMES = [f"{leg}_{part}_joint" for leg in ("FR", "FL", "RR", "RL") for part in ("hip", "thigh", "calf")]
JOINT_NAMES = LEG_NAMES + [f"joint{i}" for i in range(1, 7)]


def named_id(model, kind, name):
    result = mujoco.mj_name2id(model, kind, name)
    if result < 0:
        raise ValueError(f"Missing model {kind}: {name}")
    return result


def load_model(path):
    # MuJoCo 3.2.x's native file opener can fail on Windows Unicode paths.
    # The Python-side VFS also makes relative XML/mesh includes portable.
    assets = {}
    for file in path.parent.rglob("*"):
        if file.is_file() and file.suffix.lower() in (".xml", ".stl", ".obj", ".mtl", ".png", ".jpg"):
            if file.name in assets:
                raise ValueError(f"Duplicate VFS filename: {file.name}")
            assets[file.name] = file.read_bytes()
    return mujoco.MjModel.from_xml_string(path.read_text(encoding="utf-8"), assets=assets)


class PaintBridge:
    def __init__(self, cfg, payload_mass=0.0):
        self.cfg = cfg
        path = Path(cfg["xml_path"])
        self.model = load_model(path if path.is_absolute() else MUJOCO_ROOT / path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = float(cfg["simulation_dt"])
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.base = named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, cfg["base_body"])
        self.joint_ids = np.array([named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES])
        if np.any(self.model.jnt_type[self.joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("Expected 18 named hinge joints")
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]
        self.actuator_ids = []
        for joint in self.joint_ids:
            candidates = np.where((self.model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                                  & (self.model.actuator_trnid[:, 0] == joint))[0]
            if len(candidates) != 1:
                raise ValueError(f"Expected exactly one motor for joint id {joint}")
            self.actuator_ids.append(int(candidates[0]))
        self.actuator_ids = np.asarray(self.actuator_ids)
        self.gear = self.model.actuator_gear[self.actuator_ids, 0].copy()
        if np.any(self.gear == 0):
            raise ValueError("Zero motor gear")
        apply_profile(self.model, self.joint_ids, self.actuator_ids, cfg.get("physics_profile", "legacy"))
        self.default = np.asarray(cfg["default_joint_pos"], dtype=np.float64)
        self.kp = np.asarray(cfg["kp"], dtype=np.float64)
        self.kd = np.asarray(cfg["kd"], dtype=np.float64)
        if any(value.shape != (18,) or not np.isfinite(value).all() for value in (self.default, self.kp, self.kd)):
            raise ValueError("Expected 18 finite default positions, kp and kd values")
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.target = self.default.copy()
        delay = int(cfg["delay_steps"])
        if not 0 <= delay <= 4:
            raise ValueError("delay_steps must be in [0,4]")
        self.delay = deque([self.default.copy() for _ in range(delay + 1)], maxlen=delay + 1)
        self.model.dof_damping[self.qvel_ids] = 0.0  # damping is already in the explicit PD law
        self.model.dof_armature[self.qvel_ids] = 0.01
        self.model.dof_frictionloss[self.qvel_ids] = 0.01
        if cfg.get("disable_self_collisions", True):
            for geom in range(self.model.ngeom):
                if not (self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]):
                    continue
                robot = self.model.geom_bodyid[geom] != 0
                self.model.geom_contype[geom] = 1 if robot else 2
                self.model.geom_conaffinity[geom] = 2 if robot else 1
        self.floor = named_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, cfg["floor_geom"])
        if self.model.geom_type[self.floor] != mujoco.mjtGeom.mjGEOM_PLANE or self.model.geom_bodyid[self.floor] != 0:
            raise ValueError("This Flat bridge requires a static world plane")
        if not np.isfinite(payload_mass) or payload_mass < 0 or payload_mass > 10:
            raise ValueError("payload mass must be in [0,10] kg")
        if payload_mass:
            ee = named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "end_effector")
            self.model.body_mass[ee] += payload_mass
            self.model.body_inertia[ee] += 0.4 * payload_mass * 0.08**2
        mujoco.mj_setConst(self.model, self.data)
        free = np.where(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
        if len(free) != 1 or self.model.jnt_bodyid[free[0]] != self.base:
            raise ValueError("Expected one free root joint on the configured base")
        root = self.model.jnt_qposadr[free[0]]
        self.data.qpos[root:root + 7] = [0, 0, cfg["initial_base_height"], 1, 0, 0, 0]
        self.data.qpos[self.qpos_ids] = self.default
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        normal = self.data.geom_xmat[self.floor].reshape(3, 3)[:, 2]
        if not np.allclose(normal, [0, 0, 1], atol=1e-6):
            raise ValueError("The analytic height scan requires a horizontal plane")
        if int(cfg["height_scan_count"]) != 187:
            raise ValueError("This exported policy requires exactly 187 height scan values")
        self.floor_height = self.data.geom_xpos[self.floor, 2]

    def base_state(self):
        rotation = self.data.xmat[self.base].reshape(3, 3)
        twist_world = np.empty(6)
        # mjOBJ_BODY selects the CoM; local=True would use inertial principal axes.
        # Rotate WORLD CoM velocity into the regular base-link axes explicitly.
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base, twist_world, 0)
        return rotation.T @ twist_world[3:], rotation.T @ twist_world[:3], rotation.T @ np.array([0., 0., -1.])

    def observation(self, command):
        linear, angular, gravity = self.base_state()
        height = np.clip(self.data.xpos[self.base, 2] - self.floor_height - self.cfg["height_scan_offset"], -1, 1)
        observation = np.concatenate((linear * 2.0, angular * 0.2, gravity, command,
                                      self.data.qpos[self.qpos_ids] - self.default,
                                      self.data.qvel[self.qvel_ids] * 0.05, self.previous_action,
                                      np.full(187, height))).astype(np.float32)
        if observation.shape != (247,) or not np.isfinite(observation).all():
            raise ValueError(f"Invalid policy observation: {observation.shape}")
        return observation

    def set_action(self, action):
        action = np.asarray(action)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError(f"Expected finite 12 leg actions, got {action.shape}")
        self.previous_action = np.clip(action, -self.cfg["action_clip"], self.cfg["action_clip"]).astype(np.float32)
        self.target = self.default.copy()
        self.target[:12] += self.cfg["action_scale"] * self.previous_action

    def physics_step(self):
        self.delay.append(self.target.copy())
        torque = self.kp * (self.delay[0] - self.data.qpos[self.qpos_ids]) - self.kd * self.data.qvel[self.qvel_ids]
        control = torque / self.gear
        limits = self.model.actuator_ctrlrange[self.actuator_ids]
        limited = self.model.actuator_ctrllimited[self.actuator_ids].astype(bool)
        control[limited] = np.clip(control[limited], limits[limited, 0], limits[limited, 1])
        self.data.ctrl[self.actuator_ids] = control
        mujoco.mj_step(self.model, self.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("paint_loco.yaml"))
    parser.add_argument("--policy", type=Path, required=True, help="exported/policy.pt, NOT model_9999.pt")
    parser.add_argument("--command", nargs=3, type=float, default=[0, 0, 0], metavar=("VX", "VY", "WZ"))
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation seconds")
    parser.add_argument("--payload-mass", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", type=Path, help="Optional MP4 output; EGL/OSMesa needed on a headless Linux server")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--physics-profile", choices=PROFILES, default="legacy", help="Explicit A/B; default preserves the validated model")
    parser.add_argument("--trace", action="store_true", help="Save control-step observations, actions and joint states as NPZ")
    parser.add_argument("--check-only", action="store_true", help="Load XML and policy; check actual 247 -> 12 I/O, no rollout")
    args = parser.parse_args()
    command = np.asarray(args.command)
    if not np.isfinite(command).all() or np.any(np.abs(command) > [1, 1, 1.5]):
        parser.error("command must be finite and within vx/vy +/-1, wz +/-1.5")
    if not np.isfinite(args.duration) or args.duration <= 0 or args.fps <= 0:
        parser.error("duration/fps must be positive")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["physics_profile"] = args.physics_profile
    bridge = PaintBridge(cfg, args.payload_mass)
    torch.set_num_threads(1)
    with args.policy.expanduser().open("rb") as stream:
        policy = torch.jit.load(stream, map_location="cpu").eval()
    def infer():
        bridge.last_observation = bridge.observation(command)
        with torch.inference_mode():
            output = policy(torch.from_numpy(bridge.last_observation).unsqueeze(0))
        if not isinstance(output, torch.Tensor) or output.shape != (1, 12):
            raise ValueError("Policy must accept (1,247) and return a (1,12) tensor")
        if not torch.isfinite(output).all():
            raise ValueError("Policy returned non-finite actions")
        return output[0].cpu().numpy()
    infer()
    if hasattr(policy, "reset"):
        policy.reset()
    print("[MuJoCo] XML loaded; policy verified: 247 observations -> 12 actions; name-based joint mapping.", flush=True)
    if args.check_only:
        return 0
    if args.headless:
        context = nullcontext(None)
    else:
        from mujoco import viewer as mj_viewer
        context = mj_viewer.launch_passive(bridge.model, bridge.data)
    renderer = writer = None
    reason = "completed"
    steps = 0
    errors = np.zeros(3)
    velocities = np.zeros(3)
    arm_error_sum = 0.0
    arm_max = 0.0
    report_path = args.report or (MUJOCO_ROOT.parent / "logs/mujoco" / f"paint_{datetime.now():%Y%m%d_%H%M%S_%f}.json")
    started = time.monotonic()
    trace = {key: [] for key in ("time_s", "observation", "action", "target", "joint_pos", "joint_vel", "previous_applied_torque")}
    try:
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(bridge.model, height=480, width=640)
            writer = imageio.get_writer(str(args.video), fps=args.fps)
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation, camera.azimuth = 2.5, -20, 135
        next_frame = 0.0
        decimation = int(cfg["control_decimation"])
        if decimation < 1:
            raise ValueError("control_decimation must be positive")
        physical_steps = int(round(args.duration / bridge.model.opt.timestep))
        with context as viewer:
            for physical_step in range(physical_steps):
                wall = time.monotonic()
                if viewer is not None and not viewer.is_running():
                    reason = "viewer_closed"
                    break
                if physical_step % decimation == 0:
                    bridge.set_action(infer())
                    if args.trace:
                        for key, value in {
                            "time_s": bridge.data.time, "observation": bridge.last_observation,
                            "action": bridge.previous_action, "target": bridge.target,
                            "joint_pos": bridge.data.qpos[bridge.qpos_ids], "joint_vel": bridge.data.qvel[bridge.qvel_ids],
                            "previous_applied_torque": bridge.data.actuator_force[bridge.actuator_ids] * bridge.gear,
                        }.items():
                            trace[key].append(np.array(value, copy=True))
                    linear, angular, gravity = bridge.base_state()
                    actual = np.array([linear[0], linear[1], angular[2]])
                    errors += (actual - command)**2
                    velocities += actual
                    arm_error = bridge.data.qpos[bridge.qpos_ids[12:]] - bridge.default[12:]
                    arm_error_sum += np.mean(arm_error**2)
                    arm_max = max(arm_max, float(np.max(np.abs(arm_error))))
                    steps += 1
                    if steps == 1 or steps % 50 == 0:
                        print(f"[MuJoCo] t={bridge.data.time:.2f}s cmd={command.round(3)} "
                              f"actual={actual.round(3)} height={bridge.data.xpos[bridge.base, 2]:.3f}", flush=True)
                    if gravity[2] > -np.cos(1.0) or bridge.data.xpos[bridge.base, 2] - bridge.floor_height < 0.12:
                        reason = "fallen"
                        break
                before = bridge.data.time
                bridge.physics_step()
                if not np.isfinite(bridge.data.qpos).all() or not np.isfinite(bridge.data.qvel).all() or bridge.data.time <= before:
                    reason = "invalid_physics"
                    break
                camera.lookat[:] = bridge.data.xpos[bridge.base]
                if viewer is not None:
                    viewer.cam.lookat[:] = camera.lookat
                    viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = camera.distance, camera.elevation, camera.azimuth
                    viewer.sync()
                if renderer is not None and bridge.data.time >= next_frame:
                    renderer.update_scene(bridge.data, camera=camera)
                    writer.append_data(renderer.render())
                    next_frame += 1.0 / args.fps
                if viewer is not None:
                    time.sleep(max(0.0, bridge.model.opt.timestep - (time.monotonic() - wall)))
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    report = {"reason": reason, "policy": str(args.policy.resolve()), "observations": 247, "actions": 12,
              "policy_sha256": hashlib.sha256(args.policy.expanduser().read_bytes()).hexdigest(),
              "execution": describe_model(bridge),
              "sim_time_s": float(bridge.data.time), "wall_time_s": time.monotonic() - started,
              "control_steps": steps, "command": command.tolist(), "payload_mass_kg": args.payload_mass,
              "mean_actual_vx_vy_wz": (velocities / max(steps, 1)).tolist(),
              "velocity_rmse_vx_vy_wz": np.sqrt(errors / max(steps, 1)).tolist(),
              "arm_joint_rmse_rad": float(np.sqrt(arm_error_sum / max(steps, 1))),
              "arm_joint_max_abs_error_rad": arm_max}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if args.trace:
        trace_path = report_path.with_suffix(".trace.npz")
        np.savez_compressed(trace_path, **{key: np.asarray(value) for key, value in trace.items()})
        report["trace"] = str(trace_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[MuJoCo] {json.dumps(report)}\n[MuJoCo] Report: {report_path}", flush=True)
    return 0 if reason in ("completed", "viewer_closed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
