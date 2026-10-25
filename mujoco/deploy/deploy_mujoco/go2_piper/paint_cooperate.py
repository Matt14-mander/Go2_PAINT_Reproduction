"""Shared-payload MuJoCo pipeline: 2--4 GO2-PIPERs, replaceable LL/HL policies.

Reference mode is a hand-written admittance/formation controller, not a learned
cooperation result. Teacher mode consumes the exact clean 147-D transport actor
layout. Both modes use real 247->12 LL policies and torque-controlled free robots.
"""

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
import yaml

from cooperative_scene import build_scene, initialize_scene
from cooperative_control import DesktopCommand, motion_profile, slew_command, velocity_target
from paint_loco import JOINT_NAMES, MUJOCO_ROOT, PaintBridge, named_id


class TorchActor:
    def __init__(self, path, inputs, outputs):
        self.path = Path(path).expanduser().resolve()
        with self.path.open("rb") as stream:
            self.module = torch.jit.load(stream, map_location="cpu").eval()
        self.inputs, self.outputs = inputs, outputs
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self(np.zeros(inputs, np.float32))
        if hasattr(self.module, "reset"):
            self.module.reset()

    def __call__(self, observation):
        observation = np.asarray(observation, np.float32)
        if observation.shape != (self.inputs,) or not np.isfinite(observation).all():
            raise ValueError(f"Expected finite {self.inputs}-D observation")
        with torch.inference_mode():
            output = self.module(torch.from_numpy(observation).unsqueeze(0))
        if not isinstance(output, torch.Tensor) or output.shape != (1, self.outputs) or not torch.isfinite(output).all():
            raise ValueError(f"{self.path}: expected finite (1,{self.outputs}) actor output")
        return output[0].cpu().numpy().copy()


class RobotBridge(PaintBridge):
    """Reuse the single-robot LL equations without stepping the shared world."""
    def __init__(self, model, data, cfg, index):
        self.model, self.data, self.cfg = model, data, cfg
        self.prefix = f"r{index}/"
        self.base = named_id(model, mujoco.mjtObj.mjOBJ_BODY, self.prefix + "base_link")
        self.ee = named_id(model, mujoco.mjtObj.mjOBJ_BODY, self.prefix + "end_effector")
        self.joint_ids = np.array([named_id(model, mujoco.mjtObj.mjOBJ_JOINT, self.prefix + name) for name in JOINT_NAMES])
        self.qpos_ids, self.qvel_ids = model.jnt_qposadr[self.joint_ids], model.jnt_dofadr[self.joint_ids]
        ids = []
        for joint in self.joint_ids:
            found = np.flatnonzero((model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT) & (model.actuator_trnid[:, 0] == joint))
            if len(found) != 1:
                raise ValueError(f"{self.prefix}: expected one motor per joint")
            ids.append(found[0])
        self.actuator_ids = np.asarray(ids)
        self.gear = model.actuator_gear[self.actuator_ids, 0].copy()
        if np.any(self.gear == 0):
            raise ValueError("Zero motor gear")
        self.default = np.asarray(cfg["default_joint_pos"], float)
        self.kp, self.kd = np.asarray(cfg["kp"], float).copy(), np.asarray(cfg["kd"], float).copy()
        self.kp[12:], self.kd[12:] = 50., 4.
        self.target = self.default.copy()
        self.previous_action = np.zeros(12, np.float32)
        self.history = np.zeros((8, 18), np.float32)
        self.command = np.zeros(3)
        self.floor_height = 0.
        self.handle = named_id(model, mujoco.mjtObj.mjOBJ_SITE, f"payload/handle_{index}")
        self.eq = named_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{index}")
        model.dof_damping[self.qvel_ids] = 0
        model.dof_armature[self.qvel_ids] = .01
        model.dof_frictionloss[self.qvel_ids] = .01

    def push_history(self):
        frame = np.concatenate((self.data.qpos[self.qpos_ids[12:]] - self.default[12:],
                                self.data.qvel[self.qvel_ids[12:]], self.target[12:]))
        self.history[:-1] = self.history[1:].copy()
        self.history[-1] = frame

    def teacher_observation(self, normalized_wrench):
        return np.concatenate((self.history.ravel(), normalized_wrench)).astype(np.float32)

    def set_transport_action(self, leg_action, arm_action):
        self.set_action(leg_action)
        limits = self.model.jnt_range[self.joint_ids[12:]]
        # Match Isaac's soft_joint_pos_limit_factor=0.9 about each limit midpoint.
        midpoint, radius = limits.mean(1), (limits[:, 1] - limits[:, 0]) * .45
        self.target[12:] = np.clip(self.default[12:] + .5 * arm_action, midpoint - radius, midpoint + radius)

    def write_torques(self):
        torque = self.kp * (self.target - self.data.qpos[self.qpos_ids]) - self.kd * self.data.qvel[self.qvel_ids]
        control = torque / self.gear
        ids = self.actuator_ids
        limits = self.model.actuator_ctrlrange[ids]
        limited = self.model.actuator_ctrllimited[ids].astype(bool)
        control[limited] = np.clip(control[limited], limits[limited, 0], limits[limited, 1])
        self.data.ctrl[ids] = control


def grasp_reaction(model, data, eq_id, payload):
    """Payload->EE wrench at the EE origin, WORLD [Fx,Fy,Fz,Tx,Ty,Tz].

    Isolate this weld's generalized force on the free payload, then invert the
    payload spatial Jacobian at the EE point. No joint/sensor array offsets are
    assumed. This is a constraint reaction, NOT the training external wrench.
    """
    rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == eq_id)
    if np.count_nonzero(rows) != 6:
        raise ValueError("Expected six active rows per grasp weld")
    jacobian = data.efc_J.reshape(data.nefc, model.nv)
    generalized = jacobian[rows].T @ data.efc_force[rows]
    joint = model.body_jntadr[payload]
    dofs = np.arange(model.jnt_dofadr[joint], model.jnt_dofadr[joint] + 6)
    ee = model.eq_obj1id[eq_id]
    linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, linear, angular, data.xpos[ee], payload)
    spatial = np.vstack((linear[:, dofs], angular[:, dofs]))
    return -np.linalg.solve(spatial.T, generalized[dofs])


class CooperativeDemo:
    def __init__(self, cfg, count=2, mass=.5, spacing=1.2, self_collisions=False):
        xml, assets, self.offsets = build_scene(count, spacing, mass, cfg["simulation_dt"])
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        self.payload = initialize_scene(self.model, self.data, cfg["default_joint_pos"], self.offsets)
        self.robots = [RobotBridge(self.model, self.data, cfg, i) for i in range(count)]
        if not self_collisions:
            # Preserve cross-robot/payload/ground contacts; mask only each robot's
            # own self collision, matching the validated LL MuJoCo baseline.
            payload_bit = 1 << (count + 1)
            all_bits = (1 << (count + 2)) - 1
            for geom in range(self.model.ngeom):
                if not (self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]):
                    continue
                body = self.model.geom_bodyid[geom]
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body) or ""
                bit = payload_bit if body == self.payload else 1
                if name.startswith("r") and "/" in name:
                    bit = 1 << (int(name.split("/")[0][1:]) + 1)
                self.model.geom_contype[geom] = bit
                self.model.geom_conaffinity[geom] = all_bits ^ bit
        mujoco.mj_forward(self.model, self.data)

    def normalized_wrench(self, robot, external_world, source):
        wrench = external_world / len(self.robots) if source == "external-share" else grasp_reaction(self.model, self.data, robot.eq, self.payload)
        rotation = self.data.xmat[robot.base].reshape(3, 3)
        force, torque = rotation.T @ wrench[:3], rotation.T @ wrench[3:]
        return np.clip([force[0] / 40., force[1] / 40., torque[2] / 10.], -1, 1).astype(np.float32)

    def reference_action(self, index, external_world, formation_gain, max_speed, max_yaw,
                         world_command=None, heading_target=None):
        robot = self.robots[index]
        # Whole-group admittance to the applied horizontal force. This controller
        # has privileged force and formation state; it is not a trained teacher.
        world = external_world[:3] / (40. * len(self.robots))
        world[2] = 0
        yaw_command = external_world[5] / (10. * len(self.robots))
        if world_command is not None:
            world = np.r_[np.asarray(world_command)[:2], 0.]
            yaw_command = world_command[2]
        centers = np.array([self.data.xpos[r.base] for r in self.robots])
        payload_rotation = self.data.xmat[self.payload].reshape(3, 3)
        desired = centers.mean(0) + payload_rotation @ [0, self.offsets[index], 0]
        world[:2] += formation_gain * (desired - centers[index])[:2]
        rotation = self.data.xmat[robot.base].reshape(3, 3)
        command = rotation.T @ world
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        payload_yaw = np.arctan2(payload_rotation[1, 0], payload_rotation[0, 0])
        desired_heading = payload_yaw if heading_target is None else heading_target
        yaw_error = np.arctan2(np.sin(desired_heading - yaw), np.cos(desired_heading - yaw))
        command[2] = np.clip(yaw_command + .5 * yaw_error, -max_yaw, max_yaw)
        command[:2] = np.clip(command[:2], -max_speed, max_speed)
        return np.r_[command, np.zeros(6)]

    def status(self):
        values = []
        for robot in self.robots:
            linear, angular, gravity = robot.base_state()
            height = self.data.xpos[robot.base, 2]
            tilt = float(np.arccos(np.clip(-gravity[2], -1, 1)))
            error = float(np.linalg.norm(self.data.site_xpos[robot.handle] - self.data.xpos[robot.ee]))
            values.append({"height_m": float(height), "tilt_deg": float(np.rad2deg(tilt)), "grasp_error_m": error,
                           "actual_vx_vy_wz": [float(linear[0]), float(linear[1]), float(angular[2])]})
        return values


def policy_paths(paths, count, label):
    if len(paths) == 1:
        return paths * count
    if len(paths) != count:
        raise ValueError(f"{label}: pass one shared policy or one path per robot")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("paint_loco.yaml"))
    parser.add_argument("--ll-policy", type=Path, nargs="+", required=True)
    parser.add_argument("--hl-policy", type=Path, nargs="+", default=[])
    parser.add_argument("--controller", choices=("reference", "teacher"), default="reference")
    parser.add_argument("--drive", choices=("wrench", "velocity"), default="wrench")
    parser.add_argument("--command-world", nargs=3, type=float, default=[.3, 0., 0.], metavar=("VX", "VY", "WZ"))
    parser.add_argument("--travel-distance", type=float, help="Optional straight-line goal distance in meters; holds near goal")
    parser.add_argument("--keyboard", action="store_true", help="Reference velocity mode: W/S A/D Q/E, X stop, R resume, P pause")
    parser.add_argument("--settle-seconds", type=float, default=2.)
    parser.add_argument("--ramp-seconds", type=float, default=3.)
    parser.add_argument("--wrench-source", choices=("external-share", "grasp"), default="external-share")
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--spacing", type=float, default=1.2)
    parser.add_argument("--payload-mass", type=float, default=.5)
    parser.add_argument("--force-world", nargs=3, type=float, metavar=("FX", "FY", "TZ"),
                        help="Defaults to [8,0,0] in wrench mode; [0,0,0] in velocity mode")
    parser.add_argument("--max-speed", type=float, default=.35)
    parser.add_argument("--max-yaw-rate", type=float, default=.4)
    parser.add_argument("--formation-gain", type=float, default=.4)
    parser.add_argument("--self-collisions", action="store_true")
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-azimuth", type=float, default=0.)
    parser.add_argument("--viewer-fps", type=int, default=60)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.force_world is None:
        args.force_world = [8., 0., 0.] if args.drive == "wrench" else [0., 0., 0.]
    numbers = [args.spacing, args.payload_mass, args.duration, args.max_speed, args.max_yaw_rate,
               args.formation_gain, args.camera_azimuth, args.settle_seconds, args.ramp_seconds,
               *args.force_world, *args.command_world]
    if args.travel_distance is not None:
        numbers.append(args.travel_distance)
    if not np.isfinite(numbers).all():
        parser.error("All numeric parameters must be finite")
    if args.duration <= 0 or args.fps < 1 or not 0 < args.max_speed <= 1 or not 0 < args.max_yaw_rate <= 1.5 or args.formation_gain < 0:
        parser.error("Expected duration/fps > 0, speed in (0,1], yaw rate in (0,1.5], formation gain >= 0")
    if args.controller == "teacher" and not args.hl_policy:
        parser.error("teacher mode requires exported 147->9 --hl-policy")
    if args.controller == "reference" and args.hl_policy:
        parser.error("Use --controller teacher to enable --hl-policy")
    if args.settle_seconds < 0 or args.ramp_seconds <= 0 or args.viewer_fps < 1:
        parser.error("Expected settle >= 0, ramp > 0, viewer fps > 0")
    if args.drive == "velocity" and args.controller != "reference":
        parser.error("Velocity drive is a reference-controller demo; Teacher uses wrench drive")
    if args.keyboard and (args.headless or args.drive != "velocity"):
        parser.error("Keyboard requires desktop reference velocity mode")
    if args.travel_distance is not None and (args.travel_distance <= 0 or args.drive != "velocity" or args.keyboard
                                              or np.linalg.norm(args.command_world[:2]) == 0 or args.command_world[2] != 0):
        parser.error("Distance goal requires velocity drive, nonzero translation, zero yaw, no keyboard")
    if np.any(np.abs(args.command_world) > [args.max_speed, args.max_speed, args.max_yaw_rate]) and args.drive == "velocity":
        parser.error("World command exceeds max-speed/max-yaw-rate")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if int(cfg["control_decimation"]) != 4 or float(cfg["simulation_dt"]) != .005:
        parser.error("Current deployment contract is 200Hz physics / 50Hz control")
    if int(cfg["delay_steps"]) != 0 or float(cfg["initial_base_height"]) != .35:
        parser.error("Cooperative baseline currently requires zero actuator delay and .35m initial root height")
    demo = CooperativeDemo(cfg, args.num_robots, args.payload_mass, args.spacing, args.self_collisions)
    torch.set_num_threads(1)
    lows = [TorchActor(p, 247, 12) for p in policy_paths(args.ll_policy, args.num_robots, "LL")]
    highs = [TorchActor(p, 147, 9) for p in policy_paths(args.hl_policy, args.num_robots, "HL")] if args.hl_policy else []
    print(f"[COOP] {args.num_robots} robots, one FREE shared payload, {demo.model.neq} grasp welds; "
          f"controller={args.controller}; LL=247->12; HL=147->9; wrench={args.wrench_source}", flush=True)
    if args.check_only:
        print(f"[COOP] Initial status: {json.dumps(demo.status())}", flush=True)
        return 0
    model, data = demo.model, demo.data
    desktop = DesktopCommand(args.command_world, args.max_speed, args.max_yaw_rate)
    if args.headless:
        context = nullcontext(None)
    else:
        from mujoco import viewer
        context = viewer.launch_passive(model, data, key_callback=desktop.on_key if args.keyboard else None)
    if args.keyboard:
        print("[COOP] Click viewer: W/S forward/back, A/D left/right, Q/E yaw, X stop, R resume, P pause", flush=True)
    print(f"[COOP] drive={args.drive}; force_world={args.force_world}; world_command={args.command_world}; "
          f"distance_goal={args.travel_distance}", flush=True)
    path = args.report or MUJOCO_ROOT.parent / f"logs/mujoco_coop/coop_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = path.with_suffix(".csv")
    renderer = writer = None
    reason = "completed"
    started, next_frame, controls = time.monotonic(), 0., 0
    peak_tilt = np.zeros(args.num_robots)
    peak_grasp = np.zeros(args.num_robots)
    clipped = np.zeros(args.num_robots, int)
    initial_payload = data.xpos[demo.payload].copy()
    external = np.zeros(6)
    group_command = np.zeros(3)
    heading_target = 0.
    next_gui = 0.
    final_status = demo.status()
    warnings_before = data.warning.number.copy()
    try:
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(model, height=720, width=960)
            writer = imageio.get_writer(str(args.video), fps=args.fps)
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation, camera.azimuth = max(3., args.num_robots * 1.3), -25, args.camera_azimuth
        with trace_path.open("w", newline="", encoding="utf-8") as stream, context as display:
            if display is not None:
                with display.lock():
                    display.cam.distance, display.cam.azimuth, display.cam.elevation = camera.distance, camera.azimuth, camera.elevation
            trace = csv.writer(stream)
            trace.writerow(["time_s", "robot", "reference_cx", "reference_cy", "reference_cz", "raw_vx", "raw_vy", "raw_wz",
                            "command_vx", "command_vy", "command_wz", "actual_vx", "actual_vy", "actual_wz", "tilt_deg", "grasp_error_m"])
            step = 0
            while step < int(round(args.duration / model.opt.timestep)):
                wall = time.monotonic()
                if display is not None and not display.is_running():
                    reason = "viewer_closed"
                    break
                requested, paused = desktop.snapshot()
                if args.keyboard and paused:
                    display.sync()
                    time.sleep(.01)
                    continue
                external[:] = 0
                external[[0, 1, 5]] = np.asarray(args.force_world) * motion_profile(
                    data.time, args.duration, args.settle_seconds, args.ramp_seconds)
                data.xfrc_applied[:] = 0
                data.xfrc_applied[demo.payload] = external
                if step % 4 == 0:
                    mujoco.mj_forward(model, data)
                    final_status = demo.status()
                    for index, robot in enumerate(demo.robots):
                        state = final_status[index]
                        peak_tilt[index] = max(peak_tilt[index], state["tilt_deg"])
                        peak_grasp[index] = max(peak_grasp[index], state["grasp_error_m"])
                        if state["height_m"] < .12 or state["tilt_deg"] > np.rad2deg(1.):
                            reason = f"robot_{index}_fallen"
                        if state["grasp_error_m"] > .08:
                            reason = f"robot_{index}_grasp_error"
                    if reason != "completed":
                        break
                    if args.drive == "velocity":
                        target = velocity_target(requested, data.time, args.duration, data.xpos[demo.payload], initial_payload,
                                                 args.settle_seconds, args.ramp_seconds, args.travel_distance)
                        group_command = slew_command(group_command, target, 4 * model.opt.timestep)
                        heading_target += group_command[2] * 4 * model.opt.timestep
                    for index, robot in enumerate(demo.robots):
                        if controls:
                            robot.push_history()
                        reference = demo.normalized_wrench(robot, external, args.wrench_source)
                        raw = highs[index](robot.teacher_observation(reference)) if highs else demo.reference_action(
                            index, external, args.formation_gain, args.max_speed, args.max_yaw_rate,
                            group_command if args.drive == "velocity" else None,
                            heading_target if args.drive == "velocity" else None)
                        limits = np.array([args.max_speed, args.max_speed, args.max_yaw_rate])
                        clipped[index] += int(np.any(np.abs(raw[:3]) > limits + 1e-6))
                        robot.command = np.clip(raw[:3], -limits, limits)
                        robot.command[np.abs(robot.command) < .05] = 0
                        leg = lows[index](robot.observation(robot.command))
                        robot.set_transport_action(leg, raw[3:])
                        state = final_status[index]
                        trace.writerow([float(data.time), index, *reference, *raw[:3], *robot.command,
                                        *state["actual_vx_vy_wz"], state["tilt_deg"], state["grasp_error_m"]])
                    controls += 1
                    if controls == 1 or controls % 50 == 0:
                        displacement = data.xpos[demo.payload] - initial_payload
                        mean_actual = np.mean([s["actual_vx_vy_wz"] for s in final_status], axis=0)
                        print(f"[COOP] t={data.time:.2f}s payload={data.xpos[demo.payload].round(3)} "
                              f"travel_xy={np.linalg.norm(displacement[:2]):.2f}m world_cmd={group_command.round(2)} "
                              f"actual_body={mean_actual.round(2)} tilt={peak_tilt.round(1)} grasp={peak_grasp.round(4)}", flush=True)
                for robot in demo.robots:
                    robot.write_torques()
                before = data.time
                mujoco.mj_step(model, data)
                step += 1
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.time <= before or np.any(data.warning.number > warnings_before):
                    reason = "invalid_physics_or_solver_warning"
                    break
                camera.lookat[:] = data.xpos[demo.payload]
                if renderer is not None and data.time >= next_frame:
                    renderer.update_scene(data, camera=camera)
                    writer.append_data(renderer.render())
                    next_frame += 1 / args.fps
                if display is not None:
                    if data.time >= next_gui:
                        with display.lock():
                            display.cam.lookat[:] = camera.lookat
                        display.sync()
                        next_gui = data.time + 1 / args.viewer_fps
                    time.sleep(max(0, model.opt.timestep - (time.monotonic() - wall)))
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    mujoco.mj_forward(model, data)
    report = {"reason": reason, "controller": args.controller, "robots": args.num_robots, "payload_mass_kg": args.payload_mass,
              "drive": args.drive, "command_world_vx_vy_wz": args.command_world,
              "final_requested_world_command": desktop.snapshot()[0].tolist(),
              "travel_distance_goal_m": args.travel_distance,
              "wrench_source": args.wrench_source, "force_world_Fx_Fy_Tz": args.force_world,
              "sim_time_s": float(data.time), "wall_time_s": time.monotonic() - started, "control_steps": controls,
              "payload_displacement_world_m": (data.xpos[demo.payload] - initial_payload).tolist(),
              "max_tilt_deg": peak_tilt.tolist(), "max_grasp_error_m": peak_grasp.tolist(),
              "command_safety_clipped_fraction": (clipped / max(controls, 1)).tolist(),
              "final_status": demo.status(), "self_collisions": args.self_collisions,
              "ll_policies": [{"path": str(p.path), "sha256": p.sha256} for p in lows],
              "hl_policies": [{"path": str(p.path), "sha256": p.sha256} for p in highs], "trace": str(trace_path)}
    if args.travel_distance is not None:
        direction = np.asarray(args.command_world[:2]) / np.linalg.norm(args.command_world[:2])
        goal = initial_payload[:2] + direction * args.travel_distance
        report["goal_position_world_xy_m"] = goal.tolist()
        report["goal_error_m"] = float(np.linalg.norm(data.xpos[demo.payload, :2] - goal))
        report["goal_reached"] = report["goal_error_m"] < .15
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[COOP] {json.dumps(report)}\n[COOP] Report: {path}", flush=True)
    return 0 if reason in ("completed", "viewer_closed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
