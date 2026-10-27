"""Shared-payload MuJoCo pipeline: 2--4 GO2-PIPERs, replaceable LL/HL policies.

Reference mode is a hand-written admittance/formation controller, not a learned
cooperation result. Teacher mode consumes the exact clean 147-D transport actor
layout. Both modes use real 247->12 LL policies and torque-controlled free robots.
Leader-follow drive uses a velocity-controlled LL leader and a local FT
admittance follower, mechanically coupled through a free tow bar with no applied
guidance wrench. Sofa/table options place 1--3 followers behind shared furniture
with ideal grip extensions. Teacher policies are not used in this baseline.
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
from cooperative_control import DesktopCommand, WrenchFollower, leader_action, motion_profile, slew_command, velocity_target
from cooperative_visuals import draw_guidance, draw_leader_follow
from grasp_audit import GraspAudit, normalized_input, payload_balance
from transport_contract import load_stage2_contract, arm_target, force_profile
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
    def __init__(self, model, data, cfg, index, layout="side-by-side"):
        self.model, self.data, self.cfg = model, data, cfg
        self.prefix = f"r{index}/"
        self.base = named_id(model, mujoco.mjtObj.mjOBJ_BODY, self.prefix + "base_link")
        self.ee = named_id(model, mujoco.mjtObj.mjOBJ_BODY, self.prefix + "end_effector")
        self.attachment = (named_id(model, mujoco.mjtObj.mjOBJ_BODY, self.prefix + "tow_mount")
                           if layout in ("leader-follow", "furniture") and index == 0 else self.ee)
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

    def set_transport_action(self, leg_action, arm_action, contract=None, settling=False):
        previous = self.target[12:].copy()
        self.set_action(leg_action)
        limits = self.model.jnt_range[self.joint_ids[12:]]
        # Match Isaac's soft_joint_pos_limit_factor=0.9 about each limit midpoint.
        midpoint, radius = limits.mean(1), (limits[:, 1] - limits[:, 0]) * .45
        self.target[12:] = np.clip(self.default[12:] + .5 * arm_action, midpoint - radius, midpoint + radius)
        if contract is not None:
            soft_limits = np.column_stack((midpoint - radius, midpoint + radius))
            self.target[12:] = (self.default[12:] if settling else
                               arm_target(arm_action, self.default[12:], soft_limits, previous, contract))

    def write_torques(self):
        torque = self.kp * (self.target - self.data.qpos[self.qpos_ids]) - self.kd * self.data.qvel[self.qvel_ids]
        control = torque / self.gear
        ids = self.actuator_ids
        limits = self.model.actuator_ctrlrange[ids]
        limited = self.model.actuator_ctrllimited[ids].astype(bool)
        control[limited] = np.clip(control[limited], limits[limited, 0], limits[limited, 1])
        self.data.ctrl[ids] = control


def grasp_reaction(model, data, eq_id, payload):
    """Payload->robot wrench at the attachment origin, WORLD [F,T].

    Isolate the attachment's generalized force on the free payload, then invert
    the payload spatial Jacobian at the attachment point. No joint/sensor offsets are
    assumed. Connect anchors must be at body1's origin (the leader tow mount).
    This is a constraint reaction, NOT the training external wrench.
    """
    rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == eq_id)
    expected = 3 if model.eq_type[eq_id] == mujoco.mjtEq.mjEQ_CONNECT else 6
    if np.count_nonzero(rows) != expected:
        raise ValueError(f"Expected {expected} active rows per attachment constraint")
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
    def __init__(self, cfg, count=2, mass=.5, spacing=1.2, self_collisions=False, layout="side-by-side", payload_kind=None):
        self.layout = layout
        self.payload_kind = payload_kind or ("sofa" if layout == "furniture" else "tow-bar" if layout == "leader-follow" else "box")
        xml, assets, self.offsets = build_scene(count, spacing, mass, cfg["simulation_dt"], layout,
                                              cfg["default_joint_pos"], self.payload_kind)
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        self.payload = initialize_scene(self.model, self.data, cfg["default_joint_pos"], self.offsets, layout)
        self.robots = [RobotBridge(self.model, self.data, cfg, i, layout) for i in range(count)]
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

    def observed_world_wrench(self, robot, external_world, source):
        return external_world / len(self.robots) if source == "external-share" else grasp_reaction(self.model, self.data, robot.eq, self.payload)

    def normalized_wrench(self, robot, external_world, source):
        wrench = self.observed_world_wrench(robot, external_world, source)
        rotation = self.data.xmat[robot.base].reshape(3, 3)
        return normalized_input(wrench, rotation)[1].astype(np.float32)

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
            error = float(np.linalg.norm(self.data.site_xpos[robot.handle] - self.data.xpos[robot.attachment]))
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
    parser.add_argument("--controller", choices=("reference", "admittance", "teacher"),
                        help="Default: admittance for leader-follow; reference otherwise")
    parser.add_argument("--admittance-gain", type=float, default=1.)
    parser.add_argument("--drive", choices=("wrench", "velocity", "leader-follow"), default="wrench")
    parser.add_argument("--command-world", nargs=3, type=float, metavar=("VX", "VY", "WZ"),
                        help="Default vx=.1 for leader-follow, .3 otherwise; ONLY leader receives this in leader-follow")
    parser.add_argument("--follower-hold", action="store_true", help="Leader-follow ablation: follower LL receives zero velocity")
    parser.add_argument("--stop-hold-seconds", type=float, default=3., help="Leader-follow stationary tail after leader ramp-down")
    parser.add_argument("--travel-distance", type=float, help="Optional straight-line goal distance in meters; holds near goal")
    parser.add_argument("--keyboard", action="store_true", help="Wrench drive: keyboard changes PAYLOAD FORCE; velocity drive: changes speed")
    parser.add_argument("--settle-seconds", type=float, default=2.)
    parser.add_argument("--ramp-seconds", type=float, default=3.)
    parser.add_argument("--wrench-source", choices=("external-share", "grasp"),
                        help="Default: grasp for leader-follow; external-share otherwise")
    parser.add_argument("--audit-grasp", action="store_true", help="Log both grasp and external-share inputs without changing the chosen controller")
    parser.add_argument("--require-stage2-contract", action="store_true", help="Fail if matching Stage2 policy manifest is missing")
    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--spacing", type=float, default=1.2)
    parser.add_argument("--payload-mass", type=float, default=.5)
    parser.add_argument("--payload", choices=("box", "tow-bar", "sofa", "table"),
                        help="Default: tow-bar for leader-follow, box otherwise. Sofa/table: 1 leader + 1--3 followers")
    parser.add_argument("--force-world", nargs=3, type=float, metavar=("FX", "FY", "TZ"),
                        help="Defaults to [8,0,0] in wrench mode; [0,0,0] in velocity mode")
    parser.add_argument("--max-speed", type=float, default=.35)
    parser.add_argument("--max-yaw-rate", type=float, default=.4)
    parser.add_argument("--formation-gain", type=float, default=.4)
    parser.add_argument("--self-collisions", action="store_true")
    parser.add_argument("--freeze-arm", action="store_true", help="Required execution contract for Stage1 Teacher: hold nominal arm targets")
    parser.add_argument("--duration", type=float, default=20.)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-azimuth", type=float, help="Default: 110 degrees for leader-follow, 0 otherwise")
    parser.add_argument("--viewer-fps", type=int, default=60)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    leader_follow = args.drive == "leader-follow"
    velocity_drive = args.drive in ("velocity", "leader-follow")
    args.payload = args.payload or ("tow-bar" if leader_follow else "box")
    furniture = args.payload in ("sofa", "table")
    layout = "furniture" if furniture else "leader-follow" if leader_follow else "side-by-side"
    if (furniture and not leader_follow) or (not leader_follow and args.payload != "box") or (leader_follow and args.payload == "box"):
        parser.error("Use sofa/table/tow-bar with --drive leader-follow; box with wrench/velocity")
    args.camera_azimuth = args.camera_azimuth if args.camera_azimuth is not None else (110. if leader_follow else 0.)
    args.controller = args.controller or ("admittance" if leader_follow else "reference")
    args.wrench_source = args.wrench_source or ("grasp" if leader_follow else "external-share")
    args.command_world = args.command_world if args.command_world is not None else ([.1, 0., 0.] if leader_follow else [.3, 0., 0.])
    if args.force_world is None:
        args.force_world = [8., 0., 0.] if args.drive == "wrench" else [0., 0., 0.]
    if leader_follow and ((not furniture and args.num_robots != 2) or not 2 <= args.num_robots <= 4
                          or args.controller != "admittance" or args.wrench_source != "grasp"
                          or args.hl_policy or np.any(np.asarray(args.force_world) != 0)):
        parser.error("Leader-follow requires 2 robots for tow-bar or 2--4 for furniture, admittance, grasp reactions, no HL policy and zero external force")
    if args.follower_hold and not leader_follow:
        parser.error("--follower-hold requires --drive leader-follow")
    contract = load_stage2_contract(args.hl_policy)
    if args.require_stage2_contract and contract is None:
        parser.error("Copy the Stage2 exported manifest.json alongside policy.pt")
    if contract is not None:
        if args.freeze_arm or args.controller != "teacher":
            parser.error("Stage2 manifest requires teacher mode with the arm enabled; omit --freeze-arm")
        if not np.allclose(contract["command_limits"], [args.max_speed, args.max_speed, args.max_yaw_rate]):
            parser.error("Command limits must match the Stage2 manifest")
        args.settle_seconds = contract["settle_seconds"]
        args.ramp_seconds = contract["force_ramp_seconds"]
        force_profile(0., args.duration, contract)
        print(f"[COOP] Stage2 execution contract loaded: {json.dumps(contract)}", flush=True)
    numbers = [args.spacing, args.payload_mass, args.duration, args.max_speed, args.max_yaw_rate,
               args.formation_gain, args.camera_azimuth, args.settle_seconds, args.ramp_seconds, args.admittance_gain, args.stop_hold_seconds,
               *args.force_world, *args.command_world]
    if args.travel_distance is not None:
        numbers.append(args.travel_distance)
    if not np.isfinite(numbers).all():
        parser.error("All numeric parameters must be finite")
    if args.duration <= 0 or args.fps < 1 or not 0 < args.max_speed <= 1 or not 0 < args.max_yaw_rate <= 1.5 or args.formation_gain < 0:
        parser.error("Expected duration/fps > 0, speed in (0,1], yaw rate in (0,1.5], formation gain >= 0")
    if args.controller == "teacher" and not args.hl_policy:
        parser.error("teacher mode requires exported 147->9 --hl-policy")
    if args.controller != "teacher" and args.hl_policy:
        parser.error("Use --controller teacher to enable --hl-policy")
    if args.settle_seconds < 0 or args.ramp_seconds <= 0 or args.viewer_fps < 1:
        parser.error("Expected settle >= 0, ramp > 0, viewer fps > 0")
    if leader_follow and (args.stop_hold_seconds < 0 or args.duration < args.settle_seconds + 2 * args.ramp_seconds + args.stop_hold_seconds):
        parser.error("Leader-follow duration must include settle + two ramps + stop-hold; stop-hold must be >= 0")
    if args.drive == "velocity" and args.controller != "reference":
        parser.error("Velocity drive is a reference-controller demo; Teacher uses wrench drive")
    if args.keyboard and args.headless:
        parser.error("Keyboard requires a desktop viewer")
    if args.admittance_gain <= 0 or (args.controller == "admittance" and args.settle_seconds < 1):
        parser.error("Admittance gain must be positive; reaction calibration needs settle >= 1 second")
    if args.travel_distance is not None and (args.travel_distance <= 0 or not velocity_drive or args.keyboard
                                              or np.linalg.norm(args.command_world[:2]) == 0 or args.command_world[2] != 0):
        parser.error("Distance goal requires velocity drive, nonzero translation, zero yaw, no keyboard")
    if np.any(np.abs(args.command_world) > [args.max_speed, args.max_speed, args.max_yaw_rate]) and velocity_drive:
        parser.error("World command exceeds max-speed/max-yaw-rate")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if int(cfg["control_decimation"]) != 4 or float(cfg["simulation_dt"]) != .005:
        parser.error("Current deployment contract is 200Hz physics / 50Hz control")
    if int(cfg["delay_steps"]) != 0 or float(cfg["initial_base_height"]) != .35:
        parser.error("Cooperative baseline currently requires zero actuator delay and .35m initial root height")
    demo = CooperativeDemo(cfg, args.num_robots, args.payload_mass, args.spacing, args.self_collisions, layout, args.payload)
    torch.set_num_threads(1)
    lows = [TorchActor(p, 247, 12) for p in policy_paths(args.ll_policy, args.num_robots, "LL")]
    highs = [TorchActor(p, 147, 9) for p in policy_paths(args.hl_policy, args.num_robots, "HL")] if args.hl_policy else []
    print(f"[COOP] {args.num_robots} robots, layout={layout}, FREE {args.payload}, {demo.model.neq} attachment constraints; "
          f"controller={args.controller}; LL=247->12; HL=147->9; wrench={args.wrench_source}", flush=True)
    if args.check_only:
        print(f"[COOP] Initial status: {json.dumps(demo.status())}", flush=True)
        return 0
    model, data = demo.model, demo.data
    desktop = (DesktopCommand(args.command_world, args.max_speed, args.max_yaw_rate,
                              label="LEADER world velocity" if leader_follow else "world velocity") if velocity_drive
               else DesktopCommand(args.force_world, 80., 10., linear_step=2., yaw_step=.5, label="payload Fx/Fy/Tz [N,N,Nm]"))
    followers = [WrenchFollower(args.admittance_gain, command_limits=(
        [args.max_speed, args.max_speed, args.max_yaw_rate] if leader_follow else None)) for _ in demo.robots]
    if args.headless:
        context = nullcontext(None)
    else:
        from mujoco import viewer
        context = viewer.launch_passive(model, data, key_callback=desktop.on_key if args.keyboard else None)
    if args.keyboard:
        units = "payload Fx/Fy +/-2N, Tz +/-0.5Nm" if args.drive == "wrench" else "world velocity +/-0.05m/s, yaw +/-0.1rad/s"
        print(f"[COOP] Keyboard changes {units}: W/S X-axis, A/D Y-axis, Q/E yaw, X zero, R restore, P pause", flush=True)
    drive_description = (f"payload force Fx/Fy/Tz={args.force_world}; no prescribed transport velocity"
                         if args.drive == "wrench" else f"world velocity={args.command_world}; distance_goal={args.travel_distance}")
    print(f"[COOP] drive={args.drive}; {drive_description}", flush=True)
    if leader_follow:
        print(f"[COOP] r0=leader LL velocity; r1..r{args.num_robots - 1}=followers {'zero-command HOLD' if args.follower_hold else 'local FT admittance'}; "
              f"no applied external wrench; final stop hold={args.stop_hold_seconds}s", flush=True)
    path = args.report or MUJOCO_ROOT.parent / f"logs/mujoco_coop/coop_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = path.with_suffix(".csv")
    audit_path = path.with_suffix(".grasp.csv")
    audit = None
    renderer = writer = None
    reason = "completed"
    started, next_frame, controls = time.monotonic(), 0., 0
    peak_tilt = np.zeros(args.num_robots)
    peak_grasp = np.zeros(args.num_robots)
    clipped = np.zeros(args.num_robots, int)
    arm_offset_max = np.zeros(args.num_robots)
    arm_speed_max = np.zeros(args.num_robots)
    initial_payload = data.xpos[demo.payload].copy()
    initial_bases = np.array([data.xpos[r.base].copy() for r in demo.robots])
    command_origin = initial_bases[0].copy() if leader_follow else initial_payload.copy()
    peak_torque_fraction = np.zeros((args.num_robots, 2))  # legs, arms
    torque_saturated = np.zeros((args.num_robots, 2), int)
    peak_wrench = np.zeros((args.num_robots, 6))
    initial_separation = float(np.linalg.norm(initial_bases[0, :2] - initial_bases[1, :2]))
    separation_range = [initial_separation, initial_separation]
    follower_distance_range = np.repeat(np.linalg.norm(initial_bases[1:, :2] - initial_bases[0, :2], axis=1)[:, None], 2, axis=1)
    peak_balance = np.zeros(6)
    external = np.zeros(6)
    group_command = np.zeros(3)
    heading_target = 0.
    next_gui = 0.
    final_status = demo.status()
    warnings_before = data.warning.number.copy()
    display_wrenches = np.zeros((args.num_robots - 1, 6))

    def draw_overlay(scene):
        distance = np.linalg.norm((data.xpos[demo.payload] - initial_payload)[:2])
        if leader_follow:
            draw_leader_follow(scene, [data.xpos[r.base] for r in demo.robots],
                               [r.command for r in demo.robots], display_wrenches, distance, args.follower_hold)
        else:
            draw_guidance(scene, data.xpos[demo.payload], external, distance, args.controller)

    try:
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            renderer = mujoco.Renderer(model, height=720, width=960)
            writer = imageio.get_writer(str(args.video), fps=args.fps)
        camera = mujoco.MjvCamera()
        camera.distance = (max(4.5, args.spacing * max(1, args.num_robots - 2) + 3.) if furniture else
                           max(3.5, args.spacing + 2.) if leader_follow else max(3., args.num_robots * 1.3))
        camera.elevation, camera.azimuth = -25, args.camera_azimuth
        with trace_path.open("w", newline="", encoding="utf-8") as stream, \
                (audit_path.open("w", newline="", encoding="utf-8") if args.audit_grasp else nullcontext(None)) as audit_stream, \
                context as display:
            if audit_stream is not None:
                audit = GraspAudit(audit_stream, args.num_robots, args.settle_seconds)
            if display is not None:
                with display.lock():
                    display.cam.distance, display.cam.azimuth, display.cam.elevation = camera.distance, camera.azimuth, camera.elevation
            trace = csv.writer(stream)
            trace.writerow(["time_s", "robot", "reference_cx", "reference_cy", "reference_cz", "raw_vx", "raw_vy", "raw_wz",
                            "command_vx", "command_vy", "command_wz", "actual_vx", "actual_vy", "actual_wz", "tilt_deg", "grasp_error_m",
                            "payload_external_Fx", "payload_external_Fy", "payload_external_Tz",
                            "input_wrench_world_Fx", "input_wrench_world_Fy", "input_wrench_world_Fz",
                            "input_wrench_world_Tx", "input_wrench_world_Ty", "input_wrench_world_Tz",
                            *[f"arm_target_joint{i}_rad" for i in range(1, 7)],
                            "role", "base_world_x", "base_world_y", "base_world_z", "base_separation_m",
                            "distance_to_leader_m",
                            "leg_torque_limit_fraction", "arm_torque_limit_fraction"])
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
                force_request = requested if args.drive == "wrench" else np.asarray(args.force_world)
                profile = (force_profile(data.time, args.duration, contract) if contract is not None else
                           motion_profile(data.time, args.duration, args.settle_seconds, args.ramp_seconds))
                external[[0, 1, 5]] = force_request * profile
                data.xfrc_applied[:] = 0
                if not leader_follow:
                    data.xfrc_applied[demo.payload] = external
                if step % 4 == 0:
                    mujoco.mj_forward(model, data)
                    if audit is not None or leader_follow:
                        reactions = [grasp_reaction(model, data, r.eq, demo.payload) for r in demo.robots]
                        total, balance, extraction = payload_balance(model, data, demo.payload, demo.robots, reactions)
                        peak_balance = np.maximum(peak_balance, np.abs(balance))
                    if audit is not None:
                        for index, robot in enumerate(demo.robots):
                            audit.sample(float(data.time), index, reactions[index], data.xmat[robot.base].reshape(3, 3),
                                         external / args.num_robots, total, balance, extraction)
                    final_status = demo.status()
                    separation = float(np.linalg.norm(data.xpos[demo.robots[0].base, :2] - data.xpos[demo.robots[1].base, :2]))
                    separation_range = [min(separation_range[0], separation), max(separation_range[1], separation)]
                    distances = np.linalg.norm(np.array([data.xpos[r.base, :2] for r in demo.robots[1:]])
                                               - data.xpos[demo.robots[0].base, :2], axis=1)
                    follower_distance_range[:, 0] = np.minimum(follower_distance_range[:, 0], distances)
                    follower_distance_range[:, 1] = np.maximum(follower_distance_range[:, 1], distances)
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
                    if velocity_drive:
                        command_position = data.xpos[demo.robots[0].base] if leader_follow else data.xpos[demo.payload]
                        drive_end = args.duration - args.stop_hold_seconds if leader_follow else args.duration
                        target = velocity_target(requested, data.time, drive_end, command_position, command_origin,
                                                 args.settle_seconds, args.ramp_seconds, args.travel_distance)
                        group_command = slew_command(group_command, target, 4 * model.opt.timestep)
                        heading_target += group_command[2] * 4 * model.opt.timestep
                    for index, robot in enumerate(demo.robots):
                        if controls:
                            robot.push_history()
                        observed = demo.observed_world_wrench(robot, external, args.wrench_source)
                        reference = normalized_input(observed, data.xmat[robot.base].reshape(3, 3))[1].astype(np.float32)
                        if leader_follow and index > 0:
                            display_wrenches[index - 1] = observed
                        peak_wrench[index] = np.maximum(peak_wrench[index], np.abs(observed))
                        if leader_follow and index == 0:
                            raw = leader_action(group_command, data.xmat[robot.base].reshape(3, 3), heading_target)
                        elif leader_follow and args.follower_hold:
                            raw = np.zeros(9)
                        elif highs:
                            raw = highs[index](robot.teacher_observation(reference))
                        elif args.controller == "admittance":
                            calibrating = data.time < args.settle_seconds
                            # Ignore early landing transients; calibrate final stationary second.
                            bias_window = data.time >= args.settle_seconds - 1
                            velocity = followers[index].update(observed, data.xmat[robot.base].reshape(3, 3),
                                                                4 * model.opt.timestep, calibrating, bias_window)
                            raw = np.r_[velocity, np.zeros(6)]
                        else:
                            raw = demo.reference_action(index, external, args.formation_gain, args.max_speed, args.max_yaw_rate,
                                                        group_command if args.drive == "velocity" else None,
                                                        heading_target if args.drive == "velocity" else None)
                        limits = np.array([args.max_speed, args.max_speed, args.max_yaw_rate])
                        clipped[index] += int(np.any(np.abs(raw[:3]) > limits + 1e-6))
                        robot.command = np.clip(raw[:3], -limits, limits)
                        deadband = contract["command_deadband"] if contract is not None else .05
                        robot.command[np.abs(robot.command) < deadband] = 0
                        settling = contract is not None and data.time < contract["settle_seconds"]
                        if settling:
                            robot.command[:] = 0
                        leg = lows[index](robot.observation(robot.command))
                        previous_arm = robot.target[12:].copy()
                        robot.set_transport_action(leg, np.zeros(6) if args.freeze_arm else raw[3:], contract, settling)
                        arm_offset_max[index] = max(arm_offset_max[index], np.max(np.abs(robot.target[12:] - robot.default[12:])))
                        arm_speed_max[index] = max(arm_speed_max[index], np.max(np.abs(robot.target[12:] - previous_arm)) / .02)
                        state = final_status[index]
                        # Current commanded PD effort before actuator clipping. Log legs
                        # separately so saturated arm holding cannot hide leg overload.
                        demand = (robot.kp * (robot.target - data.qpos[robot.qpos_ids])
                                  - robot.kd * data.qvel[robot.qvel_ids]) / robot.gear
                        bounds = model.actuator_ctrlrange[robot.actuator_ids]
                        limited = model.actuator_ctrllimited[robot.actuator_ids].astype(bool)
                        directional_limit = np.where(demand >= 0, bounds[:, 1], -bounds[:, 0])
                        fraction = np.divide(np.abs(demand), directional_limit, out=np.zeros(18),
                                             where=limited & (directional_limit > 0))
                        effort = np.array([np.max(fraction[:12]), np.max(fraction[12:])])
                        peak_torque_fraction[index] = np.maximum(peak_torque_fraction[index], effort)
                        torque_saturated[index] += effort >= 1.
                        role = ("leader" if index == 0 else "follower") if leader_follow else "cooperator"
                        trace.writerow([float(data.time), index, *reference, *raw[:3], *robot.command,
                                        *state["actual_vx_vy_wz"], state["tilt_deg"], state["grasp_error_m"],
                                        *external[[0, 1, 5]], *observed, *robot.target[12:], role,
                                        *data.xpos[robot.base], separation, 0. if index == 0 else distances[index - 1], *effort])
                    controls += 1
                    if controls == 1 or controls % 50 == 0:
                        displacement = data.xpos[demo.payload] - initial_payload
                        mean_actual = np.mean([s["actual_vx_vy_wz"] for s in final_status], axis=0)
                        print(f"[COOP] t={data.time:.2f}s payload={data.xpos[demo.payload].round(3)} "
                              f"travel_xy={np.linalg.norm(displacement[:2]):.2f}m Fxy/Tz={external[[0,1,5]].round(1)} "
                              f"cmd_body={np.mean([r.command for r in demo.robots], axis=0).round(2)} "
                              f"actual_body={mean_actual.round(2)} tilt={peak_tilt.round(1)} grasp={peak_grasp.round(4)}", flush=True)
                        if leader_follow:
                            print(f"[LEADER-FOLLOW] leader_cmd={demo.robots[0].command.round(3)} "
                                  f"follower_cmd={[r.command.round(3).tolist() for r in demo.robots[1:]]} "
                                  f"actual={np.round([s['actual_vx_vy_wz'] for s in final_status], 3).tolist()} "
                                  f"follower_Fxy={display_wrenches[:, :2].round(2).tolist()}N separation={distances.round(3)}m", flush=True)
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
                    draw_overlay(renderer.scene)
                    writer.append_data(renderer.render())
                    next_frame += 1 / args.fps
                if display is not None:
                    if data.time >= next_gui:
                        with display.lock():
                            display.cam.lookat[:] = camera.lookat
                            display.user_scn.ngeom = 0
                            draw_overlay(display.user_scn)
                        display.sync()
                        next_gui = data.time + 1 / args.viewer_fps
                    time.sleep(max(0, model.opt.timestep - (time.monotonic() - wall)))
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
    mujoco.mj_forward(model, data)
    final_status = demo.status()
    for index, state in enumerate(final_status):
        peak_tilt[index] = max(peak_tilt[index], state["tilt_deg"])
        peak_grasp[index] = max(peak_grasp[index], state["grasp_error_m"])
        if reason == "completed":
            if state["height_m"] < .12 or state["tilt_deg"] > np.rad2deg(1.):
                reason = f"robot_{index}_fallen"
            elif state["grasp_error_m"] > .08:
                reason = f"robot_{index}_grasp_error"
    report = {"reason": reason, "controller": args.controller, "robots": args.num_robots, "payload_mass_kg": args.payload_mass,
              "drive": args.drive, "layout": layout,
              "payload_type": args.payload,
              "attachment_model": "ideal_virtual_handles" if furniture else "rear_ball_hitch" if leader_follow else "EE_welds",
              "command_world_vx_vy_wz": args.command_world if velocity_drive else None,
              "roles": ["leader"] + ["follower"] * (args.num_robots - 1) if leader_follow else ["cooperator"] * args.num_robots,
              "role_controllers": (["locomotion_velocity"] + ["hold" if args.follower_hold else "local_FT_admittance"] * (args.num_robots - 1)
                                   if leader_follow else [args.controller] * args.num_robots),
              "follower_hold": args.follower_hold,
              "stop_hold_seconds": args.stop_hold_seconds if leader_follow else None,
              "robot_displacement_world_m": (np.array([data.xpos[r.base] for r in demo.robots]) - initial_bases).tolist(),
              "base_separation_range_m": separation_range,
              "follower_distance_to_leader_range_m": follower_distance_range.tolist() if leader_follow else None,
              "max_abs_input_wrench_world": peak_wrench.tolist(),
              "max_abs_attachment_wrench_world": peak_wrench.tolist() if args.wrench_source == "grasp" else None,
              "max_abs_payload_balance_generalized": peak_balance.tolist() if leader_follow or audit is not None else None,
              "max_torque_limit_fraction_legs_arms": peak_torque_fraction.tolist(),
              "torque_saturated_control_fraction_legs_arms": (torque_saturated / max(controls, 1)).tolist(),
              "final_keyboard_request": desktop.snapshot()[0].tolist(),
              "keyboard_request_units": "N,N,Nm" if args.drive == "wrench" else "m/s,m/s,rad/s",
              "admittance_gain": args.admittance_gain,
              "follower_bias_world": [f.bias.tolist() for f in followers] if args.controller == "admittance" else None,
              "follower_bias_samples": [f.bias_samples for f in followers] if args.controller == "admittance" else None,
              "travel_distance_goal_m": args.travel_distance,
              "wrench_source": args.wrench_source, "force_world_Fx_Fy_Tz": args.force_world,
              "sim_time_s": float(data.time), "wall_time_s": time.monotonic() - started, "control_steps": controls,
              "payload_displacement_world_m": (data.xpos[demo.payload] - initial_payload).tolist(),
              "max_tilt_deg": peak_tilt.tolist(), "max_grasp_error_m": peak_grasp.tolist(),
              "command_safety_clipped_fraction": (clipped / max(controls, 1)).tolist(),
              "final_status": demo.status(), "self_collisions": args.self_collisions,
              "freeze_arm": args.freeze_arm or leader_follow,
              "transport_contract": contract,
              "max_arm_target_offset_rad": arm_offset_max.tolist(),
              "max_arm_target_speed_rad_s": arm_speed_max.tolist(),
              "ll_policies": [{"path": str(p.path), "sha256": p.sha256} for p in lows],
              "hl_policies": [{"path": str(p.path), "sha256": p.sha256} for p in highs], "trace": str(trace_path)}
    if args.travel_distance is not None:
        direction = np.asarray(args.command_world[:2]) / np.linalg.norm(args.command_world[:2])
        goal = command_origin[:2] + direction * args.travel_distance
        report["goal_position_world_xy_m"] = goal.tolist()
        goal_position = data.xpos[demo.robots[0].base, :2] if leader_follow else data.xpos[demo.payload, :2]
        report["goal_reference"] = "leader_base" if leader_follow else "payload"
        report["goal_error_m"] = float(np.linalg.norm(goal_position - goal))
        report["goal_reached"] = report["goal_error_m"] < .15
    if audit is not None:
        report["grasp_audit"] = audit.summary()
        report["grasp_audit"]["trace"] = str(audit_path)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[COOP] {json.dumps(report)}\n[COOP] Report: {path}", flush=True)
    return 0 if reason in ("completed", "viewer_closed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
