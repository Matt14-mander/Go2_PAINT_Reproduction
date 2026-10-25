"""Controlled PAINT evaluation options; never change observation/action schemas."""

import math
import re


def add_eval_args(parser):
    parser.add_argument("--base-command", nargs=3, type=float, metavar=("VX", "VY", "WZ"),
                        help="Fixed body-frame velocity command instead of random commands")
    parser.add_argument("--no-disturbances", action="store_true", help="Disable pushes and payload wrenches")
    parser.add_argument("--payload-mass", type=float, help="Fixed additional lumped payload mass in kg")
    parser.add_argument("--no-domain-randomization", action="store_true",
                        help="Nominal physical parameters and repeatable initial pose")
    parser.add_argument("--eval-name", default="play", help="Safe label for video and evaluation summary")


def configure_evaluation(cfg, args):
    command = getattr(args, "base_command", None)
    mass = getattr(args, "payload_mass", None)
    no_disturbances = getattr(args, "no_disturbances", False)
    nominal = getattr(args, "no_domain_randomization", False)
    name = getattr(args, "eval_name", "play")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("--eval-name must contain only letters, digits, '_' or '-'")
    if not any((command is not None, mass is not None, no_disturbances, nominal)):
        return
    if "GO2-PIPER-PAINT-Loco-" not in args.task:
        raise ValueError("Controlled evaluation options currently require a GO2-PIPER-PAINT-Loco task")
    if command is not None:
        if len(command) != 3 or any(not math.isfinite(value) for value in command):
            raise ValueError("--base-command requires three finite values")
        if any(abs(value) > limit for value, limit in zip(command, (1.0, 1.0, 1.5))):
            raise ValueError("Command exceeds the trained envelope: vx/vy +/-1, wz +/-1.5")
        velocity = cfg.commands.base_velocity
        velocity.heading_command = False
        velocity.rel_standing_envs = 0.0
        velocity.curriculum_enabled = False
        # Change both ranges: legacy curriculum can otherwise restore limit_ranges.
        for ranges in (velocity.ranges, velocity.limit_ranges):
            for key, value in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), command):
                setattr(ranges, key, (value, value))
        cfg.curriculum.lin_vel_cmd_levels = None
        cfg.curriculum.ang_vel_cmd_levels = None
    if mass is not None:
        if not math.isfinite(mass) or not 0 <= mass <= 10:
            raise ValueError("--payload-mass must be in [0, 10] kg")
        cfg.events.payload_mass.params["mass_range"] = (mass, mass)
    if no_disturbances:
        for key in ("push_robot", "payload_wrench_reset", "payload_wrench_interval", "base_external_force_torque"):
            if hasattr(cfg.events, key):
                setattr(cfg.events, key, None)
    if nominal:
        for key in ("physics_material", "add_mass", "base_com", "randomize_rigid_body_inertia", "randomize_actuator_gains"):
            if hasattr(cfg.events, key):
                setattr(cfg.events, key, None)
        for key in ("reset_robot_joints", "reset_arm_joints"):
            term = getattr(cfg.events, key, None)
            if term is not None:
                term.params["position_range"] = (1.0, 1.0)
                term.params["velocity_range"] = (0.0, 0.0)
        reset = cfg.events.reset_base
        for key in ("pose_range", "velocity_range"):
            if key in reset.params:
                reset.params[key] = {axis: (0.0, 0.0) for axis in reset.params[key]}
        cfg.observations.policy.enable_corruption = False
    print(f"[EVAL] command={command}; additional payload={mass} kg; "
          f"disturbances={'off' if no_disturbances else 'on'}; nominal_physics={nominal}", flush=True)
