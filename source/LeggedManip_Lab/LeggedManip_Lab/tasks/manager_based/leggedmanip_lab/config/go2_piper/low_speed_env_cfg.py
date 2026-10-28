"""Independent low-speed fine tuning. Original Flat/Rough configs remain intact."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from .paint_loco_env_cfg import Go2PiperPaintLocoEnvCfg
from ...mdp.low_speed import LowSpeedVelocityCommandCfg, low_speed_tracking


@configclass
class Go2PiperLowSpeedRestoreEnvCfg(Go2PiperPaintLocoEnvCfg):
    """Low-speed sampling/reward with the original Flat domain randomization."""
    def __post_init__(self):
        super().__post_init__()
        old = self.commands.base_velocity
        self.commands.base_velocity = LowSpeedVelocityCommandCfg(
            asset_name="robot", resampling_time_range=(12., 12.),
            heading_command=False, rel_standing_envs=0., debug_vis=False,
            ranges=old.limit_ranges.copy(), limit_ranges=old.limit_ranges.copy(),
            curriculum_enabled=False)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        # Two complete start/stop/reverse cycles per episode.
        self.episode_length_s = 24.
        self.scene.num_envs = 1024
        # Keep original broad tracking and all regularizers, including airtime.
        self.rewards.low_speed_vx = RewTerm(func=low_speed_tracking, weight=.75, params={"axis": 0, "width": .04})
        self.rewards.low_speed_vy = RewTerm(func=low_speed_tracking, weight=.75, params={"axis": 1, "width": .04})
        self.rewards.low_speed_wz = RewTerm(func=low_speed_tracking, weight=1.5, params={"axis": 2, "width": .06})


@configclass
class Go2PiperLowSpeedEnvCfg(Go2PiperLowSpeedRestoreEnvCfg):
    """Focused stage: light payload, modest noise/DR, no extra pushes/wrenches."""
    def __post_init__(self):
        super().__post_init__()
        self.events.payload_mass.params["mass_range"] = (0., 1.)
        self.events.push_robot = None
        self.events.payload_wrench_interval = None
        self.events.physics_material.params.update(static_friction_range=(.7, 1.0),
                                                   dynamic_friction_range=(.7, 1.0), restitution_range=(0., 0.))
        self.events.add_mass.params["mass_distribution_params"] = (.98, 1.02)
        self.events.base_com.params["com_range"] = {"x": (-.01, .01), "y": (-.01, .01), "z": (-.005, .005)}
        self.events.randomize_rigid_body_inertia.params["inertia_distribution_params"] = (.95, 1.05)
        for key in ("stiffness_distribution_params", "damping_distribution_params"):
            self.events.randomize_actuator_gains.params[key] = (.95, 1.05)
        self.events.reset_robot_joints.params["position_range"] = (.95, 1.05)
        for actuator in self.scene.robot.actuators.values():
            if hasattr(actuator, "min_delay"):
                actuator.min_delay = 0
                actuator.max_delay = 1
        for name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
            noise = getattr(self.observations.policy, name).noise
            noise.n_min *= .25
            noise.n_max *= .25
