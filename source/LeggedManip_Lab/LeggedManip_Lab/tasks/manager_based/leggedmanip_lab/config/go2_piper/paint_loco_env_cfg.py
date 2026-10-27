"""GO2-PIPER replacement low-level locomotion for the PAINT hierarchy."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .flat_env_cfg import Go2PiperFlatEnvCfg
from ... import mdp
from ...mdp.paint_locomotion import (
    LEG_JOINT_NAMES,
    ARM_JOINT_NAMES,
    PaintLegPositionActionCfg,
    privileged_payload_wrench,
    randomize_lumped_payload,
    sample_payload_wrench,
    terrain_relative_base_height,
)


@configclass
class PaintLocoPolicyCfg(ObsGroup):
    """Trained single-frame 247-D interface (includes all 18 joint states)."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
    velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5),
                        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)})
    actions = ObsTerm(func=mdp.last_action)
    height_scan = ObsTerm(func=mdp.height_scan,
                         params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1


@configclass
class Go2PiperPaintLocoEnvCfg(Go2PiperFlatEnvCfg):
    """Flat bootstrap with the same perceptive observation interface as Rough."""

    def __post_init__(self):
        super().__post_init__()
        # Carry posture is an explicit project starting point, to be verified against
        # the physical mount before hardware use. PD controls the real arm mass.
        self.scene.robot.init_state.joint_pos.pop("joint.*", None)
        self.scene.robot.init_state.joint_pos.update(
            {"joint1": 0.0, "joint2": 0.8, "joint3": -1.2,
             "joint4": 0.0, "joint5": 0.4, "joint6": 0.0}
        )
        self.scene.terrain.physics_material.restitution = 0.0
        # Remove online visual asset dependencies for headless training.
        self.scene.terrain.visual_material = None
        self.scene.sky_light.spawn.texture_file = None
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            mesh_prim_paths=["/World/ground"],
            update_period=self.decimation * self.sim.dt,
            debug_vis=False,
        )
        self.actions.joint_pos = PaintLegPositionActionCfg(
            asset_name="robot", joint_names=LEG_JOINT_NAMES,
            preserve_order=True, scale=0.25, use_default_offset=True,
            clip=None,
        )
        self.commands.ee_pose = None
        self.commands.base_velocity.debug_vis = False
        self.commands.base_velocity.curriculum_enabled = True
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.5, 1.5)
        self.curriculum.pos_cmd_levels = None
        self.rewards.end_effector_position_tracking_exp = None
        self.rewards.end_effector_orientation_tracking = None
        self.rewards.arm_deviation = None
        self.rewards.track_base_height_exp = RewTerm(
            func=terrain_relative_base_height, weight=1.0,
            params={"target_height": 0.30, "std": 0.1},
        )
        self.observations.policy = PaintLocoPolicyCfg()
        for group in (self.observations.critic,):
            group.pos_commands = None
            group.joint_pos.params = {"asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)}
            group.joint_vel.params = {"asset_cfg": SceneEntityCfg(
                "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)}
            group.height_scan = ObsTerm(
                func=mdp.height_scan,
                params={"sensor_cfg": SceneEntityCfg("height_scanner")},
                clip=(-1.0, 1.0),
            )
        self.observations.critic.ee_link0_rel_pose = None
        self.observations.critic.joint_torques.params = {"asset_cfg": SceneEntityCfg(
            "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)}
        self.observations.critic.payload_wrench = ObsTerm(func=privileged_payload_wrench, scale=0.025)

        # Restrict joint resets to legs; the arm starts exactly at the carrying pose.
        self.events.reset_robot_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)
        self.events.reset_arm_joints = EventTerm(
            func=mdp.reset_joints_by_scale, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True),
                    "position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)})
        self.events.add_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="base")
        self.events.payload_mass = EventTerm(
            func=randomize_lumped_payload, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                    "mass_range": (0.0, 10.0)},
        )
        # The action term reissues these world-frame wrenches every physics substep.
        self.events.base_external_force_torque = None
        wrench_params = {"asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                         "max_force": 40.0, "max_yaw_torque": 10.0,
                         "zero_probability": 0.2}
        # ManagerBasedRLEnv applies reset events BEFORE action.reset. Clear the
        # wrench in action.reset; the interval event introduces the next load.
        self.events.payload_wrench_interval = EventTerm(
            func=sample_payload_wrench, mode="interval", interval_range_s=(2.0, 5.0),
            params=wrench_params.copy())
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}})


@configclass
class Go2PiperPaintLocoRoughEnvCfg(Go2PiperPaintLocoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG.copy()
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 2
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class Go2PiperPaintLocoEnvCfg_PLAY(Go2PiperPaintLocoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)


@configclass
class Go2PiperPaintLocoRoughEnvCfg_PLAY(Go2PiperPaintLocoRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_play(self)
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None


def configure_play(cfg):
    """Evaluate with training loads and fall detection, but clean observations."""
    cfg.scene.num_envs = 32
    cfg.observations.policy.enable_corruption = False
    cfg.commands.base_velocity.ranges = cfg.commands.base_velocity.limit_ranges.copy()
    cfg.commands.base_velocity.curriculum_enabled = False
    cfg.curriculum.lin_vel_cmd_levels = None
    cfg.curriculum.ang_vel_cmd_levels = None
