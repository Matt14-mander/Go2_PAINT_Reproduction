"""PAINT Transport Environment tutorial port to Isaac Lab / GO2-PIPER."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg, EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from .paint_loco_env_cfg import Go2PiperPaintLocoEnvCfg
from ...mdp import transport_terms as terms
from ...mdp.transport_action import TransportActionCfg
from ...mdp.transport_payload import TransportPayloadCfg, randomize_transport_mass
from ... import mdp


@configclass
class TransportActionsCfg:
    transport = TransportActionCfg()


@configclass
class EmptyCommandsCfg:
    pass


@configclass
class TransportObservationsCfg:
    @configclass
    class Policy(ObservationGroupCfg):
        state = ObservationTermCfg(func=terms.policy_observation)
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class Teacher(ObservationGroupCfg):
        state = ObservationTermCfg(func=terms.teacher_observation)
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class Student(ObservationGroupCfg):
        state = ObservationTermCfg(func=terms.student_observation)
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class Critic(ObservationGroupCfg):
        state = ObservationTermCfg(func=terms.critic_observation)
        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    policy: Policy = Policy()
    teacher: Teacher = Teacher()
    student: Student = Student()
    critic: Critic = Critic()


@configclass
class TransportRewardsCfg:
    force_tracking = RewardTermCfg(func=terms.force_tracking_cost, weight=-2e-2)
    torque_tracking = RewardTermCfg(func=terms.torque_tracking_cost, weight=-1e-2)
    payload_height = RewardTermCfg(func=terms.payload_height, weight=-3e-3)
    joint_torque = RewardTermCfg(func=terms.joint_torque, weight=-2e-9)
    joint_dof = RewardTermCfg(func=terms.joint_dof, weight=-2e-4)
    action = RewardTermCfg(func=terms.action_cost, weight=-2e-4)
    termination = RewardTermCfg(func=terms.termination_cost, weight=-2500.0)
    arm_posture = RewardTermCfg(func=terms.arm_posture, weight=-3e-3)


@configclass
class Go2PiperPaintTransportEnvCfg(Go2PiperPaintLocoEnvCfg):
    ll_policy_path: str = ""
    observation_mode: str = "teacher"
    arm_history_length: int = 8
    student_noise: bool = True
    wrench_scale: float = 1.0
    periodic_disturbance: bool = True
    literal_tracking_rewards: bool = False
    zero_wrench_probability: float = 0.02

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        self.scene.env_spacing = 4.0
        # Author cross-asset grasp relationships independently in every env.
        # Avoid physics replication until the external constraint is validated.
        self.scene.replicate_physics = False
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        for key in (f"joint{i}" for i in range(1, 7)):
            self.scene.robot.actuators[key].stiffness = 50.0
            self.scene.robot.actuators[key].damping = 4.0
        self.scene.payload = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Payload",
            spawn=TransportPayloadCfg(size=(0.5, 0.3, 0.08),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.3, 0.1))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0, 0.8)))
        self.actions = TransportActionsCfg()
        self.commands = EmptyCommandsCfg()
        self.observations = TransportObservationsCfg()
        self.rewards = TransportRewardsCfg()
        self.rewards.termination.weight = -50.0 / (self.sim.dt * self.decimation)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        self.curriculum.pos_cmd_levels = None
        self.terminations.bad_orientation = None
        self.events.payload_mass = EventTermCfg(func=randomize_transport_mass, mode="startup", params={"mass_range": (0.0, 10.0)})
        self.events.payload_wrench_interval = None
        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.base_com = None
        self.events.randomize_rigid_body_inertia = None
        self.events.randomize_actuator_gains = None
        self.events.add_mass.params.update(mass_distribution_params=(-2.0, 5.0), operation="add")
        self.events.physics_material.params.update(static_friction_range=(0.0, 1.5), dynamic_friction_range=(0.0, 1.5))
        self.events.payload_material = EventTermCfg(func=mdp.randomize_rigid_body_material, mode="startup", params={
            "asset_cfg": SceneEntityCfg("payload"), "static_friction_range": (0.0, 1.5),
            "dynamic_friction_range": (0.0, 1.5), "restitution_range": (0.0, 0.0), "num_buckets": 64,
            "make_consistent": True})
        self.events.reset_base.params["pose_range"] = {"x": (-1, 1), "y": (-1, 1), "yaw": (-3.14159, 3.14159)}
        self.events.reset_base.params["velocity_range"] = {key: (-0.5, 0.5) for key in ("x", "y", "z", "roll", "pitch", "yaw")}


@configclass
class Go2PiperPaintTransportRoughEnvCfg(Go2PiperPaintTransportEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG.copy()
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 2
        self.curriculum.terrain_levels = CurriculumTermCfg(func=terms.terrain_progress)


@configclass
class TransportStage1RewardsCfg(TransportRewardsCfg):
    """GO2 bootstrap shaping, explicitly different from the tutorial baseline."""
    force_tracking = RewardTermCfg(func=terms.force_tracking_cost, weight=-2.0)
    torque_tracking = RewardTermCfg(func=terms.torque_tracking_cost, weight=-1.0)
    command_tracking = RewardTermCfg(func=terms.command_tracking_cost, weight=-0.5)
    base_tilt = RewardTermCfg(func=terms.base_tilt_cost, weight=-0.5)
    joint_dof = RewardTermCfg(func=terms.joint_dof, weight=-1e-5)
    action = RewardTermCfg(func=terms.action_cost, weight=-2e-3)
    # The arm is held, and the inherited height band does not match GO2's
    # nominal grasp height. Neither term should drive this velocity bootstrap.
    payload_height = None
    arm_posture = None


@configclass
class Go2PiperPaintTransportStage1EnvCfg(Go2PiperPaintTransportEnvCfg):
    """New 147/165/9 task: learn stable following with an explicitly fixed arm."""
    def __post_init__(self):
        super().__post_init__()
        self.observation_mode = "teacher"
        self.wrench_scale = 0.2
        self.zero_wrench_probability = 0.2
        self.periodic_disturbance = False
        self.literal_tracking_rewards = False
        self.actions.transport.freeze_arm = True
        self.actions.transport.command_limits = (0.35, 0.35, 0.4)
        self.rewards = TransportStage1RewardsCfg()
        self.rewards.termination.weight = -50.0 / (self.sim.dt * self.decimation)
        self.terminations.bad_orientation = TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": 0.7})
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = actuator.max_delay = 0
        self.events.payload_mass.params["mass_range"] = (0.5, 0.5)
        self.events.physics_material = None
        self.events.payload_material = None
        self.events.add_mass = None
        self.events.reset_base.params["pose_range"] = {key: (0.0, 0.0) for key in ("x", "y", "yaw")}
        self.events.reset_base.params["velocity_range"] = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
        for name in ("reset_robot_joints", "reset_arm_joints"):
            event = getattr(self.events, name)
            event.params["position_range"] = (1.0, 1.0)
            event.params["velocity_range"] = (0.0, 0.0)
