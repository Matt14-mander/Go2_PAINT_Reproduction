"""9-D transport action: frozen 247-D LL actor plus independently controlled arm."""

import hashlib
from pathlib import Path

import torch
from pxr import UsdPhysics

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul
from isaacsim.core.utils.stage import get_current_stage

from .paint_locomotion import ARM_JOINT_NAMES, LEG_JOINT_NAMES
from .transport_math import append_history, locomotion_observation, process_transport_action, wrench_profile


class TransportAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = self._asset
        self.payload = env.scene["payload"]
        self.leg_ids, _ = self.robot.find_joints(LEG_JOINT_NAMES, preserve_order=True)
        self.arm_ids, _ = self.robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        self.obs_ids = self.leg_ids + self.arm_ids
        self.ee_id = self.robot.find_bodies("end_effector")[0][0]
        if len(self.leg_ids) != 12 or len(self.arm_ids) != 6:
            raise ValueError("Expected 12 leg and 6 arm joints")
        policy_path = Path(cfg.ll_policy_path).expanduser()
        if not cfg.ll_policy_path or not policy_path.is_file():
            raise FileNotFoundError("Set env.ll_policy_path to the frozen exported/policy.pt (not model_9999.pt)")
        with policy_path.open("rb") as stream:
            self.ll_policy = torch.jit.load(stream, map_location=env.device).eval()
        for parameter in self.ll_policy.parameters():
            parameter.requires_grad_(False)
        with torch.inference_mode():
            probe = self.ll_policy(torch.zeros(1, 247, device=env.device))
        if not isinstance(probe, torch.Tensor) or probe.shape != (1, 12):
            raise ValueError("Frozen LL policy must map (N,247) to (N,12)")
        if hasattr(self.ll_policy, "reset"):
            self.ll_policy.reset()
        self.ll_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        print(f"[PAINT] Frozen LL: {policy_path}, sha256={self.ll_sha256}", flush=True)
        count, device = env.num_envs, env.device
        self._raw = torch.zeros(count, 9, device=device)
        self._processed = torch.zeros_like(self._raw)
        self.previous_hl = torch.zeros_like(self._raw)
        self.base_command = torch.zeros(count, 3, device=device)
        self.leg_actions = torch.zeros(count, 12, device=device)
        self.arm_target = self.robot.data.default_joint_pos[:, self.arm_ids].clone()
        self.leg_target = self.robot.data.default_joint_pos[:, self.leg_ids].clone()
        self.wrench_target = torch.zeros(count, 3, device=device)
        self.start_position = torch.zeros(count, 3, device=device)
        self.clean_history = torch.zeros(count, env.cfg.arm_history_length, 18, device=device)
        self.student_history = torch.zeros_like(self.clean_history)
        self.history_step = torch.full((count,), -1, device=device, dtype=torch.long)
        self.ee_velocity = torch.zeros(count, 3, device=device)
        self.payload_velocity = torch.zeros_like(self.ee_velocity)
        self.ee_acceleration = torch.zeros_like(self.ee_velocity)
        self.payload_acceleration = torch.zeros_like(self.ee_velocity)
        self.robot_forces = torch.zeros(count, self.robot.num_bodies, 3, device=device)
        self.robot_torques = torch.zeros_like(self.robot_forces)
        self.payload_forces = torch.zeros(count, 1, 3, device=device)
        self.payload_torques = torch.zeros_like(self.payload_forces)
        joint = UsdPhysics.FixedJoint(get_current_stage().GetPrimAtPath(env.scene.env_prim_paths[0] + "/Payload/GraspJoint"))
        offset = joint.GetLocalRot0Attr().Get()
        self.grasp_rotation = torch.tensor([offset.GetReal(), *offset.GetImaginary()], device=device).repeat(count, 1)
        self.grasp_position = torch.tensor(env.cfg.scene.payload.spawn.grasp_position, device=device).repeat(count, 1)
        self._substep = 0

    @property
    def action_dim(self):
        return 9

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def reference(self, offset=0.0):
        time = self._env.episode_length_buf * self._env.step_dt + offset
        return self.wrench_target * wrench_profile(time, self._env.cfg.episode_length_s).unsqueeze(-1)

    def process_actions(self, actions):
        # Penalize the sampled actor action before command clipping. Otherwise
        # e.g. vx=-1 and vx=-100 have identical dynamics AND action costs, leaving
        # PPO no extra cost for moving arbitrarily far beyond the command limit.
        self.previous_hl.copy_(self._raw)
        self._raw.copy_(actions)
        self.base_command, self.arm_target, processed = process_transport_action(
            actions, self.robot.data.default_joint_pos[:, self.arm_ids],
            self.robot.data.soft_joint_pos_limits[:, self.arm_ids], self.cfg.command_deadband)
        self._processed.copy_(processed)
        data = self.robot.data
        scanner = self._env.scene["height_scanner"].data
        heights = scanner.pos_w[:, 2:3] - scanner.ray_hits_w[..., 2] - 0.5
        obs = locomotion_observation(data.root_lin_vel_b, data.root_ang_vel_b, data.projected_gravity_b,
                                     self.base_command, data.joint_pos[:, self.obs_ids] - data.default_joint_pos[:, self.obs_ids],
                                     data.joint_vel[:, self.obs_ids] - data.default_joint_vel[:, self.obs_ids],
                                     self.leg_actions, heights)
        with torch.inference_mode():
            output = self.ll_policy(obs)
        if output.shape != self.leg_actions.shape or not torch.isfinite(output).all():
            raise RuntimeError("Invalid frozen LL output")
        self.leg_actions.copy_(output.clamp(-4.0, 4.0))
        self.leg_target = data.default_joint_pos[:, self.leg_ids] + 0.25 * self.leg_actions
        self._substep = 0

    def apply_actions(self):
        self.robot.set_joint_position_target(self.leg_target, joint_ids=self.leg_ids)
        self.robot.set_joint_position_target(self.arm_target, joint_ids=self.arm_ids)
        reference = self.reference(self._substep * self._env.physics_dt)
        force_b = torch.zeros_like(reference)
        torque_b = torch.zeros_like(reference)
        force_b[:, :2] = 40.0 * reference[:, :2]
        torque_b[:, 2] = 10.0 * reference[:, 2]
        quaternion = self.robot.data.root_quat_w
        self.payload_forces[:, 0] = quat_apply(quaternion, force_b)
        self.payload_torques[:, 0] = quat_apply(quaternion, torque_b)
        self.payload.set_external_force_and_torque(self.payload_forces, self.payload_torques, is_global=True)
        self.robot_forces.zero_()
        self.robot_torques.zero_()
        if self._env.cfg.periodic_disturbance:
            phase = 2 * torch.pi * (self._env.episode_length_buf * self._env.step_dt + self._substep * self._env.physics_dt)
            self.robot_forces[:, self.ee_id, 2] = 2.0 * torch.sin(phase)
            self.robot_torques[:, self.ee_id, 0] = torch.sin(phase + torch.pi / 4)
            self.robot_torques[:, self.ee_id, 1] = torch.sin(phase + torch.pi / 2)
        self.robot.set_external_force_and_torque(self.robot_forces, self.robot_torques, is_global=True)
        self._substep += 1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self._raw, self._processed, self.previous_hl, self.base_command, self.leg_actions,
                      self.clean_history, self.student_history, self.ee_acceleration, self.payload_acceleration,
                      self.robot_forces, self.robot_torques, self.payload_forces, self.payload_torques):
            value[ids] = 0.0
        self.arm_target[ids] = self.robot.data.default_joint_pos[ids][:, self.arm_ids]
        self.leg_target[ids] = self.robot.data.default_joint_pos[ids][:, self.leg_ids]
        self.wrench_target[ids] = 0.0
        self.history_step[ids] = self._env.common_step_counter
        # The verified LL is a stateless MLP. Do not globally reset another env's
        # LL/action history when only a subset terminates.

    def align_payload_and_resample(self, ids):
        data = self.robot.data
        ee_position = data.body_pos_w[ids, self.ee_id]
        ee_quaternion = data.body_quat_w[ids, self.ee_id]
        quaternion = quat_mul(ee_quaternion, self.grasp_rotation[ids])
        position = ee_position - quat_apply(quaternion, self.grasp_position[ids])
        angular = data.body_ang_vel_w[ids, self.ee_id]
        ee_com_position = data.body_com_pos_w[ids, self.ee_id]
        grasp_velocity = data.body_lin_vel_w[ids, self.ee_id] + torch.cross(angular, ee_position - ee_com_position, dim=-1)
        com_offset = self.payload.data.com_pos_b[ids].reshape(len(ids), 3)
        com_position = position + quat_apply(quaternion, com_offset)
        velocity = grasp_velocity + torch.cross(angular, com_position - ee_position, dim=-1)
        self.payload.write_root_state_to_sim(torch.cat((position, quaternion, velocity, angular), dim=-1), env_ids=ids)
        target = torch.empty(len(ids), 3, device=self.device).uniform_(-1, 1) * self._env.cfg.wrench_scale
        zero = torch.rand(len(ids), device=self.device) < 0.02
        target[zero] = 0.0
        self.wrench_target[ids] = target
        self.start_position[ids] = data.root_pos_w[ids]
        self.ee_velocity[ids] = data.body_lin_vel_w[ids, self.ee_id]
        self.payload_velocity[ids] = velocity

    def update_history(self):
        mask = (self.history_step != self._env.common_step_counter) & (self._env.episode_length_buf > 0)
        if not mask.any():
            return
        data = self.robot.data
        frame = torch.cat((data.joint_pos[:, self.arm_ids] - data.default_joint_pos[:, self.arm_ids],
                           data.joint_vel[:, self.arm_ids], self.arm_target), dim=-1)
        append_history(self.clean_history, frame, mask)
        noisy = frame.clone()
        if self._env.cfg.student_noise:
            noisy[:, :6] += torch.empty_like(noisy[:, :6]).uniform_(-0.01, 0.01)
            noisy[:, 6:12] += torch.empty_like(noisy[:, 6:12]).uniform_(-1.5, 1.5)
        append_history(self.student_history, noisy, mask)
        velocity_ee = data.body_lin_vel_w[:, self.ee_id]
        velocity_payload = self.payload.data.root_lin_vel_w
        self.ee_acceleration[mask] = (velocity_ee[mask] - self.ee_velocity[mask]) / self._env.step_dt
        self.payload_acceleration[mask] = (velocity_payload[mask] - self.payload_velocity[mask]) / self._env.step_dt
        self.ee_velocity[mask] = velocity_ee[mask]
        self.payload_velocity[mask] = velocity_payload[mask]
        self.history_step[mask] = self._env.common_step_counter


@configclass
class TransportActionCfg(ActionTermCfg):
    class_type: type = TransportAction
    asset_name: str = "robot"
    ll_policy_path: str = ""
    command_deadband: float = 0.05
