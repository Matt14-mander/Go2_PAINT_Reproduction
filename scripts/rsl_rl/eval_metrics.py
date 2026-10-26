"""Rollout diagnostics for PAINT. Metrics use pre-step state to avoid reset jumps."""

import torch


class LocoEvalMetrics:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.arm_ids, _ = self.robot.find_joints([f"joint{i}" for i in range(1, 7)], preserve_order=True)
        if len(self.arm_ids) != 6:
            raise ValueError("Expected six arm joints for PAINT evaluation diagnostics")
        self.error_sums = torch.zeros(4, device=env.device)
        self.velocity_sums = torch.zeros(3, device=env.device)
        self.command_sums = torch.zeros(3, device=env.device)
        self.arm_max = torch.zeros((), device=env.device)
        self.count = 0
        self.resets = 0
        self.timeouts = 0
        self.has_timeouts = False

    def sample(self, actions=None):
        data = self.robot.data
        velocity = torch.cat((data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3]), dim=-1)
        command = self.env.command_manager.get_command("base_velocity")[:, :3]
        error = velocity - command
        arm_error = data.joint_pos[:, self.arm_ids] - data.default_joint_pos[:, self.arm_ids]
        self.error_sums[:3] += error.square().sum(dim=0)
        self.error_sums[3] += arm_error.square().mean(dim=1).sum()
        self.arm_max = torch.maximum(self.arm_max, arm_error.abs().max())
        self.velocity_sums += velocity.sum(dim=0)
        self.command_sums += command.sum(dim=0)
        self.count += velocity.shape[0]

    def record_resets(self, dones, extras):
        self.resets += int(dones.count_nonzero().item())
        if "time_outs" in extras:
            self.has_timeouts = True
            self.timeouts += int(extras["time_outs"].count_nonzero().item())

    def summary(self):
        if not self.count:
            return {"samples": 0}
        rmse = (self.error_sums / self.count).sqrt().tolist()
        return {"samples": self.count,
                "mean_command_vx_vy_wz": (self.command_sums / self.count).tolist(),
                "mean_actual_vx_vy_wz": (self.velocity_sums / self.count).tolist(),
                "velocity_rmse_vx_vy_wz": rmse[:3],
                "arm_joint_rmse_rad": rmse[3], "arm_joint_max_abs_error_rad": self.arm_max.item(),
                "episode_resets": self.resets,
                "time_limit_resets": self.timeouts if self.has_timeouts else None,
                "non_timeout_resets": self.resets - self.timeouts if self.has_timeouts else None}


class TransportEvalMetrics:
    """Pre-step diagnostics; terminal counters include events hidden by auto-reset.

    Teacher-command error measures intent mapping; velocity error additionally
    contains dynamics/LL tracking. Exclude the instantaneous reset state.
    """

    def __init__(self, env):
        self.env = env
        self.action = env.action_manager.get_term("transport")
        self.sums = torch.zeros(5, 3, device=env.device)
        self.arm_squared = torch.zeros(2, device=env.device)
        self.max_tilt = torch.zeros((), device=env.device)
        self.max_roll = torch.zeros((), device=env.device)
        self.tilt_over_45 = 0
        self.count = self.resets = self.timeouts = 0
        self.has_timeouts = False
        self.saturation_counts = torch.zeros(3, device=env.device)
        self.raw_base_max = torch.zeros(3, device=env.device)

    def sample(self, actions=None):
        action = self.action
        data = action.robot.data
        mask = self.env.episode_length_buf > 0
        n = int(mask.sum())
        if not n:
            return
        reference = action.reference()[mask]
        # Inspect the new deterministic teacher command at the observed state.
        command = action.base_command if actions is None else actions[:, :3]
        limits = command.new_tensor(getattr(action.cfg, "command_limits", (1., 1., 1.5)))
        self.saturation_counts += (command[mask].abs() > limits + 1e-6).sum(0)
        self.raw_base_max = torch.maximum(self.raw_base_max, command[mask].abs().max(0).values)
        command = command.clamp(-limits, limits)
        command = torch.where(command.abs() < action.cfg.command_deadband, 0., command)[mask]
        velocity = torch.cat((data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3]), dim=-1)[mask]
        for index, values in enumerate((reference, command, velocity,
                                         (command - reference).square(), (velocity - reference).square())):
            self.sums[index] += values.sum(0)
        q = data.joint_pos[mask][:, action.arm_ids]
        self.arm_squared[0] += (q - action.arm_target[mask]).square().mean(-1).sum()
        self.arm_squared[1] += (q - data.default_joint_pos[mask][:, action.arm_ids]).square().mean(-1).sum()
        gravity = data.projected_gravity_b[mask]
        tilt = torch.acos((-gravity[:, 2]).clamp(-1., 1.))
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2]).abs()
        self.max_tilt = torch.maximum(self.max_tilt, tilt.max())
        self.max_roll = torch.maximum(self.max_roll, roll.max())
        self.tilt_over_45 += int((tilt > torch.pi / 4).sum())
        self.count += n

    def record_resets(self, dones, extras):
        self.resets += int(dones.count_nonzero())
        if "time_outs" in extras:
            self.has_timeouts = True
            self.timeouts += int(extras["time_outs"].count_nonzero())

    def summary(self):
        result = {"samples": self.count, "episode_resets": self.resets,
                  "time_limit_resets": self.timeouts if self.has_timeouts else None,
                  "non_timeout_resets": self.resets - self.timeouts if self.has_timeouts else None,
                  "ll_sha256": self.action.ll_sha256,
                  "freeze_arm": getattr(self.action.cfg, "freeze_arm", False),
                  "command_limits": list(getattr(self.action.cfg, "command_limits", (1., 1., 1.5))),
                  "sampling": "pre_step_excluding_reset_states; tilt maxima may miss the terminal physics step"}
        if self.count:
            values = self.sums / self.count
            result.update(mean_reference_vx_vy_wz=values[0].tolist(),
                          mean_teacher_command_vx_vy_wz=values[1].tolist(),
                          mean_actual_vx_vy_wz=values[2].tolist(),
                          teacher_command_rmse_vs_reference=values[3].sqrt().tolist(),
                          velocity_rmse_vs_reference=values[4].sqrt().tolist(),
                          arm_target_tracking_rmse_rad=float((self.arm_squared[0] / self.count).sqrt()),
                          arm_posture_deviation_rmse_rad=float((self.arm_squared[1] / self.count).sqrt()),
                          max_tilt_deg=float(torch.rad2deg(self.max_tilt)),
                          max_abs_roll_deg=float(torch.rad2deg(self.max_roll)),
                          tilt_over_45_sample_fraction=self.tilt_over_45 / self.count)
            result.update(raw_command_saturation_fraction_xyz=(self.saturation_counts / self.count).tolist(),
                          raw_command_max_abs_xyz=self.raw_base_max.tolist())
        return result
