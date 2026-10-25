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

    def sample(self):
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
