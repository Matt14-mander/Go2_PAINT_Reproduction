"""Isaac Lab adapter; existing Flat/Rough tasks never import this config implicitly."""

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.utils import configclass
from .cfg.command_cfg import UniformVelocityCommandCfg
from .low_speed_math import sample_commands, sequence_factor, precision_reward


class LowSpeedVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.target = torch.zeros_like(self.vel_command_b)
        self.age = torch.zeros(self.num_envs, device=self.device)
        for name in ("stand", "yaw", "translation", "mixed", "sequence"):
            self.metrics[f"fraction_{name}"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if self.cfg.fixed_command is not None:
            self.mode[ids] = 3
            self.target[ids] = self.target.new_tensor(self.cfg.fixed_command)
        else:
            mode, target = sample_commands(len(ids), self.device, self.cfg.probabilities,
                                          self.cfg.mixed_limits)
            self.mode[ids], self.target[ids] = mode, target
        self.age[ids] = 0
        self.is_standing_env[ids] = False  # standing is explicitly represented by mode 0
        self.is_heading_env[ids] = False
        self.vel_command_b[ids] = self.target[ids]
        self.vel_command_b[ids[self.mode[ids] == 4]] = 0

    def _update_command(self):
        factor = torch.where(self.mode == 4, sequence_factor(self.age), 1.)
        self.vel_command_b[:] = self.target * factor[:, None]
        self.age += self._env.step_dt

    def _update_metrics(self):
        super()._update_metrics()
        for index, name in enumerate(("stand", "yaw", "translation", "mixed", "sequence")):
            self.metrics[f"fraction_{name}"] += (self.mode == index) / self._env.max_episode_length


@configclass
class LowSpeedVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = LowSpeedVelocityCommand
    probabilities: tuple = (.15, .30, .20, .25, .10)
    mixed_limits: tuple = (1., 1., 1.5)
    fixed_command: tuple | None = None


def low_speed_tracking(env, axis: int, width: float):
    data = env.scene["robot"].data
    actual = torch.cat((data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3]), dim=-1)
    return precision_reward(env.command_manager.get_command("base_velocity"), actual, axis, width)
