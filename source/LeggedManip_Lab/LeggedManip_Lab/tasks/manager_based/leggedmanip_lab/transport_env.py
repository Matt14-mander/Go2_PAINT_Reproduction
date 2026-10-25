"""ManagerBased transport with explicit per-env payload/grasp reset semantics."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


class PaintTransportEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        if cfg.observation_mode not in ("teacher", "student"):
            raise ValueError("observation_mode must be teacher or student")
        if cfg.arm_history_length != 8 or not 0 <= cfg.wrench_scale <= 1:
            raise ValueError("Expected H=8 and wrench_scale in [0,1]")
        cfg.actions.transport.ll_policy_path = cfg.ll_policy_path
        super().__init__(cfg, **kwargs)

    def _reset_idx(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        super()._reset_idx(env_ids)
        # Root/DOF writes must be visible to FK before resetting the separate body.
        self.sim.forward()
        self.scene["robot"].update(self.physics_dt)
        self.action_manager.get_term("transport").align_payload_and_resample(env_ids)
        self.scene["payload"].update(self.physics_dt)

    def step(self, action):
        observations, reward, terminated, truncated, extras = super().step(action)
        extras["teacher_obs"] = observations["teacher"]
        extras["wrench_target"] = self.action_manager.get_term("transport").reference().clone()
        return observations, reward, terminated, truncated, extras
