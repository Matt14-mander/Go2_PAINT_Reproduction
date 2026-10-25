"""CPU checks for speed tracking/arm error metrics; requires torch, not Isaac."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

path = Path(__file__).resolve().parents[1] / "rsl_rl/eval_metrics.py"
spec = importlib.util.spec_from_file_location("eval_metrics", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class MetricsTests(unittest.TestCase):
    def test_tracking_arm_error_and_resets_are_distinct(self):
        data = NS(root_lin_vel_b=torch.tensor([[0.2, 0.0, 0.0]]),
                  root_ang_vel_b=torch.tensor([[0.0, 0.0, 0.1]]),
                  joint_pos=torch.full((1, 6), 0.2), default_joint_pos=torch.zeros(1, 6))
        robot = NS(data=data, find_joints=lambda *args, **kwargs: (list(range(6)), []))
        env = NS(scene={"robot": robot}, device="cpu",
                 command_manager=NS(get_command=lambda _: torch.tensor([[0.3, 0.0, 0.0]])))
        metric = module.LocoEvalMetrics(env)
        metric.sample()
        metric.record_resets(torch.tensor([1]), {"time_outs": torch.tensor([True])})
        metric.record_resets(torch.tensor([1]), {"time_outs": torch.tensor([False])})
        result = metric.summary()
        self.assertAlmostEqual(result["velocity_rmse_vx_vy_wz"][0], 0.1, places=5)
        self.assertAlmostEqual(result["arm_joint_rmse_rad"], 0.2, places=5)
        self.assertEqual(result["time_limit_resets"], 1)
        self.assertEqual(result["non_timeout_resets"], 1)
        self.assertEqual(result["samples"], 1)


if __name__ == "__main__":
    unittest.main()
