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
    def test_stage2_zero_tail_is_separate_from_startup(self):
        data = NS(root_lin_vel_b=torch.tensor([[.1, 0., 0.], [.02, 0., 0.]]),
                  root_ang_vel_b=torch.zeros(2, 3), joint_pos=torch.zeros(2, 6),
                  default_joint_pos=torch.zeros(2, 6),
                  projected_gravity_b=torch.tensor([[0., 0., -1.], [0., 0., -1.]]))
        action = NS(robot=NS(data=data), arm_ids=list(range(6)), arm_target=torch.full((2, 6), .004),
                    previous_arm_target=torch.zeros(2, 6), reference=lambda: torch.zeros(2, 3),
                    ll_sha256="test", cfg=NS(command_deadband=.05, command_limits=(.35, .35, .4),
                                            settle_seconds=1., stop_hold_seconds=3.))
        env = NS(action_manager=NS(get_term=lambda _: action), device="cpu", step_dt=.02,
                 episode_length_buf=torch.tensor([10, 700]), cfg=NS(episode_length_s=15.))
        metric = module.TransportEvalMetrics(env)
        metric.sample(torch.full((2, 9), .2))
        result = metric.summary()
        self.assertEqual(result["release_samples"], 1)
        self.assertAlmostEqual(result["release_velocity_rmse_vx_vy_wz"][0], .02, places=6)
        self.assertAlmostEqual(result["max_arm_target_speed_rad_s"], .2, places=6)
        torch.testing.assert_close(torch.tensor(result["mean_teacher_command_vx_vy_wz"]), torch.full((3,), .1))

    def test_transport_separates_teacher_mapping_from_motion_and_records_falls(self):
        data = NS(root_lin_vel_b=torch.tensor([[.1, 0., 0.], [99., 0., 0.]]),
                  root_ang_vel_b=torch.zeros(2, 3), joint_pos=torch.full((2, 6), .2),
                  default_joint_pos=torch.zeros(2, 6),
                  projected_gravity_b=torch.tensor([[0., -.8660254, -.5], [0., 0., -1.]]))
        action = NS(robot=NS(data=data), arm_ids=list(range(6)), arm_target=torch.full((2, 6), .1),
                    reference=lambda: torch.tensor([[.2, 0., 0.], [0., 0., 0.]]),
                    ll_sha256="frozen-test", cfg=NS(command_deadband=.05))
        env = NS(action_manager=NS(get_term=lambda _: action), device="cpu", episode_length_buf=torch.tensor([2, 0]))
        metric = module.TransportEvalMetrics(env)
        metric.sample(torch.tensor([[.2, 0., 0.], [99., 0., 0.]]))
        metric.record_resets(torch.tensor([True, False]), {"time_outs": torch.tensor([False, False])})
        metric.record_resets(torch.tensor([False, True]), {"time_outs": torch.tensor([False, True])})
        result = metric.summary()
        self.assertEqual(result["samples"], 1)  # reset state's fake velocity is excluded
        self.assertAlmostEqual(result["teacher_command_rmse_vs_reference"][0], 0.)
        self.assertAlmostEqual(result["velocity_rmse_vs_reference"][0], .1, places=6)
        self.assertAlmostEqual(result["arm_target_tracking_rmse_rad"], .1, places=6)
        self.assertAlmostEqual(result["max_tilt_deg"], 60., places=4)
        self.assertAlmostEqual(result["max_abs_roll_deg"], 60., places=4)
        self.assertEqual(result["tilt_over_45_sample_fraction"], 1)
        self.assertEqual(result["non_timeout_resets"], 1)
        self.assertEqual(result["time_limit_resets"], 1)

    def test_stage1_reports_raw_saturation_before_clipping_and_excludes_reset(self):
        data = NS(root_lin_vel_b=torch.zeros(2, 3), root_ang_vel_b=torch.zeros(2, 3),
                  joint_pos=torch.zeros(2, 6), default_joint_pos=torch.zeros(2, 6),
                  projected_gravity_b=torch.tensor([[0., 0., -1.], [0., 0., -1.]]))
        action = NS(robot=NS(data=data), arm_ids=list(range(6)), arm_target=torch.zeros(2, 6),
                    reference=lambda: torch.zeros(2, 3), ll_sha256="test",
                    cfg=NS(command_deadband=.05, command_limits=(.35, .35, .4), freeze_arm=True))
        env = NS(action_manager=NS(get_term=lambda _: action), device="cpu",
                 episode_length_buf=torch.tensor([2, 0]))
        metric = module.TransportEvalMetrics(env)
        metric.sample(torch.tensor([[.7, .01, -.8], [99., 99., 99.]]))
        result = metric.summary()
        self.assertEqual(result["raw_command_saturation_fraction_xyz"], [1., 0., 1.])
        torch.testing.assert_close(torch.tensor(result["raw_command_max_abs_xyz"]), torch.tensor([.7, .01, .8]))
        torch.testing.assert_close(torch.tensor(result["mean_teacher_command_vx_vy_wz"]),
                                   torch.tensor([.35, 0., -.4]))
        self.assertTrue(result["freeze_arm"])

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
