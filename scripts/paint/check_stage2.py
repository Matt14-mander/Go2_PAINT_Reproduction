"""Stage2 action parity, checkpoint transfer and real RSL-RL 3.x PPO checks."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
sys.path.insert(0, str(ROOT / "mujoco/deploy/deploy_mujoco/go2_piper"))
from stage2_warm_start import warm_start_stage2
from transport_contract import arm_target, force_profile, load_stage2_contract

spec = importlib.util.spec_from_file_location("stage2_math", ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/transport_math.py")
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)

CONTRACT = dict(freeze_arm=False, arm_scale=.1, arm_max_offset=.1, arm_max_speed=.2,
                command_limits=[.35, .35, .4], command_deadband=.05, control_dt=.02,
                settle_seconds=1., force_ramp_seconds=2., stop_hold_seconds=3.)


class TensorEnv:
    num_envs, num_actions, device, max_episode_length = 8, 9, "cpu", 750
    def __init__(self):
        self.episode_length_buf = torch.zeros(8, dtype=torch.long)
    def get_observations(self):
        return TensorDict({"policy": torch.randn(8, 147), "critic": torch.randn(8, 165)}, batch_size=[8])
    def step(self, actions):
        assert actions.shape == (8, 9) and torch.isfinite(actions).all()
        self.episode_length_buf += 1
        return self.get_observations(), -actions.square().mean(-1), torch.zeros(8, dtype=torch.long), {}


def make_runner():
    cfg = dict(num_steps_per_env=4, save_interval=100, obs_groups={"policy": ["policy"], "critic": ["critic"]},
               policy=dict(class_name="ActorCritic", actor_hidden_dims=[512, 256, 128],
                           critic_hidden_dims=[512, 256, 128], activation="elu", noise_std_type="log", init_noise_std=.1),
               algorithm=dict(class_name="PPO", learning_rate=5e-5, entropy_coef=5e-4,
                              num_learning_epochs=1, num_mini_batches=2, schedule="adaptive", desired_kl=.01))
    result = OnPolicyRunner(TensorEnv(), cfg, log_dir=None, device="cpu")
    result.logger_type = "tensorboard"
    return result


class Stage2Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_arm_limits_slew_reversal_and_cross_sim_parity(self):
        default = np.array([0., .8, -1.2, 0., .4, 0.])
        limits = np.array([[-3., 3.]] * 6)
        previous = default.copy()
        for value in [100.] * 40 + [-100.] * 70 + [0.] * 40:
            raw = np.array([[.2, -.1, .3, *([value] * 6)]])
            _, target, _ = math.process_transport_action(torch.tensor(raw), torch.tensor(default[None]),
                torch.tensor(limits[None]), command_limits=(.35, .35, .4),
                arm_scale=.1, arm_max_offset=.1, arm_max_speed=.2,
                previous_target=torch.tensor(previous[None]), dt=.02)
            deployed = arm_target(raw[0, 3:], default, limits, previous, CONTRACT)
            np.testing.assert_allclose(target[0].numpy(), deployed, atol=1e-10)
            self.assertLessEqual(float(np.max(np.abs(deployed - previous))), .00400001)
            self.assertLessEqual(float(np.max(np.abs(deployed - default))), .10000001)
            previous = deployed
        np.testing.assert_allclose(previous, default)

    def test_force_schedule_has_startup_and_three_second_release_tail(self):
        time = torch.tensor([0., 1., 2., 3., 9., 10., 11., 12., 14., 15.])
        expected = torch.tensor([0., 0., .5, 1., 1., 1., .5, 0., 0., 0.])
        torch.testing.assert_close(math.wrench_profile(time, 15., 1., 2., 3.), expected)
        np.testing.assert_allclose([force_profile(float(t), 15., CONTRACT) for t in time], expected.numpy())

    def test_repeated_force_has_four_release_windows_without_reset(self):
        for origin in (0., 15., 30., 45.):
            for offset, expected in ((0., 0.), (1., 0.), (2., .5), (3., 1.),
                                     (10., 1.), (11., .5), (12., 0.), (14.9, 0.)):
                self.assertAlmostEqual(force_profile(origin + offset, 60., CONTRACT, 15.), expected)
        # A partial final pulse still ends with a full zero-force tail.
        self.assertEqual(force_profile(33., 35., CONTRACT, 15.), 0.)
        for period in (0., 8., 61., float('nan')):
            with self.assertRaises(ValueError):
                force_profile(0., 60., CONTRACT, period)

    def test_manifest_requires_matching_policy_and_complete_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.pt"
            policy.write_bytes(b"fixture")
            manifest = dict(task="GO2-PIPER-PAINT-Transport-Flat-Stage2", transport_action=CONTRACT,
                            policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest())
            (policy.parent / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(load_stage2_contract([policy]), CONTRACT)
            policy.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_stage2_contract([policy])

    def test_actual_rsl3_transfer_preserves_base_resets_arm_then_learns_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "stage1.pt"
            source = make_runner()
            source.current_learning_iteration = 4999
            obs = source.env.get_observations()
            before = source.alg.policy.act_inference(obs).detach().clone()
            with checkpoint.open("wb") as stream:
                torch.save({"model_state_dict": source.alg.policy.state_dict(), "iter": 4999, "infos": {},
                            "optimizer_state_dict": source.alg.optimizer.state_dict()}, stream)
            destination = make_runner()
            metadata = warm_start_stage2(destination, checkpoint)
            after = destination.alg.policy.act_inference(obs)
            torch.testing.assert_close(after[:, :3], before[:, :3])
            torch.testing.assert_close(after[:, 3:], torch.zeros_like(after[:, 3:]))
            torch.testing.assert_close(destination.alg.policy.log_std.exp(), torch.tensor([.05] * 3 + [.1] * 6))
            self.assertEqual(destination.current_learning_iteration, 0)
            self.assertEqual(len(destination.alg.optimizer.state), 0)
            self.assertFalse(metadata["optimizer_loaded"])
            with patch("rsl_rl.runners.on_policy_runner.store_code_state", return_value=[]):
                destination.learn(num_learning_iterations=2, init_at_random_ep_len=False)
            self.assertTrue(all(torch.isfinite(p).all() for p in destination.alg.policy.parameters()))
            self.assertGreater(len(destination.alg.optimizer.state), 0)
            trained = destination.alg.policy.act_inference(obs).detach().clone()
            self.assertGreater(float(trained[:, 3:].abs().max()), 0)
            with checkpoint.open("wb") as stream:
                torch.save({"model_state_dict": destination.alg.policy.state_dict(), "iter": 77, "infos": {},
                            "optimizer_state_dict": destination.alg.optimizer.state_dict()}, stream)
            resumed = make_runner()
            resumed.load(str(checkpoint))
            self.assertEqual(resumed.current_learning_iteration, 77)
            torch.testing.assert_close(resumed.alg.policy.act_inference(obs), trained)


if __name__ == "__main__":
    unittest.main()
