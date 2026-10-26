"""CPU checks for transport interfaces and reward direction; no Isaac imports."""

import importlib.util
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/transport_math.py"
spec = importlib.util.spec_from_file_location("transport_math", PATH)
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)


class TransportMathChecks(unittest.TestCase):
    def test_stage1_holds_arm_despite_extreme_actor_output_and_limits_base(self):
        defaults = torch.tensor([[0., .8, -1.2, 0., .4, 0.]])
        limits = torch.tensor([[[-3., 3.]] * 6])
        raw = torch.full((1, 9), 100.)
        command, target, processed = math.process_transport_action(raw, defaults, limits,
            command_limits=(.35, .35, .4), freeze_arm=True)
        torch.testing.assert_close(target, defaults)
        torch.testing.assert_close(command, torch.tensor([[.35, .35, .4]]))
        torch.testing.assert_close(processed[:, 3:], torch.full((1, 6), 100.))

    def test_episode_profile(self):
        times = torch.tensor([-1, 0, .5, 1, 5, 9, 9.5, 10, 11])
        torch.testing.assert_close(math.wrench_profile(times), torch.tensor([0, 0, .5, 1, 1, 1, .5, 0, 0]))

    def test_action_limits_and_arm_independence(self):
        action = torch.tensor([[2., -.01, -3., 2., -2., 0., 0., 0., 0.]])
        limits = torch.tensor([[[-.2, .2]] * 6])
        command, target, processed = math.process_transport_action(action, torch.zeros(1, 6), limits)
        torch.testing.assert_close(command, torch.tensor([[1., 0., -1.5]]))
        torch.testing.assert_close(target, torch.tensor([[.2, -.2, 0., 0., 0., 0.]]))
        torch.testing.assert_close(processed[:, 3:5], torch.tensor([[2., -2.]]))
        _, unbounded_target, _ = math.process_transport_action(action, torch.zeros(1, 6), limits * 10)
        torch.testing.assert_close(unbounded_target[:, :2], torch.tensor([[1., -1.]]))
        with self.assertRaises(ValueError):
            math.process_transport_action(torch.zeros(1, 12), torch.zeros(1, 6), limits)

    def test_frozen_locomotion_layout(self):
        obs = math.locomotion_observation(*(torch.ones(2, n) for n in (3, 3, 3, 3, 18, 18, 12, 187)))
        self.assertEqual(obs.shape, (2, 247))
        torch.testing.assert_close(obs[:, :3], torch.full((2, 3), 2.))
        torch.testing.assert_close(obs[:, 3:6], torch.full((2, 3), .2))
        torch.testing.assert_close(obs[:, 30:48], torch.full((2, 18), .05))
        torch.testing.assert_close(obs[:, 48:60], torch.ones(2, 12))

    def test_partial_history_reset(self):
        history = torch.zeros(2, 8, 18)
        math.append_history(history, torch.ones(2, 18), torch.tensor([True, True]))
        history[0] = 0
        math.append_history(history, torch.full((2, 18), 2.), torch.tensor([True, False]))
        torch.testing.assert_close(history[0, -1], torch.full((18,), 2.))
        self.assertEqual(history[0, :-1].count_nonzero(), 0)
        torch.testing.assert_close(history[1, -1], torch.ones(18))
        self.assertEqual(history[1, :-1].count_nonzero(), 0)

    def test_maximizing_negative_cost_encourages_tracking(self):
        ref = torch.tensor([[.3, -.1, .2]])
        good = math.tracking_terms(ref, ref, ref[:, 2])
        bad = math.tracking_terms(ref, torch.zeros_like(ref), torch.zeros(1))
        self.assertTrue(all(float(value) == 0 for value in good))
        self.assertTrue(all(float(-a) > float(-b) for a, b in zip(good, bad)))
        literal = math.tracking_terms(ref, ref, ref[:, 2], True)
        self.assertTrue(all(float(value) == 1 for value in literal))

    def test_height_band(self):
        torch.testing.assert_close(math.payload_height_cost(torch.tensor([0., .05, .2, .25, .3])),
                                   torch.tensor([.0025, 0., 0., 0., .0025]))


if __name__ == "__main__":
    unittest.main()
