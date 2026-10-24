"""CPU unit checks for the frozen policy interface; requires torch, not Isaac Sim.

Load by file path because importing the project package launches Isaac dependencies.
"""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

source = Path(__file__).resolve().parents[2] / (
    "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/paint_interface.py")
spec = importlib.util.spec_from_file_location("paint_interface", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def load_locomotion_logic():
    """Stub only Isaac's base classes; run our tensor/load functions unchanged.

    This is a CPU unit check, not a replacement for the physics smoke check.
    """
    actions = types.ModuleType("isaaclab.envs.mdp.actions")

    class JointPositionAction:
        def reset(self, env_ids=None):
            pass

        def process_actions(self, action):
            self.received_action = action

        def apply_actions(self):
            pass

    actions.JointPositionAction = JointPositionAction
    actions.JointPositionActionCfg = type("JointPositionActionCfg", (), {})
    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = type("SceneEntityCfg", (), {})
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    stubs = {"isaaclab.envs.mdp.actions": actions, "isaaclab.managers": managers, "isaaclab.utils": utils}
    loco_spec = importlib.util.spec_from_file_location("paint_locomotion", source.with_name("paint_locomotion.py"))
    loco = importlib.util.module_from_spec(loco_spec)
    with patch.dict(sys.modules, stubs):
        loco_spec.loader.exec_module(loco)
    return loco


locomotion = load_locomotion_logic()


class InterfaceTests(unittest.TestCase):
    def test_command_limits_and_deadband(self):
        result = module.process_base_command(torch.tensor([[2.0, -3.0, 2.0], [0.01, -0.04, 0.0]]))
        torch.testing.assert_close(result, torch.tensor([[1.0, -1.0, 1.5], [0.0, 0.0, 0.0]]))

    def test_state_is_per_environment_and_clipped(self):
        interface = module.PaintLocoInterface(2)
        action = interface.record_action(torch.full((2, 12), 8.0))
        torch.testing.assert_close(action, torch.full((2, 12), 4.0))
        interface.reset([0])
        torch.testing.assert_close(interface.previous_action[0], torch.zeros(12))
        torch.testing.assert_close(interface.previous_action[1], torch.full((12,), 4.0))

    def test_observation_order_and_scaling(self):
        interface = module.PaintLocoInterface(2)
        observation = interface.build_observation(
            torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3), torch.ones(2, 3),
            torch.ones(2, 12), torch.ones(2, 12), torch.full((2, 187), 2.0))
        self.assertEqual(observation.shape, (2, 235))
        torch.testing.assert_close(observation[:, :3], torch.full((2, 3), 2.0))
        torch.testing.assert_close(observation[:, 24:36], torch.full((2, 12), 0.05))
        torch.testing.assert_close(observation[:, 36:48], torch.zeros(2, 12))
        torch.testing.assert_close(observation[:, 48:], torch.ones(2, 187))

    def test_reject_invalid_action(self):
        interface = module.PaintLocoInterface(2)
        for action in (torch.zeros(2, 18), torch.full((2, 12), float("nan"))):
            with self.assertRaises(ValueError):
                interface.record_action(action)


class LoadLogicTests(unittest.TestCase):
    def make_term(self):
        term = locomotion.PaintLegPositionAction.__new__(locomotion.PaintLegPositionAction)
        term.force_w = torch.zeros(256, 8, 3)
        term.torque_w = torch.zeros_like(term.force_w)
        return term

    def test_wrench_bounds_and_partial_reset(self):
        term = self.make_term()
        env = types.SimpleNamespace(num_envs=256, device="cpu",
                                    action_manager=types.SimpleNamespace(get_term=lambda _: term))
        cfg = types.SimpleNamespace(body_ids=[5])
        locomotion.sample_payload_wrench(env, None, cfg, zero_probability=0.0)
        self.assertLessEqual(torch.linalg.vector_norm(term.force_w, dim=-1).max(), 40.001)
        self.assertLessEqual(term.torque_w.abs().max(), 10.001)
        self.assertGreater(torch.count_nonzero(term.force_w[:, 5, :2]), 0)
        self.assertEqual(torch.count_nonzero(term.force_w[..., 2]), 0)
        self.assertEqual(torch.count_nonzero(term.force_w[:, :5]), 0)
        before = term.force_w[1:].clone()
        term.reset([0])
        torch.testing.assert_close(term.force_w[1:], before)
        self.assertEqual(torch.count_nonzero(term.force_w[0]), 0)
        locomotion.sample_payload_wrench(env, [2, 3], cfg, zero_probability=1.0)
        self.assertEqual(torch.count_nonzero(term.force_w[[2, 3]]), 0)
        self.assertEqual(torch.count_nonzero(term.torque_w[[2, 3]]), 0)

    def test_apply_holds_arm_and_reissues_world_wrench(self):
        term = self.make_term()
        calls = []
        term._arm_ids = [12, 13, 14, 15, 16, 17]
        term._asset = types.SimpleNamespace(
            data=types.SimpleNamespace(default_joint_pos=torch.ones(256, 18)),
            set_joint_position_target=lambda targets, joint_ids: calls.append((targets.clone(), joint_ids)),
            set_external_force_and_torque=lambda **kwargs: calls.append(kwargs),
        )
        term.process_actions(torch.full((256, 12), 9.0))
        torch.testing.assert_close(term.received_action, torch.full((256, 12), 4.0))
        term.apply_actions()
        term.apply_actions()
        self.assertEqual(len(calls), 4)
        torch.testing.assert_close(calls[0][0], torch.ones(256, 6))
        self.assertTrue(calls[1]["is_global"])
        self.assertIs(calls[3]["forces"], term.force_w)

    def test_payload_changes_selected_body_mass_and_inertia(self):
        masses = torch.ones(4, 8)
        inertias = torch.ones(4, 8, 9)
        view = types.SimpleNamespace(
            get_masses=lambda: masses,
            get_inertias=lambda: inertias,
            set_masses=lambda value, ids: masses.copy_(value),
            set_inertias=lambda value, ids: inertias.copy_(value),
        )
        env = types.SimpleNamespace(num_envs=4, scene={"robot": types.SimpleNamespace(root_physx_view=view)})
        cfg = types.SimpleNamespace(name="robot", body_ids=[5])
        locomotion.randomize_lumped_payload(env, [1, 3], cfg, mass_range=(2.0, 2.0), radius=0.1)
        torch.testing.assert_close(masses[[1, 3], 5], torch.full((2,), 3.0))
        torch.testing.assert_close(masses[[0, 2]], torch.ones(2, 8))
        torch.testing.assert_close(inertias[[1, 3], 5, 0], torch.full((2,), 1.008))
        torch.testing.assert_close(inertias[[1, 3], 5, 1], torch.ones(2))

    def test_height_reward_is_invariant_to_world_translation(self):
        base = torch.tensor([[0.0, 0.0, 0.3]])
        rays = torch.zeros(1, 187, 3)
        env = types.SimpleNamespace(scene={
            "robot": types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=base)),
            "height_scanner": types.SimpleNamespace(data=types.SimpleNamespace(ray_hits_w=rays)),
        })
        before = locomotion.terrain_relative_base_height(env)
        base[:, 2] += 5.0
        rays[..., 2] += 5.0
        torch.testing.assert_close(locomotion.terrain_relative_base_height(env), before)


if __name__ == "__main__":
    unittest.main()
