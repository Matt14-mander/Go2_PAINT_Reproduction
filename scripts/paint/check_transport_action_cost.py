"""Regression: raw-action costs must distinguish saturated actor outputs.

Execute actual action-processing/reward functions with tensor state doubles;
checks the state/cost contract, not Isaac dynamics or training convergence.
"""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
MDP = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp"
spec = importlib.util.spec_from_file_location("transport_math_checked", MDP / "transport_math.py")
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)


def load_function(path, name, namespace, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec"), namespace)
    return namespace[name]


process = load_function(MDP / "transport_action.py", "process_actions", {
    "torch": torch, "process_transport_action": math.process_transport_action,
    "locomotion_observation": math.locomotion_observation}, "TransportAction")
cost = load_function(MDP / "transport_terms.py", "action_cost", {"term": lambda env: env.action})


def action_state():
    data = NS(default_joint_pos=torch.zeros(1, 18), default_joint_vel=torch.zeros(1, 18),
              joint_pos=torch.zeros(1, 18), joint_vel=torch.zeros(1, 18),
              soft_joint_pos_limits=torch.tensor([[[-3., 3.]] * 18]),
              root_lin_vel_b=torch.zeros(1, 3), root_ang_vel_b=torch.zeros(1, 3),
              projected_gravity_b=torch.tensor([[0., 0., -1.]]))
    return NS(_raw=torch.zeros(1, 9), raw_actions=torch.zeros(1, 9), previous_hl=torch.zeros(1, 9),
              _processed=torch.zeros(1, 9), cfg=NS(command_deadband=.05, command_limits=(1., 1., 1.5), freeze_arm=False), robot=NS(data=data),
              arm_ids=list(range(12, 18)), leg_ids=list(range(12)), obs_ids=list(range(18)),
              leg_actions=torch.zeros(1, 12), ll_policy=lambda obs: torch.zeros(1, 12),
              _env=NS(scene={"height_scanner": NS(data=NS(pos_w=torch.tensor([[0., 0., .35]]),
                                                            ray_hits_w=torch.zeros(1, 187, 3)))}))


class CostChecks(unittest.TestCase):
    def test_stage2_startup_then_arm_slew_uses_previous_target(self):
        action = action_state()
        action.arm_target = torch.zeros(1, 6)
        action.previous_arm_target = torch.zeros(1, 6)
        action.cfg.arm_scale, action.cfg.arm_max_offset, action.cfg.arm_max_speed = .1, .1, .2
        action.cfg.settle_seconds = 1.
        action._env.step_dt = .02
        action._env.episode_length_buf = torch.zeros(1, dtype=torch.long)
        probe = torch.full((1, 9), 100.)
        process(action, probe)
        torch.testing.assert_close(action.base_command, torch.zeros(1, 3))
        torch.testing.assert_close(action.arm_target, torch.zeros(1, 6))
        action._env.episode_length_buf[:] = 51
        process(action, probe)
        torch.testing.assert_close(action.arm_target, torch.full((1, 6), .004))
        process(action, probe)
        torch.testing.assert_close(action.previous_arm_target, torch.full((1, 6), .004))
        torch.testing.assert_close(action.arm_target, torch.full((1, 6), .008))

    def test_same_executed_command_has_larger_cost_for_excessive_raw_output(self):
        action = action_state()
        action.raw_actions = action._raw
        env = NS(action=action)
        moderate = torch.tensor([[-1., -1., 1.5, 0, 0, 0, 0, 0, 0]])
        process(action, moderate)
        small_cost = cost(env).clone()
        command = action.base_command.clone()
        action._raw.zero_()  # independent, identical zero previous-action state
        process(action, moderate * 100)
        torch.testing.assert_close(action.base_command, command)
        torch.testing.assert_close(cost(env), small_cost * 10000)

    def test_previous_action_uses_raw_space_and_does_not_alias_input(self):
        action = action_state()
        action.raw_actions = action._raw
        first = torch.full((1, 9), 2.)
        second = torch.full((1, 9), 3.)
        process(action, first)
        process(action, second)
        first.zero_()
        second.zero_()
        torch.testing.assert_close(action.previous_hl, torch.full((1, 9), 2.))
        torch.testing.assert_close(action.raw_actions, torch.full((1, 9), 3.))
        torch.testing.assert_close(cost(NS(action=action)), torch.tensor([49.5]))


if __name__ == "__main__":
    unittest.main()
