"""Controlled evaluation config checks without Isaac; never change saved schemas."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

path = Path(__file__).resolve().parents[1] / "rsl_rl/eval_options.py"
spec = importlib.util.spec_from_file_location("eval_options", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def configuration():
    return NS(commands=NS(base_velocity=NS(
        ranges=NS(lin_vel_x=(-1, 1), lin_vel_y=(-1, 1), ang_vel_z=(-1.5, 1.5)),
        limit_ranges=NS(lin_vel_x=(-1, 1), lin_vel_y=(-1, 1), ang_vel_z=(-1.5, 1.5)),
        heading_command=False, rel_standing_envs=0.1, curriculum_enabled=False)),
        curriculum=NS(lin_vel_cmd_levels=None, ang_vel_cmd_levels=None),
        events=NS(payload_mass=NS(params={"mass_range": (0, 10)}),
                  push_robot="push", payload_wrench_interval="wrench", physics_material="material",
                  add_mass="mass", base_com="com", randomize_rigid_body_inertia="inertia",
                  randomize_actuator_gains="gains",
                  reset_robot_joints=NS(params={"position_range": (0.8, 1.2)}),
                  reset_arm_joints=NS(params={"position_range": (1, 1)}),
                  reset_base=NS(params={"pose_range": {"x": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                                        "velocity_range": {"x": (0, 0)}})),
        observations=NS(policy=NS(enable_corruption=False, joint_pos_width=18, joint_vel_width=18)),
        actions=NS(joint_pos=NS(action_dim=12)),
        scene=NS(robot=NS(actuators={"legs": NS(min_delay=0, max_delay=4)})))


def arguments(**kwargs):
    result = NS(task="GO2-PIPER-PAINT-Loco-Flat-Play", base_command=None, payload_mass=None,
                no_disturbances=False, no_domain_randomization=False, eval_name="play")
    vars(result).update(kwargs)
    return result


class EvaluationTests(unittest.TestCase):
    def test_transport_nominal_preserves_teacher_schema_and_interaction(self):
        cfg = configuration()
        cfg.wrench_scale, cfg.periodic_disturbance = .2, True
        cfg.events.payload_material = "material"
        cfg.scene = NS(robot=NS(actuators={"arm": NS(min_delay=0, max_delay=4)}))
        before = copy.deepcopy(cfg.observations)
        module.configure_evaluation(cfg, arguments(task="GO2-PIPER-PAINT-Transport-Flat",
                                                  payload_mass=.5, no_domain_randomization=True))
        self.assertEqual(cfg.wrench_scale, .2)
        self.assertTrue(cfg.periodic_disturbance)
        self.assertIsNone(cfg.events.payload_material)
        self.assertEqual(cfg.scene.robot.actuators["arm"].max_delay, 0)
        self.assertEqual(cfg.events.reset_base.params["pose_range"]["x"], (0, 0))
        self.assertEqual(cfg.events.payload_mass.params["mass_range"], (.5, .5))
        self.assertEqual(cfg.observations, before)

    def test_transport_zero_wrench_and_fixed_base_command_rejection(self):
        cfg = configuration()
        cfg.wrench_scale, cfg.periodic_disturbance = .2, True
        module.configure_evaluation(cfg, arguments(task="GO2-PIPER-PAINT-Transport-Flat", no_disturbances=True))
        self.assertEqual(cfg.wrench_scale, 0)
        self.assertFalse(cfg.periodic_disturbance)
        with self.assertRaises(ValueError):
            module.configure_evaluation(cfg, arguments(task="GO2-PIPER-PAINT-Transport-Flat", base_command=[0, 0, 0]))

    def test_default_keeps_random_evaluation(self):
        cfg = configuration()
        before = copy.deepcopy(cfg)
        module.configure_evaluation(cfg, arguments())
        self.assertEqual(cfg, before)

    def test_fixed_forward_is_not_randomly_replaced_by_standing(self):
        cfg = configuration()
        module.configure_evaluation(cfg, arguments(base_command=[0.3, 0, 0]))
        self.assertEqual(cfg.commands.base_velocity.rel_standing_envs, 0)
        for ranges in (cfg.commands.base_velocity.ranges, cfg.commands.base_velocity.limit_ranges):
            self.assertEqual(ranges.lin_vel_x, (0.3, 0.3))
            self.assertEqual(ranges.lin_vel_y, (0, 0))
            self.assertEqual(ranges.ang_vel_z, (0, 0))

    def test_nominal_unloaded_standing_keeps_checkpoint_schema(self):
        cfg = configuration()
        observation_before = copy.deepcopy(cfg.observations)
        action_before = copy.deepcopy(cfg.actions)
        module.configure_evaluation(cfg, arguments(base_command=[0, 0, 0], payload_mass=0,
                                                  no_disturbances=True, no_domain_randomization=True))
        self.assertEqual(cfg.events.payload_mass.params["mass_range"], (0, 0))
        self.assertIsNone(cfg.events.push_robot)
        self.assertIsNone(cfg.events.payload_wrench_interval)
        self.assertIsNone(cfg.events.randomize_actuator_gains)
        self.assertEqual(cfg.events.reset_base.params["pose_range"]["yaw"], (0, 0))
        self.assertEqual(cfg.observations, observation_before)
        self.assertEqual(cfg.actions, action_before)
        self.assertEqual(cfg.scene.robot.actuators["legs"].max_delay, 0)

    def test_invalid_inputs_are_rejected(self):
        for values in ({"payload_mass": -1}, {"payload_mass": float("nan")},
                       {"base_command": [2, 0, 0]}, {"base_command": [0, 0, float("inf")]},
                       {"eval_name": "../weights"}):
            with self.assertRaises(ValueError):
                module.configure_evaluation(configuration(), arguments(**values))


if __name__ == "__main__":
    unittest.main()
