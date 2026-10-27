"""Audit invariants with real MuJoCo, USD reference and explicit snapshot replay."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "mujoco/deploy/deploy_mujoco/go2_piper"
sys.path.insert(0, str(ENTRY))
from paint_loco import PaintBridge, JOINT_NAMES
from physics_audit import describe_model
from audit_mujoco_interface import compare_policies, replay_snapshot

path = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/transport_math.py"
spec = importlib.util.spec_from_file_location("consistency_math", path)
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)


class ConsistencyChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load((ENTRY / "paint_loco.yaml").read_text())
        torch.set_num_threads(1)

    def test_compiled_effort_overrides_and_opt_in_profiles(self):
        legacy = PaintBridge(self.cfg.copy())
        original = describe_model(legacy)
        self.assertEqual(original["effort_limits_Nm"][:12], [23.7, 23.7, 45.43] * 4)
        self.assertEqual(original["effort_limits_Nm"][12:], [20., 20., 15., 7., 5., 5.])
        self.assertEqual(original["feet"]["FR"]["condim"], 6)
        contact = PaintBridge({**self.cfg, "physics_profile": "point-contact"})
        changed = describe_model(contact)
        self.assertTrue(all(v["condim"] == 3 for v in changed["feet"].values()))
        self.assertEqual(changed["effort_limits_Nm"], original["effort_limits_Nm"])
        np.testing.assert_array_equal(contact.model.body_mass, legacy.model.body_mass)
        arm = PaintBridge({**self.cfg, "physics_profile": "arm-effort-100"})
        self.assertEqual(describe_model(arm)["effort_limits_Nm"][12:], [100.] * 6)
        self.assertTrue(all(v["condim"] == 6 for v in describe_model(arm)["feet"].values()))

    def test_contacts_actually_use_three_or_six_constraint_dimensions(self):
        for profile, expected in (("legacy", 6), ("point-contact", 3)):
            bridge = PaintBridge({**self.cfg, "physics_profile": profile})
            for _ in range(150):
                bridge.physics_step()
            feet = {mujoco.mj_name2id(bridge.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("FR", "FL", "RR", "RL")}
            contacts = [c for c in bridge.data.contact if c.geom1 in feet or c.geom2 in feet]
            self.assertTrue(contacts)
            self.assertTrue(all(c.dim == expected for c in contacts))

    def test_live_snapshot_replay_in_rotated_moving_state(self):
        source = PaintBridge(self.cfg.copy())
        q = source.model.jnt_qposadr[source.model.body_jntadr[source.base]]
        v = source.model.jnt_dofadr[source.model.body_jntadr[source.base]]
        quat = np.array([.8, .1, -.2, .3]); quat /= np.linalg.norm(quat)
        source.data.qpos[q:q + 7] = [.2, -.4, .6, *quat]
        source.data.qpos[source.qpos_ids] += .01
        source.data.qvel[v:v + 6] = [.3, -.2, .1, .1, -.2, .3]
        source.data.qvel[source.qvel_ids] = np.linspace(-.2, .2, 18)
        mujoco.mj_forward(source.model, source.data)
        linear_jac, angular_jac = np.zeros((3, source.model.nv)), np.zeros((3, source.model.nv))
        mujoco.mj_jacBodyCom(source.model, source.data, linear_jac, angular_jac, source.base)
        rotation = source.data.xmat[source.base].reshape(3, 3)
        linear = rotation.T @ (linear_jac @ source.data.qvel)
        angular = rotation.T @ (angular_jac @ source.data.qvel)
        gravity = rotation.T @ np.array([0., 0., -1.])
        tensor = lambda value: torch.tensor(np.asarray(value)[None], dtype=torch.float32)
        command = np.array([.1, -.1, .3])
        previous = np.linspace(-.3, .3, 12)
        obs = math.locomotion_observation(tensor(linear), tensor(angular), tensor(gravity), tensor(command),
            tensor(source.data.qpos[source.qpos_ids] - source.default), tensor(source.data.qvel[source.qvel_ids]),
            tensor(previous), torch.full((1, 187), .1)).numpy()[0]
        snapshot = {"joint_names": JOINT_NAMES, "observation": obs.tolist(), "root_pos_w": [.2, -.4, .6],
            "root_quat_wxyz": quat.tolist(), "root_lin_vel_b": linear.tolist(), "root_ang_vel_b": angular.tolist(),
            "joint_pos": source.data.qpos[source.qpos_ids].tolist(), "joint_vel": source.data.qvel[source.qvel_ids].tolist(),
            "default_joint_pos": source.default.tolist(), "command": command.tolist()}
        replay = replay_snapshot(PaintBridge(self.cfg.copy()), snapshot, None)
        self.assertLess(max(replay["observation_block_max_abs_difference"].values()), 1e-6)

    def test_policy_tensor_comparison_detects_different_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.pt", Path(directory) / "b.pt"
            layer = torch.nn.Linear(247, 12)
            def save(path):
                with path.open("wb") as stream:
                    torch.jit.save(torch.jit.trace(layer, torch.zeros(1, 247)), stream)
            save(a); save(b)
            result = compare_policies(a, b)
            self.assertTrue(result["state_dict_exactly_equal"])
            self.assertEqual(result["max_abs_output_difference"], 0.)
            with torch.no_grad():
                layer.bias += 1.
            save(b)
            self.assertFalse(compare_policies(a, b)["state_dict_exactly_equal"])

    def test_reference_records_source_motor_and_mass_values(self):
        reference = json.loads((ENTRY / "isaac_asset_reference.json").read_text())
        self.assertAlmostEqual(reference["authored_positive_mass_sum_kg"], 20.765, places=4)
        self.assertAlmostEqual(reference["joints"]["FR_calf_joint"]["drive_max_force_Nm"], 45.43, places=4)
        self.assertEqual(reference["joints"]["joint1"]["drive_max_force_Nm"], 100.)


if __name__ == "__main__":
    unittest.main()
