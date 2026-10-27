"""Real MuJoCo bridge checks, with project XML/meshes; no trained policy needed."""

import importlib.util
from pathlib import Path
import unittest
import sys

import mujoco
import numpy as np
import yaml

PATH = Path(__file__).resolve().parents[2] / "mujoco/deploy/deploy_mujoco/go2_piper/paint_loco.py"
sys.path.insert(0, str(PATH.parent))
spec = importlib.util.spec_from_file_location("paint_mujoco", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load(PATH.with_suffix(".yaml").read_text(encoding="utf-8"))

    def setUp(self):
        self.bridge = module.PaintBridge(self.cfg)

    def test_nominal_observation_and_name_mapping(self):
        bridge = self.bridge
        observation = bridge.observation(np.zeros(3))
        self.assertEqual(observation.shape, (247,))
        np.testing.assert_allclose(observation[6:9], [0, 0, -1])
        np.testing.assert_allclose(observation[12:60], 0, atol=1e-7)
        np.testing.assert_allclose(observation[60:], -0.15, atol=1e-6)
        for index, name in enumerate(module.JOINT_NAMES):
            self.assertEqual(mujoco.mj_id2name(bridge.model, mujoco.mjtObj.mjOBJ_JOINT,
                                              int(bridge.joint_ids[index])), name)
        np.testing.assert_allclose(bridge.data.xpos[bridge.base], [0, 0, 0.35], atol=1e-6)

    def test_action_clip_and_fixed_arm_target(self):
        bridge = self.bridge
        bridge.set_action(np.full(12, 10.0))
        np.testing.assert_allclose(bridge.target[:12], bridge.default[:12] + 1.0)
        np.testing.assert_allclose(bridge.target[12:], bridge.default[12:])
        np.testing.assert_allclose(bridge.observation(np.zeros(3))[48:60], 4.0)
        with self.assertRaises(ValueError):
            bridge.set_action(np.zeros(18))

    def test_world_velocity_rotates_into_regular_base_axes(self):
        bridge = self.bridge
        free = np.where(bridge.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0][0]
        qpos = bridge.model.jnt_qposadr[free]
        qvel = bridge.model.jnt_dofadr[free]
        bridge.data.qpos[qpos + 3:qpos + 7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        bridge.data.qvel[qvel:qvel + 6] = [1, 0, 0, 0, 0, 0]
        mujoco.mj_forward(bridge.model, bridge.data)
        linear, angular, gravity = bridge.base_state()
        np.testing.assert_allclose(linear, [0, -1, 0], atol=1e-6)
        np.testing.assert_allclose(angular, 0, atol=1e-6)
        np.testing.assert_allclose(gravity, [0, 0, -1], atol=1e-6)

    def test_physics_steps_are_finite_and_controls_bounded(self):
        bridge = self.bridge
        for _ in range(100):
            bridge.physics_step()
        self.assertTrue(np.isfinite(bridge.data.qpos).all())
        self.assertAlmostEqual(bridge.data.time, 0.5)
        ids = bridge.actuator_ids
        self.assertTrue(np.all(bridge.data.ctrl[ids] <= bridge.model.actuator_ctrlrange[ids, 1] + 1e-8))
        self.assertTrue(np.all(bridge.data.ctrl[ids] >= bridge.model.actuator_ctrlrange[ids, 0] - 1e-8))


if __name__ == "__main__":
    unittest.main()
