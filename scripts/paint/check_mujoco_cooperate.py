"""Real MuJoCo shared-payload physics/interface checks, without trained weights."""

from pathlib import Path
import sys
import tempfile
import io
import json
import csv
import subprocess
import hashlib
import unittest

import mujoco
import numpy as np
import torch
import yaml

ENTRY = Path(__file__).resolve().parents[2] / "mujoco/deploy/deploy_mujoco/go2_piper"
sys.path.insert(0, str(ENTRY))
from paint_cooperate import CooperativeDemo, TorchActor, grasp_reaction, policy_paths  # noqa: E402
from cooperative_control import DesktopCommand, WrenchFollower, slew_command, velocity_target  # noqa: E402
from cooperative_visuals import draw_guidance  # noqa: E402
from grasp_audit import GraspAudit, normalized_input, payload_balance  # noqa: E402


class CooperativeChecks(unittest.TestCase):
    def test_grasp_shadow_cli_writes_audit_with_teacher_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, inputs, outputs in (("ll", 247, 12), ("hl", 147, 9)):
                layer = torch.nn.Linear(inputs, outputs)
                with torch.no_grad():
                    layer.weight.zero_()
                    layer.bias.zero_()
                with (root / f"{name}.pt").open("wb") as stream:
                    torch.jit.save(torch.jit.trace(layer, torch.zeros(1, inputs)), stream)
            report = root / "audit.json"
            result = subprocess.run([sys.executable, str(ENTRY / "paint_cooperate.py"),
                                     "--ll-policy", str(root / "ll.pt"), "--controller", "teacher",
                                     "--hl-policy", str(root / "hl.pt"), "--freeze-arm",
                                     "--wrench-source", "external-share", "--audit-grasp",
                                     "--headless", "--duration", ".08", "--report", str(report)],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads(report.read_text())
            self.assertEqual(summary["wrench_source"], "external-share")
            self.assertEqual(summary["grasp_audit"]["static_samples"], [0, 0])
            with report.with_suffix(".grasp.csv").open(newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(len(rows), 9)  # four control steps, two robots
            self.assertTrue(all(len(row) == len(rows[0]) for row in rows))
            with report.with_suffix(".csv").open(newline="") as stream:
                trace = list(csv.DictReader(stream))
            self.assertTrue(all(float(row["reference_cx"]) == 0. for row in trace))

            # A Stage2 export must execute its bounded arm contract in the real CLI.
            from transport_contract import load_stage2_contract
            hl = torch.nn.Linear(147, 9)
            with torch.no_grad():
                hl.weight.zero_()
                hl.bias.zero_()
                hl.bias[3:] = 100.
            with (root / "hl.pt").open("wb") as stream:
                torch.jit.save(torch.jit.trace(hl, torch.zeros(1, 147)), stream)
            manifest = {"task": "GO2-PIPER-PAINT-Transport-Flat-Stage2",
                        "policy_sha256": hashlib.sha256((root / "hl.pt").read_bytes()).hexdigest(),
                        "transport_action": {"freeze_arm": False, "arm_scale": .1, "arm_max_offset": .1,
                            "arm_max_speed": .2, "control_dt": .02, "settle_seconds": 1.,
                            "force_ramp_seconds": 2., "stop_hold_seconds": 3.,
                            "command_limits": [.35, .35, .4], "command_deadband": .05}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertIsNotNone(load_stage2_contract([root / "hl.pt"]))
            stage2_report = root / "stage2.json"
            result = subprocess.run([sys.executable, str(ENTRY / "paint_cooperate.py"),
                "--ll-policy", str(root / "ll.pt"), "--controller", "teacher", "--hl-policy", str(root / "hl.pt"),
                "--require-stage2-contract", "--headless", "--duration", "9", "--force-period", "9", "--force-world", "0", "0", "0",
                "--report", str(stage2_report)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads(stage2_report.read_text())
            np.testing.assert_allclose(summary["max_arm_target_offset_rad"], .1, atol=1e-6)
            self.assertLessEqual(max(summary["max_arm_target_speed_rad_s"]), .20001)

    def test_grasp_audit_axes_scaling_and_missing_calibration(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        local, value = normalized_input(np.array([8., 0., -2., 0., 0., 2.]), rotation)
        np.testing.assert_allclose(local, [0., -8., -2., 0., 0., 2.])
        np.testing.assert_allclose(value, [0., -.2, .2])
        stream = io.StringIO()
        audit = GraspAudit(stream, 2, 3.)
        for time in (2., 2.5, 4.):
            audit.sample(time, 0, np.array([8., 0., -2., 0., 0., 2.]), rotation,
                         np.array([8., 0., 0., 0., 0., 2.]), np.zeros(6), np.zeros(6), np.zeros(6))
        result = audit.summary()
        self.assertEqual(result["static_samples"], [2, 0])
        self.assertIsNone(result["static_world_wrench_mean"][1])
        np.testing.assert_allclose(result["normalized_grasp_vs_external_share_rmse"][0], 0)
        self.assertEqual(len(stream.getvalue().splitlines()), 4)

    def test_recovered_grasp_balance_under_signed_loads_and_read_only(self):
        demo = CooperativeDemo(self.cfg)
        for _ in range(800):
            for robot in demo.robots:
                robot.write_torques()
            mujoco.mj_step(demo.model, demo.data)
        for force in ([8, 0, 0, 0, 0, 0], [-8, 0, 0, 0, 0, 0],
                      [0, 8, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]):
            demo.data.xfrc_applied[demo.payload] = force
            mujoco.mj_forward(demo.model, demo.data)
            before = [demo.data.qpos.copy(), demo.data.qvel.copy(), demo.data.ctrl.copy()]
            reactions = [grasp_reaction(demo.model, demo.data, r.eq, demo.payload) for r in demo.robots]
            _, residual, extraction = payload_balance(demo.model, demo.data, demo.payload, demo.robots, reactions)
            np.testing.assert_allclose(extraction, 0, atol=1e-9)
            np.testing.assert_allclose(residual, 0, atol=1e-7)
            # A wrong reaction sign must be detected, including under gravity.
            _, _, wrong = payload_balance(demo.model, demo.data, demo.payload, demo.robots, [-r for r in reactions])
            self.assertGreater(np.linalg.norm(wrong), 1.)
            for old, current in zip(before, [demo.data.qpos, demo.data.qvel, demo.data.ctrl]):
                np.testing.assert_array_equal(old, current)

    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load((ENTRY / "paint_loco.yaml").read_text(encoding="utf-8"))
        torch.set_num_threads(1)

    def test_active_transport_needs_no_payload_pull(self):
        demo = CooperativeDemo(self.cfg)
        for index, robot in enumerate(demo.robots):
            result = demo.reference_action(index, np.zeros(6), .4, .35, .4,
                                           world_command=np.array([.3, 0, 0]), heading_target=0.)
            np.testing.assert_allclose(result, [.3, 0, 0, 0, 0, 0, 0, 0, 0], atol=1e-7)
        np.testing.assert_allclose(demo.data.xfrc_applied, 0)

    def test_local_force_follower_bias_direction_release_and_independence(self):
        follower, other = WrenchFollower(), WrenchFollower()
        rotation = np.eye(3)
        static = np.array([1., 0., -2.45, 0., .6, 0.])
        for _ in range(50):
            np.testing.assert_allclose(follower.update(static, rotation, .02, calibrating=True), 0)
        np.testing.assert_allclose(follower.update(static, rotation, .02), 0)
        for _ in range(150):
            output = follower.update(static + [8., 0., 0., 0., 0., 0.], rotation, .02)
        self.assertAlmostEqual(output[0], .2, places=4)
        np.testing.assert_allclose(other.command, 0)
        for _ in range(150):
            output = follower.update(static, rotation, .02)
        self.assertLess(np.linalg.norm(output), .001)
        quarter_turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        follower = WrenchFollower()
        for _ in range(150):
            output = follower.update([8, 0, 0, 0, 0, 0], quarter_turn, .02)
        self.assertAlmostEqual(output[1], -.2, places=4)

    def test_force_keyboard_and_non_physical_arrow(self):
        keyboard = DesktopCommand([16, 0, 0], 80, 10, linear_step=2, yaw_step=.5, label="force")
        keyboard.on_key(ord("W"))
        keyboard.on_key(ord("Q"))
        np.testing.assert_allclose(keyboard.snapshot()[0], [18, 0, .5])
        keyboard.on_key(ord("X"))
        np.testing.assert_allclose(keyboard.snapshot()[0], 0)
        demo = CooperativeDemo(self.cfg)
        qpos_before = demo.data.qpos.copy()
        scene = mujoco.MjvScene(demo.model, maxgeom=32)
        draw_guidance(scene, demo.data.xpos[demo.payload], [16, 0, 0, 0, 0, 1], 2., "admittance")
        self.assertGreater(scene.ngeom, 2)
        np.testing.assert_allclose(demo.data.qpos, qpos_before)
        np.testing.assert_allclose(demo.data.xfrc_applied, 0)

    def test_motion_ramp_braking_and_slew(self):
        command = [.3, 0, 0]
        np.testing.assert_allclose(velocity_target(command, 0, 60, [0, 0, 0], [0, 0, 0]), 0)
        np.testing.assert_allclose(velocity_target(command, 3.5, 60, [0, 0, 0], [0, 0, 0]), [.15, 0, 0])
        np.testing.assert_allclose(velocity_target(command, 10, 60, [0, 0, 0], [0, 0, 0], distance=8), command)
        near = velocity_target(command, 30, 60, [7.9, .1, 0], [0, 0, 0], distance=8)
        np.testing.assert_allclose(near, [.06, -.05, 0], atol=1e-7)
        np.testing.assert_allclose(velocity_target(command, 30, 60, [8, 0, 0], [0, 0, 0], distance=8), 0)
        value = slew_command(np.zeros(3), np.array([.3, .3, 1.]), .02)
        self.assertAlmostEqual(np.linalg.norm(value[:2]), .003)
        self.assertAlmostEqual(value[2], .008)

    def test_keyboard_stop_resume_pause_and_limits(self):
        keyboard = DesktopCommand([.3, 0, 0], .35, .4)
        for _ in range(3):
            keyboard.on_key(ord("W"))
        self.assertAlmostEqual(keyboard.snapshot()[0][0], .35)
        keyboard.on_key(ord("X"))
        np.testing.assert_allclose(keyboard.snapshot()[0], 0)
        keyboard.on_key(ord("R"))
        np.testing.assert_allclose(keyboard.snapshot()[0], [.3, 0, 0])
        keyboard.on_key(ord("P"))
        self.assertTrue(keyboard.snapshot()[1])
        keyboard.on_key(ord("P"))
        self.assertFalse(keyboard.snapshot()[1])

    def test_two_and_three_robot_names_grasps_and_free_payload(self):
        for count in (2, 3):
            demo = CooperativeDemo(self.cfg, count=count)
            self.assertEqual(demo.model.nu, count * 18)
            self.assertEqual(demo.model.neq, count)
            self.assertEqual(np.count_nonzero(demo.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE), count + 1)
            for robot in demo.robots:
                self.assertEqual(robot.observation(np.zeros(3)).shape, (247,))
                self.assertEqual(robot.teacher_observation(np.zeros(3)).shape, (147,))
            self.assertFalse(set(demo.robots[0].qpos_ids) & set(demo.robots[1].qpos_ids))
            self.assertTrue(all(value["grasp_error_m"] < 1e-6 for value in demo.status()))
            eq_rows = demo.data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
            self.assertLess(np.max(np.abs(demo.data.efc_pos[eq_rows])), 1e-6)
            self.assertTrue(np.all(demo.model.eq_obj1id > 0))  # no world weld
            self.assertEqual(demo.data.ncon, 0)  # no starting handle/finger penetration

    def test_independent_arm_leg_history_and_wrench_mapping(self):
        demo = CooperativeDemo(self.cfg)
        first, second = demo.robots
        first.set_transport_action(np.ones(12), np.full(6, .2))
        np.testing.assert_allclose(first.previous_action, 1)
        np.testing.assert_allclose(second.previous_action, 0)
        first.push_history()
        np.testing.assert_allclose(first.history[-1, 12:], first.target[12:])
        np.testing.assert_allclose(second.history, 0)
        np.testing.assert_allclose(first.target[12:], np.clip(first.default[12:] + .1,
            demo.model.jnt_range[first.joint_ids[12:]].mean(1) - np.diff(demo.model.jnt_range[first.joint_ids[12:]], axis=1).ravel() * .45,
            demo.model.jnt_range[first.joint_ids[12:]].mean(1) + np.diff(demo.model.jnt_range[first.joint_ids[12:]], axis=1).ravel() * .45))
        wrench = np.array([8., 0, 0, 0, 0, 2.])
        np.testing.assert_allclose(demo.normalized_wrench(first, wrench, "external-share"), [.1, 0, .1])

    def test_physical_weight_sharing_and_grasp_release(self):
        demo = CooperativeDemo(self.cfg)
        for _ in range(1000):
            for robot in demo.robots:
                robot.write_torques()
            mujoco.mj_step(demo.model, demo.data)
        mujoco.mj_forward(demo.model, demo.data)
        self.assertTrue(np.isfinite(demo.data.qpos).all())
        self.assertTrue(all(s["grasp_error_m"] < .005 for s in demo.status()))
        reactions = np.array([grasp_reaction(demo.model, demo.data, r.eq, demo.payload) for r in demo.robots])
        self.assertAlmostEqual(float(reactions[:, 2].sum()), -.5 * 9.81, delta=.15)
        before = float(demo.data.xpos[demo.payload, 2])
        demo.data.eq_active[:] = False
        for _ in range(40):
            for robot in demo.robots:
                robot.write_torques()
            mujoco.mj_step(demo.model, demo.data)
        mujoco.mj_forward(demo.model, demo.data)
        self.assertLess(demo.data.xpos[demo.payload, 2], before - .1)

    def test_replaceable_torchscript_contract_rejects_wrong_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            layer = torch.nn.Linear(247, 12)
            with torch.no_grad():
                layer.weight.zero_()
                layer.bias.zero_()
            path = Path(directory) / "ll.pt"
            with path.open("wb") as stream:
                torch.jit.save(torch.jit.trace(layer, torch.zeros(1, 247)), stream)
            policy = TorchActor(path, 247, 12)
            np.testing.assert_allclose(policy(np.ones(247)), 0)
            with self.assertRaises((ValueError, RuntimeError)):
                TorchActor(path, 147, 9)
        self.assertEqual(policy_paths(["same"], 3, "LL"), ["same"] * 3)
        with self.assertRaises(ValueError):
            policy_paths(["a", "b"], 3, "LL")


if __name__ == "__main__":
    unittest.main()
