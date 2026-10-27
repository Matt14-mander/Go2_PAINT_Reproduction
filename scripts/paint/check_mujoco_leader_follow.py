"""Real tow-connection/controller checks; fixture actors do NOT validate locomotion."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np
import torch
import yaml

ENTRY = Path(__file__).resolve().parents[2] / "mujoco/deploy/deploy_mujoco/go2_piper"
sys.path.insert(0, str(ENTRY))
from paint_cooperate import CooperativeDemo, grasp_reaction  # noqa: E402
from cooperative_control import WrenchFollower, leader_action, velocity_target, slew_command  # noqa: E402
from cooperative_visuals import draw_leader_follow  # noqa: E402
from grasp_audit import payload_balance  # noqa: E402


class LeaderFollowChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = yaml.safe_load((ENTRY / "paint_loco.yaml").read_text(encoding="utf-8"))
        torch.set_num_threads(1)

    def scene(self):
        return CooperativeDemo(self.cfg, layout="leader-follow")

    def test_tow_geometry_anchors_independent_roots_and_collisions(self):
        demo = self.scene()
        model, data = demo.model, demo.data
        leader, follower = demo.robots
        self.assertGreater(data.xpos[leader.base, 0], data.xpos[follower.base, 0])
        np.testing.assert_allclose(data.xpos[[leader.base, follower.base], 1], 0)
        self.assertEqual(model.nu, 36)
        self.assertEqual(np.count_nonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE), 3)
        self.assertEqual(model.eq_type[leader.eq], mujoco.mjtEq.mjEQ_CONNECT)
        self.assertEqual(model.eq_type[follower.eq], mujoco.mjtEq.mjEQ_WELD)
        self.assertTrue(np.all(model.eq_obj1id > 0))
        self.assertTrue(np.all(model.eq_obj2id > 0))
        self.assertGreater(model.body_mass[demo.payload], 0)
        self.assertTrue(np.all(model.body_inertia[demo.payload] > 0))
        self.assertEqual(data.ncon, 0)
        self.assertTrue(all(s["grasp_error_m"] < 1e-9 for s in demo.status()))
        equality = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        self.assertEqual(np.count_nonzero(equality), 9)
        np.testing.assert_allclose(data.efc_pos[equality], 0, atol=1e-9)
        self.assertFalse(set(leader.qpos_ids) & set(follower.qpos_ids))
        leader.set_transport_action(np.ones(12), np.zeros(6))
        leader.push_history()
        np.testing.assert_allclose(follower.previous_action, 0)
        np.testing.assert_allclose(follower.history, 0)
        bar = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload/tow_bar")
        # Robot/bar collision masks still permit collisions away from the joint.
        for robot in demo.robots:
            base_geoms = np.flatnonzero((model.geom_bodyid == robot.base) & (model.geom_contype != 0))
            self.assertTrue(any(model.geom_contype[bar] & model.geom_conaffinity[g] for g in base_geoms))
        with self.assertRaises(ValueError):
            CooperativeDemo(self.cfg, count=3, layout="leader-follow")

    def test_physical_load_transfer_balance_and_hitch_release(self):
        demo = self.scene()
        model, data = demo.model, demo.data
        leader, follower = demo.robots
        for _ in range(1000):
            for robot in demo.robots:
                robot.write_torques()
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        self.assertTrue(all(s["grasp_error_m"] < .005 and s["tilt_deg"] < 5 for s in demo.status()))
        static = [grasp_reaction(model, data, r.eq, demo.payload) for r in demo.robots]
        self.assertAlmostEqual(sum(r[2] for r in static), -.5 * 9.81, delta=.2)
        for sign in (1, -1):
            # Test-only disturbance on the LEADER, never an operating drive force.
            data.xfrc_applied[leader.base, 0] = sign * 20
            mujoco.mj_forward(model, data)
            reactions = [grasp_reaction(model, data, r.eq, demo.payload) for r in demo.robots]
            self.assertGreater(sign * (reactions[1][0] - static[1][0]), .5)
            _, residual, extraction = payload_balance(model, data, demo.payload, demo.robots, reactions)
            np.testing.assert_allclose(residual, 0, atol=1e-7)
            np.testing.assert_allclose(extraction, 0, atol=1e-8)
        data.eq_active[leader.eq] = False
        data.xfrc_applied[:] = 0
        mujoco.mj_forward(model, data)
        released = grasp_reaction(model, data, follower.eq, demo.payload)
        data.xfrc_applied[leader.base, 0] = 20
        mujoco.mj_forward(model, data)
        # With the hitch released, changing leader load cannot reach follower.
        np.testing.assert_allclose(grasp_reaction(model, data, follower.eq, demo.payload), released, atol=1e-8)

    def test_role_separation_and_bounded_follower_recovery(self):
        follower = WrenchFollower(command_limits=[.2, .2, .4])
        for _ in range(50):
            follower.update(np.zeros(6), np.eye(3), .02, calibrating=True)
        for speed in (.1, .2, -.1):
            action = leader_action([speed, 0, 0], np.eye(3), 0.)
            self.assertAlmostEqual(action[0], speed)
            np.testing.assert_allclose(follower.update(np.zeros(6), np.eye(3), .02), 0)
        for _ in range(200):
            out = follower.update([400, 0, 0, 0, 0, 0], np.eye(3), .02)
        self.assertAlmostEqual(out[0], .2)
        for _ in range(200):
            out = follower.update(np.zeros(6), np.eye(3), .02)
        self.assertLess(np.linalg.norm(out), .001)
        quarter = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        np.testing.assert_allclose(leader_action([.1, 0, .1], quarter, np.pi / 2)[:3], [0, -.1, .1], atol=1e-9)

    def test_profile_includes_stationary_tail_and_visuals_are_read_only(self):
        command = np.zeros(3)
        for step in range(1000):
            time = step * .02
            target = velocity_target([.1, 0, 0], time, 17., [0, 0, 0], [0, 0, 0], 3., 3.)
            command = slew_command(command, target, .02)
            if time >= 17:
                np.testing.assert_allclose(command, 0, atol=1e-9)
        demo = self.scene()
        before = demo.data.qpos.copy()
        scene = mujoco.MjvScene(demo.model, maxgeom=64)
        draw_leader_follow(scene, [demo.data.xpos[r.base] for r in demo.robots],
                           [[.1, 0, 0], [.08, 0, 0]], [4, 0, 0, 0, 0, 0], 1.)
        labels = [scene.geoms[i].label for i in range(scene.ngeom)]
        self.assertTrue(any("LEADER" in label for label in labels))
        self.assertTrue(any("FOLLOWER FT" in label for label in labels))
        np.testing.assert_array_equal(demo.data.qpos, before)
        np.testing.assert_allclose(demo.data.xfrc_applied, 0)

    def test_real_cli_fixture_reports_roles_zero_external_force_and_hold_ablation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = torch.nn.Linear(247, 12)
            with torch.no_grad():
                actor.weight.zero_()
                actor.bias.zero_()
            policy = root / "fixture.pt"
            with policy.open("wb") as stream:
                torch.jit.save(torch.jit.trace(actor, torch.zeros(1, 247)), stream)
            for hold in (False, True):
                report = root / f"fixture_{hold}.json"
                args = [sys.executable, str(ENTRY / "paint_cooperate.py"), "--ll-policy", str(policy),
                        "--drive", "leader-follow", "--headless", "--duration", "5",
                        "--settle-seconds", "1", "--ramp-seconds", "1", "--stop-hold-seconds", "1",
                        "--audit-grasp", "--report", str(report)]
                if hold:
                    args.extend(["--follower-hold", "--travel-distance", ".3"])
                result = subprocess.run(args, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                summary = json.loads(report.read_text())
                self.assertEqual(summary["roles"], ["leader", "follower"])
                self.assertEqual(summary["wrench_source"], "grasp")
                self.assertEqual(summary["follower_hold"], hold)
                self.assertTrue(summary["freeze_arm"])
                self.assertEqual(summary["role_controllers"],
                                 ["locomotion_velocity", "hold" if hold else "local_FT_admittance"])
                self.assertEqual(summary["reason"], "completed")
                np.testing.assert_allclose(summary["force_world_Fx_Fy_Tz"], 0)
                self.assertLess(max(summary["max_abs_payload_balance_generalized"]), 1e-7)
                self.assertEqual(summary["control_steps"], 250)
                with report.with_suffix(".csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 500)
                leaders = [r for r in rows if r["role"] == "leader"]
                followers = [r for r in rows if r["role"] == "follower"]
                self.assertGreater(max(float(r["command_vx"]) for r in leaders), .09)
                for row in rows:
                    self.assertEqual(float(row["payload_external_Fx"]), 0)
                for row in leaders:
                    if float(row["time_s"]) >= 4:
                        self.assertEqual(float(row["command_vx"]), 0)
                if hold:
                    self.assertTrue(all(float(r["command_vx"]) == 0 for r in followers))
                    self.assertEqual(summary["goal_reference"], "leader_base")
                    np.testing.assert_allclose(summary["goal_position_world_xy_m"], [.9, 0])
                    leader_xy = np.asarray(summary["robot_displacement_world_m"])[0, :2] + [.6, 0]
                    self.assertAlmostEqual(summary["goal_error_m"], np.linalg.norm(leader_xy - [.9, 0]))
            # Reject combinations that would silently turn this into externally
            # pulled or globally commanded cooperation, before touching a policy.
            for extra in (("--force-world", "8", "0", "0"), ("--wrench-source", "external-share"),
                          ("--num-robots", "3"), ("--controller", "reference")):
                result = subprocess.run([sys.executable, str(ENTRY / "paint_cooperate.py"),
                    "--ll-policy", str(policy), "--drive", "leader-follow", "--headless", *extra],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 2)
                self.assertIn("Leader-follow requires", result.stderr)


if __name__ == "__main__":
    unittest.main()
