"""CPU checks for oracle observation, DAgger, supervised update, export/contracts."""
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import torch
from student_core import (OracleStudent, Replay, export_student, imitation_loss,
                          load_teacher_manifest, sha256, student_input, teacher_probability, freeze_teacher)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mujoco/deploy/deploy_mujoco/go2_piper"))
from transport_contract import validate_student_contract, load_stage2_contract

CONTRACT = dict(freeze_arm=False, command_limits=[.35, .35, .4], arm_scale=.1,
                arm_max_offset=.1, arm_max_speed=.2, control_dt=.02,
                settle_seconds=1., force_ramp_seconds=2., stop_hold_seconds=3., command_deadband=.05)


class StudentTests(unittest.TestCase):
    def test_jit_teacher_frozen_and_mujoco_cli_integration(self):
        teacher = freeze_teacher(torch.jit.script(torch.nn.Linear(147, 9)))
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ll = root / "ll.pt"
            with ll.open("wb") as stream:
                torch.jit.save(torch.jit.script(torch.nn.Linear(247, 12)), stream)
            policy = export_student(OracleStudent(), root / "exported", dict(
                task="GO2-PIPER-PAINT-Transport-Flat-Stage2", transport_action=CONTRACT,
                ll_policy_sha256=sha256(ll)))
            result = subprocess.run([sys.executable, str(ROOT / "mujoco/deploy/deploy_mujoco/go2_piper/paint_cooperate.py"),
                "--ll-policy", str(ll), "--hl-policy", str(policy), "--controller", "student-oracle",
                "--drive", "wrench", "--wrench-source", "external-share", "--max-speed", ".35",
                "--max-yaw-rate", ".4", "--duration", "30", "--require-stage2-contract", "--check-only"],
                cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("controller=student-oracle", result.stdout)

    def test_observation_uses_noisy_history_only_and_true_intent(self):
        teacher = torch.randn(4, 147)
        history = torch.randn(4, 144)
        x = student_input(dict(student=history, teacher=teacher))
        torch.testing.assert_close(x[:, :144], history)
        torch.testing.assert_close(x[:, -3:], teacher[:, -3:])
        teacher[:, :144] = 999
        torch.testing.assert_close(x, student_input(dict(student=history, teacher=teacher)))
        with self.assertRaises(ValueError):
            student_input(dict(student=history[:, :100], teacher=teacher))

    def test_replay_detaches_and_wraps(self):
        replay = Replay(5, "cpu")
        x = torch.ones(4, 147, requires_grad=True)
        replay.add(x, torch.zeros(4, 9))
        with torch.no_grad():
            x[:] = 100
        self.assertEqual(replay.x[:4].max(), 1)
        replay.add(torch.ones(4, 147) * 2, torch.ones(4, 9))
        self.assertEqual(replay.size, 5)
        self.assertFalse(replay.x.requires_grad)
        self.assertEqual(replay.sample(99)[0].shape, (5, 147))

    def test_teacher_schedule_reaches_student_only(self):
        self.assertEqual(teacher_probability(99, 100, 400), 1)
        self.assertGreater(teacher_probability(100, 100, 400), .99)
        self.assertEqual(teacher_probability(499, 100, 400), 0)
        self.assertEqual(teacher_probability(700, 100, 400), 0)

    def test_supervised_update_reduces_loss_and_export_matches(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        student = OracleStudent()
        x = torch.randn(32, 147)
        target = torch.tanh(x[:, :9]) * .1
        optimizer = torch.optim.Adam(student.parameters(), lr=.001)
        initial = imitation_loss(student(x), target, CONTRACT["command_limits"])[0].item()
        for _ in range(50):
            loss, _, _ = imitation_loss(student(x), target, CONTRACT["command_limits"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial * .05)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ll = root / "ll.pt"
            ll.write_bytes(b"frozen LL fixture")
            provenance = dict(task="GO2-PIPER-PAINT-Transport-Flat-Stage2",
                              transport_action=CONTRACT, ll_policy_sha256=sha256(ll))
            policy = export_student(student, root / "exported", provenance)
            with policy.open("rb") as stream:
                jit = torch.jit.load(stream)
            torch.testing.assert_close(jit(x), student(x))
            validate_student_contract([policy], [ll], "student-oracle", "external-share")
            self.assertEqual(load_stage2_contract([policy]), CONTRACT)
            for controller, wrench in (("teacher", "external-share"), ("student-oracle", "grasp")):
                with self.assertRaises(ValueError):
                    validate_student_contract([policy], [ll], controller, wrench)
            with self.assertRaises(FileExistsError):
                export_student(student, root / "exported", provenance)
            ll.write_bytes(b"wrong LL")
            with self.assertRaises(ValueError):
                validate_student_contract([policy], [ll], "student-oracle", "external-share")

    def test_teacher_source_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher, ll = root / "policy.pt", root / "ll.pt"
            teacher.write_bytes(b"teacher")
            ll.write_bytes(b"LL")
            manifest = dict(task="GO2-PIPER-PAINT-Transport-Flat-Stage2",
                            policy_sha256=sha256(teacher), ll_policy_path=str(ll), transport_action=CONTRACT)
            (root / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(load_teacher_manifest(teacher, ll), manifest)
            teacher.write_bytes(b"different teacher")
            with self.assertRaises(ValueError):
                load_teacher_manifest(teacher, ll)


if __name__ == "__main__":
    unittest.main()
