"""Standard-library tests for orchestration; never connects to a server or GPU."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("server_train", Path(__file__).with_name("server_train.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class LauncherTests(unittest.TestCase):
    def test_pipeline_order_and_exact_checkpoint_handoff(self):
        launcher = module.Launcher(module.parse_args(["pipeline", "--dry-run"]))
        with patch.object(launcher, "preflight"), patch.object(launcher, "run") as run:
            launcher.execute()
        labels = [call.args[0] for call in run.call_args_list]
        self.assertEqual(labels, ["install_project", "interface", "smoke_flat", "ppo_smoke_flat",
                                  "smoke_rough", "ppo_smoke_rough", "train_flat", "train_rough"])
        rough_args = run.call_args_list[-1].args
        self.assertIn("--resume", rough_args)
        self.assertIn("checkpoint.pt", rough_args)

    def test_smoke_failure_stops_before_large_training(self):
        launcher = module.Launcher(module.parse_args(["pipeline", "--dry-run"]))
        def fail_on_smoke(label, *args):
            if label == "smoke_flat":
                raise RuntimeError("mock smoke failure")
        with patch.object(launcher, "preflight"), patch.object(launcher, "run", side_effect=fail_on_smoke) as run:
            with self.assertRaises(RuntimeError):
                launcher.execute()
        self.assertNotIn("train_flat", [call.args[0] for call in run.call_args_list])

    def test_checkpoint_selection_is_numeric_and_run_specific(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(module, "ROOT", Path(directory)):
            run = Path(directory) / "logs/rsl_rl/experiment/2026_train_flat_unique"
            run.mkdir(parents=True)
            for name in ("model_9.pt", "model_100.pt", "model_20.pt"):
                (run / name).touch()
            other = run.parent / "2027_other_run"
            other.mkdir()
            (other / "model_99999.pt").touch()
            launcher = module.Launcher(module.parse_args(["pipeline", "--dry-run"]))
            self.assertEqual(launcher.latest_checkpoint("experiment", "train_flat_unique"), run / "model_100.pt")

    def test_resume_stages_exact_file_without_changing_source(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(module, "ROOT", Path(directory)):
            source = Path(directory) / "original.pt"
            source.write_bytes(b"test checkpoint")
            launcher = module.Launcher(module.parse_args(["rough", "--device", "cuda:1"]))
            with patch.object(launcher, "run") as run:
                launcher.train("rough", checkpoint=source)
            copies = list(Path(directory).glob("logs/rsl_rl/*/resume_*/checkpoint.pt"))
            self.assertEqual(len(copies), 1)
            self.assertEqual(copies[0].read_bytes(), source.read_bytes())
            self.assertIn("agent.device=cuda:1", run.call_args.args)


if __name__ == "__main__":
    unittest.main()
