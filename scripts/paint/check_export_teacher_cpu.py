"""CPU exporter equivalence and incompatible-checkpoint rejection checks."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

path = Path(__file__).with_name("export_teacher_cpu.py")
spec = importlib.util.spec_from_file_location("teacher_export", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def checkpoint(actor):
    return {"model_state_dict": {"actor." + key: value for key, value in actor.state_dict().items()},
            "optimizer_state_dict": {}, "iter": 9999, "infos": None}


class ExportChecks(unittest.TestCase):
    def test_serialized_actor_matches_checkpoint_and_manifest(self):
        actor = nn.Sequential(nn.Linear(147, 16), nn.ELU(), nn.Linear(16, 9)).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            params = root / "params"
            params.mkdir()
            (params / "agent.yaml").write_text("policy:\n  activation: elu\n  actor_obs_normalization: false\n", encoding="utf-8")
            source = root / "model_9999.pt"
            with source.open("wb") as stream:
                torch.save(checkpoint(actor), stream)
            output = root / "exported/model_9999"
            manifest = module.export(source, output)
            with (output / "policy.pt").open("rb") as stream:
                policy = torch.jit.load(stream)
            probe = torch.randn(8, 147)
            torch.testing.assert_close(policy(probe), actor(probe))
            self.assertEqual(manifest["checkpoint_iteration"], 9999)
            self.assertEqual(manifest["policy_sha256"], module.sha256_file(output / "policy.pt"))

    @unittest.skipUnless(importlib.util.find_spec("rsl_rl"), "Optional real RSL-RL reference not installed")
    def test_actual_rsl3_mlp_inference_is_identical(self):
        from rsl_rl.networks import MLP
        actor = MLP(147, 9, [32, 16, 8], "elu").eval()
        exported = module.actor_from_checkpoint(checkpoint(actor), "elu")
        probe = torch.randn(16, 147)
        torch.testing.assert_close(exported(probe), actor(probe), rtol=0, atol=0)

    def test_rejects_wrong_schema_or_normalization(self):
        for n_in, n_out in ((247, 12), (144, 9), (147, 12)):
            with self.assertRaises(ValueError):
                module.actor_from_checkpoint(checkpoint(nn.Sequential(nn.Linear(n_in, n_out))), "elu")
        data = checkpoint(nn.Sequential(nn.Linear(147, 9)))
        data["model_state_dict"]["actor_obs_normalizer._mean"] = torch.zeros(147)
        with self.assertRaisesRegex(ValueError, "Normalized"):
            module.actor_from_checkpoint(data, "elu")

    def test_activation_must_be_known_and_match_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.yaml"
            with self.assertRaises(ValueError):
                module.resolve_activation(path)
            self.assertEqual(module.resolve_activation(path, "elu"), "elu")
            path.write_text("actor:\n  activation: relu\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                module.resolve_activation(path, "elu")
            path.write_text("actor:\n  activation: elu\n  obs_normalization: true\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                module.resolve_activation(path)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
