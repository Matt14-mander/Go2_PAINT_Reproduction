"""Check play.py checkpoint routing without starting Isaac Sim.

Execute the actual main function's AST with dependency test doubles, stopping at
gym.make. This checks import isolation, not simulation or checkpoint weights.
"""

import ast
import builtins
import os
import hashlib
import json
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from packaging import version

SOURCE = Path(__file__).resolve().parents[1] / "rsl_rl/play.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))
MAIN = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "main")
MAIN.decorator_list = []
for argument in MAIN.args.args:
    argument.annotation = None
MAIN.returns = None
CODE = compile(ast.fix_missing_locations(ast.Module(body=[MAIN], type_ignores=[])), str(SOURCE), "exec")


class StopBeforeSimulation(Exception):
    pass


class CheckpointTests(unittest.TestCase):
    def test_export_only_has_checkpoint_provenance_and_never_rolls_out(self):
        context, env_cfg, agent = self.make_context()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model_9999.pt"
            checkpoint.write_bytes(b"checkpoint-fixture")
            output = root / "exported" / "model_9999"
            context["retrieve_file_path"].return_value = str(checkpoint)
            context["args_cli"].export_only = True
            context["args_cli"].export_dir = str(output)
            agent.device, agent.class_name, agent.clip_actions = "cpu", "OnPolicyRunner", None
            agent.to_dict = lambda: {}
            env = SimpleNamespace(unwrapped=SimpleNamespace(device="cpu"), close=Mock(),
                                  get_observations=Mock(side_effect=AssertionError("Export must not roll out")))
            runner = Mock()
            runner.alg.policy = SimpleNamespace()
            def export_jit(*args, path, filename, **kwargs):
                Path(path).mkdir(parents=True)
                (Path(path) / filename).write_bytes(b"actor-fixture")
            context.update(hashlib=hashlib, json=json, version=version,
                           DirectMARLEnv=type("DirectMARLEnv", (), {}),
                           RslRlVecEnvWrapper=lambda raw, **kwargs: env,
                           OnPolicyRunner=Mock(return_value=runner),
                           export_policy_as_jit=export_jit, export_policy_as_onnx=Mock())
            context["gym"].make = Mock(return_value=env)
            context["main"](env_cfg, agent)
            runner.load.assert_called_once_with(str(checkpoint))
            env.get_observations.assert_not_called()
            env.close.assert_called_once()
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(Path(manifest["checkpoint"]), checkpoint)
            self.assertEqual(manifest["checkpoint_sha256"], hashlib.sha256(b"checkpoint-fixture").hexdigest())
            self.assertEqual(manifest["policy_sha256"], hashlib.sha256(b"actor-fixture").hexdigest())

    def make_context(self, checkpoint="/weights/model_9999.pt", pretrained=False):
        args = SimpleNamespace(task="GO2-PIPER-PAINT-Loco-Flat-Play", num_envs=1, device=None,
                               use_pretrained_checkpoint=pretrained, checkpoint=checkpoint, video=False, eval_name="play")
        env = SimpleNamespace(scene=SimpleNamespace(num_envs=1), sim=SimpleNamespace(device="cpu"))
        agent = SimpleNamespace(seed=42, experiment_name="go2_piper_paint_loco",
                                load_run="run", load_checkpoint="model_9999.pt")
        context = dict(args_cli=args, cli_args=SimpleNamespace(update_rsl_rl_cfg=lambda cfg, _: cfg),
                       handle_deprecated_rsl_rl_cfg=lambda cfg, _: cfg, installed_version="3.0.1", os=os,
                       retrieve_file_path=Mock(return_value="/weights/model_9999.pt"),
                       get_checkpoint_path=Mock(return_value="/logs/run/model_9999.pt"),
                       configure_evaluation=lambda cfg, args: None, datetime=datetime,
                       gym=SimpleNamespace(make=Mock(side_effect=StopBeforeSimulation)))
        exec(CODE, context)
        return context, env, agent

    def block_optional_import(self, available=None):
        original = builtins.__import__
        def importer(name, *args, **kwargs):
            if name == "isaaclab_rl.utils.pretrained_checkpoint":
                if available is not None:
                    return SimpleNamespace(get_published_pretrained_checkpoint=available)
                raise ModuleNotFoundError("No module named isaaclab_rl.utils", name="isaaclab_rl.utils")
            return original(name, *args, **kwargs)
        return patch("builtins.__import__", side_effect=importer)

    def test_optional_module_not_imported_at_top_level(self):
        self.assertFalse(any(isinstance(node, ast.ImportFrom) and
                             node.module == "isaaclab_rl.utils.pretrained_checkpoint" for node in TREE.body))

    def test_explicit_checkpoint_works_without_optional_package(self):
        context, env, agent = self.make_context()
        with self.block_optional_import() as importer, self.assertRaises(StopBeforeSimulation):
            context["main"](env, agent)
        context["retrieve_file_path"].assert_called_once_with("/weights/model_9999.pt")
        self.assertFalse(any(call.args[0] == "isaaclab_rl.utils.pretrained_checkpoint"
                             for call in importer.call_args_list))
        self.assertEqual(env.log_dir, "/weights")

    def test_run_lookup_works_without_optional_package(self):
        context, env, agent = self.make_context(checkpoint=None)
        with self.block_optional_import(), self.assertRaises(StopBeforeSimulation):
            context["main"](env, agent)
        context["get_checkpoint_path"].assert_called_once()

    def test_unsupported_pretrained_request_has_actionable_error(self):
        context, env, agent = self.make_context(pretrained=True)
        with self.block_optional_import(), self.assertRaisesRegex(RuntimeError, "Omit --use_pretrained_checkpoint"):
            context["main"](env, agent)
        context["gym"].make.assert_not_called()

    def test_available_pretrained_lookup_still_used(self):
        context, env, agent = self.make_context(pretrained=True)
        lookup = Mock(return_value="/published/model.pt")
        with self.block_optional_import(available=lookup), self.assertRaises(StopBeforeSimulation):
            context["main"](env, agent)
        lookup.assert_called_once_with("rsl_rl", "GO2-PIPER-PAINT-Loco-Flat")
        self.assertEqual(env.log_dir, "/published")


if __name__ == "__main__":
    unittest.main()
