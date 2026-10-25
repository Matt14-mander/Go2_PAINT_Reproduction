"""Check pre-Kit version discovery without importing any simulator packages."""

import ast
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

DIRECTORY = Path(__file__).resolve().parents[1] / "rsl_rl"
spec = importlib.util.spec_from_file_location("runtime_checks", DIRECTORY / "runtime_checks.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RuntimeTests(unittest.TestCase):
    def test_reads_real_metadata_without_simulator(self):
        with patch.object(module.metadata, "version", return_value="3.0.1") as lookup:
            self.assertEqual(module.get_rsl_version_before_kit(), "3.0.1")
        lookup.assert_called_once_with("rsl-rl-lib")

    def test_missing_metadata_has_environment_diagnostic(self):
        with patch.object(module.metadata, "version", side_effect=module.metadata.PackageNotFoundError("rsl-rl-lib")):
            with self.assertRaisesRegex(RuntimeError, "not visible before Isaac/Kit startup") as error:
                module.get_rsl_version_before_kit()
        self.assertIn("PYTHONNOUSERSITE", str(error.exception))
        self.assertIn("pip show", str(error.exception))

    def test_version_check_precedes_isaac_import_and_launch(self):
        for filename in ("train.py", "play.py"):
            tree = ast.parse((DIRECTORY / filename).read_text(encoding="utf-8"))
            checks = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == "get_rsl_version_before_kit"]
            imports = [node.lineno for node in tree.body if isinstance(node, ast.ImportFrom)
                       and node.module == "isaaclab.app"]
            self.assertEqual(len(checks), 1, filename)
            self.assertLess(checks[0], imports[0], filename)


if __name__ == "__main__":
    unittest.main()
