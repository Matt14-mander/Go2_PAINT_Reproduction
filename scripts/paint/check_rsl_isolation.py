"""Standard-library checks for project-local RSL selection; no pip/Kit/GPU runs."""

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("with_rsl3", Path(__file__).with_name("with_rsl3.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class IsolationTests(unittest.TestCase):
    def test_environment_is_child_only_and_keeps_existing_paths(self):
        with patch.dict(os.environ, {"PYTHONPATH": "/existing/tron2"}):
            selected = module.selected_environment(Path("/paint/rsl"))
            self.assertTrue(selected["PYTHONPATH"].endswith(os.pathsep + "/existing/tron2"))
            self.assertEqual(os.environ["PYTHONPATH"], "/existing/tron2")

    def test_metadata_and_module_checked_without_importing_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "rsl"
            (target / "rsl_rl").mkdir(parents=True)
            # The probe must locate the package without executing it.
            (target / "rsl_rl/__init__.py").write_text("raise RuntimeError('must not import')\n", encoding="utf-8")
            info = target / "rsl_rl_lib-3.0.1.dist-info"
            info.mkdir()
            (info / "METADATA").write_text("Metadata-Version: 2.1\nName: rsl-rl-lib\nVersion: 3.0.1\n", encoding="utf-8")
            with patch.object(module, "ROOT", root):
                module.verify(target)

    def test_missing_target_gives_install_command(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "with_rsl3.py install"):
                module.verify(Path(directory) / "missing")

    def test_partial_install_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / "unrelated.txt").write_text("keep", encoding="utf-8")
            with patch.object(module, "verify", side_effect=RuntimeError("unknown installation")), \
                    patch.object(module.subprocess, "run") as run:
                with self.assertRaises(RuntimeError):
                    module.install(target)
            run.assert_not_called()

    def test_run_reexecutes_same_interpreter_with_selected_environment(self):
        with patch.object(module, "verify"), patch.object(module.os, "chdir"), \
                patch.object(module.os, "execve") as execute:
            module.main(["run", "scripts/rsl_rl/play.py", "--headless"])
        executable, command, environment = execute.call_args.args
        self.assertEqual(executable, module.sys.executable)
        self.assertEqual(command[0], module.sys.executable)
        self.assertEqual(command[-1], "--headless")
        self.assertTrue(environment["PYTHONPATH"].startswith(str(module.TARGET.resolve())))


if __name__ == "__main__":
    unittest.main()
