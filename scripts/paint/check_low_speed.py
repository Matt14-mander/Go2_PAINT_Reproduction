"""CPU checks with real RSL-RL 3.x and MuJoCo; no Isaac simulation claims."""

import copy
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))
from low_speed_warm_start import warm_start_low_speed
from low_speed_protocol import cases, command_at, summarize, compare_baseline
import evaluate_low_speed
import low_speed_regression

MDP = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp"
spec = importlib.util.spec_from_file_location("low_speed_math", MDP / "low_speed_math.py")
math = importlib.util.module_from_spec(spec)
spec.loader.exec_module(math)


class TensorEnv:
    num_envs, num_actions, device, max_episode_length = 8, 12, "cpu", 1200
    def __init__(self):
        self.episode_length_buf = torch.zeros(8, dtype=torch.long)
    def get_observations(self):
        return TensorDict({"policy": torch.randn(8, 247), "critic": torch.randn(8, 269)}, batch_size=[8])
    def step(self, actions):
        self.episode_length_buf += 1
        return self.get_observations(), -actions.square().mean(-1), torch.zeros(8, dtype=torch.long), {}


def runner():
    cfg = dict(num_steps_per_env=4, save_interval=100, obs_groups={"policy": ["policy"], "critic": ["critic"]},
               policy=dict(class_name="ActorCritic", actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16],
                           activation="elu", noise_std_type="log", init_noise_std=.1),
               algorithm=dict(class_name="PPO", learning_rate=5e-5, entropy_coef=.001,
                              clip_param=.1, num_learning_epochs=3, num_mini_batches=4,
                              schedule="fixed", desired_kl=.005, max_grad_norm=.5))
    value = OnPolicyRunner(TensorEnv(), cfg, log_dir=None, device="cpu")
    value.logger_type = "tensorboard"
    return value


class LowSpeedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sampling_covers_low_speed_axes_symmetry_and_original_envelope(self):
        torch.manual_seed(42)
        mode, command = math.sample_commands(50000, "cpu")
        torch.testing.assert_close(mode.bincount(minlength=5) / 50000, torch.tensor([.15, .30, .20, .25, .10]), atol=.008, rtol=0)
        self.assertEqual(int(command[mode == 0].count_nonzero()), 0)
        self.assertEqual(int(command[mode == 1, :2].count_nonzero()), 0)
        self.assertEqual(int(command[mode == 2, 2].count_nonzero()), 0)
        for selected in (1, 2, 4):
            nonzero = command[mode == selected][command[mode == selected] != 0]
            self.assertLess(abs(float((nonzero > 0).float().mean()) - .5), .03)
            self.assertGreaterEqual(float(nonzero.abs().min()), .0399)
            self.assertLessEqual(float(nonzero.abs().max()), .2201)
        self.assertGreater(float(command[mode == 3, 0].abs().max()), .99)
        self.assertGreater(float(command[mode == 3, 2].abs().max()), 1.49)

    def test_precision_rewards_zero_drift_and_correct_sign_without_discontinuity(self):
        command = torch.tensor([[0., 0., .1], [0., 0., -.1], [0., 0., 0.]])
        correct = math.precision_reward(command, command, 2, .06)
        stationary = math.precision_reward(command, torch.zeros_like(command), 2, .06)
        reverse = math.precision_reward(command, -command, 2, .06)
        self.assertTrue((correct[:2] > stationary[:2]).all())
        self.assertTrue((stationary[:2] > reverse[:2]).all())
        self.assertEqual(float(stationary[2]), 1.)
        drift = command.clone(); drift[:, 0] = .1
        self.assertTrue((math.precision_reward(command, drift, 0, .04) < .01).all())
        command = torch.tensor([[0., 0., .34999], [0., 0., .35], [0., 0., .35001]])
        self.assertLess(float(math.precision_reward(command, command, 2).max()), 1e-5)

    def test_real_command_adapter_reset_isolation_and_sequence(self):
        commands = ModuleType("isaaclab.envs.mdp.commands")
        class Uniform:
            def __init__(self, cfg, env):
                self.cfg, self._env, self.num_envs, self.device = cfg, env, 4, "cpu"
                self.vel_command_b = torch.zeros(4, 3)
                self.is_standing_env = torch.zeros(4, dtype=torch.bool)
                self.is_heading_env = self.is_standing_env.clone()
                self.metrics = {}
        commands.UniformVelocityCommand = Uniform
        utils = ModuleType("isaaclab.utils"); utils.configclass = lambda value: value
        config = ModuleType("test_low.cfg.command_cfg"); config.UniformVelocityCommandCfg = type("Cfg", (), {})
        spec = importlib.util.spec_from_file_location("test_low.low_speed", MDP / "low_speed.py")
        adapter = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"isaaclab.envs.mdp.commands": commands, "isaaclab.utils": utils,
                                     "test_low.cfg.command_cfg": config, "test_low.low_speed_math": math}):
            spec.loader.exec_module(adapter)
        cfg = SimpleNamespace(fixed_command=None, probabilities=(.15, .3, .2, .25, .1), mixed_limits=(1., 1., 1.5))
        term = adapter.LowSpeedVelocityCommand(cfg, SimpleNamespace(step_dt=.02))
        term.mode[:] = 4; term.target[:, 2] = .1
        for age, expected in ((0, 0), (2, .1), (5, 0), (7, -.1), (10, 0)):
            term.age[:] = age; term._update_command()
            torch.testing.assert_close(term.vel_command_b[:, 2], torch.full((4,), float(expected)))
        before = term.target[1:].clone()
        term._resample_command([0])
        torch.testing.assert_close(term.target[1:], before)
        cfg.fixed_command = (0., 0., -.05)
        term._resample_command([0]); term._update_command()
        torch.testing.assert_close(term.vel_command_b[0], torch.tensor([0., 0., -.05]))

    def test_real_rsl_transfer_preserves_actor_and_normal_resume_preserves_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            source = runner()
            with torch.no_grad():
                source.alg.policy.log_std.copy_(torch.tensor([.01, .1, .8] * 4).log())
            source.current_learning_iteration = 9999
            source.save(str(path))
            source_bytes = path.read_bytes()
            dest = runner()
            obs = source.env.get_observations()
            reference = source.alg.policy.act_inference(obs).detach().clone()
            meta = warm_start_low_speed(dest, path)
            torch.testing.assert_close(dest.alg.policy.act_inference(obs), reference, rtol=0, atol=0)
            torch.testing.assert_close(dest.alg.policy.log_std.exp(), torch.tensor([.05, .1, .2] * 4))
            self.assertFalse(meta["optimizer_loaded"])
            self.assertEqual(dest.current_learning_iteration, 0)
            self.assertFalse(dest.alg.optimizer.state)
            with patch("rsl_rl.runners.on_policy_runner.store_code_state", return_value=[]):
                dest.learn(num_learning_iterations=2, init_at_random_ep_len=False)
            self.assertEqual(dest.alg.learning_rate, 5e-5)
            self.assertTrue(dest.alg.optimizer.state)
            self.assertEqual(path.read_bytes(), source_bytes)
            saved = Path(directory) / "resume.pt"; dest.save(str(saved))
            resumed = runner(); resumed.load(str(saved))
            torch.testing.assert_close(resumed.alg.policy.act_inference(obs), dest.alg.policy.act_inference(obs))
            self.assertTrue(resumed.alg.optimizer.state)

    def test_acceptance_detects_deadzone_oscillation_reset_and_transitions(self):
        dt = .02; t = np.arange(1600) * dt
        commands = np.tile([0., 0., .1], (1600, 1))
        def score(actual, failures=0):
            return summarize("wz_pos_0.1", (0, 0, .1), t, commands, actual,
                             np.zeros(1600), np.zeros((1600, 12)), np.zeros((1600, 12)), failures, 1600, dt)
        self.assertTrue(score(commands)["passed"])
        self.assertFalse(score(np.zeros_like(commands))["passed"])
        self.assertFalse(score(commands, 1)["passed"])
        oscillating = commands.copy(); oscillating[:, 2] += .1 * np.sin(t * 30)
        self.assertFalse(score(oscillating)["passed"])
        command = np.array([command_at("sequence_wz", (0, 0, .1), x) for x in t])
        result = summarize("sequence_wz", (0, 0, .1), t, command, command,
                           np.zeros(1600), np.zeros((1600, 12)), np.zeros((1600, 12)), 0, 1600, dt)
        self.assertTrue(result["passed"])
        self.assertTrue(all(value <= 2 for value in result["response_time_s"].values()))

    def test_regression_gate_rejects_quality_loss_or_mismatched_protocol(self):
        row = dict(case="stand", seed=42, complete=True, steady_rmse=[.001, .001, .001],
                   action_delta_rms=.01, torque_rms_nm=2., max_tilt_deg=3.)
        original = dict(backend="mujoco", policy_sha256="original", results=[row])
        candidate = copy.deepcopy(original)
        self.assertTrue(compare_baseline(candidate, original)["passed"])
        candidate["results"][0]["steady_rmse"][2] = .1
        self.assertFalse(compare_baseline(candidate, original)["passed"])
        candidate["backend"] = "isaac"
        with self.assertRaises(ValueError):
            compare_baseline(candidate, original)

    def test_actual_mujoco_evaluator_runs_and_serializes_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            actor = torch.nn.Linear(247, 12)
            torch.nn.init.zeros_(actor.weight); torch.nn.init.zeros_(actor.bias)
            with path.open("wb") as stream:
                torch.jit.save(torch.jit.script(actor), stream)
            args = SimpleNamespace(policy=path, seeds=[42], quick=True, payload_mass=0.)
            with patch.object(evaluate_low_speed, "cases", return_value=[("stand", (0., 0., 0.))]):
                results, dt, execution = evaluate_low_speed.mujoco_rollouts(args)
            self.assertEqual(len(results), 1)
            self.assertEqual(dt, .02)
            json.dumps(results, allow_nan=False)
            self.assertGreater(results[0]["samples"], 0)

    def test_regression_hook_saves_before_check_and_stops_on_regression(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            baseline = output / "baseline.pt"; baseline.write_bytes(b"original-policy")
            active = runner()
            before = {key: value.clone() for key, value in active.alg.policy.state_dict().items()}
            exports = ModuleType("isaaclab_rl.rsl_rl")
            def export(policy, normalizer, path, filename):
                target = Path(path); target.mkdir(parents=True)
                with (target / filename).open("wb") as stream:
                    torch.jit.save(torch.jit.script(copy.deepcopy(policy.actor)), stream)
            exports.export_policy_as_jit = export
            calls = []
            def subprocess_run(command, **kwargs):
                policy_path = Path(command[command.index("--policy") + 1])
                self.assertTrue(policy_path.is_file())
                report = Path(command[command.index("--report") + 1])
                report.write_text("{}")
                calls.append(report.stem)
                return SimpleNamespace(returncode=2 if report.stem == "model_500" else 0)
            with patch.dict(sys.modules, {"isaaclab_rl.rsl_rl": exports}), patch.object(low_speed_regression.subprocess, "run", side_effect=subprocess_run):
                low_speed_regression.install_regression_hook(active, output / "run", baseline, 500)
                active.save(str(output / "model_0.pt"))
                active.save(str(output / "model_100.pt"))
                self.assertEqual(calls, ["baseline", "model_0"])
                active.low_speed_regression_final()
                self.assertEqual(calls[-1], "model_100")
                with self.assertRaisesRegex(RuntimeError, "stopped training"):
                    active.save(str(output / "model_500.pt"))
            self.assertTrue((output / "model_500.pt").is_file())
            self.assertEqual(baseline.read_bytes(), b"original-policy")
            for key, value in active.alg.policy.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_focused_configuration_keeps_restore_conditions_isolated(self):
        # Execute actual subclass post-init bodies with an isolated base fixture.
        # Real Isaac dataclass/manager construction is covered by server smoke.
        NS = SimpleNamespace
        class Ranges(NS):
            def copy(self):
                return copy.deepcopy(self)
        class Base:
            def __init__(self):
                self.__post_init__()
            def __post_init__(self):
                self.commands = NS(base_velocity=NS(limit_ranges=Ranges()))
                self.curriculum = NS()
                self.scene = NS(robot=NS(actuators={"legs": NS(min_delay=0, max_delay=4)}))
                self.rewards = NS()
                self.events = NS(**{key: NS(params={}) for key in (
                    "payload_mass", "push_robot", "payload_wrench_interval", "physics_material", "add_mass",
                    "base_com", "randomize_rigid_body_inertia", "randomize_actuator_gains", "reset_robot_joints")})
                self.events.payload_mass.params["mass_range"] = (0., 10.)
                self.observations = NS(policy=NS(**{key: NS(noise=NS(n_min=-.2, n_max=.2)) for key in
                                                     ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel")}))
        source = MDP.parent / "config/go2_piper/low_speed_env_cfg.py"
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        context = dict(configclass=lambda value: value, Go2PiperPaintLocoEnvCfg=Base,
                       LowSpeedVelocityCommandCfg=NS, RewTerm=NS, low_speed_tracking=lambda: None)
        exec(compile(tree, str(source), "exec"), context)
        focused = context["Go2PiperLowSpeedEnvCfg"]()
        restore = context["Go2PiperLowSpeedRestoreEnvCfg"]()
        self.assertEqual(focused.events.payload_mass.params["mass_range"], (0., 1.))
        self.assertEqual(restore.events.payload_mass.params["mass_range"], (0., 10.))
        self.assertEqual(focused.scene.robot.actuators["legs"].max_delay, 1)
        self.assertEqual(restore.scene.robot.actuators["legs"].max_delay, 4)
        self.assertEqual(restore.observations.policy.base_ang_vel.noise.n_max, .2)
        self.assertEqual(focused.observations.policy.base_ang_vel.noise.n_max, .05)
        self.assertIsNone(focused.events.push_robot)
        self.assertIsNotNone(restore.events.push_robot)


if __name__ == "__main__":
    unittest.main()
