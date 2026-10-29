"""Stage2 oracle-intent distillation / closed-loop evaluation (Isaac Python).

Teacher is frozen native-exported JIT, including its own normalizer. The student
is a separate MLP, NOT a PPO runner and NOT an intent estimator. See docs.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("train", "eval"))
parser.add_argument("--teacher", type=Path, required=True)
parser.add_argument("--ll-policy", type=Path, required=True)
parser.add_argument("--student", type=Path, help="Exported student policy.pt, eval only")
parser.add_argument("--output", type=Path, required=True, help="New directory; never overwritten")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iterations", type=int, default=2000)
parser.add_argument("--rollout-steps", type=int, default=32)
parser.add_argument("--updates", type=int, default=16)
parser.add_argument("--batch-size", type=int, default=2048)
parser.add_argument("--capacity", type=int, default=131072)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--teacher-warmup", type=int, default=100)
parser.add_argument("--teacher-transition", type=int, default=400)
parser.add_argument("--save-every", type=int, default=100)
parser.add_argument("--steps", type=int, default=1500, help="Evaluation steps")
parser.add_argument("--eval-controller", choices=("student", "teacher"), default="student",
                    help="Eval only: matched teacher closed-loop baseline or student closed loop")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--payload-mass", type=float, default=.5)
parser.add_argument("--wrench-scale", type=float, default=.2)
parser.add_argument("--clean-student", action="store_true", help="Disable history sensor noise for diagnostics")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
for field in ("num_envs", "iterations", "rollout_steps", "updates", "batch_size", "capacity", "teacher_transition", "save_every", "steps"):
    if getattr(args, field) < 1:
        parser.error(f"{field} must be positive")
if args.teacher_warmup < 0 or not 0 < args.learning_rate <= .01 or not 0 < args.payload_mass <= 10 or not 0 <= args.wrench_scale <= 1:
    parser.error("Invalid warmup/lr/payload/wrench scale")
if args.mode == "eval" and args.student is None:
    parser.error("eval requires --student")
if args.mode == "train" and args.student is not None:
    parser.error("Training starts a new student; --student is eval only")
args.teacher, args.ll_policy = args.teacher.expanduser().resolve(), args.ll_policy.expanduser().resolve()
if args.output.exists():
    parser.error("Choose a NEW --output directory")

# Validate input manifests before starting the simulator.
from student_core import (SCHEMA, OracleStudent, Replay, export_student, imitation_loss,
                          load_teacher_manifest, sha256, student_input, teacher_probability, freeze_teacher)
source = load_teacher_manifest(args.teacher, args.ll_policy)
launcher = AppLauncher(args)

try:
    import gymnasium as gym
    import torch
    import LeggedManip_Lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    torch.manual_seed(args.seed)
    task = "GO2-PIPER-PAINT-Transport-Flat-Stage2"
    cfg = parse_env_cfg(task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    cfg.ll_policy_path = str(args.ll_policy)
    cfg.student_noise = not args.clean_student
    cfg.wrench_scale = args.wrench_scale
    cfg.events.payload_mass.params["mass_range"] = (args.payload_mass, args.payload_mass)
    contract = source["transport_action"]
    for key, value in contract.items():
        if key != "control_dt":
            setattr(cfg.actions.transport, key, tuple(value) if key == "command_limits" else value)
    if abs(cfg.sim.dt * cfg.decimation - contract["control_dt"]) > 1e-8:
        raise ValueError("Teacher control period differs from environment")
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = dict(task=task, teacher_path=str(args.teacher), teacher_sha256=sha256(args.teacher),
                      ll_policy_path=str(args.ll_policy), ll_policy_sha256=sha256(args.ll_policy),
                      transport_action=contract, seed=args.seed, student_noise=cfg.student_noise,
                      created_at=datetime.now().isoformat(), arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    (args.output / "run.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    wrapper = gym.make(task, cfg=cfg)
    env = wrapper.unwrapped
    try:
        device = env.device
        teacher = freeze_teacher(torch.jit.load(str(args.teacher), map_location=device))
        with torch.no_grad():
            probe = teacher(torch.zeros(2, 147, device=device))
        if probe.shape != (2, 9) or not torch.isfinite(probe).all():
            raise ValueError("Teacher must be finite 147 -> 9")
        if args.mode == "train":
            model = OracleStudent().to(device)
        else:
            metadata = json.loads(args.student.with_name("manifest.json").read_text())
            for key, expected in (("policy_kind", SCHEMA), ("intent_source", "oracle"),
                                  ("teacher_sha256", provenance["teacher_sha256"]),
                                  ("ll_policy_sha256", provenance["ll_policy_sha256"]),
                                  ("transport_action", contract), ("policy_sha256", sha256(args.student))):
                if metadata.get(key) != expected:
                    raise ValueError(f"Student provenance mismatch: {key}")
            model = torch.jit.load(str(args.student), map_location=device).eval()
        obs, _ = wrapper.reset()
        action_term = env.action_manager.get_term("transport")
        limits = contract["command_limits"]

        def infer(observation):
            x = student_input(observation)
            with torch.no_grad():
                target = teacher(observation["teacher"]).detach()
                prediction = model(x).detach()
            if not all(torch.isfinite(t).all() for t in (x, target, prediction)):
                raise RuntimeError("Non-finite student input / teacher label / student action")
            return x, target, prediction

        if args.mode == "train":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
            replay = Replay(args.capacity, device)
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(args.output / "tensorboard"))

            def save(iteration):
                checkpoint = args.output / f"model_{iteration}.pt"
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                iteration=iteration, provenance=provenance), checkpoint)
                exported = export_student(model, args.output / "exported" / f"model_{iteration}",
                                          dict(provenance, iteration=iteration, checkpoint=str(checkpoint.resolve())))
                # Verify saved JIT numerically against the live actor, not only shapes.
                x, _ = replay.sample(32)
                with torch.no_grad():
                    restored = torch.jit.load(str(exported), map_location=device)
                    torch.testing.assert_close(restored(x), model(x), rtol=1e-5, atol=1e-6)
                print(f"[STUDENT] saved {exported}", flush=True)

            try:
                for iteration in range(args.iterations):
                    beta = teacher_probability(iteration, args.teacher_warmup, args.teacher_transition)
                    failures = resets = teacher_steps = 0
                    for _ in range(args.rollout_steps):
                        if not launcher.app.is_running():
                            raise RuntimeError("Simulator closed before training completed")
                        x, target, prediction = infer(obs)
                        replay.add(x, target)  # label the SAME pre-step state, including student states
                        use_teacher = torch.rand(args.num_envs, 1, device=device) < beta
                        executed = torch.where(use_teacher, target, prediction)
                        with torch.no_grad():
                            obs, _, terminated, timeout, _ = wrapper.step(executed)
                        failures += int(terminated.sum())
                        resets += int((terminated | timeout).sum())
                        teacher_steps += int(use_teacher.sum())
                    total = base_total = arm_total = 0.
                    for _ in range(args.updates):
                        x, target = replay.sample(args.batch_size)
                        loss, base_loss, arm_loss = imitation_loss(model(x), target, limits)
                        if not torch.isfinite(loss):
                            raise RuntimeError("Non-finite distillation loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        optimizer.step()
                        total += loss.item()
                        base_total += base_loss.item()
                        arm_total += arm_loss.item()
                    row = dict(iteration=iteration + 1, loss=total / args.updates,
                               base_loss=base_total / args.updates, arm_loss=arm_total / args.updates,
                               teacher_probability=beta, teacher_fraction=teacher_steps / (args.num_envs * args.rollout_steps),
                               non_timeout_resets=failures, resets=resets, replay_size=replay.size)
                    for name, value in row.items():
                        writer.add_scalar(name, value, iteration + 1)
                    with (args.output / "train.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print(f"[STUDENT] {json.dumps(row)}", flush=True)
                    if (iteration + 1) % args.save_every == 0 or iteration + 1 == args.iterations:
                        save(iteration + 1)
            finally:
                writer.close()
        else:
            # Default: pure student control. Explicit teacher mode is a separate
            # closed-loop baseline; labels always come from the current state.
            errors, velocity_errors = torch.zeros(9, device=device), torch.zeros(3, device=device)
            failures = timeouts = samples = 0
            max_tilt = 0.
            for step in range(args.steps):
                if not launcher.app.is_running():
                    raise RuntimeError("Simulator closed before evaluation completed")
                _, target, prediction = infer(obs)
                errors += (prediction - target).square().sum(0)
                reference = obs["teacher"][:, -3:].clone()
                with torch.no_grad():
                    obs, _, terminated, timeout, _ = wrapper.step(target if args.eval_controller == "teacher" else prediction)
                valid = ~(terminated | timeout)
                data = action_term.robot.data
                actual = torch.cat((data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3]), -1)
                velocity_errors += (actual[valid] - reference[valid]).square().sum(0)
                samples += int(valid.sum())
                if valid.any():
                    tilt = torch.acos((-data.projected_gravity_b[valid, 2]).clamp(-1, 1)).rad2deg().max()
                    max_tilt = max(max_tilt, float(tilt))
                failures += int(terminated.sum())
                timeouts += int((timeout & ~terminated).sum())
                if (step + 1) % 100 == 0:
                    print(f"[STUDENT-EVAL] {step+1}/{args.steps}; failures={failures}", flush=True)
            report = dict(provenance, student=str(args.student.resolve()), student_sha256=sha256(args.student),
                          control_steps=args.steps, sim_time_s=args.steps * env.step_dt,
                          raw_action_rmse_vs_teacher=(errors / (args.steps * args.num_envs)).sqrt().tolist(),
                          velocity_rmse_vs_true_intent=(velocity_errors / max(samples, 1)).sqrt().tolist(),
                          valid_motion_samples=samples, non_timeout_resets=failures, time_limit_resets=timeouts,
                          max_tilt_deg=max_tilt, sampling="post-step excluding auto-reset states; terminal tilt can be missed",
                          teacher_forcing=args.eval_controller == "teacher",
                          acceptance="diagnostic only; no automatic pass threshold")
            (args.output / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[STUDENT-EVAL] {json.dumps(report)}", flush=True)
    finally:
        wrapper.close()
finally:
    launcher.app.close()
