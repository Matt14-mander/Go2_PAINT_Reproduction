"""Read saved training config, TensorBoard scalars and NATIVE exported actor.

Does not reconstruct/export a policy, start Isaac, change weights, or train.
Synthetic probes diagnose gross output saturation, not rollout performance.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


def read_mapping(path):
    if not path.is_file():
        return {}
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return value if isinstance(value, dict) else {}


def get(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def diagnose(run, policy_path):
    env = read_mapping(run / "params/env.yaml")
    agent = read_mapping(run / "params/agent.yaml")
    report = {"run": str(run), "policy": str(policy_path),
              "config": {key: env.get(key) for key in ("observation_mode", "wrench_scale", "literal_tracking_rewards", "episode_length_s")},
              "saved_rewards": env.get("rewards"), "saved_algorithm": agent.get("algorithm"),
              "saved_actor": agent.get("actor"), "saved_native_policy": agent.get("policy"),
              "saved_action": env.get("actions"), "probes_are_synthetic": True}
    report["config"].update(sim_dt=get(env, "sim", "dt"), decimation=env.get("decimation"),
                             reset_base=get(env, "events", "reset_base"),
                             physics_material=get(env, "events", "physics_material"),
                             payload_mass=get(env, "events", "payload_mass"))
    # Only the original/native exporter output is loaded, including any native
    # observation normalizer. Do not build an actor from checkpoint parameters.
    if policy_path.is_file():
        with policy_path.open("rb") as stream:
            actor = torch.jit.load(stream, map_location="cpu").eval()
        report["policy_sha256"] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        if hasattr(actor, "reset"):
            actor.reset()
        zero = torch.zeros(147)
        default_history = torch.zeros(8, 18)
        default_history[:, 12:] = torch.tensor([0., .8, -1.2, 0., .4, 0.])
        default_history = torch.cat((default_history.flatten(), torch.zeros(3)))
        probes = []
        for history, observation in (("reset_zero_history", zero), ("stationary_default_command_history", default_history)):
            for name, reference in (("zero", [0., 0., 0.]), ("forward", [.2, 0., 0.]), ("backward", [-.2, 0., 0.]),
                                    ("left", [0., .2, 0.]), ("right", [0., -.2, 0.]),
                                    ("positive_yaw", [0., 0., .2]), ("negative_yaw", [0., 0., -.2])):
                obs = observation.clone()
                obs[-3:] = torch.tensor(reference)
                with torch.inference_mode():
                    output = actor(obs.unsqueeze(0))
                if not isinstance(output, torch.Tensor) or output.shape != (1, 9) or not torch.isfinite(output).all():
                    raise ValueError("Expected finite native 147->9 exported Teacher")
                raw = output[0]
                limits = raw.new_tensor([1., 1., 1.5])
                probes.append({"history": history, "reference": name, "wrench": reference,
                               "raw_actions_9": raw.tolist(), "bounded_base_command": raw[:3].clamp(-limits, limits).tolist(),
                               "raw_arm_offset_rad": (.5 * raw[3:]).tolist()})
        report["native_actor_probes"] = probes
    else:
        report["policy_error"] = "Native exported policy not found; configs/scalars still available"
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(str(run), size_guidance={"scalars": 1000})
        events.Reload()
        report["tensorboard_scalars"] = {}
        for tag in events.Tags().get("scalars", []):
            if not any(token in tag.lower() for token in ("reward", "loss", "std", "episode_length", "termination")):
                continue
            values = events.Scalars(tag)
            if values:
                report["tensorboard_scalars"][tag] = [{"step": v.step, "value": v.value} for v in values[-5:]]
    except Exception as error:
        # Diagnostics remain useful if TensorBoard is absent or an event file
        # cannot be read. Record the actual failure rather than report success.
        report["tensorboard_error"] = f"{type(error).__name__}: {error}"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--policy", type=Path, help="NATIVE play.py exported/policy.pt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = args.run.expanduser().resolve()
    if not run.is_dir():
        parser.error(f"Run directory not found: {run}")
    policy = (args.policy or run / "exported/policy.pt").expanduser().resolve()
    report = diagnose(run, policy)
    output = args.output or run / "evaluations/teacher_training_diagnosis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[CONFIG] " + json.dumps({k: v for k, v in report["config"].items()
                                    if k not in ("reset_base", "physics_material", "payload_mass")}), flush=True)
    rewards = report.get("saved_rewards") or {}
    print("[REWARD WEIGHTS] " + json.dumps({k: v.get("weight") for k, v in rewards.items() if isinstance(v, dict)}), flush=True)
    print("[ALGORITHM] " + json.dumps(report.get("saved_algorithm")), flush=True)
    for probe in report.get("native_actor_probes", []):
        raw = probe["raw_actions_9"]
        print(f"[PROBE] {probe['history']}/{probe['reference']}: "
              f"raw_base={[round(v, 4) for v in raw[:3]]}; "
              f"bounded={[round(v, 4) for v in probe['bounded_base_command']]}; "
              f"max_abs_raw_arm_offset={max(abs(v) for v in probe['raw_arm_offset_rad']):.4f}rad", flush=True)
    for tag, values in report.get("tensorboard_scalars", {}).items():
        print(f"[TB] {tag}: " + json.dumps(values), flush=True)
    for key in ("policy_error", "tensorboard_error"):
        if key in report:
            print(f"[DIAGNOSE] {key}: {report[key]}", flush=True)
    print(f"[DIAGNOSE] Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
