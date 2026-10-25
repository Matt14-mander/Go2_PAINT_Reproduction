"""Export this project's RSL-RL 3.x non-recurrent 147->9 Teacher without Isaac.

Requires only torch and PyYAML. Supports the standard unnormalized actor.N MLP
checkpoint layout; rejects incompatible layouts instead of guessing/truncating.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch
from torch import nn
import yaml


ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU,
               "leaky_relu": nn.LeakyReLU, "sigmoid": nn.Sigmoid}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actor_from_checkpoint(checkpoint, activation):
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Expected RSL-RL 3.x checkpoint containing model_state_dict")
    if any("actor_obs_normalizer" in key or key.startswith("memory_") for key in state):
        raise ValueError("Normalized/recurrent actor is unsupported here; use the updated Isaac play.py exporter")
    if checkpoint.get("obs_norm_state_dict") is not None:
        raise ValueError("External observation normalization is unsupported; use Isaac's native export")
    weights = {key: value for key, value in state.items() if key.startswith("actor.")}
    pattern = re.compile(r"actor\.(\d+)\.(weight|bias)$")
    if not weights or any(pattern.fullmatch(key) is None for key in weights):
        raise ValueError("Unsupported actor layout: expected actor.0.weight / actor.0.bias etc.")
    indices = sorted({int(pattern.fullmatch(key)[1]) for key in weights})
    if indices != list(range(0, 2 * len(indices), 2)):
        raise ValueError("Expected alternating Linear/activation MLP with a linear output")
    layers = []
    previous = None
    for index in indices:
        weight, bias = weights.get(f"actor.{index}.weight"), weights.get(f"actor.{index}.bias")
        if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor) or weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError(f"Invalid Linear parameters at actor.{index}")
        if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in (weight, bias)):
            raise ValueError("Expected finite float32 actor weights")
        output_size, input_size = weight.shape
        if previous is not None and input_size != previous:
            raise ValueError("Actor layer sizes are inconsistent")
        if index == 0 and input_size != 147:
            raise ValueError(f"Expected Teacher input 147, got {input_size}; LL/Student checkpoints are not interchangeable")
        layers.append(nn.Linear(input_size, output_size))
        if index != indices[-1]:
            layers.append(ACTIVATIONS[activation]())
        previous = output_size
    if previous != 9:
        raise ValueError(f"Expected 9 Teacher actions, got {previous}")
    actor = nn.Sequential(*layers)
    actor.load_state_dict({key.removeprefix("actor."): value for key, value in weights.items()}, strict=True)
    actor.eval().requires_grad_(False)
    return actor


def resolve_activation(config_path, explicit=None):
    config = {}
    if config_path.is_file():
        # BaseLoader reads mappings/scalars without constructing Python objects;
        # handles Python tags present in some Isaac YAML dumps as inert data.
        config = yaml.load(config_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        if not isinstance(config, dict):
            raise ValueError(f"Expected mapping in {config_path}")
    values = []
    for key in ("policy", "actor"):
        section = config.get(key, {})
        if isinstance(section, dict):
            if section.get("activation"):
                values.append(str(section["activation"]).lower())
            for flag in ("actor_obs_normalization", "obs_normalization"):
                if str(section.get(flag, "false")).lower() == "true":
                    raise ValueError("Agent config enables actor normalization; use Isaac's native export")
            if "class_name" in section and "recurrent" in str(section["class_name"]).lower():
                raise ValueError("Recurrent actor config is unsupported")
    if explicit:
        values.append(explicit)
    if not values:
        raise ValueError("Cannot infer activation: provide params/agent.yaml or explicitly pass --activation elu for this run")
    if len(set(values)) != 1 or values[0] not in ACTIVATIONS:
        raise ValueError(f"Conflicting/unsupported actor activations: {values}")
    return values[0]


def export(checkpoint_path, output, activation=None, agent_config=None):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    config_path = Path(agent_config) if agent_config else checkpoint_path.parent / "params/agent.yaml"
    activation = resolve_activation(config_path, activation)
    with checkpoint_path.open("rb") as stream:
        checkpoint = torch.load(stream, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary")
    actor = actor_from_checkpoint(checkpoint, activation)
    scripted = torch.jit.script(actor)
    # Compare deterministic inference before/after serialization with actual
    # parameter values; no clipping or postprocessing is embedded in the actor.
    probe = torch.linspace(-1., 1., 4 * 147).reshape(4, 147)
    with torch.inference_mode():
        expected = actor(probe)
        if not torch.isfinite(expected).all():
            raise ValueError("Actor produced non-finite actions on a finite probe")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "policy.pt"
    temporary = output / "policy.pt.tmp"
    with temporary.open("wb") as stream:
        torch.jit.save(scripted, stream)
    with temporary.open("rb") as stream:
        reloaded = torch.jit.load(stream, map_location="cpu")
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(probe), expected, rtol=1e-6, atol=1e-6)
    temporary.replace(path)
    manifest = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
                "policy_sha256": sha256_file(path), "checkpoint_iteration": checkpoint.get("iter"),
                "observations": 147, "actions": 9, "activation": activation,
                "actor_normalization": False, "exporter": "RSL-RL 3.x CPU MLP exporter",
                "agent_config": str(config_path.resolve()) if config_path.is_file() else None}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[EXPORT] Checkpoint: {checkpoint_path}\n[EXPORT] Teacher: 147 -> 9; activation={activation}; normalization=off", flush=True)
    print(f"[EXPORT] Saved: {path}\n[EXPORT] SHA256: {manifest['policy_sha256']}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--agent-config", type=Path)
    parser.add_argument("--activation", choices=tuple(ACTIVATIONS))
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output or args.checkpoint.parent / "exported" / args.checkpoint.stem
    export(args.checkpoint, output, args.activation, args.agent_config)


if __name__ == "__main__":
    main()
