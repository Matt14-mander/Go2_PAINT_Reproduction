"""Stage1 -> Stage2 transfer for the project's verified RSL-RL 3.x runner."""

import hashlib
from pathlib import Path

import torch


def warm_start_stage2(runner, checkpoint, base_std=0.05, arm_std=0.1):
    """Keep locomotion intent mapping, reset unused arm head and exploration.

    This starts a NEW experiment: no optimizer state or iteration is inherited.
    Normal Stage2 resume must use runner.load instead and preserve the arm head.
    """
    path = Path(checkpoint).expanduser().resolve()
    policy = getattr(runner.alg, "policy", None)
    if policy is None or not hasattr(policy, "actor"):
        raise RuntimeError("Stage2 warm start currently supports the verified RSL-RL 3.x ActorCritic runner")
    layers = [layer for layer in policy.actor.modules() if isinstance(layer, torch.nn.Linear)]
    if not layers or layers[0].in_features != 147 or layers[-1].out_features != 9:
        raise ValueError("Stage2 requires the 147->9 transport actor")
    deviation = getattr(policy, "log_std", None)
    logarithmic = deviation is not None
    if deviation is None:
        deviation = getattr(policy, "std", None)
    if deviation is None or deviation.shape != (9,):
        raise ValueError("Expected a separate Gaussian exploration parameter for each of the nine actions")
    runner.load(str(path), load_optimizer=False, map_location=runner.device)
    with torch.no_grad():
        layers[-1].weight[3:].zero_()
        layers[-1].bias[3:].zero_()
        std = deviation.new_tensor([base_std] * 3 + [arm_std] * 6)
        deviation.copy_(std.log() if logarithmic else std)
    runner.current_learning_iteration = 0
    return {"source_checkpoint": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "optimizer_loaded": False, "iteration_reset": True, "arm_output_head_reset": True,
            "initial_action_std": std.tolist()}
