"""Weights-only transfer for the verified RSL-RL 3.x 247->12 locomotion actor."""

import hashlib
from pathlib import Path
import torch


def warm_start_low_speed(runner, checkpoint):
    path = Path(checkpoint).expanduser().resolve()
    policy = getattr(runner.alg, "policy", None)
    if policy is None or not hasattr(policy, "actor"):
        raise RuntimeError("Low-speed transfer requires the RSL-RL 3.x ActorCritic runner")
    layers = [layer for layer in policy.actor.modules() if isinstance(layer, torch.nn.Linear)]
    if not layers or layers[0].in_features != 247 or layers[-1].out_features != 12:
        raise ValueError("Low-speed transfer requires the unchanged 247->12 actor")
    if runner.alg.optimizer.state:
        raise ValueError("Warm start must be applied to a fresh runner/optimizer")
    runner.load(str(path), load_optimizer=False, map_location=runner.device)
    parameter = getattr(policy, "log_std", None)
    log = parameter is not None
    if not log:
        parameter = getattr(policy, "std", None)
    if parameter is None or parameter.shape != (12,):
        raise ValueError("Expected 12 separate exploration standard deviations")
    with torch.no_grad():
        original = parameter.exp() if log else parameter.clone()
        if not torch.isfinite(original).all() or (original <= 0).any():
            raise ValueError("Source checkpoint contains invalid exploration std")
        # Preserve learned exploration when already modest; constrain extremes.
        initial = original.clamp(.05, .20)
        parameter.copy_(initial.log() if log else initial)
    runner.current_learning_iteration = 0
    return dict(source_checkpoint=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                optimizer_loaded=False, iteration_reset=True, actor_mean_weights_preserved=True,
                critic_weights_preserved=True, source_action_std=original.tolist(),
                initial_action_std=initial.tolist(), action_scale=.25, observation_dim=247, action_dim=12)
