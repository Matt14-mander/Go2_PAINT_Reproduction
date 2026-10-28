"""Torch-only command/reward math for the opt-in flat low-speed task."""

import torch


def sample_commands(count, device, probabilities=(.15, .30, .20, .25, .10),
                    limits=(1., 1., 1.5)):
    """Modes: stand, yaw, translation, legacy mixed, start/stop/reverse.

    Low-speed magnitudes cover .04.. .22 continuously; half are concentrated
    near .05/.10/.20. Signs are sampled independently with equal probability.
    """
    weights = torch.tensor(probabilities, device=device)
    if weights.shape != (5,) or (weights < 0).any() or not torch.isclose(weights.sum(), weights.new_tensor(1.)):
        raise ValueError("Five nonnegative command probabilities must sum to one")
    mode = torch.multinomial(weights, count, replacement=True)
    command = (torch.rand(count, 3, device=device) * 2 - 1) * weights.new_tensor(limits)
    axis = torch.randint(0, 3, (count,), device=device)
    axis[mode == 1] = 2
    axis[mode == 2] = torch.randint(0, 2, (int((mode == 2).sum()),), device=device)
    magnitude = .04 + .18 * torch.rand(count, device=device)
    concentrated = torch.rand(count, device=device) < .5
    centers = weights.new_tensor([.05, .10, .20])[torch.randint(0, 3, (count,), device=device)]
    magnitude = torch.where(concentrated, centers + .01 * (2 * torch.rand(count, device=device) - 1), magnitude)
    sign = torch.where(torch.rand(count, device=device) < .5, -1., 1.)
    single = torch.zeros_like(command)
    single.scatter_(1, axis[:, None], (magnitude * sign)[:, None])
    command = torch.where(((mode == 1) | (mode == 2) | (mode == 4))[:, None], single, command)
    command[mode == 0] = 0
    return mode, command


def sequence_factor(age):
    """12-second command cycle: stand 2s, positive 3s, stand 2s, reverse 3s, stand 2s."""
    return torch.where((age >= 2) & (age < 5), 1.,
                       torch.where((age >= 7) & (age < 10), -1., 0.))


def precision_reward(command, actual, axis, width=.05, fade_start=.20, fade_end=.35):
    """Additional bounded tracking benefit; smoothly fades outside low speeds.

    Includes zero targets on ALL three axes, so pure yaw cannot earn precision
    reward while drifting sideways. No division by the requested velocity.
    """
    activity = command.abs().amax(-1)
    x = ((activity - fade_start) / (fade_end - fade_start)).clamp(0, 1)
    gate = 1 - x.square() * (3 - 2 * x)
    return gate * torch.exp(-((actual[:, axis] - command[:, axis]) / width).square())
