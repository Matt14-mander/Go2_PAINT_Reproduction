"""CPU-testable oracle-intent student components. No Isaac/RSL imports."""
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

SCHEMA = "paint_student_oracle_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_teacher(module):
    # ScriptModule does not implement Module.requires_grad_().
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


class OracleStudent(nn.Module):
    """8 x [arm q error(6), qdot(6), applied target(6)] + true intent(3).

    History order is oldest first. Outputs are raw Stage2 actions, BEFORE the
    shared deadband/clamp/startup/arm slew logic. No independent normalizer.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(147, 256), nn.ELU(), nn.Linear(256, 128),
                                 nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 9))

    def forward(self, observation):
        return self.net(observation)


def student_input(observations):
    history, teacher = observations["student"], observations["teacher"]
    if history.ndim != 2 or history.shape[-1] != 144 or teacher.shape != (len(history), 147):
        raise ValueError("Expected student history Bx144 and teacher observation Bx147")
    return torch.cat((history, teacher[:, -3:]), -1)


def imitation_loss(prediction, target, command_limits):
    # Equal group weighting, so six arm outputs do not swamp three base outputs.
    scale = prediction.new_tensor(command_limits)
    base = ((prediction[:, :3] - target[:, :3]) / scale).square().mean()
    arm = (prediction[:, 3:] - target[:, 3:]).square().mean()
    return base + arm, base.detach(), arm.detach()


class Replay:
    def __init__(self, capacity, device):
        self.x = torch.empty(capacity, 147, device=device)
        self.y = torch.empty(capacity, 9, device=device)
        self.size = self.position = 0

    def add(self, x, y):
        x, y = x.detach()[-len(self.x):], y.detach()[-len(self.y):]
        n = len(x)
        index = (torch.arange(n, device=self.x.device) + self.position) % len(self.x)
        self.x[index], self.y[index] = x, y
        self.position = (self.position + n) % len(self.x)
        self.size = min(len(self.x), self.size + n)

    def sample(self, count):
        if not self.size:
            raise ValueError("Empty replay")
        ids = torch.randint(self.size, (min(count, self.size),), device=self.x.device)
        return self.x[ids], self.y[ids]


def teacher_probability(iteration, warmup, transition):
    if iteration < warmup:
        return 1.0
    return max(0., 1. - (iteration - warmup + 1) / transition)


def load_teacher_manifest(teacher, ll_policy):
    path = Path(teacher).parent / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("task") != "GO2-PIPER-PAINT-Transport-Flat-Stage2" or manifest.get("policy_kind") == SCHEMA:
        raise ValueError("Export the accepted Stage2 teacher with play.py first")
    if manifest.get("policy_sha256") != sha256(teacher):
        raise ValueError("Teacher manifest hash mismatch")
    # Old native exports have ll_policy_path but no LL digest. Verify the actual
    # referenced file, never silently assume a different low-level is equivalent.
    digest = manifest.get("ll_policy_sha256")
    if digest is None:
        source = Path(manifest.get("ll_policy_path") or "")
        if not source.is_file():
            raise ValueError("Teacher LL source unavailable; re-export with the original LL path")
        digest = sha256(source)
    if digest != sha256(ll_policy):
        raise ValueError("Teacher and student must use the same frozen LL policy")
    contract = manifest["transport_action"]
    if contract["freeze_arm"] or abs(contract["control_dt"] - .02) > 1e-8:
        raise ValueError("Expected active Stage2 arm at 50Hz")
    return manifest


def export_student(model, directory, provenance):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    import copy
    exported = torch.jit.script(copy.deepcopy(model).cpu().eval())
    with (directory / "policy.pt").open("wb") as stream:
        torch.jit.save(exported, stream)
    metadata = dict(provenance, policy_kind=SCHEMA,
                    intent_source="oracle", observations=147, actions=9,
                    observation_schema="oldest_first_8x18_arm_history_then_true_body_intent",
                    policy_sha256=sha256(directory / "policy.pt"))
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory / "policy.pt"
