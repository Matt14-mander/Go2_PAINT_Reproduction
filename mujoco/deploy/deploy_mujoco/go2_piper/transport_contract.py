"""Stage2 execution parameters exported by native Isaac play.py."""

import hashlib
import json
from pathlib import Path

import numpy as np


def load_stage2_contract(policy_paths):
    contracts = []
    for policy in policy_paths:
        path = Path(policy).expanduser()
        manifest = path.parent / "manifest.json"
        value = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
        if value.get("task") != "GO2-PIPER-PAINT-Transport-Flat-Stage2":
            contracts.append(None)
            continue
        if value.get("policy_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Stage2 policy/manifest hash mismatch: {path}")
        contract = value["transport_action"]
        for key in ("arm_scale", "arm_max_offset", "arm_max_speed", "control_dt", "force_ramp_seconds"):
            if not np.isfinite(contract[key]) or contract[key] <= 0:
                raise ValueError(f"Invalid Stage2 execution parameter: {key}")
        for key in ("settle_seconds", "stop_hold_seconds", "command_deadband"):
            if not np.isfinite(contract[key]) or contract[key] < 0:
                raise ValueError(f"Invalid Stage2 execution parameter: {key}")
        if contract["freeze_arm"] or not np.isclose(contract["control_dt"], .02):
            raise ValueError("Stage2 requires an active arm and 50Hz control")
        limits = np.asarray(contract["command_limits"])
        if limits.shape != (3,) or not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError("Invalid Stage2 command_limits")
        contracts.append(contract)
    active = [c for c in contracts if c is not None]
    if active and (len(active) != len(contracts) or any(c != active[0] for c in active)):
        raise ValueError("All robots must use the same Stage2 execution contract")
    return active[0] if active else None


def arm_target(raw, default, limits, previous, contract):
    residual = np.clip(np.asarray(raw) * contract["arm_scale"],
                       -contract["arm_max_offset"], contract["arm_max_offset"])
    desired = np.clip(default + residual, limits[:, 0], limits[:, 1])
    delta = contract["arm_max_speed"] * contract["control_dt"]
    return previous + np.clip(desired - previous, -delta, delta)


def force_profile(time, duration, contract):
    settle, ramp, stop = (contract[k] for k in ("settle_seconds", "force_ramp_seconds", "stop_hold_seconds"))
    if duration <= settle + 2 * ramp + stop:
        raise ValueError("Duration must include startup, force ramps, and a zero-force tail")
    return float(np.clip(min((time - settle) / ramp, (duration - stop - time) / ramp), 0., 1.))
