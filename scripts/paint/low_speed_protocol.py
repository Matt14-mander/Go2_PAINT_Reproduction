"""Shared, simulator-independent acceptance protocol. SI units throughout."""

import numpy as np


def cases(quick=False):
    result = [("stand", (0., 0., 0.))]
    for axis, name in enumerate(("vx", "vy", "wz")):
        for magnitude in ((.1, .3) if axis < 2 else (.05, .1, .2, .3)):
            for sign, label in ((1, "pos"), (-1, "neg")):
                command = [0., 0., 0.]
                command[axis] = sign * magnitude
                result.append((f"{name}_{label}_{magnitude:g}", tuple(command)))
    result += [("mixed_pos", (.3, .15, .3)), ("mixed_neg", (-.3, -.15, -.3))]
    if not quick:
        result += [(f"sequence_{name}", tuple(.1 if i == axis else 0. for i in range(3)))
                   for axis, name in enumerate(("vx", "vy", "wz"))]
    return result


def command_at(name, target, time_s):
    if not name.startswith("sequence_"):
        return np.asarray(target, dtype=float)
    factor = 1. if 3 <= time_s < 11 else -1. if 16 <= time_s < 24 else 0.
    return np.asarray(target, dtype=float) * factor


def initial_offsets(seed):
    """42 is nominal; other seeds use small, reproducible matched pose offsets."""
    if seed == 42:
        return np.zeros(12), 0.
    rng = np.random.default_rng(seed)
    return rng.uniform(-.005, .005, 12), float(rng.uniform(-.02, .02))


def summarize(name, target, time_s, command, actual, tilt, action_delta, torque,
              failures, expected_steps, dt):
    t, c, v = np.asarray(time_s), np.asarray(command), np.asarray(actual)
    if len(t) == 0:
        return dict(case=name, failures=failures, samples=0, passed=False, complete=False)
    complete = len(t) == expected_steps and failures == 0 and np.isfinite(v).all()
    steady = (t >= 5.) & (t < 30.)
    if name.startswith("sequence_"):
        steady = ((t >= 8) & (t < 11)) | ((t >= 21) & (t < 24))
    e = v - c
    result = dict(case=name, target=list(target), samples=len(t), failures=int(failures),
                  complete=bool(complete), full_rmse=np.sqrt(np.mean(e**2, axis=0)).tolist(),
                  mean_actual=v.mean(0).tolist(), steady_samples=int(steady.sum()),
                  max_tilt_deg=float(np.max(tilt)),
                  action_delta_rms=float(np.sqrt(np.mean(np.square(action_delta)))),
                  torque_rms_nm=float(np.sqrt(np.mean(np.square(torque)))))
    if not steady.any():
        result.update(passed=False)
        return result
    bias = np.abs(e[steady].mean(0))
    rmse = np.sqrt(np.mean(e[steady]**2, axis=0))
    result.update(steady_mean_actual=v[steady].mean(0).tolist(), steady_abs_bias=bias.tolist(), steady_rmse=rmse.tolist())
    limit = np.array([.04, .04, .08])
    bias_limit = np.array([.04, .04, .06])
    if name == "stand":
        limit = np.array([.01, .01, .025])
        bias_limit = limit
    elif max(abs(x) for x in target) <= .22:
        limit = np.array([.03, .03, .04 if abs(target[2]) == .05 else .05])
        bias_limit = np.array([.02, .02, .02])
    # Startup/stop timing uses a trailing 1s mean, satisfying the criterion for
    # a full subsequent 1s. Never score a fleeting threshold crossing as success.
    response = {}
    transitions = [(3., np.asarray(target)), (11., np.zeros(3)),
                   (16., -np.asarray(target)), (24., np.zeros(3))]
    if name.startswith("sequence_"):
        window = max(1, round(1. / dt))
        for start, desired in transitions:
            tolerance = np.array([.01, .01, .02]) if not desired.any() else np.array([.02, .02, .02])
            good = np.zeros(len(t), dtype=bool)
            for index in range(window - 1, len(t)):
                if t[index - window + 1] >= start:
                    good[index] = np.all(np.abs(v[index-window+1:index+1].mean(0) - desired) <= tolerance)
            elapsed = None
            for index in np.flatnonzero((t >= start) & (t <= start + 2.)):
                if index + window <= len(t) and good[index:index+window].all():
                    elapsed = float(t[index] - start)
                    break
            response[str(start)] = elapsed
        result["response_time_s"] = response
    result["limits"] = dict(steady_rmse=limit.tolist(), steady_abs_bias=bias_limit.tolist())
    result["passed"] = bool(complete and np.all(rmse <= limit) and np.all(bias <= bias_limit)
                            and result["max_tilt_deg"] < 20.
                            and all(value is not None and value <= 2. for value in response.values()))
    return result


def compare_baseline(candidate, baseline):
    """Preservation gate only; targets and preservation are separate decisions.

    Tolerances are engineering guardrails, not statistical significance tests.
    Baseline weak low-speed behavior is deliberately NOT an imitation target.
    """
    for key in ("backend", "physics_profile", "payload_mass_kg", "seeds", "quick", "dt", "duration_s"):
        if candidate.get(key) != baseline.get(key):
            raise ValueError(f"Baseline protocol mismatch: {key}")
    old = {(r["case"], r["seed"]): r for r in baseline["results"]}
    for row in baseline["results"]:
        preserve = row["case"] == "stand" or row["case"].endswith("_0.3") or row["case"].startswith("mixed_")
        if preserve and not row.get("complete"):
            raise ValueError(f"Original baseline failed a preservation case: {row['case']}; check model/inputs first")
    failures = []
    for row in candidate["results"]:
        key = row["case"], row["seed"]
        if key not in old:
            raise ValueError(f"Baseline case missing: {key}")
        ref = old[key]
        if ref.get("complete") and not row.get("complete"):
            failures.append(f"{key}: fall/incomplete rollout")
            continue
        preserve = row["case"] == "stand" or row["case"].endswith("_0.3") or row["case"].startswith("mixed_")
        if not preserve or not ref.get("complete") or not row.get("complete"):
            continue
        bound = np.asarray(ref["steady_rmse"]) * 1.2 + np.array([.005, .005, .01])
        if (np.asarray(row["steady_rmse"]) > bound).any():
            failures.append(f"{key}: existing-motion RMSE regression")
        for metric, relative, absolute in (("action_delta_rms", 1.5, .01), ("torque_rms_nm", 1.3, .5), ("max_tilt_deg", 1.3, 2.)):
            if row[metric] > ref[metric] * relative + absolute:
                failures.append(f"{key}: {metric} regression")
    if len(candidate["results"]) != len(baseline["results"]):
        raise ValueError("Baseline case count mismatch")
    return dict(passed=not failures, failures=failures,
                baseline_policy_sha256=baseline["policy_sha256"],
                note="20% + absolute RMSE tolerance; action/torque/tilt guardrails; not a statistical significance test")
