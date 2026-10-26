"""Read-only grasp diagnostics. Wrenches use [force, torque], SI units."""

import csv

import mujoco
import numpy as np


def normalized_input(world, rotation):
    """WORLD wrench at EE -> body axes, preserving the EE moment origin."""
    local = np.r_[rotation.T @ world[:3], rotation.T @ world[3:]]
    return local, np.clip(local[[0, 1, 5]] / [40., 40., 10.], -1., 1.)


def payload_balance(model, data, payload, robots, reactions):
    """Check recovered EE reactions against the payload equations of motion.

    Include every non-grasp constraint (contacts etc.) and all smooth forces.
    The six residuals are generalized forces, not six world wrench components.
    Run after mj_forward, before changing controls. This is a solver/extraction
    consistency check, not proof that the weld models a real frictional grasp.
    """
    joint = model.body_jntadr[payload]
    dofs = np.arange(model.jnt_dofadr[joint], model.jnt_dofadr[joint] + 6)
    jacobian = data.efc_J.reshape(data.nefc, model.nv)
    weld_generalized = np.zeros(model.nv)
    recovered = np.zeros(model.nv)
    total_at_com = np.zeros(6)
    for robot, wrench in zip(robots, reactions):
        rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == robot.eq)
        weld_generalized += jacobian[rows].T @ data.efc_force[rows]
        ee = model.eq_obj1id[robot.eq]
        linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, linear, angular, data.xpos[ee], payload)
        # Input is payload->robot; the payload receives its opposite.
        recovered -= linear.T @ wrench[:3] + angular.T @ wrench[3:]
        arm = data.xpos[ee] - data.xipos[payload]
        total_at_com -= np.r_[wrench[:3], wrench[3:] + np.cross(arm, wrench[:3])]
    mass_acceleration = np.zeros(model.nv)
    mujoco.mj_mulM(model, data, mass_acceleration, data.qacc)
    other_constraints = data.qfrc_constraint - weld_generalized
    residual = (mass_acceleration - data.qfrc_smooth - other_constraints - recovered)[dofs]
    extraction_error = (recovered - weld_generalized)[dofs]
    return total_at_com, residual, extraction_error


class GraspAudit:
    """Shadow logging never changes actor inputs, controller state or physics."""

    def __init__(self, stream, count, settle_seconds):
        self.writer = csv.writer(stream)
        self.writer.writerow(["time_s", "robot", "calibration_window",
                              *[f"grasp_world_{a}" for a in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")],
                              *[f"grasp_body_{a}" for a in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")],
                              "grasp_cx", "grasp_cy", "grasp_cz", "external_share_cx", "external_share_cy", "external_share_cz",
                              *[f"payload_grasp_com_{a}" for a in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")],
                              *[f"balance_generalized_{i}" for i in range(6)],
                              *[f"extraction_generalized_{i}" for i in range(6)]])
        self.settle = settle_seconds
        self.bias_sum = np.zeros((count, 6))
        self.bias_count = np.zeros(count, int)
        self.error_square = np.zeros((count, 3))
        self.count = np.zeros(count, int)
        self.balance_max = np.zeros(6)
        self.extraction_max = np.zeros(6)

    def sample(self, time, index, reaction, rotation, oracle, total, balance, extraction):
        local, normalized = normalized_input(reaction, rotation)
        _, oracle_normalized = normalized_input(oracle, rotation)
        calibration = max(0., self.settle - 1.) <= time < self.settle
        if calibration:
            self.bias_sum[index] += reaction
            self.bias_count[index] += 1
        if time >= self.settle:
            self.error_square[index] += (normalized - oracle_normalized) ** 2
            self.count[index] += 1
        self.balance_max = np.maximum(self.balance_max, np.abs(balance))
        self.extraction_max = np.maximum(self.extraction_max, np.abs(extraction))
        self.writer.writerow([time, index, int(calibration), *reaction, *local, *normalized,
                              *oracle_normalized, *total, *balance, *extraction])

    def summary(self):
        return {"static_world_wrench_mean": [
                    (self.bias_sum[i] / n).tolist() if n else None for i, n in enumerate(self.bias_count)],
                "static_samples": self.bias_count.tolist(), "post_settle_samples": self.count.tolist(),
                "normalized_grasp_vs_external_share_rmse": [
                    np.sqrt(self.error_square[i] / n).tolist() if n else None for i, n in enumerate(self.count)],
                "max_abs_payload_balance_generalized": self.balance_max.tolist(),
                "max_abs_extraction_generalized": self.extraction_max.tolist(),
                "notes": "Raw unfiltered EE reaction; no bias subtraction. Input difference is not a failure criterion. "
                         "Balance includes other constraints and smooth forces; maxima include landing. "
                         "COM wrench is robot->payload WORLD, raw grasp is payload->robot WORLD at EE."}
