"""Reference-only desktop motion planning; no MuJoCo or policy dependencies."""

from threading import Lock
import numpy as np


def motion_profile(time_s, duration, settle=2., ramp=3.):
    return float(np.clip(min((time_s - settle) / ramp, (duration - time_s) / ramp), 0, 1))


def velocity_target(requested, time_s, duration, position, origin, settle=2., ramp=3., distance=None):
    """World [vx,vy,wz], optionally brake near a straight-line distance goal."""
    requested = np.asarray(requested, float)
    target = requested.copy()
    if distance is not None:
        speed = np.linalg.norm(requested[:2])
        if speed <= 0 or requested[2] != 0:
            raise ValueError("Distance goal requires nonzero translation and zero yaw rate")
        direction = requested[:2] / speed
        displacement = np.asarray(position)[:2] - np.asarray(origin)[:2]
        progress = float(displacement @ direction)
        cross_track = displacement - progress * direction
        target[:2] = direction * np.clip(.6 * (distance - progress), -speed, speed) - .5 * cross_track
        magnitude = np.linalg.norm(target[:2])
        if magnitude > speed:
            target[:2] *= speed / magnitude
    return target * motion_profile(time_s, duration, settle, ramp)


def slew_command(current, target, dt, acceleration=.15, yaw_acceleration=.4):
    """Limit group acceleration, including keyboard changes and stop requests."""
    result = np.asarray(current, float).copy()
    difference = np.asarray(target)[:2] - result[:2]
    magnitude = np.linalg.norm(difference)
    if magnitude:
        result[:2] += difference * min(1., acceleration * dt / magnitude)
    result[2] += np.clip(target[2] - result[2], -yaw_acceleration * dt, yaw_acceleration * dt)
    return result


class DesktopCommand:
    """MuJoCo key_callback writes commands; physics thread reads snapshots."""
    def __init__(self, command, max_speed, max_yaw, linear_step=.05, yaw_step=.1, label="world command"):
        self.original = np.asarray(command, float).copy()
        self.command = self.original.copy()
        self.max_speed, self.max_yaw = max_speed, max_yaw
        self.paused = False
        self.linear_step, self.yaw_step, self.label = linear_step, yaw_step, label
        self.lock = Lock()

    def on_key(self, key):
        with self.lock:
            if key == ord("P"):
                self.paused = not self.paused
            elif key == ord("X"):
                self.command[:] = 0
            elif key == ord("R"):
                self.command[:] = self.original
            elif key in (ord("W"), ord("S")):
                self.command[0] += self.linear_step if key == ord("W") else -self.linear_step
            elif key in (ord("A"), ord("D")):
                self.command[1] += self.linear_step if key == ord("A") else -self.linear_step
            elif key in (ord("Q"), ord("E")):
                self.command[2] += self.yaw_step if key == ord("Q") else -self.yaw_step
            else:
                return
            self.command[:2] = np.clip(self.command[:2], -self.max_speed, self.max_speed)
            self.command[2] = np.clip(self.command[2], -self.max_yaw, self.max_yaw)
            print(f"[KEY] {self.label}={self.command.round(2)}; paused={self.paused}", flush=True)

    def snapshot(self):
        with self.lock:
            return self.command.copy(), self.paused


class WrenchFollower:
    """Reference FT-based admittance, not the PAINT learned intent estimator.

    Each robot only receives its own world grasp reaction and base rotation.
    Calibrate stationary load bias before external guidance, low-pass the wrench,
    then produce a speed request. No common velocity or formation state is read.
    """
    def __init__(self, gain=1., time_constant=.25):
        self.gain, self.time_constant = gain, time_constant
        self.bias = np.zeros(6)
        self.bias_samples = 0
        self.filtered = np.zeros(3)
        self.command = np.zeros(3)

    def update(self, world_wrench, base_rotation, dt, calibrating=False, gain_bias=True):
        world_wrench = np.asarray(world_wrench, float)
        if calibrating:
            if gain_bias:
                self.bias_samples += 1
                self.bias += (world_wrench - self.bias) / self.bias_samples
            self.filtered[:] = 0
            self.command[:] = 0
            return self.command.copy()
        corrected = world_wrench - self.bias
        force = np.asarray(base_rotation).T @ corrected[:3]
        torque = np.asarray(base_rotation).T @ corrected[3:]
        normalized = np.array([force[0] / 40., force[1] / 40., torque[2] / 10.])
        alpha = 1 - np.exp(-dt / self.time_constant)
        self.filtered += alpha * (normalized - self.filtered)
        self.command = slew_command(self.command, self.filtered * self.gain, dt)
        return self.command.copy()
