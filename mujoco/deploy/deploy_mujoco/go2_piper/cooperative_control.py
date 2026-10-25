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
    def __init__(self, command, max_speed, max_yaw):
        self.original = np.asarray(command, float).copy()
        self.command = self.original.copy()
        self.max_speed, self.max_yaw = max_speed, max_yaw
        self.paused = False
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
                self.command[0] += .05 if key == ord("W") else -.05
            elif key in (ord("A"), ord("D")):
                self.command[1] += .05 if key == ord("A") else -.05
            elif key in (ord("Q"), ord("E")):
                self.command[2] += .1 if key == ord("Q") else -.1
            else:
                return
            self.command[:2] = np.clip(self.command[:2], -self.max_speed, self.max_speed)
            self.command[2] = np.clip(self.command[2], -self.max_yaw, self.max_yaw)
            print(f"[KEY] world command={self.command.round(2)}; paused={self.paused}", flush=True)

    def snapshot(self):
        with self.lock:
            return self.command.copy(), self.paused
