"""Display the actual payload guidance wrench; visual geoms exert no forces."""

import mujoco
import numpy as np


def draw_guidance(scene, payload_position, world_wrench, distance, mode, show_label=True):
    """Append debug arrows/text to a render scene or an empty viewer user_scn."""
    origin = np.asarray(payload_position, float) + [0, 0, .35]

    def geometry(kind, color):
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, kind, np.zeros(3), origin, np.eye(3).ravel(), np.asarray(color, np.float32))
        scene.ngeom += 1
        return geom

    force = np.asarray(world_wrench)[:3]
    norm = np.linalg.norm(force)
    if norm > .01:
        arrow = geometry(mujoco.mjtGeom.mjGEOM_ARROW, [1, .2, .05, 1])
        if arrow is not None:
            end = origin + force / norm * min(1.5, .03 * norm)
            mujoco.mjv_connector(arrow, mujoco.mjtGeom.mjGEOM_ARROW, .025, origin, end)
    torque = world_wrench[5]
    if abs(torque) > .01:
        sign = np.sign(torque)
        center = origin + [0, 0, .08]
        for i in range(9):
            angle = sign * np.array([i, i + 1]) * np.pi / 7
            points = center + .25 * np.array([[np.cos(a), np.sin(a), 0] for a in angle])
            kind = mujoco.mjtGeom.mjGEOM_ARROW if i == 8 else mujoco.mjtGeom.mjGEOM_LINE
            geom = geometry(kind, [.1, .7, 1, 1])
            if geom is not None:
                mujoco.mjv_connector(geom, kind, .02 if i == 8 else 3., points[0], points[1])
    if not show_label:
        return
    label = geometry(mujoco.mjtGeom.mjGEOM_LABEL, [1, 1, 1, 1])
    if label is not None:
        label.pos[:] = origin + [0, 0, .35]
        label.label = (f"{mode} | F=({force[0]:.0f},{force[1]:.0f})N "
                       f"Tz={torque:.1f}Nm | {distance:.1f}m")


def draw_leader_follow(scene, positions, commands, follower_wrench, distance, hold=False):
    """Show measured follower reaction, never an artificial guidance force."""
    wrenches = np.asarray(follower_wrench)
    if wrenches.ndim == 1:
        wrenches = wrenches[None, :]
    for i, wrench in enumerate(wrenches):
        draw_guidance(scene, np.asarray(positions[i + 1]) + [0, 0, .35], wrench, distance,
                      f"Follower {i + 1} FT", show_label=len(positions) == 2)
    names = ["LEADER"] + ["FOLLOWER HOLD" if hold else "FOLLOWER FT"] * (len(positions) - 1)
    for index, name in enumerate(names):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        color = [1, .65, .1, 1] if index == 0 else [.2, .8, 1, 1]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL, np.zeros(3),
                           np.asarray(positions[index]) + [0, 0, .6], np.eye(3).ravel(), np.asarray(color, np.float32))
        geom.label = f"{name} vx={commands[index][0]:.2f}"
        if len(positions) > 2:
            geom.label = (f"LEADER vx={commands[0][0]:.2f} | {distance:.1f}m" if index == 0 else
                          f"F{index} {'HOLD' if hold else 'FT'} vx={commands[index][0]:.2f} Fx={wrenches[index - 1, 0]:.0f}N")
        scene.ngeom += 1
