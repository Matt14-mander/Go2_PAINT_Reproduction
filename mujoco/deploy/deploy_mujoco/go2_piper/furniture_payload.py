"""Procedural furniture and ideal attachment points; no downloaded assets.

One leader in front, one to three followers behind. The single free furniture
body has mass/inertia/contact geometry. Virtual handles are rigid extensions,
not an actuator, world anchor, or externally driven trajectory.
"""

from copy import deepcopy
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from paint_loco import JOINT_NAMES, named_id


def vector(value):
    return " ".join(f"{v:.12g}" for v in value)


def build_furniture(root, assets, count, spacing, mass, defaults, kind):
    world, equality, contacts = (root.find(tag) for tag in ("worldbody", "equality", "contact"))
    root.set("model", f"paint_{kind}_leader_follow")
    world.remove(world.find("body[@name='payload']"))
    equality.clear()
    for pair in list(contacts):
        if pair.get("body2") == "payload":
            contacts.remove(pair)
    follower_y = (np.arange(count - 1) - (count - 2) / 2) * spacing
    half_length = max(.85, float(max(abs(follower_y))) + .35)
    # Furniture spans x=+/-.43. Keep both robot bodies outside its silhouette;
    # virtual grip extensions bridge the remaining gap to the normal arm pose.
    positions = np.array([[.98, 0.], *[[-1.02, y] for y in follower_y]])
    for index, xy in enumerate(positions):
        world.find(f"body[@name='r{index}/base_link']").set("pos", vector([*xy, .35]))
    leader = world.find("body[@name='r0/base_link']")
    mount = ET.SubElement(leader, "body", name="r0/tow_mount", pos="-.32 0 .12")
    ET.SubElement(mount, "geom", type="capsule", fromto=".15 0 -.1 0 0 0", size=".015",
                  mass=".05", rgba=".95 .6 .1 1")
    ET.SubElement(mount, "site", name="r0/tow_site", size=".018", rgba="1 .6 .1 1")
    probe_root = deepcopy(root)
    probe = mujoco.MjModel.from_xml_string(ET.tostring(probe_root, encoding="unicode"), assets)
    state = mujoco.MjData(probe)
    for index in range(count):
        joints = [named_id(probe, mujoco.mjtObj.mjOBJ_JOINT, f"r{index}/{n}") for n in JOINT_NAMES]
        state.qpos[probe.jnt_qposadr[joints]] = defaults
    mujoco.mj_forward(probe, state)
    attachments = [state.xpos[named_id(probe, mujoco.mjtObj.mjOBJ_BODY,
                                     "r0/tow_mount" if i == 0 else f"r{i}/end_effector")].copy()
                   for i in range(count)]
    center = np.array([0., 0., .65])
    body = ET.SubElement(world, "body", name="payload", pos=vector(center))
    ET.SubElement(body, "freejoint", name="payload/free")
    # Every collision primitive contributes volume-proportional mass. MuJoCo
    # computes aggregate COM/inertia from the actual compound geometry.
    pieces = []

    def box(name, position, size, color):
        geom = ET.SubElement(body, "geom", name=f"payload/{name}", type="box",
                             pos=vector(position), size=vector(size), rgba=color, friction=".8 .02 .002")
        pieces.append((geom, float(8 * np.prod(size))))

    if kind == "sofa":
        box("frame", [0, 0, -.08], [.40, half_length, .09], ".12 .06 .035 1")
        box("back", [-.34, 0, .18], [.09, half_length, .26], ".16 .07 .035 1")
        usable = half_length - .14
        for index, y in enumerate(np.linspace(-2 * usable / 3, 2 * usable / 3, 3)):
            box(f"cushion_{index}", [.045, y, .055], [.30, usable / 3 - .008, .07], ".28 .15 .08 1")
        for sign in (-1, 1):
            box(f"armrest_{sign}", [0, sign * (half_length - .055), .09], [.40, .075, .16], ".18 .08 .045 1")
        leg_top = -.17
    else:
        box("tabletop", [0, 0, .19], [.43, half_length, .045], ".28 .16 .075 1")
        box("apron", [0, 0, .105], [.39, half_length - .045, .04], ".24 .13 .06 1")
        leg_top = .065
    bottom = -.34  # initial bottom at .31m, leaving room for natural settling
    for xsign in (-1, 1):
        for ysign in (-1, 1):
            box(f"leg_{xsign}_{ysign}", [xsign * .31, ysign * (half_length - .12), (leg_top + bottom) / 2],
                [.035, .035, (leg_top - bottom) / 2], ".12 .075 .045 1")
    volume = sum(v for _, v in pieces)
    for geom, value in pieces:
        geom.set("mass", str(mass * value / volume))
    # Ideal, non-colliding grips expose the simplifying assumption visibly.
    # Their mass is zero; all payload mass is accounted for in the furniture.
    for index, point in enumerate(attachments):
        local = point - center
        surface = np.array([.43 if index == 0 else -.43, point[1], point[2] - center[2]])
        ET.SubElement(body, "geom", name=f"payload/virtual_grip_{index}", type="capsule", size=".012",
                      fromto=vector(np.r_[surface, local]), mass="0", contype="0", conaffinity="0",
                      rgba=".65 .7 .72 1")
        ET.SubElement(body, "site", name=f"payload/handle_{index}", pos=vector(local),
                      size=".015", rgba="1 .55 .1 1" if index == 0 else ".2 .75 1 1")
        if index == 0:
            ET.SubElement(equality, "connect", name="grasp_0", body1="r0/tow_mount", body2="payload",
                          anchor="0 0 0", active="false", solref=".01 1", solimp=".99 .999 .001")
            ET.SubElement(contacts, "exclude", body1="r0/tow_mount", body2="payload")
        else:
            ET.SubElement(equality, "weld", name=f"grasp_{index}", body1=f"r{index}/end_effector",
                          body2="payload", active="false", solref=".01 1", solimp=".99 .999 .001", torquescale="1")
            for link in ("end_effector", "link7", "link8"):
                ET.SubElement(contacts, "exclude", body1=f"r{index}/{link}", body2="payload")
    return ET.tostring(root, encoding="unicode"), assets, positions
