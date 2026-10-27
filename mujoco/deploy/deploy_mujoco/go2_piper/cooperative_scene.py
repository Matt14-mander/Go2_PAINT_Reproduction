"""Compile 2--4 independently named GO2-PIPER robots and a shared free payload.

Layouts: side by side, two in tandem, or a leader and followers around furniture.
No world weld, mocap, base teleport, or kinematic payload is used. Attachments
are aligned after nominal-arm FK.
"""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from paint_loco import JOINT_NAMES, MUJOCO_ROOT, named_id


def build_scene(count=2, spacing=1.2, mass=.5, timestep=.005, layout="side-by-side", defaults=None, payload_kind=None):
    if layout not in ("side-by-side", "leader-follow", "furniture"):
        raise ValueError("Unknown robot layout")
    payload_kind = payload_kind or ("sofa" if layout == "furniture" else "tow-bar" if layout == "leader-follow" else "box")
    if ((layout == "furniture" and (payload_kind not in ("sofa", "table") or defaults is None))
            or (layout == "leader-follow" and payload_kind != "tow-bar")
            or (layout == "side-by-side" and payload_kind != "box")):
        raise ValueError("Furniture layout requires sofa/table and nominal joints; other layouts require their original payload")
    if layout == "leader-follow" and (count != 2 or defaults is None):
        raise ValueError("Leader-follow requires two robots and nominal joint positions")
    if not 2 <= count <= 4 or spacing < .9 or not .001 <= mass <= 10:
        raise ValueError("Expected 2--4 robots, spacing >= .9 m, payload mass .001--10 kg")
    source = ET.parse(MUJOCO_ROOT / "robots/go2_piper/go2piper.xml").getroot()
    for mesh in source.findall("./asset/mesh"):
        mesh.set("name", mesh.get("name", Path(mesh.get("file")).stem))
    names = {node.get("name") for node in source.iter() if node.get("name")}
    names.update(node.get("class") for node in source.iter("default") if node.get("class"))
    root = ET.Element("mujoco", model="paint_shared_transport")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep=str(timestep), integrator="implicitfast", jacobian="dense",
                  cone="elliptic", iterations="100", impratio="100")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    ET.SubElement(visual, "headlight", ambient=".45 .45 .45", diffuse=".7 .7 .7")
    sections = {tag: ET.SubElement(root, tag) for tag in ("default", "asset", "worldbody", "actuator", "sensor", "contact", "equality")}
    ET.SubElement(sections["asset"], "texture", name="ground_tex", type="2d", builtin="checker",
                  rgb1=".2 .25 .3", rgb2=".3 .35 .4", width="256", height="256")
    ET.SubElement(sections["asset"], "material", name="ground", texture="ground_tex", texrepeat="4 4", texuniform="true")
    ET.SubElement(sections["worldbody"], "geom", name="floor", type="plane", size="0 0 .05", material="ground", friction="1 .02 .002")
    ET.SubElement(sections["worldbody"], "light", pos="0 0 5", dir="0 0 -1", directional="true")
    # Non-colliding one-meter progress marks: reveal travel even as camera follows.
    for meter in range(-5, 41):
        ET.SubElement(sections["worldbody"], "geom", name=f"road_mark_{meter}", type="box",
                      pos=f"{meter} 0 .001", size=".012 4 .0005", contype="0", conaffinity="0",
                      rgba=".9 .65 .15 .7" if meter % 5 == 0 else ".7 .75 .8 .5")
    reference_attrs = {"name", "class", "childclass", "mesh", "material", "texture", "joint", "joint1", "joint2",
                       "body", "body1", "body2", "site", "site1", "site2", "objname", "target", "actuator", "tendon"}
    offsets = (np.arange(count) - (count - 1) / 2) * spacing
    for index in range(count):
        prefix = f"r{index}/"
        clone = deepcopy(source)
        for node in clone.iter():
            for key, value in list(node.attrib.items()):
                if key in reference_attrs and value in names:
                    node.set(key, prefix + value)
                if key == "file":
                    node.set(key, Path(value).name)
        base = clone.find("./worldbody/body")
        for orientation in ("euler", "axisangle", "xyaxes", "zaxis"):
            base.attrib.pop(orientation, None)
        base.set("pos", f"0 {offsets[index]} .35")
        base.set("quat", "1 0 0 0")
        for tag, destination in sections.items():
            original = clone.find(tag)
            if original is not None:
                destination.extend(list(original))
        ET.SubElement(sections["equality"], "weld", name=f"grasp_{index}", body1=prefix + "end_effector",
                      body2="payload", active="false", solref=".01 1", solimp=".99 .999 .001", torquescale="1")
        # The weld represents a closed grasp. Its EE/finger collision shapes
        # overlap the handle; retain all other robot/payload contact pairs.
        for grasp_link in ("end_effector", "link7", "link8"):
            ET.SubElement(sections["contact"], "exclude", body1=prefix + grasp_link, body2="payload")
    payload = ET.SubElement(sections["worldbody"], "body", name="payload", pos=".6 0 1")
    ET.SubElement(payload, "freejoint", name="payload/free")
    half_span = float(max(abs(offsets)))
    ET.SubElement(payload, "inertial", pos="0 0 0", mass=str(mass),
                  diaginertia=" ".join(str(mass / 3 * v) for v in
                                       ((half_span - .1)**2 + .04**2, .25**2 + .04**2, .25**2 + (half_span - .1)**2)))
    ET.SubElement(payload, "geom", name="payload/box", type="box", size=f".25 {half_span - .1} .04",
                  rgba=".65 .35 .12 1", friction=".8 .02 .002")
    for index, y in enumerate(offsets):
        ET.SubElement(payload, "site", name=f"payload/handle_{index}", pos=f"-.29 {y} .06", size=".012", rgba="1 .3 .1 1")
        for dx in (-.065, .065):
            ET.SubElement(payload, "geom", type="box", pos=f"{-.265} {y + dx} .05", size=".0325 .0125 .0125", rgba=".12 .12 .12 1")
        ET.SubElement(payload, "geom", type="box", pos=f"-.29 {y} .06", size=".0125 .0775 .0125", rgba=".12 .12 .12 1")
    assets = {p.name: p.read_bytes() for p in (MUJOCO_ROOT / "robots/go2_piper/assets").iterdir() if p.is_file()}
    if layout == "furniture":
        from furniture_payload import build_furniture
        return build_furniture(root, assets, count, spacing, mass, defaults, payload_kind)
    if layout == "leader-follow":
        return _leader_follow_scene(root, assets, spacing, mass, defaults)
    return ET.tostring(root, encoding="unicode"), assets, offsets


def _leader_follow_scene(root, assets, spacing, mass, defaults):
    """Rear ball hitch on r0, finite-mass tow bar, r1's normal forward grasp.

    Build the bar from actual nominal FK, so neither endpoint is teleported
    after activation. Only the mounting pair and closed fingers are excluded
    from bar collision; the rest of both robots remain physical obstacles.
    """
    world, equality, contacts = (root.find(tag) for tag in ("worldbody", "equality", "contact"))
    root.set("model", "paint_leader_follow")
    world.remove(world.find("body[@name='payload']"))
    equality.clear()
    for pair in list(contacts):
        if pair.get("body1", "").startswith("r0/") and pair.get("body2") == "payload":
            contacts.remove(pair)
    offsets = np.array([spacing / 2, -spacing / 2])  # r0 leader ahead, r1 follower behind
    for index, x in enumerate(offsets):
        world.find(f"body[@name='r{index}/base_link']").set("pos", f"{x} 0 .35")
    leader = world.find("body[@name='r0/base_link']")
    mount = ET.SubElement(leader, "body", name="r0/tow_mount", pos="-.32 0 .12")
    ET.SubElement(mount, "geom", type="capsule", fromto=".15 0 -.1 0 0 0", size=".015",
                  mass=".05", rgba=".9 .55 .1 1")
    ET.SubElement(mount, "site", name="r0/tow_site", size=".025", rgba="1 .5 .1 1")
    # FK in a temporary model with no payload/constraints. This is construction,
    # not a running simulation reset or a kinematic force source.
    probe_root = deepcopy(root)
    probe_root.find("contact").clear()  # payload pair references become valid below
    probe = mujoco.MjModel.from_xml_string(ET.tostring(probe_root, encoding="unicode"), assets)
    state = mujoco.MjData(probe)
    for index in range(2):
        joints = [named_id(probe, mujoco.mjtObj.mjOBJ_JOINT, f"r{index}/{n}") for n in JOINT_NAMES]
        state.qpos[probe.jnt_qposadr[joints]] = defaults
    mujoco.mj_forward(probe, state)
    front = state.xpos[named_id(probe, mujoco.mjtObj.mjOBJ_BODY, "r0/tow_mount")].copy()
    rear = state.xpos[named_id(probe, mujoco.mjtObj.mjOBJ_BODY, "r1/end_effector")].copy()
    if front[0] - rear[0] < .2:
        raise ValueError("Tow bar needs at least .2m clearance; increase --spacing")
    center = (front + rear) / 2
    vec = lambda value: " ".join(f"{v:.12g}" for v in value)
    payload = ET.SubElement(world, "body", name="payload", pos=vec(center))
    ET.SubElement(payload, "freejoint", name="payload/free")
    ET.SubElement(payload, "geom", name="payload/tow_bar", type="capsule", size=".018",
                  fromto=vec(np.r_[rear - center, front - center]), mass=str(mass), rgba=".2 .65 .9 1")
    for index, point in enumerate((front, rear)):
        ET.SubElement(payload, "site", name=f"payload/handle_{index}", pos=vec(point - center),
                      size=".02", rgba="1 .3 .1 1")
    ET.SubElement(equality, "connect", name="grasp_0", body1="r0/tow_mount", body2="payload",
                  anchor="0 0 0", active="false", solref=".01 1", solimp=".99 .999 .001")
    ET.SubElement(equality, "weld", name="grasp_1", body1="r1/end_effector", body2="payload",
                  active="false", solref=".01 1", solimp=".99 .999 .001", torquescale="1")
    ET.SubElement(contacts, "exclude", body1="r0/tow_mount", body2="payload")
    return ET.tostring(root, encoding="unicode"), assets, offsets


def initialize_scene(model, data, defaults, offsets, layout="side-by-side"):
    """Align the free payload and attachment constraints before activation."""
    ee_ids = []
    for index, y in enumerate(offsets):
        prefix = f"r{index}/"
        base = named_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link")
        free = model.body_jntadr[base]
        address = model.jnt_qposadr[free]
        xy = y if layout == "furniture" else [y, 0] if layout == "leader-follow" else [0, y]
        data.qpos[address:address + 7] = [*xy, .35, 1, 0, 0, 0]
        joints = [named_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n) for n in JOINT_NAMES]
        data.qpos[model.jnt_qposadr[joints]] = defaults
        ee_ids.append(named_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "end_effector"))
    mujoco.mj_forward(model, data)
    payload = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    address = model.jnt_qposadr[model.body_jntadr[payload]]
    if layout == "side-by-side":
        center = data.xpos[ee_ids].mean(0) + [.29, 0, -.06]
        data.qpos[address:address + 7] = [*center, 1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    # Body weld relpose is body2 pose in body1 axes (anchor defaults to body2 origin).
    for index, ee in enumerate(ee_ids):
        eq = named_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{index}")
        if model.eq_type[eq] == mujoco.mjtEq.mjEQ_CONNECT:
            data.eq_active[eq] = True
            continue  # ball hitch anchors were aligned in the constructed XML
        inverse = np.empty(4)
        relative = np.empty(4)
        mujoco.mju_negQuat(inverse, data.xquat[ee])
        mujoco.mju_mulQuat(relative, inverse, data.xquat[payload])
        model.eq_data[eq, :3] = 0
        model.eq_data[eq, 3:6] = data.xmat[ee].reshape(3, 3).T @ (data.xpos[payload] - data.xpos[ee])
        model.eq_data[eq, 6:10] = relative
        data.eq_active[eq] = True
    mujoco.mj_forward(model, data)
    return payload
