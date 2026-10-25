"""Compile 2--4 independently named GO2-PIPER robots and one shared rigid box.

The robots stand side by side facing +X. No world weld, mocap, base teleport,
or kinematic payload is used. Grasp welds are aligned after nominal-arm FK.
"""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from paint_loco import JOINT_NAMES, MUJOCO_ROOT, named_id


def build_scene(count=2, spacing=1.2, mass=.5, timestep=.005):
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
    return ET.tostring(root, encoding="unicode"), assets, offsets


def initialize_scene(model, data, defaults, offsets):
    """Align the real free payload and both sides of each weld before activation."""
    ee_ids = []
    for index, y in enumerate(offsets):
        prefix = f"r{index}/"
        base = named_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_link")
        free = model.body_jntadr[base]
        address = model.jnt_qposadr[free]
        data.qpos[address:address + 7] = [0, y, .35, 1, 0, 0, 0]
        joints = [named_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + n) for n in JOINT_NAMES]
        data.qpos[model.jnt_qposadr[joints]] = defaults
        ee_ids.append(named_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "end_effector"))
    mujoco.mj_forward(model, data)
    payload = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    address = model.jnt_qposadr[model.body_jntadr[payload]]
    center = data.xpos[ee_ids].mean(0) + [.29, 0, -.06]
    data.qpos[address:address + 7] = [*center, 1, 0, 0, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    # Body weld relpose is body2 pose in body1 axes (anchor defaults to body2 origin).
    for index, ee in enumerate(ee_ids):
        eq = named_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"grasp_{index}")
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
