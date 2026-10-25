"""Separate rigid box with two handles and one excluded fixed grasp joint per env."""

from pxr import Gf, Usd, UsdGeom, UsdPhysics
from typing import Callable
import torch

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaacsim.core.utils.stage import get_current_stage


def _handle_cube(stage, path, position, size):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube.AddScaleOp().Set(Gf.Vec3d(*size))
    cube.CreateDisplayColorAttr([(0.15, 0.15, 0.15)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def spawn_transport_payload(prim_path, cfg, translation=None, orientation=None, **kwargs):
    # The stock shape spawner handles regex cloning. Add each joint separately:
    # copying only Payload would leave a Body0 relationship pointing to env_0.
    prim = sim_utils.spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = get_current_stage()
    parents = sim_utils.find_matching_prim_paths(prim_path.rsplit("/", 1)[0])
    for parent in parents:
        payload_path = parent + "/" + prim_path.rsplit("/", 1)[1]
        robot_path = parent + "/Robot"
        robot = stage.GetPrimAtPath(robot_path)
        links = [p for p in Usd.PrimRange(robot) if p.GetName() == "end_effector" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        bases = [p for p in Usd.PrimRange(robot) if p.GetName() == "base" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(links) != 1 or len(bases) != 1:
            raise RuntimeError(f"Expected one base and rigid end_effector below {robot_path}")
        for side in (-1, 1):
            label = "near" if side < 0 else "far"
            for index, y in enumerate((-0.065, 0.065)):
                _handle_cube(stage, f"{payload_path}/handle_{label}_{index}",
                             (side * 0.27, y, 0.05), (0.065, 0.025, 0.025))
            _handle_cube(stage, f"{payload_path}/handle_{label}_bar", (side * 0.29, 0.0, 0.06), (0.025, 0.155, 0.025))
        joint = UsdPhysics.FixedJoint.Define(stage, payload_path + "/GraspJoint")
        joint.CreateBody0Rel().SetTargets([links[0].GetPath()])
        joint.CreateBody1Rel().SetTargets([stage.GetPrimAtPath(payload_path).GetPath()])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*cfg.grasp_position))
        # Preserve a fixed transform from the EE axes to the base axes of the USD
        # reference pose. Runtime reset aligns the actual payload to this joint.
        cache = UsdGeom.XformCache()
        ee_rotation = cache.GetLocalToWorldTransform(links[0]).ExtractRotationQuat()
        base_rotation = cache.GetLocalToWorldTransform(bases[0]).ExtractRotationQuat()
        offset = ee_rotation.GetInverse() * base_rotation
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(offset))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateCollisionEnabledAttr().Set(False)
        # Start the USD reference pose with coincident joint anchors. The runtime
        # reset repeats this using the actual articulation state after DOF reset.
        ee_position = cache.GetLocalToWorldTransform(links[0]).ExtractTranslation()
        matrix = Gf.Matrix4d(1.0).SetRotate(base_rotation)
        matrix.SetTranslateOnly(ee_position - matrix.TransformDir(Gf.Vec3d(*cfg.grasp_position)))
        parent_matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent))
        payload_xform = UsdGeom.Xformable(stage.GetPrimAtPath(payload_path))
        payload_xform.ClearXformOpOrder()
        payload_xform.AddTransformOp().Set(matrix * parent_matrix.GetInverse())
    return prim


@configclass
class TransportPayloadCfg(sim_utils.CuboidCfg):
    func: Callable = spawn_transport_payload
    grasp_position: tuple[float, float, float] = (-0.29, 0.0, 0.06)


def randomize_transport_mass(env, env_ids, mass_range=(0.0, 10.0)):
    payload = env.scene["payload"]
    ids = torch.arange(env.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
    masses = payload.root_physx_view.get_masses().clone()
    inertia = payload.root_physx_view.get_inertias().clone()
    # PhysX dynamic bodies require positive mass. 1 g approximates the zero end.
    sampled = torch.empty(len(ids), device="cpu").uniform_(*mass_range).clamp_min(0.001)
    old = masses[ids].reshape(len(ids), -1)[:, 0].clone()
    scale = sampled / old
    masses[ids] *= scale.reshape((-1,) + (1,) * (masses.ndim - 1))
    inertia[ids] *= scale.reshape((-1,) + (1,) * (inertia.ndim - 1))
    payload.root_physx_view.set_masses(masses, ids)
    payload.root_physx_view.set_inertias(inertia, ids)
