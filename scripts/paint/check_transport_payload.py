"""OpenUSD authoring checks with a stub shape spawner; NOT a PhysX simulation.

Requires torch and usd-core. Runs in a fresh process, without launching Isaac.
"""

import importlib.util
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[2]
stage = Usd.Stage.CreateInMemory()


def module(name):
    value = ModuleType(name)
    sys.modules[name] = value
    return value


def matching(pattern):
    # Isaac matches each path component separately; .* cannot cross a slash.
    pattern = pattern.replace(".*", "[^/]*")
    return [str(p.GetPath()) for p in stage.Traverse() if re.fullmatch(pattern, str(p.GetPath()))]


def spawn_cuboid(path, cfg, **kwargs):
    for parent in matching(path.rsplit("/", 1)[0]):
        root = UsdGeom.Xform.Define(stage, parent + "/Payload")
        UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
        UsdGeom.Cube.Define(stage, parent + "/Payload/geometry")
    return stage.GetPrimAtPath("/World/envs/env_0/Payload")


module("isaaclab")
sim = module("isaaclab.sim")
sim.CuboidCfg = type("CuboidCfg", (), {})
sim.spawn_cuboid = spawn_cuboid
sim.find_matching_prim_paths = matching
module("isaaclab.utils").configclass = lambda cls: cls
for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils"):
    module(name)
module("isaacsim.core.utils.stage").get_current_stage = lambda: stage
path = ROOT / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp/transport_payload.py"
spec = importlib.util.spec_from_file_location("transport_payload_tested", path)
payload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(payload)


class PayloadChecks(unittest.TestCase):
    def test_separate_rigid_body_and_per_environment_grasp(self):
        for i in range(2):
            parent = f"/World/envs/env_{i}"
            env = UsdGeom.Xform.Define(stage, parent)
            env.AddTranslateOp().Set(Gf.Vec3d(i * 4., 0, 0))
            robot = UsdGeom.Xform.Define(stage, parent + "/Robot")
            robot.AddRotateZOp().Set(35.)
            for name in ("base", "end_effector"):
                link = UsdGeom.Xform.Define(stage, parent + "/Robot/" + name)
                UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
                if name == "end_effector":
                    link.AddTranslateOp().Set(Gf.Vec3d(.3, .2, .8))
                    link.AddRotateYOp().Set(25.)
        cfg = payload.TransportPayloadCfg()
        payload.spawn_transport_payload("/World/envs/env_.*/Payload", cfg)
        cache = UsdGeom.XformCache()
        for i in range(2):
            parent = f"/World/envs/env_{i}"
            body = stage.GetPrimAtPath(parent + "/Payload")
            self.assertTrue(body.HasAPI(UsdPhysics.RigidBodyAPI))
            handles = [child for child in body.GetChildren() if child.GetName().startswith("handle_")]
            self.assertEqual(len(handles), 6)
            joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(parent + "/Payload/GraspJoint"))
            self.assertEqual(str(joint.GetBody0Rel().GetTargets()[0]), parent + "/Robot/end_effector")
            self.assertEqual(str(joint.GetBody1Rel().GetTargets()[0]), parent + "/Payload")
            self.assertTrue(joint.GetExcludeFromArticulationAttr().Get())
            self.assertFalse(joint.GetCollisionEnabledAttr().Get())
            ee = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(parent + "/Robot/end_effector"))
            box = cache.GetLocalToWorldTransform(body)
            anchor = box.Transform(Gf.Vec3d(*cfg.grasp_position))
            self.assertLess((anchor - ee.ExtractTranslation()).GetLength(), 1e-6)
            grasp = ee.ExtractRotationQuat() * Gf.Quatd(joint.GetLocalRot0Attr().Get())
            self.assertLess(abs(abs(Gf.Dot(grasp, box.ExtractRotationQuat())) - 1), 1e-6)

    def test_mass_and_inertia_change_only_selected_environment(self):
        class View:
            mass = torch.tensor([[.5], [.5]])
            inertia = torch.ones(2, 9)
            def get_masses(self): return self.mass
            def get_inertias(self): return self.inertia
            def set_masses(self, values, ids): self.mass = values
            def set_inertias(self, values, ids): self.inertia = values
        view = View()
        env = SimpleNamespace(num_envs=2, scene={"payload": SimpleNamespace(root_physx_view=view)})
        payload.randomize_transport_mass(env, torch.tensor([1]), (1., 1.))
        torch.testing.assert_close(view.mass, torch.tensor([[.5], [1.]]))
        torch.testing.assert_close(view.inertia[0], torch.ones(9))
        torch.testing.assert_close(view.inertia[1], torch.full((9,), 2.))
        payload.randomize_transport_mass(env, torch.tensor([1]), (0., 0.))
        self.assertAlmostEqual(float(view.mass[1]), .001, places=6)


if __name__ == "__main__":
    unittest.main()
