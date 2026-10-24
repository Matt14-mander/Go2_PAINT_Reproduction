"""Validate every robot agent config against the installed RSL API.

Launch Isaac first for package imports, then validate configs without creating a
physics environment. This catches missing config fields before the physics smoke.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)

try:
    import importlib
    import importlib.metadata
    import inspect

    from LeggedManip_Lab.rsl_rl_compat import MODERN_CONFIG_API, handle_deprecated_rsl_rl_cfg

    version = importlib.metadata.version("rsl-rl-lib")
    count = 0
    for robot in ("ago_z1", "b1_z1", "b2_z1", "go1_arx5", "go1_wx250s", "go2_arx5", "go2_piper"):
        name = f"LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.{robot}.agents.rsl_rl_ppo_cfg"
        module = importlib.import_module(name)
        for class_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != name or not class_name.endswith("PPORunnerCfg"):
                continue
            cfg = handle_deprecated_rsl_rl_cfg(cls(), version)
            cfg.validate()
            values = cfg.to_dict()
            if not MODERN_CONFIG_API:
                assert values["policy"]["class_name"] == "ActorCritic"
                assert values["policy"]["actor_hidden_dims"] == values["actor"]["hidden_dims"]
                assert values["policy"]["critic_hidden_dims"] == values["critic"]["hidden_dims"]
                assert values["obs_groups"] == {"policy": ["policy"], "critic": ["critic"]}
            count += 1
    assert count >= 16, count
    print(f"PASS: {count} runner configs; RSL-RL {version}; modern API={MODERN_CONFIG_API}")
finally:
    launcher.app.close()
