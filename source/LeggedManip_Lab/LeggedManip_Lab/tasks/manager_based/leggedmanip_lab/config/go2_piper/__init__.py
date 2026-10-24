import gymnasium as gym
from . import agents

# PAINT low-level replacement: identical actor/action interfaces on Flat and Rough.
for terrain, env_class, runner_class in (
    ("Flat", "Go2PiperPaintLocoEnvCfg", "Go2PiperPaintLocoPPORunnerCfg"),
    ("Rough", "Go2PiperPaintLocoRoughEnvCfg", "Go2PiperPaintLocoRoughPPORunnerCfg"),
):
    for suffix in ("", "-Play"):
        gym.register(
            id=f"GO2-PIPER-PAINT-Loco-{terrain}{suffix}",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.paint_loco_env_cfg:{env_class}{'_PLAY' if suffix else ''}",
                "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:{runner_class}",
            },
        )

# Flat
gym.register(
    id="GO2-PIPER-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2PiperFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperFlatPPORunnerCfg",
    },
)

gym.register(
    id="GO2-PIPER-Flat-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:Go2PiperFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperFlatPPORunnerCfg",
    },
)

# WBC
gym.register(
    id="GO2-PIPER-WBC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:Go2PiperWBCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperWBCPPORunnerCfg",
    },
)

gym.register(
    id="GO2-PIPER-WBC-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_env_cfg:Go2PiperWBCEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PiperWBCPPORunnerCfg",
    },
)

