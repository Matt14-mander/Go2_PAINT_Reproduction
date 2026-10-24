"""Project-local config adapter for RSL-RL 3.x and the newer 5.x API.

Import after AppLauncher. Never patches the installed Isaac Lab or RSL-RL packages.
The legacy path translates this project's MLP configs into native ActorCriticCfg.
"""

from packaging.version import Version

from isaaclab.utils import configclass
import isaaclab_rl.rsl_rl as native

RslRlPpoAlgorithmCfg = native.RslRlPpoAlgorithmCfg
RslRlBaseRunnerCfg = getattr(native, "RslRlBaseRunnerCfg", native.RslRlOnPolicyRunnerCfg)
MODERN_CONFIG_API = hasattr(native, "RslRlMLPModelCfg")


def _sync_legacy_policy(cfg):
    if cfg.actor is None or cfg.critic is None:
        return
    if cfg.actor.activation != cfg.critic.activation:
        raise ValueError("RSL-RL 3.x requires the same actor and critic activation")
    distribution = cfg.actor.distribution_cfg
    cfg.policy = native.RslRlPpoActorCriticCfg(
        init_noise_std=distribution.init_std,
        noise_std_type=distribution.std_type,
        actor_hidden_dims=list(cfg.actor.hidden_dims),
        critic_hidden_dims=list(cfg.critic.hidden_dims),
        activation=cfg.actor.activation,
        actor_obs_normalization=cfg.actor.obs_normalization,
        critic_obs_normalization=cfg.critic.obs_normalization,
    )


if MODERN_CONFIG_API:
    RslRlMLPModelCfg = native.RslRlMLPModelCfg

    @configclass
    class RslRlOnPolicyRunnerCfg(native.RslRlOnPolicyRunnerCfg):
        obs_groups: dict[str, list[str]] = {"actor": ["policy"], "critic": ["critic"]}
else:
    @configclass
    class RslRlMLPModelCfg:
        """Only the feed-forward Gaussian MLP features used by this repository."""

        @configclass
        class GaussianDistributionCfg:
            init_std: float = 1.0
            std_type: str = "log"

        hidden_dims: list[int] = [512, 256, 128]
        activation: str = "elu"
        obs_normalization: bool = False
        distribution_cfg: GaussianDistributionCfg = GaussianDistributionCfg()

    @configclass
    class RslRlOnPolicyRunnerCfg(native.RslRlOnPolicyRunnerCfg):
        actor: RslRlMLPModelCfg | None = None
        critic: RslRlMLPModelCfg | None = None
        obs_groups: dict[str, list[str]] = {"policy": ["policy"], "critic": ["critic"]}

        def __post_init__(self):
            # Populate policy before Hydra serializes/validates the config.
            _sync_legacy_policy(self)


def handle_deprecated_rsl_rl_cfg(cfg, installed_version):
    """Refresh legacy policy after Hydra overrides, or delegate to the new helper."""
    parsed = Version(installed_version)
    if not MODERN_CONFIG_API:
        if not Version("3.0.1") <= parsed < Version("4.0.0"):
            raise RuntimeError("The installed legacy Isaac Lab adapter requires RSL-RL 3.x >=3.0.1")
        _sync_legacy_policy(cfg)
        return cfg
    if parsed < Version("5.0.1"):
        raise RuntimeError("The installed modern Isaac Lab adapter requires RSL-RL >=5.0.1")
    converter = getattr(native, "handle_deprecated_rsl_rl_cfg", None)
    return converter(cfg, installed_version) if converter else cfg
