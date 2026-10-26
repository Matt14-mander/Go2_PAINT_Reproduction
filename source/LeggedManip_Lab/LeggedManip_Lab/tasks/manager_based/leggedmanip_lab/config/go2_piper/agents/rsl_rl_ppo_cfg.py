# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaaclab.utils import configclass

from LeggedManip_Lab.rsl_rl_compat import (
    RslRlOnPolicyRunnerCfg,
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
)

# Flat
@configclass
class Go2PiperFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    check_for_nan = True
    max_iterations = 10000
    save_interval = 1000

    experiment_name = "go2_piper_flat"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

# WBC
@configclass
class Go2PiperWBCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    check_for_nan = True
    max_iterations = 10000
    save_interval = 1000

    experiment_name = "go2_piper_wbc"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=1.0,
            std_type="log",
        ),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2PiperPaintLocoPPORunnerCfg(Go2PiperFlatPPORunnerCfg):
    experiment_name = "go2_piper_paint_loco"
    save_interval = 200
    clip_actions = 4.0


@configclass
class Go2PiperPaintLocoRoughPPORunnerCfg(Go2PiperPaintLocoPPORunnerCfg):
    experiment_name = "go2_piper_paint_loco_rough"


@configclass
class Go2PiperPaintTransportPPORunnerCfg(Go2PiperFlatPPORunnerCfg):
    """Teacher PPO wiring for environment integration; no student algorithm here."""
    experiment_name = "go2_piper_paint_transport_teacher"
    max_iterations = 15000
    save_interval = 200
    clip_actions = None  # the transport term applies per-component limits

    def __post_init__(self):
        # Preserve legacy/native policy construction from the compatibility base.
        super().__post_init__()
        self.actor.obs_normalization = False
        self.critic.obs_normalization = False


@configclass
class Go2PiperPaintTransportStage1PPORunnerCfg(Go2PiperPaintTransportPPORunnerCfg):
    experiment_name = "go2_piper_paint_teacher_stage1"
    max_iterations = 5000
    save_interval = 100
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128], activation="elu", obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="log"))

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1e-4
        self.algorithm.entropy_coef = 1e-3


@configclass
class Go2PiperPaintTransportStage2PPORunnerCfg(Go2PiperPaintTransportStage1PPORunnerCfg):
    experiment_name = "go2_piper_paint_teacher_stage2"
    max_iterations = 5000

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 5e-5
        self.algorithm.entropy_coef = 5e-4

