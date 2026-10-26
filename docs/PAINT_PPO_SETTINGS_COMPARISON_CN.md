# PAINT 教程与本项目 Teacher PPO 参数对照

依据：[训练教程的 PPO settings](https://paint-bot.github.io/docs/training.html#ppo-settings)，以及当前 `config/go2_piper/agents/rsl_rl_ppo_cfg.py`。

原Teacher继承本仓库 `Go2PiperFlatPPORunnerCfg`，没有把所有参数替换成教程值。因此原实验不是严格的教程超参数复现；网络尺寸、初始学习率和熵系数在固定臂Stage1改造前就已不同，不应把这些差异描述为本次改造才出现。

| 参数 | 教程 | 原Teacher | 新固定臂Stage1 |
|---|---|---|---|
| Actor/Critic隐藏层 | 128-128-128 | 512-256-128 | 512-256-128 |
| 激活函数 | ELU | ELU | ELU |
| 初始action std | 1.0 | 1.0 | 0.1 |
| 每环境每轮步数 | 24 | 24 | 24 |
| 学习epochs / mini-batches | 5 / 4 | 5 / 4 | 5 / 4 |
| 初始学习率 | 1e-3 | 5e-4 | 1e-4 |
| 学习率schedule / target KL | adaptive / 0.01 | adaptive / 0.01 | adaptive / 0.01 |
| gamma / GAE lambda | 0.99 / 0.95 | 0.99 / 0.95 | 0.99 / 0.95 |
| PPO clip | 0.2 | 0.2 | 0.2 |
| value loss coefficient | 1.0 | 1.0 | 1.0 |
| entropy coefficient | 0.01 | 0.005 | 0.001 |
| max gradient norm | 1.0 | 1.0 | 1.0 |
| Teacher训练轮数 | 15000 | 配置默认15000，本轮命令覆盖为10000 | 默认5000 |

以上学习率和std都是初始化值，schedule仍为adaptive，std也可学习，不能视为整个训练过程的固定值。原run的最终事实应以服务器 `params/agent.yaml`、启动参数和TensorBoard为准；本地没有读取该服务器文件。上表原run的10000轮依据实际启动命令与最终model_9999记录。

教程还给出Estimator为128-128-128，但本项目目前尚未实现完整Student意图估计器/蒸馏算法，不能声称该部分已对齐。

## 当前固定臂Stage1的定位

教程里的“Stage 1”指完整privileged Teacher；本项目新增的 `Transport-Flat-Stage1` 是Teacher内部的固定臂启动阶段，范围更窄。它是为GO2当前稳定性问题建立的工程基线，不是新的严格论文复现实验。

除了PPO参数，本阶段还改变了执行与任务：固定机械臂、底盘指令范围0.35/0.35/0.4、载荷固定0.5kg、wrench_scale=.2、20%零力回合、较少随机化、倾角保护，以及重新平衡的reward和指令引导。动作惩罚改用原始actor动作是一处独立的实现修正。

并行度也不同：[教程环境](https://paint-bot.github.io/docs/environment.html)为4096，给出的Stage1启动命令是512。两者每轮样本量分别为98304和12288；即使所有PPO标量一致，训练过程和样本预算仍不能直接视为等价。

若目标是严格参数复现，应另建128-128-128、1e-3、entropy=.01等参数的独立profile，并与GO2稳定启动方案分别记录、比较；不应在正在运行的job中途更改配置，也不能仅凭未对齐的参数列表认定它就是前轮失败的唯一原因。
