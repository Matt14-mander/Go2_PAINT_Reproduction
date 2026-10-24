<p align="center">
  <img src="docs/imgs/readme_logo.png" alt="LeggedManip Lab" width="600">
</p>

# LeggedManip Lab

GO2-PIPER 的 PAINT 底层 locomotion 改造：[任务、训练与冻结策略接口](docs/PAINT_LOCOMOTION_CN.md)。
 
[![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-5.1.0-blue.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-main%20branch-blue.svg)](https://github.com/isaac-sim/IsaacLab/tree/main)
[![RSL-RL](https://img.shields.io/badge/rsl--rl--lib-%3E%3D5.0.1-blue.svg)](https://github.com/leggedrobotics/rsl_rl)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.2.7-orange.svg?logo=mujoco)](https://mujoco.org/)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

[English](README.md) | [中文](README_CN.md)


**面向腿足操作机器人的强化学习训练框架**

腿足机器人与机械臂的协同控制面临独特挑战——需要同时协调运动与灵巧操作。LeggedManip Lab 基于 Isaac Lab 构建了统一的 RL 训练框架，支持 7 种机器人平台的全身运动-操作策略训练与部署，包含平坦地形、全身控制（WBC）等训练模式。

<div align="center">

| <div align="center"> Isaac Lab 仿真 </div> | <div align="center"> MuJoCo 部署 </div> | <div align="center"> 实机部署 </div> |
|---|---|---|
| <img src="docs/videos/isaaclab.gif" width="240px"> | <img src="docs/videos/mujoco.gif" width="240px"> | <img src="docs/videos/real.gif" width="240px"> |

</div>

---

## 支持的机器人平台

| 平台 | 腿足机器人 | 机械臂 | 预览 |
|------|:---------:|:-----:|:----:|
| AGO-Z1 | Unitree Aliengo | Unitree Z1 | <img src="./docs/imgs/ago_z1.png" alt="ago_z1" width="75"> |
| B1-Z1 | Unitree B1 | Unitree Z1 | <img src="./docs/imgs/b1_z1.png" alt="b1_z1" width="75"> |
| B2-Z1 | Unitree B2 | Unitree Z1 | <img src="./docs/imgs/b2_z1.png" alt="b2_z1" width="75"> |
| GO1-ARX5 | Unitree Go1 | ARX-X5 | <img src="./docs/imgs/go1_arx5.png" alt="go1_arx5" width="75"> |
| GO1-WX250S | Unitree Go1 | WX250S | <img src="./docs/imgs/go1_wx250s.png" alt="go1_wx250s" width="75"> |
| GO2-ARX5 | Unitree Go2 | ARX-X5 | <img src="./docs/imgs/go2_arx5.png" alt="go2_arx5" width="75"> |
| GO2-PIPER | Unitree Go2 | Agilex Piper | <img src="./docs/imgs/go2_piper.png" alt="go2_piper" width="75"> |
| ... | ... | ... | ... |
每个平台支持以下 **2** 种训练模式：

- **Flat** — 平坦地形上的移动操作
- **WBC** — 全身控制，在[**混合坐标系**](docs/WBC_MIXED_FRAME_CN.md)下进行末端执行器位姿跟踪

---

## 开发计划

| 功能 | 状态 |
|------|------|
| 平坦地形训练 | ✅ 已发布 |
| 全身控制（WBC） | ✅ 已发布 |
| Sim-to-Sim 迁移 | ✅ 已发布 |
| Sim-to-Real 迁移 | 🔜 即将发布 |

---

## 安装

### 先决条件

* 请根据 [Isaac Lab 官方安装指南](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) 安装 Isaac Lab。
  推荐使用 conda 安装方式，因为它可以更方便地在终端中调用 Python 脚本。

> [!IMPORTANT]
> Agent 配置已加入项目内兼容层，请使用配套接口：旧版 Isaac Lab policy/TensorDict + RSL-RL `3.x >=3.0.1`，或新版 actor/critic + RSL-RL `>=5.0.1`。
>
> 依赖要求：
>
> * Isaac Sim: `5.1.0`
> * Isaac Lab / rsl-rl-lib: 使用上述配套 API
> * Python: `3.11`

**[推荐一键安装脚本](https://docs.robotsfan.com/isaaclab/source/setup/oneclick_installation.html)**：

```bash
wget -O install_isaaclab.sh https://docs.robotsfan.com/install_isaaclab.sh && bash install_isaaclab.sh
```

安装完成后，请检查当前环境中的 RSL-RL 版本：

```bash
python -m pip show rsl-rl-lib
```

使用 `python scripts/paint/server_train.py diagnose` 检查实际接口，不要单独升级 RSL-RL。详见[服务器训练说明](docs/PAINT_SERVER_TRAINING_CN.md)。


### 安装 LeggedManip Lab

- 将本仓库克隆到 Isaac Lab 目录**之外**：

```bash
cd LeggedManip_Lab
python -m pip install -e source/LeggedManip_Lab
```

### 安装依赖

- 安装 MuJoCo 部署及工具所需的额外 Python 依赖：

```bash
pip install -r requirements.txt
```

---

## 快速开始

### 查看可用环境

```bash
python scripts/list_envs.py
```

### 训练

使用 RSL-RL PPO 训练策略：

```bash
# 平坦地形训练
python scripts/rsl_rl/train.py --task GO2-PIPER-Flat --num_envs 4096 --headless --max_iterations 5000

# 全身控制训练
python scripts/rsl_rl/train.py --task GO2-PIPER-WBC --num_envs 4096 --headless --max_iterations 5000
```

### 推理 / 策略评估

```bash
python scripts/rsl_rl/play.py --task GO2-PIPER-Flat --headless
```


### 调试

```bash
python scripts/zero_agent.py --task GO2-PIPER-Flat    # 零动作智能体
python scripts/random_agent.py --task GO2-PIPER-Flat  # 随机动作智能体
```

---

## MuJoCo 部署

### 运行已训练策略

**第一步 — 导出策略。** 使用 `play.py` 加载训练好的 checkpoint，脚本会自动将策略导出为 `policy.pt`（TorchScript）和 `policy.onnx`：

```bash
python scripts/rsl_rl/play.py --task GO2-PIPER-Flat
```

导出的文件保存在 `logs/rsl_rl/{实验名称}/{运行名称}/exported/` 目录下。


**第二步 — 复制策略** 到对应机器人的 policy 目录：

```bash
cp logs/rsl_rl/{实验名称}/{运行名称}/exported/policy.pt mujoco/deploy/policy/{robot}/
```

**第三步 — 在 MuJoCo 中运行** 对应平台的部署脚本：

```bash
python mujoco/deploy/deploy_mujoco/{robot}/{robot}.py {config_name}.yaml
```

**完整示例：运行 GO2-PIPER**

```bash
# 1. 导出
python scripts/rsl_rl/play.py --task GO2-PIPER-Flat

# 2. 复制
cp logs/rsl_rl/go2_piper_flat/2026-01-01_00-00-00/exported/policy.pt mujoco/deploy/policy/go2_piper/

# 3. 部署
python mujoco/deploy/deploy_mujoco/go2_piper/go2_piper.py config.yaml
```

### 键盘遥操作

| 按键 | 机器人移动 | 按键 | 末端执行器平移 | 按键 | 末端执行器旋转 |
|------|-----------|------|--------------|------|--------------|
| W | 向前移动 | I | 向前 | 1 | 绕 Roll 轴顺时针 |
| S | 向后移动 | K | 向后 | 2 | 绕 Roll 轴逆时针 |
| A | 向左移动 | J | 向左 | 3 | 绕 Pitch 轴顺时针 |
| D | 向右移动 | L | 向右 | 4 | 绕 Pitch 轴逆时针 |
| Q | 向左转向 | U | 向上 | 5 | 绕 Yaw 轴顺时针 |
| E | 向右转向 | O | 向下 | 6 | 绕 Yaw 轴逆时针 |
| R | 重置所有指令 | | | | |


---

## 文档

- [环境参数、项目结构与关键文件说明](docs/ENV_DETAILS_CN.md)


---

## 常见问题

### Pylance 扩展索引缺失

在部分 VSCode 版本中，扩展路径可能无法被自动索引。请在 `.vscode/settings.json` 中手动添加路径：

> **注意：将 `<path-to-isaac-lab>` 替换为你的实际 Isaac Lab 安装路径。**

```json
{
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/source/LeggedManip_Lab",
        "/<path-to-isaac-lab>/source/isaaclab",
        "/<path-to-isaac-lab>/source/isaaclab_assets",
        "/<path-to-isaac-lab>/source/isaaclab_mimic",
        "/<path-to-isaac-lab>/source/isaaclab_rl",
        "/<path-to-isaac-lab>/source/isaaclab_tasks"
    ]
}
```

---

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@software{junjiezhu2026LeggedManip_Lab,
  author = {Junjie Zhu},
  title  = {LeggedManip_Lab: A Reinforcement Learning Framework for Legged Manipulation},
  url    = {https://github.com/zzzJie-Robot/LeggedManip_Lab},
  year   = {2026}
}
```

---

## 致谢

本项目参考了以下开源代码：

- [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab)
- [unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
- [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)
