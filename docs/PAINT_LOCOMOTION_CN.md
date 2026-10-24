# GO2-PIPER：PAINT low-level locomotion

本阶段使用 LeggedManip Lab / Isaac Lab 实现 PAINT 的可替代底层策略。
依据：[PAINT Low-Level Locomotion](https://paint-bot.github.io/docs/locomotion.html)。
原始 ANYmal D perceptive backbone 与训练代码未公开；本实现不是原始权重或逐参数复刻。

## 已实现的范围

- 完整 GO2-PIPER 模型，策略只输出 12 个腿关节动作，机械臂每个物理步通过既有 PD 执行器保持搬运姿态。
- 控制频率 50 Hz，物理频率 200 Hz。`q_des = q_default + 0.25 * clip(action, -4, 4)`，没有 tanh。
- 平地与粗糙地形任务共享观测、动作接口，粗糙地形采用 Isaac Lab 标准地形集合与地形课程。
- 速度课程从现有 ±0.2 起步，扩展到 vx/vy ±1 m/s、wz ±1.5 rad/s。
- 末端刚体附加质量 0–10 kg，并添加半径 0.08 m 实心球的等效转动惯量。质量在 startup 随机化，每个环境固定。
- 末端持续水平力幅值不超过 40 N、世界 Z 轴力矩 ±10 N·m，episode 内每 2–5 s 重采样，20% 无扰动。
- 每个物理子步重新提交世界坐标系外力，保证机械臂转动时力仍水平；各环境 reset 独立清理，之后由 interval 事件引入下一次扰动。
- 保留质量、摩擦、惯量、质心、增益随机化与速度推扰。地形高度奖励使用扫描地面相对高度。
- 独立 PPO 日志目录，每 200 iterations 保存；Play 保留载荷、扰动与摔倒终止条件。

此处负载是**末端集中质量近似**，无新增可视物体、碰撞体或抓握约束，无法模拟物体尺寸、质心偏置与双机闭链耦合。
下一阶段 transport 环境需要实体共享 payload、固定/柔顺抓握及伙伴作用力；应再评估/微调 LL。
0–10 kg 是官方建议的训练范围，GO2-PIPER 在高负载下是否可控尚需仿真评估。
初始搬运姿态 `[0, 0.8, -1.2, 0, 0.4, 0]` rad 在仓库 MuJoCo 关节限位内，尚未通过 Isaac 运动学、碰撞与实际安装验证。

## 安装与检查

使用仓库 README 要求的 **Isaac Sim 5.1.0、Python 3.11、带新版 RSL-RL 配置 API 的 Isaac Lab、rsl-rl-lib >=5.0.1** 环境。
本任务沿用现有项目的 PhysX `root_physx_view` 和 `set_external_force_and_torque` API，不能保证未来任意 Isaac Lab main 提交兼容。
服务器实际使用的 Isaac Lab 提交应记录在实验中；不要为本任务直接升级已有可用环境。
以下命令在仓库根目录运行，Python 应指向已安装 Isaac Lab 的环境。

```bash
git lfs pull
python -m pip install -e source/LeggedManip_Lab
python -m pip show isaacsim rsl-rl-lib
python scripts/list_envs.py
python scripts/paint/check_interface.py
python scripts/paint/smoke_locomotion.py --task GO2-PIPER-PAINT-Loco-Flat --num_envs 16 --steps 100 --headless
python scripts/paint/smoke_locomotion.py --task GO2-PIPER-PAINT-Loco-Rough --num_envs 16 --steps 100 --headless
```

Smoke 检查真实关节匹配、235 维观测、12 维动作、机械臂目标与 reset、外力边界、数值有限性。
零动作策略会摔倒；Smoke 通过不代表策略已经学会行走。
本次代码改造未连接服务器，未执行 Isaac 仿真或训练；请先完成上面的集成检查。
本地已通过 8 项 CPU 单元检查、Python 编译检查与 `git diff --check`。
CPU 检查中的 Isaac 基类使用测试替身，仅验证项目自己的张量逻辑；不能替代真实 Isaac 集成检查。

## 训练

```bash
# 先用少量环境跑通 PPO；显存不足时减小 num_envs
python scripts/rsl_rl/train.py --task GO2-PIPER-PAINT-Loco-Flat --num_envs 64 --max_iterations 2 --headless

# 正式平地训练
python scripts/rsl_rl/train.py --task GO2-PIPER-PAINT-Loco-Flat --num_envs 4096 --max_iterations 5000 --headless

# 粗糙地形可以独立训练
python scripts/rsl_rl/train.py --task GO2-PIPER-PAINT-Loco-Rough --num_envs 4096 --max_iterations 10000 --headless

# 或从平地 checkpoint 继续，显式指定平地 run 的绝对路径
python scripts/rsl_rl/train.py --task GO2-PIPER-PAINT-Loco-Rough --num_envs 4096 --max_iterations 10000 --headless \
  --resume --load_run /absolute/path/to/flat/run --checkpoint model_5000.pt
```

检查实际 checkpoint 文件名再设置 `--checkpoint`。恢复会同时读取 PPO 优化器状态，但当前代码不会恢复 terrain/command curriculum 的进度；新环境重新开始课程。
不要从原有 `GO2-PIPER-Flat` 或 WBC 的 18 维动作 checkpoint 恢复。
现有 `--load_run` 由 Isaac Lab `get_checkpoint_path` 解析；若安装版本不接受绝对目录，先将平地 run 复制到 `logs/rsl_rl/go2_piper_paint_loco_rough/` 再传目录名。

## 评估与导出

```bash
python scripts/rsl_rl/play.py --task GO2-PIPER-PAINT-Loco-Rough-Play --num_envs 32 \
  --checkpoint /absolute/path/to/model_10000.pt --headless --video --video_length 1000
```

`play.py` 导出同一 run 下 `exported/policy.pt` 和 `exported/policy.onnx`，录像模式运行指定步数后退出。
需安装导出/录像依赖（如 onnx、onnxscript、视频编码依赖）；相机渲染要求对应的服务器驱动支持。
新策略不能放入现有 18 动作 MuJoCo 部署脚本直接执行；本阶段交付 Isaac Lab 训练及冻结 LL 接口。
评估应分别统计不同负载、速度与地形下的速度误差、存活时长和摔倒率，再选取冻结权重。

## 与未来高层策略的接口

策略单帧观测共 **235** 维，不堆历史，顺序如下：

| 切片 | 内容 | 处理 |
|---|---|---|
| 0:3 | base linear velocity | body frame ×2 |
| 3:6 | base angular velocity | body frame ×0.2 |
| 6:9 | projected gravity | body frame |
| 9:12 | vx, vy, wz command | m/s, m/s, rad/s |
| 12:24 | leg position − default position | rad |
| 24:36 | leg velocity | rad/s ×0.05 |
| 36:48 | previous LL action | clip ±4，reset 后为 0 |
| 48:235 | height scan | `sensor_z - ray_hit_z - 0.5`，clip ±1 |

扫描采用 `GridPatternCfg(resolution=0.1, size=[1.6,1.0])`，跟随 base yaw；需保持 Isaac Lab 网格射线顺序。
腿顺序：FR、FL、RR、RL，每条腿 hip、thigh、calf。不要按仿真内部自由度顺序直接切前 12 个关节。
Critic 可见额外的真值外力，actor 看不到该真值；没有实现原始内部 backbone 的 wrench predictor。
Actor 需要 base linear velocity（状态估计/里程计）和地形扫描。部署侧无法提供它们时，必须更换观测并重新训练，不能直接填 0。

`mdp/paint_interface.py` 的 `PaintLocoInterface` 提供一致的观测拼接、缩放、动作裁剪和逐环境 reset。
高层输出的速度先由 `process_base_command` 裁剪到训练范围并应用 0.05 的逐轴 deadband，再供给 LL。
调用方应使用处理后的速度作为高层环境中实际 LL 指令；训练随机命令本身不施加 deadband。
每 0.02 s 构造观测并调用导出策略一次，使用返回动作更新 LL 的 previous action。
该状态必须独立于高层 9 维动作，episode reset 时同步清零。导出的 actor 已含自身观测归一化，不要重复归一化。
