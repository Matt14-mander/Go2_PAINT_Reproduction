# GO2-PIPER PAINT：MuJoCo sim-to-sim 部署指南

目标是将现有 Isaac Lab Flat 策略加载到 MuJoCo，验证站立、直行和转向。这不是实机部署。
新入口：`mujoco/deploy/deploy_mujoco/go2_piper/paint_loco.py`；配置：同目录 `paint_loco.yaml`。
**不要使用原 `go2_piper.py config.yaml`**：它面向 18 动作、3 帧历史、带末端位姿命令的全身策略，与当前权重不兼容。

## 1. 所需文件与同步

需同步/保留以下文件：

- 最新 `mujoco/deploy/deploy_mujoco/go2_piper/paint_loco.py` 与 `paint_loco.yaml`。
- 整个 `mujoco/robots/go2_piper/`，包含 scene.xml、go2piper.xml 与 assets/ 内模型网格。
- 本次训练 run 下的 `exported/policy.pt`。只运行部署入口时，不需要 Isaac USD、RSL-RL 或训练 checkpoint。
- 可选检查脚本 `scripts/paint/check_mujoco_paint.py`。

本地 Git LFS 模型文件需已经下载；不要同步只有 100 多字节 LFS 指针的 STL。
部署入口通过内存 VFS 读取模型和网格，也可用于含中文的 Windows 目录；GUI 只在非 headless 模式导入。

## 2. 确认策略已导出

此前 `play.py` 已经跑出视频，因此应已执行 TorchScript 导出。在服务器执行：

```bash
cd ~/workspace/Go2_PAINT_Reproduction
RUN_DIR="$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07"
ls -lh "$RUN_DIR/exported/policy.pt"
```

**`model_9999.pt` 是训练 checkpoint，不能直接交给 MuJoCo；需要 `exported/policy.pt`。**
如果不存在，仍在原 `limx` 环境用既有权重导出并短录一次视频：

```bash
PYTHONPATH="$PWD/.runtime/rsl_rl_3_0_1${PYTHONPATH:+:$PYTHONPATH}" \
python scripts/rsl_rl/play.py --task GO2-PIPER-PAINT-Loco-Flat-Play \
  --checkpoint "$RUN_DIR/model_9999.pt" --num_envs 1 --headless --video --video_length 100
```

直接通过 `--policy "$RUN_DIR/exported/policy.pt"` 使用即可；无需复制覆盖仓库自带的其他 `policy.pt`。

## 3. MuJoCo 环境

入口只需要 Python、PyTorch、NumPy、MuJoCo、PyYAML；录像另需 imageio/ffmpeg。无需启动 Isaac Sim，也不导入 TRON2 的 RSL-RL。
可在已能加载导出策略的 `limx` 环境中安装：

```bash
python -m pip install "mujoco==3.2.7" "pyyaml>=6.0" "imageio[ffmpeg]>=2.31"
python -c "import mujoco, torch; print('MuJoCo:', mujoco.__version__, 'Torch:', torch.__version__)"
```

如果需要与现有科研环境完全隔离，可使用单独环境安装上述依赖和 PyTorch；MuJoCo 策略推理固定使用 CPU，不需要 CUDA 版 Torch。
不要运行本项目 extension 的 editable 安装：此入口不依赖 Isaac Lab 包。

## 4. 先验证模型和策略接口，不运行控制

```bash
python mujoco/deploy/deploy_mujoco/go2_piper/paint_loco.py \
  --policy "$RUN_DIR/exported/policy.pt" --headless --check-only
```

应出现：`XML loaded; policy verified: 247 observations -> 12 actions`。
这一步验证真实模型关节/电机匹配、18 个默认关节角、247 维初始观测与策略实际输出；不代表策略能在 MuJoCo 稳定行走。
如果报输入维度不匹配，不要补零、截断或修改权重，先核对导出文件是否来自本次训练。

## 5. 无窗口物理测试：站立，然后直行

```bash
mkdir -p logs/mujoco

python mujoco/deploy/deploy_mujoco/go2_piper/paint_loco.py \
  --policy "$RUN_DIR/exported/policy.pt" --headless --duration 10 --command 0 0 0 \
  --report logs/mujoco/stand.json

python mujoco/deploy/deploy_mujoco/go2_piper/paint_loco.py \
  --policy "$RUN_DIR/exported/policy.pt" --headless --duration 10 --command 0.3 0 0 \
  --report logs/mujoco/forward.json
```

这两条不创建渲染器，因此不需要 DISPLAY/EGL，适合先验证控制。
默认无额外负载、无推扰、无随机外力、固定指令、标称 PD；每仿真秒打印实际速度和基座高度。
报告包含完成原因、实际速度、逐轴速度 RMSE、机械臂关节误差。
`reason=completed` 表示跑满指定仿真时间；`fallen/invalid_physics` 会停止并返回退出码 2，不会悄悄重置继续。
跌倒判定为倾斜超过约 1 rad 或基座高度低于 0.12 m，不等同于 Isaac 中的完整基座接触终止条件。

## 6. 有桌面的可视化

在服务器本地图形桌面终端、远程桌面，或本地有图形环境的机器执行：

```bash
python mujoco/deploy/deploy_mujoco/go2_piper/paint_loco.py \
  --policy "$RUN_DIR/exported/policy.pt" --duration 30 --command 0.3 0 0
```

不要加 `--headless`。被动查看器会跟随机器人，按真实时间推进；关闭窗口即可结束。
普通无 DISPLAY 的 SSH 终端应使用无窗口测试或下一节录像。当前入口使用固定命令，不支持键盘遥控。

## 7. 无桌面服务器录制视频

在 NVIDIA Linux 服务器上，先尝试 EGL；环境变量必须在导入 MuJoCo 前设置：

```bash
MUJOCO_GL=egl python mujoco/deploy/deploy_mujoco/go2_piper/paint_loco.py \
  --policy "$RUN_DIR/exported/policy.pt" --headless --duration 10 --command 0 0 0 \
  --video logs/mujoco/stand.mp4 --report logs/mujoco/stand.json

MUJOCO_GL=egl python mujoco/deploy/deploy_mujoco/go2_piper/paint_loco.py \
  --policy "$RUN_DIR/exported/policy.pt" --headless --duration 10 --command 0.3 0 0 \
  --video logs/mujoco/forward.mp4 --report logs/mujoco/forward.json
```

默认 640×480、30 fps。EGL 是否可用取决于服务器驱动、系统库和会话环境；若渲染器创建失败，先保留无视频的物理测试，不把 EGL 错误当成策略错误。
不能将先前的 Isaac CXXABI 环境设置盲目套用到另一个 Python 环境；若出现动态库错误，按报错单独定位。

## 8. 当前适配契约与限制

| 项目 | 本入口 |
|---|---|
| Actor 输入 | 247 维，单帧，无历史 |
| 动作 | FR、FL、RR、RL，各 hip/thigh/calf，共 12 维 |
| 关节观测 | 上述 12 腿关节，再 joint1..joint6，共 18 个位置与速度 |
| 观测顺序 | lin_vel×2、ang_vel×0.2、gravity、command、q−q0、dq×0.05、previous_action、height×187 |
| 坐标系 | 质心速度从世界系转到基座 link 轴；重力投影也在该坐标系 |
| 高度扫描 | 静态水平无限平面的解析交点，`clip(base_z - floor_z - 0.5, -1, 1)` 重复 187 次 |
| 腿控制 | `q_des=q0+0.25×clip(action,-4,4)`，kp=30，kd=0.6 |
| 臂目标 | `[0,0.8,-1.2,0,0.4,0]` rad，使用训练配置的逐关节标称 PD |
| 频率 | 物理 200 Hz，策略 50 Hz |
| 物理约束 | 保留 XML 电机力矩范围，按名称映射电机；机器人自碰撞关闭 |

这里的 187 个相同高度值是平面几何计算结果，不是任意填零。此入口只支持当前 Flat 场景；台阶、斜坡或障碍物必须实现对应射线扫描，不能沿用该高度公式。
不能直接使用之前设计的 235 维 `PaintLocoInterface` 去加载当前 247 维权重。

Isaac USD 与现有 MuJoCo XML 的惯性、接触和限位不保证完全相同，XML 的电机力矩范围也未逐项与训练 USD 比对。
本入口只复现标称控制设置；不复现训练随机摩擦、增益、质心、推扰。默认执行器延迟为 0，可在 YAML 中设 `delay_steps=1..4` 做后续测试。
应以 run 下 `params/env.yaml` 为训练真值；如果训练时改过默认姿态、kp/kd、动作尺度或 dt，先更新新 YAML 再测试。
基线通过后，可尝试固定 `--payload-mass 1`，它在末端增加等效质量/球体惯量，不是共享物体抓握模型。

## 9. 本次验证范围

本地使用 MuJoCo 3.2.7 加载真实 XML/网格，4 项关节名称映射、初始 247 维观测、固定机械臂目标、坐标变换和有界力矩检查通过；使用合成零动作 TorchScript 做过接口加载、短物理运行及本机离屏 MP4 生成。Linux 服务器的 EGL 仍需实测。
没有获取服务器的真实训练权重，因此不声称真实策略已经完成 sim-to-sim 迁移。实际结果以服务器的 stand/forward 报告及视频为准。

参考：[MuJoCo Python bindings](https://mujoco.readthedocs.io/en/stable/python.html)、[mj_objectVelocity](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-objectvelocity)、[Isaac Lab root state frame semantics](https://github.com/isaac-sim/IsaacLab/blob/v2.3.0/source/isaaclab/isaaclab/assets/articulation/articulation_data.py)。
