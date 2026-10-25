# MuJoCo 多 GO2-PIPER 共享载荷 Demo

目标：先跑通多机物理连接、独立控制、策略加载、录像和诊断；之后更换策略文件即可复用。首版使用两台**同向并排**的 GO2-PIPER，机械臂连接同一个自由刚体箱子，支持 `--num-robots 2/3/4`。不是两台相向站立的机器人。

场景通过现有 MJCF 和网格在内存中组合，机器人名称分别为 `r0/…`、`r1/…`，每台有独立根部自由关节、18个关节电机、低层上一动作、机械臂8帧历史。箱子只有一个自由关节，各机器人 EE 与箱体分别通过 [MuJoCo weld equality](https://mujoco.readthedocs.io/en/3.2.7/XMLreference.html#equality-weld) 连接。初始 FK 对齐抓取锚点后才启用约束，没有与世界固定的连接、mocap驱动、运行时瞬移或载荷重力补偿。

## 控制链和可替换策略

```text
共享箱体 + 多机器人 MuJoCo 物理状态
      ↓
高层：reference 手写导纳/队形控制，或每台独立的 Teacher TorchScript
      ↓ 9维：底盘速度3 + 机械臂动作6
每台独立的冻结 LL TorchScript：247 → 12
      ↓
腿 q_default + 0.25*a_LL；臂 q_default + 0.5*a_arm
      ↓
200Hz PD力矩，统一推进同一个物理世界；高层和低层50Hz
```

| 参数 | 用途 |
|---|---|
| `--ll-policy path` | 所有机器人使用同一份已导出的247→12低层策略，实例/状态相互独立 |
| `--ll-policy path0 path1` | 各机器人分别加载自己的低层策略，路径数需等于机器人数量 |
| `--controller reference` | 默认手写参考控制器，只验证 pipeline，不能作为 Teacher 学习成果 |
| `--controller teacher --hl-policy path` | 替换高层为147→9 Teacher；也可提供每台各自的路径 |
| `--wrench-source external-share` | 默认，真实施加到箱体的世界 wrench 均分给机器人，再旋转到各自 base 坐标轴并用40N/10Nm归一化 |
| `--wrench-source grasp` | 实验选项，使用各固定连接的真实约束反力作为高层 wrench 信号 |

只接受导出的 TorchScript `policy.pt`，不接受训练 checkpoint `model_*.pt`。启动时实际探测输入/输出维度并记录 SHA256；更换文件后重启 Demo 生效，当前不支持运行中热加载。新的策略必须保持现有观测语义、默认角度、频率和动作尺度，不能仅保证维度相同。

Teacher 观测为机械臂8帧干净历史（相对角度6、速度6、上一控制步目标角度6）+ 当前归一化 wrench 3，重置时历史为零。LL 布局复用已经验证的 MuJoCo 入口，保留18个关节状态、独立12维上一低层动作和187维水平无限平面扫描。

参考模式将已知外力均分并映射为全组期望速度，额外用真实全局位置和箱体朝向保持队形，机械臂维持默认姿态。Teacher 模式没有叠加这个队形修正；如果它输出不合理，诊断会如实记录，不会自动切回参考控制器。

### 受力来源的边界

`external-share` 是可控的仿真特权信号，便于接线与策略对照。共享箱体不等同于原训练的每机独立箱体，均分是一项明确的建模选择。

`grasp` 从每个 weld 对自由箱体的广义约束力中，利用箱体在 EE 原点的空间 Jacobian 反解世界坐标六维 wrench，取反得到 payload→EE 的反作用力，再投影到 base 轴。此信号含重力、惯性和机器人间相互作用，力矩参考点为 EE。它**不等同于训练时施加在独立箱体上的伙伴外力/力矩**，不能假定现有 Teacher 无需适配就能使用。反解接口参考 [MuJoCo Jacobian API](https://mujoco.readthedocs.io/en/3.2.7/APIreference/APIfunctions.html#mj-jac)。

## 服务器运行

无需 Isaac Sim、RSL-RL，也不用停止当前训练。依赖与原 `paint_loco.py` 相同：MuJoCo3.2.7、CPU可用的PyTorch、NumPy、PyYAML；录像需要 imageio/ffmpeg。

需同步：`paint_cooperate.py`、`cooperative_scene.py`、原有 `paint_loco.py/paint_loco.yaml`，以及整个 `mujoco/robots/go2_piper/` 的真实网格。没有新增需要下载的机器人资产。

```bash
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction
DEMO=mujoco/deploy/deploy_mujoco/go2_piper/paint_cooperate.py
LL="$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt"

# 编译真实场景，加载和验证策略/关节/锚点，不进行时间推进
python "$DEMO" --ll-policy "$LL" --headless --check-only

# 两机器人静态承载，先确认真实低层权重在共享连接下稳定
python "$DEMO" --ll-policy "$LL" --headless --duration 10 \
  --force-world 0 0 0 --report logs/mujoco_coop/stand.json

# 两机器人参考控制搬运：箱体总Fx=8N，对每机归一化参考约0.1m/s
python "$DEMO" --ll-policy "$LL" --headless --duration 20 \
  --force-world 8 0 0 --report logs/mujoco_coop/reference_forward.json
```

开头2秒只站立，接下来2秒逐渐增加外力，末尾2秒撤力；`--force-world` 的三个值是世界坐标 `Fx N、Fy N、Tz Nm`，只施加在箱体上。默认 `--max-speed .25 --max-yaw-rate .4`，底盘命令小于0.05时置零。全组控制是参考导纳，执行腿运动仍依赖真实 LL 策略，不能保证仅凭场景搭建就稳定行走。

无桌面服务器录像（先通过无渲染测试）：

```bash
MUJOCO_GL=egl python "$DEMO" --ll-policy "$LL" \
  --headless --duration 20 --force-world 8 0 0 \
  --video logs/mujoco_coop/reference_forward.mp4 \
  --report logs/mujoco_coop/reference_forward_video.json
```

有桌面的服务器或本机，用相同命令去掉 `--headless` 可打开交互窗口，默认跟随箱体。可用鼠标旋转视角；录像视角可用 `--camera-azimuth` 调整。

## 替换 Teacher / 扩展机器人数量

当前 `model_8600.pt` 已发现持续输出饱和，暂时使用 reference 模式作为 pipeline 基线。新 Teacher 评估合格并由 Isaac `play.py` 导出后，只需切换入口参数：

```bash
HL="$PWD/logs/rsl_rl/go2_piper_paint_transport_teacher/2026-10-25_15-29-21_teacher_flat_light/exported/policy.pt"

python "$DEMO" --ll-policy "$LL" \
  --controller teacher --hl-policy "$HL" \
  --wrench-source external-share \
  --headless --duration 20 --force-world 8 0 0 \
  --report logs/mujoco_coop/teacher.json
```

代码同时支持两台不同策略：`--hl-policy "$HL0" "$HL1"`。接口可用于当前不合格策略的诊断，但不能把程序跑完当作 Teacher 已成功协作。

三台机器人版本可直接添加 `--num-robots 3`。箱体会随横向队形变宽，质量仍是指定的总质量；若希望每机仍对应0.1m/s的外力参考，将总 `Fx` 改为12N。四台同理16N。2/3/4台是同一结构，但尚未完成真实策略下所有规模、方向和转弯工况的验证。

## 输出与物理设置

- JSON：完成/失败原因、箱体三轴位移、每台最大倾角/抓取误差、最终状态、命令保护限幅比例，以及各策略路径和hash。
- 同名 CSV：每控制步每机器人一行，包含归一化 wrench、高层原始速度输出、限幅后的速度指令、实际速度、倾角与抓取误差。
- 跌倒（高度<0.12m或倾角>约57°）、抓取误差>8cm、非有限状态或求解器警告时停止并返回退出码2；不自动重置掩盖失败。倾角/误差峰值按50Hz采样，不包含每个物理子步的峰值。

`command_safety_clipped_fraction` 接近1时，说明高层大量输出超出 Demo 的速度保护范围；即使视频看起来平稳，也不能把结果当作原始策略合格。初始落地会使箱体高度下降，位移应结合视频和CSV判断，不能将竖直落地位移计为搬运距离。

PD：腿30/0.6，臂50/4，使用真实电机力矩限制，无执行器延迟。初始机械臂参考沿用冻结 LL，箱体位置由FK计算。默认仅关闭每台机器人自己的自碰撞，保留机器人之间、箱体和地面碰撞；EE和两个夹爪指节与载荷的碰撞被排除以表示已闭合的焊接抓取。可用 `--self-collisions` 开启自碰撞，但应单独验证。和原 Isaac Transport 自碰撞/延迟设置的差异需要在后续策略验证中考虑。

## 本地验证情况

```bash
python scripts/paint/check_mujoco_cooperate.py
python scripts/paint/check_mujoco_paint.py
```

已在真实 MuJoCo3.2.7 中检查2/3台命名隔离、自由根部/共享自由载荷、零初始约束误差、独立动作历史、力信号归一化、负载重量分担、解除抓取后的自由下落，以及 TorchScript 输入输出拒错。双机静态承载完整入口跑了6秒，三机1秒，录像路径也已实际验证。

本地使用零动作 TorchScript fixture 验证调用链与静态PD承载；该 fixture 不会走路，不是用户训练的 locomotion/Teacher 权重。**真实策略的双机行走与策略替换后的稳定性需要在服务器执行上述命令验证。** 原训练进程不受影响。
