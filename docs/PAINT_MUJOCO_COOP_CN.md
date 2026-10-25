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

### PAINT 方向的桌面演示：向载荷施力，机器人跟随（推荐）

目标依据 [PAINT 主页 Simulation Results](https://paint-bot.github.io/) 与[论文的物体交互和多机部分](https://arxiv.org/html/2604.12852v2)：伙伴通过载荷交互表达运动意图，机器人通过机械耦合响应。前一版 `--drive velocity` 是独立的速度控制接口检查，不能作为这个受力跟随目标的替代。

新增 `--controller admittance --drive wrench --wrench-source grasp` 链路：

```text
键盘修改世界坐标 Fx/Fy/Tz
  → 真实外力/力矩只施加到共享箱体
  → 固定连接把物体运动/受力传到各机械臂
  → 每机读取自己的抓取约束反力、扣除静态偏置并低通滤波
  → 每机独立生成底盘指令，冻结LL驱动腿部跟随
```

这个 FT 参考导纳控制器不接收统一预设速度，不读取其他机器人的位置，也不叠加原 reference 的全局队形修正。它是受力跟随 pipeline 的可替换基线，**不是 PAINT 训练出的无FT Student，也不能当作原论文性能复现**。目前 Teacher 输出饱和问题仍需解决。

同步最新 `paint_cooperate.py`、`cooperative_control.py`、新增 **`cooperative_visuals.py`**，并保留之前的 `cooperative_scene.py`、`paint_loco.py/yaml`。服务器桌面执行：

```bash
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction
DEMO=mujoco/deploy/deploy_mujoco/go2_piper/paint_cooperate.py
LL="$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt"

python "$DEMO" --ll-policy "$LL" \
  --controller admittance --drive wrench --wrench-source grasp \
  --force-world 16 0 0 --max-speed 0.35 \
  --settle-seconds 3 --ramp-seconds 3 \
  --keyboard --duration 120 --camera-azimuth 25 \
  --report logs/mujoco_coop/paint_force_follow.json
```

直接打开显示窗口，无 headless/录像。前3秒静止，最后1秒采样每机的静态抓取反力作为偏置，然后3秒逐渐向箱体施加总Fx=16N。默认导纳标度是40N对应1m/s、10Nm对应1rad/s，实际每机受力由共享物体动力学决定，不保证恰好均分。理想均分时每机约8N，对应约0.2m/s；需要更快时按W逐步增加到总24N，理想参考约0.3m/s。最大速度仍受保护范围限制，不能把理论参考当作实际走速。

在桌面窗口获得焦点后：

| 按键 | wrench 模式下的操作 |
|---|---|
| W / S | 箱体世界Fx增加 / 减少2N |
| A / D | 箱体世界Fy增加 / 减少2N |
| Q / E | 箱体世界Tz增加 / 减少0.5Nm |
| X | 立即撤除请求外力/力矩，机器人减速响应由物理与控制器产生 |
| R | 恢复命令行最初的施力请求 |
| P | 暂停 / 继续物理仿真 |

红色箭头显示实际施加的世界外力，蓝色弧形箭头显示偏航力矩；文字显示外力、力矩和实际水平位移。箭头抬高显示以避免被箱体遮挡，不参与物理计算。镜头跟随箱体但不锁死旋转，可用鼠标改变观察角度。地面每米标线帮助判断路程。终端和CSV分别记录实际外力、每机感知的抓取反力、生成指令和实际速度；在 wrench 模式下键盘只改力，不改速度。

参考导纳目前使用0.25秒低通滤波、0.15m/s²平动加速度上限和静态载荷偏置扣除，机械臂保持默认姿态。若需要改变响应强度，可显式设 `--admittance-gain`；默认1，不应靠无限增益掩盖低层失稳。偏置扣除是参考控制器的工程处理，与训练时的外力观测语义不同，**不会应用到 Teacher 输入**。

为隔离“反力信号适配”和“多机低层能力”，可把上述 `--wrench-source grasp` 改为 `external-share` 对照。此时参考导纳使用真实已知外力均分后的特权信号，不是每机独立感知。原有 `--controller reference --drive wrench` 仍是已知外力加全局队形辅助的基线。

后续换合格 Teacher：改成 `--controller teacher --hl-policy /path/to/exported/policy.pt`，保持 `--drive wrench`。Teacher 的147维观测接口不变。用 `external-share` 先检查真值受力输入，使用 `grasp` 时仍须考虑力矩参考点和动力学分布差异。最后再接入 Student 的本体历史意图估计器，才能验证论文中的无FT感知版本。

本阶段仍是同向并排GO2-PIPER和箱体代理模型；网页中的原始沙发/桌子/桶网格、相向抓取布局、粗糙地形及异构伙伴不在当前验证范围。本地已验证施力、反力方向、撤力后的参考指令衰减和箭头渲染，真实权重下的长距离/转向跟随需要在服务器实测。

### 独立速度控制接口检查：主动搬运8米（非 PAINT 受力跟随展示）

旧入口默认是**受力跟随**：箱体上施加8N总外力，参考控制器把它均分，换算出两台机器人各约0.1m/s的速度指令。因此物体既受到外力作用，也受到机器人腿部主动行走的作用，不能仅凭移动判断是在主动运输。

新增 **`--drive velocity` 主动搬运模式**：默认外力为0，世界坐标速度直接进入手写高层协调器，由真实 LL/PD 驱动机器人带着箱体运动。该模式用于协同搬运 pipeline 展示，不是训练好的 Teacher 策略。原有 `--drive wrench` 和 Teacher 接口保留。

同步更新 `paint_cooperate.py`、`cooperative_scene.py`，以及新增的 **`cooperative_control.py`**。在服务器图形桌面的终端执行：

```bash
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction
DEMO=mujoco/deploy/deploy_mujoco/go2_piper/paint_cooperate.py
LL="$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt"

python "$DEMO" --ll-policy "$LL" \
  --controller reference --drive velocity \
  --command-world 0.3 0 0 --max-speed 0.35 \
  --duration 60 --travel-distance 8 \
  --report logs/mujoco_coop/active_8m.json
```

直接打开窗口，不加 headless/video 参数。开始2秒站稳，3秒加速；接近世界坐标直线目标时自动减速、修正横向偏离并保持，运行到60秒或关闭窗口。0.3m/s是期望速度，8米是目标，不能保证实际策略一定完成；JSON 的 `goal_error_m`、`goal_reached`（位置误差<0.15m）报告实际情况。到达后没有直接退出，便于观察负载下停车稳定性；末尾3秒也会逐渐撤除速度指令。

速度、转向均有限幅；全组平动加速度上限0.15m/s²，偏航加速度上限0.4rad/s²。主动模式通过航向反馈抑制长时间直行时的偏航漂移。地面每1米有非碰撞标线，每5米加色，跟随相机时也能观察位移；终端显示 `travel_xy`（距初始箱体位置的水平直线位移）与 `actual_body`（各机平均机体系速度），不把初始落地高度变化算作运输距离。

图形同步由每物理步200Hz降低为最高约60Hz（默认200Hz物理步下实际约50Hz），物理和策略频率保持不变，减少桌面渲染引起的慢放。窗口按真实时间推进；鼠标仍可旋转/缩放相机。

### 桌面键盘控制

希望自由走得更远、调速或转向时，不指定距离目标，改用：

```bash
python "$DEMO" --ll-policy "$LL" \
  --controller reference --drive velocity \
  --command-world 0.3 0 0 --max-speed 0.35 \
  --keyboard --duration 300 \
  --report logs/mujoco_coop/keyboard.json
```

点击窗口使其获得键盘焦点。按键修改**世界坐标**速度指令，每次按键生效一次：

| 按键 | 操作 |
|---|---|
| W / S | 世界X速度增加 / 减少0.05m/s |
| A / D | 世界Y速度增加 / 减少0.05m/s |
| Q / E | 偏航速度增加 / 减少0.1rad/s |
| X | 请求平滑停车，仍受减速度上限约束 |
| R | 恢复命令行初始速度 |
| P | 暂停 / 继续物理仿真，暂停期间不消耗仿真时长 |

方向键式速度控制只作用于 reference/velocity 模式，不覆盖 Teacher 输出。键盘控制和固定距离目标不能同时开启。新增高速度、转弯以及长时间真实权重运行仍需逐步验证；若发生跌倒或抓取误差，仍会停止并保存报告。

### 原受力跟随模式

无需 Isaac Sim、RSL-RL，也不用停止当前训练。依赖与原 `paint_loco.py` 相同：MuJoCo3.2.7、CPU可用的PyTorch、NumPy、PyYAML；录像需要 imageio/ffmpeg。

需同步：`paint_cooperate.py`、`cooperative_scene.py`、`cooperative_control.py`、`cooperative_visuals.py`、原有 `paint_loco.py/paint_loco.yaml`，以及整个 `mujoco/robots/go2_piper/` 的真实网格。没有新增需要下载的机器人资产。

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

开头2秒只站立，接下来3秒逐渐增加外力，末尾3秒撤力；`--force-world` 的三个值是世界坐标 `Fx N、Fy N、Tz Nm`，只施加在箱体上。默认 `--max-speed .35 --max-yaw-rate .4`，底盘命令小于0.05时置零。全组控制是参考导纳，执行腿运动仍依赖真实 LL 策略，不能保证仅凭场景搭建就稳定行走。

无桌面服务器录像（先通过无渲染测试）：

```bash
MUJOCO_GL=egl python "$DEMO" --ll-policy "$LL" \
  --headless --duration 20 --force-world 8 0 0 \
  --video logs/mujoco_coop/reference_forward.mp4 \
  --report logs/mujoco_coop/reference_forward_video.json
```

有桌面的服务器或本机，用相同命令去掉 `--headless` 可打开交互窗口，默认跟随箱体。可用鼠标旋转视角；录像视角可用 `--camera-azimuth` 调整。

## 替换 Teacher / 扩展机器人数量

### 本次10000轮完成后，切换 model_9999

`exported/policy.pt` 可能是之前测试8600轮时生成的文件，不能因为训练目录新增了 `model_9999.pt` 就认为它自动更新。当前 RSL-RL 3.x、无归一化的147→9 Teacher 可直接使用独立 CPU 导出器；同步 `scripts/paint/export_teacher_cpu.py` 后执行：

```bash
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction
RUN="$PWD/logs/rsl_rl/go2_piper_paint_transport_teacher/2026-10-25_15-29-21_teacher_flat_light"
LL="$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt"

python scripts/paint/export_teacher_cpu.py \
  --checkpoint "$RUN/model_9999.pt" \
  --output "$RUN/exported/model_9999" --activation elu
```

此入口只依赖 PyTorch 和 PyYAML，不启动 Isaac、不导入 RSL-RL，也不需要 `.runtime` 的 PYTHONPATH。它读取 `params/agent.yaml` 核对激活函数，并从 `model_state_dict` 提取原始 actor MLP，验证147维输入、9维输出和序列化前后推理等价。`--activation elu` 对应本次训练配置；配置冲突会拒绝导出。`manifest.json` 记录来源 checkpoint、checkpoint/actor SHA256、迭代编号、激活函数和归一化状态。

只支持当前标准 `actor.N.weight/bias`、无观测归一化的非循环 RSL-RL 3.x Teacher。发现归一化、循环层、其他网络布局或错误维度时会报错，不能通过补零、截断或忽略状态来绕过。需要原生导出时，最新版 `play.py` 仍提供 `--export-only --export-dir`；若提示这两个参数不识别，是服务器脚本未同步，旧 parser 将未知参数传给 Hydra 导致，与权重是否有效无关。

然后在服务器桌面直接用新 Teacher：

```bash
python mujoco/deploy/deploy_mujoco/go2_piper/paint_cooperate.py \
  --ll-policy "$LL" --controller teacher \
  --hl-policy "$RUN/exported/model_9999/policy.pt" \
  --drive wrench --wrench-source external-share \
  --force-world 16 0 0 --max-speed 0.25 \
  --settle-seconds 3 --ramp-seconds 3 \
  --keyboard --duration 60 --camera-azimuth 25 \
  --report logs/mujoco_coop/teacher9999.json
```

先使用真实已知外力均分的 Teacher 特权输入，理想参考每机约0.2m/s；把抓取反力适配问题留给下一步独立比较。键盘 W/S、A/D、Q/E 改变箱体上的力/力矩，X撤力。该命令没有 reference/admittance 回退；速度和机械臂目标均由 Teacher 输出。JSON 中各 `hl_policies.path` 应包含 `exported/model_9999/`。

10000轮完成只说明训练预算结束，不代表8600轮发现的饱和现象已解决。若原始指令仍长期顶到限幅、`command_safety_clipped_fraction` 接近1，应保留JSON和CSV继续诊断，不能仅凭视频运动认定 Teacher 已学会跟随。

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
