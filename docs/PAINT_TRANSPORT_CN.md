# GO2-PIPER Transport 环境

实现依据：[PAINT Transport Environment 教程](https://paint-bot.github.io/docs/environment.html)。这是从教程迁移到本项目 Isaac Lab 的环境实现，机器人替换为 GO2-PIPER。当前包含 teacher PPO 的环境接线；student 意图估计器、蒸馏和训练算法将在后续阶段实现。

## 已实现的接口

| 项目 | 本项目实现 |
|---|---|
| Flat 任务 | `GO2-PIPER-PAINT-Transport-Flat` |
| Rough 任务 | `GO2-PIPER-PAINT-Transport-Rough`，尚未验证粗糙地形能力 |
| 仿真 / 控制 | 0.005 s / 0.02 s，decimation=4；episode 10 s / 500 步 |
| 并行规模 | 配置默认 4096，先从 4 个环境 smoke 开始 |
| 低层 | 加载导出的 TorchScript `policy.pt`，247 维输入、12 维腿动作；`eval()`、禁止梯度，高层优化器仅持有高层网络参数 |
| 高层动作 | 9 维：底盘 `[vx,vy,wz]` + 6 维机械臂动作 |
| 底盘动作 | 不缩放；分别限幅 ±1、±1 m/s、±1.5 rad/s；各分量绝对值小于 0.05 时置零 |
| 机械臂动作 | `q_des=q_default+0.5*a_arm`，最终目标限制在关节软限位；高层独立控制机械臂，不再被旧低层 action term 覆盖 |
| PD | 腿沿用已训练配置；机械臂 kp=50、kd=4；开启机器人自碰撞 |
| 独立载荷 | 0.5×0.3×0.08 m 刚体箱子、两侧把手；近侧把手与 `end_effector` 固定连接，`excludeFromArticulation=True`，保留独立刚体状态 |

载荷尺寸、抓取局部坐标 `(-0.29,0,0.06)` 是本项目的初始几何选择。抓取相对旋转取 USD 参考姿态下 EE 到 base 的变换。每次 reset 先写机器人状态并刷新 FK，再对齐载荷位置、姿态及刚体速度，避免约束两端初始分离。只禁用抓取连接的两刚体相互碰撞，其余碰撞保留。

为保证每个环境的固定连接都指向本环境机器人，当前使用 `replicate_physics=False`，逐环境创建跨资产连接。大规模启动会较慢，4096 环境的性能尚未实测。

## 观测、扰动与重置

| 组名 | 维度 | 内容 |
|---|---:|---|
| `student` | 144 | 8 帧 ×（机械臂相对默认角度6 + 角速度6 + 上一控制步的机械臂目标角度6） |
| `teacher` | 147 | 无噪声历史144 + 当前真实归一化交互 wrench 3 |
| `policy` | 147 / 144 | `observation_mode=teacher` / `student` |
| `critic` | 165 | teacher147 + EE / payload 各自的位置3、线速度3、线加速度3 |

历史从旧到新排列，重置环境清零；同一步多次读取观测不重复推进历史。student 位置噪声均匀 ±0.01 rad，速度噪声均匀 ±1.5 rad/s；teacher 使用干净历史，关闭观测归一化。critic 的位置以 base 原点为参考，位置、世界线速度和有限差分线加速度均旋转到当前 base 坐标轴。这里的速度没有减去 base 速度，角速度/姿态不额外进入 critic。

`extras["teacher_obs"]` 与 `extras["wrench_target"]` 随 step 返回。student 的 144 维环境观测需要后续算法拼接估计 wrench 3 维，才能构成教程的 student 网络输入147；直接用现有 PPO 训练144维网络不等于完成 student 蒸馏。

每次 episode reset 独立采样 `c_bar∈[-1,1]^3`，2% 概率置零；时间包络为 10% 上升、80% 保持、10% 下降。`c(t)=wrench_scale*s(t)*c_bar`，载荷受到 base 坐标系下的 `Fx=40*cx, Fy=40*cy N, Tz=10*cz Nm`，每个物理子步旋转到世界坐标后重新施加。40 N 是**每轴**范围，水平合力可达到约56.6 N。真实 `c(t)` 同时作为 teacher 输入和速度跟踪参考。

EE 额外受到世界坐标系周期扰动：`Fz=2*sin(2πt)` N，`Tx=sin(2πt+π/4)`、`Ty=sin(2πt+π/2)` Nm。

默认随机化：独立载荷质量0–10 kg（零端以1 g近似，惯量随质量同比缩放）；base 增重−2到5 kg；摩擦0–1.5；初始 x/y ±1 m、yaw ±π、根部线/角速度各分量±0.5。保留低层训练的关节执行器0–4物理步延迟和腿关节初始角度随机化。停用旧 locomotion 的 EE 增重、随机 wrench、推搡、COM/惯量/增益随机化，避免叠加两套负载模型。

终止条件为 base 接触与10秒 timeout；timeout 标记供 RSL wrapper bootstrap。Rough 地形课程改用 wrench 对应的速度参考衡量进度，不依赖已移除的速度 CommandTerm。

## 奖励与教程中的符号问题

教程给出 `exp(-error²/0.25)`，但表格同时给出负的跟踪权重。直接最大化这样的 reward 会鼓励速度偏离目标。因此默认采用 **`1-exp(-error²/0.25)` 作为 cost，保留负权重**，使优化方向正确。这是明确的解释性改动，不能声称与教程表格逐字一致。若需检查字面版本，可设置 `env.literal_tracking_rewards=true`，但不建议以此开始正式训练。

| cost | weight（Isaac RewardManager 再乘一次 dt） |
|---|---:|
| 平动跟踪 | −0.02 |
| 转动跟踪 | −0.01 |
| payload 相对 base 高度超出 [0.05,0.25] m 的平方距离 | −0.003 |
| 18关节 torque² 之和 | −2e−9 |
| 18关节 `dq²+0.00025*ddq²` 之和 | −2e−4 |
| 高层处理后动作 `Δa²+0.5*a²` 之和 | −2e−4 |
| 非 timeout 终止 | −50/dt（每次事件实际−50） |
| 机械臂偏离默认角度的 L2 范数 | −0.003 |

**GO2 几何需要实测校准：**沿用已训练 LL 的机械臂默认角度 `[0,.8,-1.2,0,.4,0]`，没有修改低层角度参考。这个姿态的 EE 较高，载荷起始高度可能超出教程0.05–0.25 m区间。smoke 会报告 `mean_payload_height_in_base_m`。看过视频和高度后再决定是否调整运输初始姿态、抓取变换或 GO2 的高度目标；不能直接把 ANYmal 的工作空间视为已适配。

## 服务器初始 smoke

仅同步代码；以下命令由你在服务器执行。无需重新训练或替换当前 locomotion checkpoint，也不需要额外 pip 升级。

```bash
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction
POLICY="$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt"
test -f "$POLICY" || { echo "找不到冻结 LL policy.pt"; exit 1; }

# 4环境、0.5kg、10% wrench、固定初始状态、取消执行器延迟和周期扰动
PYTHONPATH="$PWD/.runtime/rsl_rl_3_0_1${PYTHONPATH:+:$PYTHONPATH}" \
python scripts/paint/smoke_transport.py \
  --ll-policy "$POLICY" --num_envs 4 --steps 550 \
  --payload-mass 0.5 --wrench-scale 0.1 --nominal --headless
```

该脚本使用**真实 wrench 直接生成底盘速度、机械臂零偏移**的参考控制器，仅用于验证接线和负载下的低层稳定性，不是已训练 teacher。它检查：观测维度、数值有效性、LL 参数未改变、部分环境 reset 不污染其他环境、抓取锚点误差小于5 cm、至少存活一次完整 episode；nominal 模式若出现非 timeout 终止会失败。

报告保存在 `logs/paint_transport_smoke/<时间>/report.json`。550步约11秒仿真时间；初始化和实际墙钟耗时会更长。失败时先保留完整 traceback 和报告，排查约束/碰撞及载荷高度，不要直接开启4096环境。

录制视频与检查 student 接口：

```bash
PYTHONPATH="$PWD/.runtime/rsl_rl_3_0_1${PYTHONPATH:+:$PYTHONPATH}" \
python scripts/paint/smoke_transport.py \
  --ll-policy "$POLICY" --num_envs 4 --steps 550 \
  --payload-mass 0.5 --wrench-scale 0.1 --nominal --headless --video --mode student
```

视频也在对应 smoke 目录下。看视频确认：箱子连接在近侧把手、没有初始爆炸/穿模、机械臂能保持、地面接触正常。

通过后先将固定质量提高到1 kg、`--wrench-scale 0.2`；再去掉 `--nominal` 检查随机初始状态、摩擦、执行器延迟和周期扰动。smoke 的 `--payload-mass` 始终指定固定质量；完整0–10 kg随机化由下面训练环境默认配置提供。

## Teacher PPO 接线测试

先用16环境、2次迭代验证 rollout、reward、保存和优化链路。这仍是接口测试，不代表训练收敛：

```bash
PYTHONPATH="$PWD/.runtime/rsl_rl_3_0_1${PYTHONPATH:+:$PYTHONPATH}" \
python scripts/rsl_rl/train.py \
  --task GO2-PIPER-PAINT-Transport-Flat --num_envs 16 \
  --max_iterations 2 --headless \
  "env.ll_policy_path=$POLICY" env.wrench_scale=0.1 \
  env.periodic_disturbance=false \
  'env.events.payload_mass.params.mass_range=[0.5,0.5]'
```

它只训练新的147维输入、9维输出高层 actor，critic 输入165；LL 始终保持冻结。随机初始状态/摩擦仍开启，若频繁跌倒，需要先调试环境和初始条件。日志目录为 `logs/rsl_rl/go2_piper_paint_transport_teacher/`。

只有物理 smoke、视频检查、两次 PPO 更新全部通过后，才逐步扩大并行数和 wrench/mass 范围。例如完整教程参数的候选训练命令：

```bash
PYTHONPATH="$PWD/.runtime/rsl_rl_3_0_1${PYTHONPATH:+:$PYTHONPATH}" \
python scripts/rsl_rl/train.py \
  --task GO2-PIPER-PAINT-Transport-Flat --num_envs 1024 \
  --max_iterations 15000 --headless "env.ll_policy_path=$POLICY"
```

1024是起始规模建议，非经过显存/吞吐实测的上限。当前 LL 仅在 Flat、固定机械臂、轻载下经过用户验证；移动机械臂、完整交互 wrench、10 kg和 Rough 都是新增工作条件，需要根据存活率与跟踪误差调整课程，不能靠增加迭代数保证收敛。

## 本地验证范围

### 已训练 Teacher 的侧翻排查

`scripts/rsl_rl/play.py` 现在为 Transport 输出专用 JSON，保存到 checkpoint 同目录的 `evaluations/`，视频仍在 `videos/play/`。测试仍使用已训练 teacher，而非 smoke 的参考控制器。

若出现侧翻，先固定 checkpoint、seed、载荷，比较 `wrench_scale=0,0.05,0.1,0.2`，增加 `--no-domain-randomization --payload-mass 0.5`。这个选项会固定根部/关节初始状态，取消质量/摩擦随机化和执行器延迟；独立箱体质量仍由 `--payload-mass` 指定，受力仍保留。额外周期扰动用 `env.periodic_disturbance=false` 明确关闭。

```bash
RUN="$PWD/logs/rsl_rl/go2_piper_paint_transport_teacher/2026-10-25_15-29-21_teacher_flat_light"
PYTHONPATH="$PWD/.runtime/rsl_rl_3_0_1${PYTHONPATH:+:$PYTHONPATH}" \
python scripts/rsl_rl/play.py \
  --task GO2-PIPER-PAINT-Transport-Flat --checkpoint "$RUN/model_8600.pt" \
  --device cuda:0 --num_envs 4 --seed 42 --headless --video --video_length 1500 \
  --no-domain-randomization --payload-mass 0.5 --eval-name teacher8600_nominal_01 \
  "env.ll_policy_path=$POLICY" env.observation_mode=teacher \
  env.wrench_scale=0.1 env.periodic_disturbance=false
```

优先读取这些指标：

| 指标 | 含义 |
|---|---|
| `non_timeout_resets` | 非超时终止次数；当前环境只有 base contact 终止，接触箱体等也可能触发，不能直接等同于所有侧翻次数 |
| `max_tilt_deg` / `max_abs_roll_deg` | 控制步采样的最大倾角/绝对滚转角；自动重置可能使最终倒地帧未被采到 |
| `tilt_over_45_sample_fraction` | 倾角超过45度的有效样本比例 |
| `teacher_command_rmse_vs_reference` | 新 teacher 速度指令相对当前归一化 wrench 参考的 RMSE |
| `velocity_rmse_vs_reference` | 当前实际速度相对当前参考的 RMSE，含响应延迟与动力学误差 |
| `arm_target_tracking_rmse_rad` | 实际机械臂关节相对上一控制步目标的 RMSE |
| `arm_posture_deviation_rmse_rad` | 实际机械臂关节相对默认姿态的 RMSE |

平均受力和速度可能因正负方向抵消，不能单看均值判断跟随成功。零力也侧翻时优先检查 teacher 的零力指令、机械臂目标与碰撞；仅随机化场景侧翻时逐项恢复初始速度、摩擦、增重和延迟；随 wrench 增大而失稳时先降低课程强度。若 teacher 指令误差小、实际运动误差仍大，应继续排查低层负载能力、机械臂姿态及载荷约束。暂时不进入 student 蒸馏或 Rough 扩展，也不要仅凭一次30秒视频认定整体稳定。

所有统计来自重置前的控制步状态，跳过 episode 第0步；没有捕获所有物理子步的峰值。当前训练进程无需因评估代码更新而重启，评估应使用新进程。

```bash
python scripts/paint/check_transport_math.py
# 需要独立测试环境的 torch / usd-core；不要为此改动 limx 的 USD 安装
python scripts/paint/check_transport_payload.py
```

前者验证时间包络、动作处理、冻结 LL 观测布局、逐环境历史和奖励方向；后者使用真实 OpenUSD 与轻量 shape-spawner 替身，验证双环境独立刚体、把手、连接路径、锚点/旋转对齐、质量惯量缩放。另已用 RSL-RL 3.0.1 做147/165/9接口的两次 CPU PPO 更新。**这些检查不包含 Isaac PhysX 实际求解；服务器 smoke 尚待执行。**
