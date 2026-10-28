# Flat 底层低速微调

此分支从原 Flat `model_9999.pt` 加载 actor/critic，保持 **247 维 actor 观测、12 维动作、动作缩放 0.25、50 Hz 策略和原 PD**。原任务、checkpoint、teacher 及 demo 策略不覆盖。新训练只写 `go2_piper_paint_loco_low_speed*` 实验目录。没有自动替换 teacher 的 LL。

## 训练设计

- 新任务：`GO2-PIPER-PAINT-Loco-Flat-LowSpeed`。从原 Flat 权重热启动，优化器和迭代计数重新开始。记录来源路径、SHA256、原始和微调初始探索 std。保留 actor 均值参数；std 在原值基础上限制到 `[0.05,0.20]`，不是重置为 1。
- 指令：15% 站立，30% 低速纯转向，20% 低速单轴平移，25% 原范围混合指令，10% 启动—停止—反向。低速绝对值连续覆盖 `[0.04,0.22]`，一半集中在 `.05/.1/.2` 附近；正负等概率。比例是采样类别比例，实际零指令时间还包括序列的停止段。
- 12 秒重采样；序列：0–2s 停、2–5s 正向、5–7s 停、7–10s 反向、10–12s 停。训练 episode 24 秒。关闭旧速度范围 curriculum，避免覆盖混合分布。
- 保留旧宽范围跟踪及所有运动成本/足部奖励，额外加入每轴精细跟踪：`exp(-(error/width)^2)`；vx/vy 权重各 `.75`、width `.04 m/s`，wz 权重 `1.5`、width `.06 rad/s`。按三轴指令绝对值最大值，在 `.20→.35` 平滑淡出。零目标也计算，限制纯转向时横移。没有人为最低输出速度或指令放大。
- 初始阶段：末端等效附加质量 `[0,1]kg`，无额外推扰/外力，噪声为原幅度 25%，延迟 0–1 个物理步；保留轻微摩擦、质量、COM、增益、惯量和关节初态随机化。
- PPO：固定学习率 `5e-5`，clip `.1`，epochs 3，minibatches 4，entropy `.001`，梯度范数 `.5`，rollout 24，gamma `.99`，lambda `.95`。固定学习率避免原 adaptive 调度把学习率升高；`desired_kl` 在 fixed 调度下不是硬性 KL 限制。
- `...-LowSpeed-Restore`：同样采样/奖励，但恢复原 Flat 的噪声、0–4 步延迟、质量 `[0,10]kg`、推扰和末端外力。等初始阶段验收后，从选中的 checkpoint 进入这一步，避免未经验证直接切入强随机化。

## 服务器准备与 smoke

先把本次修改同步至服务器根目录，至少包含以下目录中的修改和新增文件：

```text
source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/
scripts/rsl_rl/
scripts/paint/
docs/PAINT_LOW_SPEED_CN.md
```

保持现有 `limx` 和 `.runtime/rsl_rl_3_0_1`。无需升级 Isaac 或 RSL-RL。周期回归依赖同环境现有的 `mujoco`、`numpy`、`PyYAML`、`torch`。

```bash
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction
GPU=1 bash scripts/paint/train_loco_low_speed.sh smoke
```

GPU 为设备编号；脚本取消 `CUDA_VISIBLE_DEVICES` 屏蔽，设置 PCI_BUS_ID 排序，通过 `--device cuda:N`、`agent.device=cuda:N` 和渲染 activeGpu 显式选卡。请选择当时空闲的卡，并核对启动日志与 nvidia-smi 的 PCI bus ID。渲染多 GPU 已关闭。纯训练子进程会清除 DISPLAY/WAYLAND_DISPLAY，关闭直播和摄像头，避免继承 ThinLinc 虚拟显示；不影响终端或桌面会话。

smoke 先运行 16 环境/650 步 Isaac 检查，覆盖 12 秒重采样、单环境 reset 隔离和序列执行，再运行 32 环境/2 次真实 PPO 更新、权重迁移、Isaac 原生 JIT 导出以及 CPU MuJoCo 回归。会创建独立 `_smoke` 实验；不会用 smoke 的输出继续正式训练。旧策略低速不达标并不是 smoke 失败；发生非有限值、接口不匹配、导出失败或已有能力退化则失败。最终应出现 `SMOKE PASSED`。

## tmux 正式训练

smoke 成功后运行：

```bash
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction
CONDA_BASE="$(conda info --base)"
tmux new-session -d -s go2_low_speed -c "$PWD" \
  "bash -lc 'source \"$CONDA_BASE/etc/profile.d/conda.sh\" && conda activate limx && GPU=1 NUM_ENVS=1024 ITERATIONS=3000 bash scripts/paint/train_loco_low_speed.sh train'"
tmux attach -t go2_low_speed
```

`Ctrl+B` 后按 `D` 脱离。512 环境可用于资源紧张时；4096 会改变每轮采样数，不作为默认直接替换。训练输出在 `logs/loco_low_speed_console/`，checkpoint 每 100 轮保存在新 run 下，约 `model_2999.pt` 结束。3000 是首轮预算，不是质量保证。

周期回归：起始基线、model_0、每 500 轮和最终 checkpoint 使用 **Isaac 原生 exporter**；评估是独立的 CPU MuJoCo 子进程，训练暂时等待，继续时不推进或重置训练环境。详细进度写在 run 的 `regression/*.console.log`，因此终端暂时没有 PPO 新行是正常的。

若回归退出非零，训练停止并保留已保存的 checkpoint；先查看 `regression/*.json` 与 console。不要删除旧策略或把失败文件覆盖到 teacher LL。原 Flat 在新增 mixed 测试本身若跌倒，回归会要求先检查该基线，而不是假设新策略通过。

## 自动防退化和正式验收

周期检查是 `point-contact`、无额外负载、seed42 名义条件，19 个固定指令，每个32秒。它用于快速防退化，不能替代完整验收。

完整验收包含上述19项及3个单轴启停反向序列，默认5个评估种子：42为名义初态，其余种子对12个腿关节施加固定可复现的 ±.005rad 初态偏差、yaw ±.02rad。两策略使用相同偏差。这些是**评估初态种子，不是5次独立训练**。不同负载需单独跑。

固定指令稳态取5–30秒；完整RMSE包含启动过程。序列为0–3s停、3–11s正向、11–16s停、16–24s反向、24–32s停。响应时间使用1秒滑动均速，并要求随后保持1秒；每次启动/停止≤2秒。报告每个指令/种子，不用总均值掩盖方向差异。

| 条件 | 稳态均值绝对误差 | 稳态RMSE |
|---|---:|---:|
| 转向 ±.1rad/s | ≤.02rad/s | ≤.05rad/s |
| 转向 ±.05rad/s | ≤.02rad/s | ≤.04rad/s |
| 转向 ±.2rad/s | ≤.02rad/s | ≤.05rad/s |
| 平移各轴 ±.1m/s | ≤.02m/s | ≤.03m/s |
| 平移 ±.3m/s | ≤.04m/s | ≤.04m/s |
| 转向 ±.3rad/s | ≤.06rad/s | ≤.08rad/s |
| 站立 | 同RMSE | 平移各轴≤.01m/s，yaw≤.025rad/s |

同时要求完整运行、无跌倒/非有限状态、最大倾角<20°；低速测试的非指令轴也限制误差。mixed测试沿用中速阈值。该套阈值是工程初始验收线，不是论文保证。

`preservation` 与 `targets_passed` 分开：原能力（站立、±.3、mixed）的RMSE超过基线 `1.2倍 + [.005,.005,.01]` 判退化；动作变化RMS超过 `1.5倍+.01`、腿力矩RMS超过 `1.3倍+.5Nm`、倾角超过 `1.3倍+2°` 也判退化。基线成功而候选跌倒的任意用例都失败。这是工程保护阈值，不代表统计显著性；不把旧策略低速不动作为模仿目标。

完成训练后，选择 checkpoint 原生导出，再用同一协议对比。以下将所有新报告和新导出放入独立目录，不写原 Flat run：

```bash
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction
export PYTHONPATH="$PWD/.runtime/rsl_rl_3_0_1${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=1
# 替换为实际新 run：
RUN="$PWD/logs/rsl_rl/go2_piper_paint_loco_low_speed/<新run目录>"
OUT="$PWD/logs/low_speed_acceptance/$(date +%Y%m%d_%H%M%S)"
OLD="$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt"
mkdir -p "$OUT"
python scripts/rsl_rl/play.py --task GO2-PIPER-PAINT-Loco-Flat-Play \
  --checkpoint "$RUN/model_2999.pt" --export-only --export-dir "$OUT/exported" \
  --device cuda:0 --num_envs 1 --headless --no-domain-randomization --no-disturbances --payload-mass 0 \
  --kit_args="--/renderer/multiGpu/enabled=false"
python scripts/paint/evaluate_low_speed.py --backend mujoco --policy "$OLD" --report "$OUT/mj_old.json"
python scripts/paint/evaluate_low_speed.py --backend mujoco --policy "$OUT/exported/policy.pt" \
  --baseline-report "$OUT/mj_old.json" --report "$OUT/mj_new.json" --fail-on-regression --require-targets
python scripts/paint/evaluate_low_speed.py --backend isaac --policy "$OLD" --report "$OUT/isaac_old.json" \
  --device cuda:0 --headless --kit_args="--/renderer/multiGpu/enabled=false"
python scripts/paint/evaluate_low_speed.py --backend isaac --policy "$OUT/exported/policy.pt" \
  --baseline-report "$OUT/isaac_old.json" --report "$OUT/isaac_new.json" --fail-on-regression --require-targets \
  --device cuda:0 --headless --kit_args="--/renderer/multiGpu/enabled=false"
```

MuJoCo不创建窗口/录像，适合无人值守批量回归；Isaac在同一场景并行测试所有用例。可用原 `paint_loco.py --physics-profile point-contact` 对候选做有窗口检查。完整验收可分别加 `--payload-mass 0.5`、`--payload-mass 1`，每个负载使用独立的新旧报告，不能跨条件比较。

## 恢复随机化

初始阶段通过名义和轻负载验收后，指定选中的低速 checkpoint，原策略仍作为回归基准：

```bash
GPU=1 NUM_ENVS=1024 ITERATIONS=2000 \
SOURCE_CKPT="$RUN/model_2999.pt" \
bash scripts/paint/train_loco_low_speed.sh restore
```

该步骤是新实验/新优化器；恢复完整原随机化可能使精度暂时下降，周期检查会发现已有能力恶化。恢复后还需用原 Flat-Play 的带随机化/扰动测试检查鲁棒性，然后再接回 Stage2 teacher。名义协议通过不代表强扰动通过，也不代表共享箱体约束下通过。

本地 CPU 测试覆盖采样/对称性/序列、奖励排序和淡出、真实RSL3迁移/PPO更新、验收死区与波动检测、真实MuJoCo执行。实际Isaac运行以及训练效果须由上述服务器smoke/训练验证。
