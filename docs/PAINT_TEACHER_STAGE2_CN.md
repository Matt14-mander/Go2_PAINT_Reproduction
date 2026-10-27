# Stage2A：小幅开放机械臂

任务名：`GO2-PIPER-PAINT-Transport-Flat-Stage2`。

从已验收的 Stage1 `model_4999.pt` 初始化，继续训练高层。LL 使用原 Flat 导出策略并保持冻结；teacher 仍观察已知外力。grasp 原始反力作为诊断输入保留，未混入本轮训练。Stage2 是 teacher 内部的分阶段训练，不是 student 或 intent estimator。

## 执行与训练参数

| 项目 | Stage2A |
|---|---|
| actor / critic / action | 147 / 165 / 9，与 Stage1 相同 |
| 机械臂目标 | 默认角 + 0.1 × 原始臂动作，每关节残差限幅 ±0.1 rad，再应用软关节限位 |
| 机械臂目标变化速度 | 每关节最大0.2 rad/s，50 Hz下每步最大0.004 rad |
| 初始高层输出 | 底盘输出保持Stage1权重；仅机械臂最后一层的6行权重和偏置置零 |
| 初始探索std | 底盘3维0.05，臂6维0.1（乘动作尺度后约0.01 rad，后续还受限速约束）；训练中std仍可学习 |
| PPO | lr 5e-5、entropy 5e-4，其余继承Stage1；新优化器，从第0轮开始，5000轮预算、每100轮保存 |
| 物理条件 | 平地、0.5 kg、当前Flat LL、无新增域随机化；20%回合零力 |
| 底盘指令 | ±0.35 m/s、±0.4 rad/s，死区0.05 |

15秒episode按以下时序运行：0–1秒强制零底盘指令和默认臂目标；1–3秒外力渐增；3–10秒保持；10–12秒渐减；12–15秒外力为零、高层继续控制，用于检查主动停稳。训练启动时不随机设置episode时间，避免跳过落稳阶段。

保留Stage1跟随、底盘倾角和原始动作成本；新增/恢复机械臂姿态成本（−0.02）、目标角速度均方成本（−0.005）、软限位越界平方成本（−0.5）、箱体在基座坐标系roll/pitch角速度平方成本（−0.02）。没有使用旧的不适配GO2的payload高度区间，也没有直接奖励“机械臂动得越多越好”。这是待物理验收的初始配方，不能保证开放臂一定优于固定臂。

这是受控的第一档幅度。通过验收后才考虑另起实验增加臂幅度；没有自动按训练轮数扩大范围，保证中途导出时执行契约明确可复现。

## 同步范围

完整同步下列目录中的本轮改动，保留服务器 logs 和 .runtime：

- `source/LeggedManip_Lab/`：任务注册、runner、动作处理、奖励和配置。
- `scripts/rsl_rl/`：训练迁移、评估、导出；新增 `stage2_warm_start.py`。
- `scripts/paint/`：smoke和启动脚本；新增 `train_teacher_stage2.sh`。
- `mujoco/deploy/deploy_mujoco/go2_piper/`：部署入口及新增 `transport_contract.py`。

首次迁移支持已验证的RSL-RL 3.x（服务器隔离环境3.0.1）。现代5.x的普通配置仍保留，但本轮warm-start不会冒用不同的内部API，遇到不支持版本会明确退出。无需更新服务器依赖。

## smoke，然后启动训练

在已有limx环境、选择有足够余量的GPU后执行；脚本不会停止其他任务。

```bash
tmux new-session -s paint_teacher_stage2
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction

GPU=1 bash scripts/paint/train_teacher_stage2.sh smoke && \
GPU=1 NUM_ENVS=512 ITERATIONS=5000 bash scripts/paint/train_teacher_stage2.sh train
```

默认使用本次Stage1路径：

```text
logs/rsl_rl/go2_piper_paint_teacher_stage1/2026-10-26_12-12-23_stage1_fixed_arm/model_4999.pt
```

通过 `STAGE1_CKPT=/absolute/path/model.pt` 和 `LL=/absolute/path/policy.pt` 可覆盖默认文件。smoke运行1600步（32秒），包含极端臂动作的启动/限速检查、实际小幅正弦臂动作、至少两个完整episode，以及LL冻结和抓取锚点检查。失败时 `&&` 阻止启动训练。smoke不是学习到的策略质量评估。

训练日志进入 `logs/rsl_rl/go2_piper_paint_teacher_stage2/`，终端日志在 `logs/teacher_stage2_console/`。每次 `train` 都是从Stage1建立新实验，不是继续已有Stage2；迁移记录在 `warm_start.json`。Ctrl+B再D脱离tmux。

恢复已有Stage2时不要再用 `--warm-start`，否则臂输出会被重新初始化。使用原生 `train.py --task GO2-PIPER-PAINT-Transport-Flat-Stage2 --resume --load_run <该Stage2目录名> --checkpoint model_XXXX.pt ...`，并保留原来的LL路径、环境和PPO配置；`--max_iterations` 是继续增加的迭代预算。

## 中期评估与导出

例如训练到1000轮，设置实际RUN后评估（可按显示环境选择加 `--headless`，默认打开窗口）：

```bash
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD/.runtime/rsl_rl_3_0_1${PYTHONPATH:+:$PYTHONPATH}"
RUN="$PWD/logs/rsl_rl/go2_piper_paint_teacher_stage2/<实际run目录>"
LL="$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt"
for CASE in force zero; do
  SCALE=0.2
  if [ "$CASE" = zero ]; then SCALE=0.0; fi
  python -u scripts/rsl_rl/play.py \
    --task GO2-PIPER-PAINT-Transport-Flat-Stage2 \
    --checkpoint "$RUN/model_1000.pt" \
    --device cuda:0 --num_envs 4 --seed 42 --eval-steps 2250 \
    --eval-name "stage2_1000_$CASE" --export-dir "$RUN/exported/model_1000" \
    "env.ll_policy_path=$LL" "env.wrench_scale=$SCALE" || break
done
```

先看异常重置、速度误差、倾角和原始动作超限比例，确认没有明显回退；再看 `max_arm_target_offset_rad <= 0.1`、`max_arm_target_speed_rad_s <= 0.2`（允许浮点误差）。这些是目标限制，不是实际关节角的硬限制。`release_velocity_rmse_vx_vy_wz` 单独统计每个episode最后1.5秒的停稳情况；`release_samples=0` 表示没有有效停稳样本，不能当作通过。

导出包含 `policy.pt` 和 `manifest.json`，两者要一起复制。manifest保存实际动作尺度、幅度、限速、启动/撤力时序、死区、控制周期与策略hash。评估Stage2必须使用Stage2任务，不能用Stage1或旧全幅动作任务。

## MuJoCo双机

Stage2不再传 `--freeze-arm`。部署默认自动识别Stage2 manifest；加 `--require-stage2-contract` 可在漏传metadata时阻止误用旧动作倍率。启动与外力渐变参数以manifest为准，最后保留3秒零力；对于60秒demo不重复15秒训练episode，也不自动reset。

```bash
python mujoco/deploy/deploy_mujoco/go2_piper/paint_cooperate.py \
  --ll-policy "$LL" --controller teacher \
  --hl-policy "$RUN/exported/model_1000/policy.pt" --require-stage2-contract \
  --drive wrench --wrench-source external-share --audit-grasp \
  --force-world 16 0 0 --max-speed 0.35 --max-yaw-rate 0.4 \
  --duration 60 --camera-azimuth 25 \
  --report logs/mujoco_coop/stage2_1000_external_16N.json
```

与Stage1使用同样外力比较，并另做零力测试。grasp日志用于比较内部拉扯、受力峰值和波动；当前版本仍不宣称teacher可直接用原始grasp反力闭环。后续intent estimator/student尚未实现。

## 批量teacher验收

新增 `scripts/paint/accept_teacher_stage2.sh`，需与最新 `paint_cooperate.py` 和 `transport_contract.py` 一起同步。脚本仅评估，不训练，默认使用本轮Stage2 run和原Flat LL，默认打开窗口、不录制。

```bash
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction
BASE="$PWD/logs/teacher_stage2_acceptance/$(date +%Y%m%d_%H%M%S)"
OUT="$BASE/mujoco" bash scripts/paint/accept_teacher_stage2.sh mujoco
OUT="$BASE/isaac" DEVICE=cuda:1 bash scripts/paint/accept_teacher_stage2.sh isaac
```

MuJoCo依次运行5项，每项60秒：总外力Fy=+16/-16 N、总Tz=+2/-2 Nm、Fx=16 N重复施加/撤去。重复项用新增 `--force-period 15`：每15秒依次1秒零力、2秒渐增、7秒保持、2秒渐减、3秒零力；60秒4个周期，机器人连续运行，不reset。普通方向项仍为一次施力并在末尾撤力。初次验收不启用keyboard，以便复现。

Isaac交叉测试seed=42/43/44、payload=0.25/0.5/1.0 kg、受力/零力，共18项，每项4环境45秒仿真时间；只改变种子、固定负载和外力尺度，其余沿用Stage2条件。0.25/1.0 kg属于训练固定0.5 kg之外的泛化测试，失败应单独记录。MuJoCo没有随机初始化入口，不用虚假的seed参数重复同一确定性场景。Isaac默认取消CUDA屏蔽，`DEVICE=cuda:1` 指实际CUDA编号；选择资源足够的卡。

每项stdout与JSON保存在对应OUT目录（Isaac原始报告仍在run/evaluations，另复制一份到OUT）。失败会记录并继续其余项，整批最后返回非零；`completed`只表示程序运行完成，不能替代质量验收。检查各方向运动是否正确、限幅比例、倾角、抓取误差、零力与撤力后速度，以及不同种子/负载的异常重置。MuJoCo重复撤力的全过程见CSV，单个最终状态不能代表所有周期都停稳。

## 验证边界

本地检查覆盖Tensor动作限幅/限速、启动与撤力时序、NumPy部署与训练计算一致性、真实RSL-RL3.0.1迁移及两次CPU PPO更新、正常resume保留臂输出、真实MuJoCo加载带metadata假策略并执行臂限幅。这里只验证接口与数值行为，不证明真实策略收敛；Isaac PhysX smoke与新训练需在服务器运行。
