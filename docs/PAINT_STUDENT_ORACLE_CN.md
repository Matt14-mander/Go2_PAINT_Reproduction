# 使用真实意图的 Student 原型

此版本只验证高层蒸馏、闭环采样和部署流程，不包含 intent estimator，也不是无特权信息的最终 Student。原 Flat、Rough、Teacher 任务和权重不覆盖。继续使用原 Flat 247→12 底层，不使用尚未验收的 low-speed model_500。

## 接口与训练

- Teacher：已验收 Stage2 原生导出的 JIT（包含 Teacher 归一化）；冻结。
- Student：独立 147→256→128→64→9 ELU MLP；不从 Teacher 复制权重。
- 输入：现有 Student 的 8×18 机械臂历史，按旧到新排列，每帧为关节位置相对默认值、关节速度、执行的机械臂目标，各6维；末尾拼接当前真实意图3维。没有复制 Teacher 的干净历史或 critic 特权状态。
- 噪声沿用已有实现：位置±.01rad、速度±1.5rad/s；可通过 `--clean-student` 做无噪声诊断。这个幅度需要根据后续数据评估，不代表真实传感器标定值。
- 输出为原始9维动作（底盘3、臂6），统一经过 Stage2 死区、限幅、启动保护和臂速率限制。用组均衡MSE监督原始动作；底盘误差按 command_limits 归一化，机械臂按原始残差单位。
- 前100轮由 Teacher 控制；随后400轮逐渐降低 Teacher 执行动作的概率；第500轮起完全由 Student 控制，每个状态仍由 Teacher 标注。每步按环境选择控制者，不把两者动作做连续平均。
- 缓存最近131072组状态/标签，每轮采32步、更新16批，批大小2048；Adam学习率1e-4、梯度裁剪1。训练损失是训练缓存上的拟合误差，不是独立验证集结果。
- 历史更新和部分环境重置复用 TransportAction 的实现，不另外累计一份历史。每次重置清零对应历史，保持与 MuJoCo 初始零历史一致。

这是有真值输入的 DAgger 式蒸馏原型。正式接入估计意图时必须另做输入分布适应和闭环验收。默认2000轮只是首轮预算，没有自动质量验收或自动替换策略。

## 服务器 smoke

同步 `scripts/paint/student_core.py`、`student_oracle.py`、`train_student_oracle.sh`、`check_student_oracle.py`，以及 MuJoCo 的 `paint_cooperate.py`、`transport_contract.py`。

```bash
source /home/matt14/.conda/etc/profile.d/conda.sh
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction

export LL="$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt"
export TEACHER="$PWD/logs/rsl_rl/go2_piper_paint_teacher_stage2/2026-10-26_16-53-46_stage2_small_arm/exported/model_4999/policy.pt"

ls "$TEACHER" "$(dirname "$TEACHER")/manifest.json" "$LL"
GPU=1 bash scripts/paint/train_student_oracle.sh smoke
```

若 Teacher 导出目录不同，修改 TEACHER 为实际目录，必须同时有原生导出的 manifest.json。脚本校验 Teacher hash、Stage2契约及LL一致性；旧manifest只有ll_policy_path时会读取该文件核对哈希，缺失则要求重新原生导出，不能猜测。

smoke：8环境、3轮，每轮256步，覆盖从 Teacher 到 Student 控制、回合重置、监督更新、保存和JIT数值一致性。然后重新启动4环境做800步纯Student评估。随机初始化的3轮Student可能跌倒；`SMOKE PASSED`只代表流程走通，不代表行为合格。异常/非有限值会失败。

训练不使用 RSL-RL runner，因此本入口不需要切换或安装RSL版本。若需重新用play.py导出Teacher，仍使用原隔离RSL3环境，不升级共享limx环境。

## 正式训练

建议先看smoke的evaluation.json，确认维度、导出、回合重置都正常，再运行：

```bash
tmux new-session -s paint_student_oracle
```

进入后重新激活limx，设置上述LL和TEACHER绝对路径，再执行：

```bash
GPU=1 NUM_ENVS=256 ITERATIONS=2000 bash scripts/paint/train_student_oracle.sh train
```

每100轮保存独立模型及 `exported/model_N/policy.pt`、manifest.json。原始模型只读。日志在 `logs/paint_student_oracle/train_时间_PID/`，包括train.jsonl、TensorBoard、run.json。当前入口不支持恢复优化器/采样状态；重新调用是新run，不冒充续训。

## Isaac 闭环对照评估

将S替换为实际Student导出；OUT必须是未使用的新目录。默认带噪观测，不更新权重。先固定seed123、0.5kg、wrench_scale=.2，再分别改变seed、负载和力大小，Teacher与Student使用相同条件。

```bash
S="$PWD/logs/paint_student_oracle/<实际训练目录>/exported/model_2000/policy.pt"
OUT="$PWD/logs/paint_student_oracle_eval/$(date +%Y%m%d_%H%M%S)"
for CONTROL in teacher student; do
  env -u DISPLAY -u WAYLAND_DISPLAY -u CUDA_VISIBLE_DEVICES \
    CUDA_DEVICE_ORDER=PCI_BUS_ID LIVESTREAM=0 ENABLE_CAMERAS=0 \
    python -u scripts/paint/student_oracle.py eval \
      --teacher "$TEACHER" --ll-policy "$LL" --student "$S" \
      --output "${OUT}_${CONTROL}" --eval-controller "$CONTROL" \
      --num_envs 16 --steps 1500 --seed 123 --payload-mass .5 --wrench-scale .2 \
      --device cuda:1 --headless \
      --kit_args="--no-window --/renderer/multiGpu/enabled=false --/renderer/activeGpu=1"
done
```

报告分开记录原始动作模仿RMSE、相对真实意图的实际速度RMSE、非超时/超时重置、最大倾角。Teacher与Student在不同闭环轨迹上运行，不能把这些轨迹当作逐状态配对；Student轨迹上的Teacher标签才是同状态监督对照。终止后自动重置状态不计入运动误差，终止瞬间倾角可能遗漏。

## MuJoCo 有窗口部署

在服务器图形桌面终端执行（保留有效DISPLAY）；设置上述LL和S。该入口只接受 `external-share`：用已知箱体外部施力均分并转换为各机器人局部参考意图。这是多机器人demo的真值代理，不是训练中单机器人真实意图的严格物理等价，仍需sim-to-sim验证。不会输入真实grasp测量。

```bash
python mujoco/deploy/deploy_mujoco/go2_piper/paint_cooperate.py \
  --ll-policy "$LL" --controller student-oracle --hl-policy "$S" \
  --require-stage2-contract --drive wrench --wrench-source external-share \
  --force-world 16 0 0 --max-speed .35 --max-yaw-rate .4 \
  --keyboard --duration 60 --camera-azimuth 25 \
  --report "logs/mujoco_coop/student_oracle_$(date +%Y%m%d_%H%M%S).json"
```

没有headless，没有录像。MuJoCo使用干净历史，训练的噪声用于鲁棒性；需噪声影响对照时运行Isaac带噪/无噪评估。Stage2 arm/限幅/施力包络参数由manifest加载，LL哈希必须相同。用相同命令改 `--controller teacher --hl-policy "$TEACHER"`，另选报告路径，即为部署基线。Student输入源被锁定，不能把 `external-share` 改成 `grasp` 就宣称已完成estimator接入。

## 本地验证范围

`python scripts/paint/check_student_oracle.py` 覆盖输入隔离、缓存覆盖/梯度隔离、控制权切换、真实监督更新、JIT一致性、策略来源与部署契约拒绝。实际Isaac训练、真实Teacher数据质量及闭环跟随性能必须在服务器smoke和正式评估中验证。
