# MuJoCo 抓取反力验证

当前抓取由末端与箱体之间的 weld 约束实现，初始化即抓牢；没有夹爪接触抓取、滑移或抓取规划。箱体是自由刚体。`grasp` 读取每个 weld 对应的约束反力，还原为世界坐标系、末端原点处的箱体→机器人六维 wrench。Teacher 取旋转到基座坐标的 Fx/Fy/Tz，除以 40/40/10 并限幅到 [-1,1]。

新增 `--audit-grasp` 是旁路测量：无论控制使用 external-share 还是 grasp，都记录实际反力与理想分摊输入。它不去偏置、不滤波、不改变控制输入，也不更改训练或策略。Stage1 部署继续使用 `--freeze-arm`。

## 同步

将以下文件同步到服务器对应目录，无需重训或重新导出：

- `mujoco/deploy/deploy_mujoco/go2_piper/paint_cooperate.py`
- `mujoco/deploy/deploy_mujoco/go2_piper/grasp_audit.py`（新文件，必须一起同步）

## 先旁路观察，再闭环

在服务器图形桌面终端执行，不录制：

```bash
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction
RUN="$PWD/logs/rsl_rl/go2_piper_paint_teacher_stage1/2026-10-26_12-12-23_stage1_fixed_arm"
LL="$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt"

python mujoco/deploy/deploy_mujoco/go2_piper/paint_cooperate.py \
  --ll-policy "$LL" --controller teacher \
  --hl-policy "$RUN/exported/model_4999/policy.pt" --freeze-arm \
  --drive wrench --wrench-source external-share --audit-grasp \
  --force-world 8 0 0 --max-speed 0.35 --max-yaw-rate 0.4 \
  --settle-seconds 3 --ramp-seconds 3 \
  --keyboard --duration 60 --camera-azimuth 25 \
  --report logs/mujoco_coop/stage1_grasp_shadow.json
```

按 X 撤力观察零力，再用 W/S、A/D、Q/E 分别改变单轴力或力矩；初始先用小力。先不要同时增加多个轴。停止后保留 JSON 和两个 CSV。为便于对比，可不加 keyboard 并分别用 `--force-world 0 0 0`、`8 0 0`、`-8 0 0`、`0 8 0`、`0 -8 0`、`0 0 0.5`、`0 0 -0.5` 独立运行，每次指定不同 report。

确认测量后，用同一命令改为 `--wrench-source grasp`，并将 report 改为 `logs/mujoco_coop/stage1_grasp_closed_loop.json`。此时 Teacher 真正使用每台机器人自身的抓取反力。当前 Teacher 是已知外力输入训练，连接反力并不等价；若反力闭环失败，先分析输入差异，不能仅凭失败认定策略或读取实现出错。

## 结果含义

- 普通 `.csv`：实际控制输入、动作、运动和箱体外力。
- `.grasp.csv`：各机器人原始 world/body 六维反力、归一化 grasp 输入、归一化 external-share 输入，以及汇总到箱体质心的机器人→箱体 wrench。
- JSON `grasp_audit.static_world_wrench_mean`：settle 最后一秒的反力均值，供观察静载偏置；没有足够窗口则为 null。它是读数统计，不保证机器人完全静止，也不自动从输入中扣除。
- `normalized_grasp_vs_external_share_rmse`：settle 之后两种归一化输入的差异。真实反力包含重力支撑、惯性及伙伴拉扯，该误差不要求为零，也不作为通过/失败标准。
- `max_abs_extraction_generalized`：将读出的反力映射回箱体自由度，与该 weld 的原始广义约束力比较，检查符号、作用点换算和提取是否一致。
- `max_abs_payload_balance_generalized`：检查箱体自由度上的动力学残差，包含全部平滑力、惯性和其他约束。不是只把外力与抓取力相加。残差应接近求解器数值精度；显著异常需结合 solver warning 检查。六分量是广义力，不能标成世界 Fx/Fy/Fz/Tx/Ty/Tz。最大值包含落地瞬态。

抓取读数的重构和动力学检查不是独立硬件传感器校验，也不能证明理想 weld 等价于真实摩擦抓取。测试先检查读数正确，再检查 teacher 对该输入分布的适应性。不得把“输入一致性验证”当成 intent estimator 或 Stage2 训练。

## 本地验证

`python scripts/paint/check_mujoco_cooperate.py` 使用真实 MuJoCo，覆盖静载重量、释放约束后箱体下落、正负外力/力矩下的广义力重构、反号故障检出、坐标旋转与归一化、日志不改变物理状态，以及假权重的命令行旁路输出。假权重只验证接口，不代表真实策略跟随性能；后者需在服务器运行上述命令验证。
