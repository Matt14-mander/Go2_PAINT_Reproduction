# Teacher 第一阶段：固定机械臂学习稳定跟随

新任务：`GO2-PIPER-PAINT-Transport-Flat-Stage1`。它针对上一轮饱和/摆臂问题建立独立的平地高层训练基线，保留原始Flat/Rough任务、LL权重和Admittance Demo。第一阶段应从新Teacher初始化开始，不加载旧8600/9999高层权重。

## 第一阶段的实际配置

| 项目 | 配置 |
|---|---|
| 观测与动作 | actor147、critic165、高层输出9，LL仍247→12 |
| 机械臂 | `actions.transport.freeze_arm=True`，无论网络的6维臂输出多大，目标都保持默认姿态；这些输出仍受原始动作成本约束 |
| 底盘指令范围 | x/y ±0.35m/s，yaw ±0.4rad/s，死区0.05 |
| 参考受力 | `wrench_scale=.2`，Fx/Fy各±8N、Tz±2Nm，沿用10%/80%/10%包络 |
| 零力回合 | 从原2%提高到20%，明确训练站稳 |
| 载荷 | 固定独立箱体0.5kg |
| 初始状态 | 默认关节角，根部相对env原点x/y/yaw为0，初始速度为0 |
| 随机化 | 暂停base增重、摩擦随机化、周期扰动和执行器延迟 |
| 自碰撞 | 暂时关闭，与已验证的MuJoCo低层/Admittance基线接近，后续阶段再单独恢复 |
| 终止 | 保留base contact；增加约40°倾角终止（0.7rad）和10秒timeout |
| PPO | 初始std0.1、学习率1e-4、entropy系数0.001、每100轮保存，默认5000轮预算 |
| 日志 | `logs/rsl_rl/go2_piper_paint_teacher_stage1/` |

这里的固定臂是固定PD目标，实际角度仍会受重力、负载和接触影响；不是把臂改成运动学物体或关闭物理。

本阶段是明确的GO2训练适配，不是教程表格的逐字复刻。高层执行界面仍保留9维，方便后续阶段衔接，但第一阶段导出策略部署到MuJoCo必须继续冻结机械臂，不能直接按原全臂Teacher的执行规则加载。

## 奖励修正

RewardManager仍只乘一次控制dt，终止事件仍为−50。

| 项目 | 原Flat | Stage1 |
|---|---:|---:|
| 平动跟踪cost | −0.02 | −2.0 |
| 偏航跟踪cost | −0.01 | −1.0 |
| 高层速度指令对真实wrench参考的平方误差 | 无 | −0.5 |
| 基座倾斜cost | 无独立项 | −0.5 |
| 关节速度/加速度cost | −2e−4 | −1e−5 |
| 原始高层动作变化和幅度cost | −2e−4 | −2e−3 |
| payload高度cost | 套用0.05–0.25m | 本阶段停用，避免默认GO2工作点不一致 |
| 机械臂姿态cost | 有 | 本阶段停用，臂目标已固定 |

速度指令引导是辅助项，实际基座速度跟踪仍有更高权重，所以不应把“网络输出接近参考”当成“实际运动已经跟上”。原始动作成本会惩罚超限网络输出；不再对 `-1` 和 `-100` 给出同样的代价。

这些参数是修正后的初始训练方案，不构成收敛保证。应查看前100/500轮的跟踪误差、异常终止和原始动作范围，避免只等待5000轮结束。

## 同步范围

服务器需要完整更新：

- `source/LeggedManip_Lab/`：注册、Stage1 env/runner、动作处理、reward、reset概率字段都在这里，不能只复制一个配置文件。
- `scripts/paint/`：更新smoke和检查脚本。
- `scripts/rsl_rl/`：更新Stage1正确限幅的评估统计、原生导出metadata。
- 后续MuJoCo验证时更新 `mujoco/deploy/deploy_mujoco/go2_piper/paint_cooperate.py`，需要新 `--freeze-arm` 参数。

保留服务器已有 `logs/`、`.runtime/rsl_rl_3_0_1/` 和低层导出文件，不需要重装环境或重新训练底层。

## tmux启动：先物理smoke，成功后自动启动正式训练

```bash
tmux new-session -s paint_teacher_stage1
```

进入会话后执行：

```bash
conda activate limx
cd ~/workspace/Go2_PAINT_Reproduction
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$PWD/.runtime/rsl_rl_3_0_1${PYTHONPATH:+:$PYTHONPATH}"
LL="$PWD/logs/rsl_rl/go2_piper_paint_loco/2026-10-24_18-05-41_train_flat_20261024_180532_13553c07/exported/policy.pt"
mkdir -p logs/teacher_stage1_console
set -o pipefail

if python -u scripts/paint/smoke_transport.py \
  --task GO2-PIPER-PAINT-Transport-Flat-Stage1 \
  --ll-policy "$LL" --device cuda:0 \
  --num_envs 4 --steps 550 --payload-mass 0.5 \
  --wrench-scale 0.2 --nominal --headless \
  2>&1 | tee logs/teacher_stage1_console/smoke.log
then
  python -u scripts/rsl_rl/train.py \
    --task GO2-PIPER-PAINT-Transport-Flat-Stage1 \
    --device cuda:0 --num_envs 512 --max_iterations 5000 \
    --run_name stage1_fixed_arm --logger tensorboard --headless \
    "env.ll_policy_path=$LL" \
    2>&1 | tee "logs/teacher_stage1_console/train_$(date +%Y%m%d_%H%M%S).log"
  echo "训练进程退出，状态码：$?"
else
  echo "Stage1 smoke 未通过，未启动正式训练；请检查上方 traceback 和 smoke 报告。"
fi
```

`CUDA_VISIBLE_DEVICES=1`选择物理GPU1供CUDA使用，程序内编号为cuda:0；如果使用另一张卡，修改该环境变量。没有`--resume`或旧checkpoint参数，Teacher会新建，冻结LL仍从旧的`policy.pt`加载。

smoke包含极端臂输入测试：在不推进物理的情况下输入100，检查机械臂目标仍是默认角、底盘实际限幅为0.35/0.35/0.4，防止只更新配置而执行代码仍是旧版本；随后运行550步物理检查、reset隔离、冻结LL和连接锚点检查。只有返回成功才开始正式训练。

按Ctrl+B再D退出tmux保持训练；重新进入：`tmux attach-session -t paint_teacher_stage1`。

## 验收与后续策略部署

先在同一个Stage1任务里用原生 `play.py` 评估新checkpoint，重点检查零力时的输出、指令/实际速度RMSE、非timeout终止、机械臂实际跟踪和最大倾角。不要把新Stage1权重用原 `GO2-PIPER-PAINT-Transport-Flat` 默认全臂配置回放。

MuJoCo部署时使用：

```bash
python mujoco/deploy/deploy_mujoco/go2_piper/paint_cooperate.py \
  --ll-policy "$LL" --controller teacher --hl-policy /path/to/stage1/exported/policy.pt \
  --drive wrench --wrench-source external-share --freeze-arm \
  --force-world 16 0 0 --max-speed 0.35 --max-yaw-rate 0.4 \
  --keyboard --duration 60 --camera-azimuth 25
```

这只是后续命令模板；应先完成Isaac中Stage1的验收。下一阶段开放小幅机械臂残差的配置本次尚未实现，不能简单去掉`--freeze-arm`就认为已经训练好了全臂策略。

本地已通过新旧官方RSL配置API校验、固定臂/动作限幅和奖励回归检查，以及实际RSL-RL3.0.1的两次CPU PPO更新（147/165/9，std=.1）。本机不能运行Isaac PhysX，服务器物理验证由上述smoke负责，本次未连接服务器或启动训练。
