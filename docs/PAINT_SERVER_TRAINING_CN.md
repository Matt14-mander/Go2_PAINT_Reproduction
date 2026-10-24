# 同步到服务器并启动 PAINT 底层训练

脚本不会自动安装 Isaac Lab、升级环境或租用 GPU。使用 Isaac Sim 5.1.0 / Python 3.11，并选择配套的 Isaac Lab/RSL-RL 接口：旧版 policy + TensorDict 包装器配 RSL-RL 3.x（>=3.0.1），新版 actor/critic 配 >=5.0.1。
下面的服务器路径是示例，请替换为实际持久化磁盘目录。代码同步本身不会启动训练。
本地已通过 8 项流程单元测试、Python 编译与补丁检查。使用官方新旧配置源码分别验证了 16 个 runner 配置，并使用实际 RSL-RL 3.0.1 在合成 TensorDict 环境完成 2 次 CPU PPO 更新（隔离 Git 日志）。这不等同于 Isaac 物理 smoke；未连接服务器或运行 Isaac 训练。

## 环境版本不匹配时先诊断

旧脚本会直接拒绝 `rsl-rl-lib 3.0.1`；新版本已加入项目内兼容层，不需要为此升级现有环境。
兼容层将网络结构、Gaussian 标准差和观测归一化字段转换为原生 `RslRlPpoActorCriticCfg`，并使用 `policy/critic` 观测组；新版使用 `actor/critic` 观测组。
7 个机器人配置统一从兼容层导入，避免任务扫描时因其他机器人的配置导入失败。
训练和评估脚本也不再强制导入旧版不存在的 `handle_deprecated_rsl_rl_cfg`。
不要只修改版本检查，也不要直接单独升级 RSL-RL，Isaac Lab 包装器必须配套。
先同步整个最新工作区，然后检查实际接口：

```bash
python scripts/paint/server_train.py diagnose
```

该命令只读取 Python/包版本、安装目录及配置类/包装器源码特征，不导入 Isaac Sim，不要求 CUDA 可用，不安装任何依赖。
未找到 pip 版本时会显示 null；源码安装仍可通过 module_paths 定位。
服务器的自定义 Isaac Lab fork 尚未在本地执行真实物理验证，最终以该环境的 smoke 为准。

针对服务器 `limx` 环境（Isaac Sim 5.1.0、isaaclab 0.48.0、isaaclab_rl 0.4.4、RSL-RL 3.0.1、Torch 2.7.0+cu128），更新后在服务器逐行执行：

```bash
cd ~/workspace/Go2_PAINT_Reproduction
python scripts/paint/server_train.py diagnose
python scripts/paint/server_train.py prepare
python scripts/paint/server_train.py smoke
```

新版 smoke 在物理检查之前，还会启动 AppLauncher 检查 16 个 runner 配置的实例化、必填字段和观测组映射。只同步 `server_train.py` 不够，需同时同步 `source/LeggedManip_Lab`、`scripts/rsl_rl` 与 `scripts/paint`。
旧接口若仍返回 tuple 而非 TensorDict，预检会停止，不会混用不兼容的 runner。

## 1. 本地 Windows 同步

在仓库根目录打开 **PowerShell 7（pwsh）**，先下载 LFS 真正的模型文件，再同步整个工作区，包括尚未提交的代码：

```powershell
git lfs pull

# 首次使用时先手动 SSH，完成主机指纹确认，并确认密钥能登录
ssh YOUR_SSH_ALIAS
# 登录后退出，再执行本地同步
pwsh -File scripts/paint/sync_to_server.ps1 `
  -SshTarget YOUR_SSH_ALIAS `
  -RemoteDirectory /workspace/Go2_PAINT_Reproduction

# 如果使用 user@host 和自定义端口/密钥
pwsh -File scripts/paint/sync_to_server.ps1 `
  -SshTarget user@host -Port 2222 `
  -IdentityFile C:/Users/Rog/.ssh/id_ed25519 `
  -RemoteDirectory /workspace/Go2_PAINT_Reproduction
```

`YOUR_SSH_ALIAS` 可以替换成本机已配置的 `RunPod-Proxy`，仅在它确实对应目标服务器时使用。
同步通过 tar + SSH 标准输入传输，不要求服务器支持 scp/SFTP。服务器必须有 Bash/兼容 shell 与 tar。
排除 `.git`、训练日志、缓存、虚拟环境、常见凭据目录和 `.env`；保留 USD 等本地实体资产。
同步会覆盖同路径代码文件，**不会删除服务器多出来的文件**，也不会移除服务器已有日志/权重。首次建议使用独立目录。
此脚本不发送本地训练 checkpoint。续训时在服务器使用已有 checkpoint，或自行上传指定文件。
不要向正在训练的同一个工作目录同步代码；需要同时训练时使用另一个项目目录。

## 2. 服务器环境与初始验证

登录服务器，进入项目目录，激活**已经安装好 Isaac Lab 的 Python 环境**：

```bash
cd /workspace/Go2_PAINT_Reproduction
conda activate YOUR_ISAACLAB_ENV
which python
nvidia-smi

# 选择一个可见 GPU；如果要使用物理 GPU 1，改为 CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=0

# 检查依赖/CUDA/LFS，并安装项目本身，不升级/补装外部依赖
python scripts/paint/server_train.py prepare

# 依次执行：CPU 接口测试、RL 配置兼容检查、Flat smoke、Flat 64 环境 PPO 2 iterations、
# Rough smoke、Rough 64 环境 PPO 2 iterations
python scripts/paint/server_train.py smoke \
  --smoke-envs 16 --smoke-steps 100 --ppo-envs 64 --ppo-iterations 2
```

`prepare` 与后续各阶段检查 Python 3.11、关键包、CUDA、RSL-RL 与 Isaac Lab 的 API 配对以及 GO2-PIPER USD 是否还是 LFS 指针。
`prepare` 仅执行 `pip install --no-deps -e source/LeggedManip_Lab`；缺失的 TensorBoard/Isaac/导出依赖需要在已有环境内补齐。
任何子进程失败均停止当前流程并返回非零状态；smoke 通过表示接口和短 PPO 能运行，不表示行走性能合格。
默认 4096 个环境是训练起点，不保证所有 GPU 都能承载；显存不足时先改为 1024 或 2048。

## 3. 大规模分阶段训练

Flat 训练：

```bash
python scripts/paint/server_train.py flat \
  --num-envs 4096 --flat-iterations 5000 --seed 42
```

查看 `logs/rsl_rl/go2_piper_paint_loco/<run>/` 中实际生成的 `model_*.pt`。检查训练曲线和行走效果后，将选定的 Flat checkpoint 用于 Rough：

```bash
python scripts/paint/server_train.py rough \
  --num-envs 4096 --rough-iterations 10000 --seed 42 \
  --checkpoint /workspace/Go2_PAINT_Reproduction/logs/rsl_rl/go2_piper_paint_loco/ACTUAL_RUN/model_ACTUAL_ITER.pt
```

恢复同一 Rough 训练也是同一命令，将 checkpoint 换成 Rough 的权重。恢复 Flat 则用 `flat --checkpoint ...`。
`--checkpoint` 必须指向现有 **PAINT LL 的 12 动作权重**，不能用原 Flat/WBC 的 18 动作权重。
脚本复制指定文件到对应实验目录下唯一的 `resume_*` 暂存目录，避免不同 Isaac Lab 版本对绝对 `--load_run` 的处理差异。
保留原 checkpoint，不依赖“最近一次实验”匹配；该副本会额外占用一份 checkpoint 的磁盘空间。
`--*-iterations` 传给现有 runner，表示本次调用要执行的 learning iterations；恢复时不是总迭代数上限。
恢复会载入 checkpoint 中的优化器等状态，但环境/地形/速度课程不会从 checkpoint 恢复。

## 4. 一次执行完整流程与后台日志

完整流程会顺序执行 prepare → smoke 与短 PPO → Flat 正式训练 → 将**本次 Flat run 中编号最大的 checkpoint**交给 Rough。
该自动交接只确认训练程序成功结束且有权重，**不会自动判断 Flat 的行走质量**；需要质量验收时使用上面的分阶段命令。

先预览所有命令（不安装、不训练、不创建日志）：

```bash
python scripts/paint/server_train.py pipeline --dry-run
```

后台运行完整流程：

```bash
mkdir -p logs/paint_launcher
STAMP=$(date +%Y%m%d_%H%M%S)
nohup python -u scripts/paint/server_train.py pipeline \
  --num-envs 4096 --flat-iterations 5000 --rough-iterations 10000 --seed 42 \
  > "logs/paint_launcher/pipeline_${STAMP}.log" 2>&1 < /dev/null &
PAINT_PID=$!
echo "$PAINT_PID" > "logs/paint_launcher/pipeline_${STAMP}.pid"
echo "PID=$PAINT_PID LOG=logs/paint_launcher/pipeline_${STAMP}.log"
tail -f "logs/paint_launcher/pipeline_${STAMP}.log"
```

`nohup` 继续使用当前已激活环境和 `CUDA_VISIBLE_DEVICES`。关闭 tail 用 Ctrl+C，不会停止后台训练。
每次 launcher 都有唯一的 `logs/paint_launcher/<tag>/`，包含环境摘要及各阶段的独立日志。
短 PPO 日志写入 `logs/rsl_rl/paint_smoke_flat/` 和 `paint_smoke_rough/`，不会成为正式实验的自动恢复来源。
正式策略日志与 checkpoint 位于 `logs/rsl_rl/go2_piper_paint_loco/` 和 `go2_piper_paint_loco_rough/`。
`Completed stage: pipeline` 表示整个流程正常返回；遇到 `ERROR:` 时查看对应阶段日志。

查看训练曲线：

```bash
tensorboard --logdir logs/rsl_rl --host 127.0.0.1 --port 6006
```

本机另开终端做 SSH 端口转发：`ssh -L 6006:127.0.0.1:6006 YOUR_SSH_ALIAS`，浏览器打开 `http://localhost:6006`。

## 5. 权重评估与导出

```bash
python scripts/rsl_rl/play.py --task GO2-PIPER-PAINT-Loco-Rough-Play \
  --checkpoint /absolute/path/to/actual/model.pt \
  --num_envs 32 --headless --video --video_length 1000
```

此命令调用现有脚本，输出同一 run 下的 `exported/policy.pt`、`policy.onnx` 及录像；需要对应导出和渲染依赖。
实际观测与负载近似的说明见 [底层改造说明](PAINT_LOCOMOTION_CN.md)。
