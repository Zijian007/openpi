# G2 wholebody × OpenPI π₀.₅（并行于 `~/work/G2_pi`）

与 **LeRobot v3 + PEFT** 路线独立：本目录吃 **LeRobot v2.1**，JAX/OpenPI 训练与部署。

| | OpenPI 线（本仓） | LeRobot 线 |
|--|------------------|------------|
| 训练 | [openpi-pi05-train-g2.md](../../../g2_wzj_docs/ops/ml/openpi-pi05-train-g2.md) | [lerobot-pi05-train-g2.md](../../../g2_wzj_docs/ops/ml/lerobot-pi05-train-g2.md) |
| 部署 | [openpi-pi05-deploy-g2.md](../../../g2_wzj_docs/ops/ml/openpi-pi05-deploy-g2.md) | [lerobot-pi05-deploy-g2.md](../../../g2_wzj_docs/ops/ml/lerobot-pi05-deploy-g2.md) |

## 布局

```text
~/openpi/
  scripts/g2/
    _env.sh / bootstrap.sh / README.md   # 共享
    train/                               # norm + 训练
    deploy/                              # serve_policy + smoke client
  data/g2_vr_lerobot_v21/                # HF_LEROBOT_HOME/<repo_id>（仅 v2.1）
  src/openpi/policies/g2_policy.py
  checkpoints/

~/work/G2_pi/                            # 另一条：v3 + PEFT（勿共用 .venv / data）
```

## 一次安装

```bash
cd ~/openpi
bash scripts/g2/bootstrap.sh                 # uv sync
bash scripts/g2/bootstrap.sh --download-base # + 拉 pi05_base（需代理时常开）
```

G2 默认跳过 ALOHA / LIBERO 子模块；需要这些示例时加 `--with-submodules`。

## 数据

```bash
# 采数机 / wholebody：launcher.lerobot.format: v2.1（或 both）
rsync -avP .../recorded/vr/lerobot_v21/ \
  ~/openpi/data/g2_vr_lerobot_v21/
```

`HF_LEROBOT_HOME` 默认 = `~/openpi/data`（见 `_env.sh`）。
需含 `meta/episodes_stats.jsonl`（OpenPI 的 LeRobot 0.1 对 v2.1 必需）。

## 训练（`scripts/g2/train/`）

日常训练参数统一放在 **[`train/config.yaml`](train/config.yaml)**。`train_pipeline.sh` 显式指定这份 YAML 并启动项目虚拟环境，
不再透传命令行参数。配置包括实验名、数据集、选臂、相机、动作表示、模型预测长度、初始权重、
学习率、batch、训练步数、checkpoint 和 W&B。

```bash
# 预览完整生效配置；无需数据集，不下载、不启动训练
.venv/bin/python scripts/g2/train/train_pipeline.py --config scripts/g2/train/config.yaml --dry-run

# 修改 train/config.yaml 后运行：norm → train
bash scripts/g2/train/train_pipeline.sh

# 独立实验 YAML 可以只写需要覆盖的字段；其余继承 train/config.yaml
.venv/bin/python scripts/g2/train/train_pipeline.py --config scripts/g2/train/my_experiment.yaml

# CLI 临时覆盖 YAML
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/g2/train/train_pipeline.py --batch-size 2 --exp-name test_run

# 短跑：batch=1、20 步、关闭自动 W&B，自动生成独立实验名
bash scripts/g2/train/train_smoke.sh

# 从同一实验的最新 checkpoint 续训；总步数是目标值，不是追加值
.venv/bin/python scripts/g2/train/train_pipeline.py --resume --num-train-steps 60000

# 只计算统计量 / 强制重算统计量
bash scripts/g2/train/compute_norm_stats.sh
.venv/bin/python scripts/g2/train/train_pipeline.py --force-norm

# 全参配置，多卡时需显式指定 FSDP 分片数
# 先在 YAML 设置 training.fsdp_devices: 2
CUDA_VISIBLE_DEVICES=0,1 bash scripts/g2/train/train_full.sh
```

四个 Shell 训练入口均不接收额外参数，固定读取 `train/config.yaml`；临时指定其他 YAML、预览或覆盖参数时，
直接调用 `.venv/bin/python scripts/g2/train/train_pipeline.py`。
Python 入口的覆盖顺序：`config.yaml` → 自定义 YAML → smoke 短跑默认值 → CLI。
`CUDA_VISIBLE_DEVICES`、`XLA_PYTHON_CLIENT_MEM_FRACTION` 和 `ARM` 环境变量仍兼容；选臂 CLI 优先于 `ARM`。
数据根目录由 YAML 的 `data.root` 决定；YAML 和 CLI 中的相对路径均相对于仓库根目录。
新实验默认不覆盖已有目录；`checkpoints.overwrite: true` 会删除同名实验的全部权重。

常用字段：

| YAML 字段 | 含义 |
|-----------|------|
| `experiment.preset` | `low_mem` LoRA / `full_ft` 全参 / `smoke` 短跑 |
| `experiment.arm` | `both` / `left` / `right` |
| `data.root` + `data.repo_id` | 本地 LeRobot v2.1 数据集位置 |
| `data.cameras` | `null` 自动选择头部和所选臂腕部；也可显式写 `[head_color]` 等子集，必须含头部 |
| `data.action_space` | 当前仅支持 `pose`；填 `joint` 会明确报错 |
| `data.action_mode` | `relative`：state.pose 作参考，xyz/rot6d 相对，夹爪绝对；`absolute`：state.joint 作状态、action.pose 绝对动作，仅支持双臂 |
| `model.action_horizon` | 每个样本监督的未来动作步数，默认 50 |
| `model.discrete_state_input` | 是否将当前状态编码进 prompt |
| `training.decay_steps` | `null` 时跟随总训练步数；smoke 保留正常预热周期 |
| `normalization.mode` | `auto` / `force` / `skip` |

归一化与训练都从同一份解析结果构建 `TrainConfig`，因此动作表示、选臂、相机、预测长度、数据路径、
batch 和 assets 路径一致。`auto` 仅在统计文件及其配置签名匹配时复用；旧统计文件没有签名时会重新计算。
数据内容发生变化但路径不变时，用 `--force-norm` 重算。
`--` 后只保留 optimizer / EMA 的高级覆盖，例如 `-- --optimizer.clip-gradient-norm=0.5`，
结构和学习率参数请直接配置 YAML，避免隐藏的第二套覆盖。

实验根目录和每份 checkpoint 的 `assets/` 都保存 `g2_train_config.yaml`，包含实际生效的配置。
部署加载带此文件的新 checkpoint 时，会自动恢复训练时的模型、动作和相机设置，不受后来修改的 YAML 影响；
老 checkpoint 继续使用原来的具名 TrainConfig。续训允许修改总步数和学习率，但不允许改变数据/模型语义。

## 部署（`scripts/g2/deploy/`）

`serve_policy.sh` 只选 checkpoint，然后调用官方 `scripts/serve_policy.py`。默认 **warmup**：监听前假图 infer 两次（付 JAX JIT）；`--no-warmup` 可关。`JAX_COMPILATION_CACHE_DIR` 在 `_env.sh` 里指到 `$OPENPI_DATA_HOME/jax_compile_cache`。

```bash
# ≠ G2_pi policy_server。与 LeRobot :8000 并存时加 --port 8001
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/deploy/serve_policy.sh \
  --config-name pi05_g2_vr_low_mem \
  --policy-dir checkpoints/pi05_g2_vr_low_mem/<exp>/<step>
# 省略 --policy-dir → 最近写入的实验里，数值最大的 step（非跨实验比 step，也非字典序）
# healthz 在 warmup 结束后才 OK；日志应有 warmup step=0 infer_ms≈10s+，step=1 ≈100ms

bash scripts/g2/deploy/smoke_infer_client.sh --port 8000   # 端口对齐 serve；首包应已是稳态
```

| 脚本 | 作用 |
|------|------|
| `deploy/serve_policy.sh` | 选 ckpt + 官方 `serve_policy.py`（默认 `:8000`，默认 warmup） |
| `deploy/smoke_infer_client.sh` | 假观测 round-trip |

域控：`start_openpi_pi05_infer.sh`（`openpi_pi05_client`）。详见 [openpi-pi05-deploy-g2.md](../../../g2_wzj_docs/ops/ml/openpi-pi05-deploy-g2.md)。

## 与 G2_pi / LeRobot 边界

- 环境 / cache / W&B（`G2_openpi`）分开；ckpt **不互通**
- `serve_policy` ≠ `G2_pi` `policy_server`
- 域控 `lerobot_pi05_client` 只服务 LeRobot 线；OpenPI 用 `openpi_pi05_client`
## 单臂 π₀.₅

使用 `ARM=both|left|right`，默认 `both`。右臂新实验：

```bash
ARM=right .venv/bin/python scripts/g2/train/train_pipeline.py --no-resume --exp-name right_arm_run
```

也可以使用 `--arm right`。训练包装器自动选择 `pi05_g2_vr_low_mem_right`（全参则为 `pi05_g2_vr_right`）；左臂对应 `_left`。独立配置名隔离 norm stats 和 checkpoint，切换选臂后需要重新计算统计量。原始双臂数据无需重新导出。

单臂模型使用头部与对应腕部相机，选取对应的 10 维 EE 状态和动作。xyz/rot6d 以 chunk 起点实测 EE 为参考，夹爪保持绝对值。模型内部仍补零到 32 维；另一腕相机使用黑色占位及 false mask，不读取或使用该相机内容。

部署时配置必须与训练一致：

```bash
ARM=right bash scripts/g2/deploy/serve_policy.sh \
  --policy-dir checkpoints/pi05_g2_vr_low_mem_right/right_arm_run/<step>
```

服务端根据本次双臂实测 EE 将单臂预测映射回 20 维绝对动作，RTC 前缀也只对所选臂进行归一化和引导。须同步更新 wholebody 执行端：它读取 `arm` metadata，仅采集头部与对应腕部相机，并不向未选中手臂和夹爪发送 policy 指令。部署脚本中已有的双臂 checkpoint 路径不能用于单臂配置，须显式指定本次实验路径。

## G2 通信协议

G2 部署使用独立 `g2-policy-protocol==0.4.1` 的 `g2.policy.v2`，与 wholebody/G2_pi 共用单份 JSON Schema。
`uv sync` 从相邻 `../g2-policy-protocol` 安装包；仅部署 wheel 时，将 `tool.uv.sources.g2-policy-protocol.path` 改为本地 wheel 文件路径，再 `uv lock && uv sync`。
`serve_policy.py` 对 robot=g2 的 metadata 选择 `G2PolicyServer`；非 G2 模型保留原生 OpenPI server。
G2 server 在入口把标准观测映射成模型内部的 `observation/*`，JPEG 解码、RTC、单臂扩展和图像处理保持在适配层。
默认 MessagePack，client 连接 `?encoding=json` 时整个会话使用 JSON text frames。
域控/GPU 需同步更新并重启；本次不兼容旧 `openpi.websocket.v1` G2 客户端。GPU-local smoke 脚本已同步使用新协议。
