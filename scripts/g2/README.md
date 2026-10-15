# G2 wholebody × OpenPI π₀.₅（并行于 `~/work/G2_pi`）

与 **LeRobot v3 + PEFT** 路线独立：本目录吃 **LeRobot v2.1**，JAX/OpenPI 训练与部署。

| | OpenPI 线（本仓） | LeRobot 线 |
|--|------------------|------------|
| 训练 | [openpi-pi05-train-g2.md](../../../g2_wzj_docs/ops/ml/openpi-pi05-train-g2.md) | [lerobot-pi05-train-g2.md](../../../g2_wzj_docs/ops/ml/lerobot-pi05-train-g2.md) |
| 部署 | [openpi-pi05-deploy-g2.md](../../../g2_wzj_docs/ops/ml/openpi-pi05-deploy-g2.md) | [lerobot-pi05-deploy-g2.md](../../../g2_wzj_docs/ops/ml/lerobot-pi05-deploy-g2.md) |

## 布局

```text
~/work/openpi/
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
cd ~/work/openpi
bash scripts/g2/bootstrap.sh                 # uv sync
bash scripts/g2/bootstrap.sh --download-base # + 拉 pi05_base（需代理时常开）
```

## 数据

```bash
# 采数机 / wholebody：launcher.lerobot.format: v2.1（或 both）
rsync -avP .../recorded/vr/lerobot_v21/ \
  ~/work/openpi/data/g2_vr_lerobot_v21/
```

`HF_LEROBOT_HOME` 默认 = `~/work/openpi/data`（见 `_env.sh`）。  
需含 `meta/episodes_stats.jsonl`（OpenPI 的 LeRobot 0.1 对 v2.1 必需）。

## 训练（`scripts/g2/train/`）

主入口 **`train_pipeline.sh`**（norm → train）。其它脚本是薄包装。

```bash
# 单卡 LoRA（默认 preset=low_mem）
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh

# 短跑验证
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh --smoke

# 全参微调（多卡 / FSDP）
CUDA_VISIBLE_DEVICES=0,1 bash scripts/g2/train/train_pipeline.sh --preset full_ft

# 只看最终 uv 命令
bash scripts/g2/train/train_pipeline.sh --smoke --dry-run

# 高级 tyro 覆盖（不改 bash）
bash scripts/g2/train/train_pipeline.sh -- --optimizer.peak-lr=1e-5

# 拆开跑
bash scripts/g2/train/compute_norm_stats.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_smoke.sh   # → --smoke
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_full.sh  # → --preset full_ft
```

| Config | 说明 |
|--------|------|
| `pi05_g2_vr` | 全参微调；单 4090 通常 OOM，需多卡 FSDP |
| `pi05_g2_vr_low_mem` | LoRA；smoke / 24GB |

Delta：默认 `state.pose` 作 `observation/state`，`actions`←`action.pose`，delta mask 来自 `g2_lerobot_contract.G2.pose`（夹爪绝对）。绝对 pose 训法：config 里 `use_delta_actions=False`（此时 state←`state.joint`）。列名在 `src/openpi/g2_lerobot_contract.py`，与 wholebody 那份文件保持一致。

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
ARM=right bash scripts/g2/train/train_pipeline.sh --no-resume --exp-name right_arm_run
```

也可以使用 `--arm right`。训练包装器自动选择 `pi05_g2_vr_low_mem_right`（全参则为 `pi05_g2_vr_right`）；左臂对应 `_left`。独立配置名隔离 norm stats 和 checkpoint，切换选臂后需要重新计算统计量。原始双臂数据无需重新导出。

单臂模型使用头部与对应腕部相机，选取对应的 10 维 EE 状态和动作。xyz/rot6d 以 chunk 起点实测 EE 为参考，夹爪保持绝对值。模型内部仍补零到 32 维；另一腕相机使用黑色占位及 false mask，不读取或使用该相机内容。

部署时配置必须与训练一致：

```bash
ARM=right bash scripts/g2/deploy/serve_policy.sh \
  --policy-dir checkpoints/pi05_g2_vr_low_mem_right/right_arm_run/<step>
```

服务端根据本次双臂实测 EE 将单臂预测映射回 20 维绝对动作，RTC 前缀也只对所选臂进行归一化和引导。须同步更新 wholebody 执行端：它读取 `arm` metadata，仅采集头部与对应腕部相机，并不向未选中手臂和夹爪发送 policy 指令。部署脚本中已有的双臂 checkpoint 路径不能用于单臂配置，须显式指定本次实验路径。
