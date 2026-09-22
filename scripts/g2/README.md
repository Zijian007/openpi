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
# 采数机 / wholebody：lerobot_format: v2.1（或 both）
rsync -avP .../recorded/vr/lerobot_v21/ \
  ~/work/openpi/data/g2_vr_lerobot_v21/
```

`HF_LEROBOT_HOME` 默认 = `~/work/openpi/data`（见 `_env.sh`）。  
需含 `meta/episodes_stats.jsonl`（OpenPI 的 LeRobot 0.1 对 v2.1 必需）。

## 训练（`scripts/g2/train/`）

```bash
# 推荐：norm stats + 训练一条龙
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh --smoke   # LoRA 短跑
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh           # full（单 4090 请 CONFIG_NAME=low_mem）
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_pipeline.sh --skip-norm

# 也可拆开
bash scripts/g2/train/compute_norm_stats.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_smoke.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train/train_full.sh
```

| Config | 说明 |
|--------|------|
| `pi05_g2_vr` | 全参微调；单 4090 通常 OOM，需多卡 FSDP |
| `pi05_g2_vr_low_mem` | LoRA；smoke / 24GB |

Delta：默认 `observation.ee` 作 state，`DeltaActions(9,-1,9,-1)`（夹爪绝对）。绝对 pose 训法：config 里 `use_delta_actions=False`。

## 部署（`scripts/g2/deploy/`）

```bash
# OpenPI 推理服务（≠ G2_pi policy_server）
# 与 LeRobot :8000 并存时用 --port 8001
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/deploy/serve_policy.sh \
  --config-name pi05_g2_vr_low_mem \
  --policy-dir checkpoints/pi05_g2_vr_low_mem/<exp>/<step>
# 省略 --policy-dir 时自动选该 config 下最新 step

# GPU 本机假观测 smoke（须先起 serve_policy）
bash scripts/g2/deploy/smoke_infer_client.sh
bash scripts/g2/deploy/smoke_infer_client.sh --port 8001 --zeros --num-infer 1
```

| 脚本 | 作用 |
|------|------|
| `deploy/serve_policy.sh` | WebSocket 推理服务（默认 `:8000`） |
| `deploy/smoke_infer_client.sh` | 假观测 round-trip 验收（shape / NaN / 耗时） |

## 与 G2_pi / LeRobot 真机栈边界

- 环境 / cache / W&B（`G2_openpi`）全部分开  
- checkpoint **不能**互通  
- `deploy/serve_policy` ≠ `G2_pi` `policy_server.sh`  
- 域控现有 `lerobot_pi05_client` **只服务 LeRobot 线**；OpenPI 线见 [openpi-pi05-deploy-g2.md](../../../g2_wzj_docs/ops/ml/openpi-pi05-deploy-g2.md)
