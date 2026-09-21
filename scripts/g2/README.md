# G2 wholebody × OpenPI π₀.₅（并行于 `~/work/G2_pi`）

与 **LeRobot v3 + PEFT** 路线独立：本目录吃 **LeRobot v2.1**，JAX/OpenPI 训练。  
总文档：[g2_wzj_docs/ops/ml/openpi-pi05-g2.md](../../../g2_wzj_docs/ops/ml/openpi-pi05-g2.md)。

## 布局

```text
~/work/openpi/                 # 本仓（从 ~/openpi 迁入；~/openpi 为兼容软链）
  scripts/g2/                  # G2 入口（对齐 G2_pi/scripts）
  data/g2_vr_lerobot_v21/      # HF_LEROBOT_HOME/<repo_id>（仅 v2.1）
  src/openpi/policies/g2_policy.py
  … training/config.py         # LeRobotG2DataConfig, pi05_g2_vr, pi05_g2_vr_low_mem
  checkpoints/                 # 训练输出

~/work/G2_pi/                  # 另一条：v3 + PEFT（勿共用 .venv / data）
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

## 训练 / 服务

```bash
# 推荐：norm stats + 训练一条龙
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train_pipeline.sh --smoke   # LoRA 短跑
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train_pipeline.sh           # full
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train_pipeline.sh --skip-norm  # 已有 assets

# 也可拆开
bash scripts/g2/compute_norm_stats.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train_smoke.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/train_full.sh

bash scripts/g2/serve_policy.sh --policy-dir checkpoints/pi05_g2_vr/<exp>/<step>
```

| Config | 说明 |
|--------|------|
| `pi05_g2_vr` | 全参微调起点；batch 默认 8 |
| `pi05_g2_vr_low_mem` | LoRA；smoke / 24GB |

Delta：默认 `observation.ee` 作 state，`DeltaActions(9,-1,9,-1)`（夹爪绝对）。绝对 pose 训法：config 里 `use_delta_actions=False`。

## 与 G2_pi 边界

- 环境 / cache / W&B project（`G2_openpi`）全部分开  
- checkpoint **不能**互通  
- `serve_policy` ≠ `G2_pi` `policy_server.sh`；域控 `pi05_client` 需另适配  
