# G2 训练

[返回入口](../README.md) · [Policy 配置](policy.md) · [字段速查](../../../../g2-policy-protocol/docs/policy-config.md)

从仓库根目录运行。默认读 [`configs/train.yaml`](../configs/train.yaml)。

## 最短路径

```bash
bash scripts/g2/bootstrap.sh --download-base
# 编辑 configs/train.yaml：data.root / data.repo_id / experiment.name / runtime.gpus
.venv/bin/python scripts/g2/train/train_pipeline.py --dry-run
bash scripts/g2/train/train_smoke.sh
bash scripts/g2/train/train_pipeline.sh
```

产物在 `checkpoints/<模型配置名>/<experiment.name>/`。相对路径以仓库根目录为准。

## 数据

OpenPI 读 **LeRobot v2.1**。目录是 `data.root / data.repo_id`，不是写死的文件夹名。需要 `meta/episodes_stats.jsonl`。

```bash
# 采数机导出 v2.1 后同步到训练机
rsync -avP .../recorded/vr/lerobot_v21/ ~/openpi/data/<data.repo_id>/
```

`HF_LEROBOT_HOME` 默认等于 YAML 的 `data.root`（见 `scripts/g2/_env.sh`）。导出列和段边界见[公共契约](../../../../g2-policy-protocol/docs/development.md#职责划分)。

## 改配置

### 多数据集

示例：[`configs/multi_dataset.yaml`](../configs/multi_dataset.yaml)，继承默认训练设置，只需修改路径、实验名和 GPU。每个 `data.sources[].path` 指向一份 v2.1 数据集；相对路径基于 `data.root`，默认 `./data`。不能同时写 `data.repo_id`；多源时拒绝 CLI 的 `--repo-id/--data-root`。

```bash
bash scripts/g2/train/train_pipeline.sh scripts/g2/configs/multi_dataset.yaml --dry-run
bash scripts/g2/train/train_pipeline.sh scripts/g2/configs/multi_dataset.yaml --smoke
```

多源 dry-run 读取本地元数据和文件信息，打印每个源的实际采样占比，不解码视频。全部不写 weight 时自然混合；全部填写时按完整 episode 重复近似加权。`data.image_size` 可省略，填写时 Pi05 仅接受 `[224, 224]`。

多源统计量写入 `mix-<指纹>` 资产目录，按展开分布计算，norm 阶段不解码视频。checkpoint 保存 asset_id，部署无需原始数据。续训校验混合身份，复用 checkpoint 统计量，但不保证数据顺序接续。

需安装 `g2-policy-protocol==0.7.0` 并同步升级训练和部署代码。旧单源 checkpoint 的 asset_id 仍为原 repo_id。完整规则见[多数据集与图像尺寸](../../../../g2-policy-protocol/docs/policy-config.md#多数据集与图像尺寸)。

先改这几项就能跑：`experiment.name`、`experiment.arm`、`data.root`、`data.repo_id`、`runtime.gpus`。
`policy` 默认已是 TeaPi：右臂实测绝对位姿，学习指令相对当前实测的分量差。改作差/坐标/历史见字段速查；自定义 YAML 只写要覆盖的叶子，其余继承 `configs/train.yaml`。本仓库的训练入口使用 `schema_version: 7`、`backend: openpi`；版本、字段翻译与 checkpoint 快照由 `openpi.adapters.g2.training_config` 维护。配置版本独立于线上 `g2.policy.v3`；共享包只约束 `policy` 的状态、动作和时序语义。

| 字段 | 含义 |
|---|---|
| `training.batch_size` | 全局 batch，沿用 OpenPI 语义 |
| `training.steps` / `training.learning_rate` | `num_train_steps` / 余弦调度器的 `peak_lr` |
| `training.num_workers` / `training.seed` | 数据加载进程数 / 随机种子 |
| `training.log_interval` / `training.save_interval` | 日志 / checkpoint 保存间隔 |
| `runtime.resume` | 恢复当前实验最新完整 checkpoint |
| `backend_options.openpi.finetuning.<模块>.mode` | `frozen` / `lora` / `full`，三个模块互不覆盖 |
| `backend_options.openpi.run.smoke` | 短跑：batch=1、20 步、独立实验名、默认关 W&B |
| `policy.timing.action_horizon` | 预测帧数，默认 50，必须写在 YAML 里 |
| `backend_options.openpi.model.discrete_state_input` | 状态是否编进 prompt；宽于 32 维必须 true |
| `backend_options.openpi.normalization.mode` | 新训练 `auto` / `force` / `skip`；续训读 checkpoint |
| `backend_options.openpi.checkpoints.overwrite` | `true` 会删掉同名实验的全部权重 |
| `runtime.gpus` | 写入 `CUDA_VISIBLE_DEVICES`；已有环境变量优先 |

```bash
# 只覆盖部分字段的实验 YAML
bash scripts/g2/train/train_pipeline.sh scripts/g2/configs/my_experiment.yaml
bash scripts/g2/train/train_pipeline.sh --config scripts/g2/configs/my_experiment.yaml --batch-size 2
```

合并顺序：`configs/train.yaml` → 自定义 YAML → CLI。`--` 之后只接受 optimizer/EMA，例如 `-- --optimizer.clip-gradient-norm=0.5`。

## 预览、短跑、正式训练

```bash
# 打印生效配置；不读数据集、不训练。续训时会读 checkpoint 元数据
.venv/bin/python scripts/g2/train/train_pipeline.py --dry-run

bash scripts/g2/train/train_smoke.sh
bash scripts/g2/train/train_pipeline.sh
CUDA_VISIBLE_DEVICES=0,1 bash scripts/g2/train/train_full.sh   # YAML 先设 backend_options.openpi.training.fsdp_devices: 2
```

`train_pipeline.sh` 按 YAML 训练；`train_smoke.sh` 打开短跑；`train_full.sh` 把 vision/language/action_expert 全设为 full；`compute_norm_stats.sh` 只算统计量。四个入口都接受 `--config PATH`。新实验默认不覆盖已有目录。

启用 W&B 在线记录时，入口在计算 norm stats、创建 checkpoint 目录和加载模型前，先用只读账号查询检查服务连接与登录，整个检查最多等待 20 秒，不创建测试 run。失败立即退出，提示检查当前终端（尤其 tmux）的代理变量和 W&B 登录；通过后才继续。`--dry-run`、`--norm-only`、关闭 W&B 或使用离线模式时跳过联网检查。

## 续训

目标步数是总数，不是再加多少步。

```bash
bash scripts/g2/train/train_pipeline.sh --resume --num-train-steps 60000 --dry-run
bash scripts/g2/train/train_pipeline.sh --resume --num-train-steps 60000
```

必须找到完整 checkpoint，不会偷偷改成从头训练。配置和统计量只读该 step 的 `assets/`，不读当前模板。可改总步数、worker、日志/保存间隔、GPU、FSDP。不可改模型、LoRA、policy、数据、batch、seed、学习率。`decay_steps: null` 时按保存时的周期恢复，拉长总步数不会自动拉长衰减。`--norm-only` 不能和 resume 一起用。

## 统计量

归一化和训练用同一份解析结果。`auto` 只在统计文件和配置签名都匹配时复用。数据变了路径没变，用 `--force-norm`。每份 checkpoint 的 `assets/g2_train_config.yaml` 保存实际生效配置；部署从这里恢复，不受后来改 YAML 的影响。

## 微调

默认 vision/language LoRA、动作专家全参。至少一个模块必须可训练。

```yaml
backend_options:
  openpi:
    finetuning:
      vision: {mode: lora, lora: {rank: 16, alpha: 16.0, targets: [attention], rslora: false}}
      language: {mode: lora, lora: {rank: 16, alpha: 16.0, targets: [attention, ffn], rslora: false}}
      action_expert: {mode: full, lora: {rank: 32, alpha: 32.0, targets: [attention, ffn], rslora: false}}
```

视觉 LoRA 只支持 `attention`；语言和动作专家可选 `attention` / `ffn`。`rslora: true` 时缩放为 alpha/sqrt(rank)。语言或动作专家用 LoRA 时关闭 EMA。改 LoRA 结构要换 `experiment.name`，不能拿旧优化器续训。当前默认会存到 `checkpoints/pi05_g2_vr_vlm_lora_action_full_right/<name>/`。

配置名保留已有预设名称；未命中预设时使用 `pi05_g2_vr_vision_<模式>_language_<模式>_action_<模式>`，单臂追加 `_left` 或 `_right`。例如视觉 LoRA、语言冻结、动作专家全参的右臂配置名为 `pi05_g2_vr_vision_lora_language_frozen_action_full_right`。checkpoint 和共享 assets 使用同一配置名，`experiment.name` 仅用于 checkpoint 的下一层目录；混合数据的 assets 子目录仍为 `mix-<指纹>`。已有目录不会自动重命名，旧自动拼接名称的实验续训前需对齐目录名称。

## 单臂

```bash
.venv/bin/python scripts/g2/train/train_pipeline.py --arm right --no-resume --exp-name right_arm_run
```

换臂会换配置名，必须重算统计量；原始双臂数据不用重导。缺的腕部相机填黑图并 mask。部署的 `--policy.dir` 必须指向这次实验的 step，不要复用双臂路径。服务端用双臂实测 EE 把单臂预测补回 20 维；域控按握手的 `arm` 只采对应相机、只向该侧发令。

## 常见问题

- `outputs`/`checkpoints` 已存在且未开 overwrite：换 `experiment.name`，或确认你在续训。
- 改了 delta/method/坐标/历史：新建实验并重算统计量。
- 自定义 YAML 在默认 `action.pose` 上只写 `delta: false`：会清掉继承的 method 和 reference。
- 宽状态却没开 `discrete_state_input`：配置阶段直接失败。


## YAML 归属与 checkpoint

外层字段为 `schema_version`、`backend`、`experiment`、`data`、`policy`、`model`、
`training`、`runtime`、`wandb`、`backend_options`。`model.type` 当前仅接受 `pi05`，
`model.pretrained_path` 映射到参数加载路径。OpenPI 专属的微调、调度器、FSDP、归一化、
显存和目录选项放在 `backend_options.openpi`，不要在那里重复声明入口字段。

`wandb.mode` 为 `auto` / `on` / `off`；`auto` 在 smoke 时关闭。
YAML 只将 true/false 作为布尔值，接受 `1e-5` 等科学计数法，并拒绝重复 key。
CLI `--steps`、`--batch-size`、`--num-workers`、`--log-interval`、`--save-interval`、
`--learning-rate` 优先于 YAML。`runtime.resume` 是布尔值，恢复当前实验的最新完整 checkpoint。

checkpoint 内的 `g2_train_config.yaml` 保存自包含的 v7 配置；恢复时不合并当前训练模板。
训练字段的归属调整沿用现有 YAML 与 checkpoint 格式，不新增迁移或兼容路径。
同名参数不保证与其他框架训练等价，尤其 LeRobot 的 batch_size 是每进程 batch。
