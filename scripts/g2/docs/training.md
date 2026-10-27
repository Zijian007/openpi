# G2 训练

[返回入口](../README.md) · [策略配置](policy.md)

## 数据

数据集位置是 `configs/train.yaml` 的 `data.root` + `data.repo_id`，不是写死的目录名。

```bash
# 采数机 / wholebody：launcher.lerobot.format: v2.1（或 both）
rsync -avP .../recorded/vr/lerobot_v21/ \
  ~/openpi/data/<data.repo_id>/
```

`HF_LEROBOT_HOME` 默认 = `~/openpi/data`（见 `scripts/g2/_env.sh`）。
需含 `meta/episodes_stats.jsonl`（OpenPI 的 LeRobot 0.1 对 v2.1 必需）。

## 训练

默认训练参数放在 **[`configs/train.yaml`](../configs/train.yaml)**。`train_pipeline.sh` 启动项目虚拟环境，
支持用 `--config PATH` 选择其他 YAML，并透传 Python 入口的 CLI 参数。v4 配置以 `policy.pose/state/action/timing` 描述策略，完整字段见
[Policy 配置说明](policy.md)。配置包括实验名、数据集、选臂、相机、动作表示、模型预测长度、初始权重、
学习率、batch、训练步数、checkpoint 和 W&B。

```bash
# 预览完整生效配置；无需数据集，不下载、不启动训练
.venv/bin/python scripts/g2/train/train_pipeline.py --config scripts/g2/configs/train.yaml --dry-run

# 修改 configs/train.yaml 后运行：norm → train
bash scripts/g2/train/train_pipeline.sh

# 独立实验 YAML 可以只写需要覆盖的字段；其余继承 configs/train.yaml
bash scripts/g2/train/train_pipeline.sh --config scripts/g2/configs/my_experiment.yaml

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

四个 Shell 训练入口均接受 `--config PATH`，不传时读取 `configs/train.yaml`。
`train_pipeline.sh` 按 YAML 训练；`train_smoke.sh` 打开独立的短跑开关；`train_full.sh` 将三个模块全部设为 full；
`compute_norm_stats.sh` 额外选择 norm-only。显式 CLI 参数排在快捷脚本的选项之后，重复选项以最后一个为准。
例如 `train_smoke.sh --config scripts/g2/configs/my_experiment.yaml --batch-size 2 --dry-run`。
配置合并顺序：`configs/train.yaml` → 自定义 YAML → CLI 选择模块模式和是否短跑 → 短跑默认值 → 显式 batch/步数/实验名覆盖。
也可以直接调用 `.venv/bin/python scripts/g2/train/train_pipeline.py`。
`CUDA_VISIBLE_DEVICES`、`XLA_PYTHON_CLIENT_MEM_FRACTION` 和 `ARM` 环境变量仍兼容；选臂 CLI 优先于 `ARM`。
数据根目录由 YAML 的 `data.root` 决定；YAML 和 CLI 中的相对路径均相对于仓库根目录。
新实验默认不覆盖已有目录；`checkpoints.overwrite: true` 会删除同名实验的全部权重。

常用字段：

| YAML 字段 | 含义 |
|-----------|------|
| `finetuning.vision/language/action_expert` | 各自选择 `frozen` / `lora` / `full`，互不隐式覆盖 |
| `run.smoke` | 独立短跑开关；保留策略和学习率周期，默认 batch=1、20 步、独立实验名、关闭自动 W&B |
| `experiment.arm` | `both` / `left` / `right` |
| `data.root` + `data.repo_id` | 本地 LeRobot v2.1 数据集位置 |
| `data.cameras` | `null` 自动选择头部和所选臂腕部；也可只要腕部。π₀.₅ 缺的相机槽位填黑图并 mask 掉 |
| `policy.pose` | State/Action的EEF部分共用的模型坐标系、TCP标定和旋转表示；磁盘/通信保持base_link / flange / rot6d |
| `policy.state.space` | `eef` / `joint` / `both`，独立于动作表示和动作参考 |
| `policy.state.spatial.mode / method` | absolute保留绝对状态；relative指定分量差、hybrid或SE(3)几何编码；joint仅支持分量差 |
| `policy.state.temporal.anchor` | 所有历史状态共用的参考；State不配置Action的chunk/step参考范围 |
| `policy.action.target.quantity / source` | 控制量与监督来源；如 `eef_pose + command` 读取 LeRobot 的 `action.pose` |
| `policy.action.spatial.mode / method` | absolute直接编码；relative指定component_difference / se3_body / se3_spatial / hybrid空间运算 |
| `policy.action.temporal.anchor / reference_scope` | 起始参考与chunk / step参考范围；absolute时时间参考字段为null；夹爪始终绝对 |
| `policy.timing.action_horizon` | 每个样本监督的动作帧数，默认 50；模型与握手共用这个名字 |
| `policy.timing.execution_steps` | 只在旧快照里出现；提交长度在 wholebody `infer.execution_steps` |
| `model.discrete_state_input` | 是否将当前状态编码进 prompt |
| `model.vision_lora_rank` / `vision_lora_alpha` | SigLIP Q/K/V/O LoRA 的 rank 和缩放；仅 vision=lora 时生效，关闭请使用 vision=frozen/full |
| `training.decay_steps` | `null` 时跟随总训练步数；smoke 保留原实验的预热和衰减周期 |
| `normalization.mode` | `auto` / `force` / `skip` |

归一化与训练都从同一份解析结果构建 `TrainConfig`，因此动作表示、选臂、相机、预测长度、数据路径、
batch 和 assets 路径一致。`auto` 仅在统计文件及其配置签名匹配时复用；旧统计文件没有签名时会重新计算。
数据内容发生变化但路径不变时，用 `--force-norm` 重算。
`--` 后只保留 optimizer / EMA 的高级覆盖，例如 `-- --optimizer.clip-gradient-norm=0.5`，
结构和学习率参数请直接配置 YAML，避免隐藏的第二套覆盖。

实验根目录和每份 checkpoint 的 `assets/` 都保存 `g2_train_config.yaml`，包含实际生效的配置。
部署加载带此文件的新 checkpoint 时，会自动恢复训练时的模型、动作和相机设置，不受后来修改的 YAML 影响；
v1快照自动迁移，保留原来的状态输入。没有YAML快照的老checkpoint继续使用固定默认值的具名TrainConfig；
这些兼容预设不读取当前训练YAML，新实验的数据集等参数仍由所选YAML覆盖。
续训允许修改总步数和学习率，但不允许改变数据、模型、模块冻结方式或 Policy 输入输出语义。
不活跃的视觉 LoRA rank/alpha 不参与续训比较。旧快照按自身的 preset 与视觉 LoRA 字段恢复原策略。

## VLM LoRA + 动作专家全参

在 `configs/train.yaml` 设置：

```yaml
finetuning:
  vision: lora
  language: lora
  action_expert: full
run:
  smoke: false
model:
  vision_lora_rank: 16
  vision_lora_alpha: 16.0
```

此组合使用 `gemma_2b_lora` 和 `gemma_300m`。当前 YAML 在 SigLIP 每层注意力的
Q/K/V/O 投影加入 LoRA，冻结视觉原始权重、FFN、LayerNorm、patch embedding 和输出 head。
语言主干冻结，仅训练其 attention/FFN 的 LoRA；动作专家、动作输入输出投影及时间 MLP 全参训练。
语言 LoRA 使用 OpenPI 的 `gemma_2b_lora` 默认设置（rank=16、alpha=16），视觉 LoRA 由上述字段配置，EMA 关闭。

视觉 LoRA 的 B 矩阵以零初始化，加载基础权重后的初始视觉输出保持不变。
`finetuning.vision: frozen` 可冻结全部视觉参数，`full` 则真正训练全部视觉参数，二者都不创建视觉 LoRA。
语言与动作专家也分别支持 frozen/lora/full。动作专家为 lora/full 时，动作投影和时间 MLP 全参训练；
frozen 时这些层一起冻结。至少一个模块须可训练。
旧快照缺少视觉 LoRA 字段时按 rank=0 恢复，不会读取当前模板来新增适配器；
已有实验切换视觉 LoRA 设置需要新实验名，不能沿用原优化器状态续训。

正常训练继续使用 `bash scripts/g2/train/train_pipeline.sh`。
右臂实验会保存到 `checkpoints/pi05_g2_vr_vlm_lora_action_full_right/<experiment.name>/`，
不会与 `low_mem` 的 checkpoint 混用；从该实验自己的 checkpoint 续训。
部署时将 `configs/deploy.yaml` 的 checkpoint 指向新实验的 step 目录，模型结构会从快照恢复。

`train_smoke.sh` 保留这份配置的视觉/语言 LoRA 和全参动作专家，只缩短训练：

```bash
bash scripts/g2/train/train_smoke.sh --config scripts/g2/configs/train.yaml
# 也可在正常入口加 --smoke，或在 YAML 设置 run.smoke: true。
# --no-smoke 可临时关闭短跑。
```

batch、学习率、训练步数和 GPU 继续由 YAML 决定。动作专家全参训练增加了优化器状态，
实际可用 batch 需按显存验证。

## 单臂训练与部署

使用 `ARM=both|left|right` 或 `--arm` 覆盖 `configs/train.yaml` 的选臂设置。右臂新实验：

```bash
ARM=right .venv/bin/python scripts/g2/train/train_pipeline.py --no-resume --exp-name right_arm_run
```

也可以使用 `--arm right`。模型配置名由模块模式和选臂推导；常用组合保留原 checkpoint 命名，
例如当前组合为 `pi05_g2_vr_vlm_lora_action_full_right`。新增组合也有独立配置名。
独立配置名隔离 norm stats 和 checkpoint，切换选臂后需要重新计算统计量。原始双臂数据无需重新导出。

单臂模型使用头部与对应腕部相机。默认每帧为 10 维 EEF 状态和 10 维 EEF 动作；也支持 joint、联合状态、历史输入、其他旋转表示及控制量。
动作参考独立于模型状态，joint 状态也能使用 EEF delta。夹爪保持绝对值；动作补零到 32 维，
联合/历史状态使用 Pi05 离散 token 表达。另一腕相机使用黑色占位和 false mask。完整组合见 [Policy 配置说明](policy.md)。

部署时配置必须与训练一致：

```bash
bash scripts/g2/deploy/serve_policy.sh \
  --policy.dir checkpoints/pi05_g2_vr_vlm_lora_action_full_right/right_arm_run/<step>
```

服务端根据本次双臂实测 EE 将单臂预测映射回 20 维绝对动作，RTC 前缀也只对所选臂进行归一化和引导。须同步更新 wholebody 执行端：它读取 `arm` metadata，仅采集头部与对应腕部相机，并不向未选中手臂和夹爪发送 policy 指令。部署脚本中已有的双臂 checkpoint 路径不能用于单臂配置，须显式指定本次实验路径。
joint 状态模型在握手中声明 requires_joint_state，客户端需额外发送 16 维 joint_state；
双臂实测 EE 仍用于单臂输出扩展，不会把关节状态当作位姿。

## 配置版本与旧命令

新 YAML 和新 checkpoint 快照使用 schema v4；v1/v2/v3 文件继续兼容，不用重导数据或转换权重。
动作目标、编码模式/方法、参考策略的字段对应见 [Policy 配置说明](policy.md#兼容性统计量与预览)。
旧 `experiment.preset` 会迁移成显式模块模式，旧 `smoke` 预设保留原 LoRA 策略并迁移为短跑开关。
CLI `--preset` 作为兼容入口保留；`--preset smoke` 等同 `--smoke`，不再切换模型。
`--full` / `train_full.sh` 真正将视觉、语言、动作专家全部设为 full。
也可临时使用 `--vision-mode frozen --language-mode lora --action-expert-mode full`。
CLI提供 `--action-quantity`、`--action-source`、`--action-mode`、`--action-spatial`、`--action-anchor`、`--action-reference-scope`。
例如 `--action-source measured --action-spatial se3_body --action-reference-scope step`，
或 `--action-quantity joint_velocity --action-mode absolute`。
显式 `--action-mode relative` 时须同时提供 `--action-spatial`；只传spatial会选择relative模式。
State提供 `--state-space`、`--state-mode`、`--state-spatial`、`--state-anchor`，使用相同的mode/spatial概念。
例如 `--state-mode relative --state-spatial se3_body --state-anchor episode_start`；`--state-mode absolute`清空相对编码。
State历史长度由YAML的 `policy.timing.history_steps` 控制；EEF坐标沿用 `policy.pose`，无需重复填写坐标系。
CLI名称保持不变，mode与spatial参数分别覆盖YAML的 `spatial.mode / method`；anchor和reference-scope覆盖temporal字段。
`--action-method`保留为 `--action-spatial` 的别名；旧 `--action-strategy fixed / previous_target` 对应新的chunk / step。
同时指定新旧参考范围参数时，以 `--action-reference-scope` 为准。
旧 `--action-encoding` / `--action-representation` 保留原候选值作为兼容入口，新的动作参数在其之后覆盖。
