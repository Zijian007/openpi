# G2 训练

[返回入口](../README.md) · [Policy 配置](policy.md) · [字段速查](../../../../g2-policy-protocol/docs/policy-config.md)

从仓库根目录运行。默认读 [`configs/train.yaml`](../configs/train.yaml)。
新服务器先完成[环境安装与验证](environment.md)，包括协议包和本地项目安装。

## 最短路径

```bash
bash scripts/g2/bootstrap.sh --download-base
# 编辑 configs/train.yaml：data.root / data.repo_id / experiment.name / runtime.gpus
.venv/bin/python scripts/g2/train/train_pipeline.py --dry-run
bash scripts/g2/train/train_smoke.sh
bash scripts/g2/train/train_pipeline.sh
```

每次实验放在 `runs/<experiment.name>/`，相对路径以仓库根目录为准：

```text
runs/<experiment.name>/
├── g2_train_config.yaml    # 实际配置；启动即写入，初始化成功后同步正式快照
├── train.log               # 本次启动配置和 norm/train 进程输出；续训追加
└── checkpoints/
    └── <step>/
        ├── params/
        ├── train_state/
        └── assets/         # 该 checkpoint 自带配置和统计量
```

启动就会创建实验目录；达到 `checkpoints.save_interval` 才出现权重步数目录。
在 tmux 等交互终端运行时保留动态进度条，并同步保存 `train.log`；非交互运行使用逐行日志。
共享统计量放在 `.cache/g2/norm/<签名>/`，签名包含数据、选臂、policy、相机和统计帧数。
`checkpoints.base_dir` 是实验根目录（默认 `./runs`）；`checkpoints.assets_dir` 是共享缓存根目录。
部署和续训使用 checkpoint 自带的统计量，不依赖共享缓存。
旧的 `checkpoints/` 和 `assets/` 兼容入口已清理；部署请使用 `runs/<实验名>/checkpoints/<步数>`。


## 数据

OpenPI 读 **LeRobot v2.1**。目录是 `data.root / data.repo_id`，不是写死的文件夹名。需要 `meta/episodes_stats.jsonl`。

```bash
# 采数机导出 v2.1 后同步到训练机
rsync -avP .../recorded/vr/lerobot_v21/ ~/openpi/data/<data.repo_id>/
```

`HF_LEROBOT_HOME` 默认等于 YAML 的 `data.root`（见 `scripts/g2/_env.sh`）。导出列和段边界见[公共契约](../../../../g2-policy-protocol/README.md#职责划分)。

## 改配置

### 多数据集

示例：[`configs/multi_dataset.yaml`](../configs/multi_dataset.yaml)，继承默认训练设置，只需修改路径、实验名和 GPU。每个 `data.sources[].path` 指向一份 v2.1 数据集；相对路径基于 `data.root`，默认 `./data`。不能同时写 `data.repo_id`；多源时拒绝 CLI 的 `--repo-id/--data-root`。

```bash
bash scripts/g2/train/train_pipeline.sh scripts/g2/configs/multi_dataset.yaml --dry-run
bash scripts/g2/train/train_pipeline.sh scripts/g2/configs/multi_dataset.yaml --smoke
```

多源 dry-run 读取本地元数据和文件信息，打印每个源的实际采样占比，不解码视频。全部不写 `weight`，或全部写 `weight: auto` 时自然混合；全部填写数字时按完整 episode 重复近似加权。`auto` 不能和数字权重混用。`data.image_size` 可省略，填写时 Pi05 仅接受 `[224, 224]`。

启动和 dry-run 的控制台仅显示各源的 episode 数、有效样本数、虚拟重复后的样本数、重复次数范围及实际占比；不展开 `resolved.mixture.lengths/plan` 的逐 episode 数字。完整列表仍保存在 checkpoint 的 `assets/g2_train_config.yaml` 中，供训练和续训恢复使用。

多源统计量写入 `.cache/g2/norm/<签名>/mix-<指纹>`，按展开分布计算，norm 阶段不解码视频。checkpoint 保存 asset_id，部署无需原始数据。续训校验混合身份，复用 checkpoint 统计量，但不保证数据顺序接续。

训练和部署代码与协议包一起更新。旧单源 checkpoint 的 asset_id 仍为原 repo_id。完整规则见[多数据集与图像尺寸](../../../../g2-policy-protocol/docs/policy-config.md#多数据集与图像尺寸)。

先改这几项就能跑：`experiment.name`、`experiment.arm`、`data.root`、`data.repo_id`、`runtime.gpus`。
`policy` 默认已是 TeaPi：右臂实测绝对位姿，学习指令相对当前实测的分量差。改作差/坐标/历史见字段速查；自定义 YAML 只写要覆盖的叶子，其余继承 `configs/train.yaml`。字段翻译与 checkpoint 快照由 `openpi.adapters.g2.training_config` 维护。训练 YAML 没有格式版本号：未知字段一律拒绝；字段含义改变时换字段名。共享包只约束 `policy` 的状态、动作和时序语义。

| 字段 | 含义 |
|---|---|
| `training.batch_size` | 全局 batch，沿用 OpenPI 语义 |
| `training.steps` / `training.lr_schedule.peak_lr` | `num_train_steps` / 余弦调度器的 `peak_lr` |
| `training.num_workers` / `training.seed` | 数据加载进程数 / 随机种子 |
| `logging.log_interval` / `checkpoints.save_interval` | 日志 / checkpoint 保存间隔 |
| `checkpoints.resume` | 恢复当前实验最新完整 checkpoint |
| `finetuning.<模块>.mode` | `frozen` / `lora` / `full`，三个模块互不覆盖 |
| `training.smoke` | 短跑：batch=1、20 步、独立实验名、默认关 W&B |
| `policy.timing.action_horizon` | 预测帧数，默认 50，必须写在 YAML 里 |
| `model.discrete_state_input` | 状态是否编进 prompt；宽于 32 维必须 true |
| `normalization.mode` | 新训练 `auto` / `force` / `skip`；续训读 checkpoint |
| `checkpoints.overwrite` | `true` 会删掉同名实验的全部权重 |
| `runtime.gpus` | 写入 `CUDA_VISIBLE_DEVICES`；已有环境变量优先 |

```bash
# 只覆盖部分字段的实验 YAML
bash scripts/g2/train/train_pipeline.sh scripts/g2/configs/my_experiment.yaml
bash scripts/g2/train/train_pipeline.sh --config scripts/g2/configs/my_experiment.yaml --batch-size 2
```

合并顺序：`configs/train.yaml` → 自定义 YAML → CLI。`--` 之后只接受 optimizer/EMA，例如 `-- --optimizer.clip-gradient-norm=0.5`。

### 自动实验名称

`experiment.name: auto` 按 `<配置文件名>_<arm>_<state>_<action>_<日期>_<版本>` 生成名称。
也可以只固定任务前缀，其余从最终生效的 policy 和选臂自动填入：

```yaml
experiment:
  name: chapai_layer3_{arm}_{state}_{action}_{date}_{version}
  arm: right
```

例如 `chapai_layer3_right_spose_abs_apose_dcomp_20261031_1430_v01`。
占位符为 `{task}`（不带扩展名的配置文件名，小写）、`{arm}`、`{state}`、`{action}`、
`{date}`（Asia/Taipei 的启动时间，YYYYMMDD_HHMM）、`{version}`（v01 起）。
位姿/关节分别用 `pose/joint`；绝对值用 `abs`，分量差用 `dcomp`，SE(3) 差用 `dse3`，
同时启用多个状态分支时拼接它们的名称。
名称只概括状态和动作表示，坐标、参考、旋转编码和时序等完整语义仍保存在 YAML 快照中。

版本号按同一实验根目录、同一模板和启动分钟下已有名称的最大版本递增，模型微调模式变化也不会覆盖同名实验。
`--dry-run` 仅预览，不创建或预留目录；并发启动同一模板时应通过 `--exp-name` 指定不同名称。
生成后的完整名称会写入 checkpoint 快照和训练日志；续训须使用完整名称，避免日期或版本变化：

```bash
bash scripts/g2/train/train_pipeline.sh --config scripts/g2/configs/G2_ChaPai.yaml \
  --exp-name 'chapai_layer3_{arm}_{state}_{action}_{date}_{version}' --dry-run
# 续训示例：名称替换为实际生成的名称
bash scripts/g2/train/train_pipeline.sh --config scripts/g2/configs/G2_ChaPai.yaml \
  --resume --exp-name chapai_layer3_right_spose_abs_apose_dcomp_20261031_1430_v01
```

手写的固定名称保持原样。短跑仍使用独立的时间戳名称，CLI 的 `--exp-name` 可覆盖。

### 按实验目录续训

```bash
bash scripts/g2/train/train_pipeline.sh --run runs/Chapai_layer3_right --steps 100000
# 只预览，不启动训练
bash scripts/g2/train/train_pipeline.sh --run runs/Chapai_layer3_right --steps 100000 --dry-run
```

`--run` 自动选择最新完整 checkpoint，并读取该 step 的 `assets/g2_train_config.yaml`，无需重新填写 YAML 或自动生成的名称。
实验根目录的配置用于记录启动情况；续训以 checkpoint 内随训练状态保存的配置为准。
步数是目标总步数，必须大于已保存步数。模型、policy、数据和优化器的续训校验仍生效。
未保存过完整 checkpoint 的实验不能续训。

## 预览、短跑、正式训练

```bash
# 打印生效配置；不读数据集、不训练。续训时会读 checkpoint 元数据
.venv/bin/python scripts/g2/train/train_pipeline.py --dry-run

bash scripts/g2/train/train_smoke.sh
bash scripts/g2/train/train_pipeline.sh
CUDA_VISIBLE_DEVICES=0,1 bash scripts/g2/train/train_full.sh   # YAML 先设 runtime.fsdp_devices: 2
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
finetuning:
  vision: {mode: lora, lora: {rank: 16, alpha: 16.0, targets: [attention], rslora: false}}
  language: {mode: lora, lora: {rank: 16, alpha: 16.0, targets: [attention, ffn], rslora: false}}
  action_expert: {mode: full}
```

视觉 LoRA 只支持 `attention`；语言和动作专家可选 `attention` / `ffn`。`rslora: true` 时缩放为 alpha/sqrt(rank)。语言或动作专家用 LoRA 时关闭 EMA。改 LoRA 结构要换 `experiment.name`，不能拿旧优化器续训。当前默认会存到 `runs/<name>/checkpoints/`。

模型配置名仍用于说明模型与微调模式，但不再参与目录层级。实验目录只使用 `experiment.name`，完整模型、数据和 policy 信息保存在配置快照中。

## 单臂

```bash
.venv/bin/python scripts/g2/train/train_pipeline.py --arm right --no-resume --exp-name right_arm_run
```

换臂会换配置名，必须重算统计量；原始双臂数据不用重导。缺的腕部相机填黑图并 mask。部署的 `--policy.dir` 必须指向这次实验的 step，不要复用双臂路径。服务端用双臂实测 EE 把单臂预测补回 20 维；域控按握手的 `arm` 只采对应相机、只向该侧发令。

## 常见问题

- `runs/<experiment.name>` 已存在且未开 overwrite：换 `experiment.name`，或通过 `--run` 续训。
- 改了 delta/method/坐标/历史：新建实验并重算统计量。
- 自定义 YAML 在默认 `action.pose` 上只写 `delta: false`：会清掉继承的 method 和 reference。
- 宽状态却没开 `discrete_state_input`：配置阶段直接失败。


## YAML 归属与 checkpoint

训练 YAML 按职责组织字段：

| 分组 | 内容 |
| --- | --- |
| `experiment` | 实验名和选臂 |
| `runtime` | `gpus`、`fsdp_devices`、`mem_fraction` |
| `model` / `finetuning` | 基础模型、状态输入方式和各模块微调策略 |
| `data` / `policy` | 数据来源，以及状态、动作和时序约定 |
| `training` | 全局 batch、总步数、worker、seed、smoke、`lr_schedule` |
| `checkpoints` | 路径、resume、overwrite、保存频率与保留周期 |
| `normalization` | 统计量复用和采样上限 |
| `logging` | 指标打印间隔和 `wandb` |

`model.type` 当前仅接受 `pi05`，`model.pretrained_path` 映射到参数加载路径。
`resolved.mixture` 是混合数据规划器生成的快照元数据，实验 YAML 不需要手填。
冻结和全参模块可省略 `lora`；缺省选项在默认值继承后补全，保存快照时包含完整参数。

`logging.wandb.mode` 为 `auto` / `on` / `off`；`auto` 在 smoke 时关闭。
YAML 只将 true/false 作为布尔值，接受 `1e-5` 等科学计数法，并拒绝重复 key。
CLI `--steps`、`--batch-size`、`--num-workers`、`--log-interval`、`--save-interval`、
`--learning-rate` 优先于 YAML；现有 CLI 参数名保持不变。
`checkpoints.resume` 是布尔值，恢复当前实验的最新完整 checkpoint。
`CUDA_VISIBLE_DEVICES` 和 `XLA_PYTHON_CLIENT_MEM_FRACTION` 优先于 YAML 的对应设置。
可见设备数须能被 `runtime.fsdp_devices` 整除，全局 batch 须能被可见设备数整除。

`train.yaml` 是默认模板，默认 GPU 为 0、实验名为 `g2_experiment`、关闭续训、自动复用统计量。
升级前的 Sing_TeaPi 实验参数保留在 `configs/Sing_TeaPi.yaml`；ChaPai 使用 `configs/G2_ChaPai.yaml`。
默认模板和实验 YAML 先各自翻译为内部配置，再合并显式覆盖。

新 checkpoint 的 `g2_train_config.yaml` 保存自包含配置；恢复时不合并当前训练模板。
早先的快照带有 `schema_version: 8`、`backend: openpi` 两个字段，读取时忽略；其他取值（如旧的 7）会被拒绝。

LeRobot 的 G2 训练 YAML 与本仓库段落对齐（batch_size 同为全局 batch）；两个框架的优化器与模型实现不同，同名参数不代表数值等价。
