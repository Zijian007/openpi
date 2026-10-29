# G2 × OpenPI π₀.₅

G2 的 LeRobot v2.1 数据训练和 WebSocket 推理入口，使用 JAX/OpenPI。
策略适配代码位于 [`src/openpi/adapters/g2/`](../../src/openpi/adapters/g2/)；
这里集中放配置、启动脚本、使用文档及入口测试。

## 文件布局

```text
scripts/g2/
├── README.md                  # 导航与快速开始
├── bootstrap.sh               # 安装依赖，可选下载基础权重
├── _env.sh                    # 共享路径、缓存和启动函数
├── configs/                   # 日常修改的配置
│   ├── train.yaml             # 数据、策略、模型和训练参数
│   └── deploy.yaml            # checkpoint、GPU、端口、预热和 RTC
├── train/                     # 统计量与训练入口
│   ├── train_pipeline.py      # 配置解析 → norm → train
│   ├── train_pipeline.sh      # 按 train.yaml 正常训练
│   ├── train_smoke.sh         # 20 步短跑
│   ├── train_full.sh          # 全参微调
│   └── compute_norm_stats.sh  # 只计算统计量
├── deploy/                    # 推理服务与通信检查
│   ├── serve_policy.py
│   ├── serve_policy.sh
│   ├── smoke_infer_client.py
│   └── smoke_infer_client.sh
├── docs/
│   ├── training.md            # 从改 YAML 到短跑、续训、选臂
│   ├── deployment.md          # 加载 checkpoint、预热和推理检查
│   ├── policy.md              # 状态、动作、坐标系、时间与协议
│   ├── rtc.md                 # RTC 前缀引导和客户端调度
│   └── architecture.md        # 适配层职责与原生 OpenPI 边界
└── tests/                     # 启动入口回归测试
```

所有命令从 OpenPI 仓库根目录运行；配置中的相对数据、权重和输出路径也以仓库根目录为基准。
数据、统计量和权重分别放在仓库的 `data/`、`assets/`、`checkpoints/`，不放在脚本目录。

## 快速开始

1. 安装：`bash scripts/g2/bootstrap.sh`；加 `--download-base` 可下载 `pi05_base`。
2. 准备 LeRobot **v2.1** 数据，并修改 [`configs/train.yaml`](configs/train.yaml) 的数据路径、选臂和训练设置。
3. 预览后短跑，再启动正式训练：

   ```bash
   .venv/bin/python scripts/g2/train/train_pipeline.py --dry-run
   bash scripts/g2/train/train_smoke.sh
   bash scripts/g2/train/train_pipeline.sh
   ```

4. 修改 [`configs/deploy.yaml`](configs/deploy.yaml) 的 checkpoint 和进程设置，启动服务：

   ```bash
   bash scripts/g2/deploy/serve_policy.sh
   ```

5. 另一个终端检查推理通信：

   ```bash
   bash scripts/g2/deploy/smoke_infer_client.sh
   ```

部署从 checkpoint 的 `assets/g2_train_config.yaml` 恢复训练语义。`policy` 字段见 [公共配置速查](../../../g2-policy-protocol/docs/policy-config.md)；本仓库怎么覆盖 YAML、微调模块和短跑见下表。
默认微调为 vision/language LoRA、动作专家全参；`run.smoke` 独立控制短跑；`train_full.sh` 将三个模块全部设为 full。详见 [训练说明](docs/training.md#微调)。

## 使用文档

| 要做什么 | 入口 |
|---|---|
| 准备数据、改训练参数、短跑、续训、选臂 | [训练说明](docs/training.md) |
| 选择权重、启动服务、检查推理 | [部署说明](docs/deployment.md) |
| 本仓库如何覆盖 policy、快照和与 LeRobot 互拷 | [Policy 配置](docs/policy.md) |
| 配置推理延迟补偿和轨迹衔接 | [RTC](docs/rtc.md) |
| 找实现位置、理解 G2 与原生代码的依赖 | [适配架构](docs/architecture.md) |

训练和部署 Shell 入口均支持 `--config PATH`。不传时分别使用 `configs/train.yaml`、`configs/deploy.yaml`。
也可直接修改 `train/train_pipeline.sh` 顶部的 `G2_TRAIN_CONFIG="..."`，保存后运行脚本。
`train_pipeline.sh` 也接受第一个参数直接指定训练YAML；相对路径以仓库根目录为准。
可将不同实验的 YAML 放在 `configs/` 下，通过启动参数选择：

```bash
bash scripts/g2/train/train_pipeline.sh --config scripts/g2/configs/train_right.yaml --dry-run
bash scripts/g2/train/train_pipeline.sh --config scripts/g2/configs/train_right.yaml
bash scripts/g2/train/compute_norm_stats.sh --config scripts/g2/configs/train_right.yaml

bash scripts/g2/deploy/serve_policy.sh --config scripts/g2/configs/deploy_right.yaml
bash scripts/g2/deploy/smoke_infer_client.sh --config scripts/g2/configs/deploy_right.yaml
```

上面的 `train_right.yaml`、`deploy_right.yaml` 是自定义文件名示例，需自行创建。
训练 YAML 支持只写覆盖字段，其余继承 `configs/train.yaml`；部署 YAML 需包含完整部署设置。
Shell 入口也支持 Python 入口的临时覆盖，例如 `--batch-size 2`、`--port 8002`。
所有相对配置路径均以仓库根目录为基准；其他目录下的 YAML 可传绝对路径。
部署保留 `--deploy-config` 别名；smoke 客户端从同一份部署 YAML 读取地址和端口，`--host` / `--port` 可覆盖。

## 开发

```bash
.venv/bin/python -m pytest -q scripts/g2/tests src/openpi/adapters/g2/tests
```

跨采数机、GPU 和域控的操作见 [训练手册](../../../g2_wzj_docs/ops/ml/openpi-pi05-train-g2.md)
和 [部署手册](../../../g2_wzj_docs/ops/ml/openpi-pi05-deploy-g2.md)。
OpenPI 与 lerobot-finetune 的数据和 checkpoint 不互通。
