# G2 新服务器环境安装

[返回入口](../README.md) · [训练](training.md) · [部署](deployment.md)

适用于 Ubuntu GPU 服务器上的 OpenPI/G2 训练与推理。以下以仓库位于 `~/openpi` 为例，
使用仓库自己的 `.venv`，Python 版本按 `.python-version` 固定为 **3.11**。
只部署已有 checkpoint 时也使用这套环境，不需要准备训练数据。

## 1. 系统与 GPU

先确认 NVIDIA 驱动可用：

```bash
nvidia-smi
```

需要支持 CUDA 12 的驱动。Python 环境会安装 JAX/PyTorch 所需的 CUDA 运行库，
通常不用另装系统 CUDA Toolkit；驱动不由 bootstrap 安装。
`nvidia-smi` 显示的 CUDA Version 是驱动支持上限，不代表已安装对应 Toolkit。

Ubuntu 22.04 可按下面准备系统工具；已安装时可跳过：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl ca-certificates build-essential ffmpeg
git lfs install
```

FFmpeg 用于视频解码。安装依赖需要访问 Python 包源和 GitHub；首次下载模型还需要访问
Google Cloud Storage 和 Hugging Face。

## 2. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

已安装 uv 时直接验证版本即可。若 `--no-sources-package` 等选项不被识别，先按原安装方式升级 uv。

## 3. 准备仓库和协议包

使用当前包含 G2 改动的仓库版本，保留 `pyproject.toml`、`uv.lock`、`src/`、`packages/` 和 `scripts/`。
从其他机器同步时不要复制 `.venv`；editable 安装和解释器路径需要在新机器重建。
数据、模型缓存与 `runs/` 可按需另行同步。

协议包有两种准备方式，选择一种。

### 使用相邻源码

将相互匹配的两个仓库版本 clone 或同步到以下位置：

```text
~/
├── openpi/
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── src/openpi/
│   └── packages/openpi-client/
└── g2-policy-protocol/
    ├── pyproject.toml
    └── src/g2_policy_protocol/
```

当前 OpenPI 要求 `g2-policy-protocol==0.7.0`。源码方式安装后不能移走相邻协议仓库。
域控客户端也需要使用同一份协议源码或同一个 wheel。

### 使用协议 wheel

如果新服务器不放协议源码，在原机器构建 wheel：

```bash
cd ~/g2-policy-protocol
uv build --wheel --out-dir dist
```

把生成的 `g2_policy_protocol-0.7.0-py3-none-any.whl` 复制到新服务器，例如 `~/g2-wheels/`。
后面的 bootstrap 命令加上 `G2_PROTOCOL_WHEELS="$HOME/g2-wheels"`，无需相邻协议源码。
该目录只提供协议包，其他依赖仍通过网络安装。

## 4. 安装项目环境

相邻源码方式：

```bash
cd ~/openpi
bash scripts/g2/bootstrap.sh
```

协议 wheel 方式：

```bash
cd ~/openpi
G2_PROTOCOL_WHEELS="$HOME/g2-wheels" bash scripts/g2/bootstrap.sh
```

bootstrap 调用 `uv sync --extra g2 --no-dev`，自动创建 `.venv`，安装第三方依赖，
同时将 **OpenPI 和 openpi-client 本身**安装为 editable package。
仅创建虚拟环境或安装第三方包还不够。

不需要先 activate，也不需要额外执行 `pip install -e .`。
即使终端显示其他环境名称，后续明确使用 `.venv/bin/python` 即可。
首次安装输出应出现 `openpi`、`openpi-client`、`g2-policy-protocol`，并最终打印 `[bootstrap] done`。
中途失败时先解决报错，再重新执行同一安装命令。

按需增加选项，可组合使用：

| 选项 | 用途 |
|---|---|
| `--download-base` | 预下载 `pi05_base`；部署已有微调 checkpoint 时可省略 |
| `--dev` | 加装 pytest、Ruff、Notebook 等开发工具 |

例如训练开发环境：

```bash
bash scripts/g2/bootstrap.sh --dev --download-base
```

wheel 方式使用这些选项时同样加上 `G2_PROTOCOL_WHEELS`。
默认不安装 ALOHA 仿真、TensorFlow/RLDS；OpenCV 使用 headless 版本。
启动训练和服务不会自动补装依赖，代码更新涉及依赖变化时重新执行 bootstrap。

## 5. 验证安装

先检查项目、协议和视频库是否可导入：

```bash
cd ~/openpi
.venv/bin/python - <<'PY'
import sys
from importlib.metadata import version
import openpi
import g2_policy_protocol
import cv2
import torchcodec

print("Python:", sys.executable)
print("OpenPI:", list(openpi.__path__))
print("Protocol:", version("g2-policy-protocol"))
print("OpenCV:", cv2.__version__)
print("TorchCodec:", version("torchcodec"))
PY
uv pip check --python .venv/bin/python
```

`OpenPI` 应指向本机 `~/openpi/src/openpi`。再验证 JAX 能使用 GPU 并完成一次计算：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda .venv/bin/python - <<'PY'
import jax
import jax.numpy as jnp

print("JAX devices:", jax.devices())
print("GPU computation:", (jnp.ones((2, 2)) + 1).block_until_ready())
PY
```

最后预览训练配置：

```bash
.venv/bin/python scripts/g2/train/train_pipeline.py --dry-run
```

默认单数据源模板的 dry-run 不读数据、不下载模型，也不启动训练。
如果选择多数据源 YAML，需要先准备数据集的本地元数据才能生成采样规划。
dry-run 成功仅证明配置可解析；训练数据、GPU 显存和模型加载还需要后续 smoke 验证。

## 6. 开始训练或部署

- **训练**：按[训练说明](training.md)准备 LeRobot v2.1 数据，修改 YAML 的数据路径、GPU、实验名，先 smoke 再正式训练。
- **部署**：复制完整 checkpoint step 目录（包含 `params/`、`assets/` 等），按[部署说明](deployment.md)设置 `checkpoint` 后启动服务；无需复制训练数据和共享 norm 缓存。
- **W&B**：使用在线日志时先执行 `.venv/bin/wandb login`；不使用时将训练 YAML 的 `logging.wandb.mode` 设为 `off`。

默认模型缓存为 `~/.cache/openpi`，可通过 `OPENPI_DATA_HOME` 改到大容量目录。
Shell 入口将 Hugging Face 缓存设为 `~/.cache/openpi-hf`，数据默认放在仓库 `data/`；
实际训练数据位置由 YAML 的 `data.root` 决定。基础权重与 tokenizer 首次使用时仍可能下载。

## 常见问题

| 报错或现象 | 处理方式 |
|---|---|
| `No module named 'openpi'` | 项目未安装到这份解释器，或复制的旧 `.venv` 内路径失效；在本机重新运行 bootstrap，再验证 `import openpi` |
| 找不到 `../g2-policy-protocol` | 按第 3 节准备相邻源码，或使用 `G2_PROTOCOL_WHEELS`；不用同名 PyPI 包替代 |
| 找不到 `openpi_client` 或 `g2_policy_protocol` | 检查 bootstrap 是否完成，以及使用的是否为仓库 `.venv/bin/python` |
| JAX 只有 CPU 或 CUDA 初始化失败 | 检查 `nvidia-smi`、`CUDA_VISIBLE_DEVICES` 和驱动兼容性；使用第 5 节 GPU 检查定位，dry-run 不能证明 GPU 可用 |
| `Could not load libtorchcodec`、缺 `libavcodec` | 检查系统 FFmpeg 及共享库是否安装；不要随意升级锁定的 Torch/TorchCodec 组合 |
| `cv2` 导入异常 | 重新执行 bootstrap；它会在移除普通 OpenCV 时重装 headless，修复共用文件 |

`PYTHONPATH=src` 只能临时影响导入搜索路径，不能代替完整安装。
不要将旧机器的 `.venv` 当作可迁移环境；在新服务器用源码、锁文件和 bootstrap 重建。
