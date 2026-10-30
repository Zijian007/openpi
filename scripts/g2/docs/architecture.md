# G2 适配架构

[返回入口](../README.md) · [策略配置](policy.md)

OpenPI 与 lerobot-finetune 环境、cache、W&B 和 checkpoint 都不互通。域控走公共 `g2.policy.v3`，模型差异留在 GPU 适配器。

## 通信

G2 使用 `g2-policy-protocol==0.6.0` 的 `g2.policy.v3`。控制量、观测、时间和 reset 都通过能力声明；见 [Policy 配置](policy.md#快照和部署)。
协议是可选依赖：`uv sync --extra g2` 从相邻 `../g2-policy-protocol` 安装。部署 wheel 时运行 `G2_PROTOCOL_WHEELS=/path/to/wheels bash scripts/g2/bootstrap.sh`，uv 仅忽略协议包的源码覆盖，从 wheel 目录解析固定版本；无需相邻协议源码。训练与服务脚本直接使用已安装的 `.venv/bin/python`。

G2 入口是 `G2PolicyServer`。服务端把标准观测映射成模型内部的 `observation/*`；JPEG 解码、RTC、单臂扩展和图像处理留在适配层。默认 MessagePack，client 连接 `?encoding=json` 时整段会话用 JSON。域控和 GPU 需同版本并重启。

## 适配层

G2 逻辑在 `openpi.adapters.g2`，启动入口在 `scripts/g2/`，依赖方向为 G2 → 原生 OpenPI：

- `g2_policy_protocol.data.policy`：解析 `policy.state/action/timing`，校验语义并推导服务能力。
- `g2_policy_protocol.data.encoding` / `dataset`：共享状态/动作编码、时序采样及数据契约检查。
- `config.py`：YAML 读取、原生配置校验与训练配置构建；预览时不加载模型。
- `training_config.py`：本仓库训练 YAML 的版本、入口字段检查、原生字段映射和 checkpoint 快照；不依赖协议包或模型框架。
- `presets.py`：Pi05 模型模板与模块冻结规则；导入时不读当前 YAML。
- `contract.py`：相机 ID 到 OpenPI 观测键；磁盘契约和几何从 `g2_policy_protocol.data` 导入。
- `data.py` / `transforms.py`：数据集接入、选臂、状态和动作编码。图像准备独立于状态维度。
- `policy.py`：先恢复 G2 配置，再委托原生 loader；包装推理结果，处理单臂扩展。
- `server.py`：协议、JPEG 解码与传输预热、metadata。
- `scripts/g2/deploy/serve_policy.py`：启动参数、预热与服务。
- `src/openpi/adapters/g2/tests/`：适配层回归。
- `scripts/g2/tests/`：训练/部署入口回归。

原生模型、loss、优化器和 DataLoader 不含 G2 分支。原生配置注册表不注册 G2；训练配置由 `load_settings / resolve_settings / build_train_config` 构建。G2 checkpoint 用 `openpi.adapters.g2.policy.create_trained_policy`。checkpoint 只走通用 `checkpoint_assets` 文本接口。

## 磁盘与部署契约

两套训练入口共用磁盘契约和能力推导。服务返回完整 H 帧，执行预算留在客户端；历史与参考由显式上下文提供。详见[架构与数据链路](../../../../g2-policy-protocol/docs/development.md#职责划分)；跨框架联调命令见[跨仓库验证](../../../../g2-policy-protocol/docs/development.md#跨仓库验证)。
