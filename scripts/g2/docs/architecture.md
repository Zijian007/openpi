# G2 适配架构

[返回入口](../README.md) · [策略配置](policy.md)

## 与 G2_pi / LeRobot 边界

- 环境 / cache / W&B（`G2_openpi`）分开；ckpt **不互通**
- `serve_policy` ≠ `G2_pi` `policy_server`
- 域控共用 `G2WebSocketPolicyClient` 与 policy runtime，模型差异留在 GPU 适配器
## G2 通信协议

G2 使用 `g2-policy-protocol==0.5.0` 中共享的 `g2.policy.v3`，保留 v2 pose20 兼容。控制量、观测、时间和 reset 都通过能力声明；详见 [Policy 配置说明](policy.md#部署协议)。
协议是可选依赖：`uv sync --extra g2` 从相邻 `../g2-policy-protocol` 安装；原生安装无需此依赖。
仅部署 wheel 时，将 `tool.uv.sources.g2-policy-protocol.path` 改为本地 wheel 文件路径，再 `uv lock` 和 `uv sync --extra g2`。
G2 专属入口使用 `G2PolicyServer`；原生服务端不再按 robot metadata 分派或识别 G2。
G2 server 在入口把标准观测映射成模型内部的 `observation/*`，JPEG 解码、RTC、单臂扩展和图像处理保持在适配层。
默认 MessagePack，client 连接 `?encoding=json` 时整个会话使用 JSON text frames。
域控/GPU 需同步更新并重启；本次不兼容旧 `openpi.websocket.v1` G2 客户端。GPU-local smoke 脚本已同步使用新协议。

## 适配层边界

G2 适配逻辑位于 `openpi.adapters.g2`，启动入口位于 `scripts/g2/`，依赖方向为 G2 → 原生 OpenPI：

- `settings_schema.py`：State/Action共用直接解析路径；旧字段只在输入边界识别，不经过新旧格式往返，不加载模型依赖。
- `config.py`：YAML 解析、校验、旧 checkpoint 迁移与训练配置构建；预览时不加载模型依赖。
- `presets.py`：从同一基础配置派生旧checkpoint的具名预设；固定旧默认值，导入时不读取当前YAML。
- `contract.py`集中维护相机ID到OpenPI观测键的映射，数据重打包、服务端、预热和smoke客户端共用。
- `contract.py` / `data.py` / `transforms.py`：磁盘数据契约、数据配置接入、选臂和状态/动作变换。
- EEF状态历史复用ActionCodec的位姿几何运算，统一使用policy.pose标定；State固定共享观测参考，Action独立选择chunk/step。
- `policy.py`：先恢复 G2 配置，再委托原生 loader 加载模型；包装推理结果，处理单臂输出扩展。
- `server.py`：G2 协议、JPEG 解码与传输预热、metadata。
- `scripts/g2/deploy/serve_policy.py`：启动参数、策略预热与服务启动。
- `src/openpi/adapters/g2/tests/`：适配层回归测试与原生代码隔离检查。
- `scripts/g2/tests/`：训练/部署入口回归测试，与 `train/`、`deploy/` 主入口分开。

原生模型、loss、优化器和 DataLoader 不包含 G2 分支。DataLoader 只提供可选 dataset_factory 通用扩展接口；G2 采样逻辑在 data.py。
原生配置注册表不注册 G2；Python 调用需使用 `openpi.adapters.g2.presets.get_config`。
原生 policy loader 不读取 G2 YAML；G2 checkpoint 使用 `openpi.adapters.g2.policy.create_trained_policy`。
G2 的 shell 启动命令不变，内部已切换至专属入口；旧的直接 `scripts/serve_policy.py ...pi05_g2...` 命令需更换入口。

checkpoint 只保留通用的 `checkpoint_assets` 文本资源接口；原生保存逻辑不知道 G2 文件名或 YAML 内容。
静态依赖检查和独立进程测试确保原生入口在禁止导入 G2 包时仍可导入。

