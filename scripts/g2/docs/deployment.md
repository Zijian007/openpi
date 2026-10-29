# G2 部署

[返回入口](../README.md) · [通信与架构](architecture.md) · [RTC](rtc.md)

`configs/deploy.yaml` 选择 checkpoint 和进程设置，`serve_policy.sh` 调用专属 `scripts/g2/deploy/serve_policy.py`，通过 G2 适配层加载模型。
原生 `scripts/serve_policy.py` 保持原生协议，不处理 G2 preset 或 G2 控制器。
默认 **warmup**：监听前进行两次普通推理；启用 RTC 时再进行两次引导推理，提前完成 JAX JIT；`--no-warmup` 可关。
`JAX_COMPILATION_CACHE_DIR` 在 `scripts/g2/_env.sh` 里指到 `$OPENPI_DATA_HOME/jax_compile_cache`。

多个推理配置可分别保存为 `configs/deploy_right.yaml`、`configs/deploy_left.yaml` 等完整 YAML，
使用 `--config PATH` 选择；原来的 `--deploy-config PATH` 仍兼容。
不传时读取 `configs/deploy.yaml`。服务端与 smoke 客户端使用同一份文件：

```bash
bash scripts/g2/deploy/serve_policy.sh --config scripts/g2/configs/deploy_right.yaml
# 另一个终端；自动读取该 YAML 的 host 和 port
bash scripts/g2/deploy/smoke_infer_client.sh --config scripts/g2/configs/deploy_right.yaml
```

上述文件名为示例，需先创建。服务端 CLI（如 `--policy.dir`、`--port`）覆盖所选 YAML。
smoke 客户端将监听地址 `0.0.0.0` / `::` 换为本机回环地址；连接远程服务时可传 `--host GPU主机地址`。
相对配置路径以仓库根目录为基准，绝对路径也支持。

```bash
# configs/deploy.yaml 默认 :8001，与 LeRobot 同时开时改其中一份 YAML
CUDA_VISIBLE_DEVICES=0 bash scripts/g2/deploy/serve_policy.sh \
  --policy.dir checkpoints/pi05_g2_vr_low_mem/<exp>/<step>
# 省略 --policy.dir 使用 configs/deploy.yaml 的 checkpoint；训练语义和 arm 从快照恢复
# healthz 在 warmup 结束后才 OK；日志应有 warmup step=0 infer_ms≈10s+，step=1 ≈100ms

bash scripts/g2/deploy/smoke_infer_client.sh   # 端口对齐 configs/deploy.yaml；首包应已是稳态
```

| 脚本 | 作用 |
|------|------|
| `deploy/serve_policy.sh` | 读取部署 YAML 后调用 G2 专属 server（默认 warmup） |
| `deploy/smoke_infer_client.sh` | 假观测 round-trip |

域控：`start_openpi_pi05_infer.sh`（`openpi_pi05_client`）。详见 [openpi-pi05-deploy-g2.md](../../../../g2_wzj_docs/ops/ml/openpi-pi05-deploy-g2.md)。
