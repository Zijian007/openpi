# 🚀 快速开始指南

## 什么是这个 Notebook？

`main_libero_pro.ipynb` 是一个交互式 Jupyter Notebook，它将原始的 Python 脚本 `main_libero_pro.py` 重新组织成 6 个独立的 Cell，使你能够：

- 📦 **分步导入库**
- ⚙️ **灵活修改超参数**
- ▶️ **按需运行评估**
- 📊 **实时查看结果**

## ⚡ 30 秒快速开始

```bash
# 1. 进入 notebook 所在目录
cd /hdd/zijianwang/openpi/examples/libero/

# 2. 启动 Jupyter
jupyter notebook main_libero_pro.ipynb

# 3. 在浏览器中按顺序运行 Cell：
#    - Cell 1-3: 初始化（自动执行）
#    - Cell 4: 修改超参数（按需修改）
#    - Cell 5: 运行评估（主要步骤）
#    - Cell 6: 查看结果（自动生成）
```

## 🎯 最常见的操作

### 修改任务套件

在 **Cell 4** 中找到这一行：
```python
task_suite_name = "libero_spatial"  # 默认
```

改为：
```python
task_suite_name = "libero_10"  # 或其他: libero_object, libero_goal, libero_90
```

### 修改试验次数

在 **Cell 4** 中找到这一行：
```python
num_trials_per_task = 50  # 默认
```

改为：
```python
num_trials_per_task = 10  # 更快的测试
```

### 修改模型服务器地址

在 **Cell 4** 中：
```python
host = "0.0.0.0"   # 改为你的服务器地址，如 "192.168.1.100"
port = 8000        # 改为实际的端口号
```

## 📋 Cell 说明

| Cell | 名称 | 用途 | 执行时间 |
|------|------|------|--------|
| 0 | Title | 标题页 | 0s |
| 1 | Import Libraries | 导入库 | < 1s |
| 2 | Helper Functions | 辅助函数 | 0s |
| 3 | Logging Functions | 日志函数 | 0s |
| 4 | Hyperparameters | **设置参数** | 0s |
| 5 | Run Evaluation | **运行评估** | 数分钟 - 数小时 |
| 6 | Summary | 结果总结 | < 1s |

## 💾 输出文件位置

运行完成后，检查以下目录：

```
./experiments/
├── logs/              ← 日志和配置
├── videos/            ← Episode 视频
└── cost/              ← 成本数据
```

## ❓ 常见问题速查

**Q: 如何快速测试？**
A: Cell 4 中将 `num_trials_per_task` 改为 1-2，只测试一个任务

**Q: 进度在哪里看？**
A: Cell 5 中有 tqdm 进度条，Cell 6 显示最终统计

**Q: 如何重新运行？**
A: 点击菜单 `Kernel → Restart & Clear Output` 后重新运行

**Q: 如何修改更多参数？**
A: Cell 4 中有详细注释，修改对应的变量即可

## 🔑 关键超参数参考

```python
# 模型服务器
host = "0.0.0.0"           # 服务器地址
port = 8000                # 服务器端口

# 任务配置
task_suite_name = "libero_spatial"  # 选择任务套件
num_trials_per_task = 50            # 每个任务的试验数

# 预处理
resize_size = 224          # 图像大小
replan_steps = 5           # 重规划间隔

# 随机性
seed = 7                   # 随机种子（设置为保证可重复性）
```

## 🎓 与原始脚本的对比

| 特性 | 原始脚本 | Notebook |
|------|--------|---------|
| 修改参数 | 需要编辑代码 | 直接修改 Cell 4 |
| 运行方式 | 全量运行 | 分步运行 |
| 中断恢复 | 需要重新开始 | 保留执行状态 |
| 调试 | 困难 | 容易 |
| 结果查看 | 查看日志文件 | 直接在 Cell 中 |
| 实验对比 | 手动管理多个脚本 | 轻松运行多个 Kernel |

## 🚨 注意事项

1. **启动前准备**：确保模型服务器已启动
2. **磁盘空间**：视频文件较大，预留足够空间
3. **长时间运行**：完整评估可能需要数小时，建议在后台运行
4. **依赖版本**：确保所有库已正确安装（见 Cell 1 的导入）

## 📞 需要帮助？

1. 查看 `README_NOTEBOOK.md` 获得详细说明
2. 查看 Cell 中的注释获得更多细节
3. 查看 `./experiments/logs/` 获得完整日志

