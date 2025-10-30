# LIBERO Pro Evaluation Jupyter Notebook

这是将原始 `main_libero_pro.py` 脚本改造成的 Jupyter Notebook，使得你可以在交互式环境中逐个执行评估步骤。

## 📋 Notebook 结构

这个 notebook 被组织成 6 个清晰的 Cell，每个 Cell 都有具体的功能：

### Cell 1: 导入库 (Import Libraries)
- 导入所有必需的依赖库
- 包括 LIBERO、OpenPI 客户端和其他工具库

### Cell 2: 定义常量和辅助函数 (Define Constants and Helper Functions)
- 定义 `LIBERO_DUMMY_ACTION` 和 `LIBERO_ENV_RESOLUTION` 常量
- 实现 `_quat2axisangle()` 四元数到轴角的转换函数
- 实现 `_get_libero_env()` 环境初始化函数

### Cell 3: 定义日志和配置管理函数 (Define Logging and Config Management Functions)
- 实现 `setup_logging()` 设置日志系统
- 实现 `save_experiment_config()` 保存实验配置
- 实现 `save_episode_video()` 保存 episode 视频

### Cell 4: 设置超参数 (Set Hyperparameters)
- **在这个 Cell 中修改各个超参数**
- 包括模型服务器参数、LIBERO 环境参数、日志路径等
- 所有参数被组织在一个字典 `args` 中，供后续使用

**可调整的关键参数：**
- `task_suite_name`: 任务套件选择 (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`)
- `num_trials_per_task`: 每个任务的试验次数（默认 50）
- `host` & `port`: 模型服务器地址
- `resize_size`: 图像大小（默认 224）
- `replan_steps`: 重规划步数（默认 5）
- `seed`: 随机种子（默认 7）

### Cell 5: 运行评估 (Run Evaluation)
- 执行主要的 `eval_libero()` 函数
- 运行 LIBERO 任务评估流程
- **这是最耗时的 Cell，实际评估在这里进行**

过程包括：
1. 设置日志和实验配置
2. 初始化 LIBERO 任务套件
3. 为每个任务构建可重用的 ValueMap
4. 逐个执行 episode，每个 episode 包含：
   - 重置环境到初始状态
   - 获取预处理图像和状态
   - 查询模型获取动作
   - 使用 ValueMap 评估轨迹成本
   - 执行最优的动作轨迹
   - 保存结果视频和成本数据

### Cell 6: 总结和结果 (Summary and Results)
- 显示评估完成信息
- 列出所有输出文件位置
- 提供后续分析建议

## 🚀 使用方法

### 1. 启动 Jupyter Notebook

```bash
cd /hdd/zijianwang/openpi/examples/libero/
jupyter notebook main_libero_pro.ipynb
```

### 2. 逐个执行 Cell

在浏览器中打开 notebook 后：

1. **执行 Cell 1-3**: 按顺序执行，设置环境和定义函数
   ```
   Shift + Enter 执行当前 cell
   或点击工具栏的 Run 按钮
   ```

2. **编辑 Cell 4**: 修改超参数以调整评估配置
   - 修改 `task_suite_name` 选择不同的任务套件
   - 调整 `num_trials_per_task` 改变试验次数
   - 修改其他参数如需要

3. **执行 Cell 5**: 启动评估
   - 这会触发完整的评估流程
   - 进度会实时显示在 notebook 中
   - **注意：这可能需要较长时间，具体取决于试验次数**

4. **执行 Cell 6**: 查看评估总结

### 3. 交互式修改工作流

与原始脚本相比，notebook 的优势在于：

- ✅ **分步执行**: 无需运行完整脚本，可以测试每个部分
- ✅ **即时修改**: 编辑完超参数后立即运行，无需重启
- ✅ **可视化调试**: 在 notebook 中查看中间结果和日志
- ✅ **保存状态**: notebook 保存了执行历史和输出

## 📊 输出文件

运行完毕后，结果会保存到以下位置：

```
./experiments/
├── logs/
│   ├── LIBERO-PRO-{task_suite}-{timestamp}.txt          # 详细日志
│   └── LIBERO-PRO-{task_suite}-{timestamp}_config.json  # 实验配置
├── videos/libero_pro/
│   └── LIBERO-PRO-{task_suite}-{timestamp}/
│       └── {task_name}/
│           ├── episode_000_success.mp4
│           ├── episode_001_failure.mp4
│           └── ...                                      # 每个 episode 的视频
└── cost/
    └── LIBERO-PRO-{task_suite}-{timestamp}/
        ├── cost_data_task_0.json                        # 任务 0 的成本数据
        └── ...
```

## 🔍 关键代码说明

### Cell 4 的超参数含义

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `host` | 模型服务器地址 | `0.0.0.0` |
| `port` | 模型服务器端口 | `8000` |
| `resize_size` | 预处理图像大小 | `224` |
| `replan_steps` | 每次规划的步数 | `5` |
| `sampling_bs` | 采样批大小 | `8` |
| `task_suite_name` | 任务套件名称 | `libero_spatial` |
| `num_steps_wait` | 物理模拟稳定等待步数 | `10` |
| `num_trials_per_task` | 每个任务的试验次数 | `50` |
| `seed` | 随机种子 | `7` |

### Cell 5 的关键流程

1. **环境初始化**
   - 加载 LIBERO 任务套件
   - 初始化 WebSocket 客户端连接到模型服务器

2. **ValueMap 构建** (根据内存 [[memory:10148411]])
   - 为每个任务构建一个可重用的 ValueMap
   - 用于评估轨迹成本，指导最优动作选择

3. **图像预处理** (根据内存 [[memory:10152401]])
   - 执行 180 度旋转 (`img[::-1, ::-1]`)
   - 这是匹配训练数据预处理的重要步骤

4. **轨迹评估**
   - 使用 `compute_eef_trajectory_from_actions()` 计算末端执行器轨迹
   - 使用 `evaluate_trajectory_with_value_map()` 评估轨迹质量
   - 选择成本最低的轨迹执行

## 💡 常见问题

### Q: 如何中断评估？
A: 点击 notebook 工具栏中的 ⏹️ 停止按钮，或按 `Ctrl+C`

### Q: 如何修改日志级别？
A: 在 Cell 5 的开头修改：`logging.basicConfig(level=logging.DEBUG)`

### Q: 如何只运行一个任务？
A: 在 Cell 4 中修改，或在 Cell 5 中的 `for task_id in range(num_tasks_in_suite):` 后添加 `break` 来限制循环

### Q: 如何查看实时进度？
A: Cell 5 会显示 tqdm 进度条，Cell 6 会显示最终统计

## 📝 注意事项

1. **模型服务器**: 请确保在运行 Cell 5 前，模型服务器已启动在 `host:port`
2. **依赖库**: 所有必需的库在 Cell 1 中导入，确保环境已正确配置
3. **磁盘空间**: 视频文件较大，请确保有足够的磁盘空间
4. **长时间运行**: 完整评估可能需要数小时，建议在稳定的环境中运行

## 🔧 扩展和修改

你可以在各个 Cell 之间添加自己的代码：

- 在 Cell 5 和 Cell 6 之间添加数据分析代码
- 修改 `eval_libero()` 函数来自定义评估流程
- 添加可视化代码来分析结果

## 📚 相关文件

- `main_libero_pro.py`: 原始 Python 脚本版本
- `util.py`: 辅助函数库（ValueMap 构建和轨迹评估）
- `third_party/LIBERO-PRO/evaluation_config.yaml`: 评估配置文件

