# G2 Policy 配置 v2

日常修改 [train/config.yaml](train/config.yaml)，`schema_version: 2`。
所有 G2 数学和数据变换在 `src/openpi/adapters/g2/`。原生模型、loss、优化器不包含 G2 分支。
原生 DataConfig 仅增加可选的通用 dataset_factory 扩展接口，统计量计算和训练共用此接口。

## 状态、动作与参考

- `state.space`：`eef`、`joint`、`both`；both 每帧按所选臂的 EEF、Joint 顺序拼接。
- `state.representation`：`absolute` 或 `relative`。relative 对指定参考的编码分量作差，夹爪绝对。
- `action.space`：`eef` 或 `joint`；与 control_quantity 一致。
- `action.control_quantity`：见下表。每只手最后一维始终是夹爪 close_ratio。
- `action.label_source`：`command` 读取指令，`measured` 读取实测轨迹。
- 左臂、右臂、双臂均支持；模型状态与动作参考独立。例如 joint 状态可预测 EEF delta。

| control_quantity | space | 标签列 | 双臂通信维度 | 单位 |
|---|---|---|---|---|
| pose_target | eef | action.pose / state.pose | 20 | 米、rot6d、夹爪 |
| joint_position | joint | action.joint / state.joint | 16 | 弧度、夹爪 |
| joint_velocity | joint | action.joint_velocity / state.joint_velocity | 16 | 弧度/秒、夹爪 |
| eef_twist | eef | action.eef_twist / state.eef_twist | 14 | 线速度 m/s、角速度 rad/s、夹爪 |
| joint_torque | joint | action.joint_torque / state.joint_torque | 16 | N·m、夹爪 |

速度列缺失时，从对应 command/measured 轨迹计算差分：内部优先前向差分，终点用后向差分，
单帧 episode 的速度为零。EEF 角速度使用 `Log(R_next R_prev^T)/dt`；线速度使用位置差分。
力矩必须来自真实记录列，不能从位置差分生成。
现有 Sing_TeaPi 数据包含 pose 和 joint，其他列以数据实际存在情况为准。

参考名称在 state/action.relative_to 中共用：

| relative_to | 数据端 | Python 推理观测 |
|---|---|---|
| current_measured_pose | 当前 state.pose；关节动作使用当前 state.joint | observation/pose；关节动作 observation/joint |
| current_command_pose | 当前 action.pose | observation/command_pose |
| previous_command_pose | 前一个采样时刻的 action.pose | observation/previous_command_pose |
| episode_start_pose | episode 第一帧 state.pose / state.joint | observation/start_pose / start_joint |
| object_pose | reference.object_pose；相对关节状态额外需 reference.object_joint | observation/object_pose / object_joint |

Joint 相对状态支持 episode_start_pose、object_pose；后者需要明确记录参考关节配置。
Joint delta 动作使用当前实测关节参考。命令、物体参考缺失时会报错，不从模型输入猜测。
每个 WebSocket 会话独立管理起点和上一条指令；重连建立新起点。
Python G2Policy.reset() 清除这些状态。部署预热结束也会 reset，假观测不会成为真实起点。

## 动作表示

绝对动作：`representation: absolute`，relative_to、delta_scope 都为 null。
非绝对动作：明确指定参考，delta_scope 选择 chunk 或 step。

| representation | 定义：参考 T_r，目标 T_y |
|---|---|
| component_delta | 位置/旋转编码分量相减，夹爪绝对；不是 SE(3) 群运算 |
| se3_body_delta | `T_r^-1 T_y` |
| se3_spatial_delta | `T_y T_r^-1` |
| hybrid_delta | 父坐标系中的位置差；旋转 `R_r^T R_y` |

SE(3) delta 存储相对变换的平移与配置的旋转表示，不是六维李代数 log 向量。
chunk 全部目标共用指定参考；step 第一帧使用指定参考，后续使用前一个目标，解码累积还原。
RTC 前缀是通信坐标系中的绝对动作/控制量，复用同一个 codec 编码，不再次做标签偏移或采样。
Joint position 支持 absolute/component_delta。速度、twist、力矩已经是控制量，只使用 absolute。

## 坐标系、TCP、旋转

磁盘和通信的位姿始终是 `base_link` 下的 `flange`，旋转为 rot6d。
policy.pose 描述模型内部坐标：

```yaml
pose:
  parent_frame: base_link
  ee_frame: flange
  rotation: rot6d
  base_to_parent: null
  flange_to_tcp: null
```

- `rotation`：rot6d 为旋转矩阵前两列按列拼接；quaternion 使用 xyzw；axis_angle 使用弧度旋转向量。
- `base_to_parent`：4×4 `T_parent_base`，将 base 坐标变到目标父坐标系。
- `flange_to_tcp`：`{left: 4×4, right: 4×4}`，分别提供 `T_flange_tcp`。
- 模型位姿为 `T_parent_base T_base_flange T_flange_tcp`；输出执行逆变换。物体参考不套用机器人的 TCP 标定。
- 改父坐标系或 TCP 名称必须提供标定，不通过改名字执行隐式变换。
- `base_to_parent: observation` 支持逐帧坐标变换；数据读 reference.base_to_parent，推理传 observation/base_to_parent。
- 变换必须是有限刚体矩阵，旋转正交、det=+1、最后一行为 [0,0,0,1]。
- eef_twist 的线/角速度按父坐标系旋转，语义是法兰几何速度；不支持附带非零 TCP 偏移的 twist。
  TCP 位姿目标支持完整平移和旋转标定。

位置是米，关节角和轴角是弧度，夹爪 close_ratio 是 0 打开、1 闭合。

## 时间采样与历史

设目标采样频率 f=sample_hz，偏移 o=target_offset_steps，间隔 s=stride_steps，预测长度 H（`action_horizon`）：

```text
观测历史：t-(L-1)/f, ..., t
标签时刻：t + (o+i*s)/f，i=0,...,H-1
动作频率：f/s
```

- `sample_hz` 可与数据集 fps 不同。低维量线性插值，旋转使用 SO(3) Slerp，图像取最近帧。
- 所有采样限制在当前 episode。起点前的历史重复起点帧，终点后的标签重复终点。
- measured 标签单独存放，不覆盖作为参考的当前 state.pose。
- `history_steps` 支持大于 1：状态按最旧到最新展开；每路相机按同顺序横向拼图，再用原生图像变换处理。
  这是适配层的历史表达方式，不改变原生视觉网络；历史越长，每帧在拼图中的分辨率越小。
- Pi05 离散状态输入保留完整联合/历史状态；宽于 32 维时要求 discrete_state_input=true，并扩大 token 长度。
  动作模型宽度仍为 32，不改变预训练动作投影参数。
- 部署历史必须由客户端提供按 f 采样的完整数组，不能把稀疏推理请求当作连续视频帧。
  Python 直接调用在未提供历史时缓存逐次观测，调用方须保证采样时间一致。
- `execution_steps` 只从旧快照读取并校验范围；v3 握手不再把它发给客户端。提交长度在 wholebody `infer.execution_steps`。
  客户端按标签时刻调度；它不改变训练标签，不进入统计量/续训语义比较。

## 部署协议

有训练配置快照的 G2 policy 使用共享 `g2.policy.v3`，兼容保留无快照语义的 v2 pose20 服务；不再使用私有 OpenPI 协议。

- action_spec 声明完整双臂 wire 控制量、维度、字段、单位、base_link/flange 坐标约定。
- observation_spec 只声明模型实际需要的 pose/joint/prompt、相机、历史和 context；例如纯关节状态+关节位置不必再索要 wire pose。模型内部的 delta、rotation、LoRA 和 token 设置不进入通信契约。
- timing 声明 action_dt 和首目标偏移；L>1 请求必须携带 observation_time_ns 与全部协商的历史数组。
- reset 在同一连接开始新 episode，清理 adapter 的参考/历史；RTC 前缀使用声明的 wire 动作空间，由同一个 ActionCodec 编入模型空间。
- `scripts/g2/deploy/config.yaml` 只选 checkpoint、进程/GPU/端口、warmup 与 OpenPI RTC 参数。checkpoint 快照恢复训练语义和选臂，不需重复 ARM/CONFIG_NAME；老 checkpoint 才指定 legacy_preset。
- 提交长度只在客户端 YAML 的 `infer.execution_steps`。v2 握手仍可带旧快照的建议值；v3 忽略 server 侧该字段。`action_horizon`、`execution_steps`、`replan_after_steps` 的执行语义由客户端负责。

共用工具在 `g2_policy_tools`，OpenPI 的原导入路径保留兼容。协议、通用 server 和新 policy 接入接口见相邻仓库 `g2-policy-protocol/docs/protocol-v3.md`、`docs/architecture.md`。

wholebody 应用绑定 pose_target 与 joint_position。关节位置复用 Gello 真机 rad 限位和每拍 slew（`arm_gello.yaml`）；任何一侧出现 unlimited/`None` 限位则拒绝该 checkpoint。速度、twist、力矩暂未绑定执行器。外部命令/物体/动态坐标系需要 provider；非零首目标偏移当前在执行前拒绝。GPU-local smoke 可用合成参考/历史检查协议，不验证实际机器人执行。

## 配置示例

关节状态 + 独立 EEF 参考，只改：

```yaml
policy:
  state:
    space: joint
```

实测轨迹、下一帧开始、隔帧预测、历史输入：

```yaml
policy:
  action:
    label_source: measured
    representation: se3_body_delta
  timing:
    history_steps: 2
    target_offset_steps: 1
    stride_steps: 2
```

关节速度：

```yaml
policy:
  state:
    space: joint
  action:
    space: joint
    control_quantity: joint_velocity
    representation: absolute
    relative_to: null
    delta_scope: null
```

标定矩阵必须来自实际设备；默认配置不预填示例标定。

## 兼容性、统计量与预览

v1 配置/快照自动迁移，保留 relative→eef、absolute→joint 的旧状态约定。
v2 老快照缺少新增标定字段时补 null，不读取当前模板改变模型语义。
YAML 递归覆盖；部分配置切到 absolute 时，省略的 relative_to/delta_scope 自动变为 null。
快照保存完整配置，部署通过 G2 loader 恢复模型、时序、codec 和参考定义。

训练前检查所需列、维度、源 fps、声明的 base/flange/rot6d 契约。
旧导出缺少可选声明时沿用 G2 契约；command 监督声明与 action 标签必须一致。
统计量与训练使用完全相同的 dataset_factory、采样与编码变换。
改变采样、参考、控制量、标定、状态/动作表示后重算统计量；数据原地变化需 force。
续训禁止改变输入输出语义，允许改变 execution_steps、总步数和学习率。

```bash
.venv/bin/python scripts/g2/train/train_pipeline.py --dry-run
```

预览只展示配置、标签列、维度和时间范围，不读取数据、下载权重或启动训练。
