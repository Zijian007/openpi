# G2 Policy 配置

[返回入口](../README.md)

`policy` 字段候选、作差和采样规则见[公共配置速查](../../../../g2-policy-protocol/docs/policy-config.md)。本页只写 OpenPI 怎么读这份配置。

修改 [configs/train.yaml](../configs/train.yaml)。自定义 YAML 是对它的叶字段覆盖，须带 `schema_version: 6`。在默认 `action.pose` 上只写 `delta: false` 会清空继承的 method 和 reference。

State 同时启用两支时，各历史时刻按 pose → joint 拼接，再按最旧到最新展开；选臂由 `experiment.arm` 决定。宽于 32 维须 `model.discrete_state_input: true`。默认右臂 action pose 有效块为 (50,10)，入模型补到 (50,32)。速度缺列时可由对应来源轨迹差分，力矩须真实记录。转换矩阵由入口补齐并写入 checkpoint；数据集标定文件尚未自动加载。

## 快照和部署

新实验的归一化、训练 worker 和 checkpoint 保存填好的整棵 `policy` 树。恢复不读取当前训练模板。修改分支、坐标、编码或采样语义需重算统计量，续训会拒绝语义不一致的配置。仅改日志或训练步数不改变统计量语义。LoRA 模块模式和实际模型结构也参与续训检查。微调见[训练说明](training.md#微调)。

部署必须提供带 `assets/g2_train_config.yaml` 的 checkpoint。服务使用 `g2.policy.v3`，通信保持完整双臂绝对控制量；模型内部的 delta 与旋转编码不出现在 wire 上。握手声明所需 pose/joint、相机、参考和历史。RTC 前缀为标准通信动作，由同一个 codec 编码。当前 wholebody 执行器绑定 pose 目标和关节位置；关节速度、力矩尚未绑定。

## 与 lerobot-finetune

两边的 G2 字段路径相同：`experiment.arm`、`data.root/repo_id/cameras`、`policy.state/action/timing`。LeRobot 用 `adapter: {type: g2}` 启用适配，模型和训练字段在 `lerobot`；OpenPI 模型和训练字段保持现有结构。`policy` 块可直接复制。LeRobot 从模型读取预测长度；OpenPI 必须在 YAML 写 `policy.timing.action_horizon`。当前 OpenPI 读 LeRobot v2.1，lerobot-finetune 读 v3，复制配置时选对应数据目录。
