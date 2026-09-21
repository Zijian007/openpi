#!/usr/bin/env python3
"""
机械臂控制演示
展示如何给机械臂发送指令并获取对应的action
"""

import robosuite as suite
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

class RobotController:
    def __init__(self):
        """初始化机械臂控制器"""
        # 创建RoboSuite环境
        self.env = suite.make(
            env_name="Lift",  # 使用Lift环境作为演示
            robots="Panda",   # 使用Panda机械臂
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,  # 简化演示，不使用相机
            control_freq=20,
        )

        # 重置环境
        self.obs = self.env.reset()

        # 从observation中获取当前末端执行器位姿
        self.current_ee_pos = self.obs["robot0_eef_pos"]  # [x, y, z]
        self.current_ee_quat = self.obs["robot0_eef_quat"]  # [x, y, z, w]

        print("机械臂控制器初始化完成")
        print(".3f")

    def get_current_pose(self):
        """获取当前末端执行器位姿"""
        return {
            'position': self.current_ee_pos.copy(),
            'orientation': self.current_ee_quat.copy()
        }

    def create_action_from_pose(self, target_pos, target_quat=None, gripper_action=0):
        """
        从目标位姿创建动作向量

        Args:
            target_pos: 目标位置 [x, y, z]
            target_quat: 目标姿态四元数 [x, y, z, w]，如果为None则保持当前姿态
            gripper_action: 夹爪动作 (-1: 张开, 0: 保持, 1: 闭合)

        Returns:
            action: 8维动作向量 [x, y, z, qw, qx, qy, qz, gripper]
        """
        if target_quat is None:
            target_quat = self.current_ee_quat

        # RoboSuite使用wxyz格式的四元数，但内部可能使用不同的顺序
        # 创建动作向量: [x, y, z, qw, qx, qy, qz, gripper]
        action = np.concatenate([target_pos, target_quat, [gripper_action]])

        return action

    def move_left(self, distance=0.1):
        """向左移动指定距离"""
        target_pos = self.current_ee_pos.copy()
        target_pos[0] -= distance  # x轴向左移动

        action = self.create_action_from_pose(target_pos)
        return action, target_pos

    def move_right(self, distance=0.1):
        """向右移动指定距离"""
        target_pos = self.current_ee_pos.copy()
        target_pos[0] += distance  # x轴向右移动

        action = self.create_action_from_pose(target_pos)
        return action, target_pos

    def move_up(self, distance=0.1):
        """向上移动指定距离"""
        target_pos = self.current_ee_pos.copy()
        target_pos[2] += distance  # z轴向上移动

        action = self.create_action_from_pose(target_pos)
        return action, target_pos

    def move_down(self, distance=0.1):
        """向下移动指定距离"""
        target_pos = self.current_ee_pos.copy()
        target_pos[2] -= distance  # z轴向下移动

        action = self.create_action_from_pose(target_pos)
        return action, target_pos

    def move_forward(self, distance=0.1):
        """向前移动指定距离"""
        target_pos = self.current_ee_pos.copy()
        target_pos[1] += distance  # y轴向前移动

        action = self.create_action_from_pose(target_pos)
        return action, target_pos

    def move_backward(self, distance=0.1):
        """向后移动指定距离"""
        target_pos = self.current_ee_pos.copy()
        target_pos[1] -= distance  # y轴向后移动

        action = self.create_action_from_pose(target_pos)
        return action, target_pos

    def execute_action(self, action):
        """执行动作并更新状态"""
        self.obs, reward, done, info = self.env.step(action)

        # 从新的observation中更新当前位置
        self.current_ee_pos = self.obs["robot0_eef_pos"]
        self.current_ee_quat = self.obs["robot0_eef_quat"]

        return self.obs, reward, done, info

    def open_gripper(self):
        """张开夹爪"""
        action = self.create_action_from_pose(self.current_ee_pos, gripper_action=-1)
        return action

    def close_gripper(self):
        """闭合夹爪"""
        action = self.create_action_from_pose(self.current_ee_pos, gripper_action=1)
        return action


def demo_basic_movements():
    """演示基本的移动指令"""
    print("=" * 50)
    print("机械臂控制演示")
    print("=" * 50)

    # 初始化控制器
    controller = RobotController()

    # 演示不同的移动指令
    movements = [
        ("向左移动 10cm", controller.move_left, 0.1),
        ("向右移动 10cm", controller.move_right, 0.1),
        ("向上移动 10cm", controller.move_up, 0.1),
        ("向下移动 10cm", controller.move_down, 0.1),
        ("向前移动 10cm", controller.move_forward, 0.1),
        ("向后移动 10cm", controller.move_backward, 0.1),
    ]

    print("\n初始位置:", controller.get_current_pose()['position'])

    for description, move_func, distance in movements:
        print(f"\n{description}:")
        action, target_pos = move_func(distance)

        print("动作向量:", action)
        print("目标位置:", target_pos)

        # 执行动作
        obs, reward, done, info = controller.execute_action(action)
        print("执行后位置:", controller.get_current_pose()['position'])

        time.sleep(0.1)  # 短暂延迟

    # 演示夹爪控制
    print("\n张开夹爪:")
    action = controller.open_gripper()
    print("动作向量:", action)
    controller.execute_action(action)

    print("\n闭合夹爪:")
    action = controller.close_gripper()
    print("动作向量:", action)
    controller.execute_action(action)

    print("\n演示完成!")


def demo_custom_movement():
    """演示自定义移动指令"""
    print("\n" + "=" * 50)
    print("自定义移动演示")
    print("=" * 50)

    controller = RobotController()

    print("当前末端执行器位姿:")
    current_pose = controller.get_current_pose()
    print(f"位置: {current_pose['position']}")
    print(f"姿态四元数: {current_pose['orientation']}")

    # 自定义移动到特定位置
    target_position = np.array([0.0, 0.5, 0.8])  # 自定义目标位置
    action = controller.create_action_from_pose(target_position)

    print(f"\n移动到目标位置: {target_position}")
    print(f"动作向量: {action}")

    controller.execute_action(action)
    print(f"执行后位置: {controller.get_current_pose()['position']}")


if __name__ == "__main__":
    demo_basic_movements()
    demo_custom_movement()
