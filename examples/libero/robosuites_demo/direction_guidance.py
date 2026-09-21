import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import imageio
import os

def move_in_direction_and_record(direction, distance, video_name="robot_movement.mp4"):
    """
    控制机械臂移动并录制视频（适用于远程服务器）
    
    参数:
        direction: 移动方向 'left', 'right', 'forward', 'backward', 'up', 'down'
        distance: 移动距离（米）
        video_name: 保存的视频文件名
    """
    # 定义方向向量
    direction_vectors = {
        'left': np.array([0, 1, 0]),      # +Y
        'right': np.array([0, -1, 0]),    # -Y
        'forward': np.array([1, 0, 0]),   # +X
        'backward': np.array([-1, 0, 0]), # -X
        'up': np.array([0, 0, 1]),        # +Z
        'down': np.array([0, 0, -1])      # -Z
    }
    
    if direction not in direction_vectors:
        print(f"未知方向: {direction}")
        return
    
    # 配置控制器
    controller_config = load_controller_config(default_controller="OSC_POSE")
    
    # 创建环境 - 关键：不开启实时渲染，只开启离线渲染
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,  # 关闭实时渲染窗口
        has_offscreen_renderer=True,  # 开启离线渲染
        use_camera_obs=True,  # 使用相机观察
        camera_names="agentview",  # 相机视角
        camera_heights=512,
        camera_widths=512,
        horizon=1000,
        control_freq=20,
    )
    
    obs = env.reset()
    
    # 获取初始位置
    initial_pos = obs["robot0_eef_pos"].copy()
    target_offset = direction_vectors[direction] * distance
    target_pos = initial_pos + target_offset
    
    print(f"开始录制视频: {video_name}")
    print(f"初始位置: {initial_pos}")
    print(f"向{direction}移动 {distance} 米")
    print(f"目标位置: {target_pos}")
    
    # 存储视频帧
    frames = []
    
    # 执行移动
    for i in range(300):
        current_pos = obs["robot0_eef_pos"]
        pos_error = target_pos - current_pos
        distance_to_target = np.linalg.norm(pos_error)
        
        # 构建动作
        action = np.zeros(env.action_dim)
        action[:3] = pos_error * 5.0  # 位置控制
        action[3:6] = [0, 0, 0]  # 姿态控制
        action[6] = -1.0  # 夹爪打开
        action = np.clip(action, -1, 1)
        
        # 执行动作
        obs, reward, done, info = env.step(action)
        
        # 获取相机图像并添加到帧列表
        frame = obs["agentview_image"][::-1]  # 上下翻转图像
        frames.append(frame)
        
        # 打印进度
        if i % 20 == 0:
            print(f"步骤 {i}: 距离目标 {distance_to_target:.4f} 米, 已录制 {len(frames)} 帧")
        
        # 检查是否到达目标
        if distance_to_target < 0.01:
            print(f"\n到达目标位置！用了 {i} 步")
            actual_movement = current_pos - initial_pos
            print(f"实际移动距离: {np.linalg.norm(actual_movement):.3f} 米")
            
            # 保持位置一段时间继续录制
            for j in range(50):
                action = np.zeros(env.action_dim)
                obs, _, _, _ = env.step(action)
                frame = obs["agentview_image"][::-1]
                frames.append(frame)
            break
        
        if done:
            print("Episode 结束，重置环境")
            obs = env.reset()
    
    env.close()
    
    # 保存视频
    print(f"\n正在保存视频到 {video_name}...")
    imageio.mimsave(video_name, frames, fps=20)
    file_size = os.path.getsize(video_name) / (1024 * 1024)  # MB
    print(f"视频保存成功！")
    print(f"文件大小: {file_size:.2f} MB")
    print(f"总帧数: {len(frames)}")
    print(f"时长: {len(frames)/20:.2f} 秒")


def record_multiple_movements(output_dir="robot_videos"):
    """
    录制多个方向的移动视频
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")
    
    # 定义要录制的移动序列
    movements = [
        ('left', 0.15, 'move_left.mp4'),
        ('right', 0.15, 'move_right.mp4'),
        ('up', 0.1, 'move_up.mp4'),
        ('down', 0.1, 'move_down.mp4'),
        ('forward', 0.1, 'move_forward.mp4'),
        ('backward', 0.1, 'move_backward.mp4'),
    ]
    
    for direction, distance, filename in movements:
        print(f"\n{'='*60}")
        video_path = os.path.join(output_dir, filename)
        move_in_direction_and_record(direction, distance, video_path)
    
    print(f"\n{'='*60}")
    print(f"所有视频已保存到目录: {output_dir}")


def record_complex_task():
    """
    录制一个复杂的任务序列
    """
    controller_config = load_controller_config(default_controller="OSC_POSE")
    
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names="agentview",
        camera_heights=512,
        camera_widths=512,
        horizon=2000,
        control_freq=20,
    )
    
    obs = env.reset()
    frames = []
    
    print("开始录制复杂任务...")
    
    # 任务序列
    task_sequence = [
        ('left', 0.15),
        ('up', 0.1),
        ('forward', 0.1),
        ('down', 0.05),
        ('right', 0.15),
        ('backward', 0.1),
    ]
    
    direction_vectors = {
        'left': np.array([0, 1, 0]),
        'right': np.array([0, -1, 0]),
        'forward': np.array([1, 0, 0]),
        'backward': np.array([-1, 0, 0]),
        'up': np.array([0, 0, 1]),
        'down': np.array([0, 0, -1])
    }
    
    for step_num, (direction, distance) in enumerate(task_sequence):
        print(f"\n步骤 {step_num + 1}: 向{direction}移动 {distance} 米")
        
        initial_pos = obs["robot0_eef_pos"].copy()
        target_pos = initial_pos + direction_vectors[direction] * distance
        
        for i in range(150):
            current_pos = obs["robot0_eef_pos"]
            pos_error = target_pos - current_pos
            distance_to_target = np.linalg.norm(pos_error)
            
            action = np.zeros(env.action_dim)
            action[:3] = pos_error * 5.0
            action[3:6] = [0, 0, 0]
            action[6] = -1.0
            action = np.clip(action, -1, 1)
            
            obs, reward, done, info = env.step(action)
            frame = obs["agentview_image"][::-1]
            frames.append(frame)
            
            if distance_to_target < 0.01:
                print(f"  到达目标！")
                # 短暂停留
                for _ in range(20):
                    action = np.zeros(env.action_dim)
                    obs, _, _, _ = env.step(action)
                    frame = obs["agentview_image"][::-1]
                    frames.append(frame)
                break
            
            if done:
                obs = env.reset()
                break
    
    env.close()
    
    # 保存视频
    video_name = "complex_task.mp4"
    print(f"\n保存视频: {video_name}")
    imageio.mimsave(video_name, frames, fps=20)
    print(f"视频保存成功！总帧数: {len(frames)}, 时长: {len(frames)/20:.2f} 秒")


if __name__ == "__main__":
    print("RoboSuite 视频录制（远程服务器模式）")
    print("="*60)
    print("\n选择录制模式:")
    print("1. 单个方向移动")
    print("2. 多个方向移动（生成多个视频）")
    print("3. 复杂任务序列（生成一个视频）")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == "1":
        print("\n可用方向: left, right, forward, backward, up, down")
        direction = input("输入移动方向: ").strip()
        distance = float(input("输入移动距离(米): ").strip())
        filename = input("输入视频文件名 (默认: movement.mp4): ").strip()
        if not filename:
            filename = "movement.mp4"
        
        move_in_direction_and_record(direction, distance, filename)
        
    elif choice == "2":
        record_multiple_movements()
        
    elif choice == "3":
        record_complex_task()
        
    else:
        print("无效选择，录制复杂任务...")
        record_complex_task()