import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import imageio

def move_to_target_simple(target_position, video_name="move_to_target.mp4"):
    """
    最简单的方法：使用比例控制移动到目标位置
    
    参数:
        target_position: 目标位置 [x, y, z]
        video_name: 保存的视频文件名
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
        horizon=1000,
        control_freq=20,
    )
    
    obs = env.reset()
    frames = []
    
    initial_pos = obs["robot0_eef_pos"].copy()
    target_pos = np.array(target_position)
    
    print(f"初始位置: {initial_pos}")
    print(f"目标位置: {target_pos}")
    print(f"需要移动: {target_pos - initial_pos}")
    print(f"距离: {np.linalg.norm(target_pos - initial_pos):.4f} 米\n")
    
    reached = False
    
    for i in range(500):
        current_pos = obs["robot0_eef_pos"]
        pos_error = target_pos - current_pos
        distance = np.linalg.norm(pos_error)
        
        # 构建 action - 简单比例控制
        action = np.zeros(7)
        action[:3] = pos_error * 5.0  # Kp = 5.0
        action[3:6] = [0, 0, 0]  # 保持姿态
        action[6] = -1.0  # 夹爪打开
        action = np.clip(action, -1, 1)
        
        obs, reward, done, info = env.step(action)
        frame = obs["agentview_image"][::-1]
        frames.append(frame)
        
        if i % 20 == 0:
            print(f"步骤 {i}: 距离 = {distance:.4f} 米, action[:3] = {action[:3]}")
        
        # 判断是否到达
        if distance < 0.01 and not reached:
            print(f"\n✓ 到达目标位置！用了 {i} 步")
            print(f"  最终位置: {current_pos}")
            print(f"  位置误差: {pos_error}")
            reached = True
            
            # 保持位置一段时间
            for j in range(100):
                action = np.zeros(7)
                obs, _, _, _ = env.step(action)
                frames.append(obs["agentview_image"][::-1])
            break
    
    env.close()
    
    # 保存视频
    imageio.mimsave(video_name, frames, fps=20)
    print(f"\n视频已保存: {video_name}")
    print(f"总帧数: {len(frames)}, 时长: {len(frames)/20:.2f} 秒")
    
    return reached

def move_to_target_pd(target_position, video_name="move_to_target_pd.mp4"):
    """
    使用 PD 控制移动到目标位置（更平滑，减少震荡）
    
    P (比例项): 根据位置误差
    D (微分项): 根据速度，提供阻尼
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
        horizon=1000,
        control_freq=20,
    )
    
    obs = env.reset()
    frames = []
    
    target_pos = np.array(target_position)
    initial_pos = obs["robot0_eef_pos"].copy()
    
    print(f"使用 PD 控制")
    print(f"初始位置: {initial_pos}")
    print(f"目标位置: {target_pos}\n")
    
    # PD 控制参数
    Kp = 5.0   # 比例增益
    Kd = 1.0   # 微分增益
    
    prev_pos = initial_pos.copy()
    dt = 1.0 / 20.0  # 控制频率 20Hz
    
    for i in range(500):
        current_pos = obs["robot0_eef_pos"]
        
        # 计算位置误差（P项）
        pos_error = target_pos - current_pos
        distance = np.linalg.norm(pos_error)
        
        # 计算速度（D项）
        velocity = (current_pos - prev_pos) / dt
        
        # PD 控制律
        action = np.zeros(7)
        action[:3] = Kp * pos_error - Kd * velocity
        action[3:6] = [0, 0, 0]
        action[6] = -1.0
        action = np.clip(action, -1, 1)
        
        prev_pos = current_pos.copy()
        
        obs, reward, done, info = env.step(action)
        frames.append(obs["agentview_image"][::-1])
        
        if i % 20 == 0:
            print(f"步骤 {i}: 距离 = {distance:.4f} 米, 速度 = {np.linalg.norm(velocity):.4f} m/s")
        
        if distance < 0.01:
            print(f"\n✓ 到达目标！用了 {i} 步")
            for j in range(100):
                obs, _, _, _ = env.step(np.zeros(7))
                frames.append(obs["agentview_image"][::-1])
            break
    
    env.close()
    imageio.mimsave(video_name, frames, fps=20)
    print(f"视频已保存: {video_name}")
    
    return True

def move_to_target_trajectory(target_position, duration=5.0, video_name="move_to_target_trajectory.mp4"):
    """
    使用轨迹插值移动到目标位置（最平滑的方法）
    
    参数:
        target_position: 目标位置 [x, y, z]
        duration: 移动时长（秒）
        video_name: 视频文件名
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
        horizon=1000,
        control_freq=20,
    )
    
    obs = env.reset()
    frames = []
    
    initial_pos = obs["robot0_eef_pos"].copy()
    target_pos = np.array(target_position)
    
    print(f"使用轨迹插值")
    print(f"初始位置: {initial_pos}")
    print(f"目标位置: {target_pos}")
    print(f"移动时长: {duration} 秒\n")
    
    # 计算总步数
    control_freq = 20  # Hz
    total_steps = int(duration * control_freq)
    
    print(f"总步数: {total_steps}")
    
    for i in range(total_steps):
        # 线性插值计算当前时刻的目标位置
        t = i / total_steps  # 归一化时间 [0, 1]
        
        # 使用平滑的 S 曲线插值（而非线性）
        # 这样加速和减速更平滑
        t_smooth = 3*t**2 - 2*t**3  # 平滑步进函数
        
        current_target = initial_pos + (target_pos - initial_pos) * t_smooth
        
        # 获取当前位置
        current_pos = obs["robot0_eef_pos"]
        
        # 计算到当前目标的误差
        pos_error = current_target - current_pos
        
        # 构建 action
        action = np.zeros(7)
        action[:3] = pos_error * 10.0  # 较大的增益以快速跟踪
        action[3:6] = [0, 0, 0]
        action[6] = -1.0
        action = np.clip(action, -1, 1)
        
        obs, reward, done, info = env.step(action)
        frames.append(obs["agentview_image"][::-1])
        
        if i % 20 == 0:
            distance_to_final = np.linalg.norm(target_pos - current_pos)
            distance_to_current_target = np.linalg.norm(pos_error)
            print(f"步骤 {i}/{total_steps}: "
                  f"到最终目标 = {distance_to_final:.4f} 米, "
                  f"跟踪误差 = {distance_to_current_target:.4f} 米, "
                  f"进度 = {t*100:.1f}%")
    
    # 保持在目标位置
    print("\n保持在目标位置...")
    for j in range(100):
        current_pos = obs["robot0_eef_pos"]
        pos_error = target_pos - current_pos
        action = np.zeros(7)
        action[:3] = pos_error * 5.0
        action = np.clip(action, -1, 1)
        obs, _, _, _ = env.step(action)
        frames.append(obs["agentview_image"][::-1])
    
    final_pos = obs["robot0_eef_pos"]
    final_error = np.linalg.norm(target_pos - final_pos)
    print(f"\n✓ 完成！")
    print(f"  最终位置: {final_pos}")
    print(f"  最终误差: {final_error:.4f} 米")
    
    env.close()
    imageio.mimsave(video_name, frames, fps=20)
    print(f"\n视频已保存: {video_name}")
    
    return True

def move_through_waypoints(waypoints, video_name="waypoint_trajectory.mp4"):
    """
    依次移动到多个路径点
    
    参数:
        waypoints: 路径点列表，例如 [[x1,y1,z1], [x2,y2,z2], ...]
        video_name: 视频文件名
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
    
    initial_pos = obs["robot0_eef_pos"].copy()
    
    print(f"初始位置: {initial_pos}")
    print(f"路径点数量: {len(waypoints)}\n")
    
    for wp_idx, waypoint in enumerate(waypoints):
        target_pos = np.array(waypoint)
        print(f"前往路径点 {wp_idx + 1}/{len(waypoints)}: {target_pos}")
        
        for i in range(300):
            current_pos = obs["robot0_eef_pos"]
            pos_error = target_pos - current_pos
            distance = np.linalg.norm(pos_error)
            
            action = np.zeros(7)
            action[:3] = pos_error * 5.0
            action[3:6] = [0, 0, 0]
            action[6] = -1.0
            action = np.clip(action, -1, 1)
            
            obs, reward, done, info = env.step(action)
            frames.append(obs["agentview_image"][::-1])
            
            if i % 30 == 0:
                print(f"  步骤 {i}: 距离 = {distance:.4f} 米")
            
            if distance < 0.015:  # 到达当前路径点
                print(f"  ✓ 到达路径点 {wp_idx + 1}\n")
                
                # 短暂停留
                for j in range(30):
                    obs, _, _, _ = env.step(np.zeros(7))
                    frames.append(obs["agentview_image"][::-1])
                break
    
    print("✓ 所有路径点完成！")
    
    env.close()
    imageio.mimsave(video_name, frames, fps=20)
    print(f"视频已保存: {video_name}")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("RoboSuite: 移动到目标位置的不同方法")
    print("="*60)
    
    print("\n选择方法:")
    print("1. 简单比例控制 (P控制) - 最简单")
    print("2. PD控制 - 更平滑，减少震荡")
    print("3. 轨迹插值 - 最平滑，可预测")
    print("4. 多点路径 - 依次经过多个位置")
    print("5. 对比所有方法")
    
    choice = input("\n请选择 (1/2/3/4/5): ").strip()
    
    # 定义一个目标位置（相对于初始位置的偏移）
    # 注意：实际目标位置需要在环境重置后根据初始位置计算
    
    if choice == "1":
        print("\n测试：简单比例控制")
        # 这里需要先获取初始位置，然后设置目标
        # 简化起见，我们在函数内部处理
        
        # 示例：移动到相对位置
        controller_config = load_controller_config(default_controller="OSC_POSE")
        env = suite.make("Lift", robots="Panda", controller_configs=controller_config,
                        has_renderer=False, has_offscreen_renderer=True,
                        use_camera_obs=True, camera_names="agentview",
                        camera_heights=512, camera_widths=512)
        obs = env.reset()
        initial = obs["robot0_eef_pos"].copy()
        target = initial + np.array([0.1, 0.15, 0.05])  # 向右前上方移动
        env.close()
        
        move_to_target_simple(target, "method1_simple_p.mp4")
        
    elif choice == "2":
        print("\n测试：PD控制")
        controller_config = load_controller_config(default_controller="OSC_POSE")
        env = suite.make("Lift", robots="Panda", controller_configs=controller_config,
                        has_renderer=False, has_offscreen_renderer=True,
                        use_camera_obs=True, camera_names="agentview")
        obs = env.reset()
        initial = obs["robot0_eef_pos"].copy()
        target = initial + np.array([0.1, 0.15, 0.05])
        env.close()
        
        move_to_target_pd(target, "method2_pd.mp4")
        
    elif choice == "3":
        print("\n测试：轨迹插值")
        controller_config = load_controller_config(default_controller="OSC_POSE")
        env = suite.make("Lift", robots="Panda", controller_configs=controller_config,
                        has_renderer=False, has_offscreen_renderer=True,
                        use_camera_obs=True, camera_names="agentview")
        obs = env.reset()
        initial = obs["robot0_eef_pos"].copy()
        target = initial + np.array([0.1, 0.15, 0.05])
        env.close()
        
        move_to_target_trajectory(target, duration=3.0, video_name="method3_trajectory.mp4")
        
    elif choice == "4":
        print("\n测试：多点路径")
        controller_config = load_controller_config(default_controller="OSC_POSE")
        env = suite.make("Lift", robots="Panda", controller_configs=controller_config,
                        has_renderer=False, has_offscreen_renderer=True,
                        use_camera_obs=True, camera_names="agentview")
        obs = env.reset()
        initial = obs["robot0_eef_pos"].copy()
        
        # 定义一个方形路径
        waypoints = [
            initial + np.array([0.15, 0, 0]),      # 向前
            initial + np.array([0.15, 0.15, 0]),   # 向前向左
            initial + np.array([0, 0.15, 0]),      # 向左
            initial + np.array([0, 0, 0.1]),       # 向上
            initial,                                # 回到起点
        ]
        env.close()
        
        move_through_waypoints(waypoints, "method4_waypoints.mp4")
        
    elif choice == "5":
        print("\n对比所有方法...")
        print("这将生成4个视频文件进行对比")
        
        # ... 运行所有方法 ...
        
    else:
        print("无效选择")