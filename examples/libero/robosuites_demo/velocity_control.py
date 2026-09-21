import numpy as np
import robosuite as suite
from robosuite.controllers import load_controller_config
import imageio

def explain_velocity_control():
    """
    详细解释关节速度控制的工作原理
    """
    
    print("="*70)
    print("关节速度控制详解")
    print("="*70)
    
    # 创建环境，注意 control_freq
    controller_config = load_controller_config(default_controller="JOINT_VELOCITY")
    
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
        control_freq=20,  # ← 关键！每秒20个控制周期
    )
    
    obs = env.reset()

    print("\n【关键参数】")
    print(f"control_freq = 20 Hz")
    print(f"控制周期 Δt = 1/20 = 0.05 秒")
    print(f"即：每次 env.step() 执行 0.05 秒的仿真")

    print("\n【速度控制原理】")
    print("┌─────────────────────────────────────────────────┐")
    print("│ action[i] = 关节i的目标角速度 (弧度/秒)         │")
    print("│                                                  │")
    print("│ 每次 step():                                    │")
    print("│   关节转动角度 = 速度 × Δt                      │")
    print("│                = action[i] × 0.05               │")
    print("└─────────────────────────────────────────────────┘")

    # 示例1：恒定速度旋转
    print("\n" + "="*70)
    print("示例1：关节2以 1.0 rad/s 的速度旋转")
    print("="*70)

    # robosuite使用sin/cos编码关节位置，需要解码
    def decode_joint_pos(obs):
        joint_pos_cos = obs["robot0_joint_pos_cos"]
        joint_pos_sin = obs["robot0_joint_pos_sin"]
        return np.arctan2(joint_pos_sin, joint_pos_cos)

    initial_joint_pos = decode_joint_pos(obs)
    initial_joint2 = initial_joint_pos[1]
    print(f"\n初始位置: {initial_joint2:.4f} 弧度")
    
    target_velocity = 0.1  # rad/s
    print(f"目标速度: {target_velocity} 弧度/秒")
    print(f"\n理论上每次 step():")
    print(f"  转动角度 = {target_velocity} × 0.05 = {target_velocity * 0.05:.4f} 弧度")
    print(f"  转动角度 = {target_velocity * 0.05 * 180 / np.pi:.2f} 度")
    
    frames = []
    positions = []
    
    print("\n开始旋转 100 步 (5秒)...")
    print("\n步骤    时间(s)    关节位置(rad)    实际转动(rad)    累计转动(rad)")
    print("-" * 75)
    
    for i in range(100):
        # 构建action：只让关节2旋转
        action = np.zeros(8)
        action[1] = target_velocity  # 关节2的速度
        action[7] = -1.0  # 夹爪打开
        
        # 执行一步
        obs, reward, done, info = env.step(action)

        current_joint_pos = decode_joint_pos(obs)
        current_joint2 = current_joint_pos[1]
        positions.append(current_joint2)
        
        # 录制视频帧
        frame = obs["agentview_image"][::-1]
        frames.append(frame)
        
        # 每10步打印一次
        if i % 10 == 0:
            time_elapsed = i * 0.05
            total_rotation = current_joint2 - initial_joint2
            step_rotation = positions[-1] - positions[-2] if i > 0 else 0
            
            print(f"{i:3d}     {time_elapsed:5.2f}      {current_joint2:8.4f}         "
                  f"{step_rotation:8.4f}          {total_rotation:8.4f}")

    final_joint_pos = decode_joint_pos(obs)
    final_joint2 = final_joint_pos[1]
    total_rotation = final_joint2 - initial_joint2
    expected_rotation = target_velocity * (100 * 0.05)  # 速度 × 总时间
    
    print("-" * 75)
    print(f"\n【结果分析】")
    print(f"初始位置:     {initial_joint2:.4f} 弧度")
    print(f"最终位置:     {final_joint2:.4f} 弧度")
    print(f"实际转动:     {total_rotation:.4f} 弧度 = {total_rotation * 180 / np.pi:.2f} 度")
    print(f"理论转动:     {expected_rotation:.4f} 弧度 = {expected_rotation * 180 / np.pi:.2f} 度")
    print(f"总时间:       {100 * 0.05:.2f} 秒")
    print(f"平均速度:     {total_rotation / (100 * 0.05):.4f} 弧度/秒")
    
    env.close()
    imageio.mimsave("velocity_control_constant.mp4", frames, fps=20)
    print(f"\n视频已保存: velocity_control_constant.mp4")
    
    return positions


def demonstrate_speed_change():
    """
    演示：改变速度会如何影响运动
    """

    print("\n" + "="*70)
    print("示例2：速度变化演示")
    print("="*70)

    controller_config = load_controller_config(default_controller="JOINT_VELOCITY")

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

    # robosuite使用sin/cos编码关节位置，需要解码
    def decode_joint_pos(obs):
        joint_pos_cos = obs["robot0_joint_pos_cos"]
        joint_pos_sin = obs["robot0_joint_pos_sin"]
        return np.arctan2(joint_pos_sin, joint_pos_cos)

    initial_joint_pos = decode_joint_pos(obs)
    initial_joint2 = initial_joint_pos[1]
    
    print("\n阶段1 (0-2秒): 速度 = 0.5 rad/s")
    print("阶段2 (2-4秒): 速度 = 1.0 rad/s  (加速)")
    print("阶段3 (4-6秒): 速度 = 0.0 rad/s  (停止)")
    print("阶段4 (6-8秒): 速度 = -0.5 rad/s (反向)")
    
    print("\n时间(s)  速度(rad/s)  位置(rad)  说明")
    print("-" * 60)
    
    for i in range(160):  # 8秒 = 160步
        time = i * 0.05
        
        # 根据时间设置不同速度
        if time < 2.0:
            velocity = 0.5
            stage = "慢速正转"
        elif time < 4.0:
            velocity = 1.0
            stage = "快速正转"
        elif time < 6.0:
            velocity = 0.0
            stage = "停止"
        else:
            velocity = -0.5
            stage = "慢速反转"
        
        action = np.zeros(8)
        action[1] = velocity
        action[7] = -1.0
        
        obs, reward, done, info = env.step(action)
        current_joint_pos = decode_joint_pos(obs)
        current_joint2 = current_joint_pos[1]
        
        frame = obs["agentview_image"][::-1]
        frames.append(frame)
        
        if i % 20 == 0:  # 每1秒打印一次
            print(f"{time:5.2f}    {velocity:6.2f}      {current_joint2:7.4f}    {stage}")

    final_joint_pos = decode_joint_pos(obs)
    final_joint2 = final_joint_pos[1]
    
    print("-" * 60)
    print(f"\n最终位置: {final_joint2:.4f} 弧度")
    print(f"相对初始位置移动: {final_joint2 - initial_joint2:.4f} 弧度")
    
    print("\n【理论计算】")
    print("阶段1: 0.5 × 2.0 = 1.0 弧度")
    print("阶段2: 1.0 × 2.0 = 2.0 弧度")
    print("阶段3: 0.0 × 2.0 = 0.0 弧度")
    print("阶段4: -0.5 × 2.0 = -1.0 弧度")
    print("总计: 1.0 + 2.0 + 0.0 - 1.0 = 2.0 弧度")
    
    env.close()
    imageio.mimsave("velocity_control_varying.mp4", frames, fps=20)
    print(f"\n视频已保存: velocity_control_varying.mp4")


def demonstrate_position_via_velocity():
    """
    演示：用速度控制实现位置控制
    """
    
    print("\n" + "="*70)
    print("示例3：用速度控制实现'旋转到指定位置'")
    print("="*70)
    
    controller_config = load_controller_config(default_controller="JOINT_VELOCITY")
    
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

    # robosuite使用sin/cos编码关节位置，需要解码
    def decode_joint_pos(obs):
        joint_pos_cos = obs["robot0_joint_pos_cos"]
        joint_pos_sin = obs["robot0_joint_pos_sin"]
        return np.arctan2(joint_pos_sin, joint_pos_cos)

    initial_joint_pos = decode_joint_pos(obs)
    initial_joint2 = initial_joint_pos[1]
    target_position = initial_joint2 + 1.57  # 旋转90度 (π/2)
    
    print(f"\n目标：从 {initial_joint2:.4f} 旋转到 {target_position:.4f} 弧度")
    print(f"即旋转 {(target_position - initial_joint2) * 180 / np.pi:.1f} 度")
    
    print("\n策略：使用比例控制")
    print("  速度 = Kp × (目标位置 - 当前位置)")
    print("  Kp = 2.0")
    
    print("\n步骤  当前位置  距离目标  速度命令  说明")
    print("-" * 70)
    
    Kp = 2.0  # 比例增益
    
    for i in range(200):
        current_joint_pos = decode_joint_pos(obs)
        current_joint2 = current_joint_pos[1]
        error = target_position - current_joint2
        distance = abs(error)
        
        # 比例控制：速度正比于误差
        velocity = Kp * error
        
        # 限制最大速度
        max_velocity = 1.0
        velocity = np.clip(velocity, -max_velocity, max_velocity)
        
        action = np.zeros(8)
        action[1] = velocity
        action[7] = -1.0
        
        obs, reward, done, info = env.step(action)
        
        frame = obs["agentview_image"][::-1]
        frames.append(frame)
        
        if i % 20 == 0:
            status = "接近中..." if distance > 0.01 else "已到达！"
            print(f"{i:3d}   {current_joint2:7.4f}   {distance:7.4f}   {velocity:7.4f}   {status}")
        
        # 到达目标
        if distance < 0.01:
            print(f"\n到达目标！用了 {i} 步 ({i * 0.05:.2f} 秒)")
            # 保持位置
            for _ in range(40):
                action = np.zeros(8)
                action[1] = 0.0  # 速度为0
                action[7] = -1.0
                obs, _, _, _ = env.step(action)
                frame = obs["agentview_image"][::-1]
                frames.append(frame)
            break
    
    env.close()
    imageio.mimsave("velocity_to_position.mp4", frames, fps=20)
    print(f"\n视频已保存: velocity_to_position.mp4")


def compare_control_frequencies():
    """
    演示：control_freq 如何影响速度控制
    """
    
    print("\n" + "="*70)
    print("示例4：control_freq 的影响")
    print("="*70)
    
    print("\n实验：相同速度命令，不同控制频率")
    
    for freq in [10, 20, 50]:
        print(f"\n--- control_freq = {freq} Hz ---")
        print(f"控制周期 Δt = {1/freq:.4f} 秒")
        
        controller_config = load_controller_config(default_controller="JOINT_VELOCITY")
        
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
            control_freq=freq,
        )
        
        obs = env.reset()

        # robosuite使用sin/cos编码关节位置，需要解码
        def decode_joint_pos(obs):
            joint_pos_cos = obs["robot0_joint_pos_cos"]
            joint_pos_sin = obs["robot0_joint_pos_sin"]
            return np.arctan2(joint_pos_sin, joint_pos_cos)

        initial_pos = decode_joint_pos(obs)[1]
        
        velocity = 1.0  # rad/s
        num_steps = freq  # 执行1秒
        
        print(f"速度命令: {velocity} rad/s")
        print(f"执行时间: 1 秒 ({num_steps} 步)")
        print(f"每步转动: {velocity / freq:.4f} 弧度")
        
        for i in range(num_steps):
            action = np.zeros(8)
            action[1] = velocity
            action[7] = -1.0
            obs, _, _, _ = env.step(action)

        final_pos = decode_joint_pos(obs)[1]
        actual_rotation = final_pos - initial_pos
        
        print(f"理论转动: {velocity * 1.0:.4f} 弧度")
        print(f"实际转动: {actual_rotation:.4f} 弧度")
        print(f"误差: {abs(actual_rotation - velocity):.6f} 弧度")
        
        env.close()


if __name__ == "__main__":
    print("\n关节速度控制完全指南")
    print("="*70)
    
    print("\n核心要点：")
    print("1. 速度控制不是'转多久'，而是'每个控制周期转多少'")
    print("2. 每次 env.step() = 一个控制周期")
    print("3. 控制周期长度 = 1 / control_freq")
    print("4. 转动角度 = 速度 × 控制周期")
    
    print("\n" + "="*70)
    print("选择演示:")
    print("1. 恒定速度旋转 (理解基本原理)")
    print("2. 变化速度 (理解速度如何影响运动)")
    print("3. 用速度实现位置控制 (实用技巧)")
    print("4. control_freq 的影响")
    print("5. 运行所有演示")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == "1":
        explain_velocity_control()
    elif choice == "2":
        demonstrate_speed_change()
    elif choice == "3":
        demonstrate_position_via_velocity()
    elif choice == "4":
        compare_control_frequencies()
    else:
        print("\n运行所有演示...\n")
        explain_velocity_control()
        demonstrate_speed_change()
        demonstrate_position_via_velocity()
        compare_control_frequencies()