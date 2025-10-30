import logging
import numpy as np
from utils.interfaces import LMP_interface
from utils.visualizers import ValueMapVisualizer
from libero.libero.envs import VoxRenderEnv


def compute_eef_trajectory_from_actions(env, action_chunk):
    """
    通过env.step()获取action chunk对应的末端执行器3D坐标轨迹
    使用env.step()方法, 最准确但可能较慢

    Args:
        env: Libero环境实例
        action_chunk: action chunk, 形状为 (T, 7), 前6维是joint delta, 最后1维是gripper

    Returns:
        eef_positions: 末端执行器3D坐标轨迹, 形状为 (T, 3)
    """
    # 保存当前状态
    current_state = env.sim.get_state()
    current_timestep = env.env.timestep
    eef_positions = []

    try:
        for action in action_chunk:
            # 直接使用env.step()获取最准确的位置
            obs, reward, done, info = env.step(action.tolist())
            if done:
                env.env.done = False  # 重置done标志
            eef_pos = obs["robot0_eef_pos"].copy()
            eef_positions.append(eef_pos)

    finally:
        # 恢复原始状态
        env.regenerate_obs_from_state(current_state)
        env.env.timestep = current_timestep
        env.env.done = False  # 重置done标志

    return np.array(eef_positions)


def test_action_rollback(env, action_chunk, tolerance=1e-3, relaxed_tolerance=0.02):
    """
    测试动作执行回退功能

    1. 保存初始状态
    2. 正向执行整个action chunk
    3. 对每个action取反，反向执行
    4. 对比返回后的状态和动作执行前的状态是否一致

    Args:
        env: Libero环境实例
        action_chunk: action chunk, 形状为 (T, 7), 前6维是joint delta, 最后1维是gripper
        tolerance: 回退测试的容差（米）
        relaxed_tolerance: 宽松容差（米）

    Returns:
        dict: 包含回退测试结果的字典
    """
    # 1. 保存初始状态
    initial_state = env.sim.get_state()
    initial_obs = env.env._get_observations()
    initial_eef_pos = initial_obs["robot0_eef_pos"].copy()
    initial_eef_quat = initial_obs["robot0_eef_quat"].copy()
    initial_gripper_qpos = initial_obs["robot0_gripper_qpos"].copy()

    logging.info("开始动作回退测试...")
    logging.info(f"初始末端执行器位置: {initial_eef_pos}")
    logging.info(f"初始末端执行器四元数: {initial_eef_quat}")
    logging.info(f"初始夹爪位置: {initial_gripper_qpos}")

    try:
        # 2. 正向执行整个action chunk
        logging.info("正向执行action chunk...")
        forward_obs = []
        for i, action in enumerate(action_chunk):
            obs, reward, done, info = env.step(action.tolist())
            forward_obs.append(obs)
            logging.info(f"正向步骤 {i}: 末端执行器位置 {obs['robot0_eef_pos']}")

        # 记录正向执行后的状态
        final_obs = forward_obs[-1]
        final_eef_pos = final_obs["robot0_eef_pos"].copy()
        final_eef_quat = final_obs["robot0_eef_quat"].copy()
        final_gripper_qpos = final_obs["robot0_gripper_qpos"].copy()

        logging.info(f"正向执行后末端执行器位置: {final_eef_pos}")
        logging.info(f"正向执行后末端执行器四元数: {final_eef_quat}")
        logging.info(f"正向执行后夹爪位置: {final_gripper_qpos}")

        # 3. 反向执行（对每个action取反）
        logging.info("反向执行action chunk...")
        reverse_obs = []
        for i, action in enumerate(reversed(action_chunk)):
            # 对action取反：前6维joint delta取反，gripper保持不变
            reverse_action = np.array(action)
            reverse_action[:6] = -reverse_action[:6]  # joint delta取反
            # gripper保持不变，因为gripper动作通常不需要反向

            obs, reward, done, info = env.step(reverse_action.tolist())
            reverse_obs.append(obs)
            logging.info(f"反向步骤 {i}: 末端执行器位置 {obs['robot0_eef_pos']}")

            # 记录每步的累积误差
            if i == 0:  # 第一步
                step_error = np.linalg.norm(obs["robot0_eef_pos"] - initial_eef_pos)
                logging.info(f"反向步骤 {i} 累积位置误差: {step_error:.6f}")

        # 记录反向执行后的状态
        rollback_obs = reverse_obs[-1]
        rollback_eef_pos = rollback_obs["robot0_eef_pos"].copy()
        rollback_eef_quat = rollback_obs["robot0_eef_quat"].copy()
        rollback_gripper_qpos = rollback_obs["robot0_gripper_qpos"].copy()

        logging.info(f"回退后末端执行器位置: {rollback_eef_pos}")
        logging.info(f"回退后末端执行器四元数: {rollback_eef_quat}")
        logging.info(f"回退后夹爪位置: {rollback_gripper_qpos}")

        # 4. 计算状态差异
        eef_pos_error = np.linalg.norm(rollback_eef_pos - initial_eef_pos)
        eef_quat_error = np.linalg.norm(rollback_eef_quat - initial_eef_quat)
        gripper_error = np.linalg.norm(rollback_gripper_qpos - initial_gripper_qpos)

        # 判断是否成功回退（使用传入的容差）
        strict_success = eef_pos_error < tolerance and eef_quat_error < tolerance and gripper_error < tolerance
        relaxed_success = (
            eef_pos_error < relaxed_tolerance and eef_quat_error < relaxed_tolerance and gripper_error < tolerance
        )
        success = strict_success  # 主要使用严格容差

        logging.info("=" * 50)
        logging.info("回退测试结果:")
        logging.info(f"末端执行器位置误差: {eef_pos_error:.6f} (严格容差: {tolerance}, 宽松容差: {relaxed_tolerance})")
        logging.info(
            f"末端执行器四元数误差: {eef_quat_error:.6f} (严格容差: {tolerance}, 宽松容差: {relaxed_tolerance})"
        )
        logging.info(f"夹爪位置误差: {gripper_error:.6f} (容差: {tolerance})")
        logging.info(f"严格回退成功: {'是' if strict_success else '否'}")
        logging.info(f"宽松回退成功: {'是' if relaxed_success else '否'}")

        # 详细分析
        if not success:
            logging.info("=" * 30)
            logging.info("详细误差分析:")
            pos_diff = rollback_eef_pos - initial_eef_pos
            logging.info(f"X轴位置偏差: {pos_diff[0]:.6f}m")
            logging.info(f"Y轴位置偏差: {pos_diff[1]:.6f}m")
            logging.info(f"Z轴位置偏差: {pos_diff[2]:.6f}m")

            quat_diff = rollback_eef_quat - initial_eef_quat
            logging.info(f"四元数偏差: {quat_diff}")

            gripper_diff = rollback_gripper_qpos - initial_gripper_qpos
            logging.info(f"夹爪位置偏差: {gripper_diff}")
            logging.info("=" * 30)

        logging.info("=" * 50)

        return {
            "success": success,
            "strict_success": strict_success,
            "relaxed_success": relaxed_success,
            "eef_pos_error": eef_pos_error,
            "eef_quat_error": eef_quat_error,
            "gripper_error": gripper_error,
            "initial_eef_pos": initial_eef_pos,
            "rollback_eef_pos": rollback_eef_pos,
            "tolerance": tolerance,
            "relaxed_tolerance": relaxed_tolerance,
        }

    except Exception as e:
        logging.error(f"回退测试过程中发生错误: {e}")
        return {
            "success": False,
            "strict_success": False,
            "relaxed_success": False,
            "error": str(e),
            "eef_pos_error": float("inf"),
            "eef_quat_error": float("inf"),
            "gripper_error": float("inf"),
        }

    finally:
        # 恢复初始状态
        env.regenerate_obs_from_state(initial_state)
        logging.info("已恢复初始状态")


def test_state_based_rollback(env, action_chunk, tolerance=1e-3, relaxed_tolerance=0.02):
    """
    基于状态的回退测试功能

    1. 保存初始状态（关节位置和夹爪位置）
    2. 正向执行整个action chunk，记录每步的状态
    3. 直接恢复到初始状态
    4. 对比恢复后的状态和初始状态是否一致

    Args:
        env: Libero环境实例
        action_chunk: action chunk, 形状为 (T, 7), 前6维是joint delta, 最后1维是gripper
        tolerance: 回退测试的容差（米）
        relaxed_tolerance: 宽松容差（米）

    Returns:
        dict: 包含回退测试结果的字典
    """
    # 1. 保存初始状态
    initial_state = env.sim.get_state()
    initial_obs = env.env._get_observations()
    initial_eef_pos = initial_obs["robot0_eef_pos"].copy()
    initial_eef_quat = initial_obs["robot0_eef_quat"].copy()
    initial_gripper_qpos = initial_obs["robot0_gripper_qpos"].copy()
    initial_joint_pos = initial_obs["robot0_joint_pos"].copy()

    logging.info("开始基于状态的回退测试...")
    logging.info(f"初始末端执行器位置: {initial_eef_pos}")
    logging.info(f"初始末端执行器四元数: {initial_eef_quat}")
    logging.info(f"初始夹爪位置: {initial_gripper_qpos}")
    logging.info(f"初始关节位置: {initial_joint_pos}")

    try:
        # 2. 正向执行整个action chunk，记录每步状态
        logging.info("正向执行action chunk并记录状态...")
        forward_states = []
        forward_obs = []

        for i, action in enumerate(action_chunk):
            obs, reward, done, info = env.step(action.tolist())
            forward_obs.append(obs)

            # 记录每步的关节状态
            step_state = {
                "joint_pos": obs["robot0_joint_pos"].copy(),
                "gripper_qpos": obs["robot0_gripper_qpos"].copy(),
                "eef_pos": obs["robot0_eef_pos"].copy(),
                "eef_quat": obs["robot0_eef_quat"].copy(),
            }
            forward_states.append(step_state)

            logging.info(f"正向步骤 {i}: 末端执行器位置 {obs['robot0_eef_pos']}")
            logging.info(f"正向步骤 {i}: 关节位置 {obs['robot0_joint_pos']}")

        # 记录正向执行后的状态
        final_obs = forward_obs[-1]
        final_eef_pos = final_obs["robot0_eef_pos"].copy()
        final_eef_quat = final_obs["robot0_eef_quat"].copy()
        final_gripper_qpos = final_obs["robot0_gripper_qpos"].copy()
        final_joint_pos = final_obs["robot0_joint_pos"].copy()

        logging.info(f"正向执行后末端执行器位置: {final_eef_pos}")
        logging.info(f"正向执行后关节位置: {final_joint_pos}")

        # 3. 直接恢复到初始状态
        logging.info("直接恢复到初始状态...")
        try:
            # 使用restore_robot_joint_positions方法直接恢复
            env.restore_robot_joint_positions(initial_joint_pos, initial_gripper_qpos)

            # 获取恢复后的状态
            restored_obs = env.env._get_observations()
            restored_eef_pos = restored_obs["robot0_eef_pos"].copy()
            restored_eef_quat = restored_obs["robot0_eef_quat"].copy()
            restored_gripper_qpos = restored_obs["robot0_gripper_qpos"].copy()
            restored_joint_pos = restored_obs["robot0_joint_pos"].copy()

            logging.info(f"恢复后末端执行器位置: {restored_eef_pos}")
            logging.info(f"恢复后关节位置: {restored_joint_pos}")

        except Exception as restore_error:
            logging.error(f"状态恢复失败: {restore_error}")
            # 如果直接恢复失败，尝试使用regenerate_obs_from_state
            env.regenerate_obs_from_state(initial_state)

            restored_obs = env.env._get_observations()
            restored_eef_pos = restored_obs["robot0_eef_pos"].copy()
            restored_eef_quat = restored_obs["robot0_eef_quat"].copy()
            restored_gripper_qpos = restored_obs["robot0_gripper_qpos"].copy()
            restored_joint_pos = restored_obs["robot0_joint_pos"].copy()

            logging.info(f"使用sim.set_state恢复后末端执行器位置: {restored_eef_pos}")

        # 4. 计算状态差异
        eef_pos_error = np.linalg.norm(restored_eef_pos - initial_eef_pos)
        eef_quat_error = np.linalg.norm(restored_eef_quat - initial_eef_quat)
        gripper_error = np.linalg.norm(restored_gripper_qpos - initial_gripper_qpos)
        joint_error = np.linalg.norm(restored_joint_pos - initial_joint_pos)

        # 判断是否成功回退
        strict_success = (
            eef_pos_error < tolerance
            and eef_quat_error < tolerance
            and gripper_error < tolerance
            and joint_error < tolerance
        )
        relaxed_success = (
            eef_pos_error < relaxed_tolerance
            and eef_quat_error < relaxed_tolerance
            and gripper_error < tolerance
            and joint_error < tolerance
        )
        success = strict_success

        logging.info("=" * 50)
        logging.info("基于状态的回退测试结果:")
        logging.info(f"末端执行器位置误差: {eef_pos_error:.6f} (严格容差: {tolerance}, 宽松容差: {relaxed_tolerance})")
        logging.info(
            f"末端执行器四元数误差: {eef_quat_error:.6f} (严格容差: {tolerance}, 宽松容差: {relaxed_tolerance})"
        )
        logging.info(f"夹爪位置误差: {gripper_error:.6f} (容差: {tolerance})")
        logging.info(f"关节位置误差: {joint_error:.6f} (容差: {tolerance})")
        logging.info(f"严格回退成功: {'是' if strict_success else '否'}")
        logging.info(f"宽松回退成功: {'是' if relaxed_success else '否'}")

        # 详细分析
        if not success:
            logging.info("=" * 30)
            logging.info("详细误差分析:")
            pos_diff = restored_eef_pos - initial_eef_pos
            logging.info(f"X轴位置偏差: {pos_diff[0]:.6f}m")
            logging.info(f"Y轴位置偏差: {pos_diff[1]:.6f}m")
            logging.info(f"Z轴位置偏差: {pos_diff[2]:.6f}m")

            quat_diff = restored_eef_quat - initial_eef_quat
            logging.info(f"四元数偏差: {quat_diff}")

            gripper_diff = restored_gripper_qpos - initial_gripper_qpos
            logging.info(f"夹爪位置偏差: {gripper_diff}")

            joint_diff = restored_joint_pos - initial_joint_pos
            logging.info(f"关节位置偏差: {joint_diff}")
            logging.info("=" * 30)

        logging.info("=" * 50)

        return {
            "success": success,
            "strict_success": strict_success,
            "relaxed_success": relaxed_success,
            "eef_pos_error": eef_pos_error,
            "eef_quat_error": eef_quat_error,
            "gripper_error": gripper_error,
            "joint_error": joint_error,
            "initial_eef_pos": initial_eef_pos,
            "restored_eef_pos": restored_eef_pos,
            "tolerance": tolerance,
            "relaxed_tolerance": relaxed_tolerance,
        }

    except Exception as e:
        logging.error(f"基于状态的回退测试过程中发生错误: {e}")
        return {
            "success": False,
            "strict_success": False,
            "relaxed_success": False,
            "error": str(e),
            "eef_pos_error": float("inf"),
            "eef_quat_error": float("inf"),
            "gripper_error": float("inf"),
            "joint_error": float("inf"),
        }

    finally:
        # 确保恢复到初始状态
        try:
            env.restore_robot_joint_positions(initial_joint_pos, initial_gripper_qpos)
        except:
            env.regenerate_obs_from_state(initial_state)
        logging.info("已恢复初始状态")


def visualize_action_chunk_trajectory_in_valuemap(env, action_chunk, task_description, target_objects, avoid_objects):
    """
    在valuemap中可视化action chunk轨迹

    Args:
        env: Libero环境实例
        action_chunk: action chunk, 形状为 (T, 7), 前6维是joint delta, 最后1维是gripper
        task_description: 任务描述
        target_objects: 目标对象列表
        avoid_objects: 避免对象列表

    Returns:
        dict: 包含轨迹可视化信息的字典
    """
    # 保存当前状态
    current_state = env.sim.get_state()
    current_obs = env.env._get_observations()
    current_eef_pos = current_obs["robot0_eef_pos"].copy()

    try:
        # 计算action chunk对应的末端执行器轨迹
        eef_trajectory = compute_eef_trajectory_from_actions(env, action_chunk)

        # 简化的轨迹可视化 - 直接返回轨迹信息
        info = {
            "action_chunk_trajectory": eef_trajectory,
            "action_chunk_length": len(action_chunk),
            "trajectory_start_pos": current_eef_pos,
            "trajectory_end_pos": eef_trajectory[-1] if len(eef_trajectory) > 0 else current_eef_pos,
            "trajectory_points": len(eef_trajectory),
            "task_description": task_description,
            "target_objects": target_objects,
            "avoid_objects": avoid_objects,
        }

        logging.info(f"Action chunk轨迹可视化完成，轨迹长度: {len(eef_trajectory)}")
        logging.info(f"起始位置: {current_eef_pos}")
        logging.info(f"结束位置: {eef_trajectory[-1] if len(eef_trajectory) > 0 else current_eef_pos}")
        logging.info(f"目标对象: {target_objects}")
        logging.info(f"避免对象: {avoid_objects}")

        return info

    except Exception as e:
        logging.error(f"Action chunk轨迹可视化过程中发生错误: {e}")
        return {"error": str(e), "action_chunk_trajectory": [], "action_chunk_length": 0}

    finally:
        # 恢复原始状态
        env.regenerate_obs_from_state(current_state)
        logging.info("已恢复原始状态")


def build_value_map_with_clone(env, task_description, target_objects=None, avoid_objects=None, eef_traj=None):
    """
    构建valuemap功能，先从当前环境中clone一个环境

    Args:
        env: 主环境实例
        task_description: 任务描述
        target_objects: 目标对象列表（可选，如果为None则自动获取）
        avoid_objects: 避免对象列表（可选，如果为None则自动获取）

    Returns:
        dict: 包含valuemap构建结果的字典
    """
    # 保存当前环境状态
    current_state = env.sim.get_state()
    current_obs = env.env._get_observations()

    try:
        # 创建VoxRender环境用于valuemap构建
        # 参考复制的代码实现
        visualizer = ValueMapVisualizer()

        # 获取环境参数
        env_args = {
            "bddl_file_name": getattr(env.env, "bddl_file_name", None),
            "camera_heights": getattr(env.env.camera_heights[0], "camera_heights", 256),
            "camera_widths": getattr(env.env.camera_widths[0], "camera_widths", 256),
        }

        # 创建VoxRender环境
        env_vox = VoxRenderEnv(visualizer=visualizer, **env_args)

        # 拷贝主环境状态到VoxRender环境
        sim_state = env.get_sim_state()
        start_obs = env_vox.reset(sim_state)

        # 获取目标对象和避免对象
        objects_of_interest = env_vox.obj_of_interest if hasattr(env_vox, "obj_of_interest") else []
        object_names = env_vox.get_object_names() if hasattr(env_vox, "get_object_names") else []
        target_objects = objects_of_interest[:1] if objects_of_interest else []
        avoid_objects = [obj for obj in object_names if obj not in target_objects]

        # 构建valuemap
        lmp_env = LMP_interface(env_vox, task_description)
        movable_gripper, gripper_map, affordance_map, avoidance_map = lmp_env.build_value_map(
            target_objects, avoid_objects
        )
        info = lmp_env.execute(
            eef_traj,
            movable_gripper,
            affordance_map=affordance_map,
            avoidance_map=avoidance_map,
            gripper_map=gripper_map,
        )

        # 准备返回信息
        valuemap_info = {
            "movable_gripper": movable_gripper,
            "gripper_map": gripper_map,
            "affordance_map": affordance_map,
            "avoidance_map": avoidance_map,
            "target_objects": target_objects,
            "avoid_objects": avoid_objects,
            "task_description": task_description,
            "map_size": lmp_env._map_size,
            "resolution": lmp_env._resolution,
        }

        logging.info(f"ValueMap构建完成:")
        logging.info(f"  目标对象: {target_objects}")
        logging.info(f"  避免对象: {avoid_objects}")
        logging.info(f"  地图大小: {lmp_env._map_size}")
        logging.info(f"  分辨率: {lmp_env._resolution}")
        logging.info(f"  当前末端执行器位置: {current_obs['robot0_eef_pos']}")

        return valuemap_info

    except Exception as e:
        logging.error(f"ValueMap构建过程中发生错误: {e}")
        return {
            "error": str(e),
            "movable_gripper": None,
            "gripper_map": None,
            "affordance_map": None,
            "avoidance_map": None,
        }

    finally:
        # 恢复原始环境状态
        env.regenerate_obs_from_state(current_state)
        logging.info("已恢复原始环境状态")


def build_value_map_with_trajectory_evaluation(
    env, task_description, target_objects, avoid_objects, action_chunk=None, trajectory_info=None
):
    """
    构建valuemap并评估轨迹

    Args:
        env: 主环境实例
        task_description: 任务描述
        target_objects: 目标对象列表
        avoid_objects: 避免对象列表
        action_chunk: 可选的action chunk
        trajectory_info: 可选的轨迹信息

    Returns:
        dict: 包含valuemap构建和轨迹评估结果的字典
    """
    # 保存当前环境状态
    current_state = env.sim.get_state()
    current_obs = env.env._get_observations()

    try:
        # 构建valuemap
        valuemap_info = build_value_map_with_clone(env, task_description, target_objects, avoid_objects)

        if "error" in valuemap_info:
            return valuemap_info

        # 如果有action chunk，计算轨迹
        if action_chunk is not None:
            eef_trajectory = compute_eef_trajectory_from_actions(env, action_chunk)
            valuemap_info["action_chunk_trajectory"] = eef_trajectory
            valuemap_info["action_chunk_length"] = len(action_chunk)
            valuemap_info["trajectory_start_pos"] = current_obs["robot0_eef_pos"]
            valuemap_info["trajectory_end_pos"] = (
                eef_trajectory[-1] if len(eef_trajectory) > 0 else current_obs["robot0_eef_pos"]
            )

            logging.info(f"轨迹计算完成，轨迹长度: {len(eef_trajectory)}")

        # 如果有轨迹信息，添加到结果中
        if trajectory_info is not None:
            valuemap_info["trajectory_info"] = trajectory_info

        return valuemap_info

    except Exception as e:
        logging.error(f"ValueMap构建和轨迹评估过程中发生错误: {e}")
        return {
            "error": str(e),
            "movable_gripper": None,
            "gripper_map": None,
            "affordance_map": None,
            "avoidance_map": None,
        }

    finally:
        # 恢复原始环境状态
        env.regenerate_obs_from_state(current_state)
        logging.info("已恢复原始环境状态")


def build_reusable_value_map(env, task_description, target_objects=None, avoid_objects=None):
    """
    构建可重复使用的valuemap（不包含gripper_map，因为gripper_map会变化）

    Args:
        env: 主环境实例
        task_description: 任务描述
        target_objects: 目标对象列表（可选，如果为None则自动获取）
        avoid_objects: 避免对象列表（可选，如果为None则自动获取）

    Returns:
        dict: 包含可重复使用的valuemap信息
    """
    # 保存当前环境状态
    current_state = env.sim.get_state()
    current_obs = env.env._get_observations()

    try:
        # 创建VoxRender环境用于valuemap构建
        visualizer = ValueMapVisualizer()

        # 获取环境参数
        env_args = {
            "bddl_file_name": getattr(env.env, "bddl_file_name", None),
            "camera_heights": getattr(env.env.camera_heights[0], "camera_heights", 256),
            "camera_widths": getattr(env.env.camera_widths[0], "camera_widths", 256),
        }

        # 创建VoxRender环境
        env_vox = VoxRenderEnv(visualizer=visualizer, **env_args)

        # 拷贝主环境状态到VoxRender环境
        sim_state = env.get_sim_state()
        start_obs = env_vox.reset(sim_state)

        # 获取目标对象和避免对象
        objects_of_interest = env_vox.obj_of_interest if hasattr(env_vox, "obj_of_interest") else []
        object_names = env_vox.get_object_names() if hasattr(env_vox, "get_object_names") else []
        target_objects = objects_of_interest[:1] if objects_of_interest else []
        avoid_objects = [obj for obj in object_names if obj not in target_objects]

        # 构建valuemap（不包含gripper_map）
        lmp_env = LMP_interface(env_vox, task_description)
        movable_gripper, _, affordance_map, avoidance_map = lmp_env.build_value_map(target_objects, avoid_objects)

        # 准备返回信息（不包含gripper_map）
        valuemap_info = {
            "movable_gripper": movable_gripper,
            "affordance_map": affordance_map,
            "avoidance_map": avoidance_map,
            "target_objects": target_objects,
            "avoid_objects": avoid_objects,
            "task_description": task_description,
            "map_size": lmp_env._map_size,
            "resolution": lmp_env._resolution,
            "lmp_env": lmp_env,  # 保存lmp_env实例以便后续使用
            "env_vox": env_vox,  # 保存环境实例以便后续使用
        }

        logging.info(f"可重复使用ValueMap构建完成:")
        logging.info(f"  目标对象: {target_objects}")
        logging.info(f"  避免对象: {avoid_objects}")
        logging.info(f"  地图大小: {lmp_env._map_size}")
        logging.info(f"  分辨率: {lmp_env._resolution}")
        logging.info(f"  当前末端执行器位置: {current_obs['robot0_eef_pos']}")

        return valuemap_info

    except Exception as e:
        logging.error(f"可重复使用ValueMap构建过程中发生错误: {e}")
        return {
            "error": str(e),
            "movable_gripper": None,
            "affordance_map": None,
            "avoidance_map": None,
        }

    finally:
        # 恢复原始环境状态
        env.regenerate_obs_from_state(current_state)
        logging.info("已恢复原始环境状态")


def evaluate_trajectory_with_value_map(valuemap_info, eef_traj, current_env=None):
    """
    使用已构建的valuemap评估轨迹

    Args:
        valuemap_info: 已构建的valuemap信息
        eef_traj: 末端执行器轨迹
        current_env: 当前环境实例（可选，用于获取当前gripper状态）

    Returns:
        dict: 包含轨迹评估结果的信息
    """
    if "error" in valuemap_info:
        return {"error": "ValueMap构建失败，无法评估轨迹"}

    try:
        lmp_env = valuemap_info["lmp_env"]
        affordance_map = valuemap_info["affordance_map"]
        avoidance_map = valuemap_info["avoidance_map"]

        # 获取当前的gripper状态
        if current_env is not None:
            # 使用当前环境获取gripper状态
            current_gripper_pos = current_env.get_ee_pos()
            current_gripper_vox = lmp_env._world_to_voxel(current_gripper_pos)
            movable_gripper = {
                "name": "gripper",
                "position": current_gripper_vox,
                "aabb": np.array([current_gripper_vox, current_gripper_vox]),
                "_position_world": current_gripper_pos,
            }
        else:
            # 如果没有提供当前环境，使用初始状态
            movable_gripper = valuemap_info["movable_gripper"]
            logging.warning("未提供当前环境，使用初始gripper状态")

        # 构建当前的gripper_map（因为gripper状态会变化）
        gripper_map = lmp_env.get_empty_gripper_map()
        gripper_map[:, :, :] = 1  # 默认开启

        # 使用lmp_env的execute方法评估轨迹
        step_info = lmp_env.execute(
            eef_traj,
            movable_gripper,
            affordance_map=affordance_map,
            avoidance_map=avoidance_map,
            gripper_map=gripper_map,
        )

        return {"step_info": step_info, "trajectory_cost": step_info.get("costmap", None), "success": True}

    except Exception as e:
        logging.error(f"轨迹评估过程中发生错误: {e}")
        return {"error": str(e), "success": False}
