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


def build_reusable_value_map(env, task_description, stage:int = 0, object_path = "/hdd/zijianwang/openpi/third_party/LIBERO-PRO/reordered_objects_with_descriptions.json"):
    """
    构建可重复使用的valuemap（不包含gripper_map，因为gripper_map会变化）

    Args:
        env: 主环境实例
        task_description: 任务描述
        object_path: JSON文件的路径

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
        all_target_objects = get_reordered_objects(object_path, task_description)
        target_objects = all_target_objects[stage:stage+1] if stage < len(all_target_objects) else all_target_objects[:1]
        logging.info(f"**All target objects**: {all_target_objects}")
        logging.info(f"**Using target object in stage {stage}**: {target_objects}")
        # target_objects = objects_of_interest[:1] if objects_of_interest else []
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

        logging.info(f"ValueMap构建完成: 目标对象: {target_objects}, 避免对象: {avoid_objects}")
        # logging.info(f"  地图大小: {lmp_env._map_size}")
        # logging.info(f"  分辨率: {lmp_env._resolution}")
        # logging.info(f"  当前末端执行器位置: {current_obs['robot0_eef_pos']}")

        return valuemap_info

    except Exception as e:
        logging.error(f"ValueMap构建过程中发生错误: {e}")
        return {
            "error": str(e),
            "movable_gripper": None,
            "affordance_map": None,
            "avoidance_map": None,
        }

    finally:
        # 恢复原始环境状态
        env.regenerate_obs_from_state(current_state)
        logging.info("ValueMap构建完成, 已恢复原始环境状态")


def setup_task_valuemap(env, task_description, logger=None):
    """
    为任务设置可重复使用的valuemap

    这是一个高级包装函数，封装了构建reusable_valuemap的完整流程，包括：
    - ValueMap构建
    - 日志记录
    - 错误处理

    注意：调用者需要在调用此函数前已经将环境设置到所需状态

    Args:
        env: 主环境实例（调用前已设置到所需状态）
        task_description: 任务描述
        logger: 日志记录器（可选），如果为None则使用logging模块

    Returns:
        dict: 可重复使用的valuemap信息，包含以下键：
            - error: 错误信息（仅在失败时存在）
            - target_objects: 目标对象列表
            - avoid_objects: 避免对象列表
            - movable_gripper: 可移动的gripper状态
            - affordance_map: affordance map
            - avoidance_map: avoidance map
            - map_size: 地图大小
            - resolution: 分辨率
            - lmp_env: LMP环境实例
            - env_vox: VoxRender环境实例
        None: 在构建失败时返回None
    """
    if logger is None:
        logger = logging

    logger.info("Building reusable ValueMap...")

    reusable_valuemap = None
    try:
        # 构建可重复使用的valuemap
        reusable_valuemap = build_reusable_value_map(env, task_description)

        if "error" not in reusable_valuemap:
            logger.info(f"Reusable ValueMap built successfully:")
            logger.info(f"  Target objects: {reusable_valuemap.get('target_objects', [])}")
            logger.info(f"  Avoid objects: {reusable_valuemap.get('avoid_objects', [])}")
            logger.info(f"  Map size: {reusable_valuemap.get('map_size', 'N/A')}")
            logger.info(f"  Resolution: {reusable_valuemap.get('resolution', 'N/A')}")
        else:
            logger.warning(f"Failed to build reusable ValueMap: {reusable_valuemap.get('error', 'Unknown error')}")
            reusable_valuemap = None

    except Exception as e:
        logger.warning(f"Failed to build reusable ValueMap: {e}")
        reusable_valuemap = None

    return reusable_valuemap


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

def get_reordered_objects(json_file_path, task_description):
    """
    根据任务描述从JSON文件中获取重新排序的对象列表
    
    Args:
        json_file_path (str): JSON文件的路径
        task_description (str): 任务描述
        
    Returns:
        list: 重新排序的对象列表，如果未找到则返回None
    """
    import json
    
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 遍历所有任务套件
        for suite_name, tasks in data.items():
            for task_id, task_data in tasks.items():
                if task_data['task_description'] == task_description:
                    return task_data['reordered_objects']
        
        # 如果没有找到匹配的任务描述
        print(f"未找到任务描述: {task_description}")
        return None
        
    except FileNotFoundError:
        print(f"文件未找到: {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"JSON文件格式错误: {json_file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return None


def is_gripper_closed(obs, tolerance=0.005):
    """
    根据 robosuite 的 observation 中 'robot0_gripper_qpos' 的值判断夹爪是否闭合。

    Args:
        obs (Dict[str, Any]): 来自 env.step() 的观察结果字典。
        tolerance (float): 判断夹爪是否闭合时的容差阈值。

    Returns:
        bool: 如果夹爪被认为是闭合的，则返回 True，否则返回 False。
    """
    if 'robot0_gripper_qpos' not in obs:
        raise KeyError("Observation dictionary does not contain 'robot0_gripper_qpos'.")

    gripper_qpos = obs['robot0_gripper_qpos']
    
    # 假设夹爪是双指且左右关节位置大小相等、符号相反。
    # 我们可以计算两个关节位置的绝对值之和，或计算它们之间的距离。
    # 当夹爪闭合时，关节位置接近于0。
    # 这里的 total_closure 是一个简化的度量，表示夹爪的闭合程度。
    total_closure = np.sum(np.abs(gripper_qpos))
    
    # 如果总闭合度小于容差，则认为夹爪已闭合。
    return total_closure < tolerance


def detect_pre_grasp_state(
    t, 
    linear_speed_trajectory, 
    gripper_state_trajectory,
    speed_percentage_threshold=0.3,
    low_speed_window=1,
    gripper_change_lookahead=5,
    min_history_window=10
):
    """
    检测机械臂的抓取状态（基于历史信息）。
    
    本函数可以检测两种状态：
    1. 即将抓取状态 (is_pre_grasp)：
       - 机械臂减速到低速
       - 低速状态持续一段时间
       - 夹爪刚刚闭合
    
    2. 稳定抓取并准备移动状态 (is_stable_grasp_and_moving)：
       - 夹爪已连续闭合N个时刻
       - 速度在最近N步内呈递增趋势（加速离开抓取点）
    
    注意：本函数只使用t时刻及之前的历史信息，不依赖未来信息。
    
    Args:
        t (int): 当前时间步
        linear_speed_trajectory (list or np.ndarray): 线速度轨迹，每个元素为标量速度值
        gripper_state_trajectory (list or np.ndarray): 夹爪状态轨迹，每个元素为布尔值
            (True表示闭合，False表示打开)
        speed_percentage_threshold (float): 判断为"低速"的速度百分比阈值（默认0.3，即30%）
            当前速度小于历史最高速度的这个百分比时，判断为低速
        low_speed_window (int): 需要保持低速的最小时间窗口（默认1步）
        gripper_change_lookahead (int): 向后回溯检查夹爪状态变化的时间窗口（默认5步）
        min_history_window (int): 计算历史最高速度的最小窗口大小（默认10步）
    
    Returns:
        dict: 包含以下键的检测结果字典:
            - is_pre_grasp (bool): 是否处于即将抓取状态
            - pre_grasp_confidence (float): 即将抓取的置信度 [0.0, 1.0]
            - is_stable_grasp_and_moving (bool): 是否处于稳定抓取并准备移动状态
            - stable_grasp_confidence (float): 稳定抓取的置信度 [0.0, 1.0]
            - details (dict): 详细信息
                速度相关：
                - current_speed (float): 当前速度
                - max_historical_speed (float): 历史最高速度
                - speed_threshold (float): 实际使用的速度阈值
                - speed_percentage (float): 当前速度占历史最高速度的百分比
                - is_low_speed (bool): 当前是否为低速
                - low_speed_duration (int): 连续低速的持续时间
                
                夹爪状态相关：
                - current_gripper_state (bool): 当前夹爪状态
                - gripper_just_closed (bool): 夹爪是否刚刚闭合
                - gripper_closed_steps_ago (int): 夹爪在多少步之前闭合（-1表示未闭合）
                - gripper_closed_duration (int): 夹爪连续闭合的持续时间
                
                稳定抓取相关：
                - speed_increasing (bool): 速度是否在递增
                - speed_increase_ratio (float): 速度递增的比例
    """
    # 输入验证
    if t < 0:
        return {
            "is_pre_grasp": False,
            "pre_grasp_confidence": 0.0,
            "is_stable_grasp_and_moving": False,
            "stable_grasp_confidence": 0.0,
            "details": {"error": "Invalid timestep"}
        }
    
    if t >= len(linear_speed_trajectory) or t >= len(gripper_state_trajectory):
        return {
            "is_pre_grasp": False,
            "pre_grasp_confidence": 0.0,
            "is_stable_grasp_and_moving": False,
            "stable_grasp_confidence": 0.0,
            "details": {"error": "Timestep out of bounds"}
        }
    
    # 转换为numpy数组便于操作
    speeds = np.array(linear_speed_trajectory)
    grippers = np.array(gripper_state_trajectory)
    
    # 1. 计算历史最高速度和动态速度阈值
    current_speed = speeds[t]
    
    # 获取历史速度数据（从0到当前时刻t）
    history_start = max(0, t - min_history_window)
    historical_speeds = speeds[history_start:t+1]
    
    # 计算历史最高速度
    if len(historical_speeds) > 0:
        max_historical_speed = np.max(historical_speeds)
    else:
        max_historical_speed = current_speed
    
    # 避免除以零的情况
    if max_historical_speed < 1e-6:
        max_historical_speed = 1e-6
    
    # 动态速度阈值 = 历史最高速度 × 百分比
    speed_threshold = max_historical_speed * speed_percentage_threshold
    
    # 判断当前是否为低速
    is_low_speed = current_speed < speed_threshold
    # print(f"is_low_speed: {is_low_speed}")
    
    # 2. 计算连续低速的持续时间
    low_speed_duration = 0
    if is_low_speed:
        # 向前回溯，计算连续低速的时间
        for i in range(t, max(-1, t - low_speed_window - 5), -1):
            if speeds[i] < speed_threshold:
                low_speed_duration += 1
            else:
                break
    # print(f"low_speed_duration: {low_speed_duration}")
    # 3. 检查夹爪状态变化（基于历史信息）
    current_gripper = grippers[t]
    
    # 检查夹爪是否刚刚从打开变为闭合（在最近几步内）
    # 这表示机械臂刚刚完成或正在进行抓取动作
    gripper_just_closed = False
    gripper_closed_steps_ago = -1
    
    if current_gripper:  # 当前夹爪是闭合的
        # 向前回溯，查看夹爪是在多少步之前闭合的
        lookback_window = min(gripper_change_lookahead, t)
        for i in range(t, max(-1, t - lookback_window - 1), -1):
            if i >= 0 and grippers[i]:
                # 找到最早的闭合点
                gripper_closed_steps_ago = t - i
            else:
                # 找到了打开状态，说明夹爪是在 (i+1) 步闭合的
                if i + 1 <= t:
                    gripper_just_closed = True
                    gripper_closed_steps_ago = t - (i + 1)
                break
        
        # 如果在整个回溯窗口内都是闭合的，也认为是刚刚闭合
        # （可能是在更早之前闭合的）
        if gripper_closed_steps_ago >= 0 and gripper_closed_steps_ago <= gripper_change_lookahead:
            gripper_just_closed = True
    
    # print(f"gripper_just_closed: {gripper_just_closed}, steps_ago: {gripper_closed_steps_ago}")

    # 4. 检查是否已经稳定抓取并准备移动
    # 核心条件：
    # - 夹爪已经连续闭合了N个时刻
    # - 速度在最近N步内呈递增趋势（加速离开）
    
    is_stable_grasp_and_moving = False
    stable_grasp_confidence = 0.0
    gripper_closed_duration = 0
    speed_increasing = False
    speed_increase_ratio = 0.0
    
    if current_gripper:  # 当前夹爪是闭合的
        # 计算夹爪连续闭合的时长
        for i in range(t, -1, -1):
            if grippers[i]:
                gripper_closed_duration += 1
            else:
                break
        
        # 检查速度是否在递增（加速）
        # 比较最近N步的速度变化
        acceleration_window = min(gripper_change_lookahead, t, gripper_closed_duration)
        if acceleration_window >= 2:
            # 获取窗口内的速度
            window_start = max(0, t - acceleration_window + 1)
            speed_window = speeds[window_start:t+1]
            
            # 计算速度递增的比例（有多少步是递增的）
            increasing_count = 0
            for i in range(1, len(speed_window)):
                if speed_window[i] > speed_window[i-1]:
                    increasing_count += 1
            
            if len(speed_window) > 1:
                speed_increase_ratio = increasing_count / (len(speed_window) - 1)
                # 如果超过60%的步数都在递增，认为是加速状态
                if speed_increase_ratio >= 0.6:
                    speed_increasing = True
            
            # 额外检查：当前速度是否高于窗口起始速度
            if len(speed_window) >= 2:
                speed_delta = speed_window[-1] - speed_window[0]
                speed_increasing = speed_increasing and (speed_delta > 0)
        
        # 判断是否满足稳定抓取并移动的条件
        min_closed_duration = max(2, low_speed_window)  # 至少闭合2步
        if gripper_closed_duration >= min_closed_duration and speed_increasing:
            is_stable_grasp_and_moving = True
            
            # 计算稳定抓取的置信度
            # 因素1: 夹爪闭合时间越长越稳定
            duration_factor = min(1.0, gripper_closed_duration / (min_closed_duration * 2))
            # 因素2: 速度递增比例越高越好
            acceleration_factor = speed_increase_ratio
            # 因素3: 当前速度相对于阈值
            speed_factor = min(1.0, current_speed / speed_threshold) if speed_threshold > 1e-6 else 0.5
            
            stable_grasp_confidence = (duration_factor * 0.4 + acceleration_factor * 0.4 + speed_factor * 0.2)

    # 5. 综合判断是否处于即将抓取状态
    # 核心条件：
    # - 当前处于低速状态
    # - 低速状态已持续足够长时间（表示已接近目标）
    # - 夹爪刚刚闭合（表示正在抓取）
    
    is_pre_grasp = False
    pre_grasp_confidence = 0.0
    
    if is_low_speed and low_speed_duration >= low_speed_window:
        is_pre_grasp = True
        # 计算置信度
        # 因素1: 速度越低越好
        speed_confidence = max(0.0, 1.0 - current_speed / speed_threshold)
        # 因素2: 低速持续时间越长越好（最多到2倍窗口）
        duration_confidence = min(1.0, low_speed_duration / (low_speed_window * 2))
        # 因素3: 夹爪刚刚闭合的话提高置信度
        if gripper_just_closed:
            gripper_confidence = max(0.5, 1.0 - gripper_closed_steps_ago / gripper_change_lookahead)
        else:
            gripper_confidence = 0.3  # 没有夹爪闭合信号，给一个较低的基础置信度
        
        # 综合置信度（加权平均）
        pre_grasp_confidence = (speed_confidence * 0.4 + duration_confidence * 0.3 + gripper_confidence * 0.3)
    
    # 构建返回结果
    result = {
        "is_pre_grasp": is_pre_grasp,
        "pre_grasp_confidence": pre_grasp_confidence,
        "is_stable_grasp_and_moving": is_stable_grasp_and_moving,
        "stable_grasp_confidence": stable_grasp_confidence,
        "details": {
            # 速度相关
            "current_speed": float(current_speed),
            "max_historical_speed": float(max_historical_speed),
            "speed_threshold": float(speed_threshold),
            "speed_percentage": float(current_speed / max_historical_speed if max_historical_speed > 1e-6 else 0.0),
            "is_low_speed": bool(is_low_speed),
            "low_speed_duration": int(low_speed_duration),
            
            # 夹爪状态相关
            "current_gripper_state": bool(current_gripper),
            "gripper_just_closed": bool(gripper_just_closed),
            "gripper_closed_steps_ago": int(gripper_closed_steps_ago),
            "gripper_closed_duration": int(gripper_closed_duration),
            
            # 稳定抓取相关
            "speed_increasing": bool(speed_increasing),
            "speed_increase_ratio": float(speed_increase_ratio),
        }
    }
    
    return result


def search_best_action_chunk(client, element, env, reusable_valuemap, replan_steps, logger, t, current_episode_costs):
    """
    通过采样多个动作轨迹并搜索最优轨迹来获取最佳动作序列。
    
    Args:
        client: 模型客户端，用于推理获取动作序列
        element (Dict): 包含观察、状态和提示的输入元素
        env: LIBERO 环境实例
        reusable_valuemap: 预构建的价值地图，用于评估轨迹
        replan_steps (int): 重规划步数
        logger: 日志记录器
        t (int): 当前时间步
        current_episode_costs (List): 当前episode的成本记录列表，函数会将成本数据添加到此列表
        
    Returns:
        np.ndarray: 最佳动作序列 (shape: [replan_steps, action_dim])
    """
    # Query model to get action
    action_chunk = client.infer(element)["actions"]
    assert action_chunk.shape[-2] >= replan_steps, (
        f"We want to replan every {replan_steps} steps, but policy only predicts {action_chunk.shape[-2]} steps."
    )

    # Evaluate trajectory using built valuemap
    try:
        if action_chunk.ndim == 2:
            action_chunk = np.expand_dims(action_chunk, axis=0)

        batch_eef_positions = []
        for i in range(action_chunk.shape[0]):
            eef_traj = compute_eef_trajectory_from_actions(env, action_chunk[i])
            batch_eef_positions.append(eef_traj)
        eef_trajs = np.stack(batch_eef_positions, axis=0)

        if reusable_valuemap is not None:
            evaluation_result = evaluate_trajectory_with_value_map(
                reusable_valuemap, eef_trajs, current_env=env
            )
            
            traj_cost = evaluation_result["step_info"]["traj_cost"] # shape: (num_traj, num_steps)
            best_traj_id = evaluation_result["step_info"]["best_traj_id"]
            best_action_chunk = action_chunk[best_traj_id]
            best_traj_cost = traj_cost[best_traj_id, : replan_steps]

            # Record cost data for current step
            step_cost_data = {
                "step": t,
                "best_traj_cost": best_traj_cost.copy(),
                "best_traj_id": best_traj_id,
                "replan_steps": replan_steps,
            }
            current_episode_costs.append(step_cost_data)
            logger.info(f"Step {t} - Best traj cost: {best_traj_cost}")
        else:
            logger.warning("No available valuemap, skipping trajectory evaluation")
            best_action_chunk = action_chunk[0]

    except Exception as e:
        logger.warning(f"Trajectory evaluation failed: {e}")
        best_action_chunk = action_chunk[0]

    return best_action_chunk