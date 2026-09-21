import collections
import dataclasses
import json
import logging
import math
import os
import cv2
import pathlib
import time
from datetime import datetime

import imageio
import numpy as np
# import perturbation
import tqdm
import tyro
import yaml
from PIL import Image
import matplotlib.pyplot as plt

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from util import (
    compute_eef_trajectory_from_actions, 
    build_reusable_value_map, 
    evaluate_trajectory_with_value_map,
    is_gripper_closed,
    search_best_action_chunk,
    detect_pre_grasp_state,
    get_reordered_objects,
)

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    sampling_bs: int = 4
    sampling_std: float = 2.0  # Standard deviation for sampling actions
    search_start_step: int = 10  # Only use search_best_action_chunk when t >= this value

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # LIBERO Pro parameters
    #################################################################################################################
    evaluation_config_path: str = "third_party/LIBERO-PRO/evaluation_config.yaml"  # Path to evaluation config
    #################################################################################################################
    # Logging and experiment tracking parameters
    #################################################################################################################
    local_log_dir: str = "./experiments/logs"  # Local directory for experiment logs
    run_id_note: str = None  # Extra note to add to end of run ID for logging
    save_experiment_config: bool = True  # Whether to save experiment configuration
    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "./experiments/videos"  # Path to save videos
    img_out_path: str = "./experiments/imgs"  # Path to save imgs

    seed: int = 7  # Random Seed (for reproducibility)


def setup_logging(args: Args):
    """Setup experiment logging"""
    # Create run ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"LIBERO-PRO-{args.task_suite_name}-{timestamp}"
    if args.run_id_note is not None:
        run_id += f"-{args.run_id_note}"

    # Create log directory
    os.makedirs(args.local_log_dir, exist_ok=True)
    log_filepath = os.path.join(args.local_log_dir, f"{run_id}.txt")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_filepath, encoding="utf-8")],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Experiment run ID: {run_id}")
    logger.info(f"Log file path: {log_filepath}")

    return logger, run_id, log_filepath


def save_experiment_config(args: Args, run_id: str, log_filepath: str):
    """Save experiment configuration"""
    if not args.save_experiment_config:
        return

    config_data = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "args": dataclasses.asdict(args),
        "log_file": log_filepath,
    }

    config_filepath = os.path.join(args.local_log_dir, f"{run_id}_config.json")
    with open(config_filepath, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    logging.info(f"Experiment configuration saved to: {config_filepath}")


def save_episode_video(
    replay_images: list, task_description: str, episode_idx: int, success: bool, run_id: str, args: Args
):
    """Save individual episode video"""
    # Create task-specific video directory
    task_segment = task_description.replace(" ", "_").replace("/", "_")
    task_video_dir = os.path.join(args.video_out_path, run_id, task_segment)
    os.makedirs(task_video_dir, exist_ok=True)

    # Generate video filename
    suffix = "success" if success else "failure"
    video_filename = f"episode_{episode_idx:03d}_{suffix}.mp4"
    video_filepath = os.path.join(task_video_dir, video_filename)

    # Save video
    try:
        imageio.mimwrite(
            video_filepath,
            [np.asarray(x) for x in replay_images],
            fps=10,
        )
        logging.info(f"Episode video saved: {video_filepath}")
    except Exception as e:
        logging.error(f"Failed to save episode video: {e}")


def _plot_trajectory(trajectory, output_path):
    """
    Plot trajectory with N dimensions as different colored lines.
    
    Args:
        trajectory: List of N-dimensional velocity vectors or 1D array
        output_path: Path to save the plot
    """
    if len(trajectory) == 0:
        logging.warning("No velocity data to plot")
        return
    
    velocity_array = np.array(trajectory)  # Shape: (T, N) or (T,)
    time_steps = np.arange(len(trajectory))
    
    # Handle 1D case (T,) by reshaping to (T, 1)
    if velocity_array.ndim == 1:
        velocity_array = velocity_array.reshape(-1, 1)
    
    # Get number of dimensions
    num_dims = velocity_array.shape[1]
    
    # Create figure with good size
    plt.figure(figsize=(12, 6))
    
    # Define colors - use a colormap for arbitrary number of dimensions
    colors = plt.cm.tab10(np.linspace(0, 1, max(num_dims, 1)))
    dimension_labels = [f'Joint {i+1}' for i in range(num_dims)]
    
    # Plot each dimension with different color
    for dim in range(num_dims):
        plt.plot(time_steps, velocity_array[:, dim], 
                color=colors[dim], label=dimension_labels[dim], linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Trajectory', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    logging.info(f"Plot saved to: {output_path}")


def add_text_to_image(temp_img, CoA_step):
    """Add text overlay to image showing length and step number.
    
    Args:
        temp_img (np.ndarray): Input image of shape (224, 224, 3)
        CoA_step (int): Current step number
        
    Returns:
        np.ndarray: Image with text overlay
    """
    img = temp_img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"step: {CoA_step}"
    
    # Get text size to position it in upper right
    (text_width, text_height), _ = cv2.getTextSize(text, font, 0.5, 1)
    
    # Position text 10 pixels from right and top edges
    text_x = img.shape[1] - text_width - 10
    text_y = text_height + 10
    
    # Add white text with black outline for visibility
    cv2.putText(img, text, (text_x, text_y), font, 0.5, (0,0,0), 2)
    cv2.putText(img, text, (text_x, text_y), font, 0.5, (255,255,255), 1)
    
    return img


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def eval_libero(args: Args) -> None:
    # Setup logging
    logger, run_id, log_filepath = setup_logging(args)

    # Save experiment configuration
    save_experiment_config(args, run_id, log_filepath)

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if "libero_spatial" in args.task_suite_name:
        max_steps = 220  # longest training demo has 193 steps
    elif "libero_object" in args.task_suite_name:
        max_steps = 280  # longest training demo has 254 steps
    elif "libero_goal" in args.task_suite_name:
        max_steps = 300  # longest training demo has 270 steps
    elif "libero_10" in args.task_suite_name:
        max_steps = 520  # longest training demo has 505 steps
    elif "libero_90" in args.task_suite_name:
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0

    logger.info(f"Starting evaluation of task suite: {args.task_suite_name}")
    logger.info(f"Number of tasks: {num_tasks_in_suite}")
    logger.info(f"Trials per task: {args.num_trials_per_task}")

    # Get current timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_out_dir = pathlib.Path(args.video_out_path) / f"{timestamp}_{args.task_suite_name}"
    video_out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Videos will be saved to: {video_out_dir}")

    object_path = "/home/zijianwang/openpi/third_party/LIBERO-PRO/reordered_objects_with_descriptions.json"

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        all_target_objects = get_reordered_objects(object_path, task_description)
        num_stages = len(all_target_objects)

        # Start episodes
        task_episodes, task_successes = 0, 0
        logger.info(f"\nStarting task: {task_description}, This task has {num_stages} stages.")

        # 初始化cost记录
        all_episode_costs = []  # 存储所有episode的cost数据
        current_episode_costs = []  # 存储当前episode的cost数据

        ### Entering task loop
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):

            use_search = False
            is_pre_grasp = False
            is_next_stage = False
            has_rebuilt = False

            logger.info(f"\nTask: {task_description}")
            logger.info(f"Episode {episode_idx + 1}/{args.num_trials_per_task}")
            # Reset environment
            env.reset()
            robot_instance = env.robots[0]
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            ### Build reusable valuemap before task starts
            reusable_valuemap = build_reusable_value_map(env, task_description, stage=0)
            # reusable_valuemap = None

            # Setup
            t = 0
            replay_images, current_episode_costs, linear_speed_trajectory, gripper_state_trajectory = [], [], [], []
            logger.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                if is_next_stage == True and num_stages > 1 and has_rebuilt == False:
                    reusable_valuemap = build_reusable_value_map(env, task_description, stage=1)
                    has_rebuilt = True
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    linear_speed, gripper_state = 0, 0
                    linear_speed_trajectory.append(linear_speed)
                    gripper_state_trajectory.append(gripper_state)
                    t += 1
                    continue

                # Get preprocessed image - note the 180 degree rotation
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))

                # Image.fromarray(np.uint8(img)).save("./experiments/tmp/live_image.png")
                tempimg = add_text_to_image(img, t)
                replay_images.append(tempimg)

                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk

                    # Use search-based approach only after threshold step
                    if t >= args.search_start_step and is_pre_grasp == False:
                        use_search = True

                    if use_search == True:
                        # Search for best action chunk through trajectory evaluation
                        element = {
                                    "observation/image": img,
                                    "observation/wrist_image": wrist_img,
                                    "observation/state": np.concatenate(
                                        (obs["robot0_eef_pos"],
                                        _quat2axisangle(obs["robot0_eef_quat"]),
                                        obs["robot0_gripper_qpos"],)
                                    ),
                                    "prompt": str(task_description),
                                    "sampling_bs": int(args.sampling_bs),
                                    "sampling_std": float(args.sampling_std),
                                    }

                        best_action_chunk = search_best_action_chunk(
                            client, element, env, reusable_valuemap, args.replan_steps, 
                            logger, t, current_episode_costs)

                    elif use_search == False:
                        # Normal inference without search (before threshold)
                        element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],)
                            ),
                        "prompt": str(task_description),
                        "sampling_bs": 1,
                        "sampling_std": 1.0,
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert action_chunk.shape[-2] >= args.replan_steps, (f"We want to replan every {args.replan_steps} steps, but policy only predicts {action_chunk.shape[-2]} steps.")
                        best_action_chunk = action_chunk[0]

                    action_plan.extend(best_action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                # Execute action in environment
                # print(action)
                obs, reward, done, info = env.step(action.tolist())

                # 或者机器人本体信息
                gripper_is_closed_result = is_gripper_closed(obs)
                eef_total_velocity = robot_instance._hand_total_velocity  # vx, vy, vz, rx, ry, rz 3个线速度, 3个角速度
                # 计算线速度的幅值（前3个分量）
                linear_speed = np.linalg.norm(eef_total_velocity[:3])

                linear_speed_trajectory.append(linear_speed)
                gripper_state_trajectory.append(gripper_is_closed_result)

                # 检测机械臂是否处于抓取状态
                proprioception_res = detect_pre_grasp_state(t, 
                                                            linear_speed_trajectory, 
                                                            gripper_state_trajectory,
                                                            speed_percentage_threshold = 0.3,
                                                            low_speed_window = 1,
                                                            gripper_change_lookahead = 3,
                                                            min_history_window = 10)                              
                is_pre_grasp = proprioception_res["is_pre_grasp"]
                is_next_stage = proprioception_res["is_stable_grasp_and_moving"]
                if is_pre_grasp == True:
                    use_search = False
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            ##################################################################################################################
            ##################################################################################################################
            task_episodes += 1
            total_episodes += 1

            # Save cost data for current episode
            episode_data = {
                "episode_idx": episode_idx,
                "task_description": task_description,
                "success": done,
                "total_steps": t,
                "costs": current_episode_costs.copy(),
            }
            all_episode_costs.append(episode_data)

            logger.info(f"Episode {episode_idx + 1} completed - Success: {done}, Steps: {t}, Cost records: {len(current_episode_costs)}")

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_filename = f"task{task_id:02d}_ep{episode_idx:03d}_{task_segment}_{suffix}.mp4"
            video_path = video_out_dir / video_filename
            
            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=18,
            )
            logging.info(f"Video saved to: {video_path}")

            # Save velocity trajectory plot
            velocity_plot_filename = f"task{task_id:02d}_ep{episode_idx:03d}_{task_segment}_{suffix}_velocity.png"
            velocity_plot_path = video_out_dir / velocity_plot_filename
            _plot_trajectory(linear_speed_trajectory, str(velocity_plot_path))

            # Save gripper state trajectory plot
            gripper_state_plot_filename = f"task{task_id:02d}_ep{episode_idx:03d}_{task_segment}_{suffix}_gripper_state.png"
            gripper_state_plot_path = video_out_dir / gripper_state_plot_filename
            _plot_trajectory(gripper_state_trajectory, str(gripper_state_plot_path))

            # Log current results
            logger.info(f"Episode result: {'Success' if done else 'Failure'}")
            logger.info(f"Completed episodes: {total_episodes}")
            logger.info(f"Successful episodes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0
        total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

        logger.info(f"Current task success rate: {task_success_rate:.4f} ({task_success_rate * 100:.1f}%)")
        logger.info(f"Overall success rate: {total_success_rate:.4f} ({total_success_rate * 100:.1f}%)")
        logger.info(f"Current task episodes: {task_episodes}, successful: {task_successes}")
        logger.info(f"Total episodes: {total_episodes}, total successful: {total_successes}")

        # Save cost data for current task
        task_cost_file = f"./experiments/cost/{run_id}/cost_data_task_{task_id}.json"
        os.makedirs(os.path.dirname(task_cost_file), exist_ok=True)
        with open(task_cost_file, "w") as f:
            json.dump(all_episode_costs, f, indent=2, default=str)
        logger.info(f"Cost data saved to: {task_cost_file}")

    # Calculate final results
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0

    # Log final results
    logger.info("=" * 60)
    logger.info("Experiment completed - Final results:")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Total successful: {total_successes}")
    logger.info(f"Final success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)")

    logger.info("=" * 60)
    logger.info(f"Experiment run ID: {run_id}")
    logger.info(f"Log file: {log_filepath}")
    logger.info("Experiment completed!")




if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_libero(args)

