import collections
import dataclasses
import json
import logging
import math
import os
import pathlib
import time
from datetime import datetime

import imageio
import numpy as np
import perturbation
import tqdm
import tyro
import yaml
from PIL import Image

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

from util import compute_eef_trajectory_from_actions, build_reusable_value_map, evaluate_trajectory_with_value_map

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
    sampling_bs: int = 8

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
    video_out_path: str = "./experiments/videos/libero_pro"  # Path to save videos

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


def eval_libero(args: Args) -> None:
    # Setup logging
    logger, run_id, log_filepath = setup_logging(args)

    # Save experiment configuration
    save_experiment_config(args, run_id, log_filepath)

    # Set random seed
    np.random.seed(args.seed)

    # initialize environment perturbation for LIBERO Pro
    with open(args.evaluation_config_path) as f:
        evaluation_cfg = yaml.safe_load(f)

    evaluation_cfg["bddl_files_path"] = evaluation_cfg.get("bddl_files_path", "") + "/" + args.task_suite_name
    evaluation_cfg["task_suite_name"] = args.task_suite_name

    if not os.path.exists(evaluation_cfg.get("init_file_dir", "") + args.task_suite_name + "_temp/"):
        perturbation.create_env(
            configs=evaluation_cfg,
        )
    # args.task_suite_name = args.task_suite_name + "_temp"
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

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        logger.info(f"\nStarting task: {task_description}")

        # 初始化cost记录
        all_episode_costs = []  # 存储所有episode的cost数据
        current_episode_costs = []  # 存储当前episode的cost数据

        # 初始化cost记录
        all_episode_costs = []  # 存储所有episode的cost数据
        current_episode_costs = []  # 存储当前episode的cost数据

        # 在任务开始前构建一次可重复使用的valuemap
        logger.info("构建可重复使用的ValueMap...")
        reusable_valuemap = None
        try:
            # 先重置环境到初始状态
            env.reset()
            obs = env.set_init_state(initial_states[0])  # 使用第一个初始状态

            # 构建可重复使用的valuemap
            reusable_valuemap:dict = build_reusable_value_map(env, task_description)

            if "error" not in reusable_valuemap:
                logger.info(f"可重复使用ValueMap构建完成:")
                logger.info(f"  目标对象: {reusable_valuemap.get('target_objects', [])}")
                logger.info(f"  避免对象: {reusable_valuemap.get('avoid_objects', [])}")
                logger.info(f"  地图大小: {reusable_valuemap.get('map_size', 'N/A')}")
                logger.info(f"  分辨率: {reusable_valuemap.get('resolution', 'N/A')}")
            else:
                logger.warning(f"可重复使用ValueMap构建失败: {reusable_valuemap.get('error', 'Unknown error')}")
                reusable_valuemap = None
        except Exception as e:
            logger.warning(f"可重复使用ValueMap构建失败: {e}")
            reusable_valuemap = None

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logger.info(f"\nTask: {task_description}")
            logger.info(f"Episode {episode_idx + 1}/{args.num_trials_per_task}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            # 重置当前episode的cost记录
            current_episode_costs = []

            logger.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Get preprocessed image
                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )


                Image.fromarray(np.uint8(img)).save("./experiments/tmp/live_image.png")

                # Save preprocessed image for replay video
                replay_images.append(img)

                if not action_plan:
                    # Finished executing previous action chunk -- compute new chunk
                    # Prepare observations dict
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                        "sampling_bs": int(args.sampling_bs),
                    }

                    # Query model to get action
                    action_chunk = client.infer(element)["actions"]
                    assert action_chunk.shape[-2] >= args.replan_steps, (
                        f"We want to replan every {args.replan_steps} steps, but policy only predicts {action_chunk.shape[-2]} steps."
                    )

                    # 使用已构建的valuemap评估轨迹
                    try:
                        if action_chunk.ndim == 2:
                            action_chunk = np.expand_dims(action_chunk, axis=0)


                        start_time = time.time()
                        batch_eef_positions = []
                        for i in range(action_chunk.shape[0]):
                            eef_traj = compute_eef_trajectory_from_actions(env, action_chunk[i])
                            batch_eef_positions.append(eef_traj)
                        eef_trajs = np.stack(batch_eef_positions, axis=0)
                        timer_eef = time.time() - start_time
                        logger.info(f"compute_eef_trajectory_from_actions耗时: {timer_eef:.3f}s")
                        start_eval_time = time.time()
                        if reusable_valuemap is not None:
                            # 使用已构建的valuemap评估轨迹，传入当前环境以获取当前gripper状态
                            evaluation_result = evaluate_trajectory_with_value_map(
                                reusable_valuemap, eef_trajs, current_env=env
                            )
                            timer_eval = time.time() - start_eval_time
                            logger.info(f"evaluate_trajectory_with_value_map耗时: {timer_eval:.3f}s")
                            traj_cost = evaluation_result["step_info"]["traj_cost"]
                            traj_cost = evaluation_result["step_info"]["traj_cost"]
                            best_traj_id = evaluation_result["step_info"]["best_traj_id"]
                            best_action_chunk = action_chunk[best_traj_id]
                            best_traj_cost = traj_cost[best_traj_id, : args.replan_steps]

                            # 记录当前step的cost数据
                            step_cost_data = {
                                "step": t,
                                "best_traj_cost": best_traj_cost.copy(),  # 复制数组避免引用问题
                                "best_traj_id": best_traj_id,
                                "replan_steps": args.replan_steps,
                            }
                            current_episode_costs.append(step_cost_data)

                            logger.info(f"Step {t} - Best traj cost: {best_traj_cost}")

                            if evaluation_result.get("success", False):
                                logger.info(f"轨迹评估完成:")
                                logger.info(f"  轨迹长度: {len(eef_traj)}")
                                logger.info(f"  评估成功: {evaluation_result.get('success', False)}")
                            else:
                                logger.warning(f"轨迹评估失败: {evaluation_result.get('error', 'Unknown error')}")
                        else:
                            logger.warning("没有可用的valuemap，跳过轨迹评估")
    
                    except Exception as e:
                        logger.warning(f"轨迹评估失败: {e}")

                    action_plan.extend(best_action_chunk[: args.replan_steps])

                action = action_plan.popleft()

                # Execute action in environment
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

                # except Exception as e:
                #     logging.error(f"Caught exception: {e}")
                #     break

            task_episodes += 1
            total_episodes += 1

            # 保存当前episode的cost数据
            episode_data = {
                "episode_idx": episode_idx,
                "task_description": task_description,
                "success": done,
                "total_steps": t,
                "costs": current_episode_costs.copy(),
            }
            all_episode_costs.append(episode_data)

            logger.info(
                f"Episode {episode_idx + 1} completed - Success: {done}, Steps: {t}, Cost records: {len(current_episode_costs)}"
            )

            # Save a replay video of the episode
            save_episode_video(replay_images, task_description, episode_idx, done, run_id, args)

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

        # 保存当前任务的cost数据
        task_cost_file = f"./experiments/cost/{run_id}/cost_data_task_{task_id}.json"
        os.makedirs(os.path.dirname(task_cost_file), exist_ok=True)
        with open(task_cost_file, "w") as f:
            json.dump(all_episode_costs, f, indent=2, default=str)
        logger.info(f"Cost data saved to: {task_cost_file}")

        break

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_libero(args)

