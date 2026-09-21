import collections
import dataclasses
import logging
import re
import math, sys, os
import pathlib
from datetime import datetime
from typing import List, Optional
sys.path.append("/hdd/zijianwang/openpi/third_party/LIBERO-PRO")
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from scipy.ndimage import distance_transform_edt, gaussian_filter

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

from util import (
    compute_eef_trajectory_from_actions, 
    build_reusable_value_map, 
    evaluate_trajectory_with_value_map,
    is_gripper_closed,
    search_best_action_chunk,
    detect_pre_grasp_state,
    get_reordered_objects,
)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_10"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    task_ids: Optional[List[int]] = None  # Specific task IDs to run (if None, run all tasks)
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = x.min(), x.max()
    return (x - mn) / (mx - mn + 1e-8)


def visualize_affordance_map(affordance_map: np.ndarray, avoidance_map: np.ndarray, lmp_env, env_vox) -> None:
    target_map = _normalize_map(distance_transform_edt(1 - affordance_map))
    obstacle_map = _normalize_map(
        gaussian_filter(avoidance_map, sigma=lmp_env.obstacle_map_gaussian_sigma)
    )
    costmap = _normalize_map(
        target_map * lmp_env.target_map_weight + obstacle_map * lmp_env.obstacle_map_weight
    )

    targets_voxel = np.argwhere(affordance_map == 1)
    targets_world = lmp_env._voxel_to_world(targets_voxel)

    step_info = {
        "costmap": costmap,
        "raw_target_map": affordance_map,
        "targets_world": targets_world,
        "start_pos_world": env_vox.get_ee_pos(),
    }
    env_vox.visualizer.visualize(step_info, show=False, save=True)


def query_affordance_value_at_voxel(affordance_map: np.ndarray, voxel_xyz) -> float:
    voxel_xyz = np.round(np.asarray(voxel_xyz)).astype(int)
    if voxel_xyz.shape != (3,):
        raise ValueError(f"voxel_xyz must be shape (3,), got {voxel_xyz.shape}")

    x, y, z = voxel_xyz.tolist()
    if (
        x < 0 or x >= affordance_map.shape[0]
        or y < 0 or y >= affordance_map.shape[1]
        or z < 0 or z >= affordance_map.shape[2]
    ):
        raise IndexError(f"voxel index out of range: {voxel_xyz}, map shape={affordance_map.shape}")

    return float(affordance_map[x, y, z])


def query_affordance_value_at_world(affordance_map: np.ndarray, world_xyz, lmp_env) -> float:
    voxel_xyz = lmp_env._world_to_voxel(np.asarray(world_xyz, dtype=np.float32))
    return query_affordance_value_at_voxel(affordance_map, voxel_xyz)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Get current timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create timestamped video output directory
    video_out_dir = pathlib.Path(args.video_out_path) / f"{timestamp}_{args.task_suite_name}"
    video_out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Videos will be saved to: {video_out_dir}")

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

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
    task_ids_to_run = args.task_ids if args.task_ids is not None else list(range(num_tasks_in_suite))
    for task_id in tqdm.tqdm(task_ids_to_run):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            reusable_valuemap = build_reusable_value_map(env, task_description, stage=0)
            env_vox = reusable_valuemap["env_vox"]
            lmp_env = reusable_valuemap["lmp_env"]
            avoidance_map = reusable_valuemap["avoidance_map"]
            affordance_map = reusable_valuemap["affordance_map"]

            # Visualize 3D value map once per episode
            visualize_affordance_map(affordance_map, avoidance_map, lmp_env, env_vox)

            # Example: query affordance value at current ee world coordinate
            current_affordance_value = query_affordance_value_at_world(
                affordance_map, env_vox.get_ee_pos(), lmp_env
            )
            logging.info(f"Current EE affordance value: {current_affordance_value:.4f}")

            
            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
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
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

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
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        # print(f"action_chunk.shape: {action_chunk.shape}")
                        # assert (
                        #     len(action_chunk) >= args.replan_steps
                        # ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        assert action_chunk.shape[-2] >= args.replan_steps, (
                        f"We want to replan every {args.replan_steps} steps, but policy only predicts {action_chunk.shape[-2]} steps."
                    )
                        action_chunk = action_chunk[0]
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env_vox.step(action.tolist())
                    action_length = len(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode with unique filename
            # Include task_id, episode_idx, and success/failure status
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_filename = f"task{task_id:02d}_ep{episode_idx:03d}_{task_segment}_{suffix}.mp4"
            video_path = video_out_dir / video_filename
            
            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=24,
            )
            logging.info(f"Video saved to: {video_path}")

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"All videos saved to: {video_out_dir}")


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


