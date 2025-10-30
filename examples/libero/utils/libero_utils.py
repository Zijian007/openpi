"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SegmentationRenderEnv, VoxRenderEnv

from .visualizers import ValueMapVisualizer

from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)
import numpy as np
import cv2
import imageio

def find_most_recent_image_in_folder(folder_path):
    img_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        return None
    most_recent_file = max(img_files, key=lambda f: os.path.getctime(os.path.join(folder_path, f)))
    oldest_file = min(img_files, key=lambda f: os.path.getctime(os.path.join(folder_path, f)))
    recent_path = os.path.join(folder_path, most_recent_file)
    oldest_path = os.path.join(folder_path, oldest_file)
    return recent_path, oldest_path

def add_labeled_box(img, label, position, color):
    """
    在图像上添加带框的标签。

    参数:
        img (np.ndarray): 输入图像。
        label (str): 标签内容。
        position (tuple): 标签位置 (x, y)。
        color (tuple): 标签颜色 (B, G, R)。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    # 计算文本大小
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_width, text_height = text_size

    # 计算框的位置
    box_padding = 10
    top_left = (position[0] - box_padding, position[1] - text_height - box_padding)
    bottom_right = (position[0] + text_width + box_padding, position[1] + box_padding)

    # 绘制白色底框 + 黑边框
    cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), -1)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 2)

    # 绘制文本
    text_position = (position[0], position[1])
    cv2.putText(img, label, text_position, font, font_scale, color, font_thickness)
    
def label_observation_frames(annotated_img_paths, task_dir):
    """
    在三张图像的左上角分别标注 "A"、"B" 和 "C"，并保存标注后的图像。

    参数:
        annotated_img_paths (list): 包含三张图像路径的列表 [recent_path, oldest_path, cur_path]。
        task_dir (str): 保存标注图像的目录路径。

    返回:
        list: 标注后的图像路径列表 [save_a, save_b, save_c]。
    """

    # 加载图像
    oldest_path, recent_path, cur_path = annotated_img_paths
    begin_frame = cv2.imread(oldest_path)
    previous_frame = cv2.imread(recent_path)
    current_frame = cv2.imread(cur_path)

    # 确保图像为 uint8 类型
    if begin_frame.dtype != np.uint8:
        begin_frame = (np.clip(begin_frame, 0.0, 1.0) * 255).astype(np.uint8)
    if previous_frame.dtype != np.uint8:
        previous_frame = (np.clip(previous_frame, 0.0, 1.0) * 255).astype(np.uint8)
    if current_frame.dtype != np.uint8:
        current_frame = (np.clip(current_frame, 0.0, 1.0) * 255).astype(np.uint8)

    # 在子任务开始帧标注 "A"
    add_labeled_box(begin_frame, "A", (20, 40), (0, 0, 255))  # 红色

    # 在上一步帧标注 "B"
    add_labeled_box(previous_frame, "B", (20, 40), (0, 0, 255))  # 红色

    # 在当前帧标注 "C"
    add_labeled_box(current_frame, "C", (20, 40), (0, 0, 255))  # 红色

    # 保存标注后的图像
    labeled_query_path = os.path.join(task_dir, 'labeled_query')
    os.makedirs(labeled_query_path, exist_ok=True)
    save_a = os.path.join(labeled_query_path, 'image_a.jpg')
    save_b = os.path.join(labeled_query_path, 'image_b.jpg')
    save_c = os.path.join(labeled_query_path, 'image_c.jpg')
    cv2.imwrite(save_a, begin_frame)
    cv2.imwrite(save_b, previous_frame)
    cv2.imwrite(save_c, current_frame)

    # print(f"✅ 标注 'A' 的图像已保存：{save_a}")
    # print(f"✅ 标注 'B' 的图像已保存：{save_b}")
    # print(f"✅ 标注 'C' 的图像已保存：{save_c}")

    return [save_a, save_b, save_c]

def save_img(img, save_path=None):
    """
    保存图像为 PNG 格式。

    参数:
        img (np.ndarray): 输入图像 (H, W, 3)。
        save_path (str): 图像保存路径 (.jpg)，如果为 None，则使用默认路径。
    """
    if save_path is None:
        save_path = "./tmp/action_replay.jpg"

    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)

    imageio.imwrite(save_path, img)
    print(f"✅ 图像已保存：{save_path}")
    

def save_img_with_multiple_trajectories(replay_img, all_objects_positions, sim, camera_name="agentview", save_path=None):
    """
    将多个物体的轨迹可视化到单张图像上，并保存为 PNG 图像（带编号框）
    
    参数:
        replay_img: 单张图像 (H, W, 3)
        all_objects_positions: list of list of 3D positions，每个元素是某个物体的轨迹 [(T,3), (T,3), ...]
        sim: 模拟器实例，用于获取相机参数
        camera_name: 相机名
        save_path: 图像保存路径 (.jpg)
    """
    if save_path is None:
        save_path = "./tmp/multi_object_trajectory.jpg"

    h, w = replay_img.shape[:2]
    intrinsics = get_camera_intrinsics(sim, camera_name, w, h)
    extrinsics = get_camera_extrinsics(sim, camera_name)

    # 投影所有物体轨迹
    projected_trajectories = []
    for positions in all_objects_positions:
        pts = np.array(positions.reshape(1,-1))  # shape: (T, 3)
        projected_pts = point_to_pixel(pts, intrinsics, extrinsics)  # shape: (T, 2)
        projected_trajectories.append(projected_pts)

    # 设置不同颜色
    colors = [
        (255, 0, 0),    # 红
        (0, 255, 0),    # 绿
        (0, 0, 255),    # 蓝
        (255, 255, 0),  # 青
        (255, 0, 255),  # 紫
        (0, 255, 255),  # 黄
    ]

    img = replay_img.copy()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
    img = np.flip(img, axis=0)
    img = np.ascontiguousarray(img)

    # 绘制每个物体轨迹及编号
    for idx, projected_pts in enumerate(projected_trajectories):
        color = colors[idx % len(colors)]

        # 绘制轨迹线和点
        for j in range(1, len(projected_pts)):
            pt1 = tuple(np.round(projected_pts[j - 1]).astype(int))
            pt2 = tuple(np.round(projected_pts[j]).astype(int))
            if all(0 <= pt < dim for pt, dim in zip(pt1, (w, h))) and all(0 <= pt < dim for pt, dim in zip(pt2, (w, h))):
                cv2.line(img, pt1, pt2, color=color, thickness=2)

        # 当前点（轨迹终点）
        px, py = projected_pts[-1].astype(int)
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(img, (px, py), radius=5, color=color, thickness=2)

            # 编号文本框
            displayed_text = f"{idx}"
            text_length = len(displayed_text)
            box_width = 30 + 10 * (text_length - 1)
            box_height = 30
            top_left = (px - box_width // 2, py - box_height // 2)
            bottom_right = (px + box_width // 2, py + box_height // 2)

            # 白色底框 + 黑边
            cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), -1)
            cv2.rectangle(img, top_left, bottom_right, (0, 0, 0), 2)

            # 文本
            org = (px - 7 * text_length, py + 7)
            cv2.putText(img, displayed_text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # # 保存图像
    # imageio.imwrite(save_path, img)
    # print(f"✅ 多物体轨迹图像已保存：{save_path}")
    
    return img
    
    
def save_gif_with_trajectory(replay_imgs, ee_positions, sim, camera_name="agentview", save_path=None, fps=5):
    """
    将动作图像序列 + ee 轨迹（完整累计）绘制为动态图 GIF
    """
    if save_path is None:
        save_path = f"/home/siyu/deployment/openvla-oft/tmp/action_replay_with_traj.gif"

    h, w = replay_imgs[0].shape[:2]
    intrinsics = get_camera_intrinsics(sim, camera_name, w, h)
    extrinsics = get_camera_extrinsics(sim, camera_name)

    # 投影所有点为像素坐标
    pts = np.array(ee_positions)
    projected_pts = point_to_pixel(pts, intrinsics, extrinsics)  # shape (N, 2)

    frames = []
    for i in range(len(replay_imgs)):
        img = replay_imgs[i].copy()
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        img = np.flip(img, axis=0)  # 上下翻转
        # if img.shape[2] == 3:
        #     img = img[..., ::-1]  # BGR → RGB
        img = np.ascontiguousarray(img)

        # ✅ 画从起点到当前步所有轨迹线
        for j in range(1, i + 1):
            pt1 = tuple(np.round(projected_pts[j - 1]).astype(int))
            pt2 = tuple(np.round(projected_pts[j]).astype(int))
            cv2.line(img, pt1, pt2, color=(0, 255, 0), thickness=2)

        # ✅ 画从起点到当前所有轨迹点
        for j in range(i + 1):
            px, py = projected_pts[j].astype(int)
            cv2.circle(img, (px, py), radius=3, color=(255, 0, 0), thickness=-1)

        # ✅ 可选：当前末端位置再加个大圈强调
        px, py = projected_pts[i].astype(int)
        cv2.circle(img, (px, py), radius=5, color=(0, 0, 255), thickness=2)

        frames.append(img)

    imageio.mimsave(save_path, frames, fps=fps)
    print(f"✅ 动作轨迹 GIF 保存完成：{save_path}")
    

def save_gif(replay_imgs, save_name="action_replay.gif", fps=10):
    """
    将 replay_imgs 图像序列保存为 GIF 动画。
    """
    # 将图片转换为 uint8 格式（如果尚未）
    frames = []
    for img in replay_imgs:
        if img.dtype != np.uint8:
            img = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        img = np.flip(img, axis=0)  # 确保方向一致
        if img.shape[2] == 3:       # BGR → RGB
            img = img[..., ::-1]
        frames.append(img)

    root = "/home/siyu/deployment/openvla-oft/tmp"
    save_path = os.path.join(root, save_name)
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"✅ 动作执行过程已保存为 GIF：{save_path}")
    
def get_camera_intrinsics(env, camera_name, img_width, img_height):
    cam_id = env.sim.model.camera_name2id(camera_name)
    fovy = env.sim.model.cam_fovy[cam_id]
    fovy_rad = np.deg2rad(fovy)

    fy = img_height / (2 * np.tan(fovy_rad / 2))
    fx = fy  # 假设像素长宽比为 1:1

    cx = img_width / 2
    cy = img_height / 2

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    return K

def get_camera_extrinsics(env, camera_name):
    cam_id = env.sim.model.camera_name2id(camera_name)
    cam_pos = env.sim.model.cam_pos[cam_id]
    cam_mat = env.sim.model.cam_mat0[cam_id].reshape(3, 3)

    R_cw = cam_mat.T
    t_cw = -R_cw @ cam_pos

    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R_cw
    extrinsics[:3, 3] = t_cw

    return extrinsics

def point_to_pixel(pt, intrinsics, extrinsics):
    """
    pt -- (N, 3) world-frame 3D points
    intrinsics -- (3, 3)
    extrinsics -- (4, 4)
    """
    pt_homo = np.hstack((pt, np.ones((pt.shape[0], 1))))  # (N, 4)
    pt_in_cam = (extrinsics @ pt_homo.T)  # (4, N)

    pt_in_cam[1, :] *= -1
    pt_in_cam[2, :] *= -1
    pt_in_cam = pt_in_cam[:3, :]

    pt_proj = intrinsics @ pt_in_cam
    pt_proj /= pt_proj[2, :]  # z 除法

    return pt_proj[:2, :].T  # (N, 2)

def plot_ee_trajectory_with_projection(sim, img, positions, camera_name="agentview", save_name="ee_traj_projected.jpg"):
    """
    用真实内参和外参矩阵，将末端执行器轨迹绘制在图像上。
    """
    img = img.copy()
    if img.shape[0] > 10:
        img = np.flip(img, axis=0)  # 上下翻转

    if img.shape[2] == 3:
        img = img[..., ::-1]  # BGR → RGB

    img = img.astype(np.float32) / 255.0

    h, w = img.shape[:2]
    intrinsics = get_camera_intrinsics(sim, camera_name, w, h)
    extrinsics = get_camera_extrinsics(sim, camera_name)

    pts = np.array(positions)
    pixels = point_to_pixel(pts, intrinsics, extrinsics)

    # 可视化
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)

    xs, ys = pixels[:, 0], pixels[:, 1]
    ax.plot(xs, ys, color='lime', linewidth=2, marker='o', markersize=4)

    ax.set_title("EE Trajectory Projection")
    ax.axis("off")

    root = "/home/siyu/deployment/openvla-oft/tmp"
    save_path = os.path.join(root, save_name)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"✅ 精准轨迹图已保存至：{save_path}")
    plt.close()

    
    
def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    visualizer = ValueMapVisualizer()
    env_dummy = VoxRenderEnv(visualizer=visualizer, **env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, env_dummy, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
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
