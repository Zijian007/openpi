"""Plotly-Based Visualizer"""

import plotly.graph_objects as go
import numpy as np
import os
import datetime


class ValueMapVisualizer:
    """
    A Plotly-based visualizer for 3D value map and planned path.
    """

    def __init__(self, save_dir="experiments/tmp/visualizations", quality="low", map_size=100):
        self.scene_points = None
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.quality = quality
        self.update_quality(self.quality)
        self.map_size = map_size

    def update_bounds(self, lower, upper):
        self.workspace_bounds_min = lower
        self.workspace_bounds_max = upper
        self.plot_bounds_min = lower - 0.15 * (upper - lower)
        self.plot_bounds_max = upper + 0.15 * (upper - lower)
        xyz_ratio = 1 / (self.workspace_bounds_max - self.workspace_bounds_min)
        scene_scale = np.max(xyz_ratio) / xyz_ratio
        self.scene_scale = scene_scale

    def update_quality(self, quality):
        self.quality = quality
        if self.quality == "low":
            self.downsample_ratio = 4
            self.max_scene_points = 150000
            self.costmap_opacity = 0.2 * 0.6
            self.costmap_surface_count = 10
        elif self.quality == "low-full-scene":
            self.downsample_ratio = 4
            self.max_scene_points = 1000000
            self.costmap_opacity = 0.2 * 0.6
            self.costmap_surface_count = 10
        elif self.quality == "low-half-scene":
            self.downsample_ratio = 4
            self.max_scene_points = 250000
            self.costmap_opacity = 0.2 * 0.6
            self.costmap_surface_count = 10
        elif self.quality == "medium":
            self.downsample_ratio = 2
            self.max_scene_points = 300000
            self.costmap_opacity = 0.1 * 0.6
            self.costmap_surface_count = 30
        elif self.quality == "medium-full-scene":
            self.downsample_ratio = 2
            self.max_scene_points = 1000000
            self.costmap_opacity = 0.1 * 0.6
            self.costmap_surface_count = 30
        elif self.quality == "medium-half-scene":
            self.downsample_ratio = 2
            self.max_scene_points = 500000
            self.costmap_opacity = 0.1 * 0.6
            self.costmap_surface_count = 30
        elif self.quality == "high":
            self.downsample_ratio = 1
            self.max_scene_points = 500000
            self.costmap_opacity = 0.07 * 0.6
            self.costmap_surface_count = 50
        elif self.quality == "best":
            self.downsample_ratio = 1
            self.max_scene_points = 500000
            self.costmap_opacity = 0.05 * 0.6
            self.costmap_surface_count = 100
        else:
            raise ValueError(f"Unknown quality: {self.quality}; should be one of [low, medium, high]")

    def update_scene_points(self, points, colors=None):
        points = points.astype(np.float16)
        assert colors.dtype == np.uint8
        self.scene_points = (points, colors)

    def visualize(self, info, show=False, save=True):
        """visualize the path and relevant info using plotly"""
        fig_data = []

        if "traj_world" in info:
            traj_world = info["traj_world"]
            best_traj_id = info.get("best_traj_id", 0)

            # 检查是否为batch轨迹 (B, T, 3) 还是单条轨迹 (T, 3)
            if traj_world.ndim == 3:
                # batch轨迹处理
                batch_size = traj_world.shape[0]
                for i, traj in enumerate(traj_world):
                    if i == best_traj_id:
                        # 绘制最佳轨迹 - 红色粗线
                        for j in range(traj.shape[0] - 1):
                            fig_data.append(
                                go.Scatter3d(
                                    x=traj[j : j + 2, 0],
                                    y=traj[j : j + 2, 1],
                                    z=traj[j : j + 2, 2],
                                    mode="lines",
                                    name="best_traj" if j == 0 else None,
                                    line=dict(width=8, color="red"),
                                    showlegend=j == 0,
                                )
                            )
                        # 添加最佳轨迹的起点和终点标记
                        fig_data.append(
                            go.Scatter3d(
                                x=[traj[0, 0]],
                                y=[traj[0, 1]],
                                z=[traj[0, 2]],
                                mode="markers",
                                name="best_start",
                                marker=dict(size=8, color="darkred", symbol="circle"),
                            )
                        )
                        fig_data.append(
                            go.Scatter3d(
                                x=[traj[-1, 0]],
                                y=[traj[-1, 1]],
                                z=[traj[-1, 2]],
                                mode="markers",
                                name="best_end",
                                marker=dict(size=8, color="darkred", symbol="diamond"),
                            )
                        )
                    else:
                        # 绘制其他轨迹 - 灰色细线
                        for j in range(traj.shape[0] - 1):
                            fig_data.append(
                                go.Scatter3d(
                                    x=traj[j : j + 2, 0],
                                    y=traj[j : j + 2, 1],
                                    z=traj[j : j + 2, 2],
                                    mode="lines",
                                    name="other_traj" if j == 0 else None,
                                    line=dict(width=3, color="gray"),
                                    opacity=0.6,
                                    showlegend=j == 0,
                                )
                            )
            else:
                # 单条轨迹处理（保持原有逻辑）
                waypoints_world = traj_world
                fig_data.append(
                    go.Scatter3d(
                        x=waypoints_world[:, 0],
                        y=waypoints_world[:, 1],
                        z=waypoints_world[:, 2],
                        mode="markers",
                        name="single_traj",
                        line=dict(width=2, color="gray"),
                    )
                )
                # 添加路径连线
                for i in range(waypoints_world.shape[0] - 1):
                    fig_data.append(
                        go.Scatter3d(
                            x=waypoints_world[i : i + 2, 0],
                            y=waypoints_world[i : i + 2, 1],
                            z=waypoints_world[i : i + 2, 2],
                            mode="lines",
                            name="path",
                            line=dict(width=8, color="orange"),
                        )
                    )

        # plot costmap
        if "costmap" in info:
            costmap = info["costmap"][:: self.downsample_ratio, :: self.downsample_ratio, :: self.downsample_ratio]
            skip_ratio = (self.workspace_bounds_max - self.workspace_bounds_min) / (
                self.map_size / self.downsample_ratio
            )
            x, y, z = np.mgrid[
                self.workspace_bounds_min[0] : self.workspace_bounds_max[0] : skip_ratio[0],
                self.workspace_bounds_min[1] : self.workspace_bounds_max[1] : skip_ratio[1],
                self.workspace_bounds_min[2] : self.workspace_bounds_max[2] : skip_ratio[2],
            ]
            grid_shape = costmap.shape
            x = x[: grid_shape[0], : grid_shape[1], : grid_shape[2]]
            y = y[: grid_shape[0], : grid_shape[1], : grid_shape[2]]
            z = z[: grid_shape[0], : grid_shape[1], : grid_shape[2]]
            fig_data.append(
                go.Volume(
                    x=x.flatten(),
                    y=y.flatten(),
                    z=z.flatten(),
                    value=costmap.flatten(),
                    isomin=0,
                    isomax=1,
                    opacity=self.costmap_opacity,
                    surface_count=self.costmap_surface_count,
                    colorscale="Jet",
                    showlegend=True,
                    showscale=False,
                )
            )
        # plot start position
        if "start_pos_world" in info:
            start_pos_world = np.array(info["start_pos_world"]).squeeze()
            fig_data.append(
                go.Scatter3d(
                    x=[start_pos_world[0]],
                    y=[start_pos_world[1]],
                    z=[start_pos_world[2]],
                    mode="markers",
                    name="start",
                    marker=dict(size=6, color="blue"),
                )
            )
        # plot target as dots extracted from target_map
        if "raw_target_map" in info:
            targets_world = info["targets_world"]
            fig_data.append(
                go.Scatter3d(
                    x=targets_world[:, 0],
                    y=targets_world[:, 1],
                    z=targets_world[:, 2],
                    mode="markers",
                    name="target",
                    marker=dict(size=6, color="green", opacity=0.7),
                )
            )

        if "sample_pos_world" in info:
            sample_pos_world = np.array(info["sample_pos_world"])
            fig_data.append(
                go.Scatter3d(
                    x=sample_pos_world[:, 0],
                    y=sample_pos_world[:, 1],
                    z=sample_pos_world[:, 2],
                    mode="markers",
                    name="sample",
                    marker=dict(size=6, color="yellow", opacity=0.7),
                )
            )

        # visualize scene points
        if self.scene_points is None:
            print("no scene points to overlay, skipping...")
            scene_points = None
        else:
            scene_points, scene_point_colors = self.scene_points
            scene_points = scene_points.astype(np.float32)
            # resample to reduce the number of points
            if scene_points.shape[0] > self.max_scene_points:
                resample_idx = np.random.choice(
                    scene_points.shape[0], min(scene_points.shape[0], self.max_scene_points), replace=False
                )
                scene_points = scene_points[resample_idx]
                if scene_point_colors is not None:
                    scene_point_colors = scene_point_colors[resample_idx]
            if scene_point_colors is None:
                scene_point_colors = scene_points[:, 2]
            else:
                scene_point_colors = scene_point_colors / 255.0
            # add scene points
            fig_data.append(
                go.Scatter3d(
                    x=scene_points[:, 0],
                    y=scene_points[:, 1],
                    z=scene_points[:, 2],
                    mode="markers",
                    marker=dict(size=3, color=scene_point_colors, opacity=1.0),
                )
            )

        fig = go.Figure(data=fig_data)

        # set bounds and ratio
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[self.plot_bounds_min[0], self.plot_bounds_max[0]], autorange=False),
                yaxis=dict(range=[self.plot_bounds_min[1], self.plot_bounds_max[1]], autorange=False),
                zaxis=dict(range=[self.plot_bounds_min[2], self.plot_bounds_max[2]], autorange=False),
            ),
            scene_aspectmode="manual",
            scene_aspectratio=dict(x=self.scene_scale[0], y=self.scene_scale[1], z=self.scene_scale[2]),
        )

        # do not show grid and axes
        fig.update_layout(
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title="", visible=False),
                yaxis=dict(showgrid=False, showticklabels=False, title="", visible=False),
                zaxis=dict(showgrid=False, showticklabels=False, title="", visible=False),
            )
        )

        # set background color as white
        fig.update_layout(template="none")

        # save and show
        if save and self.save_dir is not None:
            curr_time = datetime.datetime.now()
            log_id = f"{curr_time.hour}:{curr_time.minute}:{curr_time.second}"
            save_path = os.path.join(self.save_dir, log_id + ".html")
            latest_save_path = os.path.join(self.save_dir, "latest.html")
            print("** saving visualization to", save_path, "...")
            fig.write_html(save_path)
            print("** saving visualization to", latest_save_path, "...")
            fig.write_html(latest_save_path)
            print(f"** save to {save_path}")
        if show:
            fig.show()

        return fig

    def visualize_scene(self, show=False, save=True):
        """visualize the path and relevant info using plotly"""
        # visualize scene points
        scene_points, scene_point_colors = self.scene_points
        print(f"Scene points min: {scene_points.min(axis=0)}")
        print(f"Scene points max: {scene_points.max(axis=0)}")

        if scene_point_colors is not None:
            print(f"Scene point colors min: {scene_point_colors.min(axis=0)}")
            print(f"Scene point colors max: {scene_point_colors.max(axis=0)}")
        else:
            print("Scene point colors are None")

        print(f"Number of scene points after sampling: {scene_points.shape[0]}")

        print(f"Scene points shape: {scene_points.shape}")
        print(f"Scene points dtype: {scene_points.dtype}")

        fig_data = []

        if self.scene_points is None:
            print("no scene points to overlay, skipping...")
            scene_points = None
        else:
            scene_points, scene_point_colors = self.scene_points
            # 转换为 float32
            scene_points = scene_points.astype(np.float32)
            # 采样
            if scene_points.shape[0] > self.max_scene_points:
                resample_idx = np.random.choice(scene_points.shape[0], self.max_scene_points, replace=False)
                scene_points = scene_points[resample_idx]
                if scene_point_colors is not None:
                    scene_point_colors = scene_point_colors[resample_idx]
            # 处理颜色
            if scene_point_colors is None:
                scene_point_colors = scene_points[:, 2]
            else:
                scene_point_colors = scene_point_colors / 255.0
            # 添加点云
            fig_data.append(
                go.Scatter3d(
                    x=scene_points[:, 0],
                    y=scene_points[:, 1],
                    z=scene_points[:, 2],
                    mode="markers",
                    marker=dict(size=5, color=scene_point_colors, opacity=1.0),
                )
            )

        fig = go.Figure(data=fig_data)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-1, 1]),
                yaxis=dict(range=[-1, 1]),
                zaxis=dict(range=[0, 1]),
            )
        )

        # save and show
        if save and self.save_dir is not None:
            curr_time = datetime.datetime.now()
            log_id = f"{curr_time.hour}:{curr_time.minute}:{curr_time.second}"
            save_path = os.path.join(self.save_dir, log_id + ".html")
            latest_save_path = os.path.join(self.save_dir, "latest.html")
            print("** saving visualization to", save_path, "...")
            fig.write_html(save_path)
            print("** saving visualization to", latest_save_path, "...")
            fig.write_html(latest_save_path)
            print(f"** save to {save_path}")
        if show:
            fig.show()

        return fig
