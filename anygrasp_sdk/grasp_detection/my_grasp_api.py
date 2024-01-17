import os
import argparse
import torch
import cv2
import random
import numpy as np
import open3d as o3d
from PIL import Image

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup


class AnyGraspAPI:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
        parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
        parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps')
        parser.add_argument('--debug', action='store_true', help='Enable visualization')
        cfgs = parser.parse_args()
        cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
        cfgs.checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", "checkpoint_detection.tar")
        self.cfgs = cfgs

        xmin, xmax = -0.12, 0.12
        ymin, ymax = -0.12, 0.12
        zmin, zmax = 0.0, 1.0
        self.lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        self.anygrasp = AnyGrasp(self.cfgs)
        self.anygrasp.load_net()
        pass

    def infer(self, pc_full, pc_colors=None, pc_segments=None):
        if pc_colors is None:
            pc_colors = np.ones_like(pc_full)
        
        gg, cloud = self.anygrasp.get_grasp(pc_full, pc_colors, self.lims)
        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
        gg = gg.nms().sort_by_score()
        gg_scores = gg.scores
        gg_rot_mat = gg.rotation_matrices
        gg_translation = gg.translations
        gg_T = np.zeros((len(gg), 4, 4))
        for i in range(len(gg)):
            gg_T[i, :3, :3] = gg_rot_mat[i]
            gg_T[i, :3, 3] = gg_translation[i]
            gg_T[i, 3, 3] = 1

        # # debug visualization
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # grippers = gg.to_open3d_geometry_list()
        # coord_systems = []
        # for i in range(len(gg)):
        #     coord_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        #     coord_system.transform(gg_T[i])
        #     coord_systems.append(coord_system)
        # vis_coord_systems = coord_systems[0:5]
        # vis_grippers = grippers[0:5]
        # o3d.visualization.draw_geometries([*vis_coord_systems, *vis_grippers, pcd])

        return gg_T, gg_scores


if __name__ == '__main__':
    anygrasp_api = AnyGraspAPI()

    from PIL import Image
    import open3d as o3d

    colors = np.array(Image.open("/home/huangdehao/github_projects/pybullet_franka/anygrasp_sdk/grasp_detection/example_data/color.png"), dtype=np.float32) / 255.0
    depths = np.array(Image.open("/home/huangdehao/github_projects/pybullet_franka/anygrasp_sdk/grasp_detection/example_data/depth.png"))

    # get camera intrinsics
    fx, fy = 927.17, 927.37
    cx, cy = 651.32, 349.62
    scale = 1000.0

    # set workspace
    xmin, xmax = -0.19, 0.12
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # remove outlier
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # get prediction
    gg = anygrasp_api.infer(points, colors)

    grippers = gg.to_open3d_geometry_list()

    gg_rot_mat = gg.rotation_matrices
    gg_translation = gg.translations
    gg_T = np.zeros((len(gg), 4, 4))
    for i in range(len(gg)):
        gg_T[i, :3, :3] = gg_rot_mat[i]
        gg_T[i, :3, 3] = gg_translation[i]
        gg_T[i, 3, 3] = 1
    
    coord_systems = []
    for i in range(len(gg)):
        coord_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        coord_system.transform(gg_T[i])
        coord_systems.append(coord_system)
    
    vis_coord_systems = coord_systems[0:5]
    vis_grippers = grippers[0:5]

    o3d.visualization.draw_geometries([*vis_coord_systems, *vis_grippers, pcd])
    o3d.visualization.draw_geometries([*grippers, pcd])
    o3d.visualization.draw_geometries([grippers[0], pcd])