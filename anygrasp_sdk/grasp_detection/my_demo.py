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

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps')
parser.add_argument('--debug', action='store_true', help='Enable visualization')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def demo(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # # get data
    # colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    # depths = np.array(Image.open(os.path.join(data_dir, 'depth.png')))

    # # get camera intrinsics
    # fx, fy = 927.17, 927.37
    # cx, cy = 651.32, 349.62
    # scale = 1000.0

    # # set workspace
    xmin, xmax = -0.19, 0.12
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # # get point cloud
    # xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    # xmap, ymap = np.meshgrid(xmap, ymap)
    # points_z = depths / scale
    # points_x = (xmap - cx) / fx * points_z
    # points_y = (ymap - cy) / fy * points_z

    # # remove outlier
    # mask = (points_z > 0) & (points_z < 1)
    # points = np.stack([points_x, points_y, points_z], axis=-1)
    # points = points[mask].astype(np.float32)
    # colors = colors[mask].astype(np.float32)
    # print(points.min(axis=0), points.max(axis=0))
    data_path = "/home/huangdehao/github_projects/anygrasp_sdk/grasp_detection/example_data_me/textured_simple.obj"
    mesh = o3d.io.read_triangle_mesh(data_path, True)
    rotation_matrix = mesh.get_rotation_matrix_from_xyz((0, 0, np.pi))
    mesh.rotate(rotation_matrix, center=(0, 0, 0))

    # o3d.visualization.draw_geometries([mesh])
    # point_cloud = mesh.sample_points_uniformly(number_of_points=10000)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = mesh.vertices
    point_cloud.colors = mesh.vertex_colors
    points = np.asarray(point_cloud.points).astype(np.float32)
    colors = np.asarray(point_cloud.colors).astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))
    pass

    # get prediction
    gg, cloud = anygrasp.get_grasp(points, colors, lims)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    gg_translation = [g.translation for g in gg]
    gg_translation_z = [g.translation[2] for g in gg]
    index_gg = np.argsort(np.array(gg_translation_z))
    gg_pick = gg[0:20]
    print(gg_pick.scores)
    print('grasp score:', gg_pick[0].score)

    # visualization
    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)

        # o3d.visualization.draw_geometries([*grippers, mesh])
        grippers = np.array(grippers)[index_gg][0:10]
        while True:
            grippers = random.sample(list(grippers), 6)
            o3d.visualization.draw_geometries([*grippers, mesh])
            # o3d.visualization.draw_geometries([grippers[0], mesh])

        # o3d.visualization.draw_geometries([*grippers, cloud])
        # o3d.visualization.draw_geometries([grippers[0], cloud])


if __name__ == '__main__':
    
    demo('./example_data/')
