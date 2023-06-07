import os
import open3d as o3d
import sys
import argparse
import numpy as np
import time
import glob
import cv2

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))
import config_utils as config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
# from contact_graspnet.visualization_utils import visualize_grasps, show_image


class ContactGraspAPI:
    def __init__(self) -> None:
        ckpt_dir = os.path.join(BASE_DIR, 'checkpoints', 'scene_test_2048_bs3_hor_sigma_001')
        global_config = config_utils.load_config(ckpt_dir, batch_size=1, arg_configs=[])

        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()

        # Add ops to save and restore all the variables.
        self.saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)

        # Load weights
        self.grasp_estimator.load_weights(self.sess, self.saver, ckpt_dir, mode='test')
        pass

    def infer(self, pc_full, pc_colors=None, pc_segments=None):
        # segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)
        segmap = None
        rgb, depth = None, None
        cam_K = None
        
        print('Generating Grasps...')
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(self.sess, pc_full, pc_segments=pc_segments)  

        # Save results
        # np.savez('results/predictions.npz', pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

        pred_grasps_cam = pred_grasps_cam[-1]
        scores = scores[-1]
        Ts = []
        for i in range(pred_grasps_cam.shape[0]):
            T = pred_grasps_cam[i, ...]
            Ts.append(T)

        return Ts, scores


if __name__ == "__main__":
    api = ContactGraspAPI()
    
    world_to_camera_T_txt_path = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/runs/2023-03-07-153759/0_Inbox/camera_to_world.txt"
    world_to_camera_T = np.loadtxt(world_to_camera_T_txt_path)

    pcd_path = "/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/runs/2023-03-07-142325/2_colmap_ws/acmh_dense/ACMP/ACMP_model.ply"
    pcd = o3d.io.read_point_cloud(pcd_path)
        
    pcd = pcd.transform(world_to_camera_T)

    # points = np.asarray(pcd.points)
    # points[:, 2] = -points[:, 2]
    # pcd.points = o3d.utility.Vector3dVector(points)

    pcd = pcd.remove_radius_outlier(nb_points=32, radius=0.005)[0]
    pcd = pcd.random_down_sample(0.2)
    # plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
    #                                     ransac_n=3,
    #                                     num_iterations=1000)
    # inlier_cloud = pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([1.0, 0, 0])
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # pcd = outlier_cloud

    # pcd = pcd.remove_radius_outlier(nb_points=32, radius=0.005)[0]

    pc_full = np.asarray(pcd.points)
    pc_colors = np.asarray(pcd.colors)

    api.infer(pc_full, pc_colors)
    pass