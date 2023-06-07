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
import config_utils
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
# from visualization_utils import visualize_grasps, show_image

def inference(pc_full, pc_colors, global_config, checkpoint_dir, local_regions=True, skip_border_objects=False, filter_grasps=True, segmap_id=None, z_range=[0.2,1.8], forward_passes=1):
    """
    Predict 6-DoF grasp distribution for given model and input data
    
    :param global_config: config.yaml from checkpoint directory
    :param checkpoint_dir: checkpoint directory
    :param input_paths: .png/.npz/.npy file paths that contain depth/pointcloud and optionally intrinsics/segmentation/rgb
    :param K: Camera Matrix with intrinsics to convert depth to point cloud
    :param local_regions: Crop 3D local regions around given segments. 
    :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
    :param filter_grasps: Filter and assign grasp contacts according to segmap.
    :param segmap_id: only return grasps from specified segmap_id.
    :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
    :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
    """
    
    # Build the model
    grasp_estimator = GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(sess, saver, checkpoint_dir, mode='test')
    
    os.makedirs('results', exist_ok=True)

    # segmap, rgb, depth, cam_K, pc_full, pc_colors = load_available_input_data(p, K=K)
    segmap = None
    rgb, depth = None, None
    cam_K = None
    
    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(sess, pc_full)  

    

    # Save results
    np.savez('results/predictions.npz', pred_grasps_cam=pred_grasps_cam, scores=scores, contact_pts=contact_pts)

    # Visualize results          
    # show_image(rgb, segmap)
    # visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
    pass
        
        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default='checkpoints/scene_test_2048_bs3_hor_sigma_001', help='Log dir [default: checkpoints/scene_test_2048_bs3_hor_sigma_001]')
    # parser.add_argument('--np_path', default='test_data/7.npy', help='Input data: npz/npy file with keys either "depth" & camera matrix "K" or just point cloud "pc" in meters. Optionally, a 2D "segmap"')
    # parser.add_argument('--png_path', default='', help='Input data: depth map png in meters')
    # parser.add_argument('--K', default=None, help='Flat Camera Matrix, pass as "[fx, 0, cx, 0, fy, cy, 0, 0 ,1]"')
    parser.add_argument('--z_range', default=[0.2,1.8], help='Z value threshold to crop the input point cloud')
    parser.add_argument('--local_regions', action='store_true', default=False, help='Crop 3D local regions around given segments.')
    parser.add_argument('--filter_grasps', action='store_true', default=False,  help='Filter grasp contacts according to segmap.')
    parser.add_argument('--skip_border_objects', action='store_true', default=False,  help='When extracting local_regions, ignore segments at depth map boundary.')
    parser.add_argument('--forward_passes', type=int, default=1,  help='Run multiple parallel forward passes to mesh_utils more potential contact points.')
    parser.add_argument('--segmap_id', type=int, default=0,  help='Only return grasps of the given object id')
    parser.add_argument('--arg_configs', nargs="*", type=str, default=[], help='overwrite config parameters')
    FLAGS = parser.parse_args()

    global_config = config_utils.load_config(FLAGS.ckpt_dir, batch_size=FLAGS.forward_passes, arg_configs=FLAGS.arg_configs)
    
    print(str(global_config))
    print('pid: %s'%(str(os.getpid())))

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

    inference(pc_full, pc_colors, global_config, FLAGS.ckpt_dir)

    # vis
    predictions = np.load('results/predictions.npz', allow_pickle=True)
    pred_grasps_cam = predictions['pred_grasps_cam'].item()[-1]
    scores = predictions['scores'].item()[-1]
    axis_list = []
    for i in range(pred_grasps_cam.shape[0]):
        T = pred_grasps_cam[i, ...]
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08, origin=[0, 0, 0])
        axis_pcd.transform(T)
        axis_list.append(axis_pcd)
    o3d.visualization.draw_geometries(axis_list + [pcd])
    pass

