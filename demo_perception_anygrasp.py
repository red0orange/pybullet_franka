import os
import sys
import time
from datetime import datetime
import shutil
import cv2
# os.environ['QT_QPA_PLATFORM'] = 'xcb'
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull
import trimesh.transformations as tra

import tf
import rospy
import rospkg
import actionlib
import message_filters
from cv_bridge import CvBridge, CvBridgeError

import std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField, CameraInfo, Image
from geometry_msgs.msg import PoseStamped, Pose
from my_robot.msg import GraspAction, GraspGoal, GraspResult, GraspFeedback

from utils.T_7dof import *
from utils.tk_combo_box import *

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

project_root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root_dir, "anygrasp_sdk", "grasp_detection"))
from my_grasp_api import AnyGraspAPI

# @note 选择 baseline
# sys.path.append("/home/huangdehao/github_projects/taskgrasp_ws/GraspGPT/gcngrasp")
# sys.path.append("/home/huangdehao/github_projects/taskgrasp_ws/GraspGPT_exp/gcngrasp")
sys.path.append("/home/huangdehao/github_projects/taskgrasp_ws/TaskGrasp/gcngrasp")

from infer_sample import MyGraspGPTAPI


def dilate_mask(mask, kernel_size=5, iterations=1):
    # Convert to uint8
    mask_uint8 = (mask.astype(np.uint8) * 255)
    
    # Create a kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Dilate the mask
    dilation_uint8 = cv2.dilate(mask_uint8, kernel, iterations=iterations)
    
    # Convert back to bool and return
    return (dilation_uint8 > 0)


def publish_poses(pose_Ts, publisher):
    # Create a PoseStamped message object
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = 'base_link'  # Set the frame ID according to your setup

    # Iterate through the pose_T list and publish each pose
    for pose_T in pose_Ts:
        sevenDof = T2sevendof(T=pose_T)

        # Set the position and orientation of the pose
        pose_msg.pose.position.x = sevenDof[0]
        pose_msg.pose.position.y = sevenDof[1]
        pose_msg.pose.position.z = sevenDof[2]
        pose_msg.pose.orientation.x = sevenDof[3]
        pose_msg.pose.orientation.y = sevenDof[4]
        pose_msg.pose.orientation.z = sevenDof[5]
        pose_msg.pose.orientation.w = sevenDof[6]

        # Publish the pose
        publisher.publish(pose_msg)

        # Add a delay if needed to control the publishing rate
        rospy.sleep(0.1)  # Adjust the delay time as per your requirement
    pass


def filter_segment(contact_pts, segment_pc, thres=0.00001):
    """
    Filter grasps to obtain contacts on specified point cloud segment
    
    :param contact_pts: Nx3 contact points of all grasps in the scene
    :param segment_pc: Mx3 segmented point cloud of the object of interest
    :param thres: maximum distance in m of filtered contact points from segmented point cloud
    :returns: Contact/Grasp indices that lie in the point cloud segment
    """
    filtered_grasp_idcs = np.array([],dtype=np.int32)
    
    if contact_pts.shape[0] > 0 and segment_pc.shape[0] > 0:
        try:
            dists = contact_pts[:,:3].reshape(-1,1,3) - segment_pc.reshape(1,-1,3)           
            min_dists = np.min(np.linalg.norm(dists,axis=2),axis=1)
            filtered_grasp_idcs = np.where(min_dists<thres)
        except:
            pass
        
    return filtered_grasp_idcs


def center_filter_segment(center_pts, segment_pc, thres=0.05):
    filtered_grasp_idcs = np.array([],dtype=np.int32)
    
    if center_pts.shape[0] > 0 and segment_pc.shape[0] > 0:
        try:
            dists = center_pts[:,:3].reshape(-1,1,3) - segment_pc.reshape(1,-1,3)           
            min_dists = np.min(np.linalg.norm(dists,axis=2),axis=1)
            filtered_grasp_idcs = np.where(min_dists<thres)
        except:
            pass
        
    return filtered_grasp_idcs


def project_to_xy(T):
    # Extract rotation matrix R from the homogeneous transformation matrix T
    R = T[:3, :3]
    
    # Calculate the rotation angles around x, y, and z axes
    # using the Euler angles representation (roll, pitch, yaw)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    z_axis = R[:, 2]
    z_direction = 'Up' if z_axis[2] > 0 else 'Down'
    
    # Create a new rotation matrix R_new using the extracted Euler angles
    pitch = 0
    if z_direction == 'Down':
        pitch = np.pi
    R_direction_correction = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    R_new = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R_new = np.matmul(R_new, R_direction_correction)
    
    # Combine the new rotation matrix R_new with the original translation vector
    T_new = np.eye(4)
    T_new[:3, :3] = R_new
    T_new[:3, 3] = T[:3, 3]
    
    return T_new


def read_matrices_from_txt(filepath):
    # Open the file in read mode
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Initialize an empty dictionary to hold matrices
    matrices = {}

    # Iterate over the lines in the file
    i = 0
    while i < len(lines):
        if lines[i].strip():
            matrix_name = lines[i].strip()
            matrix_lines = []
            i += 1

            # Read the matrix lines until an empty line or end of file is reached
            while i < len(lines) and lines[i].strip():
                matrix_lines.append(list(map(float, lines[i].split())))
                i += 1

            # Convert list of lists into numpy array and add to dictionary
            matrices[matrix_name] = np.array(matrix_lines)
        else:
            i += 1

    return matrices


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Mouse click at coordinates : ', x, ',', y)
        param.append((x, y))

def get_click_coordinates(image):
    # create an empty list to store the coordinates
    coordinates = []

    # display the image in a window
    cv2.imshow('image', image)

    # set the callback function for mouse events
    cv2.setMouseCallback('image', click_event, param=coordinates)

    # wait until any key is pressed
    cv2.waitKey(0)

    # return the coordinates
    return coordinates


def apply_mask(image, mask):
    if mask.dtype != "uint8":
        mask = (mask * 255).astype(image.dtype)
    # 将掩码应用于图像
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def filter_point_cloud(point_cloud, colors=None, downsample_ratio=None, min_neighbors=40, radius=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32) / 255.0)

    pcd, _ = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    if downsample_ratio is not None:
        pcd = pcd.random_down_sample(downsample_ratio)

    if colors is not None:
        return np.asarray(pcd.points), np.asarray(pcd.colors)
    return np.asarray(pcd.points)


def compute_bounding_box(points, scale):
    # 计算点云的凸包
    hull = ConvexHull(points)

    # 提取凸包的顶点坐标
    vertices = points[hull.vertices]

    # 找到最小外接矩形的边界坐标
    x_min, y_min, z_min = np.min(vertices, axis=0)
    x_max, y_max, z_max = np.max(vertices, axis=0)

    # 稍微放大矩形
    scale = scale - 1.0
    x_min -= scale * (x_max - x_min)
    x_max += scale * (x_max - x_min)
    y_min -= scale * (y_max - y_min)
    y_max += scale * (y_max - y_min)
    z_min -= scale * (z_max - z_min)
    z_max += scale * (z_max - z_min)

    return x_min, x_max, y_min, y_max, z_min, z_max


def numpy_to_pointcloud2(points_array, rgb=None, frame_id="base_link"):
    """
    Convert a nx3 points array to a sensor_msgs/PointCloud2 message.

    Parameters:
    points_array: A nx3 numpy array representing a point cloud.
    rgb: An optional nx3 numpy array representing the RGB color of each point.
    frame_id: The frame in which the point cloud will be published (default: "base_link").

    Returns:
    sensor_msgs/PointCloud2: A sensor_msgs/PointCloud2 message containing the points from the input array.
    """
    # Create header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Create fields
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
    ]

    if rgb is not None:
        fields.append(PointField(name='rgb', offset=12, datatype=PointField.UINT8, count=3))

    # Convert numpy array to list
    points_list = points_array.flatten().tolist()

    if rgb is not None:
        points_list += rgb.flatten().tolist()

    # Create PointCloud2 message
    pc2 = PointCloud2(
        header=header,
        height=1,
        width=len(points_array),
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=12 + (3 if rgb is not None else 0),
        row_step=(12 + (3 if rgb is not None else 0)) * len(points_array),
        data=np.asarray(points_list, np.float32).tostring()
    )

    return pc2


def downsample_pc(pc, nsample):
    if pc.shape[0] < nsample:
        print(
            'Less points in pc {}, than target dimensionality {}'.format(
                pc.shape[0],
                nsample))
    chosen_one = np.random.choice(
        pc.shape[0], nsample, replace=pc.shape[1] > nsample)
    return pc[chosen_one, :], chosen_one


def visualize_point_cloud(point_cloud, pcd_color=None, bbox=None, poses=None):
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    
    # 将点云数组转换为Open3D点云格式
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if pcd_color is not None:
        pcd.colors = o3d.utility.Vector3dVector(pcd_color)

    if bbox is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        bbox = o3d.geometry.AxisAlignedBoundingBox([x_min, y_min, z_min], [x_max, y_max, z_max])
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # 将点云添加到可视化窗口
    vis.add_geometry(pcd)
    if poses is not None:
        for pose in poses:
            # 创建一个单位坐标轴
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01)

            # 转换位姿为4x4的矩阵
            pose_matrix = np.eye(4)
            pose_matrix[:3, :3] = pose[:3, :3]
            pose_matrix[:3, 3] = pose[:3, 3]

            # 将单位坐标轴应用于位姿矩阵
            coord_frame.transform(pose_matrix)

            vis.add_geometry(coord_frame)

    if bbox is not None:
        bbox.color = (1, 0, 0)
        vis.add_geometry(bbox)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 0.01  # 设置点的大小
    
    # 运行可视化窗口
    vis.run()
    
    # 关闭可视化窗口
    vis.destroy_window()


def depth2pc(depth, K, rgb=None, mask=None, max_depth=1.0):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    if mask is not None:
        mask = np.where((depth < max_depth) & (depth > 0) & (mask != 0))
    else:
        mask = np.where((depth < max_depth) & (depth > 0))
    x,y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]
        
    pc = np.vstack((world_x, world_y, world_z)).T
    return pc, rgb, np.vstack((x,y)).T


def T2pose(T):
    pose_msg = Pose()
    sevenDof_pose = T2sevendof(T)
    pose_msg.position.x = sevenDof_pose[0]
    pose_msg.position.y = sevenDof_pose[1]
    pose_msg.position.z = sevenDof_pose[2]
    pose_msg.orientation.x = sevenDof_pose[3]
    pose_msg.orientation.y = sevenDof_pose[4]
    pose_msg.orientation.z = sevenDof_pose[5]
    pose_msg.orientation.w = sevenDof_pose[6]
    return pose_msg


class Sampler(object):
    def __init__(self, rgb_topic_name, depth_topic_name, camera_info_topic_name):
        # == tk
        task_list = ['pour', 'scoop', 'cut', 'spray', 'brush', 'handover', 'mix']
        obj_list = ['saucepan.n.01', 'mug.n.01', 'bottle.n.01', 'cup.n.01', 'bottle.n.01', 'spatula.n.01', 'ladle.n.01', 'knife.n.01', 'hammer.n.01', 'book.n.01', 'tongs.n.01', 'fork.n.01', 'whisk.n.01', 'tablespoon.n.01', 'brush.n.01', 'screwdriver.n.01']
        combox_box = run_combo_box(cands_1=task_list, cands_2=obj_list)

        # == graspgpt
        self.graspgpt_api = MyGraspGPTAPI()

        # == SAM
        # self.sam = sam_model_registry["vit_b"](checkpoint="/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/sam_vit_b_01ec64.pth")
        self.sam = sam_model_registry["vit_h"](checkpoint="/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/sam_vit_h_4b8939.pth")
        self.sam.cuda()
        # self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.predictor = SamPredictor(self.sam)

        # == anygrasp API
        self.grasp_interface = AnyGraspAPI()

        # == image subscriber
        self.tf_listener = tf.listener.TransformListener()
        self.bridge = CvBridge()

        # == for debug
        self.object_pcd2_pub = rospy.Publisher("/object_pcd2", PointCloud2, queue_size=1)
        self.env_pcd2_pub    = rospy.Publisher("/env_pcd2", PointCloud2, queue_size=1)
        self.grasp_pose_pub = rospy.Publisher("/debug_grasp_poses", PoseStamped, queue_size=1)

        # == begin
        rospy.loginfo("Waiting grasp interface")
        self.client = actionlib.SimpleActionClient("/grasp_interface", GraspAction)
        self.client.wait_for_server()

        rospy.loginfo("Waiting camera_info")
        camera_info = rospy.wait_for_message(camera_info_topic_name, CameraInfo)
        self.fx, self.fy, self.cx, self.cy = (
            camera_info.K[0],
            camera_info.K[4],
            camera_info.K[2],
            camera_info.K[5],
        )
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.image_width, self.image_height = camera_info.width, camera_info.height
        rospy.loginfo("Get camera_info")

        # == 主动进入一次采样
        while True:
            save_data_dict = {
                "K": None,
                "image": None,
                "depth": None,
                "world_pc": None,
                "world_pc_color": None,
                "world_grasps": None,
                "camera_pose": None,
            }

            rospy.loginfo("================Click to one sample!================")
            cv2.namedWindow("click to one sample", cv2.WINDOW_NORMAL)
            cv2.imshow("click to one sample", np.zeros([100, 100]))
            cv2.waitKey(0)
            # cv2.destroyAllWindows()

            rospy.loginfo("Getting RGBD Image!")
            rgb_msg = rospy.wait_for_message(rgb_topic_name, Image)
            depth_msg = rospy.wait_for_message(depth_topic_name, Image)
            rospy.loginfo("Get RGBD Image!")
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            depth_image = depth_image.astype(np.float32) / 1000.0
            rospy.loginfo("Obtaining Pose!")
            # ee_pose_T = self.get_cur_pose()
            tool_frame = "tool_frame"
            base_frame = "base_link"
            self.tf_listener.waitForTransform(base_frame, tool_frame, rospy.Time(), rospy.Duration(4.0))
            (trans, quat) = self.tf_listener.lookupTransform(base_frame, tool_frame, rospy.Time(0))
            ee_pose_T = sevenDof2T(list(trans) + list(quat))

            data = "0.0318925 0.0342034 -0.0573687 0.00927742 0.000113984 0.999956 -0.000995543"
            ee_to_camera_pose = np.array(list(map(float, data.split())))

            ee_to_camera_T = sevenDof2T(ee_to_camera_pose)
            camera_pose_T = np.dot(ee_pose_T, ee_to_camera_T)
            rospy.loginfo("Obtain Pose!")
            output = self.test_input(rgb_image, depth_image, self.K, pcd_T=camera_pose_T)
            if output is None:
                continue
            grasp_Ts, object_pc, env_pc, full_pc = output
            object_pc, object_pc_color = object_pc[:, :3], object_pc[:, 3:]

            object_pc, object_pc_color = filter_point_cloud(object_pc, colors=object_pc_color, min_neighbors=40, radius=0.01)
            # height = 0.15
            # object_pc_color = np.array([object_pc_color[i] for i in range(len(object_pc)) if object_pc[i][2] > height])
            # object_pc = np.array([p for p in object_pc if p[2] > height])

            save_data_dict["K"] = self.K
            save_data_dict["image"] = rgb_image
            save_data_dict["depth"] = (depth_image * 1000.0).astype(np.uint16)
            save_data_dict["world_pc"] = object_pc
            save_data_dict["world_pc_color"] = object_pc_color
            save_data_dict["world_grasps"] = grasp_Ts
            save_data_dict["camera_pose"] = camera_pose_T

            current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
            save_dir = "/home/huangdehao/github_projects/taskgrasp_ws/GraspGPT/data/sample_data/test_" + current_datetime
            if os.path.exists(save_dir): shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=False)
            np.save(os.path.join(save_dir, "ori_data.npy"), save_data_dict)
            np.save(os.path.join(save_dir, "fused_pc_clean.npy"), np.concatenate([save_data_dict["world_pc"], save_data_dict["world_pc_color"]], axis=1))
            np.save(os.path.join(save_dir, "fused_grasps_clean.npy"), save_data_dict["world_grasps"])
            np.save(os.path.join(save_dir, "fused_pc.npy"), np.concatenate([save_data_dict["world_pc"], save_data_dict["world_pc_color"]], axis=1))
            np.save(os.path.join(save_dir, "fused_grasps.npy"), save_data_dict["world_grasps"])

            cv2.imwrite(os.path.join(save_dir, "0_color.png"), save_data_dict["image"])
            np.save(os.path.join(save_dir, "0_color.npy"), save_data_dict["image"])
            np.save(os.path.join(save_dir, "0_depth.npy"), save_data_dict["depth"])
            np.save(os.path.join(save_dir, "0_camera_info.npy"), save_data_dict["camera_pose"])
            np.save(os.path.join(save_dir, "0_camera_pose.npy"), save_data_dict["camera_pose"])

            # # == TODO 测试阶段，不操作
            obj_name = "test_" + current_datetime

            task = combox_box.value1
            obj_class = combox_box.value2
            best_grasps, best_grasp, probs = self.graspgpt_api.infer(task=task, obj_class=obj_class, obj_name=obj_name)
            best_grasp = best_grasps[0]

            # 针对物体决定是否旋转 180°
            obj_cat = "pot"
            if obj_cat == "pot":
                object_pc_center = np.mean(object_pc, axis=0)

                grasp_center = best_grasp[:3, 3]

                transform = tra.euler_matrix(0, 0, 0)
                transform[1, 3] = 0.01
                forward_best_grasp = np.array(np.matmul(best_grasp, transform))
                forward_grasp_center = forward_best_grasp[:3, 3]

                normal_distance = np.linalg.norm(object_pc_center - grasp_center)
                forward_distance = np.linalg.norm(object_pc_center - forward_grasp_center)
                
                if forward_distance > normal_distance:
                    transform = tra.euler_matrix(0, 0, np.deg2rad(180))
                    best_grasp = np.array(np.matmul(best_grasp, transform))

                if topdown:
                    # 投影为 top-down
                    transform = tra.euler_matrix(0, 0, 0)
                    transform[2, 3] = 0.07
                    transform_inv = np.linalg.inv(transform)
                    best_grasp = np.array(np.matmul(best_grasp, transform))
                    best_grasp = project_to_xy(best_grasp)
                    best_grasp = np.array(np.matmul(best_grasp, transform_inv))

            best_grasps[0] = best_grasp


            graspgpt_data_dict = {
                "task": task,
                "obj_class": obj_class,
                "obj_name": obj_name,

                "best_grasps": best_grasps,
                "best_grasp": best_grasp,
                "probs": probs,
            }
            np.save(os.path.join(save_dir, "graspgpt_data.npy"), graspgpt_data_dict)

            cv2.namedWindow("exit to one sample", cv2.WINDOW_NORMAL)
            cv2.imshow("exit to one sample", np.zeros([100, 100]))
            key = cv2.waitKey(0)
            if key == ord("q"):
                continue

            # goal_grasp_T = grasps[0]
            goal_grasp_T = best_grasp
            # continue

            # == 发送
            goal = GraspGoal()

            goal.full_cloud = numpy_to_pointcloud2(full_pc, frame_id="base_link")
            goal.env_cloud = numpy_to_pointcloud2(env_pc, frame_id="base_link")
            goal.obj_cloud = numpy_to_pointcloud2(object_pc, frame_id="base_link")

            # goal_grasp_T = grasp_Ts[0]
            goal.grasp_pose.pose = T2pose(goal_grasp_T)

            # debug
            # pose_stamp_msgs = [PoseStamped(pose=T2pose(T)) for T in grasp_Ts]
            pose_stamp_msgs = [PoseStamped(pose=T2pose(T)) for T in best_grasps]
            for pose_stamp_msg in pose_stamp_msgs:
                goal.debug_grasp_poses.append(pose_stamp_msg)
            
            self.client.send_goal(goal)

            for i in range(20):
                self.object_pcd2_pub.publish(numpy_to_pointcloud2(object_pc, frame_id="base_link"))
                self.env_pcd2_pub.publish(numpy_to_pointcloud2(env_pc, frame_id="base_link"))
                time.sleep(0.5)

            # rate = rospy.Rate(hz=5)
            # cnt = 0
            # while not rospy.is_shutdown():
            #     cnt += 1
            #     if cnt > 20:
            #         break

            #     self.object_pcd2_pub.publish(numpy_to_pointcloud2(object_pc, frame_id="base_link"))
            #     self.env_pcd2_pub.publish(numpy_to_pointcloud2(env_pc, frame_id="base_link"))
            #     key = cv2.waitKey(1)
            #     if key == ord("q"):
            #         break

            #     rate.sleep()

            # self.client.wait_for_result()
            # result = self.client.get_result()

            # cv2.destroyAllWindows()
            rospy.loginfo("================Finish!================")

        # self.rgb_subscriber = message_filters.Subscriber(rgb_topic_name, Image)
        # self.depth_subscriber = message_filters.Subscriber(depth_topic_name, Image)
        # self.rgbd_subscriber = message_filters.ApproximateTimeSynchronizer(
        #     [self.rgb_subscriber, self.depth_subscriber],
        #     1,
        #     0.5,
        # )
        # self.rgbd_subscriber.registerCallback(self.rgbd_cb)
        # self.flag = True

        pass

    def test_input(self, rgb_image, depth_image, K, pcd_T=None):
        # DEBUG = True
        DEBUG = False

        # == 完整点云进行 Grasp Predict
        full_pcd, object_pcd_color, xy = depth2pc(depth_image, K, rgb=rgb_image, max_depth=0.9)
        # @note predict grasp
        Ts, grasp_scores = self.grasp_interface.infer(full_pcd, object_pcd_color)
        
        x_z_T = np.array([
            0, 0, 1, 0,
            0, 1, 0, 0,
            1, 0, 0, 0,
            0, 0, 0, 1
        ]).reshape([4, 4])
        Ts = [np.matmul(T, x_z_T) for T in Ts]
        rot_T = np.array([
            0, -1, 0, 0,
            1, 0, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        ]).reshape([4, 4])
        Ts = [np.matmul(T, rot_T) for T in Ts]
                
        center_pts = np.array([T[:3, 3] for T in Ts])
        if len(Ts) < 10:
            rospy.loginfo("No Grasp!")
            return
        if DEBUG: my_visualize_grasps(full_pcd, Ts, grasp_scores)  # debug

        # == prompt point 进行 SAM 物体选择
        prompt_point = get_click_coordinates(rgb_image)[-1]
        self.predictor.set_image(rgb_image)
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array([prompt_point]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
        masks = sorted(masks, key=lambda x: np.sum(x), reverse=True)
        masks = [i for i in masks if np.sum(i) < 50000]  # @DEBUG 

        # Select default
        mask = masks[0]
        mask = dilate_mask(mask, kernel_size=5, iterations=2)
        # # if DEBUG: 
        # if True:
        #     cv2.imshow("mask", mask.astype(np.uint8) * 255)  # debug
        #     key = cv2.waitKey(0)
        #     if key == ord("q"):
        #         return None

        # 手动选择点云
        for test_mask in masks:
            cv2.imshow("test_mask", test_mask.astype(np.uint8) * 255)  # debug
            key = cv2.waitKey(0)
            if key == ord("q"):
                continue
            else:
                mask = test_mask
                break

        # == 根据 mask 过滤物体的 Grasp
        env_pc, _, _ = depth2pc(depth_image, K, mask=np.bitwise_not(mask), rgb=rgb_image)
        env_pc = filter_point_cloud(env_pc, radius=0.01, min_neighbors=50)
        segment_pc, segment_pc_color, _ = depth2pc(depth_image, K, mask=mask, rgb=rgb_image)
        filtered_indexes = center_filter_segment(center_pts, segment_pc, thres=0.03)
        filtered_Ts = np.array(Ts)[filtered_indexes]
        filtered_grasp_scores = np.array(grasp_scores)[filtered_indexes]
        # filtered_Ts = filtered_Ts[filtered_grasp_scores > 0.2]
        if DEBUG: my_visualize_grasps(segment_pc, filtered_Ts, filtered_grasp_scores)  # debug

        trans_T = np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, -0.1,
            0, 0, 0, 1
        ]).reshape([4, 4])
        filtered_Ts = [np.matmul(T, trans_T) for T in filtered_Ts]

        # # @note debug visualization
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(segment_pc)
        # pcd.colors = o3d.utility.Vector3dVector(segment_pc_color / 255.0)
        # coord_systems = []
        # for i in range(len(filtered_Ts)):
        #     coord_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        #     coord_system.transform(filtered_Ts[i])
        #     coord_systems.append(coord_system)
        # vis_coord_systems = coord_systems
        # o3d.visualization.draw_geometries([*vis_coord_systems, pcd])

        # == 转换到 panda_link0 坐标系下
        if pcd_T is not None:
            filtered_Ts = [pcd_T @ i for i in filtered_Ts]
            segment_pc = (pcd_T @ np.concatenate([segment_pc, np.ones((segment_pc.shape[0], 1))], axis=1).T).T[:, :3]
            env_pc = (pcd_T @ np.concatenate([env_pc, np.ones((env_pc.shape[0], 1))], axis=1).T).T[:, :3]
            full_pcd = (pcd_T @ np.concatenate([full_pcd, np.ones((full_pcd.shape[0], 1))], axis=1).T).T[:, :3]
            if DEBUG: my_visualize_grasps(segment_pc, filtered_Ts, filtered_grasp_scores)  # debug

        segment_pc = np.concatenate([segment_pc, segment_pc_color], axis=1)
        return filtered_Ts, segment_pc, env_pc, full_pcd

        # publish_pcd = numpy_to_pointcloud2(object_pcd)
        # rate = rospy.Rate(hz=1)
        # while True:
        #     publish_poses([Ts[0]], self.grasp_pose_pub)
        #     self.pcd_pub.publish(publish_pcd)
        #     rate.sleep()
        # pass

    def test_input_2(self):
        object_pcd = np.load("/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/paint_roller.npy")
        object_pcd, object_pcd_color = object_pcd[:, :3], object_pcd[:, 3:]
        object_pcd, object_pcd_color = filter_point_cloud(object_pcd, object_pcd_color, min_neighbors=40, radius=0.01)

        # DEBUG
        # visualize_point_cloud(object_pcd)

        Ts, scores = self.grasp_interface.infer(object_pcd)

        my_visualize_grasps(object_pcd, Ts, scores, pc_colors=object_pcd_color)
        pass


def main():
    rospy.init_node("my_sampler_test_input", anonymous=True)

    camera_info_topic_name = "/camera/color/camera_info"
    rgb_topic_name = "/camera/color/image_raw"
    depth_topic_name = "/camera/aligned_depth_to_color/image_raw"

    sampler = Sampler(
        camera_info_topic_name=camera_info_topic_name,
        rgb_topic_name=rgb_topic_name,
        depth_topic_name=depth_topic_name,
    )

    # image_path = "/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/graspgpt_2/images/00000001.png"
    # depth_path = "/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/graspgpt_2/depth/00000001.png"
    # T_path = "/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/graspgpt_2/cams/00000001_cam.txt"
    # rgb_image = cv2.imread(image_path)
    # depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / 1000.0
    # K = np.array([
    #     601.3460693359375,0.0,316.9475402832031, 
    #     0.0,601.8807373046875,239.77159118652344,
    #     0.0,0.0,1.0 
    # ])
    # K = K.reshape([3, 3])
    # matrices = read_matrices_from_txt(T_path)
    # pcd_T = matrices["extrinsic"]

    # sampler.test_input(rgb_image, depth_image, K, pcd_T=pcd_T)
    # sampler.test_input_2()
    pass


if __name__ == "__main__":
    main()