import os
import sys
import cv2
# os.environ['QT_QPA_PLATFORM'] = 'xcb'
import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from scipy.spatial import ConvexHull

import rospy
import rospkg
import actionlib

from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
project_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_root_dir, "3rd_contact_graspnet", "contact_graspnet"))
from my_grasp_api import ContactGraspAPI
from visualization_utils import my_visualize_grasps


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
        pcd.colors = o3d.utility.Vector3dVector(colors)

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


def numpy_to_pointcloud2(points_array, frame_id="base_link"):
    """
    Convert a nx3 points array to a sensor_msgs/PointCloud2 message.

    Parameters:
    points_array: A nx3 numpy array representing a point cloud.
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

    # Convert numpy array to list
    points_list = points_array.flatten().tolist()

    # Create PointCloud2 message
    pc2 = PointCloud2(
        header=header,
        height=1,
        width=len(points_array),
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=12,
        row_step=12 * len(points_array),
        data=np.asarray(points_list, np.float32).tostring()
    )

    return pc2


def visualize_point_cloud(point_cloud, pcd_color=None, bbox=None):
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
    if bbox is not None:
        bbox.color = (1, 0, 0)
        vis.add_geometry(bbox)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 1.5  # 设置点的大小
    
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


class Sampler(object):
    def __init__(self):
        # self.sam = sam_model_registry["vit_b"](checkpoint="/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/sam_vit_b_01ec64.pth")
        self.sam = sam_model_registry["vit_h"](checkpoint="/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/sam_vit_h_4b8939.pth")
        self.sam.cuda()
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

        self.grasp_interface = ContactGraspAPI()

        # # for debug
        # self.pcd_pub = rospy.Publisher("/pcd", PointCloud2, queue_size=1)
        pass

    def sample(self):
        # TODO: 实际运行时，需要从摄像头获取图像
        pass

    def test_input(self, rgb_image, depth_image, K, pcd_T=None):
        # @note Debug: 显示整个点云场景
        # object_pcd, object_pcd_color, _ = depth2pc(depth_image, K, rgb=rgb_image)
        # visualize_point_cloud(object_pcd)

        # 测试输入
        print("generate masks")
        masks = self.mask_generator.generate(rgb_image)
        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        masks = [i for i in masks if i["area"] > 16000 and i["area"] < 20000]
        # masks = [i for i in masks if i["area"] > 18000 and i["area"] < 25000]
        # masks = [i for i in masks if i["area"] > 20000 and i["area"] < 30000]
        # masks = [i for i in masks if i["area"] > 28000 and i["area"] < 38000]
        # masks = [i for i in masks if i["area"] > 3000 and i["area"] < 5000]
        # masks = [i for i in masks if i["area"] > 6000 and i["area"] < 7000]
        # masks = sorted(masks, key=lambda x: x["bbox"][0] - (rgb_image.shape[1] / 2))

        segm = np.full((len(masks), *rgb_image.shape[:2]), fill_value=-1, dtype=np.uint8)
        for i, mask in enumerate(iterable=masks):
            segm[i, ...] = mask["segmentation"]

        # DEBUG
        for i in range(len(masks)):
            print("Area:", masks[i]["area"])
            # cv2.imshow(f"segm{i}", segm[i].astype(np.uint8) * 255)
            # cv2.waitKey(0)
            # plt.imshow(segm[i].astype(np.uint8) * 255)
            plt.imshow(apply_mask((depth_image * 255.0).astype(np.uint8), segm[i]))
            plt.show()
            # plt.imshow(apply_mask(rgb_image, segm[i]))
            # plt.show()

        object_pcd, object_pcd_color, _ = depth2pc(depth_image, K, mask=segm[0], rgb=rgb_image)
        # object_pcd, object_pcd_color = filter_point_cloud(object_pcd, object_pcd_color)
        if pcd_T is not None:
            object_pcd = (pcd_T @ np.concatenate([object_pcd, np.ones([object_pcd.shape[0], 1])], axis=1).T).T[:, :3]
        # DEBUG
        visualize_point_cloud(object_pcd)

        Ts, scores = self.grasp_interface.infer(object_pcd)

        my_visualize_grasps(object_pcd, Ts, scores, pc_colors=object_pcd_color)
        pass

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

    sampler = Sampler()

    # # test
    # image_path = "/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/2_color.png"
    # depth_path = "/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/2_depth.png"
    # K = np.array([[908.5330810546875, 0.0, 647.282470703125], [0.0, 909.3223876953125, 354.7244873046875], [0.0, 0.0, 1.0]])

    # rgb_image = cv2.imread(image_path)
    # depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / 1000.0

    # # pcd, _, _ = depth2pc(depth_image, K)
    # # visualize_point_cloud(pcd)

    # ori_image_width, ori_image_height = rgb_image.shape[1], rgb_image.shape[0]
    # rgb_image = cv2.resize(rgb_image, (ori_image_width // 2, ori_image_height // 2))
    # depth_image = cv2.resize(depth_image, (ori_image_width // 2, ori_image_height // 2))
    # K[0, 0] /= 2.0
    # K[0, 2] /= 2.0
    # K[1, 1] /= 2.0
    # K[1, 2] /= 2.0

    # object_pcd = np.load("/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/124_paint_roller/1_pc.npy")
    # object_pcd, object_pcd_color = object_pcd[:, :3], object_pcd[:, 3:]
    # visualize_point_cloud(point_cloud=object_pcd)

    # test 2
    # image_path = "/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/124_paint_roller/1_color.png"
    # depth_path = "/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/124_paint_roller/1_depth.npy"
    # K_path     = "/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/124_paint_roller/1_camerainfo.npy"
    # rgb_image = cv2.imread(image_path)
    # depth_image = np.load(depth_path) / 1000.0
    # K = np.load(K_path)

    # test3
    image_path = "/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/graspgpt_2/images/00000001.png"
    depth_path = "/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/graspgpt_2/depth/00000001.png"
    T_path = "/home/huangdehao/github_projects/graspgpt_ros_ws/src/my_sampler/test_dataset/graspgpt_2/cams/00000001_cam.txt"
    rgb_image = cv2.imread(image_path)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / 1000.0
    K = np.array([
        601.3460693359375,0.0,316.9475402832031, 
        0.0,601.8807373046875,239.77159118652344,
        0.0,0.0,1.0 
    ])
    K = K.reshape([3, 3])
    matrices = read_matrices_from_txt(T_path)
    pcd_T = matrices["extrinsic"]

    sampler.test_input(rgb_image, depth_image, K, pcd_T=pcd_T)
    # sampler.test_input_2()
    pass


if __name__ == "__main__":
    main()