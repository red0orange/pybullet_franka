#!/usr/bin/python3
from multiprocessing.connection import wait
import sys
import time
import copy
import re
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg

from math import pi, tau, dist, fabs, cos
from scipy.spatial.transform import Rotation as R
import cv2

import actionlib

from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list

from moveit_msgs.msg import PlanningScene, PlanningSceneWorld, PlanningSceneComponents
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene

from my_robot.msg import GraspAction

from utils.tf import *
import utils.geometry as geometry


def pose_msg_to_T(pose_msg):
    T = np.identity(4)
    T[:3, :3] = tf.transformations.quaternion_matrix(
        [pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w])[:3, :3]
    T[:3, 3] = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    return T

def T_to_pose_msg(T):
    pose_msg = geometry_msgs.msg.Pose()
    sevenDof = T2sevendof(T)
    pose_msg.position.x = sevenDof[0]
    pose_msg.position.y = sevenDof[1]
    pose_msg.position.z = sevenDof[2]
    pose_msg.orientation.x = sevenDof[3]
    pose_msg.orientation.y = sevenDof[4]
    pose_msg.orientation.z = sevenDof[5]
    pose_msg.orientation.w = sevenDof[6]
    return pose_msg


def pose_msg_to_c(pose_msg):
    position = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    orientation = [pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w]
    c = geometry.Coordinate(position=position, quaternion=orientation)
    return c

def c_to_pose_msg(c):
    pose_msg = geometry_msgs.msg.Pose()
    pose_msg.position.x = c.position[0]
    pose_msg.position.y = c.position[1]
    pose_msg.position.z = c.position[2]
    pose_msg.orientation.x = c.quaternion[0]
    pose_msg.orientation.y = c.quaternion[1]
    pose_msg.orientation.z = c.quaternion[2]
    pose_msg.orientation.w = c.quaternion[3]
    return pose_msg


def compute_z_rotation(T):
    # Extract rotation matrix from homogeneous transformation matrix
    rotation_matrix = T[:3, :3]
    # Compute z rotation from rotation matrix
    # Use the atan2 function to ensure correct quadrant for the angle
    z_rotation = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return z_rotation

def sort_grasp_poses(grasp_poses):
    # Sort grasp poses based on z rotation
    grasp_poses.sort(key=lambda pose: compute_z_rotation(pose))


# 计算齐次变换矩阵和xy平面的夹角
def calculate_angle_with_xy_plane(homo_matrix):
    # 提取旋转矩阵的z轴方向
    z_axis = homo_matrix[0:3, 2]
    # 计算和xy平面法向量(0,0,1)的夹角，由于计算的是和Z轴的夹角，所以直接取Z轴的Z分量即可
    cos_angle = z_axis[2]
    angle = np.arccos(np.clip(cos_angle, -1, 1)) # 取值范围防止计算错误
    # 判断z轴方向，若z轴向下则夹角为负
    if z_axis[2] < 0:
        angle = -angle
    # 转换为角度制
    angle = np.degrees(angle)
    return angle


class DemoMoveitInterface(object):
    def __init__(self) -> None:
        super(DemoMoveitInterface, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("demo_moveit_interface", anonymous=True)

        self.server = actionlib.SimpleActionServer(
            "/grasp_interface", GraspAction, self.grasp_callback, False
        )
        self.server.start()

        # init
        robot_description = "/my_gen3/robot_description"
        ns = "/my_gen3"
        self.robot = moveit_commander.RobotCommander(robot_description=robot_description, ns=ns)
        # self.scene = moveit_commander.PlanningSceneInterface()
        self.arm_move_group = moveit_commander.MoveGroupCommander("arm", robot_description=robot_description, ns=ns)
        self.gripper_move_group = moveit_commander.MoveGroupCommander("gripper", robot_description=robot_description, ns=ns)

        # collision pub
        self.collision_pcd_pub = rospy.Publisher("/collision_pcd2", sensor_msgs.msg.PointCloud2, queue_size=1)

        # print franka state
        planning_frame = self.arm_move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)
        eef_link = self.arm_move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)
        group_names = self.robot.get_group_names()
        print("============ Available Planning Groups:", group_names)
        robot_cur_state = self.robot.get_current_state()
        joint_state = "\n".join(re.findall(
            "position" + '.*', str(robot_cur_state)))
        print("============ Printing robot state: {}".format(joint_state))
        pass

    @staticmethod
    def get_planning_scene():
        get_planning_scene_service_name = "/my_gen3/get_planning_scene"
        rospy.wait_for_service(get_planning_scene_service_name, 10.0)
        get_planning_scene = rospy.ServiceProxy(get_planning_scene_service_name, GetPlanningScene)
        request = PlanningSceneComponents()
        original_response = get_planning_scene(request)
        ori_planning_scene = original_response.scene
        return ori_planning_scene

    @staticmethod
    def apply_planning_scene(planning_scene):
        set_planning_scene_service_name = "/my_gen3/apply_planning_scene"
        rospy.wait_for_service(set_planning_scene_service_name, 10.0)
        apply_planning_scene = rospy.ServiceProxy(set_planning_scene_service_name, ApplyPlanningScene)
        apply_planning_scene(planning_scene)
        pass

    @staticmethod
    def translate_pose_msg(pose_msg, translation):
        c = pose_msg_to_c(pose_msg=pose_msg)
        c.translate(translation, wrt="local")
        pose_msg = c_to_pose_msg(c)
        return pose_msg

    def grasp_callback(self, grasp_goal):
        rospy.loginfo("Waiting env pcd pointcloud")
        rgb_msg = rospy.wait_for_message("/env_pcd2", sensor_msgs.msg.PointCloud2)

        # == begin
        grasp_pose = grasp_goal.grasp_pose
        header = grasp_pose.header

        # debug
        grasp_poses = grasp_goal.debug_grasp_poses
        grasp_Ts     = [pose_msg_to_T(pose.pose) for pose in grasp_poses]
        angle_z_xy_plane = [abs(calculate_angle_with_xy_plane(T)) for T in grasp_Ts]
        print(angle_z_xy_plane)
        grasp_Ts = [grasp_Ts[i] for i in range(len(grasp_Ts)) if angle_z_xy_plane[i] > 120]
        grasp_Ts = sorted(grasp_Ts, key=lambda x: x[2, 3], reverse=False)

        grasp_T = grasp_Ts[0]

        # debug
        # grasp_pose = sorted(grasp_poses, key=lambda x: x.pose.position.y, reverse=True)[len(grasp_poses) // 2]
        grasp_pose = T_to_pose_msg(grasp_T)
        grasp_pose = geometry_msgs.msg.PoseStamped(pose=grasp_pose, header=header)
        # 按照

        # clear planning scene and wait for update
        ori_planning_scene = self.get_planning_scene()
        ori_planning_scene.world = PlanningSceneWorld()
        self.apply_planning_scene(ori_planning_scene)
        for i in range(10):
            self.collision_pcd_pub.publish(grasp_goal.full_cloud)
            time.sleep(0.1)
        time.sleep(1)

        # == moveit pre grasp
        print("moving to pre grasp pose")
        # pose_goal = grasp_pose.pose
        pose_goal = self.translate_pose_msg(grasp_pose.pose, [0, 0, -0.02])
        print(pose_goal)
        assert(type(pose_goal) == geometry_msgs.msg.Pose)
        self.arm_move_group.set_pose_target(pose_goal)
        success = self.arm_move_group.go(wait=True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()

        # clear planning scene and wait for update
        ori_planning_scene = self.get_planning_scene()
        ori_planning_scene.world = PlanningSceneWorld()
        self.apply_planning_scene(ori_planning_scene)
        for i in range(10):
            self.collision_pcd_pub.publish(grasp_goal.env_cloud)
            time.sleep(0.1)
        time.sleep(1)

        # == moveit grasp
        print("moving to grasp pose")
        pose_goal = self.translate_pose_msg(grasp_pose.pose, [0, 0, 0.10])
        print(pose_goal)
        assert(type(pose_goal) == geometry_msgs.msg.Pose)
        self.arm_move_group.set_pose_target(pose_goal)
        success = self.arm_move_group.go(wait=True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()

        pass

    def verify_pose(goal, actual, tolerance):
        """
        Convenience method for testing if the values in two lists are within a tolerance of each other.
        For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
        between the identical orientations q and -q is calculated correctly).
        @param: goal       A list of floats, a Pose or a PoseStamped
        @param: actual     A list of floats, a Pose or a PoseStamped
        @param: tolerance  A float
        @returns: bool
        """
        if type(goal) is list:
            for index in range(len(goal)):
                if abs(actual[index] - goal[index]) > tolerance:
                    return False

        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return DemoMoveitInterface.verify_pose(goal.pose, actual.pose, tolerance)

        elif type(goal) is geometry_msgs.msg.Pose:
            x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
            x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
            # Euclidean distance
            d = dist((x1, y1, z1), (x0, y0, z0))
            # phi = angle between orientations
            cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
            return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

        return True

    def get_cur_pose(self):
        return self.arm_move_group.get_current_pose().pose

    def move_to_pose_goal(self, pose_goal):
        assert(type(pose_goal) == geometry_msgs.msg.Pose)

        self.arm_move_group.set_pose_target(pose_goal)
        success = self.arm_move_group.go(wait=True)

        cv2.imshow("click to excute", np.zeros((100, 100)))
        cv2.waitKey(0)

        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()

        return DemoMoveitInterface.verify_pose(pose_goal, self.get_cur_pose(), 0.01)


if __name__ == "__main__":
    demo_interface = DemoMoveitInterface()

    # # control down 0.05
    # cur_pose = demo_interface.get_cur_pose()
    # goal_pose = copy.copy(cur_pose)
    # goal_pose.position.z += 0.10
    # result = demo_interface.move_to_pose_goal(goal_pose)
    # print("result: {}".format("True" if result else "False"))

    rospy.spin()
    pass
