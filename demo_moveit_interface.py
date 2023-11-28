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
from kortex_driver.srv import *
from kortex_driver.msg import *

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
    sevenDof = np.array(sevenDof).astype(np.float64)
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


def compute_z_angle(T):
    # 提取旋转矩阵
    R = T[:3, :3]

    # 提取z轴的方向向量
    z_axis = R[:, 2]

    # 基坐标系xy平面的法向量
    n = np.array([0, 0, 1])

    # 计算两向量的点积
    dot_product = np.dot(z_axis, n)

    # 计算z轴向量的模
    norm_z = np.linalg.norm(z_axis)

    # 计算法向量的模
    norm_n = np.linalg.norm(n)

    # 计算角度
    cos_theta = dot_product / (norm_z * norm_n)

    # 用逆余弦函数得到角度（弧度）
    theta = np.arccos(cos_theta)

    theta = np.degrees(theta)

    theta = 180 - theta
    theta = 90 - theta

    return theta


def sort_grasp_poses(grasp_poses):
    # Sort grasp poses based on z rotation
    grasp_poses.sort(key=lambda pose: compute_z_rotation(pose))


# 计算齐次变换矩阵和xy平面的夹角
def calculate_angle_with_xy_plane(homo_matrix):
    # 提取旋转矩阵的z轴方向
    z_axis_vector = homo_matrix[0:3, 2]

    # 计算该向量与 XY 平面法向量的夹角
    xy_plane_normal = np.array([0, 0, 1])
    dot_product = np.dot(xy_plane_normal, z_axis_vector)

    # 使用点积公式计算夹角
    angle = np.arccos(dot_product / (np.linalg.norm(xy_plane_normal) * np.linalg.norm(z_axis_vector)))

    # 将弧度转换为度数，如果你需要的话
    angle_degrees = np.degrees(angle)

    return angle_degrees


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
        send_gripper_command_full_name = ns + '/base/send_gripper_command'
        rospy.wait_for_service(send_gripper_command_full_name)
        self.send_gripper_command = rospy.ServiceProxy(send_gripper_command_full_name, SendGripperCommand)

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

        print(self.get_cur_pose())

        self.back_to_home_state()
        pass

    def get_cur_pose(self):
        return self.arm_move_group.get_current_pose().pose

    def back_to_home_state(self, release_gripper=True):
        # gripper
        if release_gripper:
            self.example_send_gripper_command(value=0.2)

        # home pose
        # home_joint_goal = [0.03153623608649902, -0.9909734268950627, -3.017123961090542, -2.1596175153858903, -0.06072380127933297, -1.2676343188412904, 1.69711937836117]
        # home_joint_goal = [-0.006377738178521497, -0.555639266673043, 3.1360338374464707, -2.163956071117738, -0.014774685095541251, -1.0588206514581397, 1.564804081184164]
        home_joint_goal = [-0.039876576266192565, -1.2166917756136009, 3.0783202071472546, -2.010595010268326, 0.023798230124570448, -1.5123021239801782, 1.4359715329227085]
        self.arm_move_group.go(home_joint_goal, wait=True)
        pass

    def back_to_grasp_state(self):
        # grasp_state = [-0.003840278291893817, -0.3875597134270894, -3.1351339553141333, -2.2356531613526336, -0.0005928196586522105, -1.2238317104961034, 1.5737672161489347]
        grasp_state = [-0.03870052432881366, -0.3805699808300673, 3.104453806924202, -1.8106635177528076, -0.015250325666229081, -1.6416582497119432, 1.497566720510715]
        self.arm_move_group.go(grasp_state, wait=True)
        pass

    def go_to_specify_joint(self, joint):
        self.arm_move_group.go(joint, wait=True)
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
    def translate_pose_msg(pose_msg, translation, wrt="local"):
        c = pose_msg_to_c(pose_msg=pose_msg)
        c.translate(translation, wrt=wrt)
        pose_msg = c_to_pose_msg(c)
        return pose_msg

    def example_send_gripper_command(self, value):
        # Initialize the request
        # Close the gripper
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION

        rospy.loginfo("Sending the gripper command...")

        # Call the service 
        try:
            self.send_gripper_command(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            time.sleep(2)
            return True

    def grasp_callback(self, grasp_goal):
        enable_collision = False

        # == moveit pose grasp
        # self.back_to_home_state()
        self.back_to_grasp_state()

        # time.sleep(5)
        # self.back_to_home_state()

        # == begin
        grasp_pose = grasp_goal.grasp_pose
        header = grasp_pose.header

        # debug
        grasp_poses = grasp_goal.debug_grasp_poses
        grasp_Ts     = [pose_msg_to_T(pose.pose) for pose in grasp_poses]
        angle_z_xy_plane = [calculate_angle_with_xy_plane(T) for T in grasp_Ts]
        print(angle_z_xy_plane)

        def get_hard_T(grasp_Ts):
            hard_grasp_Ts = [grasp_Ts[i] for i in range(len(grasp_Ts)) if angle_z_xy_plane[i] < 100]
            hard_grasp_Ts = sorted(hard_grasp_Ts, key=lambda x: x[2, 3], reverse=True)

            if len(hard_grasp_Ts) == 0:
                return None

            grasp_T = hard_grasp_Ts[0]
            return grasp_T

        def get_simple_T(grasp_Ts):
            simple_grasp_Ts = [grasp_Ts[i] for i in range(len(grasp_Ts)) if angle_z_xy_plane[i] > 120]
            simple_grasp_Ts = sorted(simple_grasp_Ts, key=lambda x: x[2, 3], reverse=True)

            if len(simple_grasp_Ts) == 0:
                return None

            grasp_T = simple_grasp_Ts[0]
            return grasp_T

        # hard_grasp_T = get_hard_T(grasp_Ts)
        # simple_grasp_T = get_simple_T(grasp_Ts)
        # hard_grasp_T = simple_grasp_T

        # hard_grasp_T = get_simple_T(grasp_Ts)
        # simple_grasp_T = get_hard_T(grasp_Ts)
        grasp_Ts[0][1, 3] -= 0.03
        hard_grasp_T = grasp_Ts[0]
        simple_grasp_T = grasp_Ts[0]

        # if True:
        if False:
            if hard_grasp_T[:3, 1][0] > 0:
                pose = T2pybullet(T=hard_grasp_T)
                c = geometry.Coordinate(*pose)
                c.rotate(euler=[0, 0, np.deg2rad(180)], wrt="local")
                simple_grasp_T = pybullet2T(c.pose)
                # 旋转180
                pass
            else:
                pose = T2pybullet(T=hard_grasp_T)
                c = geometry.Coordinate(*pose)
                c.rotate(euler=[0, 0, np.deg2rad(180)], wrt="local")
                hard_grasp_T = pybullet2T(c.pose)

        # if hard_grasp_T is None:
        #     print("hard grasp T is None")
        #     hard_grasp_T = simple_grasp_T
        #     if simple_grasp_T is None:
        #         print("simple grasp T is None")
        #         return

        # == graspgpt
        # hard_grasp_T = pose_msg_to_T(grasp_pose.pose)

        # clear planning scene and wait for update
        ori_planning_scene = self.get_planning_scene()
        ori_planning_scene.world = PlanningSceneWorld()
        self.apply_planning_scene(ori_planning_scene)
        if enable_collision:
            for i in range(10):
                self.collision_pcd_pub.publish(grasp_goal.full_cloud)
                # self.collision_pcd_pub.publish(grasp_goal.env_cloud)
                time.sleep(0.1)
                # cur_planning_scene = self.get_planning_scene()
                # if len(cur_planning_scene.world.collision_objects) != 0:
                #     break
        time.sleep(1)

        # == moveit pre grasp
        print("moving to pre grasp pose")
        # pose_goal = grasp_pose.pose
        grasp_pose = T_to_pose_msg(hard_grasp_T)
        grasp_pose = geometry_msgs.msg.PoseStamped(pose=grasp_pose, header=header)
        pose_goal = self.translate_pose_msg(grasp_pose.pose, [0, 0, -0.03])
        # print(pose_goal)
        assert(type(pose_goal) == geometry_msgs.msg.Pose)
        self.arm_move_group.set_pose_target(pose_goal)
        success = self.arm_move_group.go(wait=True)
        self.arm_move_group.stop()
        self.arm_move_group.clear_pose_targets()
        verify_reach = self.verify_pose(pose_goal, self.get_cur_pose(), 0.01)
        if not verify_reach:
            print("pre grasp pose not reached")
            self.back_to_home_state()
            return

            # # try simple again
            # print("moving to pre grasp pose")
            # # pose_goal = grasp_pose.pose
            # grasp_pose = T_to_pose_msg(simple_grasp_T)
            # grasp_pose = geometry_msgs.msg.PoseStamped(pose=grasp_pose, header=header)
            # pose_goal = self.translate_pose_msg(grasp_pose.pose, [0, 0, -0.03])
            # # print(pose_goal)
            # assert(type(pose_goal) == geometry_msgs.msg.Pose)
            # self.arm_move_group.set_pose_target(pose_goal)
            # success = self.arm_move_group.go(wait=True)
            # self.arm_move_group.stop()
            # self.arm_move_group.clear_pose_targets()

            # verify_reach = self.verify_pose(pose_goal, self.get_cur_pose(), 0.01)
            # if not verify_reach:
            #     print("pre grasp pose not reached")

            #     self.back_to_home_state()
            #     return


        # clear planning scene and wait for update
        ori_planning_scene = self.get_planning_scene()
        ori_planning_scene.world = PlanningSceneWorld()
        self.apply_planning_scene(ori_planning_scene)
        if enable_collision:
            for i in range(10):
            # while True:
                self.collision_pcd_pub.publish(grasp_goal.env_cloud)
                time.sleep(0.1)
                # cur_planning_scene = self.get_planning_scene()
                # if len(cur_planning_scene.world.collision_objects) != 0:
                #     break
        time.sleep(1)

        # == moveit grasp
        print("moving to grasp pose")
        # pose_goal = self.translate_pose_msg(grasp_pose.pose, [0, 0, 0.108])

        # pose_goal = self.translate_pose_msg(grasp_pose.pose, [0, 0, 0.100])
        pose_goal = self.translate_pose_msg(grasp_pose.pose, [0, 0, 0.088])
        waypoints = [pose_goal]
        (plan, fraction) = self.arm_move_group.compute_cartesian_path(
            waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
        )
        print(plan)
        success = self.arm_move_group.execute(plan, wait=True)
        # self.arm_move_group.stop()
        # self.arm_move_group.clear_pose_targets()

        # print(pose_goal)
        # assert(type(pose_goal) == geometry_msgs.msg.Pose)
        # self.arm_move_group.set_pose_target(pose_goal)
        # success = self.arm_move_group.go(wait=True)
        # self.arm_move_group.stop()
        # self.arm_move_group.clear_pose_targets()

        # verify_reach = self.verify_pose(pose_goal, self.get_cur_pose(), 0.03)
        # if not verify_reach:
        #     print("grasp pose not reached")
        #     self.back_to_home_state()
        #     return

        # == moveit pose grasp
        print("grasp")
        # self.example_send_gripper_command(value=0.5)
        self.example_send_gripper_command(value=1.0)

        # 进行 manipulation
        action = "normal"
        # action = "pour"
        # action = "place"
        # action = "cut"

        if action == "cut":
            # 举起来
            pose_goal = self.get_cur_pose()
            pose_goal.position.z += 0.15
            waypoints = [pose_goal]
            (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )
            success = self.arm_move_group.execute(plan, wait=True)

            # 归位
            joint = [-0.01698990249026977, 0.24362972159073162, 3.1230834176976336, -1.4502275673959568, 0.06931690665539374, -1.8496830887802123, 1.573086112700146]
            self.go_to_specify_joint(joint)

            # 目标位置
            pose = [0.4551979578004415, -0.09460965088200658, 0.2913777939533833]
            pose_goal = T_to_pose_msg(sevenDof2T(pose + [0, 0, 0, 1]))
            pose_goal.orientation = self.get_cur_pose().orientation
            waypoints = [pose_goal]
            (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )
            success = self.arm_move_group.execute(plan, wait=True)

            # 往上铲
            c = pose_msg_to_c(self.get_cur_pose())
            c.rotate([-np.deg2rad(-10), 0, 0])
            pose_goal = T_to_pose_msg(pybullet2T(c.pose))
            waypoints = [pose_goal]
            (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )
            success = self.arm_move_group.execute(plan, wait=True)
            time.sleep(1)

            # 往斜下方铲
            pose_goal = self.get_cur_pose()
            pose_goal.position.x += 0.08
            pose_goal.position.z -= 0.08
            waypoints = [pose_goal]
            (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )
            success = self.arm_move_group.execute(plan, wait=True)
            time.sleep(2)

            # 往上铲
            c = pose_msg_to_c(self.get_cur_pose())
            c.rotate([-np.deg2rad(10), 0, 0])
            c.translate([0.01, 0, 0.0])
            pose_goal = T_to_pose_msg(pybullet2T(c.pose))
            waypoints = [pose_goal]
            (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )
            success = self.arm_move_group.execute(plan, wait=True)
            time.sleep(1)

            # 抬起来
            pose_goal = self.get_cur_pose()
            pose_goal.position.z += 0.15
            waypoints = [pose_goal]
            (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )
            success = self.arm_move_group.execute(plan, wait=True)
            time.sleep(2)

            # # 移动到目标位置
            # pose = [0.2947997610512299, 0.11903731873544133, 0.2743777939533833]
            # pose_goal = T_to_pose_msg(sevenDof2T(pose + [0, 0, 0, 1]))
            # pose_goal.orientation = self.get_cur_pose().orientation
            # waypoints = [pose_goal]
            # (plan, fraction) = self.arm_move_group.compute_cartesian_path(
            #     waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            # )
            # success = self.arm_move_group.execute(plan, wait=True)

            # # 倒
            # cur_pose = self.get_cur_pose()
            # cur_pose = [[cur_pose.position.x, cur_pose.position.y, cur_pose.position.z], [cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w]]
            # z_angle = compute_z_angle(pybullet2T(cur_pose))
            # if z_angle > 45:  # 竖着抓的
            #     c = geometry.Coordinate(cur_pose[0], cur_pose[1])
            #     c.rotate([0, np.deg2rad(-90), 0], wrt="local")
            #     pose_goal = T_to_pose_msg(pybullet2T(c.pose))

            #     waypoints = [pose_goal]
            #     (plan, fraction) = self.arm_move_group.compute_cartesian_path(
            #         waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            #     )
            #     success = self.arm_move_group.execute(plan, wait=True)
            #     self.arm_move_group.stop()
            #     self.arm_move_group.clear_pose_targets()
            #     time.sleep(5)

            #     # self.arm_move_group.set_pose_target(pose_goal)
            #     # success = self.arm_move_group.go(wait=True)
            #     # self.arm_move_group.stop()
            #     # self.arm_move_group.clear_pose_targets()

            #     # verify_reach = self.verify_pose(pose_goal, self.get_cur_pose(), 0.01)
            #     # if not verify_reach:
            #     #     print("pour pose not reached")
            #     #     c = geometry.Coordinate(cur_pose[0], cur_pose[1])
            #     #     c.rotate([0, np.deg2rad(-90), 0], wrt="local")
            #     #     pose_goal = T_to_pose_msg(pybullet2T(c.pose))

            #     #     waypoints = [pose_goal]
            #     #     (plan, fraction) = self.arm_move_group.compute_cartesian_path(
            #     #         waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            #     #     )
            #     #     success = self.arm_move_group.execute(plan, wait=True)
            #     #     self.arm_move_group.stop()
            #     #     self.arm_move_group.clear_pose_targets()

            #     #     # self.arm_move_group.set_pose_target(pose_goal)
            #     #     # success = self.arm_move_group.go(wait=True)
            #     #     # self.arm_move_group.stop()
            #     #     # self.arm_move_group.clear_pose_targets()

            #     #     verify_reach = self.verify_pose(pose_goal, self.get_cur_pose(), 0.01)
            #     #     if not verify_reach:
            #     #         print("pour pose not reached")
            #     #         self.back_to_home_state()
            # else:  # 横着抓的
            #     c = geometry.Coordinate(cur_pose[0], cur_pose[1])
            #     c.rotate([0, 0, np.deg2rad(120)], wrt="local")
            #     pose_goal = T_to_pose_msg(pybullet2T(c.pose))
            #     self.arm_move_group.set_pose_target(pose_goal)
            #     success = self.arm_move_group.go(wait=True)
            #     self.arm_move_group.stop()
            #     self.arm_move_group.clear_pose_targets()
            #     verify_reach = self.verify_pose(pose_goal, self.get_cur_pose(), 0.01)
            #     if not verify_reach:
            #         print("pour pose not reached")
            #         c = geometry.Coordinate(cur_pose[0], cur_pose[1])
            #         c.rotate([0, 0, np.deg2rad(-110)], wrt="local")
            #         pose_goal = T_to_pose_msg(pybullet2T(c.pose))
            #         self.arm_move_group.set_pose_target(pose_goal)
            #         success = self.arm_move_group.go(wait=True)
            #         self.arm_move_group.stop()
            #         self.arm_move_group.clear_pose_targets()
            #         verify_reach = self.verify_pose(pose_goal, self.get_cur_pose(), 0.01)
            #         if not verify_reach:
            #             print("pour pose not reached")
            #             self.back_to_home_state()
            
            # # 放下勺子
            # self.back_to_grasp_state()

            # pose = [0.2853296484923709, 0.09734829990176541, 0.3]
            # pose_goal = T_to_pose_msg(sevenDof2T(pose + [0, 0, 0, 1]))
            # pose_goal.orientation.x = 0.018073192860724323
            # pose_goal.orientation.y = 0.9992255928086577
            # pose_goal.orientation.z = 0.003314989414865834
            # pose_goal.orientation.w = 0.03479346520807617
            # waypoints = [pose_goal]
            # (plan, fraction) = self.arm_move_group.compute_cartesian_path(
            #     waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            # )
            # success = self.arm_move_group.execute(plan, wait=True)

            # pose_goal = self.get_cur_pose()
            # pose_goal.position.z -= 0.16
            # waypoints = [pose_goal]
            # (plan, fraction) = self.arm_move_group.compute_cartesian_path(
            #     waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            # )
            # success = self.arm_move_group.execute(plan, wait=True)

            self.example_send_gripper_command(value=0.2)
            pass

        elif action == "pour":
            # 举起来
            pose_goal = self.get_cur_pose()
            pose_goal.position.z += 0.15
            waypoints = [pose_goal]
            (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )
            success = self.arm_move_group.execute(plan, wait=True)

            # 归位
            self.back_to_grasp_state()

            # 移到目标位置
            pose = [0.4881979578004415, -0.12460965088200658, 0.3043777939533833]
            pose_goal = T_to_pose_msg(sevenDof2T(pose + [0, 0, 0, 1]))
            pose_goal.orientation = self.get_cur_pose().orientation
            waypoints = [pose_goal]
            (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )
            success = self.arm_move_group.execute(plan, wait=True)

            # 倒
            cur_pose = self.get_cur_pose()
            cur_pose = [[cur_pose.position.x, cur_pose.position.y, cur_pose.position.z], [cur_pose.orientation.x, cur_pose.orientation.y, cur_pose.orientation.z, cur_pose.orientation.w]]
            z_angle = compute_z_angle(hard_grasp_T)
            if z_angle > 45:  # 竖着抓的
                c = geometry.Coordinate(cur_pose[0], cur_pose[1])
                c.rotate([0, np.deg2rad(-110), 0], wrt="local")
                pose_goal = T_to_pose_msg(pybullet2T(c.pose))

                waypoints = [pose_goal]
                (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                    waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
                )
                success = self.arm_move_group.execute(plan, wait=True)
                self.arm_move_group.stop()
                self.arm_move_group.clear_pose_targets()
                time.sleep(5)

                # self.arm_move_group.set_pose_target(pose_goal)
                # success = self.arm_move_group.go(wait=True)
                # self.arm_move_group.stop()
                # self.arm_move_group.clear_pose_targets()

                # verify_reach = self.verify_pose(pose_goal, self.get_cur_pose(), 0.01)
                # if not verify_reach:
                #     print("pour pose not reached")
                #     c = geometry.Coordinate(cur_pose[0], cur_pose[1])
                #     c.rotate([0, np.deg2rad(-90), 0], wrt="local")
                #     pose_goal = T_to_pose_msg(pybullet2T(c.pose))

                #     waypoints = [pose_goal]
                #     (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                #         waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
                #     )
                #     success = self.arm_move_group.execute(plan, wait=True)
                #     self.arm_move_group.stop()
                #     self.arm_move_group.clear_pose_targets()

                #     # self.arm_move_group.set_pose_target(pose_goal)
                #     # success = self.arm_move_group.go(wait=True)
                #     # self.arm_move_group.stop()
                #     # self.arm_move_group.clear_pose_targets()

                #     verify_reach = self.verify_pose(pose_goal, self.get_cur_pose(), 0.01)
                #     if not verify_reach:
                #         print("pour pose not reached")
                #         self.back_to_home_state()
            else:  # 横着抓的
                c = geometry.Coordinate(cur_pose[0], cur_pose[1])
                c.rotate([0, 0, np.deg2rad(110)], wrt="local")
                pose_goal = T_to_pose_msg(pybullet2T(c.pose))
                self.arm_move_group.set_pose_target(pose_goal)
                success = self.arm_move_group.go(wait=True)
                self.arm_move_group.stop()
                self.arm_move_group.clear_pose_targets()
                verify_reach = self.verify_pose(pose_goal, self.get_cur_pose(), 0.01)
                if not verify_reach:
                    print("pour pose not reached")
                    c = geometry.Coordinate(cur_pose[0], cur_pose[1])
                    c.rotate([0, 0, np.deg2rad(-110)], wrt="local")
                    pose_goal = T_to_pose_msg(pybullet2T(c.pose))
                    self.arm_move_group.set_pose_target(pose_goal)
                    success = self.arm_move_group.go(wait=True)
                    self.arm_move_group.stop()
                    self.arm_move_group.clear_pose_targets()
                    verify_reach = self.verify_pose(pose_goal, self.get_cur_pose(), 0.01)
                    if not verify_reach:
                        print("pour pose not reached")
                        self.back_to_home_state()

            time.sleep(2)

            # # 放下来
            # self.back_to_grasp_state()

            # pose = [0.3053296484923709, 0.02734829990176541, 0.3]
            # pose_goal = T_to_pose_msg(sevenDof2T(pose + [0, 0, 0, 1]))
            # pose_goal.orientation.x = 0.018073192860724323
            # pose_goal.orientation.y = 0.9992255928086577
            # pose_goal.orientation.z = 0.003314989414865834
            # pose_goal.orientation.w = 0.03479346520807617
            # waypoints = [pose_goal]
            # (plan, fraction) = self.arm_move_group.compute_cartesian_path(
            #     waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            # )
            # success = self.arm_move_group.execute(plan, wait=True)

            # pose_goal = self.get_cur_pose()
            # pose_goal.position.z -= 0.12
            # waypoints = [pose_goal]
            # (plan, fraction) = self.arm_move_group.compute_cartesian_path(
            #     waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            # )
            # success = self.arm_move_group.execute(plan, wait=True)

            # joint_goal = [-0.1692540072866775, 0.24677103660977778, 3.054517139640235, -2.0491738281353227, 0.05966495755824166, -0.7797085860446034, 1.27590383350015]
            # self.go_to_specify_joint(joint_goal)

            self.example_send_gripper_command(value=0.2)
            pass

        elif action == "place":
            # 举起来
            pose_goal = self.get_cur_pose()
            pose_goal.position.z += 0.15
            waypoints = [pose_goal]
            (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )
            success = self.arm_move_group.execute(plan, wait=True)

            # # 移到目标位置
            joint = [0.4352411493209497, -0.09204736938862634, 2.3975259513620477, -1.6666046122749334, -0.2740957351842628, -0.7511797391832378, 2.323698864887307]
            self.go_to_specify_joint(joint)

            # pose = [0.39228432546185626, 0.30755358627124346, 0.3034245668226253]
            # pose_goal = T_to_pose_msg(sevenDof2T(pose + [0, 0, 0, 1]))
            # pose_goal.orientation = self.get_cur_pose().orientation
            # waypoints = [pose_goal]
            # (plan, fraction) = self.arm_move_group.compute_cartesian_path(
            #     waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            # )
            # success = self.arm_move_group.execute(plan, wait=True)

            # == 放置 
            time.sleep(3.0)
            print("gripper release")
            self.example_send_gripper_command(value=0.2)
            pass

        elif action == "cut":
            pass
        else:  # 正常的动作
            # == moveit pose grasp
            print("moving to grasp pose")
            pose_goal = self.get_cur_pose()
            pose_goal.position.z += 0.2
            # pose_goal = self.translate_pose_msg(grasp_pose.pose, [0, 0, 0.108])
            waypoints = [pose_goal]
            (plan, fraction) = self.arm_move_group.compute_cartesian_path(
                waypoints, 0.01, 0.0  # waypoints to follow  # eef_step
            )
            success = self.arm_move_group.execute(plan, wait=True)
            # self.arm_move_group.stop()
            # self.arm_move_group.clear_pose_targets()

            # pose_goal = self.translate_pose_msg(pose_goal, [0, 0, 0.22], wrt="world")
            # print(pose_goal)
            # assert(type(pose_goal) == geometry_msgs.msg.Pose)
            # self.arm_move_group.set_pose_target(pose_goal)
            # success = self.arm_move_group.go(wait=True)
            # self.arm_move_group.stop()
            # self.arm_move_group.clear_pose_targets()

            verify_reach = self.verify_pose(pose_goal, self.get_cur_pose(), 0.01)
            if not verify_reach:
                print("grasp pose not reached")
                self.back_to_home_state()
                return

            # == 
            time.sleep(1.0)
            print("gripper release")
            self.example_send_gripper_command(value=0.2)

        # == 
        time.sleep(0.5)
        self.back_to_home_state()
        pass

    @staticmethod
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
