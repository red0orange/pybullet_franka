#!/usr/bin/python3
import os
import re
import sys

import numpy as np
from scipy import interpolate

import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg

from moveit_commander.conversions import pose_to_list
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from kortex_driver.srv import *
from kortex_driver.msg import *

from demo_utils import adjust_angles_for_continuity, resample_trajectory, compute_velocity_acceleration, limit_velocity_acceleration, process_traj, wait_for_desired_input, wait_for_valid_input, joints_traj_to_robot_traj


class DemonstrationCollector(object):
    def __init__(self):
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manipulate_trajs")

        # @note 初始化
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("demo_moveit_interface", anonymous=True)

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
        
        # @note 主循环
        while True:
            user_input = wait_for_desired_input(
                "============\n"
                "Please input command: \n"
                "1: back to home pose\n"
                "2: back to grasp pose\n"
                "3: begin to collect traj\n"
                "4: test traj\n"
                "5: quit\n"
                "============\n",
                ["1", "2", "3", "4", "5"]
            )
            if user_input == "1":
                self.back_to_home_state()
            elif user_input == "2":
                self.back_to_grasp_state()
            elif user_input == "3":
                self.traj_poses = []
                self.traj_v = []
                input("Press any key to start")
                print("!Start! to collect trajectory")
                self.sub = rospy.Subscriber("/my_gen3/joint_states", sensor_msgs.msg.JointState, self.collect_traj_pose)
                # self.timer = rospy.Timer(rospy.Duration(nsecs=(1 / fps) * 1e9), self.collect_traj_pose) 
                # self.timer.shutdown()
                input("Press any key to stop")
                self.sub.unregister()
                print("!Stop! to collect trajectory")
                print("Trajectory length: {}".format(len(self.traj_poses)))

                # @note 保存轨迹
                self.traj_poses = np.array(self.traj_poses)
                traj_data_dict = {
                    "joints_traj": self.traj_poses,
                    "v_traj": self.traj_v,
                }
                np.save("tmp_traj.npy", traj_data_dict)    # @note 临时备份轨迹
                file_name = wait_for_valid_input(prompt="Please input file name (no posfix): ")
                file_path = os.path.join(self.save_dir, file_name + ".npy")
                np.save(file_path, traj_data_dict)
                os.remove("tmp_traj.npy")   # 删除临时备份轨迹
            elif user_input == "4":
                seq_name = wait_for_valid_input(prompt="Please input file name (no posfix): ")
                self.test_traj(seq_name)
            elif user_input == "5":
                break
            else:
                print("Invalid input: {}".format(user_input))
                continue


        pass

    def collect_traj_pose(self, msg):
        cur_joints = msg.position
        self.traj_poses.append(cur_joints[:7])
        self.traj_v.append(msg.velocity[:7])
        pass

    def test_traj(self, manipulation_seq_name):
        # @note debug pose 的 header frame_id 用来放 manipulation 的指定序列
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manipulate_trajs")
        mani_seq_path = os.path.join(save_dir, manipulation_seq_name + ".npy")
        print("===================================")
        if os.path.exists(mani_seq_path):
            print("manipulation sequence found: {}".format(mani_seq_path))
        else:
            print("manipulation sequence not found, will not do manipulation")
        print("===================================")

        mani_seq_data_dict = np.load(mani_seq_path, allow_pickle=True).item()

        fps = 20
        finish_flag = False
        cnt = 0
        while not finish_flag and cnt < 3:
            cnt += 1
            mani_seq = mani_seq_data_dict["joints_traj"]
            mani_v = mani_seq_data_dict["v_traj"]

            downsample_ratio = 4
            mani_seq = mani_seq[::downsample_ratio]
            mani_v = mani_v[::downsample_ratio]

            print("original manipulation sequence length: {}".format(len(mani_seq)))
            mani_seq = process_traj(mani_seq, threshold=0.01)   # @note
            # mani_seq = resample_trajectory(mani_seq, num_points=len(mani_seq))   # @note
            print("processed manipulation sequence length: {}".format(len(mani_seq)))

            # 去到 manipulation 的起始点
            mani_start = mani_seq[0]
            self.go_to_specify_joint(mani_start)
            robot_traj = joints_traj_to_robot_traj(mani_seq, velocity=mani_v, duration=(1.0 / fps))
            # robot_traj = joints_traj_to_robot_traj(mani_seq, duration=(1.0 / fps))
            finish_flag = self.arm_move_group.execute(robot_traj, wait=True)

            fps /= 2  # @note 降低 fps
        pass

    def go_to_specify_joint(self, joint):
        self.arm_move_group.go(joint, wait=True)
        pass

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

    def get_cur_joints(self):
        robot_cur_state = self.arm_move_group.get_current_joint_values()
        return robot_cur_state

    def get_cur_pose(self):
        return self.arm_move_group.get_current_pose().pose

    def back_to_home_state(self, release_gripper=True):
        # gripper
        if release_gripper:
            self.example_send_gripper_command(value=0.2)

        # home pose
        home_joint_goal = [-0.039876576266192565, -1.2166917756136009, 3.0783202071472546, -2.010595010268326, 0.023798230124570448, -1.5123021239801782, 1.4359715329227085]
        self.arm_move_group.go(home_joint_goal, wait=True)
        pass

    def back_to_grasp_state(self):
        grasp_state = [-0.04285931668708187, -0.2992556832666775, 3.1413361911768183, -2.0276917055183077, 0.05497603801785843, -1.1224648751888529, 1.4403721403079552]
        self.arm_move_group.go(grasp_state, wait=True)
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
    

if __name__ == "__main__":
    DemonstrationCollector()
    pass