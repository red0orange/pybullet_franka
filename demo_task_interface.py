import time
import IPython

import numpy as np
import pybullet as p
import pybullet_planning as pp

import rospy
import actionlib
from geometry_msgs.msg import PoseStamped

from my_robot.msg import GraspAction, GraspGoal

from base_task_interface import BaseTaskInterface
from utils.reorient_modules import reorientbot
import utils.geometry as geometry
from utils.tf import *


def pose_msg_to_T(pose_msg):
    T = np.identity(4)
    T[:3, :3] = tf.transformations.quaternion_matrix(
        [pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w])[:3, :3]
    T[:3, 3] = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    return T


class DemoTaskInterface(BaseTaskInterface):
    def __init__(self, real):
        super().__init__(real)

        # output_topic_name = "/grasp_poses"
        # self.grasp_pose_subscriber = rospy.Subscriber(output_topic_name, PoseStamped, self.grasp_callback)
        # self.grasp_poses = []

        self.server = actionlib.SimpleActionServer(
            "/grasp_interface", GraspAction, self.grasp_callback, False
        )
        self.server.start()
        pass

    def grasp_callback(self, grasp_goal):
        print(grasp_goal)
        grasp_pose = grasp_goal.grasp_pose

        # debug
        grasp_poses = grasp_goal.debug_grasp_poses
        print(grasp_poses)
        debug_T = [pose_msg_to_T(pose.pose) for pose in grasp_poses]
        for i, T in enumerate(debug_T):
            if i % 3 == 0:
                pp.draw_pose(T2pybullet(T))

        # debug
        grasp_pose = sorted(grasp_poses, key=lambda x: x.pose.position.y, reverse=True)[0]

        goal_T = pose_msg_to_T(grasp_pose.pose)
        goal_pose = T2pybullet(goal_T)

        pp.draw_pose(goal_pose)

        # pre grasp
        c = reorientbot.geometry.Coordinate(goal_pose[0], goal_pose[1])
        c.translate([0, 0, -0.04], wrt="local")
        goal_pose = c.pose
        target_j = self._env.pi.solve_ik(
            goal_pose,
            move_target=self._env.pi.robot_model.panda_hand,
            n_init=100,
            thre=0.05,
            rthre=np.deg2rad(5),
            # obstacles=self.object_ids,
            validate=True,
        )
        if target_j is None:
            print("target_j is None")
            return
        # Planning
        target_js = self._env.pi.planj(
            target_j, 
            # obstacles=self.object_ids
        )
        if target_js is None:
            print("target_js is None")
            return
        # Control
        self.movejs(target_js, time_scale=5, retry=True)

        # grasp
        pp.draw_pose(goal_pose)
        c = reorientbot.geometry.Coordinate(goal_pose[0], goal_pose[1])
        c.translate([0, 0, 0.04], wrt="local")
        goal_pose = c.pose

        target_j = self._env.pi.solve_ik(
            goal_pose,
            move_target=self._env.pi.robot_model.panda_hand,
            n_init=100,
            thre=0.05,
            rthre=np.deg2rad(5),
            # obstacles=self.object_ids,
            validate=True,
        )
        if target_j is None:
            print("target_j is None")
            return
        # Planning
        target_js = self._env.pi.planj(
            target_j, 
            # obstacles=self.object_ids
        )
        if target_js is None:
            print("target_js is None")
            return
        # Control
        self.movejs(target_js, time_scale=5, retry=True)
        

        self.start_grasp()
        time.sleep(2)
        self.stop_grasp()

        # 回归 Home pose
        self.reset_pose()
        pass


    def test_real_pick(self):
        # 回归 Home pose
        self.reset_pose()
        self.stop_grasp()

        # 设置抓取 Pose
        eye = np.array([0.4, 0, 0.5])
        target = eye + np.array([0, 0, -0.1])
        c = geometry.Coordinate.from_matrix(
            geometry.look_at(eye, target, None)
        )
        goal_pose = c.pose
        # 求解 IK
        target_j = self._env.pi.solve_ik(
            goal_pose,
            move_target=self._env.pi.robot_model.panda_hand,
            n_init=100,
            thre=0.05,
            rthre=np.deg2rad(5),
            # obstacles=self.object_ids,
            validate=True,
        )
        # Planning
        target_js = self._env.pi.planj(
            target_j, 
            # obstacles=self.object_ids
        )
        if target_js is None:
            print("target_js is None")
            return
        # Control
        self.movejs(target_js, time_scale=5, retry=True)
        self.start_grasp()
        time.sleep(2)
        self.stop_grasp()

        # 回归 Home pose
        self.reset_pose()
        pass


def main():
    rospy.init_node("grasp_task_interface")
    
    # real = False
    real = True
    task_interface = DemoTaskInterface(real)

    # time.sleep(2)
    # task_interface.test_real_pick()

    rospy.spin()

    self = task_interface
    IPython.embed(header="base_task_interface")
    pass

    
if __name__ == '__main__':
    main()
    pass