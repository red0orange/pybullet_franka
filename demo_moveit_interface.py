#!/usr/bin/python3
from multiprocessing.connection import wait
import sys
import copy
import re
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

from math import pi, tau, dist, fabs, cos


from std_msgs.msg import String
from moveit_commander.conversions import pose_to_list


class DemoMoveitInterface(object):
    def __init__(self) -> None:
        super(DemoMoveitInterface, self).__init__()

        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("demo_moveit_interface", anonymous=True)

        # init
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander("panda_arm")

        # print franka state
        planning_frame = self.move_group.get_planning_frame()
        print("============ Planning frame: %s" % planning_frame)
        eef_link = self.move_group.get_end_effector_link()
        print("============ End effector link: %s" % eef_link)
        group_names = self.robot.get_group_names()
        print("============ Available Planning Groups:", group_names)
        robot_cur_state = self.robot.get_current_state()
        joint_state = "\n".join(re.findall(
            "position" + '.*', str(robot_cur_state)))
        print("============ Printing robot state: {}".format(joint_state))
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
        return self.move_group.get_current_pose().pose

    def move_to_pose_goal(self, pose_goal):
        assert(type(pose_goal) == geometry_msgs.msg.Pose)

        self.move_group.set_pose_target(pose_goal)
        success = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        return DemoMoveitInterface.verify_pose(pose_goal, self.get_cur_pose(), 0.01)


if __name__ == "__main__":
    demo_interface = DemoMoveitInterface()

    # control down 0.05
    cur_pose = demo_interface.get_cur_pose()
    goal_pose = copy.copy(cur_pose)
    goal_pose.position.z += 0.10
    result = demo_interface.move_to_pose_goal(goal_pose)
    print("result: {}".format("True" if result else "False"))

    rospy.spin()
    pass
