import actionlib
import control_msgs.msg
import franka_gripper.msg

from skrobot.interfaces.ros.base import ROSRobotInterfaceBase


WIDTH_MAX = 0.08


class PandaROSRobotInterface(ROSRobotInterfaceBase):

    def __init__(self, *args, **kwargs):
        super(PandaROSRobotInterface, self).__init__(*args, **kwargs)

        self.gripper_grasp = actionlib.SimpleActionClient(
            '/franka_gripper/grasp', franka_gripper.msg.GraspAction)
        self.gripper_grasp.wait_for_server()

        self.gripper_move = actionlib.SimpleActionClient(
            'franka_gripper/move',
            franka_gripper.msg.MoveAction)
        self.gripper_move.wait_for_server()

        self.gripper_stop = actionlib.SimpleActionClient(
            'franka_gripper/stop',
            franka_gripper.msg.StopAction)
        self.gripper_stop.wait_for_server()

    def grasp(self, width=0, **kwargs):
        # self.move_gripper(width=width, **kwargs)
        self.gripper_grasp(width=width, **kwargs)

    def ungrasp(self, **kwargs):
        self.move_gripper(width=WIDTH_MAX, **kwargs)

    def gripper_grasp(self, width, speed=WIDTH_MAX, wait=True):
        goal_epsilon = franka_gripper.msg.GraspEpsilon(inner=0.01, outer=0.01)
        goal = franka_gripper.msg.GraspGoal(
            width=width, epsilon=goal_epsilon, speed=speed, force=5.0)
        if wait:
            self.gripper_grasp.send_goal_and_wait(goal)
        else:
            self.gripper_grasp.send_goal(goal)
        pass

    def move_gripper(self, width, speed=WIDTH_MAX, wait=True):
        goal = franka_gripper.msg.MoveGoal(width=width, speed=speed)
        if wait:
            self.gripper_move.send_goal_and_wait(goal)
        else:
            self.gripper_move.send_goal(goal)

    def stop_gripper(self, wait=True):
        goal = franka_gripper.msg.StopGoal()
        if wait:
            self.gripper_move.send_goal_and_wait(goal)
        else:
            self.gripper_move.send_goal(goal)

    @property
    def rarm_controller(self):
        return dict(
            controller_type='rarm_controller',
            controller_action='position_joint_trajectory_controller/follow_joint_trajectory',  # NOQA
            controller_state='position_joint_trajectory_controller/state',
            action_type=control_msgs.msg.FollowJointTrajectoryAction,
            joint_names=[j.name for j in self.robot.rarm.joint_list],
        )

    def default_controller(self):
        return [self.rarm_controller]
