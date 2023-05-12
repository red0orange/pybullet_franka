#!/home/red0orange/anaconda3/envs/franka/bin/python

import argparse
import time
from functools import partial

import imgviz
import IPython
import numpy as np
import pybullet as p
import pybullet_planning as pp

import actionlib
from actionlib_msgs.msg import GoalStatus
import cv_bridge
from franka_msgs.msg import ErrorRecoveryAction
from franka_msgs.msg import ErrorRecoveryGoal
from franka_msgs.msg import FrankaState
import franka_gripper.msg
import rospy
from std_msgs.msg import Int16
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool
import tf

from modules import reorientbot
from modules import _env
from modules._message_subscriber import MessageSubscriber
from modules._panda import Panda, MyPanda
from modules._panda_ros_robot_interface import PandaROSRobotInterface


class BaseTaskInterface:
    def __init__(self, real, gui=True):
        self.real = real
        self.base_link = "panda_link0"

        self._env = _env.Env(
            robot_model="franka_panda/panda_finger",
            debug=False,
            gui=gui
        )
        self._env.reset()

        # @note Manully Control Joint Interface
        self.incremental_control_joints_sub = rospy.Subscriber(
            "~/my_incremental_control_joint", JointState, partial(self.control_joints_cb, method="incremental"), queue_size=3)
        self.absolute_control_joints_sub = rospy.Subscriber(
            "~/my_absolute_control_joint", JointState, partial(self.control_joints_cb, method="absolute"), queue_size=1)
        self.absolute_control_joints_sub = rospy.Subscriber(
            "~/my_control_gripper", Int16, self.control_gripper_cb, queue_size=1)

        if self.real:
            self._tf_listener = tf.listener.TransformListener(
                cache_time=rospy.Duration(60)
            )

            # self._ri = PandaROSRobotInterface(robot=Panda())
            self._ri = PandaROSRobotInterface(robot=MyPanda())

            self._real2robot()

            self._sub_points = MessageSubscriber(
                [
                    ("/camera/color/camera_info", CameraInfo),
                    ("/camera/color/image_rect_color", Image),
                    ("/camera/aligned_depth_to_color/image_raw", Image),
                ],
                callback=self._sub_points_callback,
            )
            self._sub_points_density = 1 / 9
            self._sub_points_update_rate = 1
            self._sub_points_stamp = None
            self._sub_points_pybullet_id = None

            self._workspace_initialized = False

        else:
            self.robot_state_pub = rospy.Publisher(
                "/joint_states", JointState, queue_size=1)
            pass

    # ******************************************************************
    # Base task interface functions which can be called public
    # All these function are suitable for real and sim
    # ******************************************************************
    def spinOnce(self):
        if self.real:
            self._real2robot()
            pass
        else:
            self._update_robot_state()
            self._env.update_obs()
            pp.step_simulation()
            pass

    def get_joint_state(self):
        if self.real:
            self._real2robot()
        return self._env.pi.getj()

    def movejs(self, js, *argc, **argv):
        if self.real:
            self._real_movejs(js, *argc, **argv)
        else:
            self._sim_movejs(js, *argc, **argv)
        pass

    def start_grasp(self):
        if self.real:
            self.ri.grasp(wait=True)
        else:
            self._env.pi.grasp()
        pass

    def stop_grasp(self):
        if self.real:
            self.ri.ungrasp(wait=True)
        else:
            self._env.pi.ungrasp()
        pass

    def reset_pose(self, *args, **kwargs):
        self.movejs([self._env.pi.homej], *args, **kwargs)
        pass

    # ******************************************************************
    # Directly control in sim and don't need update states,
    # But publishing states for rviz is need
    # ******************************************************************
    def _sim_movejs(self, js, sleep_time=1/240, *argc, **argv):
        for j in js:
            for _ in self._env.pi.movej(j):
                pp.step_simulation()
                time.sleep(sleep_time)
        pass

    def _update_robot_state(self):
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = [0] * len(self._env.pi.joints)
        joint_state.position = [0] * len(self._env.pi.joints)
        for joint_i in self._env.pi.joints:
            joint_name = pp.get_joint_name(
                self._env.pi.robot, joint_i).decode()
            joint_position = p.getJointState(
                self._env.pi.robot, joint_i)[0]

            joint_state.name[joint_i] = joint_name
            joint_state.position[joint_i] = joint_position
        self.robot_state_pub.publish(joint_state)
        pass

    # ******************************************************************
    # Control in real and Update states to pybullet
    # ******************************************************************
    def _real_movejs(self, js, time_scale=None, wait=True, retry=False, wait_callback=None, *argc, **argv):
        if not self._recover_from_error():
            return
        if time_scale is None:
            time_scale = 3
        js = np.asarray(js)

        self._real2robot()
        j_init = self._env.pi.getj()

        self.ri.angle_vector_sequence(
            js, time_scale=time_scale,
        )
        if wait:
            success = self._wait_interpolation(callback=wait_callback)
            if success or not retry:
                return

            self._real2robot()  # update

            j_curr = self._env.pi.getj()

            js = np.r_[[j_init], js]

            for i in range(len(js) - 1):
                dj1 = js[i + 1] - j_curr
                dj2 = js[i + 1] - js[i]
                dj1[abs(dj1) < 0.01] = 0
                dj2[abs(dj2) < 0.01] = 0
                if (np.sign(dj1) == np.sign(dj2)).all():
                    break
            else:
                return
            self.movejs(
                js[i + 1:], time_scale=time_scale, wait=wait, retry=False
            )
        pass

    def _wait_interpolation(self, callback=None):
        self._sub_points.subscribe()
        controller_actions = self.ri.controller_table[self.ri.controller_type]
        while True:
            states = [action.get_state() for action in controller_actions]
            if all(s >= GoalStatus.SUCCEEDED for s in states):
                break
            self._real2robot()
            if callback is not None:
                callback()
            rospy.sleep(0.01)
        self._sub_points.unsubscribe()
        if not all(s == GoalStatus.SUCCEEDED for s in states):
            rospy.logwarn("Some joint control requests have failed")
            return False
        return True

    # @note real2robot, update real robot states
    def _real2robot(self):
        # Update states of real robot to sim robot
        self.ri.update_robot_state()  # Get real robot states
        # Update arm joints
        self._env.pi.setj(self.ri.potentio_vector())
        # Update finger gripper
        self._env.pi.set_finger(self.ri.finger_positions())

        for attachment in self._env.pi.attachments:
            attachment.assign()
        pass

    def _recover_from_error(self):
        state_msg = rospy.wait_for_message(
            "/franka_state_controller/franka_states", FrankaState
        )
        if state_msg.robot_mode == FrankaState.ROBOT_MODE_MOVE:
            return True

        client = actionlib.SimpleActionClient(
            "/franka_control/error_recovery", ErrorRecoveryAction
        )
        client.wait_for_server()

        if client.get_state() == GoalStatus.SUCCEEDED:
            return True

        goal = ErrorRecoveryGoal()
        state = client.send_goal_and_wait(goal)
        succeeded = state == GoalStatus.SUCCEEDED

        if succeeded:
            rospy.loginfo("Recovered from error")
        else:
            rospy.logerr("Failed to recover from error")
        return succeeded

    # ******************************************************************
    # Planning in pybullet
    # ******************************************************************
    def _solve_ik_for_look_at(self, eye, target, up=None, rotation_axis=True):
        c = reorientbot.geometry.Coordinate.from_matrix(
            reorientbot.geometry.look_at(eye, target, up)
        )
        if rotation_axis is True:
            for _ in range(4):
                c.rotate([0, 0, np.deg2rad(90)])
                if abs(c.euler[2] - np.deg2rad(-90)) < np.pi / 4:
                    break
        j = self._env.pi.solve_ik(
            c.pose,
            move_target=self._env.pi.robot_model.camera_link,
            n_init=20,
            thre=0.05,
            # rthre=np.deg2rad(15),
            # rotation_axis=rotation_axis,
            validate=True,
        )
        if j is None:
            rospy.logerr("j is not found")
            return
        return j

    def _look_at(self, eye, target, rotation_axis=True, *args, **kwargs):
        j = self._solve_ik_for_look_at(eye, target, rotation_axis)
        self.movejs([j], *args, **kwargs)

    def _init_workspace(self):
        if self._workspace_initialized:
            return

        # light
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS, True, lightPosition=(100, -100, 0.5)
        )

        # table
        pp.set_texture(self._env.plane)

        # left wall
        obj = pp.create_box(w=3, l=0.01, h=1.05, color=(0.6, 0.6, 0.6, 1))
        pp.set_pose(
            obj,
            (
                (-0.0010000000000000002, 0.6925000000000028, 0.55),
                (0.0, 0.0, 0.0194987642109932, 0.9998098810245096),
            ),
        )
        self._env.bg_objects.append(obj)

        # back wall
        obj = pp.create_box(w=0.01, l=3, h=1.05, color=(0.7, 0.7, 0.7, 1))
        pp.set_pose(obj, ([-0.4, 0, 1.05 / 2], [0, 0, 0, 1]))
        self._env.bg_objects.append(obj)

        # ceiling
        obj = pp.create_box(w=3, l=3, h=0.5, color=(1, 1, 1, 1))
        pp.set_pose(obj, ([0, 0, 0.25 + 1.05], [0, 0, 0, 1]))
        self._env.bg_objects.append(obj)

        self._workspace_initialized = True

        # reorientbot.pybullet.annotate_pose(obj)

    # ******************************************************************
    # Others
    # ******************************************************************
    @property
    def ri(self):
        return self._ri

    def control_gripper_cb(self, msg: Int16):
        control_flag = msg.data
        if control_flag == 0:
            self.stop_grasp()
        elif control_flag == 1:
            self.start_grasp()
        pass

    def control_joints_cb(self, msg: JointState, method):
        print("Control")
        # cal goal joint state
        if method == "incremental":
            cur_joint_state = self.get_joint_state()
            inc_joint_state = msg.position
            assert len(cur_joint_state) == len(inc_joint_state)
            goal_joint_state = (np.array(cur_joint_state) +
                                np.array(inc_joint_state)).tolist()
        elif method == "absolute":
            goal_joint_state = msg.position
        else:
            raise BaseException("Error Method")

        # control
        if self.real:
            self.movejs([goal_joint_state], time_scale=20)
        else:
            self.movejs([goal_joint_state])
        pass

    # no idea
    def _sub_points_callback(self, info_msg, rgb_msg, depth_msg):
        if self._sub_points_stamp is not None and (
            info_msg.header.stamp - self._sub_points_stamp
        ) < rospy.Duration(1 / self._sub_points_update_rate):
            return

        K = np.array(info_msg.K).reshape(3, 3)
        bridge = cv_bridge.CvBridge()
        rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")
        depth = bridge.imgmsg_to_cv2(depth_msg)
        assert depth.dtype == np.uint16
        depth = depth.astype(np.float32) / 1000
        depth[depth == 0] = np.nan

        try:
            camera_to_base = self.lookup_transform(
                "panda_link0",
                info_msg.header.frame_id,
                time=info_msg.header.stamp,
                timeout=rospy.Duration(1 / self._sub_points_update_rate),
            )
        except tf.ExtrapolationException:
            return

        pcd = reorientbot.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        pcd = reorientbot.geometry.transform_points(
            pcd, reorientbot.geometry.transformation_matrix(*camera_to_base)
        )

        height = int(round(rgb.shape[0] * np.sqrt(self._sub_points_density)))
        rgb = imgviz.resize(rgb, height=height)
        pcd = imgviz.resize(pcd, height=height)

        sub_points_pybullet_id = reorientbot.pybullet.draw_points(
            pcd, rgb, size=1
        )

        if self._sub_points_pybullet_id is not None:
            pp.remove_debug(self._sub_points_pybullet_id)
        self._sub_points_pybullet_id = sub_points_pybullet_id
        self._sub_points_stamp = info_msg.header.stamp

    # no idea
    def lookup_transform(self, target_frame, source_frame, time, timeout=None):
        if timeout is not None:
            self._tf_listener.waitForTransform(
                target_frame=target_frame,
                source_frame=source_frame,
                time=time,
                timeout=timeout,
            )
        return self._tf_listener.lookupTransform(
            target_frame=target_frame, source_frame=source_frame, time=time
        )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-c", dest="cmd")
    args = parser.parse_args()

    rospy.init_node("base_task_interface")
    self = BaseTaskInterface(real=True)  # NOQA

    while True:
        self._real2robot()
        time.sleep(0.01)

    # if args.cmd:
    #     exec(args.cmd)
    # IPython.embed(header="base_task_interface")


if __name__ == "__main__":
    main()
