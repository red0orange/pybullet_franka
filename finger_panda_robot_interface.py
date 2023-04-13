import contextlib
import itertools

from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning

import geometry
import utils
import pybullet_utils
from ompl_planning import PbPlanner
from finger_gripper import FingerGripper

import skrobot


here = path.Path(__file__).abspath().parent


class FingerPandaRobotInterface:
    def __init__(
        self,
        pose=None,
        max_force=10,
        surface_threshold=np.deg2rad(10),
        surface_alignment=True,
        planner="RRTConnect",
        robot_model="franka_panda/panda_finger",
    ):
        self.pose = pose

        urdf_file = here / f"data/{robot_model}.urdf"
        self.robot_model = skrobot.models.urdf.RobotModelFromURDF(
            urdf_file=urdf_file
        )
        # @note load Panda in Pybullet
        self.robot = pybullet_planning.load_pybullet(
            urdf_file, fixed_base=True
        )
        self.ee_link_name = "panda_hand"
        self.ee = pybullet_planning.link_from_name(
            self.robot, self.ee_link_name)
        self.left_finger = pybullet_planning.link_from_name(
            self.robot, "panda_finger_link1")
        self.right_finger = pybullet_planning.link_from_name(
            self.robot, "panda_finger_link2")

        self.gripper = FingerGripper(
            self.robot,
            self.left_finger,
            self.right_finger,
            max_force=max_force,
            surface_threshold=surface_threshold,
            surface_alignment=surface_alignment,
        )

        self.attachments = []

        if self.pose is not None:
            self.robot_model.translate(pose[0])
            self.robot_model.orient_with_matrix(
                geometry.quaternion_matrix(pose[1])[:3, :3]
            )

            pybullet_planning.set_pose(self.robot, self.pose)

        # Get revolute joint indices of robot (skip fixed joints).
        n_joints = p.getNumJoints(self.robot)
        joints = [p.getJointInfo(self.robot, i) for i in range(n_joints)]
        self.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

        self.homej = [0, -np.pi / 4, 0, -np.pi / 2, 0, np.pi / 4, np.pi / 4]
        for joint, joint_angle in zip(self.joints, self.homej):
            p.resetJointState(self.robot, joint, joint_angle)
        p.resetJointState(self.robot, self.left_finger, 0.04)
        p.resetJointState(self.robot, self.right_finger, 0.04)
        # p.resetJointState(self.robot, self.left_finger, 0.0)
        # p.resetJointState(self.robot, self.right_finger, 0.0)
        self.update_robot_model()

        self.planner = planner

        lower, upper = self.get_bounds()
        for joint, min_angle, max_angle in zip(self.joints, lower, upper):
            joint_name = pybullet_planning.get_joint_name(
                self.robot, joint
            ).decode()
            getattr(self.robot_model, joint_name).min_angle = min_angle
            getattr(self.robot_model, joint_name).max_angle = max_angle

    def get_bounds(self):
        lower_bounds = []
        upper_bounds = []
        for joint in self.joints:
            lower, upper = p.getJointInfo(self.robot, joint)[8:10]
            center = (upper + lower) / 2
            width = upper - lower
            width = width * 0.96
            upper = center + width / 2
            lower = center - width / 2
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        return lower_bounds, upper_bounds

    def step_simulation(self):
        self.gripper.step_simulation()

    def update_robot_model(self, j=None):
        if j is None:
            j = self.getj()
        for joint, joint_angle in zip(self.joints, j):
            joint_name = pybullet_planning.get_joint_name(
                self.robot, joint
            ).decode()
            getattr(self.robot_model, joint_name).joint_angle(joint_angle)

    def setj(self, joint_positions):
        # set arm joints
        for joint, joint_position in zip(self.joints, joint_positions):
            p.resetJointState(self.robot, joint, joint_position)
        for attachment in self.attachments:
            attachment.assign()

    def set_finger(self, finger_positions):
        p.resetJointState(self.robot, self.left_finger, finger_positions[0])
        p.resetJointState(self.robot, self.right_finger, finger_positions[1])
        pass

    def getj(self):
        joint_positions = []
        for joint in self.joints:
            joint_positions.append(p.getJointState(self.robot, joint)[0])
        return joint_positions

    def movej(self, targj, speed=0.01, timeout=5, raise_on_timeout=False):
        assert len(targj) == len(self.joints)
        for i in itertools.count():
            currj = [p.getJointState(self.robot, i)[0] for i in self.joints]
            currj = np.array(currj)
            diffj = targj - currj
            if all(np.abs(diffj) < 1e-2):
                return

            # Move with constant velocity
            norm = np.linalg.norm(diffj)
            v = diffj / norm if norm > 0 else 0
            stepj = currj + v * speed
            gains = np.ones(len(self.joints))
            p.setJointMotorControlArray(
                bodyIndex=self.robot,
                jointIndices=self.joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=stepj,
                positionGains=gains,
            )
            yield i

            if i >= (timeout / pybullet_planning.get_time_step()):
                if raise_on_timeout:
                    raise RuntimeError("timeout in joint motor control")
                else:
                    logger.error("timeout in joint motor control")
                    return

    def solve_ik(
        self,
        pose,
        move_target=None,
        n_init=1,
        random_state=None,
        validate=False,
        obstacles=None,
        **kwargs,
    ):
        if move_target is None:
            move_target = self.robot_model.panda_hand
        if random_state is None:
            random_state = np.random.RandomState()

        def sample_fn():
            lower, upper = self.get_bounds()
            extents = upper - lower
            scale = random_state.uniform(size=len(lower))
            return lower + scale * extents

        self.update_robot_model()
        c = geometry.Coordinate(*pose)
        for _ in range(n_init):
            result = self.robot_model.inverse_kinematics(
                c.skrobot_coords,
                move_target=move_target,
                **kwargs,
            )
            if result is not False:
                if not validate:
                    break
                if self.validatej(result, obstacles=obstacles):
                    break
            self.update_robot_model(sample_fn())
        else:
            # logger.warning("Failed to solve IK")
            return
        j = []
        for joint in self.joints:
            joint_name = pybullet_planning.get_joint_name(
                self.robot, joint
            ).decode()
            j.append(getattr(self.robot_model, joint_name).joint_angle())
        return j

    @contextlib.contextmanager
    def enabling_attachments(self):
        robot_model = self.robot_model
        try:
            self.robot_model = self.get_skrobot(attachments=self.attachments)
            yield
        finally:
            self.robot_model = robot_model

    def get_skrobot(self, attachments=None):
        attachments = attachments or []

        self.update_robot_model()

        link_list = self.robot_model.link_list.copy()
        joint_list = self.robot_model.joint_list.copy()
        for i, attachment in enumerate(attachments):
            position, quaternion = attachment.grasp_pose
            link = skrobot.model.Link(
                parent=self.robot_model.panda_hand,
                pos=position,
                rot=geometry.quaternion_matrix(quaternion)[:3, :3],
                name=f"attachment_link{i}",
            )
            joint = skrobot.model.FixedJoint(
                child_link=link,
                parent_link=self.robot_model.panda_hand,
                name=f"attachment_joint{i}",
            )
            link.joint = joint
            link_list.append(link)
            joint_list.append(joint)
        return skrobot.model.RobotModel(
            link_list=link_list,
            joint_list=joint_list,
            # root_link=self.robot_model.root_link,
        )

    def validatej(self, j, obstacles=None, min_distances=None):
        planner = PbPlanner(
            self,
            obstacles=obstacles,
            min_distances=min_distances,
            planner=self.planner,
        )
        return planner.validityChecker.isValid(j)

    def planj(
        self,
        j,
        obstacles=None,
        min_distances=None,
        min_distances_start_goal=None,
        planner_range=0,
    ):
        if self.planner == "Naive":
            return [j]

        planner = PbPlanner(
            self,
            obstacles=obstacles,
            min_distances=min_distances,
            min_distances_start_goal=min_distances_start_goal,
            planner=self.planner,
            planner_range=planner_range,
        )

        planner.validityChecker.start = self.getj()
        planner.validityChecker.goal = j

        if not planner.validityChecker.isValid(self.getj()):
            logger.warning("Start state is invalid")
            return

        if not planner.validityChecker.isValid(j):
            logger.warning("Goal state is invalid")
            return

        result = planner.plan(self.getj(), j)

        if result is None:
            logger.warning("No solution found")
            return

        ndof = len(self.joints)
        state_count = result.getStateCount()
        path = np.zeros((state_count, ndof), dtype=float)
        for i_state in range(state_count):
            state = result.getState(i_state)
            path_i = np.zeros((ndof,), dtype=float)
            for i_dof in range(ndof):
                path_i[i_dof] = state[i_dof]
            path[i_state] = path_i

        if not np.allclose(j, path[-1]):
            # the goal is not reached
            return

        return path

    def grasp(self):
        self.gripper.activate()
        pass

    def ungrasp(self):
        self.gripper.release()
        # if hasattr(self, "virtual_grasped_object"):
        #     p.removeBody(self.virtual_grasped_object)
        #     del self.virtual_grasped_object
        self.attachments = []
        pass

    def add_link(self, name, pose, parent=None):
        if parent is None:
            parent = self.ee
        parent_name = pybullet_planning.get_link_name(self.robot, parent)

        link_list = self.robot_model.link_list.copy()
        joint_list = self.robot_model.joint_list.copy()
        parent_link = getattr(self.robot_model, parent_name)
        link = skrobot.model.Link(
            parent=parent_link,
            pos=pose[0],
            rot=geometry.quaternion_matrix(pose[1])[:3, :3],
            name=name,
        )
        joint = skrobot.model.FixedJoint(
            child_link=link,
            parent_link=parent_link,
            name=f"{parent_name}_to_{name}_joint",
        )
        link.joint = joint
        link_list.append(link)
        joint_list.append(joint)
        self.robot_model = skrobot.model.RobotModel(
            link_list=link_list,
            joint_list=joint_list,
            # root_link=self.robot_model.root_link,
        )

    def get_pose(self, name):
        self.update_robot_model()
        T_a_to_world = getattr(self.robot_model, name).worldcoords().T()
        a_to_world = geometry.Coordinate.from_matrix(T_a_to_world).pose
        return a_to_world

    def add_camera(
        self,
        pose,
        fovy=np.deg2rad(42),
        height=480,
        width=640,
        parent=None,
    ):
        if parent is None:
            parent = self.ee
        self.add_link("camera_link", pose=pose, parent=parent)

        # pybullet_planning.draw_pose(
        #     pose, parent=self.robot, parent_link=parent
        # )
        pybullet_utils.draw_camera(
            fovy=fovy,
            height=height,
            width=width,
            pose=pose,
            parent=self.robot,
            parent_link=parent,
        )

        self.camera = dict(fovy=fovy, height=height, width=width)

    def get_camera_image(self):
        if not hasattr(self.robot_model, "camera_link"):
            raise ValueError

        self.update_robot_model()
        return pybullet_utils.get_camera_image(
            T_cam2world=self.robot_model.camera_link.worldcoords().T(),
            fovy=self.camera["fovy"],
            height=self.camera["height"],
            width=self.camera["width"],
        )

    def get_ori_camera_image(self):
        if not hasattr(self.robot_model, "camera_link"):
            raise ValueError

        self.update_robot_model()
        return pybullet_utils.get_ori_camera_image(
            T_cam2world=self.robot_model.camera_link.worldcoords().T(),
            fovy=self.camera["fovy"],
            height=self.camera["height"],
            width=self.camera["width"],
        )

    def get_opengl_intrinsic_matrix(self):
        return geometry.opengl_intrinsic_matrix(
            fovy=self.camera["fovy"],
            height=self.camera["height"],
            width=self.camera["width"],
        )

    def move_to_homej(self, bg_object_ids, object_ids, speed=0.01, timeout=5):
        obstacles = bg_object_ids + object_ids
        if self.attachments and self.attachments[0].child in obstacles:
            obstacles.remove(self.attachments[0].child)

        js = None
        min_distance = 0
        while True:
            js = self.planj(
                self.homej,
                obstacles=obstacles,
                min_distances=utils.StaticDict(value=min_distance),
            )
            if js is not None:
                break

            if min_distance <= -0.05:
                js = [self.homej]
                break
            logger.warning(f"js is None w/ min_distance={min_distance}")
            min_distance -= 0.01
        for j in js:
            for _ in self.movej(j, speed=speed, timeout=timeout / len(js)):
                yield

    def get_cartesian_path(self, j=None, pose=None, rotation_axis=True):
        if not (j is None) ^ (pose is None):
            raise ValueError("Either j or coords must be given")

        p_start = self.get_pose(self.ee_link_name)

        with pybullet_planning.WorldSaver():
            if j is None:
                j = self.solve_ik(pose, rotation_axis=rotation_axis)
                if j is None:
                    raise RuntimeError("IK failure")
            else:
                self.setj(j)
            j_end = j

            self.setj(j_end)
            p_end = self.get_pose(self.ee_link_name)

            js_reverse = [j_end]
            for pose in pybullet_planning.interpolate_poses(p_end, p_start):
                j = self.solve_ik(
                    pose, rotation_axis=rotation_axis, validate=True
                )
                if j is None:
                    return
                js_reverse.append(j)
                self.setj(j)

        js = np.array(js_reverse[::-1])
        return js
