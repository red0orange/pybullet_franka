from cached_property import cached_property
import six
import path

import skrobot
from skrobot.utils import urdf
from skrobot.utils.urdf import URDF
from skrobot.coordinates import rpy_angle
from skrobot.coordinates import Coordinates
from skrobot.model.link import Link
from skrobot.model.joint import LinearJoint
from skrobot.model.joint import RotationalJoint
from skrobot.model.joint import FixedJoint

from . import reorientbot


class RobotModel(skrobot.model.RobotModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pass

    def load_urdf_file(self, file_obj):
        # result = super().load_urdf_file(file_obj)

        if isinstance(file_obj, six.string_types):
            self.urdf_path = file_obj
        else:
            self.urdf_path = getattr(file_obj, 'name', None)
        self.urdf_robot_model = URDF.load(file_obj=file_obj)
        root_link = self.urdf_robot_model.base_link

        links = []
        for urdf_link in self.urdf_robot_model.links:
            link = Link(name=urdf_link.name)
            link.collision_mesh = urdf_link.collision_mesh
            link.visual_mesh = self._meshes_from_urdf_visuals(
                urdf_link.visuals)
            links.append(link)
        link_maps = {l.name: l for l in links}

        joint_list = []
        whole_joint_list = []
        for j in self.urdf_robot_model.joints:
            if j.limit is None:
                j.limit = urdf.JointLimit(0, 0)
            if j.axis is None:
                j.axis = 'z'
            if j.joint_type == 'fixed':
                joint = FixedJoint(
                    name=j.name,
                    parent_link=link_maps[j.parent],
                    child_link=link_maps[j.child])
            elif j.joint_type == 'revolute':
                joint = RotationalJoint(
                    axis=j.axis,
                    name=j.name,
                    parent_link=link_maps[j.parent],
                    child_link=link_maps[j.child],
                    min_angle=j.limit.lower,
                    max_angle=j.limit.upper,
                    max_joint_torque=j.limit.effort,
                    max_joint_velocity=j.limit.velocity)
            elif j.joint_type == 'continuous':
                joint = RotationalJoint(
                    axis=j.axis,
                    name=j.name,
                    parent_link=link_maps[j.parent],
                    child_link=link_maps[j.child],
                    min_angle=-np.inf,
                    max_angle=np.inf,
                    max_joint_torque=j.limit.effort,
                    max_joint_velocity=j.limit.velocity)
            elif j.joint_type == 'prismatic':
                # http://wiki.ros.org/urdf/XML/joint
                # meters for prismatic joints
                joint = LinearJoint(
                    axis=j.axis,
                    name=j.name,
                    parent_link=link_maps[j.parent],
                    child_link=link_maps[j.child],
                    min_angle=j.limit.lower,
                    max_angle=j.limit.upper,
                    max_joint_torque=j.limit.effort,
                    max_joint_velocity=j.limit.velocity)

            if j.joint_type not in ['fixed']:
                if j.name not in ["panda_finger_joint1", "panda_finger_joint2"]:
                    joint_list.append(joint)
            whole_joint_list.append(joint)

            # TODO(make clear the difference between assoc and add_child_link)
            link_maps[j.parent].assoc(link_maps[j.child])
            link_maps[j.child].add_joint(joint)
            link_maps[j.child].add_parent_link(link_maps[j.parent])
            link_maps[j.parent].add_child_link(link_maps[j.child])

        for j in self.urdf_robot_model.joints:
            if j.origin is None:
                rpy = np.zeros(3, dtype=np.float32)
                xyz = np.zeros(3, dtype=np.float32)
            else:
                rpy = rpy_angle(j.origin[:3, :3])[0]
                xyz = j.origin[:3, 3]
            link_maps[j.child].newcoords(rpy,
                                         xyz)
            # TODO(fix automatically update default_coords)
            link_maps[j.child].joint.default_coords = Coordinates(
                pos=link_maps[j.child].translation,
                rot=link_maps[j.child].rotation)

        # TODO(duplicate of __init__)
        self.link_list = links
        self.joint_list = joint_list

        self.joint_names = []
        for joint in self.joint_list:
            self.joint_names.append(joint.name)

        for link in self.link_list:
            self.__dict__[link.name] = link
        for joint in whole_joint_list:
            self.__dict__[joint.name] = joint
        self.root_link = self.__dict__[root_link.name]
        self.assoc(self.root_link)

        # Add hook of mimic joint.
        for j in self.urdf_robot_model.joints:
            if j.mimic is None:
                continue
            joint_a = self.__dict__[j.mimic.joint]
            joint_b = self.__dict__[j.name]
            multiplier = j.mimic.multiplier
            offset = j.mimic.offset
            joint_a.register_mimic_joint(joint_b, multiplier, offset)

        self._relevance_predicate_table = \
            self._compute_relevance_predicate_table()
        pass


class RobotModelFromURDF(RobotModel):

    def __init__(self, urdf=None, urdf_file=None):
        super(RobotModelFromURDF, self).__init__()

        if urdf and urdf_file:
            raise ValueError(
                "'urdf' and 'urdf_file' cannot be given at the same time"
            )
        if urdf:
            self.load_urdf(urdf=urdf)
        elif urdf_file:
            self.load_urdf_file(file_obj=urdf_file)
        else:
            self.load_urdf_file(file_obj=self.default_urdf_path)

    @property
    def default_urdf_path(self):
        raise NotImplementedError


class Panda(RobotModelFromURDF):

    """Panda Robot Model.

    https://frankaemika.github.io/docs/control_parameters.html
    """

    def __init__(self, *args, **kwargs):
        super(Panda, self).__init__(*args, **kwargs)
        self.reset_pose()

    @cached_property
    def default_urdf_path(self):
        return panda_urdfpath()

    def reset_pose(self):
        angle_vector = [
            0.03942226991057396,
            -0.9558116793632507,
            -0.014800949953496456,
            -2.130282163619995,
            -0.013104429468512535,
            1.1745525598526,
            0.8112226724624634,
        ]
        for link, angle in zip(self.rarm.link_list, angle_vector):
            link.joint.joint_angle(angle)
        return self.angle_vector()

    @cached_property
    def rarm(self):
        link_names = ['panda_link{}'.format(i) for i in range(1, 8)]
        links = [getattr(self, n) for n in link_names]
        joints = [l.joint for l in links]
        model = RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self.panda_hand
        return model


class MyPanda(Panda):
    def __init__(self, *args, **kwargs):
        root_dir = path.Path(reorientbot.__file__).parent
        urdf_file = (
            root_dir / "_pybullet/data/franka_panda/panda_finger.urdf"
        )  # NOQA
        super().__init__(urdf_file=urdf_file)
        print(self.joint_list)
        print(self.link_list)
        pass

    @property
    def rarm(self):
        link_names = ["panda_link{}".format(i) for i in range(1, 8)]
        links = [getattr(self, n) for n in link_names]
        joints = [link.joint for link in links]
        model = skrobot.model.RobotModel(link_list=links, joint_list=joints)
        model.end_coords = self.panda_hand
        return model


def main():
    import IPython
    import numpy as np

    viewer = skrobot.viewers.TrimeshSceneViewer()

    robot_model = Panda()
    viewer.add(robot_model)
    viewer.add(skrobot.model.Box((2, 2, 0), vertex_colors=(0.7, 0.7, 0.7)))

    viewer.set_camera(
        angles=[np.deg2rad(80), 0, np.deg2rad(60)],
        distance=2,
        center=(0, 0, 0.5),
    )

    viewer.show()
    IPython.embed()


if __name__ == "__main__":
    main()
