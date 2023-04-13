from cmath import pi
import copy
import pickle
import time

# from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

from finger_panda_robot_interface import FingerPandaRobotInterface
from _get_heightmap import get_heightmap
import geometry
import utils

home = path.Path("~").expanduser()


class Env:

    # parameters
    IMAGE_HEIGHT = 480
    IMAGE_WIDTH = 640

    TABLE_OFFSET = 0.0

    HEIGHTMAP_PIXEL_SIZE = 0.004
    HEIGHTMAP_IMAGE_SIZE = 128
    HEIGHTMAP_SIZE = HEIGHTMAP_PIXEL_SIZE * HEIGHTMAP_IMAGE_SIZE

    # PILES_DIR = home / ".cache/reorientbot/pile_generation"
    # PILE_TRAIN_IDS = np.arange(0, 1000)
    # PILE_EVAL_IDS = np.arange(1000, 1200)
    PILE_POSITION = np.array([0.5, 0, TABLE_OFFSET])
    CAMERA_POSITION = np.array([PILE_POSITION[0], PILE_POSITION[1], 0.7])

    def __init__(self, robot_model):
        self._gui = True
        self._robot_model = robot_model
        # self.debug = debug
        # self.eval = False
        self.random_state = np.random.RandomState()

        pass

    def shutdown(self):
        pp.disconnect()

    def launch(self):
        pp.connect(use_gui=self._gui)
        pp.add_data_path()

    def reset(self, pile_file=None):
        if not pp.is_connected():
            self.launch()

        # @note load plane
        pp.reset_simulation()
        pp.enable_gravity()
        p.setGravity(0, 0, -9.8)
        pp.set_camera_pose((1, -0.7, 0.8), (0.1, 0.1, 0.35))
        with pp.LockRenderer():
            # self.plane = pp.load_pybullet("plane.urdf")
            self.plane = pp.load_pybullet("plane.obj")
            p.changeVisualShape(self.plane, -1, rgbaColor=[1, 1, 1, 1])
            pp.set_pose(self.plane, ([0, 0, self.TABLE_OFFSET], [0, 0, 0, 1]))

        # @note load Panda
        self.pi = FingerPandaRobotInterface(
            max_force=None,
            surface_threshold=np.inf,
            surface_alignment=False,
            planner="RRTConnect",
            robot_model=self._robot_model,
        )

        c = geometry.Coordinate()
        c.translate([0.05, 0, 0.05])
        c.rotate([0, 0, np.pi / 2])
        pose = c.pose
        self.pi.add_camera(
            pose=pose,
            fovy=np.deg2rad(54),
            height=self.IMAGE_HEIGHT,
            width=self.IMAGE_WIDTH,
        )
        pass

    def update_obs(self):
        rgb, depth, segm = self.pi.get_camera_image()
        # if pp.has_gui():
        #     import imgviz

        #     imgviz.io.cv_imshow(
        #         np.hstack((rgb, imgviz.depth2rgb(depth))), "update_obs"
        #     )
        #     imgviz.io.cv_waitkey(100)
        fg_mask = segm == self.fg_object_id
        camera_to_world = self.pi.get_pose("camera_link")

        K = self.pi.get_opengl_intrinsic_matrix()
        pcd_in_camera = reorientbot.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        pcd_in_world = reorientbot.geometry.transform_points(
            pcd_in_camera,
            reorientbot.geometry.transformation_matrix(*camera_to_world),
        )

        aabb = np.array(
            [
                self.PILE_POSITION - self.HEIGHTMAP_SIZE / 2,
                self.PILE_POSITION + self.HEIGHTMAP_SIZE / 2,
            ]
        )
        aabb[0][2] = 0
        aabb[1][2] = 0.5
        _, _, segmmap, pointmap = get_heightmap(
            points=pcd_in_world,
            colors=rgb,
            ids=segm,
            aabb=aabb,
            pixel_size=self.HEIGHTMAP_PIXEL_SIZE,
        )

        self.obs = dict(
            rgb=rgb,
            depth=depth,
            K=self.pi.get_opengl_intrinsic_matrix(),
            camera_to_world=np.hstack(camera_to_world),
            segmmap=segmmap,
            pointmap=pointmap,
            target_instance_id=self.fg_object_id,
            fg_mask=fg_mask.astype(np.uint8),
            segm=segm,
        )


def main():

    import IPython

    env = Env(robot_model="franka_panda/panda_finger")
    env.reset()
    IPython.embed()


if __name__ == "__main__":
    main()
