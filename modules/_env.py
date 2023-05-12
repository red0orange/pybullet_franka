from cmath import pi
import copy
import pickle
import time

from loguru import logger
import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

from . import reorientbot
from . import _utils
from ._get_heightmap import get_heightmap


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

    def __init__(
        self,
        gui=True,
        retime=1,
        step_callback=None,
        mp4=None,
        face="front",
        robot_model="franka_panda/panda_finger",
        debug=True,
    ):
        super().__init__()

        self._gui = gui
        self._retime = retime
        self._step_callback = step_callback
        self._mp4 = mp4
        self._face = face
        self._robot_model = robot_model

        self.debug = debug
        self.eval = False
        self.random_state = np.random.RandomState()

    def shutdown(self):
        pp.disconnect()

    def launch(self):
        pp.connect(use_gui=self._gui, mp4=self._mp4)
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
        self.pi = reorientbot.pybullet.FingerPandaRobotInterface(
            max_force=None,
            surface_threshold=np.inf,
            surface_alignment=False,
            planner="RRTConnect",
            robot_model=self._robot_model,
        )

        c = reorientbot.geometry.Coordinate()
        c.translate([0.05, 0, 0.05])
        c.rotate([0, 0, np.pi / 2])
        pose = c.pose
        self.pi.add_camera(
            pose=pose,
            fovy=np.deg2rad(54),
            height=self.IMAGE_HEIGHT,
            width=self.IMAGE_WIDTH,
        )

        self.object_ids = None
        self.fg_object_id = None
        self.PLACE_POSE = None
        self.LAST_PRE_PLACE_POSE = None
        self.PRE_PLACE_POSE = None
        self._shelf = -1

        self.bg_objects = [self.plane, self._shelf]

    def setj_to_camera_pose(self):
        self.pi.setj(self.pi.homej)
        j = None
        while j is None:
            c = reorientbot.geometry.Coordinate(
                *self.pi.get_pose("camera_link"))
            c.position = self.CAMERA_POSITION
            j = self.pi.solve_ik(
                c.pose, move_target=self.pi.robot_model.camera_link)
        self.pi.setj(j)

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
    import argparse

    import IPython

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    choices = ["franka_panda/panda_suction", "franka_panda/panda_drl"]
    parser.add_argument(
        "--robot-model",
        default=choices[0],
        choices=choices,
        help=" ",
    )
    args = parser.parse_args()

    env = Env(class_ids=[2, 3, 5, 11, 12, 15], robot_model=args.robot_model)
    env.reset()
    IPython.embed()


if __name__ == "__main__":
    main()
