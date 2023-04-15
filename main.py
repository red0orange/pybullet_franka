import rospy
from tqdm import tqdm
from cmath import pi
import copy
import pickle
import time

import numpy as np
import path
import pybullet as p
import pybullet_planning as pp

from finger_panda_robot_interface import FingerPandaRobotInterface
from _get_heightmap import get_heightmap
import geometry
import utils

home = path.Path("~").expanduser()


class Interface:

    table_center = np.array([0.55, 0, -0.15])
    pile_center = np.array([0.45, 0, -0.02])
    pile_size = np.array([[-0.15, -0.25, 0.10], [0.15, 0.25, 0.35]])
    table_center.setflags(write=False)
    pile_center.setflags(write=False)

    def __init__(self):
        self._env = Env("franka_panda/panda_finger")
        self._env.reset()
        pass

    def movejs(self, js, sleep_time=1/240, *argc, **argv):
        for j in js:
            for _ in self._env.pi.movej(j):
                pp.step_simulation()
                time.sleep(sleep_time)
        pass

    def scan(self):
        ############################# 生成扫描路径
        all_scan_eyes = []
        all_scan_targets = []

        def sample_ellipse(center_x, center_y, a, b, num_points):
            t = np.linspace(0, 2*np.pi, num_points)
            x = center_x + a*np.cos(t)
            y = center_y + b*np.sin(t)
            return x, y

        # cycle 1
        scan_eye_height = 0.4
        scan_target_height = 0.2
        scan_ell_a_scale_fator = 1.6
        scan_ell_b_scale_fator = 1.2
        scan_target_size_factor = 1 / 4
        scan_num_points = 50

        pile_center = self.pile_center
        pile_size = self.pile_size[:, :2]
        pile_width = pile_size[1, 0] - pile_size[0, 0]
        pile_height = pile_size[1, 1] - pile_size[0, 1]

        scan_ell_a = (pile_width / 2) * scan_ell_a_scale_fator
        scan_ell_b = (pile_height / 2) * scan_ell_b_scale_fator

        scan_ell_dx, scan_ell_dy = sample_ellipse(0, 0, scan_ell_a, scan_ell_b, scan_num_points)
        scan_eyes = np.concatenate([(scan_ell_dx + pile_center[0])[None, ...], (scan_ell_dy + pile_center[1])[None, ...], np.full_like(scan_ell_dx, scan_eye_height)[None, ...]], axis=0).T
        scan_targets = np.concatenate([(scan_ell_dx * scan_target_size_factor + pile_center[0])[None, ...], (scan_ell_dy * scan_target_size_factor + pile_center[1])[None, ...], np.full_like(scan_ell_dx, scan_target_height)[None, ...]], axis=0).T
        all_scan_eyes.append(scan_eyes)
        all_scan_targets.append(scan_targets)

        # # cycle 2
        scan_eye_height = 0.5
        scan_target_height = 0.1
        scan_ell_a_scale_fator = 1.6
        scan_ell_b_scale_fator = 1.2
        scan_target_size_factor = 1 / 4
        scan_num_points = 30

        pile_center = self.pile_center
        pile_size = self.pile_size[:, :2]
        pile_width = pile_size[1, 0] - pile_size[0, 0]
        pile_height = pile_size[1, 1] - pile_size[0, 1]

        scan_ell_a = (pile_width / 2) * scan_ell_a_scale_fator
        scan_ell_b = (pile_height / 2) * scan_ell_b_scale_fator

        scan_ell_dx, scan_ell_dy = sample_ellipse(0, 0, scan_ell_a, scan_ell_b, scan_num_points)
        scan_eyes = np.concatenate([(scan_ell_dx + pile_center[0])[None, ...], (scan_ell_dy + pile_center[1])[None, ...], np.full_like(scan_ell_dx, scan_eye_height)[None, ...]], axis=0).T
        scan_targets = np.concatenate([(scan_ell_dx * scan_target_size_factor + pile_center[0])[None, ...], (scan_ell_dy * scan_target_size_factor + pile_center[1])[None, ...], np.full_like(scan_ell_dx, scan_target_height)[None, ...]], axis=0).T
        all_scan_eyes.append(scan_eyes)
        all_scan_targets.append(scan_targets)

        all_scan_eyes = np.concatenate(all_scan_eyes, axis=0)
        all_scan_targets = np.concatenate(all_scan_targets, axis=0)

        # for i in range(len(scan_eyes)):
        #     eye = scan_eyes[i]
        #     target = scan_targets[i]
        #     pp.draw_pose([[eye[0], eye[1], eye[2]], [0, 0, 0, 1]], 0.1)
        #     pp.draw_pose([[target[0], target[1], target[2]], [0, 0, 0, 1]], 0.1)

        # @note TODO 保存好的扫描路径，可以直接读取
        js = []
        flip_flags = []
        for i in range(len(all_scan_eyes)):
            eye = all_scan_eyes[i]
            target = all_scan_targets[i]

            # up = None
            if eye[0] > pile_center[0]:
                up = pile_center[0], pile_center[1], target[2]+0.50 # 始终朝中心
            else:
                up = pile_center[0], pile_center[1], target[2]-0.50 # 始终朝中心
            up = up - eye

            c = geometry.Coordinate.from_matrix(
                geometry.look_at(eye, target, up)
            )

            j = self._env.pi.solve_ik(
                c.pose,
                move_target=self._env.pi.robot_model.camera_link,
                n_init=10,
                thre=0.05,
                rthre=np.deg2rad(5),
                obstacles=[2, 3, 4, 5],
                validate=True,
            )
            if j is None:
                rospy.logerr("j is not found")
                continue

            js.append(j)
            flip_flags.append(eye[0] > pile_center[0])
            self._env.pi.setj(j)

            pp.draw_pose([c.position, c.quaternion], 0.1)

        ############################# 运行扫描的路径
        self.movejs([[0.010523236721068734, -1.4790833239639014, 0.10027132190633238, -2.416246868493765, 0.08994288870361117, 1.4022499498261343, 0.8516519819506339]])
        time.sleep(1)

        # 开始移动
        for i, j in tqdm(enumerate(js), desc="Scan Scene"):
            for _ in self._env.pi.movej(j):
                pp.step_simulation()
                time.sleep(1 / 240)
        pass


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
        pcd_in_camera = geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        pcd_in_world = geometry.transform_points(
            pcd_in_camera,
            geometry.transformation_matrix(*camera_to_world),
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

    interface = Interface()
    interface.scan()

    IPython.embed()


if __name__ == "__main__":
    main()
