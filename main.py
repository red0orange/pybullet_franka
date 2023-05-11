import time
import path
import random
from tqdm import tqdm
import copy
import pickle
import rospy

import tf
import numpy as np
import pybullet as p
import pybullet_planning as pp

from finger_panda_robot_interface import FingerPandaRobotInterface
from suction_panda_robot_interface import SuctionPandaRobotInterface
from _get_heightmap import get_heightmap
import geometry
import utils

home = path.Path("~").expanduser()


class Interface:

    table_center = np.array([0.75, 0, -0.20])
    pile_center = np.array([0.45, 0, 0.32])
    pile_size = np.array([[-0.10, -0.15, 0.0], [0.35, 0.15, 0.60]])
    table_center.setflags(write=False)
    pile_center.setflags(write=False)

    def __init__(self):
        # self._env = Env("franka_panda/panda_finger")
        self._env = Env("franka_panda/panda_suction")
        self._env.reset()

        # 放置障碍物
        seed = 111
        cube_num = 15
        fail_stop_thres = 1000
        pile_aabb = [
            self.pile_center + self.pile_size[0],
            self.pile_center + self.pile_size[1],
        ]
        # pp.draw_aabb(pile_aabb)
        self.build_scene(pile_aabb, fail_stop_thres, cube_num, seed=seed)
        pass

    def build_scene(self, pile_aabb, fail_stop_thres, cube_num, seed=111):
        def create_block(pos, orientation, block_scale, block_mass=1.0):
            block_visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                                        halfExtents=[block_scale]*3,
                                                        rgbaColor=[1, 1, 0, 1])
            block_collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                            halfExtents=[block_scale]*3)
            block_body_id = p.createMultiBody(baseMass=block_mass, baseVisualShapeIndex=block_visual_shape_id,
                                            baseCollisionShapeIndex=block_collision_shape_id,
                                            basePosition=pos, baseOrientation=orientation)

            return block_body_id

        # 放桌子
        with pp.LockRenderer():
            self.table_id = pp.load_pybullet("table/table.obj", scale=1.0)   # table_id: 2
            quat = tf.transformations.quaternion_from_euler(0, 0, np.pi/2)
            pp.set_pose(self.table_id, ([self.table_center[0] - 0.1, self.table_center[1], self.table_center[2]], quat))

        block_scale = 0.05
        col_num = 3
        row_num = 3
        height_num = 3
        # col_num = 2
        # row_num = 2
        # height_num = 3

        random.seed(seed)
        cube_matrix = np.random.randint(1, height_num, size=(row_num, col_num))
        self.cube_centers = np.zeros((row_num, col_num, 3))

        object_ids = [self.table_id]
        start_pos = pile_aabb[0] + [0, 0, 0]
        for i in range(row_num):
            for j in range(col_num):
                self.cube_centers[i, j] = [start_pos[0] + i*block_scale*2, start_pos[1] + j*block_scale*2, start_pos[2] + (cube_matrix[i, j])*(block_scale*2) + 0.02]
                for k in range(cube_matrix[i, j]):
                    pos = [start_pos[0] + i*block_scale*2, start_pos[1] + j*block_scale*2, start_pos[2] + k*(block_scale*2)]
                    obj_id = create_block(pos, [0, 0, 0, 1], block_scale)
                    object_ids.append(obj_id)
        self.object_ids = object_ids

        for _ in range(int(1 / pp.get_time_step())):
            p.stepSimulation()
            if self._env._gui:
                time.sleep(pp.get_time_step())

        return object_ids

    def set_env(self):
        range_x = (-0.5, 0.5)
        range_y = (-0.5, 0.5)
        range_z = (0, 0) # 将方块放置在平面上
        num_blocks = 3

        for i in range(num_blocks):
            pos = random_position(range_x, range_y, range_z)
            block_id = create_block(pos)
        pass

    def movejs(self, js, sleep_time=1/240, *argc, **argv):
        for j in js:
            for _ in self._env.pi.movej(j):
                pp.step_simulation()
                time.sleep(sleep_time)
        pass

    def pick(self):
        all_test_poses = np.zeros(self.cube_centers.shape[:2], dtype=object)
        for i, j in np.ndindex(self.cube_centers.shape[:2]):
            eye = self.cube_centers[i, j]
            target = eye + [0, 0, -0.1]
            c = geometry.Coordinate.from_matrix(
                geometry.look_at(eye, target, None)
            )
            all_test_poses[i, j] = c.pose

        # 1b. 使用 Integrated RRT 的方法
        use_traditional = False
        if not use_traditional:
            for i, j in np.ndindex(self.cube_centers.shape[:2]):
                pose = all_test_poses[i, j]
                T = utils.tf.pybullet2T(pose)
                target_js = self._env.pi.plan(T, obstacles=self.object_ids)
                if target_js is None:
                    print("target_js is None")
                    continue
                self.movejs(target_js, time_scale=5, retry=True)
                print("Goal Finish: {} {}".format(i, j))

                # home
                home_T = utils.tf.pybullet2T(self._env.pi.home_pose)
                home_js = self._env.pi.plan(home_T, obstacles=self.object_ids)
                if home_js is None:
                    print("home_js is None")
                    continue
                self.movejs(home_js, time_scale=5, retry=True)
                print("Home Finish: {} {}".format(i, j))
        else:
            all_test_js = np.zeros(self.cube_centers.shape[:2], dtype=object)
            for i, j in np.ndindex(self.cube_centers.shape[:2]):
                pp.draw_pose(all_test_poses[i, j], length=0.1, width=0.01)
                joint = self._env.pi.solve_ik(
                    all_test_poses[i, j],
                    move_target=self._env.pi.robot_model.tipLink,
                    n_init=100,
                    thre=0.05,
                    rthre=np.deg2rad(5),
                    obstacles=self.object_ids,
                    validate=True,
                )
                all_test_js[i, j] = joint

            for i, j in np.ndindex(self.cube_centers.shape[:2]):
                if i != 3: continue
                # target
                target_j = all_test_js[i, j]
                if target_j is None:
                    print("target_j is None")
                    continue
                target_js = self._env.pi.planj(target_j, obstacles=self.object_ids)
                if target_js is None:
                    print("target_js is None")
                    continue
                self.movejs(target_js, time_scale=5, retry=True)

                # home
                home_js = self._env.pi.planj(self._env.pi.homej, obstacles=self.object_ids)
                if home_js is None:
                    print("home_js is None")
                    continue
                self.movejs(home_js, time_scale=5, retry=True)

            # for i, j in tqdm(enumerate(all_test_js), desc="Scan Scene"):
            #     if j is None: continue
            #     for _ in self._env.pi.movej(j):
            #         pp.step_simulation()
            #         time.sleep(1 / 240)
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
        # self.pi = FingerPandaRobotInterface(
        #     max_force=None,
        #     surface_threshold=np.inf,
        #     surface_alignment=False,
        #     planner="RRTConnect",
        #     robot_model=self._robot_model,
        # )
        self.pi = SuctionPandaRobotInterface(
            suction_max_force=None,
            suction_surface_threshold=np.inf,
            suction_surface_alignment=False,
            planner="RRTConnect",
            robot_model=self._robot_model,
        )

        c = geometry.Coordinate()
        c.translate([0.00, -0.1, -0.1])
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
    interface.pick()

    while True:
        p.stepSimulation()
        time.sleep(0.05)

    IPython.embed()


if __name__ == "__main__":
    main()
