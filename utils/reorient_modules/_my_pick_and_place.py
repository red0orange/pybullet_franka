import itertools
import time

import cv2
from loguru import logger
import numpy as np
import pybullet_planning as pp
from pybullet_planning.interfaces.debug_utils.debug_utils import draw_pose

from . import reorientbot
from . import _utils


def my_get_pre_place_pose(place_pose, axis="z", value=0.3):
    c = reorientbot.geometry.Coordinate(*place_pose)
    if axis == "x":
        c.translate([value, 0, 0], wrt="world")
    elif axis == "y":
        c.translate([0, value, 0], wrt="world")
    elif axis == "z":
        c.translate([0, 0, value], wrt="world")
    else:
        raise BaseException("error param")
    return c.pose


def my_get_query_ocs(
    object_id,
    pre_place_pose,
    place_pose,
):
    """生成place object的place pose

    Args:
        object_id (_type_): _description_
        pre_place_pose (_type_): _description_
        place_pose (_type_): _description_

    Returns:
        _type_: _description_
    """
    lock_renderer = pp.LockRenderer()
    world_saver = pp.WorldSaver()

    pp.set_pose(object_id, place_pose)

    T_camera_to_world = reorientbot.geometry.look_at(
        pre_place_pose[0], place_pose[0])
    fovy = np.deg2rad(60)
    height = 240
    width = 240

    if False:
        reorientbot.pybullet.draw_camera(
            fovy,
            height,
            width,
            pose=reorientbot.geometry.pose_from_matrix(T_camera_to_world),
        )

    rgb, depth, segm = reorientbot.pybullet.get_camera_image(
        T_camera_to_world, fovy, height, width
    )

    K = reorientbot.geometry.opengl_intrinsic_matrix(fovy, height, width)
    pcd_in_camera = reorientbot.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    pcd_in_world = reorientbot.geometry.transform_points(
        pcd_in_camera, T_camera_to_world
    )
    normals_in_world = reorientbot.geometry.normals_from_pointcloud(
        pcd_in_world)
    normals_in_world *= -1  # flip normals

    mask = segm == object_id

    normals_on_obj = normals_in_world.copy()
    normals_on_obj[~mask] = 0
    laplacian = cv2.Laplacian(normals_on_obj, cv2.CV_64FC3)
    magnitude = np.linalg.norm(laplacian, axis=2)
    edge_mask = magnitude > 0.5
    edge_mask = (
        cv2.dilate(np.uint8(edge_mask) * 255,
                   kernel=np.ones((12, 12)), iterations=2)
        == 255
    )
    mask = mask & ~edge_mask

    # if pp.has_gui():
    #     import imgviz

    #     imgviz.io.cv_imshow(
    #         np.hstack(
    #             (
    #                 rgb,
    #                 imgviz.depth2rgb(depth),
    #                 imgviz.bool2ubyte(
    #                     np.concatenate([edge_mask[..., None]] * 3, axis=2)
    #                 ),
    #                 imgviz.bool2ubyte(np.concatenate([mask[..., None]] * 3, axis=2)),
    #             )
    #         ),
    #         "get_query_ocs",
    #     )
    #     imgviz.io.cv_waitkey(100)

    world_to_obj = pp.invert(pp.get_pose(object_id))
    pcd_in_obj = reorientbot.geometry.transform_points(
        pcd_in_world[mask],
        reorientbot.geometry.transformation_matrix(*world_to_obj),
    )
    normals_in_obj = (
        reorientbot.geometry.transform_points(
            pcd_in_world[mask] + normals_in_world[mask],
            reorientbot.geometry.transformation_matrix(*world_to_obj),
        )
        - pcd_in_obj
    )

    world_saver.restore()
    lock_renderer.restore()
    return pcd_in_obj, normals_in_obj


def my_plan_place(
    env,
    pick_pose,
    grasp_poses,
    last_pre_place_pose,
    pre_place_pose,
    place_pose,
):
    end_effortor_name = "panda_hand"

    obj_to_world = pick_pose

    j_init = env.pi.getj()

    nfail_j_grasp = 0
    max_nfail_j_grasp = 9
    for grasp_pose in grasp_poses:
        if nfail_j_grasp >= max_nfail_j_grasp:
            continue

        world_saver = pp.WorldSaver()

        ee_to_world = pp.multiply(obj_to_world, np.hsplit(grasp_pose, [3]))

        # find self-collision-free j_grasp
        for dg in np.random.uniform(-np.pi, np.pi, size=(6,)):
            result = {}

            c = reorientbot.geometry.Coordinate(*ee_to_world)
            c.rotate([0, 0, dg])

            c.translate([0, 0, -0.07])

            j = env.pi.solve_ik(c.pose)
            if j is None or not env.pi.validatej(j, obstacles=env.bg_objects):
                world_saver.restore()
                env.pi.attachments = []
                print("no j_grasp")
                nfail_j_grasp += 1
                continue
            result["j_grasp"] = j

            env.pi.setj(result["j_grasp"])
            ee_to_world = c.pose

            c = reorientbot.geometry.Coordinate(*ee_to_world)
            c.translate([0, 0, -0.1])
            j = env.pi.solve_ik(c.pose)
            if j is None or not env.pi.validatej(j, obstacles=env.bg_objects):
                world_saver.restore()
                env.pi.attachments = []
                print("no j_pre_grasp")
                continue
            result["j_pre_grasp"] = j

            ee_to_obj = pp.multiply(pp.invert(obj_to_world), ee_to_world)
            result["attachments"] = [
                pp.Attachment(
                    env.pi.robot,
                    env.pi.ee,
                    pp.invert(ee_to_obj),
                    env.fg_object_id,
                )
            ]
            env.pi.attachments = result["attachments"]

            c = reorientbot.geometry.Coordinate(*ee_to_world)
            c.translate([0, 0, -0.2], wrt="local")
            j = env.pi.solve_ik(c.pose)
            if j is None or not env.pi.validatej(j, obstacles=env.bg_objects):
                world_saver.restore()
                env.pi.attachments = []
                print("no j_post_grasp")
                continue
            result["j_post_grasp"] = j

            env.pi.setj(result["j_post_grasp"])

            with env.pi.enabling_attachments():
                j = env.pi.solve_ik(
                    pre_place_pose,
                    move_target=env.pi.robot_model.attachment_link0,
                    n_init=10,
                    validate=True,
                )
            if j is None:
                world_saver.restore()
                env.pi.attachments = []
                print("no j_pre_place")
                continue
            result["j_pre_place"] = j

            env.pi.setj(result["j_pre_place"])
            # env.pi.attachments[0].assign()

            if last_pre_place_pose is None:
                result["j_last_pre_place"] = None
            else:
                with env.pi.enabling_attachments():
                    j = env.pi.solve_ik(
                        last_pre_place_pose,
                        move_target=env.pi.robot_model.attachment_link0,
                        n_init=3,
                    )
                if j is None:
                    world_saver.restore()
                    env.pi.attachments = []
                    print("no j_last_pre_place")
                    continue
                result["j_last_pre_place"] = j

            if result["j_last_pre_place"] is not None:
                env.pi.setj(result["j_last_pre_place"])

            with env.pi.enabling_attachments():
                j = env.pi.solve_ik(
                    place_pose,
                    move_target=env.pi.robot_model.attachment_link0,
                )
            env.pi.attachments = []
            if j is None or not env.pi.validatej(j, obstacles=env.bg_objects):
                world_saver.restore()
                env.pi.attachments = []
                print("no j_place")
                continue
            env.pi.attachments = result["attachments"]
            result["j_place"] = j

            break
        else:
            world_saver.restore()
            env.pi.attachments = []
            continue

        env.pi.attachments = []

        env.pi.setj(j_init)
        js = env.pi.planj(
            result["j_pre_grasp"],
            obstacles=env.bg_objects + env.object_ids,  # @note 设置障碍
            min_distances=reorientbot.utils.StaticDict(-0.01),
        )
        if js is None:
            logger.warning("js_pre_grasp is not found")
            world_saver.restore()
            env.pi.attachments = []
            continue
        result["js_pre_grasp"] = js

        # @note add
        env.pi.setj(result["j_pre_grasp"])
        js = env.pi.planj(
            result["j_grasp"],
            obstacles=env.bg_objects + env.object_ids,  # @note 设置障碍
            min_distances=reorientbot.utils.StaticDict(-0.01),
        )
        if js is None:
            logger.warning("js_grasp is not found")
            world_saver.restore()
            env.pi.attachments = []
            continue
        result["js_grasp"] = js
        ################

        env.pi.setj(result["j_post_grasp"])

        obstacles = env.bg_objects + env.object_ids
        obstacles.remove(env.fg_object_id)

        env.pi.attachments = result["attachments"]
        js = env.pi.planj(
            result["j_pre_place"],
            obstacles=obstacles,
        )
        if js is None:
            logger.warning("js_pre_place is not found")
            world_saver.restore()
            env.pi.attachments = []
            continue
        result["js_pre_place"] = js

        env.pi.setj(result["j_pre_place"])
        pose1 = env.pi.get_pose(end_effortor_name)
        if result["j_last_pre_place"] is None:
            env.pi.setj(result["j_place"])
        else:
            env.pi.setj(result["j_last_pre_place"])
        pose2 = env.pi.get_pose(end_effortor_name)

        env.pi.setj(result["j_pre_place"])
        js = []
        env.pi.attachments = []
        for pose in pp.interpolate_poses_by_num_steps(pose1, pose2, num_steps=5):
            j = env.pi.solve_ik(pose, rthre=np.deg2rad(10), thre=0.01)
            if j is None or not env.pi.validatej(
                j,
                obstacles=obstacles,
                min_distances=reorientbot.utils.StaticDict(-0.01),
            ):
                break
            env.pi.setj(j)
            js.append(j)
        if len(js) != 6:
            logger.warning("js_place is not found")
            world_saver.restore()
            env.pi.attachments = []
            continue
        env.pi.setj(result["j_place"])
        pose = env.pi.get_pose(end_effortor_name)
        j = env.pi.solve_ik(pose)
        if j is not None:
            js.append(j)
        result["js_place"] = js

        env.pi.setj(result["j_place"])
        pose1 = env.pi.get_pose(end_effortor_name)
        env.pi.setj(result["j_pre_place"])
        pose2 = env.pi.get_pose(end_effortor_name)
        js = []
        for pose in pp.interpolate_poses_by_num_steps(pose1, pose2, num_steps=5):
            j = env.pi.solve_ik(pose)
            if j is not None:
                env.pi.setj(j)
                js.append(j)
        js.append(result["j_pre_place"])
        result["js_post_place"] = js

        break

    world_saver.restore()
    env.pi.attachments = []
    return result


def execute_place(env, result):
    for _ in (_ for j in result["js_pre_grasp"] for _ in env.pi.movej(j)):
        pp.step_simulation()
        if pp.has_gui():
            time.sleep(pp.get_time_step())

    env.pi.grasp()

    for _ in env.pi.movej(env.pi.homej):
        pp.step_simulation()
        if pp.has_gui():
            time.sleep(pp.get_time_step())

    js = result["js_pre_place"]
    for _ in (_ for j in js for _ in env.pi.movej(j)):
        pp.step_simulation()
        if pp.has_gui():
            time.sleep(pp.get_time_step())

    js = result["js_place"]
    for _ in (_ for j in js for _ in env.pi.movej(j, speed=0.005)):
        pp.step_simulation()
        if pp.has_gui():
            time.sleep(pp.get_time_step())

    for _ in range(240):
        pp.step_simulation()
        if pp.has_gui():
            time.sleep(pp.get_time_step())

    env.pi.ungrasp()

    for _ in range(240):
        pp.step_simulation()
        if pp.has_gui():
            time.sleep(pp.get_time_step())

    js = result["js_place"][::-1]
    for _ in (_ for j in js for _ in env.pi.movej(j, speed=0.005)):
        pp.step_simulation()
        if pp.has_gui():
            time.sleep(pp.get_time_step())

    for _ in env.pi.movej(env.pi.homej):
        pp.step_simulation()
        if pp.has_gui():
            time.sleep(pp.get_time_step())


def get_query_ocs(env):
    lock_renderer = pp.LockRenderer()
    world_saver = pp.WorldSaver()

    pp.set_pose(env.fg_object_id, env.PLACE_POSE)

    T_camera_to_world = reorientbot.geometry.look_at(
        env.PRE_PLACE_POSE[0], env.PLACE_POSE[0]
    )
    fovy = np.deg2rad(60)
    height = 240
    width = 240
    if env.debug:
        reorientbot.pybullet.draw_camera(
            fovy,
            height,
            width,
            pose=reorientbot.geometry.pose_from_matrix(T_camera_to_world),
        )
    rgb, depth, segm = reorientbot.pybullet.get_camera_image(
        T_camera_to_world, fovy, height, width
    )

    K = reorientbot.geometry.opengl_intrinsic_matrix(fovy, height, width)
    pcd_in_camera = reorientbot.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    pcd_in_world = reorientbot.geometry.transform_points(
        pcd_in_camera, T_camera_to_world
    )
    normals_in_world = reorientbot.geometry.normals_from_pointcloud(
        pcd_in_world)
    normals_in_world *= -1  # flip normals

    mask = segm == env.fg_object_id

    normals_on_obj = normals_in_world.copy()
    normals_on_obj[~mask] = 0
    laplacian = cv2.Laplacian(normals_on_obj, cv2.CV_64FC3)
    magnitude = np.linalg.norm(laplacian, axis=2)
    edge_mask = magnitude > 0.5
    edge_mask = (
        cv2.dilate(np.uint8(edge_mask) * 255,
                   kernel=np.ones((12, 12)), iterations=2)
        == 255
    )
    mask = mask & ~edge_mask

    # if pp.has_gui():
    #     import imgviz

    #     imgviz.io.cv_imshow(
    #         np.hstack(
    #             (
    #                 rgb,
    #                 imgviz.depth2rgb(depth),
    #                 imgviz.bool2ubyte(
    #                     np.concatenate([edge_mask[..., None]] * 3, axis=2)
    #                 ),
    #                 imgviz.bool2ubyte(np.concatenate([mask[..., None]] * 3, axis=2)),
    #             )
    #         ),
    #         "get_query_ocs",
    #     )
    #     imgviz.io.cv_waitkey(100)

    world_to_obj = pp.invert(pp.get_pose(env.fg_object_id))
    pcd_in_obj = reorientbot.geometry.transform_points(
        pcd_in_world[mask],
        reorientbot.geometry.transformation_matrix(*world_to_obj),
    )
    normals_in_obj = (
        reorientbot.geometry.transform_points(
            pcd_in_world[mask] + normals_in_world[mask],
            reorientbot.geometry.transformation_matrix(*world_to_obj),
        )
        - pcd_in_obj
    )

    world_saver.restore()
    lock_renderer.restore()
    return pcd_in_obj, normals_in_obj
