import bisect
import math
import copy
import itertools
from functools import partial

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
import pybullet as p
import pybullet_planning as pp

from utils.tf import *


class pbValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, is_valid_func, sample_state_func):
        super().__init__(si)
        self._isValid = is_valid_func
        self._sample_state = sample_state_func

        self.start = None
        self.goal = None
        pass

    def isValid(self, state):
        return self._isValid(end_state=state)

    def sample_state(self, *args, **kwargs):
        return self._sample_state(*args, **kwargs)


class Node:
    def __init__(self, n):
        self.coord = np.array(n)
        self.parent = None


class RRTPlanner:
    def __init__(self, ri, obstacles, iter_num, min_distances=None, min_distances_start_goal=None):
        # 操作 Agent
        self.ri = ri
        
        # 参数
        self.iter_max = iter_num
        self.ndof = len(self.ri.joints)
        self.obstacles = obstacles or []
        self.lower, self.upper = ri.get_bounds()
        self.lower = np.asarray(self.lower)
        self.upper = np.asarray(self.upper)
        self.min_distances = min_distances or {}
        self.min_distances_start_goal = min_distances_start_goal or {}

        # 构造 OMPL PathSimplifier 所需要的变量
        bounds = ob.RealVectorBounds(self.ndof)
        for i in range(self.ndof):
            bounds.setLow(i, self.lower[i])
            bounds.setHigh(i, self.upper[i])
        self.space = ob.RealVectorStateSpace(self.ndof)
        self.space.setBounds(bounds)
        self.si = ob.SpaceInformation(self.space)
        self.validityChecker = pbValidityChecker(self.si, partial(self.isValid, start_state=None), self.sample_state)
        self.si.setStateValidityChecker(self.validityChecker)
        self.si.setup()
        pass

    ############################# Planning ############################# 
    def sample_state(self):
        raise BaseException("Not Implementation!")        

    def plan(self):
        raise BaseException("Not Implementation!")        

    def path_simplify(self, path):
        og_path = og.PathGeometric(self.si)
        for q in path:
            state = self.si.allocState()
            state[0] = q[0]
            state[1] = q[1]
            state[2] = q[2]
            state[3] = q[3]
            state[4] = q[4]
            state[5] = q[5]
            state[6] = q[6]
            og_path.append(state)
        simplifier = og.PathSimplifier(self.si)
        simplifier.simplifyMax(og_path)

        state_count = og_path.getStateCount()
        path = np.zeros((state_count, self.ndof), dtype=float)
        for i_state in range(state_count):
            state = og_path.getState(i_state)
            path_i = np.zeros((self.ndof), dtype=float)
            for i_dof in range(self.ndof):
                path_i[i_dof] = state[i_dof]
            path[i_state] = path_i
        return path

    ############################# Agent ############################# 
    def isValid(self, start_state, end_state, *args, **kwargs):
        state = end_state
        if not self.check_joint_limits(state):
            return False

        j = [state[i] for i in range(self.ndof)]

        if self.min_distances_start_goal:
            if self.start is not None and np.allclose(j, self.start):
                min_distances = self.min_distances_start_goal
            elif self.goal is not None and np.allclose(j, self.goal):
                min_distances = self.min_distances_start_goal
            else:
                min_distances = self.min_distances
        else:
            min_distances = self.min_distances

        with pp.WorldSaver():
            self.ri.setj(j)

            is_valid = self.check_self_collision(
                min_distances=min_distances
            ) and self.check_collision(
                self.obstacles, min_distances=min_distances
            )

        return is_valid

    def check_self_collision(self, min_distances=None):
        min_distances = min_distances or {}

        is_colliding = False

        links = pp.get_links(self.ri.robot)
        for link_a, link_b in itertools.combinations(links, 2):
            link_name_a = pp.get_link_name(self.ri.robot, link_a)
            link_name_b = pp.get_link_name(self.ri.robot, link_b)

            assert link_b > link_a
            if link_b - link_a == 1:
                continue

            distance = 0

            # XXX: specific configurations for panda_drl.urdf
            # panda_link5: front arm
            # panda_link6: arm head
            # panda_link7: wrist
            # panda_link8: palm tip
            if (
                link_name_a == "panda_link7"
                and link_name_b == "panda_suction_gripper"
            ):
                continue
            if (
                link_name_a == "panda_link5"
                and link_name_b == "panda_suction_gripper"
            ):
                continue

            # XXX: specific configurations for panda_suction.urdf
            # panda_link7: wrist
            if link_name_a == "panda_link7" and link_name_b == "baseLink":
                continue

            # XXX: specific configurations for panda_suction.urdf
            # panda_link7: wrist
            if (
                link_name_a == "panda_link7"
                and link_name_b == "panda_hand"
            ):
                continue
            if (
                link_name_a == "panda_hand"
                and link_name_b == "panda_finger_link1"
            ):
                continue
            if (
                link_name_a == "panda_hand"
                and link_name_b == "panda_finger_link2"
            ):
                continue

            is_colliding_i = (
                len(
                    p.getClosestPoints(
                        bodyA=self.ri.robot,
                        linkIndexA=link_a,
                        bodyB=self.ri.robot,
                        linkIndexB=link_b,
                        distance=distance,
                    )
                )
                > 0
            )
            # if is_colliding_i:
            #     from loguru import logger
            #
            #     logger.warning(
            #         f"{link_a}:{link_name_a} and {link_b}:{link_name_b} "
            #         f"is self-colliding with distance of {distance}"
            #     )
            is_colliding |= is_colliding_i

        for attachment in self.ri.attachments:
            assert attachment.parent == self.ri.robot
            min_distance = min_distances.get((attachment.child, -1), 0)
            for link in links:
                if link == attachment.parent_link:
                    continue
                is_colliding |= (
                    len(
                        p.getClosestPoints(
                            bodyA=attachment.child,
                            linkIndexA=-1,
                            bodyB=self.ri.robot,
                            linkIndexB=link,
                            distance=min_distance,
                        )
                    )
                    > 0
                )

        return not is_colliding

    def check_collision(self, ids_to_check, min_distances=None):
        min_distances = min_distances or {}

        if len(ids_to_check) == 0:
            return True

        is_colliding = False

        # 利用Pybullet检查每个Link与最后一个障碍物的最小距离？
        # 为什么只检查最后一个物体？看下面，用递归来逐个检查。
        for link in pp.get_links(self.ri.robot):
            min_distance = min_distances.get((self.ri.robot, link), 0)
            is_colliding |= (
                len(
                    p.getClosestPoints(
                        bodyA=self.ri.robot,
                        linkIndexA=link,
                        bodyB=ids_to_check[-1],
                        linkIndexB=-1,
                        distance=min_distance,
                    )
                )
                > 0
            )

        for attachment in self.ri.attachments:
            min_distance = min_distances.get((attachment.child, -1), 0)
            is_colliding |= (
                len(
                    p.getClosestPoints(
                        attachment.child,
                        ids_to_check[-1],
                        distance=min_distance,
                    )
                )
                > 0
            )

        if is_colliding:
            return False
        else:
            # 这里用递归来检查每一个
            return self.check_collision(
                ids_to_check[0:-1], min_distances=min_distances
            )

    def check_joint_limits(self, state):
        for i in range(self.ndof):
            if state[i] > self.upper[i] or state[i] < self.lower[i]:
                return False
        return True


class RRTConnectPlanner(RRTPlanner):
    def __init__(self, ri, obstacles, step_len, goal_sample_rate, iter_num, min_distances=None, min_distances_start_goal=None):
        super().__init__(ri, obstacles, iter_num, min_distances, min_distances_start_goal)
        self.goal_sample_rate = goal_sample_rate
        self.step_len = step_len

    def sample_state(self, *args, **kwargs):
        # 一定概率直接启发式地返回目标点
        if np.random.random() < self.goal_sample_rate:
            return copy.deepcopy(self.s_goal)

        # 完全随机
        # @note 改为优先在小范围内随机
        q = (
            np.random.random(self.ndof) * (self.upper - self.lower)
            + self.lower
        )
        if self.isValid(q, q):
            return Node(q)
        else:
            return self.sample_state()

    def plan(self, start_j, goal_j):
        path = self.ori_plan(start_j, goal_j)
        if path is None:
            return None
        path = self.path_simplify(path)
        return path

    def ori_plan(self, start_j, goal_j):
        self.s_start = Node(start_j)
        self.s_goal = Node(goal_j)

        self.V1 = [self.s_start]
        self.V2 = [self.s_goal]
        for i in range(self.iter_max):
            node_rand = self.sample_state()
            node_near = self.nearest_neighbor(self.V1, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if node_new and self.isValid(node_near.coord, node_new.coord):
                self.V1.append(node_new)
                node_near_prim = self.nearest_neighbor(self.V2, node_new)
                node_new_prim = self.new_state(node_near_prim, node_new)

                if node_new_prim and not self.isValid(node_new_prim.coord, node_near_prim.coord):
                    self.V2.append(node_new_prim)

                    while True:
                        node_new_prim2 = self.new_state(node_new_prim, node_new)
                        if node_new_prim2 and not self.is_collision(node_new_prim2, node_new_prim):
                            self.V2.append(node_new_prim2)
                            node_new_prim = self.change_node(node_new_prim, node_new_prim2)
                        else:
                            break

                        if self.is_node_same(node_new_prim, node_new):
                            break

                if self.is_node_same(node_new_prim, node_new):
                    return self.extract_path(node_new, node_new_prim)

            if len(self.V2) < len(self.V1):
                list_mid = self.V2
                self.V2 = self.V1
                self.V1 = list_mid

        return None

    def nearest_neighbor(self, node_list, n):
        return node_list[int(np.argmin([np.linalg.norm(nd.coord - n.coord) for nd in node_list]))]

    def new_state(self, node_start, node_end):
        v = node_end.coord - node_start.coord
        dist = np.linalg.norm(v)
        unit_v = v / (dist+1e-8)  # @note avoid zero division
        dist = min(self.step_len, dist)
        new_coord = node_start.coord + dist * unit_v
        node_new = Node(new_coord)
        node_new.parent = node_start
        return node_new

    @staticmethod
    def extract_path(node_new, node_new_prim):
        path1 = [node_new.coord]
        node_now = node_new
        while node_now.parent is not None:
            node_now = node_now.parent
            path1.append(node_now.coord)
        path2 = [node_new_prim.coord]
        node_now = node_new_prim
        while node_now.parent is not None:
            node_now = node_now.parent
            path2.append(node_now.coord)
        return np.concatenate((np.array(path1[::-1]), np.array(path2)), axis=0)

    @staticmethod
    def change_node(n1, n2):
        new_node = Node(n2.coord)
        new_node.parent = n1
        return new_node

    @staticmethod
    def is_node_same(node_new_prim, node_new):
        if ((node_new_prim.coord - node_new.coord) < 1e-4).all():
            return True

        return False


class INode:
    def __init__(self, n):
        self.coord = np.array(n)
        self.parent = None

        # distance
        self.self_collision_closest_distance = None
        self.collision_closest_distance = None
        self.limit_distance = None

        # score
        self.goal_score = None
        self.obstacle_score = None
        self.score = None

        # 
        self.joint_links_pose = None
        self.ee_pose = None

        self.failure_cnt = 0

    def get_pose(self, ri, ready=False):
        if self.joint_links_pose is None:
            if ready == False:
                with pp.WorldSaver():
                    ri.setj(self.coord)
                    joint_links_T, ee_T = ri.get_whole_pose()
                    self.joint_links_pose = joint_links_T
                    self.ee_pose = ee_T
            else:
                joint_links_T, ee_T = ri.get_whole_pose()
                self.joint_links_pose = joint_links_T
                self.ee_pose = ee_T
        else:
            # print("不重复初始化")
            pass
        pass

    
class IntegratedRRTPlanner(RRTPlanner):
    def __init__(self, ri, obstacles, failure_max_cnt, step_len, iter_num, min_distances=None, min_distances_start_goal=None):
        super().__init__(ri, obstacles, iter_num, min_distances, min_distances_start_goal)

        # 参数
        self.failure_max_cnt = failure_max_cnt
        pass

    def plan(self, start_j, goal_pose_T):
        path = self.ori_plan(start_j, goal_pose_T)
        if path is None:
            return None
        # path = self.path_simplify(path)
        return path

    def ori_plan(self, start_j, goal_pose_T):
        self.s_start = INode(start_j)
        self.goal_pose_T = goal_pose_T
        self.T = [self.s_start]
        self.ranking_nodes = [self.s_start]
        self.node_init(self.s_start)

        if not self.isValid(self.s_start):
            print("起点不可行")
        self.node_init(self.s_start)
        
        for i in range(self.iter_max):
            ############ EXTEND
            # node_rand = self.sample_state()
            # node_near = self.nearest_neighbor(self.T, node_rand)
            # node_new = self.new_state(node_near, node_rand)

            # if node_new and not self.is_collision(node_near, node_new):
            #     self.T.append(node_new)
            #     dist = self.node_distance(self.s_goal, node_new)
            #     if dist < self.step_len:  # 小于步长，直接连接到终点
            #         node_new = self.new_state(node_new, self.s_goal)
            #         return self.extract_path(node_new)
            ############ EXTEND

            ############ EXTEND_HEURISTIC
            node_rand = self.sample_state()
            node_near = self.ranking_nodes[-1] 
            node_new = self.new_state(node_near, node_rand)
            self.node_init(node_new)

            node_near_goal_score = self.goal_score(node_near)
            node_new_goal_score = self.goal_score(node_new)
            if node_new and self.isValid(node_new) and (node_new_goal_score > node_near_goal_score):
                self.add_node(node_new)
                # return
                print("node_new_goal_score: ", node_new_goal_score)

                # 判断是否到终点
                trans, angle = self.goal_dist(node_new)
                # if trans < 0.05 and angle < 20:
                if trans < 0.02:
                    return self.extract_path(node_new)
            else:
                node_near.failure_cnt += 1
                if node_near.failure_cnt > self.failure_max_cnt:
                    self.ranking_nodes.remove(node_near)
                    node_near.parent.failure_cnt = self.failure_max_cnt
            
            ############ EXTEND_HEURISTIC
        return None

    ############################# Planning ############################# 
    def sample_state(self, *args, **kwargs):
        # TODO 添加 non-uniform sample

        # sample_range = len(self.upper) * [np.max(self.upper - self.lower)]
        # delta = np.random.random(self.ndof) * sample_range
        # q = (delta + self.lower)

        q = (
            np.random.random(self.ndof) * (self.upper - self.lower)
            + self.lower
        )
        n = INode(q)
        return n
        # if self.isValid(n):
        #     return n
        # else:
        #     return self.sample_state()

    @staticmethod
    def insort_with_key(data, item, key_func=None):
        if key_func is None:
            # 如果没有提供 key 函数，使用默认的 bisect_left
            position = bisect.bisect_left(data, item)
        else:
            # 使用 key 函数处理 item 和 data 中的每个元素
            item_key = key_func(item)
            position = bisect.bisect_left([key_func(x) for x in data], item_key)

        # 插入元素
        data.insert(position, item)

    def add_node(self, n):
        self.T.append(n)
        # 高效地插入新节点，保持 ranking 的顺序
        self.insort_with_key(self.ranking_nodes, n, key_func=lambda x: x.score)
        pass

    def node_init(self, n, goal_weight=1, obstacle_weight=1):
        n.get_pose(self.ri, ready=False)

        # 根据与障碍物的距离和、终点的距离计算某个节点的得分
        # 根据得分维持 ranking 列表的有序性
        n.goal_score = self.goal_score(n)
        n.obstacle_score = self.obstacle_score(n)
        n.score = goal_weight * n.goal_score + obstacle_weight * n.obstacle_score
        return n

    def goal_dist(self, n):
        trans = np.linalg.norm(n.ee_pose[:3, 3] - self.goal_pose_T[:3, 3])
        # angle = angle_between_z_axis(n.ee_pose, self.goal_pose_T) / np.pi * 180
        angle = np.abs(np.arccos(-n.ee_pose[2, 2])) / np.pi * 180
        return trans, angle

    def goal_score(self, n):
        # TODO
        trans, angle = self.goal_dist(n)
        angle = angle / 180 * np.pi
        if trans > 0.2:
            score = -(1 * np.exp(trans) + (1/4) * np.exp(np.pi))
        else:
            score = -(1 * np.exp(trans) + (1/4) * np.exp(angle))

        # score = -(1 * np.exp(trans) + (1/4) * np.exp(angle))
        return score

    def obstacle_score(self, n):
        # TODO
        return 0

    def node_distance(self, n1, n2):
        return np.linalg.norm(n1.coord - n2.coord)

    def nearest_neighbor(self, node_list, n):
        return node_list[int(np.argmin([np.linalg.norm(nd.coord - n.coord) for nd in node_list]))]

    @staticmethod
    def cal_scale(q_d, closest_distance):
        # From: Geraerts, Roland, and Mark H. Overmars. "On improving the clearance for robots in high-dimensional configuration spaces." 2005 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2005.
        # q_d: normalized direction joint between cur_node and rand_new_node
        # weight from its paper: 6, 6, 6, 2, 2, 2, 2
        # min scale: 0.1
        weights = np.array([6, 6, 6, 2, 2, 2, 2])
        min_scale = 0.01
        scale = closest_distance / np.linalg.norm(weights * q_d)
        return max(min_scale, scale)

    def new_state(self, node_start, node_end):
        v = node_end.coord - node_start.coord
        dist = np.linalg.norm(v)
        unit_v = v / (dist+1e-8)  # @note avoid zero division

        # min_closest_distance = min(node_start.self_collision_closest_distance, node_start.collision_closest_distance)
        min_closest_distance = node_start.collision_closest_distance
        scale = self.cal_scale(unit_v, min_closest_distance)

        new_coord = node_start.coord + scale * unit_v
        node_new = INode(new_coord)
        node_new.parent = node_start
        return node_new

    @staticmethod
    def extract_path(node_new):
        path1 = [node_new.coord]
        node_now = node_new
        while node_now.parent is not None:
            node_now = node_now.parent
            path1.append(node_now.coord)
        return np.array(path1[::-1])

    @staticmethod
    def change_node(n1, n2):
        new_node = INode(n2.coord)
        new_node.parent = n1
        return new_node

    @staticmethod
    def is_node_same(node_new_prim, node_new):
        if ((node_new_prim.coord - node_new.coord) < 1e-4).all():
            return True

        return False

    ############################# Agent ############################# 
    def isValid(self, n):
        state = n.coord

        limit_is_valid, limit_distances = self.check_joint_limits(state)
        if not limit_is_valid:
            return False

        j = [state[i] for i in range(self.ndof)]

        with pp.WorldSaver():
            self.ri.setj(j)

            n.get_pose(self.ri, ready=True)

            self_collision_is_valid, self_collision_closest_distance = self.check_self_collision() 
            collision_is_valid, collision_closest_distance = self.check_collision(self.obstacles)
            pass

        if (not self_collision_is_valid) or (not collision_is_valid):
            return False 

        n.limit_distance = np.max(limit_distances)
        n.collision_closest_distance = collision_closest_distance
        n.self_collision_closest_distance = self_collision_closest_distance

        return True

    def check_self_collision(self, min_distances=None):
        # 添加返回距离
        min_distances = min_distances or {}

        is_colliding = False

        links = pp.get_links(self.ri.robot)
        distance = 0.1  # TODO 可能影响效率，需要调整
        closest_distance = distance
        for link_a, link_b in itertools.combinations(links, 2):
            link_name_a = pp.get_link_name(self.ri.robot, link_a)
            link_name_b = pp.get_link_name(self.ri.robot, link_b)

            assert link_b > link_a
            if link_b - link_a == 1:
                continue

            # XXX: specific configurations for panda_drl.urdf
            # panda_link5: front arm
            # panda_link6: arm head
            # panda_link7: wrist
            # panda_link8: palm tip
            if (
                link_name_a == "panda_link7"
                and link_name_b == "panda_suction_gripper"
            ):
                continue
            if (
                link_name_a == "panda_link5"
                and link_name_b == "panda_suction_gripper"
            ):
                continue

            # XXX: specific configurations for panda_suction.urdf
            # panda_link7: wrist
            if link_name_a == "panda_link7" and link_name_b == "baseLink":
                continue

            # XXX: specific configurations for panda_suction.urdf
            # panda_link7: wrist
            if (
                link_name_a == "panda_link7"
                and link_name_b == "panda_hand"
            ):
                continue
            if (
                link_name_a == "panda_hand"
                and link_name_b == "panda_finger_link1"
            ):
                continue
            if (
                link_name_a == "panda_hand"
                and link_name_b == "panda_finger_link2"
            ):
                continue
            
            closest_points_info = p.getClosestPoints(
                                        bodyA=self.ri.robot,
                                        linkIndexA=link_a,
                                        bodyB=self.ri.robot,
                                        linkIndexB=link_b,
                                        distance=distance,
                                    )
            if len(closest_points_info) == 0:
                is_colliding_i = False
            else:
                closest_distances = [i[8] for i in closest_points_info]
                closest_distances = sorted(closest_distances)
                if closest_distances[0] < closest_distance:
                    closest_distance = closest_distances[0]
                is_colliding_i = (closest_distances[0] < 0)
            is_colliding |= is_colliding_i

        return (not is_colliding), closest_distance

    def check_collision(self, ids_to_check):
        # 添加返回距离

        if len(ids_to_check) == 0:
            return True, np.inf

        is_colliding = False
        # 只考虑小于 distance 的惩罚
        distance = 0.1  # TODO 可能影响效率，需要调整
        closest_distance = distance

        # 利用Pybullet检查每个Link与最后一个障碍物的最小距离？
        # 为什么只检查最后一个物体？看下面，用递归来逐个检查。
        for link in pp.get_links(self.ri.robot):
            link_name = pp.get_joint_name(self.ri.robot, link).decode()
            closest_points_info = p.getClosestPoints(
                                    bodyA=self.ri.robot,
                                    linkIndexA=link,
                                    bodyB=ids_to_check[-1],
                                    linkIndexB=-1,
                                    distance=distance,
                                )
            if len(closest_points_info) != 0:
                closest_distances = [i[8] for i in closest_points_info]
                closest_distances = sorted(closest_distances)
                if link_name != "panda_joint1" and link_name != "panda_joint2":
                    if closest_distances[0] < closest_distance:
                        closest_distance = closest_distances[0]
                is_colliding |= (closest_distances[0] < 0)

        if is_colliding:
            return False, closest_distance
        else:
            # 这里用递归来检查每一个
            is_colliding, sub_closest_distance = self.check_collision(ids_to_check[0:-1])
            if sub_closest_distance < closest_distance:
                closest_distance = sub_closest_distance
            return is_colliding, closest_distance

    def check_joint_limits(self, state):
        limit_distances = [None] * self.ndof
        for i in range(self.ndof):

            upper_distance = self.upper[i] - state[i]
            lower_distance = state[i] - self.lower[i]
            limit_distances[i] = max(upper_distance, lower_distance)

            if state[i] > self.upper[i] or state[i] < self.lower[i]:
                return False, limit_distances

        return True, limit_distances