import bisect
import copy
import itertools
from functools import partial

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou
import pybullet as p
import pybullet_planning as pp


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
        # score
        self.goal_score = None
        self.obstacle_score = None
        self.score = None
        self.failure_cnt = 0

    
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
        self.T = [self.s_start]
        self.ranking_nodes = [self.s_start]
        self.node_score(self.s_start)
        
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

            node_near_goal_score = self.goal_score(node_near)
            node_new_goal_score = self.goal_score(node_new)
            if node_new and self.isValid(node_new.coord, node_new.coord) and (node_new_goal_score > node_near_goal_score):
                self.add_node(node_new)
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
        # 完全随机
        q = (
            np.random.random(self.ndof) * (self.upper - self.lower)
            + self.lower
        )
        if self.isValid(q, q):
            return INode(q)
        else:
            return self.sample_state()

    def add_node(self, n):
        self.T.append(n)
        self.node_score(n)
        # 高效地插入新节点，保持 ranking 的顺序
        bisect.insort(self.ranking_nodes, n, key=lambda x: x.score)
        pass

    def node_score(self, n, goal_weight=1, obstacle_weight=1):
        # 根据与障碍物的距离和、终点的距离计算某个节点的得分
        # 根据得分维持 ranking 列表的有序性
        goal_score = self.goal_score(n)
        obstacle_score = self.obstacle_score(n)
        weighted_sum = goal_weight * goal_score + obstacle_weight * obstacle_score

        n.goal_score = goal_score
        n.obstacle_score = obstacle_score
        n.score = weighted_sum        
        return weighted_sum, goal_score, obstacle_score

    def goal_score(self, n):
        # TODO
        # 1. 主要关注末端 pose，先计算末端 pose
        return -np.linalg.norm(n.coord - self.s_goal.coord)

    def obstacle_score(self, n):
        # TODO
        return 0

    def node_distance(self, n1, n2):
        return np.linalg.norm(n1.coord - n2.coord)

    def nearest_neighbor(self, node_list, n):
        return node_list[int(np.argmin([np.linalg.norm(nd.coord - n.coord) for nd in node_list]))]

    def new_state(self, node_start, node_end):
        v = node_end.coord - node_start.coord
        dist = np.linalg.norm(v)
        unit_v = v / (dist+1e-8)  # @note avoid zero division

        # TODO 改为 dynamic scale
        dist = min(self.step_len, dist)

        new_coord = node_start.coord + dist * unit_v
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