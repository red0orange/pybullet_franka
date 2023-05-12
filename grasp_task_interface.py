import time
import IPython

import rospy
import numpy as np

from base_task_interface import BaseTaskInterface
import geometry


class GraspTaskInterface(BaseTaskInterface):
    def __init__(self, real):
        super().__init__(real)

        pass

    def test_real_pick(self):
        # 回归 Home pose
        self.reset_pose()
        self.stop_grasp()

        # 设置抓取 Pose
        eye = np.array([0.4, 0, 0.5])
        target = eye + np.array([0, 0, -0.1])
        c = geometry.Coordinate.from_matrix(
            geometry.look_at(eye, target, None)
        )
        goal_pose = c.pose
        # 求解 IK
        target_j = self._env.pi.solve_ik(
            goal_pose,
            move_target=self._env.pi.robot_model.panda_hand,
            n_init=100,
            thre=0.05,
            rthre=np.deg2rad(5),
            # obstacles=self.object_ids,
            validate=True,
        )
        # Planning
        target_js = self._env.pi.planj(
            target_j, 
            # obstacles=self.object_ids
        )
        if target_js is None:
            print("target_js is None")
            return
        # Control
        self.movejs(target_js, time_scale=5, retry=True)
        self.start_grasp()
        time.sleep(2)
        self.stop_grasp()

        # 回归 Home pose
        self.reset_pose()
        pass


def main():
    rospy.init_node("grasp_task_interface")
    
    # real = False
    real = True
    task_interface = GraspTaskInterface(real)

    time.sleep(2)
    task_interface.test_real_pick()

    self = task_interface
    IPython.embed(header="base_task_interface")
    pass

    
if __name__ == '__main__':
    main()
    pass