import time
import IPython

import rospy

from base_task_interface import BaseTaskInterface


class GraspTaskInterface(BaseTaskInterface):
    def __init__(self, real):
        super().__init__(real)

        pass

    def test_real_pick(self):
        # 回归 Home pose
        self.reset_pose()

        # 设置抓取 Pose
        eye = [0.4, 0, 0.5]
        target = eye + [0, 0, -0.1]
        c = geometry.Coordinate.from_matrix(
            geometry.look_at(eye, target, None)
        )
        goal_pose = c.pose
        # 求解 IK
        joint = self._env.pi.solve_ik(
            goal_pose,
            move_target=self._env.pi.robot_model.tipLink,
            n_init=100,
            thre=0.05,
            rthre=np.deg2rad(5),
            obstacles=self.object_ids,
            validate=True,
        )
        # Planning
        target_js = self._env.pi.planj(target_j, obstacles=self.object_ids)
        if target_js is None:
            print("target_js is None")
            return
        # Control
        self.movejs(target_js, time_scale=5, retry=True)
        self.start_grasp()
        time.sleep(2)

        # 回归 Home pose
        self.reset_pose()
        pass


def main():
    rospy.init_node("grasp_task_interface")
    
    real = False
    task_interface = GraspTaskInterface(real)

    time.sleep(2)
    task_interface.test_real_pick()

    self = task_interface
    IPython.embed(header="base_task_interface")
    pass

    
if __name__ == '__main__':
    main()
    pass