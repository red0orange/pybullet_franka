import IPython

import rospy

from base_task_interface import BaseTaskInterface


class GraspTaskInterface(BaseTaskInterface):
    def __init__(self, real):
        super().__init__(real)


def main():
    rospy.init_node("grasp_task_interface")
    
    real = False
    task_interface = GraspTaskInterface(real)

    IPython.embed(header="base_task_interface")
    pass

    
if __name__ == '__main__':
    main()
    pass