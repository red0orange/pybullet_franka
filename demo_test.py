import os

import rospy

from sensor_msgs.msg import PointCloud2, PointField, CameraInfo, Image


class DemoTest(object):
    def __init__(self):
        rospy.init_node("demo_test")

        self.buffer = []

        while True:
            user_input = input("Please input command: \n"
                            "1: back to home pose\n"
                            "2: back to grasp pose\n"
                            "5: quit\n")
            if user_input == "1":
                print("1")
            elif user_input == "2":
                print("2")
                fps = 100
                timer = rospy.Timer(rospy.Duration(nsecs=(1 / fps) * 1e9), self.read_one_msg) 
                stop_input = input("Press any key to stop")
                timer.shutdown()
                print(self.buffer)
                print(len(self.buffer))
            elif user_input == "5":
                break
            else:
                print("Invalid input: {}".format(user_input))
                continue
    
    def read_one_msg(self, event):
        camera_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        self.buffer.append(camera_info)
        pass
        


if __name__ == "__main__":
    import numpy as np 
    data = np.load("tmp.npy", allow_pickle=True)
    print(data)

    try:
        demo_test = DemoTest()
    except Exception():
        exit(100)
    pass