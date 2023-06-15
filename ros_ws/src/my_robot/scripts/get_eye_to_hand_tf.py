#! /usr/bin/env python
import numpy as np
import rospy
import tf
from tf import transformations as T


if __name__ == "__main__":
    rospy.init_node("get_fake_eye_hand_tf")

    tf_listener = tf.TransformListener()

    # Read From Realsense TF
    tf_listener.waitForTransform(
        'camera_link', 'camera_color_optical_frame', rospy.Time(), rospy.Duration(4.0))
    (camera_link_to_optical_frame_position, camera_link_to_optical_frame_quaternion) = tf_listener.lookupTransform(
        'camera_link', 'camera_color_optical_frame', rospy.Time(0))
    camera_link_to_optical_frame_homo_T = np.identity(4, dtype=np.float32)
    camera_link_to_optical_frame_homo_T[:3, -
                                        1] = camera_link_to_optical_frame_position
    camera_link_to_optical_frame_homo_T[:3, :3] = T.quaternion_matrix(
        camera_link_to_optical_frame_quaternion)[:3, :3]

    optical_frame_to_camera_link_homo_T = np.linalg.inv(
        camera_link_to_optical_frame_homo_T)

    # Calibration Result
    (hand_to_optical_frame_position, hand_to_optical_frame_quaternion) = (
        0.0285166,0.0361366,-0.0144262), (0.0108551,0.0126138,0.998982,0.041935)
    hand_to_optical_frame_homo_T = np.identity(4, dtype=np.float32)
    hand_to_optical_frame_homo_T[:3, -1] = hand_to_optical_frame_position
    hand_to_optical_frame_homo_T[:3, :3] = T.quaternion_matrix(
        hand_to_optical_frame_quaternion)[:3, :3]

    hand_to_camera_link_homo_T = hand_to_optical_frame_homo_T @ optical_frame_to_camera_link_homo_T
    hand_to_camera_link_position = hand_to_camera_link_homo_T[:3, -1]
    hand_to_camera_link_quaternion = T.quaternion_from_matrix(
        hand_to_camera_link_homo_T)

    print("{} {} {}      {} {} {} {}".format(
        *(list(hand_to_camera_link_position) + list(hand_to_camera_link_quaternion))))
    pass
