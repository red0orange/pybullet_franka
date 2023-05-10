from scipy.spatial.transform import Rotation as R
import numpy as np
import tf


def angle_between_z_axis(T1, T2):
    # 提取旋转矩阵
    R1 = T1[:3, :3]
    R2 = T2[:3, :3]
    
    # 计算相对旋转矩阵
    R_relative = np.dot(R1.T, R2)
    
    # 从相对旋转矩阵中提取z轴夹角（单位：弧度）
    angle_z = np.arctan2(R_relative[1, 0], R_relative[0, 0])

    return abs(angle_z)


def pybullet2sevenDof(pose):
    (position, quaternion) = pose
    return [*position, *quaternion]


def sevenDof2pybullet(pose):
    return np.hsplit(pose, [3])


def T2sevendof(T):
    translation_vector = T[:3, 3]
    quat = tf.transformations.quaternion_from_matrix(T)
    return [*translation_vector, *quat]


def T2pybullet(T):
    translation_vector = T[:3, 3]
    quat = tf.transformations.quaternion_from_matrix(T)
    return [translation_vector, quat]


def sevenDof2T(pose):
    (position, quaternion) = pose[:3], pose[3:]
    homo_T = np.identity(4, dtype=np.float32)
    homo_T[:3, -1] = position
    homo_T[:3, :3] = tf.transformations.quaternion_matrix(quaternion)[:3, :3]
    return homo_T


def pybullet2T(pose):
    (position, quaternion) = pose
    homo_T = np.identity(4, dtype=np.float32)
    homo_T[:3, -1] = position
    homo_T[:3, :3] = tf.transformations.quaternion_matrix(quaternion)[:3, :3]
    return homo_T
