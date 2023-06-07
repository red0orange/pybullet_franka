from scipy.spatial.transform import Rotation as R
import numpy as np
import tf

# 目前实现三个格式的Pose之间的转换
# 格式 1：T
# 格式 2：[x, y, z, qx, qy, qz, qw]
# 格式 3: [[x, y, z], [qx, qy, qz, qw]]


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


def mean_Ts(Ts):
    quaternions = [R.from_matrix(T[:3, :3]).as_quat() for T in Ts]
    mean_quaternion = np.mean(quaternions, axis=0)
    mean_rotation = R.from_quat(mean_quaternion).as_matrix()

    translations = np.array([T[:3, 3] for T in Ts])
    mean_translation = np.mean(translations, axis=0)
    
    T_mean = np.eye(4)
    T_mean[:3, :3] = mean_rotation
    T_mean[:3, 3] = mean_translation
    return T_mean


def rotation_distance(T1, T2):
    q1 = R.from_matrix(T1[:3, :3]).as_quat()
    q2 = R.from_matrix(T2[:3, :3]).as_quat()

    # 计算四元数之间的角度差
    angle_diff = np.arccos(2 * np.dot(q1, q2)**2 - 1)
    return angle_diff


def T_distance(T1, T2):
    angle_diff = rotation_distance(T1, T2)
    trans_diff = np.linalg.norm(T1[:3, 3] - T2[:3, 3])
    return angle_diff, trans_diff


def ransac_average_Ts(Ts, angle_threshold=0.1, trans_threshold=0.05, max_iter_num=10, sample_ratio=0.5):
    best_T_mean = None
    best_ratio = 0
    
    for i in range(max_iter_num):
        indices = np.random.choice(list(range(len(Ts))), int(len(Ts) * sample_ratio))
        sample_Ts = [Ts[i] for i in indices]
        T_mean = mean_Ts(sample_Ts)

        good_num = 0
        for T in Ts:
            angle_diff, trans_diff = T_distance(T, T_mean)
            if angle_diff < angle_threshold and trans_diff < trans_threshold:
                good_num += 1

        ratio = float(good_num) / len(Ts)
        if ratio > best_ratio:
            best_T_mean = T_mean
            best_ratio = ratio

    print('best_ratio: ', best_ratio)
    return best_T_mean


def remove_xy_rotation(R):
    # Extract the rotation angle around the z-axis from R:
    theta = np.arctan2(R[1, 0], R[0, 0])
    
    # Construct a new rotation matrix around the z-axis with the same angle:
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])
    return Rz
