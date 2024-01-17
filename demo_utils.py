import re
import numpy as np
from scipy import interpolate

import rospy

from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


def joints_traj_to_robot_traj(joints_traj, velocity=None, duration=0.05, joint_names=None):
    velocity, acceleration = compute_velocity_acceleration(joints_traj, duration, velocity=velocity)
    velocity, acceleration = limit_velocity_acceleration(velocity, acceleration, max_velocity=1.08, max_acceleration=1.08)
    robot_trajectory = RobotTrajectory()

    for i, point in enumerate(joints_traj):
        # 创建一个轨迹点
        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = point
        trajectory_point.velocities = velocity[i]
        # trajectory_point.accelerations = acceleration[i]
        trajectory_point.time_from_start = rospy.Duration(nsecs=(i * duration * 1e9))
        # 将轨迹点添加到轨迹中
        robot_trajectory.joint_trajectory.points.append(trajectory_point)
        
    print(np.max(velocity))
    print(np.max(acceleration))
    if joint_names is None:
        joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"]
    robot_trajectory.joint_trajectory.joint_names = joint_names
    return robot_trajectory


def adjust_angles_for_continuity(angles):
    """
    调整角度值以保持连续性，特别是当跨越0弧度时。

    :param angles: 一维或二维的numpy数组，表示关节角度，单位为弧度。
    :return: 调整后连续的角度值。
    """
    adjusted_angles = np.unwrap(angles)  # np.unwrap对角度进行展开以保持连续性
    return adjusted_angles

def resample_trajectory(trajectory, num_points):
    """
    使用B样条曲线对轨迹进行时间基的重新采样，并处理周期性边界。

    :param trajectory: nx7 的 numpy 数组，表示机械臂的运动轨迹。
    :param num_points: 重采样后的点数。
    :return: 重采样后的轨迹。
    """
    # 提取原始轨迹的时间
    time_original = np.linspace(0, 1, len(trajectory))
    
    # 处理周期性边界问题
    adjusted_trajectory = np.apply_along_axis(adjust_angles_for_continuity, 0, trajectory)

    # 创建B样条曲线
    tck, u = interpolate.splprep(adjusted_trajectory.T, s=0)

    # 生成新的时间点并在这些点上评估B样条曲线
    time_new = np.linspace(0, 1, num_points)
    resampled_trajectory = np.array(interpolate.splev(time_new, tck)).T

    return resampled_trajectory

def compute_velocity_acceleration(resampled_trajectory, delta_t, velocity=None):
    """
    计算重新采样轨迹的速度和加速度。

    :param resampled_trajectory: 重新采样后的轨迹。
    :param delta_t: 重采样后每个点之间的时间间隔。
    :return: 速度和加速度数组。
    """
    # 速度：使用中心差分计算速度
    if velocity is None:
        velocity = (resampled_trajectory[2:] - resampled_trajectory[:-2]) / (2 * delta_t)

    # 加速度：使用中心差分计算加速度
    acceleration = (resampled_trajectory[2:] - 2 * resampled_trajectory[1:-1] + resampled_trajectory[:-2]) / (delta_t ** 2)

    # 对于第一个和最后一个点，我们不能使用中心差分
    # 可以选择使用前向/后向差分，或将它们设置为0或其他适当的值
    velocity = np.vstack((velocity[0], velocity, velocity[-1]))
    acceleration = np.vstack((acceleration[0], acceleration, acceleration[-1]))

    return velocity, acceleration

def limit_velocity_acceleration(velocity, acceleration, max_velocity, max_acceleration):
    """
    限制速度和加速度以确保它们不会超过机器人的最大速度和加速度。

    :param velocity: 速度数组
    :param acceleration: 加速度数组
    :param max_velocity: 允许的最大速度
    :param max_acceleration: 允许的最大加速度
    """
    # 计算速度和加速度的最大比例
    velocity_scale = max_velocity / np.max(np.abs(velocity))
    acceleration_scale = max_acceleration / np.max(np.abs(acceleration))

    # 选择最小的比例作为整体缩放因子
    scale = min(velocity_scale, acceleration_scale)

    # 如果比例小于1，则缩放速度和加速度
    if scale < 1:
        velocity *= scale
        acceleration *= scale

    return velocity, acceleration

def process_traj(trajectory, threshold):
    """
    减少机械臂运动轨迹中的停顿。

    :param trajectory: nx7 的 numpy 数组，表示机械臂的运动轨迹。
    :param threshold: 浮点数，用于确定何时认为机械臂停顿。
    :return: 减少停顿后的新轨迹。
    """
    # 初始化一个列表来存储减少停顿后的轨迹
    reduced_trajectory = []
    
    # 遍历轨迹中的每一帧
    for i in range(len(trajectory) - 1):
        # 计算当前帧与下一帧之间的差异
        frame_diff = np.abs(trajectory[i] - trajectory[i + 1])
        
        # 如果差异大于阈值，说明机械臂在移动，保留这一帧
        if np.sum(frame_diff) > threshold:
            reduced_trajectory.append(trajectory[i])
    
    # 添加最后一帧，因为它没有后续帧与之比较
    reduced_trajectory.append(trajectory[-1])
    
    return np.array(reduced_trajectory)

def is_valid_filename(input_str):
    """
    验证输入的字符串是否为有效的文件名。
    
    :param input_str: 用户的输入字符串。
    :return: 如果输入字符串是一个有效的文件名，则返回 True，否则返回 False。
    """
    # 文件名不能包含 \/:*?"<>| 等字符，且不能以空格开始或结束。
    if re.match("^[^.\\/:*?\"<>|\r\n]*[^ .\\/:*?\"<>|\r\n]$", input_str) is None:
        return False
    return True

def wait_for_valid_input(prompt, validation_func=None):
    """
    等待直到接收到有效的输入字符串。
    
    :param prompt: 提示用户输入的字符串。
    :param validation_func: 一个函数，用于验证输入的字符串。如果输入有效，则应该返回 True。
                            如果没有提供这个函数，则只检查输入是否为非空字符串。
    :return: 有效的输入字符串。
    """
    if validation_func is None:
        validation_func = is_valid_filename
    while True:
        user_input = input(prompt)

        if len(user_input) == 0:
            print("输入不能为空，请再次尝试。")
            continue
        
        # 检查输入是否有效
        if not validation_func(user_input):
            print("无效输入，请再次尝试。")
            continue

        break
        
    return user_input


def wait_for_desired_input(prompt, desired_inputs):
    while True:
        user_input = input(prompt)
        if len(user_input) == 0:
            print("Empty input")
            continue
        if user_input[-1] in desired_inputs:
            break
        else:
            print("Invalid input: {}".format(user_input))
    return user_input