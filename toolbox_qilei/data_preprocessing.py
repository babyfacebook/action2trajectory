__author__ = 'qilei'
import os
import re
import numpy as np
def get_all_files(root_dir, pattern = None):
    # the list which srore the filtered files
    f_names = []
    for dir_path, dir_names, file_names in os.walk(root_dir):
        if pattern is not None:
            for f in file_names:
                if (re.search(pattern, f)):
                    f_names.append(dir_path + '/' + f)
        else:
            for f in file_names:
                    f_names.append(dir_path + '/' + f)
    return f_names
def get_all_trajetories(positions, L_trajectory, time_scale = 1):
    time_step = 0
    # the variable trajectories stores the information of trajectories
    trajectories = []
    for i in range(time_scale):
        time_step = time_step + 1
        for j in range((len(positions) - 1) - ((L_trajectory)*time_step) + 1):
            q1 = [(positions[1+k*time_step] - positions[k*time_step]) for k in range(L_trajectory)]
            q1 = np.array(q1).transpose()[:-1].transpose()
            s1 = [label_the_trajectory(q1[k+1], [1, 3, 5, 7]) for k in range(L_trajectory - 1 )]
            q1 = q1[:-1, :]
            s1 = np.array(s1)
            q1 = np.concatenate((q1, s1), axis=1)
            trajectories.append(q1)
    return trajectories

def label_the_trajectory(motion_direction, threshold):
    '''
    This function is designed to map a three-demension vector (motion direction) to a 64-dimension zero-one vector (4*4*4)
    The reason why we do it is to convert the prediction problem to classification problem
    :param motion_direction: three-dimension vector which represents the directions of motion
    :param threshold: threshold which determines the motion speed( motionless, slow, normal, or fast)
    :return: motion_speed, the speed of motion
    '''
    motion_threshold = [0, 0, 0]
    speed = np.zeros([1,])
    motion_direction = map(abs, motion_direction)
    if motion_direction[0] <= threshold[0]:
        motion_threshold[0] = 0
    elif motion_direction[0] > threshold[0] and motion_direction[0] <= threshold[1]:
        motion_threshold[0] = 1
    elif motion_direction[0] > threshold[1] and motion_direction[0] <= threshold[2]:
        motion_threshold[0] = 2
    elif motion_direction[0] > threshold[2] and motion_direction[0] <= threshold[3]:
        motion_threshold[0] = 3

    if motion_direction[1] <= threshold[0]:
        motion_threshold[1] = 0
    elif motion_direction[1] > threshold[0] and motion_direction[1] <= threshold[1]:
        motion_threshold[1] = 1
    elif motion_direction[1] > threshold[1] and motion_direction[1] <= threshold[2]:
        motion_threshold[1] = 2
    elif motion_direction[1] > threshold[2] and motion_direction[1] <= threshold[3]:
        motion_threshold[1] = 3

    if motion_direction[2] <= threshold[0]:
        motion_threshold[2] = 0
    elif motion_direction[2] > threshold[0] and motion_direction[2] <= threshold[1]:
        motion_threshold[2] = 1
    elif motion_direction[2] > threshold[1] and motion_direction[2] <= threshold[2]:
        motion_threshold[2] = 2
    elif motion_direction[2] > threshold[2] and motion_direction[2] <= threshold[3]:
        motion_threshold[2] = 3

    speed[0] = motion_threshold[0]   + motion_threshold[1] * 4 + motion_threshold[2] * 16

    #motion_speed[speed] = 1
    return speed



