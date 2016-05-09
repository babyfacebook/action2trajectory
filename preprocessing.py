import numpy as np
import os
import re
import toolbox_qilei.data_preprocessing
import cPickle
#parameters

#the lenth of trajectory
L_traj = 15
#the total number of trajectories
N_traj = 1000

#load data
data_path = './joints_data'
#the regular expression in pattern help choose the file which meet the requirement
pattern = re.compile('^a01_')
f_names = toolbox_qilei.data_preprocessing.get_all_files(data_path, pattern)

#extract the positions of a certain joint
# the list storing the positions
position_of_joint = []
pattern = re.compile('[0-9].[0-9]+e[\+\-][0-9]+')
zz =0
for f in f_names:
    print zz
    zz = zz + 1
    q1 = open(f).readlines()
    q2 = [q1[i*20] for i in range(len(q1)/20)]
    for i in range(len(q2)):
        q3 = re.findall(pattern, q2[i])
        q3 = [int(float(j)) for j in q3]
        q2[i] = np.array(q3)
    position_of_joint.append(q2)
a = 100
#extract trajectories
L_trajectory = 16
time_scale = 3
trajectories = []
for i in range(0, len(position_of_joint)*1/2, 1):
    positions = position_of_joint[i]
    print i
    trajectories.extend(toolbox_qilei.data_preprocessing.get_all_trajetories(positions, L_trajectory, time_scale))
f_traj = open('./data/train_data_traj.save', 'w+')
data_traj = cPickle.dump(trajectories, f_traj, protocol=0)
f_traj.close()

trajectories = []
for i in range(len(position_of_joint)*1/2, len(position_of_joint)*2/3, 1):
    positions = position_of_joint[i]
    print i
    trajectories.extend(toolbox_qilei.data_preprocessing.get_all_trajetories(positions, L_trajectory, time_scale))
f_traj = open('./data/valid_data_traj.save', 'w+')
data_traj = cPickle.dump(trajectories, f_traj, protocol=0)
f_traj.close()

trajectories = []
for i in range(len(position_of_joint)*2/3, len(position_of_joint), 1):
    positions = position_of_joint[i]
    print i
    trajectories.extend(toolbox_qilei.data_preprocessing.get_all_trajetories(positions, L_trajectory, time_scale))
f_traj = open('./data/test_data_traj.save', 'w+')
data_traj = cPickle.dump(trajectories, f_traj, protocol=0)
f_traj.close()


