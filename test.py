__author__ = 'qilei'
import os
import cPickle
data_path = './data/'
train_data = cPickle.load(open(data_path + 'train_data_traj.save'))
m = 100