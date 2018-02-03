# -*- coding: UTF-8 -*-
import numpy as np

class LMS():
	def __init__(self, weight=[0,0,0], mse=[]):
		self.weight = weight
		self.mse = mse
	
	# lms算法
	def lms_train(self, data_set, ita=0.0001, train_count=2):
		x_mat = np.concatenate((data_set[:,0:2], np.ones([data_set.shape[0], 1])), axis=1)
		weight_list = np.empty(0)
		for n in range(train_count):
			for x, data in zip(x_mat, data_set):
				weight_array = np.array([self.weight])
				if weight_list.shape[0]==0:
					weight_list = weight_array
				else:
					weight_list = np.concatenate((weight_list, weight_array), axis=0)
				
				e = data[2] - np.inner(x, self.weight)
				self.weight += ita * e * x
				y_vec = np.dot(x_mat, self.weight.T)
				e_vec = data_set[:,2] - y_vec
				e_mean = np.sum(e_vec**2) / e_vec.shape[0]
				self.mse.append(e_mean)
		return weight_list
