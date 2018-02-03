# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import dbmoon as dm
from lms import LMS

def plt_result(weight, weight_list, mse):
    x_mse = list(range(len(mse)))
    y_mse = mse
    plt.figure(1)
    plt.scatter(x_mse, y_mse, s=3, c='red')

    plt.figure(2)
    plt.scatter(weight_list[:,0], weight_list[:,1], s=3, c='blue')
    plt.xlabel('w1')
    plt.ylabel('w2')

    x_values = np.linspace(-30, 30, 100)
    y_values = -weight[0]/weight[1]*x_values -weight[2]/weight[1]
    plt.figure(3)
    plt.plot(dbmoon_data[0:1000,0], dbmoon_data[0:1000,1], 'r*', dbmoon_data[1000:2000,0], dbmoon_data[1000:2000,1], 'b*')
    plt.plot(x_values, y_values, 'g')
    plt.show()

def test_lms(data_train):
    weight = np.zeros(3)
    mse = []
    my_lms = LMS(weight, mse)
    weight_list = my_lms.lms_train(data_train)
    plt_result(weight, weight_list, mse)




# 主程序
if __name__ == "__main__":

    # 得到测试数据和训练数据
    dbmoon_data = dm.dbmoon(1000, 1, 10, 6)
    index = np.random.randint(0, 2*1000, 1000)
    data_train = dbmoon_data.take(index, axis=0)

    test_lms(data_train)

