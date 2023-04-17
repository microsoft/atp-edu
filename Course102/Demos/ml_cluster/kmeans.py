# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:54:02 2022

@author: hongzhihou
"""
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

class Kmeans():
    
    def __init__(self, k=1):
        self.k = k

    def train(self, x, epochs=10):
        '''
        x: 样本集合， m个d维向量
        '''
        x_cluster_list = []
        kernels_list = []
        
        m = len(x)
        
        # 从x中随机抽出k个作为初始化核心
        # kernels全部的质心
        kernels = [x[i] for i in random.sample(range(0, m), self.k)]                
        
        # 样本的归属
        x_cluster = np.zeros(m, dtype=int)    
        
        # 循环
        for epoch in range(epochs):          
            cluster_x = [[] for i in range(self.k)]

            # 分配样本            
            for i in range(m):
               
                min_kernel_index = 0
                min_distance = math.inf

                for j in range(self.k):
                    
                    temp_distance = self.distance(x[i], kernels[j])
                    
                    if temp_distance < min_distance:
                        
                        min_distance = temp_distance
                        min_kernel_index = j

                x_cluster[i] = min_kernel_index
                cluster_x[min_kernel_index].append(i)            
            
            kernels_list.append(copy.deepcopy(kernels))
            x_cluster_list.append(copy.deepcopy(x_cluster))                        
            
            # 更新质心
            for i in range(self.k):
                # 计算重心，j为被归属到第i类的样本的index
                kernel = sum([x[j] for j in cluster_x[i]]) / len(cluster_x[i])
                kernels[i] = kernel
                  
        return x_cluster_list, kernels_list
                
    @staticmethod
    def distance(x1, x2):
        '''
        x1 和 x2 间的距离，这里采用欧拉距离
        即x1 - x2的二范数
        '''
        #return np.sqrt(np.sum(x1-x2))
        return np.linalg.norm(x1-x2)


# test
color = ['cyan', 'red', 'green', 'yellow', 'blue', 'black', 'magenta', 'white']

# 生成数据
   
x1 = 1 + 1*np.random.randn(50)
y1 = x1 + 1*np.random.randn(50)
x2 = 8 + np.random.randn(50)
y2 = x2 + 2*np.random.randn(50)
x3 = 15 + 2*np.random.randn(50)
y3 = x3 + 5*np.random.randn(50)

x = [x1, x2, x3]
y = [y1, y2, y3]

plt.figure(1)

for i in range(3):
    plt.scatter(x[i], y[i], c=color[i], marker='*')

plt.title("original")

plt.figure(2)    
x = np.array([x1, x2, x3]).reshape(1, -1)[0]
y = np.array([y1, y2, y3]).reshape(1, -1)[0]

plt.scatter(x, y, c='r', marker='*')
plt.title("train")

# 训练
k = 4
kmeans = Kmeans(k)

X = np.array([x ,y]).T

res_list, kernels_list = kmeans.train(X)

res = res_list[-1]
kernels = kernels_list[-1]

# 可视化
plt.figure(3)  

for epoch in range(len(res_list)):
    
    plt.clf()
    X_pred = [[] for i in range(k)]
    
    for i,value in enumerate(res_list[epoch]):
        X_pred[value].append(X[i])
          
    for i in range(k):
        plt.scatter(np.array(X_pred[i])[:, 0], np.array(X_pred[i])[:, 1], c=color[i], marker='*')    
        
    for kernel in kernels_list[epoch]:
        plt.scatter(kernel[0], kernel[1], c='black', marker='o')
        
    plt.title("predict k={}, epoch={}".format(k, epoch))  
    plt.pause(0.5)

