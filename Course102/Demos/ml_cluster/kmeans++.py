# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 10:27:51 2022

@author: hongzhihou
"""
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import copy


# kmeans++ algorithm
class Kmeanspp:
    
    def __init__(self, k=1):
        self.k = k
    
    def train(self, x, epochs=10):
        '''
        x: 样本集合， m个d维向量
        '''
        # 每次迭代x的label
        x_cluster_list = []
        # 每次迭代的kernels
        kernels_list = []
        
        m = len(x)
        
        # 从x中随机抽出k个作为初始化核心
        #kernels = self.initialize_kernels_randomly(x)

        # 根据距离初始化k个核心
        kernels = self.initialize_kernels(x)
        
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
    
    # 随机抽取k个点作为kernel初值
    def initialize_kernels_randomly(self, x):
        return [x[i] for i in random.sample(range(0, len(x)), self.k)]
    
    # kmeans++ algorithm 抽取k个点作为kernel初值
    def initialize_kernels(self, x):
        
        # 随机选取一个样本作为第一个kernel
        kernels = [x[i] for i in random.sample(range(0, len(x)), 1)]        
        min_distance_list = [math.inf for i in range(len(x))]

        # 依距离选取k-1个kernel
        for i in range(self.k-1):
            kernels.append(self.initialize_kernel(kernels[i], x, min_distance_list))
            
        return kernels
    
    # 在已抽出kernels的情况下再抽出一个kernel 加入到kernels中
    def initialize_kernel(self, pre_kernel, x, min_distance_list):
        
        # 根据上一次选出的kernel 更新min_distance_list
        for i in range(len(x)):
            
            temp_distance = self.distance(x[i], pre_kernel)
            
            if temp_distance < min_distance_list[i]:
                min_distance_list[i] = temp_distance
        
        # 计算概率
        d2 = [i*i for i in min_distance_list]
        probs = [i / sum(d2) for i in d2]        
        
        # 依概率抽取kernel
        return self.sample_with_prob(x, probs)
     

    @staticmethod
    def sample_with_prob(seq, probs):
        x = random.uniform(0, 1)
        
        curr_prob = 0.0
        
        for i, prob in enumerate(probs):

            curr_prob += prob

            if x < curr_prob:
                return seq[i]

        return seq[-1]
    
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
   
x1 = 1 + 1*np.random.randn(100)
y1 = x1 + 1*np.random.randn(100)
x2 = 8 + np.random.randn(100)
y2 = x2 + 2*np.random.randn(100)
x3 = 15 + 2*np.random.randn(100)
y3 = x3 + 5*np.random.randn(100)

x = [x1, x2, x3]
y = [y1, y2, y3]

#plt.figure(1)

#for i in range(3):
#    plt.scatter(x[i], y[i], c=color[i], marker='*')

#plt.title("original")

#plt.figure(2)    
x = np.array([x1, x2, x3]).reshape(1, -1)[0]
y = np.array([y1, y2, y3]).reshape(1, -1)[0]

#plt.scatter(x, y, c='r', marker='*')
#plt.title("train")

# 训练
k = 4
kmeanspp = Kmeanspp(k)

X = np.array([x, y]).T

res_list, kernels_list = kmeanspp.train(X)

res = res_list[-1]
kernels = kernels_list[-1]

# 可视化
plt.figure(3)  

for epoch in range(len(res_list)):
    
    plt.clf()
    X_pred = [[] for i in range(k)]
    
    for i, value in enumerate(res_list[epoch]):
        X_pred[value].append(X[i])
          
    for i in range(k):
        plt.scatter(np.array(X_pred[i])[:, 0], np.array(X_pred[i])[:, 1], c=color[i], marker='*')    
        
    for kernel in kernels_list[epoch]:
        plt.scatter(kernel[0], kernel[1], c='black', marker='o')
        
    plt.title("predict k={}, epoch={}".format(k, epoch))  
    plt.pause(0.5)
