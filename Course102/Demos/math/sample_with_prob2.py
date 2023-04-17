# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 00:38:48 2022

@author: hongzhihou
"""

import random
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import scipy.stats as st


DELTA = 0.0001


# 概率分布函数cumulative distribution function
def my_cdf(x):
    '''
    if x <= 1:
        return 0
    elif x <= 3:
        return 1/4 *(x-1)**2
    else:
        return 1
    '''

    if x < 1:
        return 0
    if x < 3.5:
        return 0.3
    if x < 7:
        return 0.6
    else:
        return 1


# 概率密度函数probability density function
def my_pdf(x, cdf,delta=DELTA):
    
    return cdf(x) - cdf(x-delta)    


# 随机变量的抽样 - o(n)
def sample_with_distribute(cdf, start, end, delta=DELTA):
    
    x = random.uniform(0, 1)
    i = start
    delta = delta * (end - start)
    
    while x > cdf(i):
        i += delta

    return i


# 随机变量的抽样 - 二分查找 o(log())
def sample_with_distribute_binary_search(cdf, start=-10000, end=10000, delta=DELTA):
    
    x = random.uniform(0, 1)
    
    mid = (start + end)/2
    
    while start<end:

        if x > cdf(mid):
            start = mid+delta
            mid = (start + end)/2        
        
        elif x < cdf(mid):
            end = mid-delta
            mid = (start + end)/2       
        
        else:
            break   
    
    return mid


samples = 100
'''res = [sample_with_distribute(cdf=my_cdf, start=1, end=3) for i in range(samples)]

plt.figure(0)

plt.hist(res, 100, density=True)'''


# res = [sample_with_distribute_binary_search(cdf=st.norm.cdf) for i in range(samples)]
res = [sample_with_distribute_binary_search(cdf=my_cdf) for i in range(samples)]
print(sum(res)/samples)
plt.figure(1)

plt.hist(res, 100, density=False)

"""
# 求期望
k = -20
mean = 0
while k <= 20:
    mean += k*my_pdf(k, my_cdf)
    k += DELTA
print(mean)


# 中心极限定理 样本均值收敛到期望
def test2(samples, mean):
    
    res_sum = []
    epochs = 100
    
    for epoch in range(epochs):
        
        # res = [sample_with_prob(seq, probs) for i in range(samples)]
        res = [sample_with_distribute_binary_search(cdf=my_cdf) for i in range(samples)]
        res_sum.append(sum(res)/samples)
    
    plt.figure()
    plt.axvline(mean, color='r')    
    plt.hist(res_sum)
    plt.xlim((mean-1, mean+1))
    plt.title("expectation = {} \n num of samples = {}".format(mean, samples))
    plt.xlabel('mean of samples')
"""

# for i in [10,100,1000,10000]:
    #test2(i, mean)
    
