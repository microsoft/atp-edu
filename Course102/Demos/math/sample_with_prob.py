# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:58:49 2022

@author: hongzhihou
"""

import random
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


# 离散型随机变量的抽样
def sample_with_prob(seq, probs):
    
    x = random.uniform(0, 1)

    curr_prob = 0.0
    
    for i, prob in enumerate(probs):
        
        curr_prob += prob
        
        if x <= curr_prob:
            return seq[i]


# 大数定律 频率收敛到概率       
def test1(samples=1000):

    seq = [1, 2, 3, 4, 5, 6]
    probs = [0.2, 0.1, 0.3, 0.3, 0.05, 0.05]
    
    res = []

    for i in range(samples):
        
        res.append(sample_with_prob(seq, probs))
    
    counter = Counter()
    counter.update(res)
    
    freq = [counter[item]/samples for item in seq]
    
    width = 0.4
    x = np.arange(len(seq))
    
    plt.figure()
    plt.bar(x-width/2, freq, width)
    plt.bar(x+width/2, probs, width, color='r')
    plt.xticks(range(len(seq)), labels=seq)
    plt.xlabel('category')
    #plt.ylabel('frequency')
    
    plt.legend(['freq', 'prob'])
    
    plt.ylim((0, max(probs)+0.1))
    plt.title('num of samples = {} \n'.format(samples))


# 中心极限定理 样本均值收敛到期望
def test2(samples=1000):

    seq = [1, 2, 3, 4, 5, 6]
    probs = [0.2, 0.1, 0.3, 0.3, 0.05, 0.05]
    
    res_sum = []
    epochs = 100
    
    for epoch in range(epochs):
        
        res = [sample_with_prob(seq, probs) for i in range(samples)]
        res_sum.append(sum(res)/samples)
    
    plt.figure()
    plt.hist(res_sum)
    plt.xlim((2, 4))
    mean = sum([se*pb for se, pb in zip(seq, probs)])
    plt.axvline(mean, color='r')
    plt.title("expectation = {} \n num of samples = {}".format(mean, samples))
    plt.xlabel('mean of samples')
    

'''for i in [10, 50, 100, 200, 500, 1000]:
    test2(i)'''


for i in [10, 100, 1000, 10000]:
    test1(i)