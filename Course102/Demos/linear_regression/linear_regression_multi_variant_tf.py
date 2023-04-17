# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 17:17:46 2022

@author: hongzhihou
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


class MultiLinearRegressionTf:

    def __init__(self, Theta=[], n=1):
        """
        theta: 参数
        n: feature数

        y_hat = theta0 + theta1 * x1 + theta2 * x2 + ··· + thetan * xn

        令 x0 = 1

        y_hat = theta0 * x0 + theta1 * x1 + ···+ thetan * xn

        Y = X * Theta
        """
        if Theta == []:
            self.n = n
            self.Theta = tf.Variable(np.array([np.ones(n+1)]).T,dtype=tf.float32)
            
        else:
            self.Theta = Theta
            self.n = len(Theta) + 1
   
    @tf.function        
    def predict(self, X):
        return tf.matmul(tf.transpose(self.Theta),tf.transpose(self.extend(X)))
    
    @tf.function
    def loss(self, X, Y):
        Y_hat = self.predict(X)
        return tf.reduce_mean(tf.square(tf.subtract(Y, Y_hat)))    
    
    @tf.function
    def train(self, X, Y, epochs=100, learning_rate=0.0001):
        loss = []
        
        for epoch in range(epochs):

            with tf.GradientTape() as tape:
               
               curr_loss = self.loss(X, Y)
               gradient = tape.gradient(curr_loss, self.Theta)
               self.Theta.assign_sub(learning_rate*gradient)
               loss.append(curr_loss)
               
        return loss
        
    @staticmethod
    @tf.function
    def extend(X):
        return tf.concat([tf.ones([len(X), 1], dtype=tf.float32), X], axis=1)
    
def test():
    
    data = pd.read_csv('./winequality-red.csv', sep=';')
    data_np = data.to_numpy()
    
    X = tf.constant(data_np[:, :-1], dtype=tf.float32)
    Y = tf.constant(np.array([data_np[:, -1]]).T, dtype=tf.float32)
    
    lr = MultiLinearRegressionTf(n=X.shape[1])
    
    loss = lr.train(X, Y) 
    
    step = np.linspace(1, len(loss), len(loss))
   
    plt.figure()
    plt.ion()
    plt.show()
    plt.xlim(0, 120)
    plt.ylim(0, 250)    
    
    for i in range(0, len(loss)):
        
        if i % 3 == 0:         
            plt.scatter(step[i], loss[i])
            plt.pause(0.3)

test()

