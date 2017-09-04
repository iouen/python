#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivan
# Date: 2017-08-26
import argparse
import time

from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy as np


# Load the MNIST dataset from the official website.
mnist = input_data.read_data_sets("mnist/", one_hot=True)

num_train, num_feats = mnist.train.images.shape
num_test = mnist.test.images.shape[0]
num_classes = mnist.train.labels.shape[1]

# Set hyperparameters of MLP.
rseed = 42
batch_size = 200
lr = 1e-1
num_hiddens = 500
num_epochs = 20
np.random.seed(rseed)
# Your code here to create model parameters globally.

def U(fan_in,fan_out):
    # Initialize model parameters, sample W ~ [-U, U], where U = sqrt(6.0 / (fan_in + fan_out)).
    U = np.sqrt(6.0 / (fan_in + fan_out))
    return U

u = U(num_feats,num_hiddens)
w1 = 2*u*np.random.random_sample(size=(num_feats, num_hiddens)) - u
u = U(num_hiddens,num_classes)
w2 = 2*u*np.random.random_sample(size=(num_hiddens, num_classes)) - u
b1 = np.zeros(num_hiddens)
b2 = np.zeros(num_classes)

# Used to store the gradients of model parameters.


dw1 = np.zeros((num_feats, num_hiddens))
db1 = np.zeros(num_hiddens)
dw2 = np.zeros((num_hiddens, num_classes))
db2 = np.zeros(num_classes)

# Helper functions.
def ReLU(inputs):
    """
    Compute the ReLU: max(x, 0) nonlinearity.
    """
    outputs = inputs.copy()
    outputs[outputs<0] = 0
    return outputs


def fully_connect_forward(x, w, b):
    """
    全连接前向网络
    输入：
        x: 输入向量，（batch_size， N）形状的矩阵
        w: 权重，（N, M）形状的矩阵
        b: 偏置，一个大小为M的向量
    输出：
        outputs =  x .* w + b
    """
    outputs = np.dot(x , w) + b
    return outputs


def softmax(inputs):
    """
    Compute the softmax nonlinear activation function.

    Forward evaluation of the model.
    输入：
        inputs：形如（batch_size,10）的np.narray对象，代表batch_size个图片，每个图片已降到10维。
    输出：
        onputs：形如（batch_size,10）的np.narray对象。已规范化，每行的和为1。
    """
    exp_lists =np.e ** inputs
    batch_size,_ = exp_lists.shape

    for i in range(batch_size):
        sum = np.sum(exp_lists[i])
        exp_lists[i] = exp_lists[i]/sum

    return exp_lists


def ReLU_backward(din,x):
    """
    ReLU层反向传播
    """
    dout = din.copy()
    dout[x <= 0] = 0
    return dout


def fully_connect_backward(dout, x, w):
    """
    全连接层反向传播
    输入：
        dout: 上一层传下来的误差，（batch_size ， N）形状的矩阵
        w: 本层权重，（N, M）形状的矩阵
        x: 本层输入，上一层输出
    输出：
        dw: 对w的偏导
        dx: 对x的偏导
        db: 对b的偏导
    """
    batch_size,_ = dout.shape
    dw = np.dot(x.T,dout) / batch_size
    dx = np.dot(dout, w.T) / batch_size
    db = np.dot(np.ones(batch_size) ,dout)  /batch_size
    return dw,dx,db

def softmax_backward(inputs,labels):
    """
    Make predictions based on the model probability.
    输入：
        inputs: 预测标签，形状（N，10）的narray
        labels: 真实标签，形状（N，10）的narray，结果用one-hot形式表示出来
    输出：
        dout = -( e(x) - y(x) )
    """

    return -(labels - inputs)

def forward(input):
    """
    Forward evaluation of the model.
    整合前向网络层：
    全连接 + ReLU + 全连接 + softmax

    """
    # 前向算法
    # 添加全连接神经网络层+relu层
    outputs_1 = ReLU(fully_connect_forward(input, w1, b1))


    # 再添加全连接神经网络层+softmax层
    outputs = softmax(fully_connect_forward(outputs_1, w2, b2))

    return outputs, outputs_1


def backward(probs, labels, x1, x2):
    """
    Backward propagation of errors.
    """
    dout = softmax_backward(probs, labels)

    dw2, dx2, db2 = fully_connect_backward(dout, x2, w2)

    dout = ReLU_backward(dx2,x2)

    dw1, dx1, db1 = fully_connect_backward(dout, x1, w1)
    grad = [dw1,dw2,db1,db2]
    return  grad


def predict(probs):
    """
    Make predictions based on the model probability.
    输入：可能标签，结果用one-hot形式表示出来
    输出：返回最大可能的位置
    """
    return np.argmax(probs,axis=1)


def evaluate(inputs, labels):
    """
    输入：
        inputs: 原始数据
        labels: 真实标签，形状（N，10）的narray，结果用one-hot形式表示出来
    输出:
        outputs: 平均
    """

    probs,_ = forward(inputs)
    prob_loc  = predict(probs)
    label_loc  = np.argmax(probs, axis=1)
    return np.mean(prob_loc==label_loc),np.mean(np.max(probs,axis=1))



# Training using stochastic gradient descent.
time_start = time.time()
num_batches = int(num_train / batch_size)
train_accs, valid_accs = [], []
for i in range(num_epochs):
    for j in range(num_batches):
        # Fetch the j-th mini-batch of the data.
        insts = mnist.train.images[batch_size * j: batch_size * (j+1), :]
        labels = mnist.train.labels[batch_size * j: batch_size * (j+1), :]

        # 前向算法

        outputs, x2  = forward(insts)
        # 后向算法
        dw1, dw2, db1, db2 = backward(outputs,labels,insts,x2)
        # Gradient update.
        w1 -= lr * dw1
        w2 -= lr * dw2
        b1 -= lr * db1
        b2 -= lr * db2
    # Evaluate on both training and validation set.
    train_acc = evaluate(mnist.train.images, mnist.train.labels)
    valid_acc = evaluate(mnist.validation.images, mnist.validation.labels)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    print("Number of iteration: {}, classification accuracy on training set = {}, classification accuracy on validation set: {}".format(i, train_acc, valid_acc))
time_end = time.time()
# Compute test set accuracy.

for i in range(num_epochs):
    for j in range(num_batches):
        # Fetch the j-th mini-batch of the data.
        insts = mnist.validation.images[batch_size * j: batch_size * (j+1), :]
        labels = mnist.validation.labels[batch_size * j: batch_size * (j+1), :]

        # 前向算法
        outputs, x2 = forward(insts)
        # 后向算法
        dw1, dw2, db1, db2 = backward(outputs, labels, insts, x2)
        # Gradient update.
        w1 -= lr * dw1
        w2 -= lr * dw2
        b1 -= lr * db1
        b2 -= lr * db2



acc = evaluate(mnist.test.images, mnist.test.labels)
print("Final classification accuracy on test set = {}".format(acc))
print("Time used to train the model: {} seconds.".format(time_end - time_start))

# Plot classification accuracy on both training and validation set for better visualization.
plt.figure()
plt.plot(train_accs, "bo-", linewidth=2)
plt.plot(valid_accs, "go-", linewidth=2)
plt.legend(["training accuracy", "validation accuracy"], loc=4)
plt.grid(True)
plt.show()
