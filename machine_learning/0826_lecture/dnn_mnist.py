#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivan
# Date: 2017-08-26
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


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

# Initialize model parameters, sample W ~ [-U, U], where U = sqrt(6.0 / (fan_in + fan_out)).
np.random.seed(rseed)
# Your code here to create model parameters globally.



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
    # Your code here.
    pass

def softmax(inputs):
    """
    Compute the softmax nonlinear activation function.
    """
    # Your code pass
    pass


def forward(inputs):
    """
    Forward evaluation of the model.
    """
    # Your code here.
    pass

"""
python3 不支持 modify
def backward(probs, labels, (x, h1, h2)):
"""
def backward(probs, labels, x, h1, h2):
    """
    Backward propagation of errors.
    """
    # Your code here.
    pass


def predict(probs):
    """
    Make predictions based on the model probability.
    """
    # Your code here.
    pass


def evaluate(inputs, labels):
    """
    Evaluate the accuracy of current model on (inputs, labels).
    """
    # Your code here.
    pass


# Training using stochastic gradient descent.
time_start = time.time()
#强制转成int
num_batches = int (num_train/batch_size)
train_accs, valid_accs = [], []
#python3 没有xrange这中方式
for i in range(num_epochs):
    for j in range(num_batches):
        # Fetch the j-th mini-batch of the data.
        insts = mnist.train.images[batch_size * j: batch_size * (j+1), :]
        labels = mnist.train.labels[batch_size * j: batch_size * (j+1), :]
        # Forward propagation.
        # Your code here.

        # Backward propagation.
        # Your code here.
        
        # Gradient update.
        w1 = w1- lr * dw1
        w2 = w2- lr * dw2
        b1 = b1- lr * db1
        b2 = b2- lr * db2
    # Evaluate on both training and validation set.
    train_acc = evaluate(mnist.train.images, mnist.train.labels)
    valid_acc = evaluate(mnist.validation.images, mnist.validation.labels)
    train_accs.append(train_acc)
    valid_accs.append(valid_acc)
    print ("Number of iteration: {}, classification accuracy on training set = {}, classification accuracy on validation set: {}".format(i, train_acc, valid_acc))
time_end = time.time()
# Compute test set accuracy.
acc = evaluate(mnist.test.images, mnist.test.labels)
print ("Final classification accuracy on test set = {}".format(acc))
print ("Time used to train the model: {} seconds.".format(time_end - time_start))

# Plot classification accuracy on both training and validation set for better visualization.
plt.figure()
plt.plot(train_accs, "bo-", linewidth=2)
plt.plot(valid_accs, "go-", linewidth=2)
plt.legend(["training accuracy", "validation accuracy"], loc=4)
plt.grid(True)
plt.show()
