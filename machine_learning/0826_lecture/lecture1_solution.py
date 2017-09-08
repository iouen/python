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
u1 =  2.0 * np.sqrt(6.0 / (num_feats + num_hiddens))
w1 = (np.random.rand(num_feats, num_hiddens) - 0.5) * u1
b1 = np.zeros(num_hiddens)
u2 = 2.0 * np.sqrt(6.0 / (num_hiddens + num_classes))
w2 = (np.random.rand(num_hiddens, num_classes) - 0.5) * u2
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
    inputs[inputs < 0.0] = 0.0
    return inputs


def softmax(inputs):
    """
    Compute the softmax nonlinear activation function.
    """
    probs = np.exp(inputs)
    probs /= np.sum(probs, axis=1)[:, np.newaxis]
    return probs


def forward(inputs):
    """
    Forward evaluation of the model.
    """
    h1 = ReLU(np.dot(inputs, w1) + b1)
    h2 = np.dot(h1, w2) + b2
    return (h1, h2), softmax(h2)


def backward(probs, labels, x, h1, h2):
    """
    Backward propagation of errors.
    """
    n = probs.shape[0]
    e2 = probs - labels
    e1 = np.dot(e2, w2.T)
    e1[h1 <= 0.0] = 0.0
    dw2[:] = np.dot(h1.T, e2) / n
    db2[:] = np.mean(e2, axis=0)
    dw1[:] = np.dot(x.T, e1) / n
    db1[:] = np.mean(e1, axis=0)


def predict(probs):
    """
    Make predictions based on the model probability.
    """
    return np.argmax(probs, axis=1)


def evaluate(inputs, labels):
    """
    Evaluate the accuracy of current model on (inputs, labels).
    """
    _, probs = forward(inputs)
    preds = predict(probs)
    trues = np.argmax(labels, axis=1)
    return np.mean(preds == trues)


# Training using stochastic gradient descent.
time_start = time.time()
num_batches = int  (num_train / batch_size)
train_accs, valid_accs = [], []
for i in range(num_epochs):
    for j in range(num_batches):
        # Fetch the j-th mini-batch of the data.
        insts = mnist.train.images[batch_size * j: batch_size * (j+1), :]
        labels = mnist.train.labels[batch_size * j: batch_size * (j+1), :]
        # Forward propagation.
        (h1, h2), probs = forward(insts)
        # Backward propagation.
        backward(probs, labels, insts, h1, h2)
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
