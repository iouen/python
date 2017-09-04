#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivan
# Date: 2017-08-27
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
tf.nn.relu

# Using Tensorflow's default tools to fetch data, this is the same as what we did in the first homework assignment.
mnist = input_data.read_data_sets('./mnist', one_hot=True) 

# Random seed.
rseed = 42
batch_size = 200
lr = 1e-1
num_epochs = 20
num_hiddens = 500


num_train, num_feats = mnist.train.images.shape
num_test = mnist.test.images.shape[0]
num_classes = mnist.train.labels.shape[1]

# Placeholders that should be filled with training pairs (x, y). Use None to unspecify the first dimension 
# for flexibility.
x = tf.placeholder(tf.float32, [None, num_feats], name="x")
y = tf.placeholder(tf.int32, [None, num_classes], name="y")

# Model weights initialization.
# Your code here.
def U(fan_in,fan_out):
    # Initialize model parameters, sample W ~ [-U, U], where U = sqrt(6.0 / (fan_in + fan_out)).
    U = np.sqrt(6.0 / (fan_in + fan_out))
    return U

w1 = tf.Variable(tf.random_uniform([num_feats, num_hiddens],minval=-U(num_feats,num_hiddens),maxval=U(num_feats,num_hiddens) ,name="w1"), dtype=tf.float32)
w2 = tf.Variable(tf.random_uniform([num_hiddens, num_classes],minval=-U(num_hiddens,num_classes),maxval=U(num_hiddens,num_classes),name="w2"), dtype=tf.float32)
b1 = tf.Variable(tf.zeros([1,num_hiddens],name="b1"))
b2 = tf.Variable(tf.zeros([1,num_classes],name="b2"))
# logits is the log-probablity of each classes, forward computation.
# Your code here.

# 前向算法
# 添加全连接神经网络层 + relu层
output_1 = tf.nn.relu( tf.matmul(x, w1) + b1 )
# 再添加全连接神经网络层 + softmax层
outputs = tf.matmul(output_1, w2) + b2

#正则化


# Use TensorFlow's default implementation to compute the cross-entropy loss of classification.
# Your code here.
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=y, name="loss")
loss = tf.reduce_mean(cross_entropy)

#正则化
l2_loss = loss + tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2)
# Build prediction function.
# Your code here.

# Use TensorFlow's default implementation for optimziation algorithm. 
# Your code here.
optimizer = tf.train.GradientDescentOptimizer(lr).minimize(loss)


# Start training!
num_batches = int(num_train / batch_size)
losses = []
train_accs, valid_accs = [], []
time_start = time.time()
with tf.Session() as sess:
    # Before evaluating the graph, we should initialize all the variables.
    sess.run(tf.global_variables_initializer())
    for i in range(num_epochs):
        # Each training epoch contains num_batches of parameter updates.
        total_loss = 0.0
        for _ in range(num_batches):
            # Fetch next mini-batch of data using TensorFlow's default method.
            x_batch, y_batch = mnist.train.next_batch(batch_size)
            # Note that we also need to include optimizer into the list in order to update parameters, but we 
            # don't need the return value of optimizer.
            _, loss_batch = sess.run([optimizer, loss], feed_dict={x: x_batch, y: y_batch})
            total_loss += loss_batch

        _, loss_batch = sess.run([optimizer, loss], feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
        total_loss += loss_batch
        # Compute training set and validation set accuracy after each epoch.
        train_acc = 1 - sess.run(loss, feed_dict={x: mnist.train.images, y: mnist.train.labels})# your code here.
        valid_acc = 1 - sess.run(loss, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})# your code here.
        losses.append(total_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        print("Number of iteration: {}, total_loss = {}, train accuracy = {}, validation accuracy = {}".format(i, total_loss, train_acc, valid_acc))
    # Evaluate the test set accuracy at the end.
    test_acc = 1 - sess.run(loss, feed_dict={x: mnist.test.images, y: mnist.test.labels})
time_end = time.time()
print("Time used for training = {} seconds.".format(time_end - time_start))
print("MNIST image classification accuracy on test set = {}".format(test_acc))

# Plot the losses during training.
plt.figure()
plt.title("MLP-784-500-10 with TensorFlow")
plt.plot(losses, "b-o", linewidth=2)
plt.grid(True)
plt.xlabel("Iteration")
plt.ylabel("Cross-entropy")
plt.show()
