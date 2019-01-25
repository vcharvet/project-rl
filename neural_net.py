import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers import xavier_initializer

import scipy.misc
import random
import gym


class QNetwork:
    def __init__(self, env, id, img_shape, h_size, out_size, lr=0.0001):
        img_row, img_col, nb_channels = img_shape

        # initialization of weights and biases
        # self.wc1 = tf.get_variable('wc1'+ id, shape=[3, 3, nb_channels, h_size],
        #                            initializer=xavier_initializer())
        # self.bc1 = tf.Variable(tf.zeros([h_size]), name='bc1'+id)
        # self.wc2 = tf.get_variable('wc2'+id, shape=[3, 3, h_size, h_size],
        #                            initializer=xavier_initializer())
        # self.bc2 = tf.Variable(tf.zeros([h_size]), name='bc2'+id)
        # self.wout = tf.Variable(tf.random.normal([h_size*h_size*nb_channels, out_size]),
        #                         name='wout'+id)
        # self.bout = tf.Variable(tf.zeros(out_size), name='bout'+id)
        self.X = tf.placeholder(shape=[None, img_row, img_col, nb_channels], dtype=tf.float32)
        # self.conv1 = tf.nn.conv2d(self.X, self.wc1, padding='SAME', strides=[1, 1, 1, 1],
        #                           name='conv1'+id, data_format='NHWC')
        self.conv1 = tf.layers.conv2d(inputs=self.X,
                                      filters=h_size,
                                      kernel_size=[3, 3],
                                      strides=[2, 2],
                                      padding='SAME',
                                      kernel_initializer=xavier_initializer(),
                                      name='conv1'+id)
        # self.z1 = tf.nn.dropout(tf.nn.relu(tf.add(self.conv1, self.bc1)), 0.75)
        self.z1 = tf.nn.relu(self.conv1, name='a1'+id)

        # self.conv2 = tf.nn.conv2d(self.z1, self.wc2, padding='SAME', strides=[1, 2, 2, 1],
        #                           name='conv2'+id, data_format='NHWC')
        # self.a2 = tf.nn.dropout(tf.nn.relu(tf.add(self.conv2, self.bc2)), 0.75)
        # self.z2 = tf.nn.max_pool(self.a2, ksize=[1, 2, 2, 1],
        #                          strides=[1, 2, 2, 1], padding='SAME', name='maxpool2'+id)
        self.conv2 = tf.layers.conv2d(self.z1,
                                      filters=h_size,
                                      kernel_size=[3, 3],
                                      strides=[2, 2],
                                      padding='SAME',
                                      kernel_initializer=xavier_initializer(),
                                      name='conv2'+id)
        self.z2 = tf.nn.relu(self.conv2, name='a2'+id)

        # self.aout = tf.matmul(tf.reshape(self.z2, [-1, self.wout.shape[0]]), self.wout)
        # self.zout = tf.add(self.aout, self.aout)
        self.conv3 = tf.layers.conv2d(self.z2,
                                      filters=2*h_size,
                                      kernel_size=[3, 3],
                                      strides=[2, 2],
                                      padding='SAME',
                                      name='conv3'+id)
        self.z3 = tf.nn.relu(self.conv3, name='a2'+id)

        #split the output from the final layer between advantage and value streams
        # self.streamAC, self.streamVC = tf.split(self.zout, 2, 1)
        self.streamA = tf.layers.flatten(self.z3)
        self.streamV = tf.layers.flatten(self.z3)
        # self.AW = tf.get_variable('AW'+id, shape=[out_size //2, env.action_space.n],
        #                           initializer=xavier_initializer())
        # self.VW = tf.get_variable('VW'+id, shape=[out_size//2, 1],
        #                          initializer=xavier_initializer())
        self.AW = tf.layers.dense(inputs=self.streamA, units=out_size,
                                  activation=tf.nn.relu,
                                  kernel_initializer=xavier_initializer(),
                                  name='AW'+id)
        self.Advantage = tf.layers.dense(inputs=self.AW, units=env.action_space.n,
                                         activation=None,
                                         kernel_initializer=xavier_initializer(),
                                         name='Advantage'+id)

        self.VW = tf.layers.dense(inputs=self.streamV, units=out_size,
                                  activation=tf.nn.relu,
                                  kernel_initializer=xavier_initializer(),
                                  name='VW'+id)
        self.Value = tf.layers.dense(inputs=self.VW, units=1,
                                     activation=None,
                                     kernel_initializer=xavier_initializer(),
                                     name='Value'+id)

        # combine both outputs
        self.Qout = self.Value + tf.math.subtract(self.Advantage,
                                              tf.reduce_mean(self.Advantage,
                                                             axis=1,
                                                             keepdims=True))
        self.predict = tf.argmax(self.Qout, 1)

        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.action_space.n,
                                         dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot),
                               axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_model = self.trainer.minimize(self.loss)







