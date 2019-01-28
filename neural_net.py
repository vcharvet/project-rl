import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorflow.contrib.layers import xavier_initializer

import scipy.misc
import random
import gym


class DoubleQNetwork:
    def __init__(self, env, id, img_shape, h_size, out_size, lr=0.0001):
        img_row, img_col, nb_channels = img_shape

        # input is a tensor of dimension 3
        self.X = tf.placeholder(shape=[None, img_row, img_col, nb_channels], dtype=tf.float32)

        self.conv1 = tf.layers.conv2d(inputs=self.X,
                                      filters=h_size,
                                      kernel_size=[7, 7],
                                      strides=[3, 3],
                                      padding='SAME',
                                      kernel_initializer=xavier_initializer(),
                                      name='conv1'+id)
        self.a1 = tf.layers.batch_normalization(tf.nn.dropout(self.conv1, 0.5))
        self.z1 = tf.nn.relu(self.a1, name='a1'+id)

        self.conv2 = tf.layers.conv2d(self.z1,
                                      filters=2*h_size,
                                      kernel_size=[5, 5],
                                      strides=[2, 2],
                                      padding='SAME',
                                      kernel_initializer=xavier_initializer(),
                                      name='conv2'+id)
        self.a2 = tf.layers.batch_normalization(tf.nn.dropout(self.conv2, 0.5))
        self.z2 = tf.nn.relu(self.a2, name='a2'+id)

        self.conv3 = tf.layers.conv2d(self.z2,
                                      filters=h_size,
                                      kernel_size=[3, 3],
                                      strides=[1, 1],
                                      padding='SAME',
                                      name='conv3'+id)
        self.a3 = tf.layers.batch_normalization(tf.nn.dropout(self.conv3, 0.5))
        self.z3 = tf.nn.relu(self.a3, name='z3'+id)

        self.conv4 = tf.layers.conv2d(self.z3,
                                      filters=h_size,
                                      kernel_size=[3, 3],
                                      strides=[1, 1],
                                      padding='SAME',
                                      kernel_initializer=xavier_initializer(),
                                      name='conv4'+id)
        self.a4 = tf.layers.batch_normalization(tf.nn.dropout(self.conv4, 0.5))
        self.z4 = tf.nn.relu(self.a4, name='z4'+id)

        # splitting the network for DDQN
        self.streamA = tf.layers.flatten(self.z4)
        self.streamV = tf.layers.flatten(self.z4)
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

        # target inputs
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],
                                      dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.action_space.n,
                                         dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot),
                               axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_model = self.trainer.minimize(self.loss)







