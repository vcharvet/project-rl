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

        self.conv1 = tf.layers.conv2d(inputs=self.X / 255.,
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
        self.streamA = tf.layers.batch_normalization(tf.layers.flatten(self.z4))
        self.streamV = tf.layers.batch_normalization(tf.layers.flatten(self.z4))
        AW = tf.layers.dense(inputs=self.streamA, units=out_size,
                                  activation=tf.nn.relu,
                                  kernel_initializer=xavier_initializer(),
                                  name='AW'+id)
        self.AW = tf.layers.batch_normalization(AW)
        self.Advantage = tf.layers.dense(inputs=self.AW, units=env.action_space.n,
                                         activation=None,
                                         kernel_initializer=xavier_initializer(),
                                         name='Advantage'+id)

        VW = tf.layers.dense(inputs=self.streamV, units=out_size,
                                  activation=tf.nn.relu,
                                  kernel_initializer=xavier_initializer(),
                                  name='VW'+id)
        self.VW = tf.layers.batch_normalization(VW)
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



class A2CPolicy(object):
    def __init__(self, env, img_shape, id, h_size, out_size, lr):
        img_row, img_col, nb_channels = img_shape

        # self.pd_type = make_pdtype(env.action_space.n)

        # input is a tensor of dimension 3
        self.X = tf.placeholder(shape=[None, img_row, img_col, nb_channels],
                                dtype=tf.float32)
        self.targetQ = tf.placeholder(dtype=tf.float32)
        self.action = tf.placeholder(shape=None, dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.action, env.action_space.n,
                                         dtype=tf.float32)


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

        self.flatten = tf.layers.flatten(self.z4)

        self.fc_common = tf.layers.dense(self.flatten,
                                         units=out_size,
                                         activation='relu')

        # self.pd, self.pi = self.pd_type.pdfromlatent(self.fc_common,
        #                                              init_scale=0.1)
        self.AW = tf.layers.dense(self.fc_common, env.action_space.n,
                                  kernel_initializer=xavier_initializer())
        self.action_probs = tf.squeeze(tf.nn.softmax(self.AW))
        self.picked_action_probs = tf.gather(self.action_probs, self.action)

        self.value_estim = tf.layers.dense(inputs=self.fc_common, units=1,
                                           activation=None,
                                           kernel_initializer=xavier_initializer(),
                                           name='Value' + id)[:, 0]

        self.actor_loss = -tf.log(self.picked_action_probs) * self.targetQ
        self.critic_loss = (self.value_estim - self.targetQ) ** 2

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        self.train_actor = self.optimizer.minimize(self.actor_loss)
        self.train_critic = self.optimizer.minimize(self.critic_loss)






