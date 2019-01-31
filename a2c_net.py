import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers import xavier_initializer


def conv_layer(inputs, h_size, filter_size, strides):
    return tf.layers.conv2d(inputs=inputs,
                            filters=h_size,
                            kernel_size=[filter_size, filter_size],
                            strides=[strides, strides],
                            padding='SAME',
                            kernel_initializer=xavier_initializer())

def dense_layer(inputs, units, activation=None, std=0.1):
    return tf.layers.dense(inputs, units=units, activation=activation,
                           kernel_initializer=tf.random_normal_initializer(stddev=std),
                           bias_initializer=tf.constant_initializer(1.))

class ActorCritic(object):
    def __init__(self, sess, env, img_shape, h_size, out_size, clip_length,
                 reuse, gamma=0.9, lr=0.001, beta=0.05):
        img_row, img_col, nb_channels = img_shape
        self.beta = tf.constant(beta)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=5*lr)
        self.sess = sess
        # input is a tensor of dimension 3
        self.X = tf.placeholder(shape=[None, img_row, img_col, nb_channels],
                                dtype=tf.float32)
        self.td_error = tf.placeholder(shape=[None], dtype=tf.float32)
        self.value_next = tf.placeholder(shape=[None], dtype=tf.float32)
        self.reward = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action = tf.placeholder(shape=[None], dtype=tf.int32)
        #inputs for preferences
        self.segment1 = tf.placeholder(shape=None,
                                       dtype=tf.float32)
        self.segment2 = tf.placeholder(shape=None,
                                       dtype=tf.float32)

        # self.actions_onehot = tf.one_hot(self.action, env.action_space.n,
        #                                  dtype=tf.int32)

        #building common input layers to extract image features
        self.conv1 = conv_layer(self.X / 255., h_size, 7, 3)
        self.a1 = tf.layers.batch_normalization(tf.nn.dropout(self.conv1, 0.5)) #removed batch_norm
        self.z1 = tf.nn.relu(self.a1)

        self.conv2 = conv_layer(self.z1, h_size, 5, 2)
        self.a2 = tf.layers.batch_normalization(tf.nn.dropout(self.conv2, 0.5))
        self.z2 = tf.nn.relu(self.a2)

        self.conv3 = conv_layer(self.z2, h_size, 3, 1)
        self.a3 = tf.layers.batch_normalization(tf.nn.dropout(self.conv3, 0.5))
        self.z3 = tf.nn.relu(self.a3)

        self.conv4 = conv_layer(self.z3, h_size, 3, 1)
        self.a4 = tf.layers.batch_normalization(tf.nn.dropout(self.conv4, 0.5))
        self.z4 = tf.nn.relu(self.a4)

        self.flatten = tf.layers.batch_normalization(tf.layers.flatten(self.z4))
        self.fc_out = tf.layers.batch_normalization(dense_layer(self.flatten,
                                                                out_size,
                                                                activation=tf.nn.relu))
        # actor output
        self.advantage = dense_layer(self.fc_out, env.action_space.n)
        probs = tf.squeeze(tf.nn.softmax(self.advantage))
        self.action_probs = tf.clip_by_value(probs, 10e-6, 0.999999)
        self.log_action_probs = tf.log(self.action_probs)
        #critic output
        self.value_estim = dense_layer(self.fc_out, 1, std=1)

        picked_action_prob = tf.gather(self.action_probs, self.action, axis=1)
        # picked_action_prob = tf.batch_gather(self.action_probs, self.actions_onehot)
        self.actor_loss = tf.reduce_sum(-tf.log(picked_action_prob) * self.td_error)
        self.actor_loss_entropy = (self.actor_loss) - self.beta * \
            tf.reduce_sum(tf.multiply(self.action_probs, self.log_action_probs))

        self.train_actor = self.optimizer.minimize(self.actor_loss_entropy)

        self.td_err_out = tf.reduce_sum(
            self.reward + gamma * self.value_next - self.value_estim,
            axis=1)
        self.critic_loss = (self.td_err_out ** 2)
        self.train_critic = self.optimizer.minimize(self.critic_loss)

        #part for preferences
        self.estim_values1 = tf.reduce_sum(self.value_estim[:clip_length])
        self.estim_values2 = tf.reduce_sum(self.value_estim[clip_length:])

        self.exp_values1 = self.estim_values1 ** 2  #tf.exp(self.estim_values1)
        self.exp_values2 = self.estim_values2 ** 2#tf.exp(self.estim_values2)

        self.P1 = self.exp_values1 / (self.exp_values1 + self.exp_values2)
        self.P2 = 1 - self.P1

        self.pref_loss = - (self.segment1 * tf.log(self.P1) \
            + self.segment2 * tf.log(self.P2))
        self.train_pref = self.optimizer.minimize(self.pref_loss)

    # def learn_actor(self, state, action, td_err):
    #     train_actor = self.optimizer.minimize(self.actor_loss)
    #     loss, _ = self.sess.run([self.actor_loss, train_actor],
    #                             feed_dict={self.X: state,
    #                                        self.action: action,
    #                                        self.td_error: td_err})
    #     return loss
    #
    # def learn_critic(self, state, reward, next_state):
    #     train_critic = self.optimizer.minimize(self.critic_loss)
    #     loss, _ = self.sess.run([self.critic_loss, train_critic],
    #                             feed_dict={self.X: state,
    #                                        self.reward: reward,
    #                                        self.value_next: next_state})

    def pick_action(self, state):
        a = self.sess.run(self.action_probs,
                          feed_dict={self.X: state})
        return a




