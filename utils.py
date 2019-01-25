import numpy as np
import tensorflow as tf

from neural_net import QNetwork
from buffer import experience_buffer
#import tf.layers.{dense, conv2d
import os


def processState(states, img_shape):
    return np.reshape(states, [img_shape[0] * img_shape[1] * img_shape[2]])

def update_target_graph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0: total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(\
            (var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)


def train_dqn(env, img_shape, batch_size=32, update_freq=4,  y=0.99, startE=1, endE=0.1,
              annealing_steps=1000, num_episodes=10000, pre_train_episodes=100,
              max_ep_length=50, load_model=False, save_path='./data/dqn',
              lr=0.0001, h_size=64, out_size=512, render=False, verbosity=20,
              tau=0.001):
    tf.reset_default_graph()
    # img_shape = 250, 160, 3
    mainQN = QNetwork(env, 'main', img_shape, h_size, out_size, lr)
    targetQN = QNetwork(env, 'target', img_shape, h_size, out_size, lr)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    trainables = tf.trainable_variables()
    target_ops = update_target_graph(trainables, tau)
    my_buffer = experience_buffer()

    e = startE
    step_drop = (startE - endE) / annealing_steps

    jList, rList, total_steps = [], [], 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with tf.Session() as sess:
        sess.run(init)
        if load_model:
            print('Loading model...')
            ckpt = tf.train.get_checkpoint_state(save_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(num_episodes):
            episode_buffer = experience_buffer()
            s = env.reset()
            s = s.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))
            rAll, j = 0, 0
            while j < max_ep_length:
                j +=1
                if np.random.rand(1) < e or i < pre_train_episodes:
                    a = np.random.randint(0, env.action_space.n)
                else:
                    a = sess.run(mainQN.predict, feed_dict={mainQN.X: s})[0]

                s1, r, d, info = env.step(a)
                total_steps +=1
                episode_buffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

                if render:
                    env.render()
                if i > pre_train_episodes:
                    if e > endE:
                        e -= step_drop
                    if total_steps % (update_freq) == 0:
                        train_batch = my_buffer.sample(batch_size)

                        states = np.vstack(train_batch[:, 3]).reshape((batch_size, img_shape[0],
                                                            img_shape[1], img_shape[2]))
                        Q1 = sess.run(mainQN.predict,
                                      feed_dict={mainQN.X: states})
                        Q2 = sess.run(targetQN.Qout,
                                      feed_dict={targetQN.X: states})
                        end_multiplier = -(train_batch[:, 4] - 1)
                        doubleQ = Q2[:batch_size, Q1]
                        targetQ = train_batch[:, 2] + (y*doubleQ.dot(end_multiplier))

                        old_states = np.vstack(train_batch[:, 0]).reshape((batch_size, img_shape[0],
                                                            img_shape[1], img_shape[2]))
                        _, loss = sess.run([mainQN.update_model, mainQN.loss], feed_dict= \
                            {mainQN.X: old_states, mainQN.targetQ: targetQ,
                             mainQN.actions: train_batch[:, 1]})
                        # print('Step {} on iter {}: need to update the main model '
                        #       'with loss={:.3f}'.format(j, i, loss))

                        update_target(target_ops, sess)
                # print("Finish iteration {}, return={}"\
                #       .format(i, r))
                rAll+=r
                if d:
                    print('Arriving on terminal state in iteration {}'.format(i))
                    break
            print('Episode terminated in {} steps'.format(j))
            my_buffer.add(episode_buffer.buffer)
            jList.append(j)
            rList.append(rAll)
            print('Reward of  iteration {}: {:.3f}\n'.format(i, rAll/j))
            if i % verbosity == 0:
                print('On iteration {} mean return={:.3f} ---- total '
                      'steps:{}'.format(i, sum(rList)/(i+1), total_steps))
    return jList, rList, num_episodes








