import numpy as np
import tensorflow as tf
import pandas as pd

from neural_net import DoubleQNetwork
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


def train_double_dqn(env, img_shape, batch_size=32, update_freq=4,  y=0.99,
                     startE=0.7, endE=0.95, annealing_steps=1000,
                     num_episodes=10000, max_ep_length=50,
                     pre_train_steps=10, checkpoint=50,
                     load_model=False, save_path='models/ddqn/',
                     lr=0.0001, h_size=64, out_size=64, render=False,
                     verbosity=20, tau=0.001):
    tf.reset_default_graph()
    # img_shape = 250, 160, 3
    mainQN = DoubleQNetwork(env, 'main', img_shape, h_size, out_size, lr)
    targetQN = DoubleQNetwork(env, 'target', img_shape, h_size, out_size, lr)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    trainables = tf.trainable_variables()
    target_ops = update_target_graph(trainables, tau)
    my_buffer = experience_buffer()

    e = startE
    step_drop = (endE - startE) / annealing_steps

    jList, rList, total_steps = [], [], 0

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with tf.Session() as sess:
        sess.run(init)
        if load_model:
            print('Loading model...')
            ckpt = tf.train.get_checkpoint_state(save_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # pretraining steps: random actions
        print('Performing random pretraining steps...')
        for i in range(pre_train_steps):
            episode_buffer = experience_buffer()
            s = env.reset()
            # s = s.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))
            j = 0
            while j < max_ep_length:
                j+=1
                a = np.random.randint(0, env.action_space.n)

                s, _, _, _ = env.step(a)
                s1, r, d, info = env.step(a)
                # s1 = s1.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))
                episode_buffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                s = s1

                if d:
                    break

            my_buffer.add(episode_buffer.buffer )
        print('Random steps done\n Begin training...')

        for i in range(num_episodes):
            if e < endE:
                e += step_drop

            episode_buffer = experience_buffer()
            s = env.reset()
            # s = s.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))
            rAll, j = 0, 0
            while j < max_ep_length:
                j +=1

                if np.random.rand(1)[0] > e:
                    a = np.random.randint(0, env.action_space.n)
                else:
                    s0 = s.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))
                    a = sess.run(mainQN.predict, feed_dict={mainQN.X: s0})[0]

                s1, r, d, info = env.step(a)
                total_steps +=1
                episode_buffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
                s = s1#.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))

                if render:
                    env.render()

                if total_steps % (update_freq) == 0:
                    train_batch = my_buffer.sample(batch_size)

                    states = np.vstack(train_batch[:, 3]).reshape(
                        (batch_size, img_shape[0],
                         img_shape[1], img_shape[2]))
                    Q1 = sess.run(mainQN.predict,
                                  feed_dict={mainQN.X: states})
                    Q2 = sess.run(targetQN.Qout,            #targetQN
                                  feed_dict={targetQN.X: states})
                    end_multiplier = -(train_batch[:, 4] - 1)
                    doubleQ = Q2[:batch_size, Q1]
                    targetQ = train_batch[:, 2] + (
                            y * doubleQ.dot(end_multiplier))

                    old_states = np.vstack(train_batch[:, 0]).reshape(
                        (batch_size, img_shape[0],
                         img_shape[1], img_shape[2]))
                    _, loss = sess.run([mainQN.update_model, mainQN.loss],
                                       feed_dict= \
                                           {mainQN.X: old_states,
                                            mainQN.targetQ: targetQ,
                                            mainQN.actions: train_batch[:, 1]})
                    if total_steps % (verbosity*5) == 0:
                        print('Total step {} loss={:.3}'.format(total_steps, loss))
                    # print('Step {} on iter {}: need to update the main model '
                    #       'with loss={:.3f}'.format(j, i, loss))

                    # update_target(target_ops, sess)
                rAll+=r
                if d:
                    print('Arriving on terminal state in iteration {}'.format(i))
                    break

            print('Episode terminated in {} steps --- curent e:{}'.format(j, e))
            my_buffer.add(episode_buffer.buffer)
            jList.append(j)
            rList.append(rAll)
            print('Reward of  iteration {}: {:.3f}\n'.format(i, rAll))
            if i % verbosity == 0:
                print('On iteration {} mean return={:.3f} ---- total '
                      'steps:{}'.format(i, sum(rList)/(i+1), total_steps))
            if (i+1)%checkpoint == 0:
                path = saver.save(sess, save_path)
                print('Model saved in path: {}'.format(path))
                df = pd.DataFrame([jList, rList], index=['j', 'reward']).T
                df.to_csv('models/ddqn/rewards.csv', sep=';')
    return jList, rList, num_episodes








