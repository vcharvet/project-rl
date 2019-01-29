import tensorflow as tf
import numpy as np
import gym

from dueling_dqn import update_target_graph
from neural_net import A2CPolicy
from a2c_net import ActorCritic
from buffer import experience_buffer




def actor_critic(env, img_shape, batch_size=32, update_freq=4,  y=0.99,
                     startE=0.7,endE=0.95, annealing_steps=1000,
                     num_episodes=10000, max_ep_length=50,
                     pre_train_steps=10, discount=0.9,
                     load_model=False, save_path='./data/dqn',
                     lr=0.0001, h_size=64, out_size=64, render=False,
                     verbosity=20, tau=0.001):
    lr_actor, lr_critic = lr, 5*lr
    tf.reset_default_graph()
    sess = tf.Session()
    total_steps = 0
    # actorNet = ActorCritic(sess, env, img_shape, 32, 64, False, lr_actor)
    # criticNet = ActorCritic(sess, env, img_shape, 32, 64, True, lr_critic)
    ACNet = ActorCritic(sess, env, img_shape, 32, 64, False, 0.01)
    sess.run(tf.global_variables_initializer())
    e = startE
    step_drop = (endE - startE) / annealing_steps
    jList, rList = [], []

    my_buffer = experience_buffer()
    print('Performing random pretraining steps...')
    for i in range(pre_train_steps):
        episode_buffer = experience_buffer()
        s = env.reset()
        # s = s.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))
        j = 0
        while j < max_ep_length:
            j += 1
            a = np.random.randint(0, env.action_space.n)

            s1, r, d, info = env.step(a)
            # s1 = s1.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))
            episode_buffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
            s = s1

            if d:
                break
        my_buffer.add(episode_buffer.buffer)
    print('Random steps done\n Begin training...')

    for i in range(num_episodes):
        if e < endE:
            e += step_drop

        episode_buffer = experience_buffer()
        s = env.reset()

        rAll, j = 0, 0
        while j < max_ep_length:
            j += 1

            if np.random.rand(1)[0] > e:
                a = np.random.randint(0, env.action_space.n)
            else:
                s0 = s.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))
                a = ACNet.pick_action(s0)
                # if total_steps % verbosity == 0:
                #     print('Action probabilities:', a)
                a = np.random.choice(np.arange(len(a)), p=a)

            s1, r, d, info = env.step(a)
            total_steps += 1
            episode_buffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))
            rAll += r

            if render:
                env.render()

            s1 = s1.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))
            s = s.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))

            # td_error = criticNet.learn_critic(s, r, s1)
            # actor_loss = actorNet.learn_actor(s, a, td_error)
            value_next = sess.run(ACNet.value_estim,
                                  {ACNet.X: s1})
            td_err, _ = sess.run([ACNet.td_err_out,
                                  ACNet.train_critic],
                                 {ACNet.X: s,
                                  ACNet.reward: r,
                                  ACNet.value_next: value_next})
            actor_loss, _ = sess.run([ACNet.actor_loss,
                                      ACNet.train_actor],
                                     {ACNet.X: s,
                                      ACNet.action: a,
                                      ACNet.td_error: td_err})

            # print(td_err, actor_loss)
            if total_steps % verbosity == 0:
                    print('Total step {}: actor_loss={:.3f} - td_error={:.3f}'\
                          .format(total_steps, actor_loss, td_err))

            if d:
                print('Arriving on terminal state in iteration {}'.format(i))
                break
            s = s1
        print('Episode terminated in {} steps --- curent e:{}'.format(j, e))
        my_buffer.add(episode_buffer.buffer)
        jList.append(j)
        rList.append(rAll)
        print('Reward of  iteration {}: {:.3f}\n'.format(i, rAll))
        if i % verbosity == 0:
            print('On iteration {} mean return={:.3f} ---- total '
                  'steps:{}'.format(i, sum(rList) / (i + 1), total_steps))
    return jList, rList, num_episodes


#@deprecated
def actor_critic_deprecated(env, img_shape, batch_size=32, update_freq=4,  y=0.99,
                     startE=0.7,endE=0.95, annealing_steps=1000,
                     num_episodes=10000, max_ep_length=50,
                     pre_train_steps=10, discount=0.9,
                     load_model=False, save_path='./data/dqn',
                     lr=0.0001, h_size=64, out_size=64, render=False,
                     verbosity=20, tau=0.001):
    tf.reset_default_graph()

    total_steps = 0

    actorNet = A2CPolicy(env, img_shape, 'actor', h_size, out_size, lr)
    criticNet = A2CPolicy(env, img_shape, 'critic', h_size, out_size, lr)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    trainables = tf.trainable_variables()
    critic_ops = update_target_graph(trainables, tau)

    e = startE
    step_drop = (endE - startE) / annealing_steps

    my_buffer = experience_buffer()
    jList, rList = [], []

    with tf.Session() as sess:
        sess.run(init)
        if load_model:
            print('not implemented yet')
            quit()

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

            rAll, j = 0, 0
            while j < max_ep_length:
                j += 1

                if np.random.rand(1)[0] > e:
                    a = np.random.randint(0, env.action_space.n)
                else:
                    s0 = s.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))
                    action_prob = sess.run(actorNet.action_probs,
                                           feed_dict={actorNet.X: s0})
                    a = np.random.choice(np.arange(len(action_prob)),
                                         p=action_prob)

                s1, r, d, info = env.step(a)
                total_steps += 1
                episode_buffer.add(np.reshape(np.array([s, a, r, s1, d]), [1, 5]))

                if render:
                    env.render()

                rAll += r
                s1 = s1.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))
                s = s.reshape((-1, img_shape[0], img_shape[1], img_shape[2]))

                value_estim = sess.run(criticNet.value_estim,
                                       feed_dict={criticNet.X: s1})
                td_target = r + discount * value_estim
                value_next =  sess.run(criticNet.value_estim,
                                       feed_dict={criticNet.X: s})

                actor_loss, _ = sess.run([actorNet.actor_loss,
                                          actorNet.train_actor],
                                         feed_dict={actorNet.X: s,
                                                    actorNet.action: a,
                                                    actorNet.targetQ: r})
                # update_target_graph(critic_ops, tau)
                critic_loss, _ = sess.run([criticNet.critic_loss,
                                           criticNet.train_critic],
                                          feed_dict={criticNet.X: s,
                                                     criticNet.action: a,
                                                     criticNet.targetQ: r})

                print(actor_loss, critic_loss)
                if total_steps % verbosity == 0:
                    print('Total step {}: actor_loss={:.3f} - critic_loss={:.3f}'\
                          .format(actor_loss, critic_loss[0]))

                if d:
                    print('Arriving on terminal state in iteration {}'.format(i))
                    break
                s = s1

            print('Episode terminated in {} steps --- curent e:{}'.format(j, e))
            my_buffer.add(episode_buffer.buffer)
            jList.append(j)
            rList.append(rAll)
            print('Reward of  iteration {}: {:.3f}\n'.format(i, rAll))
            if i % verbosity == 0:
                print('On iteration {} mean return={:.3f} ---- total '
                      'steps:{}'.format(i, sum(rList)/(i+1), total_steps))
    return jList, rList, num_episodes

