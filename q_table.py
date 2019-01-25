import numpy as np
import tensorflow as tf



def q_table_learning(env, lr=0.8, y=0.95, num_episodes=2000, verbosity=100):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    jList, rList = [], []

    for i in range(num_episodes):
        s = env.reset()
        rAll = 0
        # d = False
        j = 0
        while j < 99:
            j+=1
            a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) *
                          (1/(i+1)))
            s1, r, d, info = env.step(a)
            Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
            rAll += r
            s=s1
            if d: #state is terminal
                break
        if (i % verbosity) == 0:
            print('Mean Return on iteration {} = {:.3f}'\
                .format(i, sum(rList)/(i+1)))
        rList.append(rAll)
        jList.append(j)
    return jList, rList, num_episodes


def nn_learning(env, lr=0.1, y=0.99, num_episodes=2000, e=0.1, verbosity=100):
    tf.reset_default_graph()
    # S, A = env.observation_space.n, env.action_space.n
    S, A = env.observation_space.shape[0], env.action_space.n
    Id = np.identity(S)

    inputs1 = tf.placeholder(shape=[S],
                             dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([S, A], 0, 0.01))
    Qout = tf.matmul(inputs1, W)
    predict = tf.argmax(Qout, 1)

    nextQ = tf.placeholder(shape=[1, A], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))

    trainer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    update_model = trainer.minimize(loss)

    init = tf.global_variables_initializer()

    jList, rList = [], []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            s = env.reset()
            rAll = 0
            j = 0
            while j < 99:
                # env.render()
                j+=1
                # choose action greedily from the network output
                next_a, allQ = sess.run([predict, Qout], \
                    feed_dict={inputs1: s})  #Id[s:s+1]})
                #exploration sample
                if np.random.rand(1) < e:
                    next_a[0] = env.action_space.sample()
                next_state, r, d, info = env.step(next_a[0])

                ### get Q" values with next_state through the network
                Q1 = sess.run(Qout, feed_dict={inputs1: Id[next_state:next_state+1]})
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, next_a[0]] = r + y * maxQ1

                # train the model
                _ = sess.run([update_model], \
                    feed_dict={inputs1: Id[s:s+1], nextQ: targetQ})

                # count the reward
                rAll += r
                s = next_state
                if d: #end state
                  e = 1 / ((i / 50) + 10)
                  break
            if (i % verbosity) == 0:
                # env.render()
                print('Mean Return on iteration {} = {:.3f}' \
                      .format(i, sum(rList) / (i + 1)))
            jList.append(j)
            rList.append(rAll)
    return jList, rList, num_episodes