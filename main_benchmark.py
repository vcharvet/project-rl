from q_table import q_table_learning, nn_learning
from utils import train_dqn

import numpy as np
import gym
import os
import argparse

"""
Benchmark main to test reinforcement withou humain supervision
"""
def main():
    parser = argparse.ArgumentParser(description='RL without humain supervision')
    parser.add_argument('--env', dest='env', type=str, default='FrozenLake-v0',
                        help='name of environment - must be gym-compatible')

    args = parser.parse_args()
    env_name = args.env

    env = gym.make(env_name)
    # env.render()
    # _, rList, num = q_table_learning(env)
    # _, rList, num = nn_learning(env)
    _, rList, num = train_dqn(env, img_shape=(210, 160, 3), render=False,
                              lr=0.001, batch_size=16, max_ep_length=250,
                              pre_train_episodes=25,
                              out_size=256)




    print('Mean return: {:.3f}'.format(sum(rList)/num))


    return 0



if __name__ == "__main__":
    main()
