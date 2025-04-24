import os
import pickle
import matplotlib.pyplot as plt
plt.ion()

from Env_ST_D3QN import Environment
import numpy as np
import argparse
from ST_D3QN import ST_D3QN

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/ST_D3QN/')
parser.add_argument('--ckpt_dir_pkl', type=str, default='./data/ST_D3QN/')
args = parser.parse_args()

def test():
    env = Environment()
    agent = ST_D3QN(
        alpha=0.001,
        state_dim=env.state_space,
        action_dim=env.action_space,
        fc1_dim=256,
        fc2_dim=256,
        ckpt_dir=args.ckpt_dir,
        gamma=0.98,
        tau=0.001,
        epsilon=0.0,
        eps_end=0.0,
        max_size=1000000,
        batch_size=64
    )
    agent.load_models()
    total_rewards = []

    num_episodes = 100

    for episode in range(num_episodes):
        episode_reward = 0
        env.reset()
        done = False
        observation = env.get_state()
        while not done:
            action = agent.choose_action(observation, range(env.action_space), isTrain=False)
            observation_, reward, done, info = env.step(action)
            episode_reward += reward
            observation = observation_

            env.render(episode)

            if done:
                break

        total_rewards.append(episode_reward)
        print('Test Episode:{} Reward:{}'.format(episode + 1, episode_reward))

    avg_reward = np.mean(total_rewards)
    print('Average Reward over {} episodes: {}'.format(num_episodes, avg_reward))

    plt.ioff()
    plt.show()

if __name__ == '__main__':
    test()