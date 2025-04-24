import os
import pickle

import matplotlib.pyplot as plt

from Env_ST_D3QN import Environment
import numpy as np
import argparse
from utils import create_directory, ensure_dir_exists
from ST_D3QN import ST_D3QN

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=1000)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/ST_D3QN/')
parser.add_argument('--ckpt_dir_pkl', type=str, default='./data/ST_D3QN/')
args = parser.parse_args()

def main(numberOfExperiment):
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
        epsilon=0.99,
        eps_end=0.01,
        max_size=1000000,
        batch_size=64
    )
    create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, epsilon_history = [], [], []
    episode_losses = []

    steps = 0

    for episode in range(args.max_episodes):
        episode_reward = 0
        episode_loss = 0
        env.reset()
        done = False
        observation = env.get_state()
        while not done:
            action = agent.choose_action(observation, range(env.action_space), isTrain=True)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            steps += 1

            loss = agent.learn()
            if loss is not None:
                agent.decrement_epsilon(steps)
                episode_loss += loss

            episode_reward += reward
            observation = observation_

            env.render(episode)

        total_rewards.append(episode_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        epsilon_history.append(agent.epsilon)
        episode_losses.append(episode_loss)

        print('EP:{} Reward:{} Avg_reward:{} Epsilon:{}'.format(
            episode + 1, episode_reward, avg_reward, agent.epsilon
        ))

        if (episode + 1) % 100 == 0:
            agent.save_models()

    episodes = [i + 1 for i in range(args.max_episodes)]
    data = {
        'episodes': episodes,
        'total_rewards': total_rewards,
        'avg_rewards': avg_rewards,
        'epsilon_history': epsilon_history,
        'episode_losses': episode_losses
    }
    
    ensure_dir_exists(args.ckpt_dir_pkl)
    with open(args.ckpt_dir_pkl + 'training_data_{}.pkl'.format(numberOfExperiment), 'wb') as f:
        pickle.dump(data, f)

    plt.ioff()

if __name__ == '__main__':
    MAX = 50
    for i in range(0, MAX):
        print("Experiment:{}".format(i))
        main(i)