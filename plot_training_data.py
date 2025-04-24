import os
import pickle
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./data/ST_D3QN/')
parser.add_argument('--experiment_num', type=int, default=0)
args = parser.parse_args()

def main():
    data_file = os.path.join(args.data_dir, 'training_data_{}.pkl'.format(args.experiment_num))
    if not os.path.isfile(data_file):
        print('Data file not found: {}'.format(data_file))
        return

    with open(data_file, 'rb') as f:
        data = pickle.load(f)

    episodes = data['episodes']
    total_rewards = data['total_rewards']
    avg_rewards = data['avg_rewards']
    epsilon_history = data['epsilon_history']
    episode_losses = data['episode_losses']

    # 绘制总奖励（每个Episode的Reward）
    plt.figure(figsize=(10,6))
    plt.plot(episodes, total_rewards, label='Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制平均奖励（过去100个Episode的平均Reward）
    plt.figure(figsize=(10,6))
    plt.plot(episodes, avg_rewards, label='Average Reward (Last 100 Episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward (Last 100 Episodes)')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制epsilon的变化
    plt.figure(figsize=(10,6))
    plt.plot(episodes, epsilon_history, label='Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay over Episodes')
    plt.legend()
    plt.grid()
    plt.show()

    # 绘制每个Episode的Loss
    plt.figure(figsize=(10,6))
    plt.plot(episodes, episode_losses, label='Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss per Episode')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    main()
