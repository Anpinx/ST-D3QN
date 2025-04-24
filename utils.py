import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
from utils import *
from A_star import *
import math
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=2000)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/DQN/')
parser.add_argument('--ckpt_dir_pkl_data', type=str, default='./')

parser.add_argument('--ckpt_dir_pkl_ST_D3QN_dec', type=str, default='./data/ST_D3QN_dec/')
parser.add_argument('--ckpt_dir_pkl_ST_D3QN', type=str, default='./data/ST_D3QN/')
parser.add_argument('--ckpt_dir_pkl_D3QN', type=str, default='./data/D3QN/')
parser.add_argument('--ckpt_dir_pkl_DQN', type=str, default='./data/DQN/')
parser.add_argument('--ckpt_dir_pkl_DDQN', type=str, default='./data/DDQN/')

parser.add_argument('--reward_path', type=str, default='./output_images/reward.png')
parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')
parser.add_argument('--total_rewards_path', type=str, default='./output_images/total_rewards.png')
args = parser.parse_args()

def create_directory(directory, sub_dirs=[]):
    if not os.path.exists(directory):
        os.makedirs(directory)
    for sub_dir in sub_dirs:
        path = os.path.join(directory, sub_dir)
        if not os.path.exists(path):
            os.makedirs(path)

def ensure_dir_exists(directory):
    """如果目录不存在，则创建目录"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"目录 {directory} 已创建")
    else:
        print(f"目录 {directory} 已存在")
