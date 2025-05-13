import numpy as np
import pandas as pd
import random as rn
import os
import gym
import matplotlib.pyplot as plt
import Environment.environment
import drac
import torch
import argparse

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Pendulum environment')
parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')
parser.add_argument('--action_dim', type=int, default=1, help='action_dim')
parser.add_argument('--state_dim', type=int, default=26, help='state_dim')
parser.add_argument('--action_limit', type=int, default=7.6, help='action limit')
parser.add_argument('--hidden_sizes', type=int, default=256, help='hidden_sizes')
parser.add_argument('--T_horizon', type=int, default=30, help='rollout')
parser.add_argument('--train_step', type=int, default=11, help='train_step')
parser.add_argument('--mode', type=str, default='train', help='train')
parser.add_argument('--speed_range', type=int, default=15.0, help='speed range')
parser.add_argument('--save_dir_model', type=str, default='model/', help='the path to save models')
parser.add_argument('--save_dir_data', type=str, default='result/', help='the path to save data')
parser.add_argument('--save_dir_train_data', type=str, default='train/', help='the path to save training data')
args = parser.parse_args()

# Set the environment
env = gym.make('traffic-v0')

# Set a random seed
env.seed(args.seed)
np.random.seed(args.seed)
rn.seed(args.seed)
torch.manual_seed(args.seed)

def train():
    env.start(gui=True)
    model_v = drac.Agent(args.state_dim, args.action_dim, args.action_limit, args.hidden_sizes)

    if not os.path.exists(args.save_dir_model):
        os.mkdir(args.save_dir_model)
    model_v.train()
    print("The model is training")

    score = 0.0
    total_reward = []
    episode = []
    print_interval = 10

    v = []
    v_epi = []
    v_epi_mean = []

    cn = 0.0
    cn_epi = []

    sn = 0.0
    sn_epi = []

    for n_epi in range(args.train_step):
        s = env.reset()

        for t in range(args.T_horizon):
            s = np.array(s, dtype=float)
            a, a_pi = model_v.select_a(s, args.mode)
            r, next_s, done, r_, c_, info = env.step(a)

            model_v.replay_buffer.add(s, a_pi, r/10, next_s, done, c_)
            s = next_s

            score += r
            v.append(s[24]*args.speed_range)
            v_epi.append(s[24]*args.speed_range)
            xa = info[0]
            ya = info[1]

            if args.mode == "train" and n_epi > 10:
                model_v.train_model()

            if done:
                break

        if done is True:
            cn += 1

        if xa < -50 and ya > 4.0 and done is False:
            sn += 1

        if (n_epi+1) % print_interval == 0:
            print("# of episode :{}, avg score_v : {:.1f}".format(n_epi+1, score / print_interval))

            episode.append(n_epi+1)
            total_reward.append(score / print_interval)
            cn_epi.append(cn/print_interval)
            sn_epi.append(sn/print_interval)
            print("######cn & sn rate:", cn/print_interval, sn/print_interval)

            v_mean = np.mean(v_epi)
            v_epi_mean.append(v_mean)

            v_epi = []
            score = 0.0
            cn = 0.0
            sn = 0.0

        if args.mode == "train" and (n_epi+1) % 10 == 0 and n_epi > 1800:
            model_v.save_model(n_epi+1, args.save_dir_model)
            print("#The models are saved!#")

    plt.plot(episode, total_reward)
    plt.xlabel('episode')
    plt.ylabel('total_reward')
    plt.show()

    df = pd.DataFrame([])
    df["n_epi"] = episode
    df["total_reward"] = total_reward
    df["v_epi_mean"] = v_epi_mean
    df["cn_epi"] = cn_epi
    df["sn_epi"] = sn_epi

    if not os.path.exists(args.save_dir_data):
        os.mkdir(args.save_dir_data)
    train_data_path = os.path.join(args.save_dir_data, args.save_dir_train_data)
    if not os.path.exists(train_data_path):
        os.mkdir(train_data_path)

    df.to_csv('./' + train_data_path + '/train.csv', index=0)
    env.close()


if __name__ == "__main__":
    train()