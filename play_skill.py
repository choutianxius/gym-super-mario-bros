"""
Generate the `clear_times.png` and `mean_scores.png` metrics figures
"""

import torch
from agent import Mario, MarioNet
from environment import wrapped

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm

import os


env = wrapped('rgb_array')


def test_model_skill(model_idx=96):
    save_path = f'save/2023-05-21T07-54-29/mario_net_{model_idx}.chkpt'
    data = torch.load(save_path)
    state_dict = data.get('online_model')
    output_dim = data.get('output_dim')

    model = MarioNet(output_dim=output_dim)
    model.load_state_dict(state_dict)
    model.eval()
    mario = Mario(action_dim=output_dim, model=model, enable_explore=True)

    scores = []
    clear_times = 0
    episodes = 100

    for episode in tqdm(range(episodes)):
        env.reset()
        state, reward, done, trunc, info = env.step(action=0)
        score = reward
        while not (done or trunc):
            action = mario.act(state)
            state, reward, done, trunc, info = env.step(action)
            score += reward
        if info['flag_get']:
            clear_times += 1
        scores.append(score)

    mean_score = np.mean(scores)
    return mean_score, clear_times


def main():
    if os.path.isfile('./mean_scores.npy'):
        mean_scores = np.load('mean_scores.npy')
        clear_times1 = np.load('clear_times.npy')
        model_indices = np.load('model_indices.npy')
    else:
        mean_scores = []
        clear_times1 = []
        model_indices = [6, 16, 26, 36, 46, 56, 66, 76, 86, 96]
        for idx in model_indices:
            mean_score, clear_times = test_model_skill(idx)
            mean_scores.append(mean_score)
            clear_times1.append(clear_times)
        data = {
            'model_indices': model_indices,
            'mean_scores': mean_scores,
            'clear_times': clear_times1,
        }
        for name in data.keys():
            np.save(name, data[name])

    fig, ax = plt.subplots()
    ax.plot(range(1, len(model_indices) + 1), mean_scores)
    ax.set_xticks(range(1, len(model_indices) + 1))
    ax.set_xticklabels([str(x)+'%' for x in range(10, 101, 10)])
    plt.savefig('./mean_scores.png', dpi=300)

    fig, ax = plt.subplots()
    ax.plot(range(1, len(model_indices) + 1), clear_times1)
    ax.set_xticks(range(1, len(model_indices) + 1))
    ax.set_xticklabels([str(x)+'%' for x in range(10, 101, 10)])
    ax.set_ylim(-0.5, np.max(clear_times1) + 5)
    ax.set_yticks(range(0, np.max(clear_times1) + 5, 5))
    plt.savefig('./clear_times.png', dpi=300)

    plt.close()


if __name__ == '__main__':
    main()
