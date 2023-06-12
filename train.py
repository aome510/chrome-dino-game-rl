import enum
from itertools import count
import os
from PIL import Image
import gymnasium as gym
import torchvision.transforms as transforms
from torchvision.utils import torch
import random
import math
import envs as _
import numpy as np
import datetime

from model import Model

# A majority of the codes in this file is based on Pytorch's DQN tutorial [1]
# [1]: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


class HyperParameters:
    lr = 1e-4
    batch_size = 64

    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000


env = gym.make("Env-v0", render_mode="rgb_array")

# define the RL model
input_shape = (128, 256)
output_shape = env.action_space.n
policy_net = Model(input_shape, output_shape)
target_net = Model(input_shape, output_shape)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize(input_shape, antialias=True)]
)

num_episodes = 10
steps_done = 0


def select_action(state: torch.Tensor) -> torch.Tensor:
    global steps_done

    sample = random.random()
    eps_threshold = HyperParameters.eps_end + (
        HyperParameters.eps_start - HyperParameters.eps_end
    ) * math.exp(-1.0 * steps_done / HyperParameters.eps_decay)

    steps_done += 1

    # determine whether to explore or exploit with the eps_threshold
    if sample > eps_threshold:
        # exploit
        return policy_net(state).max(1)[1]
    else:
        # explore
        return torch.tensor(env.action_space.sample())


def save_experiment_results(obs_arr: list[np.ndarray]):
    if not os.path.exists("results"):
        os.mkdir("results")

    folder_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")

    folder_path = os.path.join("results", folder_name)
    if os.path.exists(folder_path):
        os.rmdir(folder_path)
    os.mkdir(folder_path)

    for i, obs in enumerate(obs_arr):
        img = Image.fromarray(obs, "RGB")
        img.save(os.path.join(folder_path, f"episode-{i}.png"))


obs_arr = []
for i_episode in range(num_episodes):
    steps_done = 0
    obs, _ = env.reset()
    state = transform(obs).unsqueeze(0)

    print(state.size())

    for t in count():
        action = select_action(state)
        obs, reward, terminated, truncated, _ = env.step(action.item())

        done = terminated or truncated
        if done:
            print(f"Done in {t} steps")
            obs_arr.append(obs)
            break


save_experiment_results(obs_arr)

env.close()
