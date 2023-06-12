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
import multiprocessing

from model import Model

# A majority of the codes in this file is based on Pytorch's DQN tutorial [1]
# [1]: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


class HyperParameters:
    lr = 1e-4
    batch_size = 64

    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000


def get_eps_threshold(t: int) -> float:
    return HyperParameters.eps_end + (
        HyperParameters.eps_start - HyperParameters.eps_end
    ) * math.exp(-1.0 * t / HyperParameters.eps_decay)


def select_action(state: torch.Tensor, eps_threshold: float) -> torch.Tensor:
    sample = random.random()

    # determine whether to explore or exploit with the eps_threshold
    if sample > eps_threshold:
        # exploit
        return policy_net(state).max(1)[1]
    else:
        # explore
        return torch.tensor(env.action_space.sample())


def save_obs_result(data: tuple[int, list[np.ndarray], str]):
    episode_i, obs_arr, folder_path = data

    frames = [Image.fromarray(obs, "RGB") for obs in obs_arr]
    file_path = os.path.join(folder_path, f"episode-{episode_i}.gif")

    frames[0].save(
        file_path,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=100,
        loop=0,
    )


def save_experiment_results(episode_obs_arr: list[list[np.ndarray]]):
    folder_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    folder_path = os.path.join("results", folder_name)
    if os.path.exists(folder_path):
        os.rmdir(folder_path)
    os.makedirs(folder_path)

    # use multiprocessing to speed up the gif saving process
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    n_episodes = len(episode_obs_arr)
    pool.map(
        save_obs_result,
        zip(range(n_episodes), episode_obs_arr, [folder_path] * n_episodes),
    )


if __name__ == "__main__":
    # Initialize the gym environment
    env = gym.make("Env-v0", render_mode="rgb_array")

    # Define the RL model
    input_shape = (128, 256)
    output_shape = env.action_space.n
    policy_net = Model(input_shape, output_shape)
    target_net = Model(input_shape, output_shape)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(input_shape, antialias=True)]
    )

    episode_obs_arr = []
    for episode_i in range(10):
        steps_done = 0
        obs, _ = env.reset()
        state = transform(obs).unsqueeze(0)

        obs_arr = []
        for t in count():
            action = select_action(state, eps_threshold=get_eps_threshold(t))

            obs, reward, terminated, truncated, _ = env.step(action.item())

            obs_arr.append(obs)

            done = terminated or truncated
            if done:
                print(f"{episode_i} episode, done in {t} steps")
                break

        episode_obs_arr.append(obs_arr)

    save_experiment_results(episode_obs_arr)

    env.close()
