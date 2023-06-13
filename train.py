from collections import deque, namedtuple
from itertools import count
import os
from PIL import Image
import gymnasium as gym
import torchvision.transforms as transforms
from torchvision.utils import torch
import torch.nn as nn
import random
import envs as _
import numpy as np
import datetime
import shutil

from model import Model

# A majority of the codes in this file is based on Pytorch's DQN tutorial [1]
# [1]: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # type: ignore


class HP:
    lr = 1e-4
    batch_size = 32

    replay_size = 10_000

    sync_frame_freq = 1_000

    input_shape = (128, 256)

    gamma = 0.99

    eps_start = 1.0
    eps_end = 0.01
    eps_decay = 1_000


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def get_eps_threshold(steps_done: int) -> float:
    return HP.eps_end + (HP.eps_start - HP.eps_end) * np.exp(
        -1.0 * steps_done / HP.eps_decay
    )


def select_action(
    state: torch.Tensor, policy_net: Model, steps_done: int
) -> torch.Tensor:
    eps_threshold = get_eps_threshold(steps_done)

    sample = random.random()

    # determine whether to explore or exploit with the eps_threshold
    if sample > eps_threshold:
        # exploit
        with torch.no_grad():
            return policy_net(state.unsqueeze(0)).max(dim=1)[1][0]
    else:
        # explore
        return torch.tensor(env.action_space.sample(), device=device)


def save_obs_result(episode_i: int, obs_arr: list[np.ndarray], folder_path: str):
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


def optimize_model(
    optimizer: torch.optim.Optimizer,
    replay_memory: ReplayMemory,
    policy_net: Model,
    target_net: Model,
):
    if len(replay_memory) < HP.batch_size:
        return

    transitions = replay_memory.sample(HP.batch_size)
    # Convert batch-array of Transitions to a Transition of batch-arrays
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        dtype=torch.bool,
        device=device,
    )
    non_final_next_state_batch = torch.stack(
        [s for s in batch.next_state if s is not None]
    )
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute batch "Q(s, a)"
    # The model returns "Q(s)", then we select the columns of actions taken.
    state_action_values = policy_net(state_batch).gather(
        1, action_batch.reshape((-1, 1))
    )

    # Compute batch "max_{a'} Q(s', a')"
    next_state_values = torch.zeros(HP.batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_state_batch).max(
            1
        )[0]
    expected_state_action_values = (next_state_values * HP.gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    print(f"Loss: {loss.item()}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()


if __name__ == "__main__":
    # Initialize the gym environment
    env = gym.make("Env-v0", render_mode="rgb_array", game_mode="train")

    # Define the RL model
    input_shape = HP.input_shape
    output_shape = env.action_space.n  # type: ignore

    policy_net = Model(output_shape).to(device)
    target_net = Model(output_shape).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    replay_memory = ReplayMemory(HP.replay_size)

    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=HP.lr, amsgrad=True)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(input_shape, antialias=True)]  # type: ignore
    )

    folder_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
    folder_path = os.path.join("results", folder_name)
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

    steps_done = 0

    for episode_i in range(1000):
        obs, _ = env.reset()
        state = transform(obs).to(device)

        obs_arr = []
        for t in count():
            action = select_action(state, policy_net, steps_done)

            obs, reward, terminated, *_ = env.step(action.item())
            obs_arr.append(obs)

            next_state = None
            if not terminated:
                next_state = transform(obs).to(device)

            replay_memory.push(
                state, action, next_state, torch.tensor(reward, device=device)
            )

            if next_state is not None:
                state = next_state

            # Perform one step of model optimization
            if steps_done % HP.sync_frame_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            optimize_model(optimizer, replay_memory, policy_net, target_net)

            if terminated:
                print(f"{episode_i} episode, done in {t+1} steps")
                break

            steps_done += 1

        if episode_i % 50 == 0:
            save_obs_result(episode_i, obs_arr, folder_path)

    env.close()
