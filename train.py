from PIL import Image
from collections import deque, namedtuple
import datetime
from itertools import count
import os
import shutil
import gymnasium as gym
from torchvision.utils import torch
import torch.nn as nn
import random
import envs
import numpy as np

from model import DQN

# A majority of the codes in this file is based on Pytorch's DQN tutorial [1]
# [1]: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminated")
)


class MemoryReplay(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Trainer:
    def __init__(
        self,
        env: envs.Wrapper,
        policy_net: DQN,
        target_net: DQN,
        n_episodes=1_000,
        lr=1e-4,
        batch_size=32,
        replay_size=10_000,  # experience replay's buffer size
        learning_start=10_000,  # number of frames before learning starts
        target_update_freq=1_000,  # number of frames between every target network update
        optimize_freq=1,
        gamma=0.99,  # reward decay factor
        # explore/exploit eps-greedy policy
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=10_000,
    ):
        self.env = env

        self.policy_net = policy_net.to(self.device)
        self.target_net = target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory_replay = MemoryReplay(replay_size)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # type: ignore

        self.n_steps = 0

        self.n_episodes = n_episodes
        self.batch_size = batch_size

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

        self.learning_start = learning_start
        self.target_update_freq = target_update_freq
        self.optimize_freq = optimize_freq

        self.gamma = gamma

        self._get_eps = lambda n_steps: eps_end + (eps_start - eps_end) * np.exp(
            -1.0 * n_steps / eps_decay
        )

        # Initialize folder to save training results
        folder_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        folder_path = os.path.join("results", folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        self.folder_path = folder_path

    def _select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select the next action given the current state following the eps-greedy policy"""
        eps = self._get_eps(self.n_steps)

        if random.random() > eps:
            # exploit
            with torch.no_grad():
                return self.policy_net(state.unsqueeze(0)).max(dim=1)[1][0]
        else:
            # explore
            return torch.tensor(self.env.action_space.sample(), device=self.device)

    def _optimize(self):
        transitions = self.memory_replay.sample(self.batch_size)

        # Convert batch-array of Transitions to a Transition of batch-arrays
        batch = Transition(*zip(*transitions))

        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        next_state_batch = torch.stack(batch.next_state)
        reward_batch = torch.stack(batch.reward)
        terminated_batch = torch.tensor(
            batch.terminated, device=self.device, dtype=torch.float
        )

        # Compute batch "Q(s, a)"
        # The model returns "Q(s)", then we select the columns of actions taken.
        Q_values = (
            self.policy_net(state_batch)
            .gather(1, action_batch.unsqueeze(-1))
            .squeeze(-1)
        )

        # Compute batch "max_{a'} Q(s', a')"
        with torch.no_grad():
            next_Q_values = self.target_net(next_state_batch).max(1)[0]
        expected_Q_values = (
            1.0 - terminated_batch
        ) * next_Q_values * self.gamma + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q_values, expected_Q_values)

        # print(f"Loss: {loss.item()}")

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        for episode_i in range(self.n_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, device=self.device)

            total_reward = 0.0

            for t in count():
                self.n_steps += 1

                action = self._select_action(state)

                next_state, reward, terminated, *_ = self.env.step(
                    envs.Action(action.item())
                )
                next_state = torch.tensor(next_state, device=self.device)

                total_reward += float(reward)

                self.memory_replay.push(
                    state,
                    action,
                    next_state,
                    torch.tensor(reward, device=self.device),
                    terminated,
                )

                # Synchronize the target network with the policy network
                if (
                    self.n_steps > self.learning_start
                    and self.n_steps % self.target_update_freq == 0
                ):
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # Optimize the policy network
                if (
                    self.n_steps > self.learning_start
                    and self.n_steps % self.optimize_freq == 0
                ):
                    self._optimize()

                if terminated:
                    print(
                        f"{episode_i} episode, done in {t+1} steps, total reward: {total_reward}"
                    )
                    break
                else:
                    state = next_state

            if episode_i % 50 == 0:
                self.save_obs_result(episode_i, self.env.frames)
                self.save_model_weights(episode_i)

        self.env.close()

    def save_obs_result(self, episode_i: int, obs_arr: list[np.ndarray]):
        frames = [Image.fromarray(obs, "RGB") for obs in obs_arr]
        file_path = os.path.join(self.folder_path, f"episode-{episode_i}.gif")

        frames[0].save(
            file_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=100,
            loop=0,
        )

    def save_model_weights(self, episode_i: int):
        file_path = os.path.join(self.folder_path, f"model-{episode_i}.pth")
        torch.save(self.policy_net, file_path)


if __name__ == "__main__":
    env = gym.make("Env-v0", render_mode="rgb_array", game_mode="train")
    env = envs.Wrapper(env, k=4)

    # Define the DQN networks
    obs_space = env.observation_space.shape
    assert obs_space is not None
    in_channels = obs_space[0]
    out_channels = self.env.action_space.n  # type: ignore

    policy_net = DQN(in_channels, out_channels)
    target_net = DQN(in_channels, out_channels)

    trainer = Trainer(env, policy_net, target_net)
    trainer.train()
