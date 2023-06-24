import argparse
import gymnasium as gym
from matplotlib import os
from torchvision.utils import torch
import envs
from play import play_with_model

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_path")
    parser.add_argument("-s", "--seed", type=int, default=510)
    args = parser.parse_args()

    env = gym.make("Env-v0", render_mode="rgb_array")
    env = envs.Wrapper(env)

    path = args.result_path
    seed = args.seed

    chosen_episode_milestones = range(50, 1000, 50)

    for milestone in chosen_episode_milestones:
        model_path = os.path.join(path, f"model-{milestone}.pth")
        policy_net = torch.load(model_path).to(device)
        policy_net.eval()

        total_reward = play_with_model(env, policy_net, device, seed=seed)

        print(
            f"Milestone: {milestone}: total reward: {total_reward}, number of frames: {len(env.frames)}"
        )
