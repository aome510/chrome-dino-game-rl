import argparse
from PIL import Image
import gymnasium as gym
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import torch
import envs
from play import play_with_model
from tqdm import tqdm

if __name__ == "__main__":
    if not os.path.exists("experiments"):
        os.mkdir("experiments")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # type: ignore

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--result_path")
    parser.add_argument("-n", "--n_runs", type=int, default=10)
    parser.add_argument("-s", "--seed", type=int, default=510)
    parser.add_argument("-e", "--episodes", type=int, nargs="+")
    parser.add_argument("--save_play", action="store_true")
    parser.add_argument("--save_plot", action="store_true")
    args = parser.parse_args()

    env = gym.make("Env-v0", render_mode="rgb_array")
    env = envs.Wrapper(env)

    progress_bar = tqdm(total=args.n_runs * len(args.episodes), desc="Progress")

    avg_rewards = []
    avg_frame_counts = []
    # Run experiments on models after training for a specific number of episodes
    for milestone in args.episodes:
        model_path = os.path.join(args.result_path, f"model-{milestone}.pth")
        policy_net = torch.load(model_path).to(device)
        policy_net.eval()

        rewards = []
        frame_counts = []
        for i in range(args.n_runs):
            total_reward = play_with_model(
                env, policy_net, device, seed=(i + 1) * args.seed
            )
            rewards.append(total_reward)
            frame_counts.append(len(env.frames))

            if args.save_play and i == 0:
                # Only save play of the first run
                frames = [Image.fromarray(obs, "RGB") for obs in env.frames]

                frames[0].save(
                    f"experiments/model-{milestone}.gif",
                    save_all=True,
                    append_images=frames[1:],
                    optimize=True,
                    duration=100,
                    loop=0,
                )

            progress_bar.update(1)

        avg_rewards.append(np.array(rewards, dtype=np.float32).mean())
        avg_frame_counts.append(np.array(frame_counts, dtype=np.float32).mean())

    # Draw plots to see average total rewards and average number of frames
    # each model got over `n_runs` runs.
    fig, axs = plt.subplots(1, 2)

    axs[0].plot(
        args.episodes,
        avg_rewards,
        marker=".",
        markeredgecolor="red",
        markerfacecolor="red",
    )
    axs[0].set_title("Average total reward")
    axs[0].set_xlabel("Episodes")
    axs[0].set_ylabel("Reward")

    axs[1].plot(
        args.episodes,
        avg_frame_counts,
        marker=".",
        markeredgecolor="red",
        markerfacecolor="red",
    )
    axs[1].set_title("Average number of frames")
    axs[1].set_xlabel("Episodes")
    axs[1].set_ylabel("Frames")

    plt.subplots_adjust(wspace=0.5)
    if args.save_plot:
        plt.savefig("experiments/plot.png")
    plt.show()
