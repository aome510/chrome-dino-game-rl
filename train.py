import gymnasium as gym
from numpy import random
import envs

env = gym.make("Env-v0", render_mode="human")
env.reset()

for _ in range(100):
    env.step(random.choice(list(envs.Action)))

env.close()
