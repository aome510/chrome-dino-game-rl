import gymnasium as gym

env = gym.make("Env-v0", render_mode="human")
env.reset()

for _ in range(1000):
    ...

env.close()
