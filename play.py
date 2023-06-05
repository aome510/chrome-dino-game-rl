import gymnasium as gym
import pygame
import envs

env = gym.make("Env-v0", render_mode="human")
env.reset()

for _ in range(1000):
    userInput = pygame.key.get_pressed()
    action = envs.Action.STAND
    if userInput[pygame.K_UP] or userInput[pygame.K_SPACE]:
        action = envs.Action.JUMP
    elif userInput[pygame.K_DOWN]:
        action = envs.Action.DUCK

    _, _, terminated, _, _ = env.step(action)
    if terminated:
        break

env.close()
