import gymnasium as gym
import pygame
import envs

env = gym.make("Env-v0", render_mode="human")
env.reset()

total_reward = 0.0
while True:
    userInput = pygame.key.get_pressed()
    action = envs.Action.STAND
    if userInput[pygame.K_UP] or userInput[pygame.K_SPACE]:
        action = envs.Action.JUMP
    elif userInput[pygame.K_DOWN]:
        action = envs.Action.DUCK

    _, reward, terminated, _, _ = env.step(action)

    print(reward)

    total_reward += float(reward)
    if terminated:
        break

print(f"Total reward: {total_reward}")

env.close()
