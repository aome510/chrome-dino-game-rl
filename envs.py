import enum
import os
from typing import Any, TypedDict
import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.registration import register


class Observation(TypedDict):
    agent: np.ndarray


class Action(enum.Enum):
    UP = 0
    DOWN = 1


class RenderMode(str, enum.Enum):
    HUMAN = "human"
    RGB = "rgb_array"


class Assets:
    def __init__(self):
        self.track = pygame.image.load(os.path.join("assets", "Track.png"))


class Env(gym.Env):
    metadata = {"render_fps": 10, "render_modes": [RenderMode.HUMAN, RenderMode.RGB]}

    def __init__(
        self,
        render_mode: RenderMode | None,
        window_size: tuple[int, int] = (1024, 512),
    ) -> None:
        self.assets = Assets()

        self.render_mode = render_mode
        self.window_size = window_size

        self.score = 0
        self.speed = 20

        self.track_offset_x = 0
        self.track_w = self.assets.track.get_width()
        self.track_h = self.assets.track.get_height()

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, 200, shape=(2,), dtype=np.int32),
            }
        )

        self.window = None
        self.clock = None

        if self.render_mode == RenderMode.HUMAN:
            pygame.init()
            pygame.display.init()

            self.window = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()

        super().__init__()

    def _get_obs(self) -> Observation:
        return {"agent": np.zeros((2,), dtype=np.int32)}

    def _get_info(self) -> dict:
        return {}

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict]:
        super().reset(seed=seed, options=options)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict]:
        terminated = False
        reward = 0.0
        obs = self._get_obs()
        info = self._get_info()

        self.score += 1
        if self.score % 100 == 0:
            self.speed += 1
        # Negative offset to slide the running track image to the left
        self.track_offset_x -= self.speed

        if self.render_mode == RenderMode.HUMAN:
            self._render_frame()

        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode == RenderMode.RGB:
            return self._render_frame()

    def _render_track(self, canvas: pygame.Surface):
        # Render the running track image moved to the left by `track_offset_x`
        canvas.blit(
            self.assets.track,
            (self.track_offset_x, self.window_size[1] - self.track_h),
        )

        # If the moved image doesn't cover the screen, render the left space
        # with a second image to create a "loop" effect.
        if self.track_offset_x + self.track_w < self.window_size[0]:
            # Find the starting position to render the second image
            # -10 here because the running track image starts with a small gap
            start_x = self.track_offset_x + self.track_w - 10
            canvas.blit(
                self.assets.track,
                (start_x, self.window_size[1] - self.track_h),
            )

            # If the starting position is negative, which means the moved image
            # doesn't intersect with the screen, start rendering a new image with
            # a new offset equal to the starting position
            if start_x <= 0:
                self.track_offset_x = start_x

    def _render_frame(self):
        canvas = pygame.Surface(self.window_size)
        canvas.fill((255, 255, 255))

        self._render_track(canvas)

        if self.window is not None and self.clock is not None:
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


register(
    id="Env-v0",
    entry_point="envs:Env",
    max_episode_steps=300,
)
