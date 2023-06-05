from abc import ABC
import enum
import os
from typing import Any, TypedDict
import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.registration import register

WINDOW_SIZE = (1024, 512)


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

        # dino assets
        self.dino_run1 = pygame.image.load(os.path.join("assets", "DinoRun1.png"))
        self.dino_run2 = pygame.image.load(os.path.join("assets", "DinoRun2.png"))


class EnvObject(ABC):
    def __init__(self, assets: Assets, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def render(self, canvas: pygame.Surface, *args, **kwargs):
        pass


class Dino(EnvObject):
    def __init__(self, assets: Assets):
        self._run_assets = [assets.dino_run1, assets.dino_run2]

    def step(self):
        self._run_assets[0], self._run_assets[1] = (
            self._run_assets[1],
            self._run_assets[0],
        )

    def render(self, canvas: pygame.Surface):
        canvas.blit(
            self._run_assets[0],
            (50, WINDOW_SIZE[1] - self._run_assets[0].get_height()),
        )


class Track(EnvObject):
    def __init__(self, assets: Assets):
        self._asset = assets.track

        self._track_offset_x = 0
        self._track_w = self._asset.get_width()
        self._track_h = self._asset.get_height()

    def step(self, speed: int):
        # Negative offset to slide the running track image to the left
        self._track_offset_x -= speed

    def render(self, canvas: pygame.Surface):
        # Render the running track image moved to the left by `track_offset_x`
        canvas.blit(
            self._asset,
            (self._track_offset_x, WINDOW_SIZE[1] - self._track_h),
        )

        # If the moved image doesn't cover the screen, render the left space
        # with a second image to create a "loop" effect.
        if self._track_offset_x + self._track_w < WINDOW_SIZE[0]:
            # Find the starting position to render the second image
            # -10 here because the running track image starts with a small gap
            start_x = self._track_offset_x + self._track_w - 10
            canvas.blit(
                self._asset,
                (start_x, WINDOW_SIZE[1] - self._track_h),
            )

            # If the starting position is negative, which means the moved image
            # doesn't intersect with the screen, start rendering a new image with
            # a new offset equal to the starting position
            if start_x <= 0:
                self._track_offset_x = start_x


class Env(gym.Env):
    metadata = {"render_fps": 10, "render_modes": [RenderMode.HUMAN, RenderMode.RGB]}

    def __init__(
        self,
        render_mode: RenderMode | None,
        window_size: tuple[int, int] = (1024, 512),
    ) -> None:
        # Initialize `gym.Env` required fields
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, 200, shape=(2,), dtype=np.int32),
            }
        )

        # Other (private) fields
        self._assets = Assets()

        self._score = 0
        self._speed = 20

        # Initialize environment's objects' states
        self._track = Track(self._assets)
        self._dino = Dino(self._assets)

        # Initialize `pygame` data
        self._window = None
        self._clock = None

        if self.render_mode == RenderMode.HUMAN:
            pygame.init()
            pygame.display.init()

            self._window = pygame.display.set_mode(WINDOW_SIZE)
            self._clock = pygame.time.Clock()

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

        self._score += 1
        if self._score % 100 == 0:
            self._speed += 1

        self._track.step(self._speed)
        self._dino.step()

        if self.render_mode == RenderMode.HUMAN:
            self._render_frame()

        return obs, reward, terminated, False, info

    def render(self):
        if self.render_mode == RenderMode.RGB:
            return self._render_frame()

    def _render_frame(self):
        canvas = pygame.Surface(WINDOW_SIZE)
        canvas.fill((255, 255, 255))

        self._track.render(canvas)
        self._dino.render(canvas)

        if self._window is not None and self._clock is not None:
            self._window.blit(canvas, canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

            self._clock.tick(self.metadata["render_fps"])
        else:
            # return the canvas as a rgb array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()


register(
    id="Env-v0",
    entry_point="envs:Env",
    max_episode_steps=300,
)
