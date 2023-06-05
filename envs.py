from abc import ABC
import enum
import os
from typing import Any, TypedDict
import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.registration import register

WINDOW_SIZE = (1024, 512)
JUMP_DURATION = 11
JUMP_VEL = 100
BASE_OBSTACLE_SPAWN_PROB = 0.3
RENDER_FPS = 15


class Observation(TypedDict):
    agent: np.ndarray


class Action(int, enum.Enum):
    STAND = 0
    JUMP = 1
    DUCK = 2


class DinoState(int, enum.Enum):
    STAND = 0
    JUMP = 1
    DUCK = 2


class RenderMode(str, enum.Enum):
    HUMAN = "human"
    RGB = "rgb_array"


class Assets:
    def __init__(self):
        self.track = pygame.image.load(os.path.join("assets", "Track.png"))

        # dino assets
        self.dino_runs = [
            pygame.image.load(os.path.join("assets", "DinoRun1.png")),
            pygame.image.load(os.path.join("assets", "DinoRun2.png")),
        ]
        self.dino_ducks = [
            pygame.image.load(os.path.join("assets", "DinoDuck1.png")),
            pygame.image.load(os.path.join("assets", "DinoDuck2.png")),
        ]
        self.dino_jump = pygame.image.load(os.path.join("assets", "DinoJump.png"))

        # cactus
        self.cactuses = [
            pygame.image.load(os.path.join("assets", "LargeCactus1.png")),
            pygame.image.load(os.path.join("assets", "LargeCactus2.png")),
            pygame.image.load(os.path.join("assets", "LargeCactus3.png")),
            pygame.image.load(os.path.join("assets", "SmallCactus1.png")),
            pygame.image.load(os.path.join("assets", "SmallCactus2.png")),
            pygame.image.load(os.path.join("assets", "SmallCactus3.png")),
        ]


class EnvObject(ABC):
    def __init__(self, assets: Assets, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def render(self, canvas: pygame.Surface, *args, **kwargs):
        pass


class Cactus(EnvObject):
    def __init__(self, assets: Assets, id: int):
        self._asset = assets.cactuses[id]
        self._x = WINDOW_SIZE[0]

    def step(self, speed: int):
        self._x -= speed

    def is_inside(self) -> bool:
        return self._x + self._asset.get_width() > 0

    def render(self, canvas: pygame.Surface):
        canvas.blit(
            self._asset,
            (self._x, WINDOW_SIZE[1] - self._asset.get_height() - 7),
        )


class Dino(EnvObject):
    def __init__(self, assets: Assets):
        self._run_assets = assets.dino_runs
        self._duck_assets = assets.dino_ducks
        self._jump_asset = assets.dino_jump

        self._jump_timer = 0
        self._state = DinoState.STAND

    def step(self, action: Action):
        self._run_assets[0], self._run_assets[1] = (
            self._run_assets[1],
            self._run_assets[0],
        )
        self._duck_assets[0], self._duck_assets[1] = (
            self._duck_assets[1],
            self._duck_assets[0],
        )

        # Check if the jump animation is finished
        if self._state == DinoState.JUMP:
            self._jump_timer -= 1
            if self._jump_timer < 0:
                self._state = DinoState.STAND

        # If dino is not jumping, transition to a new state based on the action
        if self._state != DinoState.JUMP:
            match action:
                case Action.STAND:
                    self._state = DinoState.STAND
                case Action.JUMP:
                    self._state = DinoState.JUMP
                    self._jump_timer = JUMP_DURATION
                case Action.DUCK:
                    self._state = DinoState.DUCK

    def _get_jump_offset(self) -> int:
        a = -JUMP_VEL / (JUMP_DURATION / 2)
        t = JUMP_DURATION - self._jump_timer
        d = int(JUMP_VEL * t + 0.5 * a * (t**2))
        return d

    def render(self, canvas: pygame.Surface):
        match self._state:
            case DinoState.STAND:
                canvas.blit(
                    self._run_assets[0],
                    (50, WINDOW_SIZE[1] - self._run_assets[0].get_height()),
                )
            case DinoState.JUMP:
                canvas.blit(
                    self._jump_asset,
                    (
                        50,
                        WINDOW_SIZE[1]
                        - self._get_jump_offset()
                        - self._jump_asset.get_height(),
                    ),
                )
            case DinoState.DUCK:
                canvas.blit(
                    self._duck_assets[0],
                    (50, WINDOW_SIZE[1] - self._duck_assets[0].get_height()),
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
    metadata = {
        "render_fps": RENDER_FPS,
        "render_modes": [RenderMode.HUMAN, RenderMode.RGB],
    }

    def __init__(
        self,
        render_mode: RenderMode | None,
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

        self._frame = 0
        self._speed = 20
        self._spawn_prob = BASE_OBSTACLE_SPAWN_PROB

        # Initialize environment's objects' states
        self._track = Track(self._assets)
        self._dino = Dino(self._assets)
        self._obstacles: list[Cactus] = []

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

        self._frame += 1
        # increase the difficulty of the game every 20 frames
        if self._frame % 20 == 0:
            self._speed += 1
            self._spawn_prob = min(0.7, self._spawn_prob * 1.01)

            print(f"New speed: {self._speed}, new prob: {self._spawn_prob}")

        self._track.step(self._speed)
        self._dino.step(action)
        for o in self._obstacles:
            o.step(self._speed)

        # Filter inside obstacles after each step
        self._obstacles = [o for o in self._obstacles if o.is_inside()]

        # Should we spawn a new obstacle?
        if (
            self._frame % 20 == 0
            and self.np_random.choice(2, 1, p=[1 - self._spawn_prob, self._spawn_prob])[
                0
            ]
        ):
            id = self.np_random.choice(len(self._assets.cactuses), 1)[0]
            self._obstacles.append(Cactus(self._assets, id))

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
        for o in self._obstacles:
            o.render(canvas)

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
