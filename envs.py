from abc import ABC
import enum
import os
from typing import Any
import gymnasium as gym
import numpy as np
import pygame
from gymnasium.envs.registration import register

# environment's constants
WINDOW_SIZE = (1024, 512)
JUMP_DURATION = 12
JUMP_VEL = 100
OBSTACLE_MIN_CNT = 400
MAX_SPEED = 100
MAX_SPAWN_PROB = 0.7
BASE_SPAWN_PROB = 0.3
RENDER_FPS = 15


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

        # bird
        self.birds = [
            pygame.image.load(os.path.join("assets", "Bird1.png")),
            pygame.image.load(os.path.join("assets", "Bird2.png")),
        ]


class EnvObject(ABC):
    def __init__(self, assets: Assets, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def render(self, canvas: pygame.Surface, *args, **kwargs):
        pass


class Obstacle(EnvObject, ABC):
    def collide(self, other: pygame.Rect) -> bool:
        return False

    def is_inside(self) -> bool:
        return False


class Bird(Obstacle):
    def __init__(self, assets: Assets):
        self._assets = assets.birds
        self._rect = self._assets[0].get_rect()
        self._rect.x = WINDOW_SIZE[0]
        self._rect.y = 375

    def step(self, speed: int):
        self._rect.x -= speed
        self._assets[0], self._assets[1] = (
            self._assets[1],
            self._assets[0],
        )

    def collide(self, other: pygame.Rect) -> bool:
        return self._rect.colliderect(other)

    def is_inside(self) -> bool:
        return self._rect.x + self._assets[0].get_width() > 0

    def render(self, canvas: pygame.Surface):
        canvas.blit(
            self._assets[0],
            self._rect,
        )


class Cactus(Obstacle):
    def __init__(self, assets: Assets, id: int):
        self._asset = assets.cactuses[id]
        self._rect = self._asset.get_rect()
        self._rect.x = WINDOW_SIZE[0]
        self._rect.y = WINDOW_SIZE[1] - self._asset.get_height() - 7

    def step(self, speed: int):
        self._rect.x -= speed

    def collide(self, other: pygame.Rect) -> bool:
        return self._rect.colliderect(other)

    def is_inside(self) -> bool:
        return self._rect.x + self._asset.get_width() > 0

    def render(self, canvas: pygame.Surface):
        canvas.blit(
            self._asset,
            self._rect,
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

    def get_data(self) -> tuple[pygame.Surface, pygame.Rect]:
        match self._state:
            case DinoState.STAND:
                asset = self._run_assets[0]
                y = WINDOW_SIZE[1] - asset.get_height()
            case DinoState.JUMP:
                asset = self._jump_asset
                y = WINDOW_SIZE[1] - self._get_jump_offset() - asset.get_height()
            case DinoState.DUCK:
                asset = self._duck_assets[0]
                y = WINDOW_SIZE[1] - asset.get_height()

        rect = pygame.Rect(50, y, asset.get_width(), asset.get_height())

        return asset, rect

    def _get_jump_offset(self) -> int:
        a = -JUMP_VEL / (JUMP_DURATION / 2)
        t = JUMP_DURATION - self._jump_timer
        d = int(JUMP_VEL * t + 0.5 * a * (t**2))
        return d

    def render(self, canvas: pygame.Surface):
        asset, rect = self.get_data()
        canvas.blit(asset, rect)


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
        # the observation space is the rgb image of the current frame
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8
        )

        self._assets = Assets()

        # Initialize `pygame` data
        self._window = None
        self._clock = None

        if self.render_mode == RenderMode.HUMAN:
            pygame.init()
            pygame.display.init()

            self._window = pygame.display.set_mode(WINDOW_SIZE)
            self._clock = pygame.time.Clock()

        self._init_game_data()

        super().__init__()

    def _init_game_data(self):
        """Initialize game's data, which should be re-initialized when the environment is reset"""
        self._frame = 0
        self._speed = 20
        self._spawn_prob = BASE_SPAWN_PROB
        self._obstacle_cnt = OBSTACLE_MIN_CNT

        # Initialize environment's objects' states
        self._track = Track(self._assets)
        self._dino = Dino(self._assets)
        self._obstacles: list[Obstacle] = []

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        self._init_game_data()

        obs = self._render_frame()

        return obs, {}

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict]:
        terminated = False
        reward = 1.0

        self._frame += 1
        self._obstacle_cnt += self._speed
        # increase the difficulty of the game every 20 frames
        if self._frame % 20 == 0:
            self._speed = min(MAX_SPEED, self._speed + 1)
            self._spawn_prob = min(MAX_SPAWN_PROB, self._spawn_prob * 1.01)

        self._track.step(self._speed)
        self._dino.step(action)
        for o in self._obstacles:
            o.step(self._speed)

        # Filter inside obstacles after each step
        self._obstacles = [o for o in self._obstacles if o.is_inside()]

        _, dino_rect = self._dino.get_data()
        for o in self._obstacles:
            if o.collide(dino_rect):
                reward = 0.0
                terminated = True

        # Should we spawn a new obstacle?
        if self._obstacle_cnt > max(OBSTACLE_MIN_CNT, JUMP_DURATION * self._speed):
            if self.np_random.choice(2, 1, p=[1 - self._spawn_prob, self._spawn_prob])[
                0
            ]:
                id = self.np_random.choice(len(self._assets.cactuses), 1)[0]
                self._obstacles.append(Cactus(self._assets, id))

            elif self.np_random.choice(2, 1, p=[0.9, 0.1])[0]:
                self._obstacles.append(Bird(self._assets))

            self._obstacle_cnt = 0

        obs = self._render_frame()

        return obs, reward, terminated, False, {}

    def render(self):
        if self.render_mode == RenderMode.RGB:
            return self._render_frame()

    def _render_frame(self) -> np.ndarray:
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

        # return the canvas as a rgb array
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()


register(
    id="Env-v0",
    entry_point="envs:Env",
    max_episode_steps=300,
)
