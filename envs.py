from abc import ABC
from collections import deque
import enum
import os
from typing import Any
from PIL import Image
import gymnasium as gym
import numpy as np
import pygame.freetype
import pygame
from gymnasium.envs.registration import register

# Environment's constants
WINDOW_SIZE = (1024, 512)  # (w, h)
# It took (JUMP_DURATION / 2) to jump to the peak and another (JUMP_DURATION / 2) to fall to the ground
JUMP_DURATION = 12
JUMP_VEL = 100
OBSTACLE_MIN_CNT = 400
MAX_SPEED = 100
MAX_CACTUS_SPAWN_PROB = 0.7
BASE_CACTUS_SPAWN_PROB = 0.3
BIRD_SPAWN_PROB = 0.3
RENDER_FPS = 15
COLLISION_THRESHOLD = 20
DIFFICULTY_INCREASE_FREQ = 20


class Action(int, enum.Enum):
    STAND = 0
    JUMP = 1
    DUCK = 2


class DinoState(int, enum.Enum):
    STAND = 0
    JUMP = 1
    DUCK = 2


class GameMode(str, enum.Enum):
    NORMAL = "normal"
    # In the train mode, when the agent collide with obstacles,
    # it gets negative rewards instead of losing the game.
    TRAIN = "train"


class RenderMode(str, enum.Enum):
    HUMAN = "human"
    RGB = "rgb_array"


class Assets:
    def __init__(self):
        # running track
        self.track = pygame.image.load(os.path.join("assets", "Track.png"))

        # dino
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
    rect: pygame.Rect

    def __init__(self, assets: Assets, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def render(self, canvas: pygame.Surface, *args, **kwargs):
        pass


class Obstacle(EnvObject, ABC):
    # A flag indicates if the agent already passes or collides the obstacle.
    # This is used to avoid "duplicating" rewards for passing/colliding an obstacle.
    needs_collision_check = True

    def collide(self, o: pygame.Rect) -> bool:
        return self.rect.colliderect(
            o.left + COLLISION_THRESHOLD,
            o.top + COLLISION_THRESHOLD,
            o.width - 2 * COLLISION_THRESHOLD,
            o.height - 2 * COLLISION_THRESHOLD,
        )

    def is_inside(self) -> bool:
        return False


class Bird(Obstacle):
    def __init__(self, assets: Assets):
        self._assets = assets.birds
        self.rect = self._assets[0].get_rect()
        self.rect.x = WINDOW_SIZE[0]
        self.rect.y = 360

    def step(self, speed: int):
        self.rect.x -= speed
        # Alternate the assets to create a moving animation
        self._assets[0], self._assets[1] = (
            self._assets[1],
            self._assets[0],
        )

    def is_inside(self) -> bool:
        return self.rect.x + self._assets[0].get_width() > 0

    def render(self, canvas: pygame.Surface):
        canvas.blit(
            self._assets[0],
            self.rect,
        )


class Cactus(Obstacle):
    def __init__(self, assets: Assets, id: int):
        self._asset = assets.cactuses[id]
        self.rect = self._asset.get_rect()
        self.rect.x = WINDOW_SIZE[0]
        self.rect.y = WINDOW_SIZE[1] - self._asset.get_height() - 7

    def step(self, speed: int):
        self.rect.x -= speed

    def is_inside(self) -> bool:
        return self.rect.x + self._asset.get_width() > 0

    def render(self, canvas: pygame.Surface):
        canvas.blit(
            self._asset,
            self.rect,
        )


class Dino(EnvObject):
    def __init__(self, assets: Assets):
        self._run_assets = assets.dino_runs
        self._duck_assets = assets.dino_ducks
        self._jump_asset = assets.dino_jump

        self._jump_timer = 0
        self.state = DinoState.STAND

    def step(self, action: Action):
        # Alternate the assets to create a moving animation
        self._run_assets[0], self._run_assets[1] = (
            self._run_assets[1],
            self._run_assets[0],
        )
        self._duck_assets[0], self._duck_assets[1] = (
            self._duck_assets[1],
            self._duck_assets[0],
        )

        # Check if the jump animation is finished
        if self.state == DinoState.JUMP:
            self._jump_timer -= 1
            if self._jump_timer < 0:
                self.state = DinoState.STAND

        # If dino is not jumping, transition to a new state based on the action
        if self.state != DinoState.JUMP:
            match action:
                case Action.STAND:
                    self.state = DinoState.STAND
                case Action.JUMP:
                    self.state = DinoState.JUMP
                    self._jump_timer = JUMP_DURATION
                case Action.DUCK:
                    self.state = DinoState.DUCK

    def get_data(self) -> tuple[pygame.Surface, pygame.Rect]:
        match self.state:
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
        # Compute the jump distance from acceleration, initial speed, and time
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
        # Negative offset means moving the running track image to the left
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
        game_mode: GameMode = GameMode.NORMAL,
        train_frame_limit=500,  # the upper limit for number of frames during the train mode
    ) -> None:
        # Initialize `gym.Env` required fields
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(len(list(Action)))
        # The observation space is the dimension of the current frame (rgb image)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8
        )

        self._game_mode = game_mode
        self._train_frame_limit = train_frame_limit

        # Initialize `pygame` data
        self._window = None
        self._clock = None

        pygame.freetype.init()
        self._game_font = pygame.freetype.SysFont(
            pygame.freetype.get_default_font(), 24
        )
        if self.render_mode == RenderMode.HUMAN:
            pygame.init()
            pygame.display.init()

            self._window = pygame.display.set_mode(WINDOW_SIZE)
            self._clock = pygame.time.Clock()

        self._init_game_data()

        super().__init__()

    def _init_game_data(self):
        self._assets = Assets()

        """Initialize game's data, which should be re-initialized when the environment is reset"""
        self._frame = 0
        self._speed = 20
        self._spawn_prob = BASE_CACTUS_SPAWN_PROB
        # The counter (in pixels) for spawning a new obstacle
        self._obstacle_cnt = OBSTACLE_MIN_CNT

        # Initialize environment's objects' states
        self._track = Track(self._assets)
        self._agent = Dino(self._assets)
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
        reward = 0.0

        self._frame += 1
        self._obstacle_cnt += self._speed
        # Increase the difficulty of the game every fixed number of frames
        if self._frame % DIFFICULTY_INCREASE_FREQ == 0:
            self._speed = min(MAX_SPEED, self._speed + 1)
            self._spawn_prob = min(MAX_CACTUS_SPAWN_PROB, self._spawn_prob * 1.01)

        self._track.step(self._speed)
        self._agent.step(action)
        for o in self._obstacles:
            o.step(self._speed)

        # Filter out outside obstacles after each step
        self._obstacles = [o for o in self._obstacles if o.is_inside()]

        # Check if the agent collides with an obstacle
        _, agent_rect = self._agent.get_data()
        for o in self._obstacles:
            if not o.needs_collision_check:
                continue
            if o.collide(agent_rect):
                o.needs_collision_check = False
                reward -= 1.0
                if self._game_mode == GameMode.NORMAL:
                    terminated = True
            else:
                # Agent passes an obstacle without colliding with the object, give a reward
                if agent_rect.left > o.rect.right:
                    o.needs_collision_check = False
                    reward += 1.0

        if self._game_mode == GameMode.TRAIN and self._frame >= self._train_frame_limit:
            terminated = True

        # Should we spawn a new obstacle?
        self._spawn_obstacle_maybe()

        obs = self._render_frame()

        return obs, reward, terminated, False, {}

    def _spawn_obstacle_maybe(self):
        if self._obstacle_cnt > max(OBSTACLE_MIN_CNT, JUMP_DURATION * self._speed):
            if self.np_random.choice(2, 1, p=[1 - self._spawn_prob, self._spawn_prob])[
                0
            ]:
                id = self.np_random.choice(len(self._assets.cactuses), 1)[0]
                self._obstacles.append(Cactus(self._assets, id))

            elif self.np_random.choice(2, 1, p=[0.9, 0.1])[0]:
                self._obstacles.append(Bird(self._assets))

            self._obstacle_cnt = 0

    def render(self):
        if self.render_mode == RenderMode.RGB:
            return self._render_frame()

    def _render_frame(self) -> np.ndarray:
        canvas = pygame.Surface(WINDOW_SIZE)
        canvas.fill((255, 255, 255))

        self._track.render(canvas)
        self._agent.render(canvas)
        for o in self._obstacles:
            o.render(canvas)

        # Display the current scores (number of frames)
        text_surface, _ = self._game_font.render(f"score: {self._frame}", (0, 0, 0))
        canvas.blit(text_surface, (10, 10))

        if self._window is not None and self._clock is not None:
            self._window.blit(canvas, canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

            self._clock.tick(self.metadata["render_fps"])

        # Return the canvas as a rgb array
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()


class Wrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, k=4, image_size=(128, 64)):
        super().__init__(env)

        self.env = env
        self.k = k
        self.image_size = image_size

        obs_space = env.observation_space.shape
        assert obs_space is not None
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.k, self.image_size[1], self.image_size[0]),
            dtype=np.uint8,
        )

        self.frames: list[np.ndarray] = []
        self.stack = deque([], maxlen=self.k)

    def _transform(self, obs: np.ndarray) -> np.ndarray:
        # Convert the observation image from the environment to
        # gray scale and resize it to a corresponding size
        return np.array(
            Image.fromarray(obs).convert("L").resize(self.image_size), dtype=np.float32
        )

    def _get_obs(self) -> np.ndarray:
        # Stack the last "k" frames into a single "np.ndarray"
        assert len(self.stack) == self.k
        return np.stack(self.stack)

    def reset(self, *args, **kwargs) -> tuple[np.ndarray, dict]:
        self.frames = []
        self.stack = deque([], maxlen=self.k)

        obs, _ = self.env.reset(*args, **kwargs)
        self.frames.append(obs)
        obs = self._transform(obs)

        for _ in range(self.k):
            self.stack.append(obs)

        return self._get_obs(), {}

    def step(self, action: Action) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = False

        for _ in range(self.k):
            obs, reward, term, *_ = self.env.step(action)
            self.frames.append(obs)
            obs = self._transform(obs)

            self.stack.append(obs)

            total_reward += float(reward)
            if term:
                terminated = True
                break

        return self._get_obs(), total_reward, terminated, False, {}


register(
    id="Env-v0",
    entry_point="envs:Env",
    max_episode_steps=300,
)
