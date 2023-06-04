import enum
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register


class Action(enum.Enum):
    UP = 0
    DOWN = 1


class Observation:
    ...


class Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode: str | None = None) -> None:
        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, 200, shape=(2,), dtype=np.int32),
            }
        )

        super().__init__()

    def _get_obs(self) -> dict:
        return {}

    def _get_info(self) -> dict:
        return {}

    def reset(self, seed: int | None = None) -> tuple[dict, dict]:
        super().reset(seed=seed)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: Action) -> tuple[dict, float, bool, bool, dict]:
        terminated = False
        reward = 0.0
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, False, info

    def render(self):
        ...

    def close(self):
        ...


register(
    id="Env-v0",
    entry_point="env:Env",
    max_episode_steps=300,
)
