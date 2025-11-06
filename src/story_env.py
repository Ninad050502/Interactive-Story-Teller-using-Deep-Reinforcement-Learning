import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from dataset_loader import load_story
from state_encoder import StateEncoder


# ----------- 1️⃣ Original StoryEnv -----------
class StoryEnv:
    def __init__(self, story_path: str, next_prob: float = 0.7):
        self.story = load_story(story_path)
        self.encoder = StateEncoder()
        self.next_prob = next_prob
        self.n_states = len(self.story)
        self.reset()

    def reset(self):
        self.idx = 0
        state_text = self.story[self.idx]
        return self.encoder.encode(state_text)

    def step(self, action: int):
        done = self.idx >= self.n_states - 1
        if done:
            return None, 0.0, True, {}

        if random.random() < self.next_prob:
            next_idx = self.idx + 1
        else:
            next_idx = min(self.idx + random.choice([1, 2]), self.n_states - 1)

        reward = 1.0 if next_idx == self.idx + 1 else -1.0
        self.idx = next_idx
        next_state = self.encoder.encode(self.story[self.idx])
        done = self.idx == self.n_states - 1
        info = {"line": self.story[self.idx]}
        return next_state, reward, done, info


# ----------- 2️⃣ Gym Wrapper -----------
class StoryEnvGym(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, story_path: str):
        super().__init__()
        self.inner_env = StoryEnv(story_path)  # ✅ Now StoryEnv is defined above
        self.state_dim = 768
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        state = self.inner_env.reset()
        return state.numpy().astype(np.float32), {}

    def step(self, action):
        if action == 0:
            self.inner_env.next_prob = 0.9
        else:
            self.inner_env.next_prob = 0.4

        next_state, reward, done, info = self.inner_env.step(action)
        return next_state.numpy().astype(np.float32), reward, done, False, info

    def render(self):
        print(f"Current index: {self.inner_env.idx}")
