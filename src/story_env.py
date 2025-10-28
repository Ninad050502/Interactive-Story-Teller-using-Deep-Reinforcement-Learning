import random
import numpy as np
from dataset_loader import load_story
from state_encoder import StateEncoder

class StoryEnv:
    """
    A lightweight stochastic environment that simulates
    story progression through sentences.
    """

    def __init__(self, story_path: str, next_prob: float = 0.7):
        self.story = load_story(story_path)
        self.encoder = StateEncoder()
        self.next_prob = next_prob
        self.n_states = len(self.story)
        self.reset()

    def reset(self):
        """Resets the environment to the beginning of the story."""
        self.idx = 0
        state_text = self.story[self.idx]
        return self.encoder.encode(state_text)

    def step(self, action: int):
        """
        Performs one environment step.
        Since Phase 1 doesn’t yet use true actions,
        'action' is ignored — stochastic transition instead.
        """
        done = self.idx >= self.n_states - 1
        if done:
            return None, 0.0, True, {}

        # stochastic transition: usually next line, occasionally skip or repeat
        if random.random() < self.next_prob:
            next_idx = self.idx + 1
        else:
            next_idx = min(self.idx + random.choice([1, 2]), self.n_states - 1)

        # reward = +1 if sequential, −1 if jump (simulating coherence)
        reward = 1.0 if next_idx == self.idx + 1 else -1.0

        self.idx = next_idx
        next_state = self.encoder.encode(self.story[self.idx])
        done = self.idx == self.n_states - 1
        info = {"line": self.story[self.idx]}
        return next_state, reward, done, info
