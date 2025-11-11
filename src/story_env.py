import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, List
from dataset_loader import load_story
from state_encoder import StateEncoder
from reward_calculator import RewardCalculator


# ----------- 1️⃣ Original StoryEnv (Legacy Support) -----------
class StoryEnv:
    def __init__(self, story_path: Optional[str] = None, 
                 story_data: Optional[Dict] = None, 
                 next_prob: float = 0.7,
                 use_enhanced_rewards: bool = True,
                 reward_weights: Optional[Dict[str, float]] = None):
        """
        Initialize StoryEnv with either a story path or story data.
        
        Args:
            story_path: Path to JSON story file (legacy support)
            story_data: Story dictionary from dataset (new format)
            next_prob: Probability of following correct sequence
            use_enhanced_rewards: Whether to use enhanced reward calculation
            reward_weights: Optional reward weights for enhanced rewards
        """
        # Determine if we should use character features
        has_annotations = False
        if story_data is not None:
            self.story = story_data['lines']
            self.story_id = story_data.get('storyid', 'unknown')
            self.story_title = story_data.get('title', 'Unknown')
            self.annotations = story_data.get('annotations', None)
            has_annotations = self.annotations is not None
        elif story_path is not None:
            self.story = load_story(story_path)
            self.story_id = 'legacy'
            self.story_title = 'Legacy Story'
            self.annotations = None
        else:
            raise ValueError("Either story_path or story_data must be provided")
        
        # Initialize encoder with character features if annotations available
        self.encoder = StateEncoder(include_character_features=has_annotations and use_enhanced_rewards)
        
        # Initialize reward calculator
        self.use_enhanced_rewards = use_enhanced_rewards and has_annotations
        self.reward_calculator = RewardCalculator(reward_weights=reward_weights) if self.use_enhanced_rewards else None
        
        self.next_prob = next_prob
        self.n_states = len(self.story)
        
        # Store previous state and character info for enhanced rewards
        self.prev_state = None
        self.prev_char_info = None
        
        self.reset()

    def reset(self):
        """Reset environment to start of story."""
        self.idx = 0
        state_text = self.story[self.idx]
        
        # Get character info for current line if annotations available
        char_info = None
        if self.annotations and 'lines' in self.annotations:
            line_key = str(self.idx + 1)  # Lines are 1-indexed in annotations
            if line_key in self.annotations['lines']:
                char_info = self.annotations['lines'][line_key]
        
        state = self.encoder.encode(state_text, character_info=char_info)
        self.prev_state = state
        self.prev_char_info = char_info
        return state

    def step(self, action: int):
        """Take a step in the environment."""
        done = self.idx >= self.n_states - 1
        if done:
            return None, 0.0, True, {}

        prev_idx = self.idx
        
        if random.random() < self.next_prob:
            next_idx = self.idx + 1
        else:
            next_idx = min(self.idx + random.choice([1, 2]), self.n_states - 1)

        self.idx = next_idx
        
        # Get current state and character info
        state_text = self.story[self.idx]
        char_info = None
        if self.annotations and 'lines' in self.annotations:
            line_key = str(self.idx + 1)  # Lines are 1-indexed
            if line_key in self.annotations['lines']:
                char_info = self.annotations['lines'][line_key]
        
        next_state = self.encoder.encode(state_text, character_info=char_info)
        
        # Calculate reward
        if self.use_enhanced_rewards and self.reward_calculator:
            reward = self.reward_calculator.calculate_reward(
                prev_idx=prev_idx,
                current_idx=self.idx,
                prev_state=self.prev_state,
                current_state=next_state,
                prev_char_info=self.prev_char_info,
                current_char_info=char_info
            )
        else:
            # Simple sequence reward
            reward = 1.0 if next_idx == prev_idx + 1 else -1.0
        
        # Update previous state and character info
        self.prev_state = next_state
        self.prev_char_info = char_info
        
        done = self.idx == self.n_states - 1
        info = {
            "line": self.story[self.idx],
            "story_id": self.story_id,
            "story_title": self.story_title,
            "line_idx": self.idx
        }
        return next_state, reward, done, info
    
    def set_story(self, story_data: Dict):
        """Set a new story (for multi-story training)."""
        self.story = story_data['lines']
        self.story_id = story_data.get('storyid', 'unknown')
        self.story_title = story_data.get('title', 'Unknown')
        self.annotations = story_data.get('annotations', None)
        
        # Update encoder and reward calculator based on new annotations
        has_annotations = self.annotations is not None
        if has_annotations != self.encoder.include_character_features:
            self.encoder = StateEncoder(include_character_features=has_annotations and self.use_enhanced_rewards)
            self.use_enhanced_rewards = self.use_enhanced_rewards and has_annotations
            if self.use_enhanced_rewards and not self.reward_calculator:
                self.reward_calculator = RewardCalculator()
        
        self.n_states = len(self.story)
        self.prev_state = None
        self.prev_char_info = None
        self.reset()


# ----------- 2️⃣ Gym Wrapper (Legacy Support) -----------
class StoryEnvGym(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, story_path: Optional[str] = None, 
                 story_data: Optional[Dict] = None):
        """
        Initialize Gym environment.
        
        Args:
            story_path: Path to JSON story file (legacy support)
            story_data: Story dictionary from dataset (new format)
        """
        super().__init__()
        self.inner_env = StoryEnv(story_path=story_path, story_data=story_data)
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
        if next_state is None:
            # Handle end of story
            return np.zeros(self.state_dim, dtype=np.float32), reward, done, False, info
        return next_state.numpy().astype(np.float32), reward, done, False, info

    def render(self):
        print(f"Current index: {self.inner_env.idx}")
    
    def set_story(self, story_data: Dict):
        """Set a new story (for multi-story training)."""
        self.inner_env.set_story(story_data)


# ----------- 3️⃣ Multi-Story Environment (New) -----------
class MultiStoryEnvGym(gym.Env):
    """
    Environment that supports multiple stories from a dataset.
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self, dataset_manager, use_enhanced_rewards: bool = True,
                 reward_weights: Optional[Dict[str, float]] = None):
        """
        Initialize multi-story environment.
        
        Args:
            dataset_manager: DatasetManager instance
            use_enhanced_rewards: Whether to use enhanced rewards
            reward_weights: Optional reward weights for enhanced rewards
        """
        super().__init__()
        self.dataset_manager = dataset_manager
        self.use_enhanced_rewards = use_enhanced_rewards
        self.reward_weights = reward_weights
        
        # Determine state dimension (will be updated when first story is loaded)
        self.state_dim = 768  # Default, will be updated if character features used
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        # Initialize with first story
        self.current_story = None
        self.inner_env = None
        self._load_new_story()
    
    def _load_new_story(self):
        """Load a new random story from the dataset."""
        self.current_story = self.dataset_manager.get_story()
        if self.inner_env is None:
            self.inner_env = StoryEnv(
                story_data=self.current_story,
                use_enhanced_rewards=self.use_enhanced_rewards,
                reward_weights=self.reward_weights
            )
            # Update state dimension based on encoder
            if hasattr(self.inner_env.encoder, 'include_character_features'):
                if self.inner_env.encoder.include_character_features:
                    self.state_dim = 800  # 768 + 32 character features
                else:
                    self.state_dim = 768
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
                )
        else:
            self.inner_env.set_story(self.current_story)
            # Update state dimension if needed
            if hasattr(self.inner_env.encoder, 'include_character_features'):
                if self.inner_env.encoder.include_character_features:
                    self.state_dim = 800
                else:
                    self.state_dim = 768
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
                )
    
    def reset(self, seed=None, options=None):
        """Reset environment with a new random story."""
        self._load_new_story()
        state = self.inner_env.reset()
        return state.numpy().astype(np.float32), {}
    
    def step(self, action):
        """Take a step in the current story."""
        if action == 0:
            self.inner_env.next_prob = 0.9
        else:
            self.inner_env.next_prob = 0.4
        
        next_state, reward, done, info = self.inner_env.step(action)
        
        if next_state is None:
            # End of story
            return np.zeros(self.state_dim, dtype=np.float32), reward, done, False, info
        
        return next_state.numpy().astype(np.float32), reward, done, False, info
    
    def render(self):
        """Render current state."""
        if self.inner_env:
            print(f"Story: {self.inner_env.story_title} | Line: {self.inner_env.idx + 1}/{self.inner_env.n_states}")
    
    def get_current_story_info(self):
        """Get information about current story."""
        if self.current_story:
            return {
                'story_id': self.current_story.get('storyid', 'unknown'),
                'title': self.current_story.get('title', 'Unknown'),
                'has_annotations': self.current_story.get('annotations') is not None
            }
        return None
