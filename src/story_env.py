import random
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, List
from dataset_loader import load_story
from state_encoder import StateEncoder
from reward_calculator import RewardCalculator
from story_generator import StoryGenerator


# ----------- 1️⃣ Original StoryEnv (Legacy Support) -----------
class StoryEnv:
    def __init__(self, story_path: Optional[str] = None, 
                 story_data: Optional[Dict] = None, 
                 next_prob: float = 0.7,
                 use_enhanced_rewards: bool = True,
                 reward_weights: Optional[Dict[str, float]] = None,
                 use_generation: bool = False,
                 story_generator: Optional[StoryGenerator] = None):
        """
        Initialize StoryEnv with either a story path or story data.
        
        Args:
            story_path: Path to JSON story file (legacy support)
            story_data: Story dictionary from dataset (new format)
            next_prob: Probability of following correct sequence
            use_enhanced_rewards: Whether to use enhanced reward calculation
            reward_weights: Optional reward weights for enhanced rewards
            use_generation: Whether to use story generation (multi-choice mode)
            story_generator: StoryGenerator instance (created if None and use_generation=True)
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
        
        # Initialize story generator if using generation mode
        self.use_generation = use_generation
        if use_generation:
            # Use config settings if available
            try:
                import config
                gen_config = getattr(config, 'GENERATION_CONFIG', {})
                filter_inappropriate = gen_config.get('filter_inappropriate', True)
                model_name = gen_config.get('model_name', 'gpt2')
            except:
                filter_inappropriate = True
                model_name = 'gpt2'
            
            if story_generator is None:
                self.story_generator = StoryGenerator(
                    model_name=model_name,
                    filter_inappropriate=filter_inappropriate
                )
            else:
                self.story_generator = story_generator
        else:
            self.story_generator = None
        
        self.next_prob = next_prob
        self.n_states = len(self.story)
        
        # Store previous state and character info for enhanced rewards
        self.prev_state = None
        self.prev_char_info = None
        
        # Store generated story so far (for generation mode)
        self.generated_story = []
        self.current_options = []  # [true, generated1, generated2]
        
        self.reset()

    def reset(self):
        """Reset environment to start of story."""
        self.idx = 0
        state_text = self.story[self.idx]
        self.generated_story = [state_text]  # Start with first sentence
        
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
        
        # If using generation mode, agent chooses among continuations
        if self.use_generation and self.story_generator:
            # Get story context so far
            story_context = self.generated_story.copy()
            
            # Get true continuation from dataset (if available)
            true_continuation = None
            if self.idx + 1 < len(self.story):
                true_continuation = self.story[self.idx + 1]
            
            # Generate 2 alternative continuations
            try:
                # Use config settings if available
                try:
                    import config
                    gen_config = getattr(config, 'GENERATION_CONFIG', {})
                    num_options = gen_config.get('num_generated_options', 2)
                    max_length = gen_config.get('max_length', 50)
                    temp_range = gen_config.get('temperature_range', (0.6, 0.8))
                    max_attempts = gen_config.get('max_attempts', 5)
                except:
                    num_options = 2
                    max_length = 50
                    temp_range = (0.6, 0.8)
                    max_attempts = 5
                
                generated_options = self.story_generator.generate_continuations(
                    context=story_context,
                    num_options=num_options,
                    max_length=max_length,
                    temperature_range=temp_range,
                    max_attempts=max_attempts
                )
            except Exception as e:
                # Fallback if generation fails
                print(f"Warning: Generation failed: {e}. Using fallback options.")
                generated_options = ["The story continued.", "Something happened."]
            
            # Create 3 options: [true, generated1, generated2]
            self.current_options = []
            if true_continuation:
                self.current_options.append(true_continuation)
            self.current_options.extend(generated_options)
            
            # Ensure we have at least 3 options (pad if needed)
            while len(self.current_options) < 3:
                self.current_options.append("The story continued.")
            
            # Agent's action selects which continuation to use (0, 1, or 2)
            if action >= len(self.current_options):
                action = len(self.current_options) - 1
            
            selected_continuation = self.current_options[action]
            
            # Update story with selected continuation
            self.idx += 1
            self.generated_story.append(selected_continuation)
            
            # Get character info for selected continuation
            char_info = None
            if self.annotations and 'lines' in self.annotations:
                line_key = str(self.idx + 1)  # Lines are 1-indexed
                if line_key in self.annotations['lines']:
                    char_info = self.annotations['lines'][line_key]
            
            next_state = self.encoder.encode(selected_continuation, character_info=char_info)
            
            # Calculate reward based on agent's choice
            reward = self._calculate_generation_reward(
                action=action,
                selected=selected_continuation,
                true_continuation=true_continuation,
                generated_options=generated_options,
                prev_state=self.prev_state,
                current_state=next_state,
                prev_char_info=self.prev_char_info,
                current_char_info=char_info
            )
            
        else:
            # Original behavior: probabilistic sequence following
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
            "line": self.generated_story[-1] if self.use_generation else self.story[self.idx],
            "story_id": self.story_id,
            "story_title": self.story_title,
            "line_idx": self.idx,
            "chose_true": action == 0 if self.use_generation and self.current_options else None,
            "options": self.current_options if self.use_generation else None
        }
        return next_state, reward, done, info
    
    def _calculate_generation_reward(self, action: int, selected: str, true_continuation: Optional[str],
                                    generated_options: List[str], prev_state: torch.Tensor,
                                    current_state: torch.Tensor, prev_char_info: Optional[Dict],
                                    current_char_info: Optional[Dict]) -> float:
        """
        Calculate reward for generation mode.
        
        Args:
            action: Agent's action (0=true, 1=generated1, 2=generated2)
            selected: The selected continuation text
            true_continuation: True continuation from dataset (if available)
            generated_options: List of generated continuations
            prev_state: Previous state embedding
            current_state: Current state embedding
            prev_char_info: Previous character info
            current_char_info: Current character info
        
        Returns:
            Reward value
        """
        base_reward = 0.0
        
        # Reward for choosing true continuation
        if action == 0 and true_continuation:
            base_reward = 1.0
        elif action > 0:
            # Evaluate quality of generated continuation
            base_reward = self._evaluate_generation_quality(
                selected, true_continuation, generated_options
            )
        else:
            base_reward = 0.5  # Neutral if no true continuation available
        
        # Add coherence reward (how well it fits context)
        coherence_reward = 0.0
        if prev_state is not None and current_state is not None:
            # Use cosine similarity
            prev_norm = prev_state / (torch.norm(prev_state) + 1e-8)
            curr_norm = current_state / (torch.norm(current_state) + 1e-8)
            coherence_reward = torch.dot(prev_norm, curr_norm).item() * 0.3
        
        # Add character consistency (if using enhanced rewards)
        char_reward = 0.0
        if self.use_enhanced_rewards and self.reward_calculator:
            if prev_char_info and current_char_info:
                char_reward = self.reward_calculator._calculate_character_consistency(
                    prev_char_info, current_char_info
                ) * 0.2
        
        total_reward = base_reward + coherence_reward + char_reward
        return total_reward
    
    def _evaluate_generation_quality(self, selected: str, true_continuation: Optional[str],
                                    generated_options: List[str]) -> float:
        """
        Evaluate quality of generated continuation.
        
        Returns:
            Reward between 0.0 and 1.0
        """
        # Simple heuristic: reward diversity and reasonable length
        reward = 0.5  # Base reward for generation
        
        # Bonus for reasonable length (not too short, not too long)
        length = len(selected.split())
        if 5 <= length <= 25:
            reward += 0.2
        
        # Penalty if too similar to other generated options (encourage diversity)
        if len(generated_options) > 1:
            # Simple check: if very similar, reduce reward
            if selected in generated_options:
                other_idx = 1 if generated_options[0] == selected else 0
                if other_idx < len(generated_options):
                    other = generated_options[other_idx]
                    # Simple similarity check (word overlap)
                    selected_words = set(selected.lower().split())
                    other_words = set(other.lower().split())
                    if len(selected_words) > 0:
                        overlap = len(selected_words & other_words) / len(selected_words)
                        if overlap > 0.8:  # Too similar
                            reward -= 0.2
        
        return max(0.0, min(1.0, reward))
    
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
                 reward_weights: Optional[Dict[str, float]] = None,
                 use_generation: bool = False,
                 story_generator: Optional[StoryGenerator] = None):
        """
        Initialize multi-story environment.
        
        Args:
            dataset_manager: DatasetManager instance
            use_enhanced_rewards: Whether to use enhanced rewards
            reward_weights: Optional reward weights for enhanced rewards
            use_generation: Whether to use story generation (multi-choice mode)
            story_generator: StoryGenerator instance (shared across stories)
        """
        super().__init__()
        self.dataset_manager = dataset_manager
        self.use_enhanced_rewards = use_enhanced_rewards
        self.reward_weights = reward_weights
        self.use_generation = use_generation
        
        # Initialize story generator if using generation mode
        if use_generation:
            # Use config settings if available
            try:
                import config
                gen_config = getattr(config, 'GENERATION_CONFIG', {})
                filter_inappropriate = gen_config.get('filter_inappropriate', True)
                model_name = gen_config.get('model_name', 'gpt2')
            except:
                filter_inappropriate = True
                model_name = 'gpt2'
            
            if story_generator is None:
                self.story_generator = StoryGenerator(
                    model_name=model_name,
                    filter_inappropriate=filter_inappropriate
                )
            else:
                self.story_generator = story_generator
        else:
            self.story_generator = None
        
        # Determine state dimension (will be updated when first story is loaded)
        self.state_dim = 768  # Default, will be updated if character features used
        # Action space: 3 if generation, 2 otherwise
        self.action_space = spaces.Discrete(3 if use_generation else 2)
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
                reward_weights=self.reward_weights,
                use_generation=self.use_generation,
                story_generator=self.story_generator
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
            # Update generation mode
            self.inner_env.use_generation = self.use_generation
            if self.use_generation and not self.inner_env.story_generator:
                self.inner_env.story_generator = self.story_generator
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
        # In generation mode, action directly selects continuation (0, 1, or 2)
        # In non-generation mode, action controls probability
        if not self.use_generation:
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
