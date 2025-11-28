import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import sys
from collections import deque
from typing import Optional

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
except ImportError:
    # Fallback if config not available
    config = None

from story_env import StoryEnvGym, MultiStoryEnvGym
from dataset_manager import DatasetManager

# ----------------- Q-Network -----------------
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_size),
        )

    def forward(self, x):
        return self.net(x)

# ----------------- DQN Agent -----------------
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.9, lr=1e-3,
                 batch_size=32, buffer_size=10000, epsilon_decay=0.995, 
                 epsilon_min=0.1, target_update_frequency=10):
        self.q_net = QNetwork(state_size, action_size)
        self.target_net = QNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer = deque(maxlen=buffer_size)
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_frequency = target_update_frequency
        self.step_count = 0

    def act(self, state, action_size=None):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            action_size: Number of actions (if None, uses network output size)
        """
        if action_size is None:
            action_size = self.q_net.net[-1].out_features
        
        if random.random() < self.epsilon:
            return random.randrange(action_size)
        with torch.no_grad():
            q_vals = self.q_net(torch.tensor(state).float().unsqueeze(0))
        return torch.argmax(q_vals).item()

    def memorize(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))
        self.step_count += 1

    def replay(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        s, a, r, s2, done = zip(*batch)
        s = torch.tensor(np.array(s)).float()
        a = torch.tensor(a)
        r = torch.tensor(r).float()
        s2 = torch.tensor(np.array(s2)).float()
        done = torch.tensor(done).float()

        q_pred = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0]
            q_target = r + self.gamma * q_next * (1 - done)

        loss = nn.MSELoss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# ----------------- Training Loop (Legacy - Single Story) -----------------
def train_dqn_legacy(episodes=100, story_path="../data/story_sample.json"):
    """Legacy training function for single story."""
    env = StoryEnvGym(story_path=story_path)
    agent = DQNAgent(state_size=768, action_size=2)
    rewards = []

    for ep in range(episodes):
        s, _ = env.reset()
        total = 0
        done = False
        while not done:
            a = agent.act(s)
            s2, r, done, _, _ = env.step(a)
            agent.memorize(s, a, r, s2, done)
            agent.replay()
            s = s2
            total += r
        agent.update_target()
        agent.decay_epsilon()
        rewards.append(total)
        print(f"Episode {ep+1:03d} | Reward: {total:.2f} | Epsilon: {agent.epsilon:.3f}")

    torch.save(agent.q_net.state_dict(), "../models/saved_dqn.pt")
    print("✅ Training complete. Model saved.")
    return rewards

# ----------------- Training Loop (Multi-Story) -----------------
def train_dqn_multi_story(csv_path: str, json_annotations_path: Optional[str] = None,
                          split: str = 'train', max_stories: Optional[int] = None,
                          episodes: int = 1000, use_annotations: bool = False,
                          use_generation: bool = False):
    """
    Train DQN on multiple stories from dataset.
    
    Args:
        csv_path: Path to rocstorysubset.csv
        json_annotations_path: Optional path to annotations.json
        split: 'train', 'dev', or 'test'
        max_stories: Optional maximum number of stories to use
        episodes: Number of training episodes
        use_annotations: Whether to use annotations (for future enhancement)
        use_generation: Whether to use story generation (multi-choice mode)
    """
    # Load dataset
    print(f"Loading dataset from {csv_path}...")
    dataset_manager = DatasetManager(
        csv_path=csv_path,
        json_annotations_path=json_annotations_path if use_annotations else None,
        split=split,
        max_stories=max_stories
    )
    
    # Create environment with enhanced rewards if annotations are used
    reward_weights = config.REWARD_WEIGHTS if config else None
    # Get config settings for stochastic emotions
    use_stochastic_emotions = True
    if config:
        use_stochastic_emotions = getattr(config, 'USE_STOCHASTIC_EMOTIONS', True)
    
    env = MultiStoryEnvGym(
        dataset_manager,
        use_enhanced_rewards=use_annotations,
        reward_weights=reward_weights,
        use_generation=use_generation,
        use_stochastic_emotions=use_stochastic_emotions
    )
    
    # Get actual state dimension and action size from environment
    state_dim = env.state_dim
    action_size = env.action_space.n  # 3 if generation, 2 otherwise
    
    # Create agent with correct state dimension and action size
    agent_config = config.AGENT_CONFIG.copy() if config else {
        'gamma': 0.9,
        'lr': 1e-3,
        'batch_size': 32,
        'buffer_size': 10000,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.1,
        'target_update_frequency': 10
    }
    agent_config['state_size'] = state_dim
    agent_config['action_size'] = action_size
    
    agent = DQNAgent(**agent_config)
    
    # Training metrics
    rewards = []
    episode_rewards = []
    
    print(f"\n🚀 Starting training on {len(dataset_manager)} stories...")
    print(f"Episodes: {episodes} | State dim: {state_dim} | Action size: {action_size} | Epsilon: {agent.epsilon:.3f}")
    print(f"Enhanced rewards: {use_annotations} | Generation mode: {use_generation} | Reward weights: {reward_weights}\n")
    
    # Metrics tracking
    best_avg_reward = float('-inf')
    best_episode = 0
    
    for ep in range(episodes):
        s, _ = env.reset()
        total = 0
        steps = 0
        done = False
        
        while not done:
            a = agent.act(s, action_size=action_size)
            s2, r, done, _, info = env.step(a)
            agent.memorize(s, a, r, s2, done)
            agent.replay()
            s = s2
            total += r
            steps += 1
        
        # Update target network periodically
        if (ep + 1) % agent.target_update_frequency == 0:
            agent.update_target()
        
        agent.decay_epsilon()
        rewards.append(total)
        episode_rewards.append(total)
        
        # Track best performance
        if len(episode_rewards) >= 10:
            avg_reward = np.mean(episode_rewards[-10:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_episode = ep + 1
        
        # Print progress
        if (ep + 1) % 10 == 0 or ep == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total
            max_reward = max(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total
            min_reward = min(episode_rewards[-10:]) if len(episode_rewards) >= 10 else total
            print(f"Episode {ep+1:04d} | Reward: {total:.2f} | Avg (last 10): {avg_reward:.2f} | "
                  f"Max: {max_reward:.2f} | Min: {min_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | Buffer: {len(agent.buffer)}")
    
    # Save model
    model_path = config.MODEL_SAVE_PATH if config else "../models/saved_dqn.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(agent.q_net.state_dict(), model_path)
    
    # Print final statistics
    print(f"\n✅ Training complete. Model saved to {model_path}")
    print(f"\n📊 Training Statistics:")
    print(f"  Final epsilon: {agent.epsilon:.3f}")
    print(f"  Total experiences: {len(agent.buffer)}")
    print(f"  Average reward (all episodes): {np.mean(rewards):.2f}")
    print(f"  Best average reward (last 10): {best_avg_reward:.2f} at episode {best_episode}")
    print(f"  Final average reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"  Max reward: {max(rewards):.2f} | Min reward: {min(rewards):.2f}")
    
    return rewards

# ----------------- Main Training Function -----------------
def train_dqn(episodes=100, use_dataset=True, max_stories=None, use_generation=False):
    """
    Main training function with automatic mode selection.
    
    Args:
        episodes: Number of episodes
        use_dataset: Whether to use multi-story dataset (True) or legacy single story (False)
        max_stories: Optional limit on number of stories (for testing)
        use_generation: Whether to use story generation (multi-choice mode)
    """
    if use_dataset and config:
        # Use multi-story dataset
        train_dqn_multi_story(
            csv_path=config.CSV_STORIES_PATH,
            json_annotations_path=config.JSON_ANNOTATIONS_PATH if config.USE_ANNOTATIONS else None,
            split=config.TRAIN_SPLIT,
            max_stories=max_stories or config.MAX_STORIES,
            episodes=episodes,
            use_annotations=config.USE_ANNOTATIONS,
            use_generation=use_generation
        )
    else:
        # Legacy mode - single story
        story_path = config.LEGACY_STORY_PATH if config else "../data/story_sample.json"
        train_dqn_legacy(episodes=episodes, story_path=story_path)

if __name__ == "__main__":
    # Check if config is available
    if config and os.path.exists(config.CSV_STORIES_PATH):
        print("Using multi-story dataset mode...")
        # Set use_generation=True to enable multi-choice story continuation
        use_gen = getattr(config, 'USE_GENERATION', False)
        train_dqn(episodes=100, use_dataset=True, max_stories=10, use_generation=use_gen)
    else:
        print("Using legacy single-story mode...")
        train_dqn(episodes=50, use_dataset=False)
