"""
Training Utilities for DQN Training
Includes metrics tracking, checkpointing, and evaluation utilities
"""

import torch
import os
import json
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime


class TrainingMetrics:
    """
    Track training metrics and statistics.
    """
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.epsilon_values = []
        self.timestamps = []
    
    def add_episode(self, reward: float, length: int, epsilon: float, loss: Optional[float] = None):
        """Add metrics for an episode."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.epsilon_values.append(epsilon)
        if loss is not None:
            self.losses.append(loss)
        self.timestamps.append(datetime.now().isoformat())
    
    def get_statistics(self, window: int = 10) -> Dict:
        """Get statistics over recent episodes."""
        if len(self.episode_rewards) == 0:
            return {}
        
        recent_rewards = self.episode_rewards[-window:] if len(self.episode_rewards) >= window else self.episode_rewards
        recent_lengths = self.episode_lengths[-window:] if len(self.episode_lengths) >= window else self.episode_lengths
        
        stats = {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards),
            'avg_reward_recent': np.mean(recent_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'avg_length': np.mean(self.episode_lengths),
            'avg_length_recent': np.mean(recent_lengths),
            'current_epsilon': self.epsilon_values[-1] if self.epsilon_values else 0.0,
        }
        
        if self.losses:
            stats['avg_loss'] = np.mean(self.losses)
            stats['avg_loss_recent'] = np.mean(self.losses[-window:]) if len(self.losses) >= window else np.mean(self.losses)
        
        return stats
    
    def save(self, filepath: str):
        """Save metrics to JSON file."""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_values': self.epsilon_values,
            'losses': self.losses,
            'timestamps': self.timestamps,
            'statistics': self.get_statistics()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load metrics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.episode_rewards = data['episode_rewards']
        self.episode_lengths = data['episode_lengths']
        self.epsilon_values = data['epsilon_values']
        self.losses = data.get('losses', [])
        self.timestamps = data.get('timestamps', [])


def save_checkpoint(model, optimizer, episode: int, metrics: TrainingMetrics,
                   filepath: str, additional_info: Optional[Dict] = None):
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        episode: Current episode number
        metrics: TrainingMetrics instance
        filepath: Path to save checkpoint
        additional_info: Optional additional information to save
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': {
            'episode_rewards': metrics.episode_rewards,
            'episode_lengths': metrics.episode_lengths,
            'epsilon_values': metrics.epsilon_values,
        },
        'statistics': metrics.get_statistics(),
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(filepath: str, model, optimizer=None):
    """
    Load training checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load state into
        optimizer: Optional PyTorch optimizer to load state into
    
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def evaluate_agent(env, agent, num_episodes: int = 10, verbose: bool = True) -> Dict:
    """
    Evaluate agent performance on environment.
    
    Args:
        env: Environment instance
        agent: DQN agent
        num_episodes: Number of episodes to evaluate
        verbose: Whether to print progress
    
    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    # Save current epsilon and set to 0 (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        if verbose:
            print(f"Evaluation Episode {ep+1}/{num_episodes} | Reward: {total_reward:.2f} | Steps: {steps}")
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    metrics = {
        'num_episodes': num_episodes,
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
    }
    
    if verbose:
        print(f"\nEvaluation Results:")
        print(f"  Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"  Reward Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
        print(f"  Average Length: {metrics['avg_length']:.2f} ± {metrics['std_length']:.2f}")
    
    return metrics

