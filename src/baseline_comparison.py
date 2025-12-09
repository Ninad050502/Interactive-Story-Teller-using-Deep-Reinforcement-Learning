"""
Baseline Comparison Script
Compares DQN agent performance against Random and Oracle baselines
"""

import sys
import os
import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
except ImportError:
    config = None

from story_env import MultiStoryEnvGym
from dataset_manager import DatasetManager
from dqn_train import DQNAgent
from train_and_evaluate import load_trained_agent


class RandomBaseline:
    """Random baseline: chooses random action at each step."""
    
    def __init__(self, action_size: int):
        self.action_size = action_size
        self.name = "Random"
    
    def act(self, state):
        """Choose random action."""
        return np.random.randint(0, self.action_size)


class OracleBaseline:
    """Oracle baseline: always chooses true continuation (action 0) when available."""
    
    def __init__(self, action_size: int):
        self.action_size = action_size
        self.name = "Oracle"
    
    def act(self, state):
        """Always choose action 0 (true continuation)."""
        return 0


def evaluate_baseline(env: MultiStoryEnvGym, baseline, num_episodes: int = 100) -> Dict:
    """
    Evaluate a baseline on the environment.
    
    Args:
        env: Environment instance
        baseline: Baseline agent (RandomBaseline or OracleBaseline)
        num_episodes: Number of episodes to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    rewards = []
    episode_lengths = []
    true_continuation_picks = []
    ending_rewards = []
    ending_qualities = []
    
    # Progress indicator
    print(f"   Progress: ", end="", flush=True)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        chose_true_count = 0
        episode_ending_reward = 0.0
        episode_ending_quality = None
        
        while not done:
            action = baseline.act(state)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _, info = step_result
            
            total_reward += reward
            steps += 1
            
            # Track true continuation picks
            if info.get('chose_true') is not None:
                if info['chose_true']:
                    chose_true_count += 1
            
            # Track ending metrics
            if done:
                episode_ending_reward = info.get('ending_reward', 0.0)
                episode_ending_quality = info.get('ending_quality')
            
            state = next_state
        
        rewards.append(total_reward)
        episode_lengths.append(steps)
        true_continuation_picks.append(chose_true_count / max(1, steps))
        ending_rewards.append(episode_ending_reward)
        if episode_ending_quality is not None:
            ending_qualities.append(episode_ending_quality)
        
        # Progress indicator (every 10 episodes or at end)
        if (episode + 1) % 10 == 0 or (episode + 1) == num_episodes:
            print(f"{episode + 1}/{num_episodes} ", end="", flush=True)
    
    print()  # New line after progress
    
    return {
        'name': baseline.name,
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'avg_episode_length': np.mean(episode_lengths),
        'avg_true_pick_rate': np.mean(true_continuation_picks) if true_continuation_picks else 0.0,
        'avg_ending_reward': np.mean(ending_rewards) if ending_rewards else 0.0,
        'avg_ending_quality': np.mean(ending_qualities) if ending_qualities else None,
        'rewards': rewards
    }


def evaluate_dqn_agent(env: MultiStoryEnvGym, agent: DQNAgent, 
                       model_path: str, num_episodes: int = 100) -> Dict:
    """
    Evaluate trained DQN agent.
    
    Args:
        env: Environment instance
        agent: DQN agent (will be loaded from model_path)
        model_path: Path to saved model
        num_episodes: Number of episodes to evaluate
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load trained agent with correct dimensions
    state_dim = env.state_dim
    action_size = env.action_space.n
    agent = load_trained_agent(model_path, state_dim=state_dim, action_size=action_size)
    
    # Set epsilon to 0 for evaluation (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    rewards = []
    episode_lengths = []
    true_continuation_picks = []
    ending_rewards = []
    ending_qualities = []
    
    # Progress indicator
    print(f"   Progress: ", end="", flush=True)
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        chose_true_count = 0
        episode_ending_reward = 0.0
        episode_ending_quality = None
        
        while not done:
            action = agent.act(state, action_size=env.action_space.n)
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _, info = step_result
            
            total_reward += reward
            steps += 1
            
            # Track true continuation picks
            if info.get('chose_true') is not None:
                if info['chose_true']:
                    chose_true_count += 1
            
            # Track ending metrics
            if done:
                episode_ending_reward = info.get('ending_reward', 0.0)
                episode_ending_quality = info.get('ending_quality')
            
            state = next_state
        
        rewards.append(total_reward)
        episode_lengths.append(steps)
        true_continuation_picks.append(chose_true_count / max(1, steps))
        ending_rewards.append(episode_ending_reward)
        if episode_ending_quality is not None:
            ending_qualities.append(episode_ending_quality)
        
        # Progress indicator (every 10 episodes or at end)
        if (episode + 1) % 10 == 0 or (episode + 1) == num_episodes:
            print(f"{episode + 1}/{num_episodes} ", end="", flush=True)
    
    print()  # New line after progress
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    return {
        'name': 'DQN Agent',
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'avg_episode_length': np.mean(episode_lengths),
        'avg_true_pick_rate': np.mean(true_continuation_picks) if true_continuation_picks else 0.0,
        'avg_ending_reward': np.mean(ending_rewards) if ending_rewards else 0.0,
        'avg_ending_quality': np.mean(ending_qualities) if ending_qualities else None,
        'rewards': rewards
    }


def compare_baselines(num_episodes: int = 100, use_generation: bool = None,
                     use_stochastic_emotions: bool = None):
    """
    Compare DQN agent against Random and Oracle baselines.
    
    Args:
        num_episodes: Number of episodes for evaluation
        use_generation: Whether to use generation mode (None = use from config)
        use_stochastic_emotions: Whether to use stochastic emotions (None = use from config)
    """
    print("=" * 70)
    print("Baseline Comparison: Random vs Oracle vs DQN Agent")
    print("=" * 70)
    
    # Load dataset
    if not config or not os.path.exists(config.CSV_STORIES_PATH):
        print("Error: Dataset not found. Cannot run baseline comparison.")
        return
    
    # Use config defaults if not specified
    if use_generation is None:
        use_generation = getattr(config, 'USE_GENERATION', False) if config else False
    if use_stochastic_emotions is None:
        use_stochastic_emotions = getattr(config, 'USE_STOCHASTIC_EMOTIONS', True) if config else True
    
    dataset_manager = DatasetManager(
        csv_path=config.CSV_STORIES_PATH,
        json_annotations_path=config.JSON_ANNOTATIONS_PATH if config.USE_ANNOTATIONS else None,
        split='test',  # Use test split for evaluation
        max_stories=100  # Limit for faster evaluation
    )
    
    print(f"\nLoaded {len(dataset_manager)} stories from test split\n")
    
    # Create environment
    reward_weights = config.REWARD_WEIGHTS if config else None
    env = MultiStoryEnvGym(
        dataset_manager,
        use_enhanced_rewards=config.USE_ANNOTATIONS if config else False,
        reward_weights=reward_weights,
        use_generation=use_generation,
        use_stochastic_emotions=use_stochastic_emotions
    )
    
    action_size = env.action_space.n
    state_dim = env.state_dim
    
    print(f"Environment Configuration:")
    print(f"  Action space: {action_size} actions")
    print(f"  State dimension: {state_dim}")
    print(f"  Generation mode: {use_generation}")
    print(f"  Stochastic emotions: {use_stochastic_emotions}")
    print(f"  Enhanced rewards: {config.USE_ANNOTATIONS if config else False}\n")
    
    results = {}
    
    # Evaluate Random Baseline
    print("Evaluating Random Baseline...")
    print(f"   (This may take a few minutes with generation mode - generating 2 continuations per step)")
    random_baseline = RandomBaseline(action_size)
    results['random'] = evaluate_baseline(env, random_baseline, num_episodes)
    print(f"Random baseline complete\n")
    
    # Evaluate Oracle Baseline
    print("Evaluating Oracle Baseline...")
    print(f"   (This may take a few minutes with generation mode - generating 2 continuations per step)")
    oracle_baseline = OracleBaseline(action_size)
    results['oracle'] = evaluate_baseline(env, oracle_baseline, num_episodes)
    print(f"Oracle baseline complete\n")
    
    # Evaluate DQN Agent
    print("Evaluating DQN Agent...")
    print(f"   (This may take a few minutes with generation mode - generating 2 continuations per step)")
    model_path = config.MODEL_SAVE_PATH if config else "models/saved_dqn.pt"
    
    # Create a dummy agent (will be replaced in evaluate_dqn_agent)
    # The actual agent will be loaded with correct dimensions inside evaluate_dqn_agent
    dummy_agent = DQNAgent(state_size=state_dim, action_size=action_size)
    results['dqn'] = evaluate_dqn_agent(env, dummy_agent, model_path, num_episodes)
    print(f"DQN agent evaluation complete\n")
    
    # Print comparison
    print("=" * 70)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 70)
    
    for name in ['random', 'oracle', 'dqn']:
        r = results[name]
        print(f"\n{r['name']}:")
        print(f"  Average Reward: {r['avg_reward']:.2f} ± {r['std_reward']:.2f}")
        print(f"  Reward Range: [{r['min_reward']:.2f}, {r['max_reward']:.2f}]")
        print(f"  Avg Episode Length: {r['avg_episode_length']:.1f}")
        if r['avg_true_pick_rate'] > 0:
            print(f"  True Continuation Pick Rate: {r['avg_true_pick_rate']*100:.1f}%")
        if r['avg_ending_reward'] > 0:
            print(f"  Avg Ending Reward: {r['avg_ending_reward']:.2f}")
        if r['avg_ending_quality'] is not None:
            print(f"  Avg Ending Quality: {r['avg_ending_quality']:.2f}")
    
    # Improvement over random
    if 'random' in results and 'dqn' in results:
        improvement = results['dqn']['avg_reward'] - results['random']['avg_reward']
        improvement_pct = (improvement / abs(results['random']['avg_reward']) * 100) if results['random']['avg_reward'] != 0 else 0
        print(f"\nDQN Improvement over Random: {improvement:+.2f} ({improvement_pct:+.1f}%)")
    
    # Gap to oracle
    if 'oracle' in results and 'dqn' in results:
        gap = results['oracle']['avg_reward'] - results['dqn']['avg_reward']
        gap_pct = (gap / abs(results['oracle']['avg_reward']) * 100) if results['oracle']['avg_reward'] != 0 else 0
        print(f"Gap to Oracle: {gap:.2f} ({gap_pct:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Baseline comparison complete!")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare baselines with DQN agent')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--no-generation', action='store_true',
                       help='Disable generation mode (default: generation enabled)')
    parser.add_argument('--no-stochastic-emotions', action='store_true',
                       help='Disable stochastic emotional transitions (default: use from config)')
    
    args = parser.parse_args()
    
    # Default to generation=True, unless --no-generation is specified
    use_gen = False if args.no_generation else True
    # If --no-stochastic-emotions is specified, disable; otherwise None (will use config)
    use_stoch = False if args.no_stochastic_emotions else None
    
    compare_baselines(
        num_episodes=args.episodes,
        use_generation=use_gen,
        use_stochastic_emotions=use_stoch
    )

