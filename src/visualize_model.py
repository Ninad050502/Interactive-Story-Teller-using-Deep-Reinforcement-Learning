"""
Visualize Model Behavior - Interactive Story Teller
Shows detailed information about model decisions, rewards, and story progression
"""

import torch
import os
import sys
import numpy as np
from typing import Dict, List, Optional

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


def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_section(title: str, char="="):
    """Print a section header."""
    print_separator(char)
    print(f"  {title}")
    print_separator(char)


def format_reward(reward: float) -> str:
    """Format reward with color coding."""
    if reward >= 7.0:
        return f"🟢 {reward:.2f}"  # High reward (green)
    elif reward >= 4.0:
        return f"🟡 {reward:.2f}"  # Medium reward (yellow)
    else:
        return f"🔴 {reward:.2f}"  # Low reward (red)


def visualize_episode(env: MultiStoryEnvGym, agent: DQNAgent, 
                      episode_num: int = 1, verbose: bool = True) -> Dict:
    """
    Visualize a single episode with detailed information.
    
    Returns:
        Dictionary with episode statistics
    """
    # Set epsilon to 0 for deterministic evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    step = 0
    episode_info = {
        'steps': [],
        'total_reward': 0.0,
        'story_id': None,
        'story_title': None,
        'chose_true_count': 0,
        'chose_generated_count': 0
    }
    
    # Get story info
    story_info = env.get_current_story_info()
    if story_info:
        episode_info['story_id'] = story_info.get('story_id', 'unknown')
        episode_info['story_title'] = story_info.get('title', 'Unknown')
    
    if verbose:
        print_section(f"📖 EPISODE {episode_num}: {episode_info['story_title']}")
        print(f"Story ID: {episode_info['story_id']}")
        print()
    
    while not done:
        step += 1
        
        # Get Q-values for all actions
        with torch.no_grad():
            state_tensor = torch.tensor(state).float().unsqueeze(0)
            q_values = agent.q_net(state_tensor).squeeze().cpu().numpy()
            action = agent.act(state, action_size=env.action_space.n)
        
        # Take step (Gymnasium returns 5 values: state, reward, terminated, truncated, info)
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            # Fallback for older gym versions
            next_state, reward, done, info = step_result
        
        # Collect step information
        step_info = {
            'step': step,
            'action': action,
            'reward': reward,
            'q_values': q_values.tolist(),
            'line': info.get('line', 'N/A'),
            'chose_true': info.get('chose_true', None),
            'options': info.get('options', None),
            'ending_quality': info.get('ending_quality', None),
            'ending_reward': info.get('ending_reward', 0.0)
        }
        episode_info['steps'].append(step_info)
        total_reward += reward
        
        # Track action choices
        if step_info['chose_true'] is not None:
            if step_info['chose_true']:
                episode_info['chose_true_count'] += 1
            else:
                episode_info['chose_generated_count'] += 1
        
        # Display step information
        if verbose:
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"📍 STEP {step}")
            print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            
            # Show Q-values
            print(f"\n🎯 Q-Values (Expected Future Reward):")
            action_names = ["True Continuation", "Generated Option 1", "Generated Option 2"]
            for i, (q_val, name) in enumerate(zip(q_values, action_names)):
                marker = "👉" if i == action else "  "
                print(f"  {marker} Action {i} ({name}): {q_val:.3f}")
            
            # Show action taken
            action_name = action_names[action]
            print(f"\n✅ Action Taken: {action} ({action_name})")
            
            # Show options if available
            if step_info['options']:
                print(f"\n📋 Available Options:")
                for i, option in enumerate(step_info['options']):
                    marker = "✅" if i == action else "  "
                    print(f"  {marker} Option {i}: {option[:80]}..." if len(option) > 80 else f"  {marker} Option {i}: {option}")
            
            # Show selected line
            print(f"\n📝 Selected Continuation:")
            print(f"   {step_info['line']}")
            
            # Show reward
            print(f"\n💰 Step Reward: {format_reward(reward)}")
            if step_info['ending_reward'] > 0:
                print(f"   🎁 Ending Quality Bonus: +{step_info['ending_reward']:.2f}")
                if step_info['ending_quality']:
                    print(f"   📊 Ending Quality Score: {step_info['ending_quality']:.2f}")
            
            # Show cumulative reward
            print(f"\n📈 Cumulative Reward: {format_reward(total_reward)}")
            print()
        
        state = next_state
    
    episode_info['total_reward'] = total_reward
    
    # Restore epsilon
    agent.epsilon = original_epsilon
    
    if verbose:
        print_separator()
        print(f"📊 EPISODE SUMMARY")
        print_separator()
        print(f"Total Steps: {step}")
        print(f"Total Reward: {format_reward(total_reward)}")
        print(f"Average Reward per Step: {format_reward(total_reward / step)}")
        print(f"True Continuation Choices: {episode_info['chose_true_count']} / {max(1, step - 1)}")
        print(f"Generated Continuation Choices: {episode_info['chose_generated_count']} / {max(1, step - 1)}")
        if step > 1:
            true_rate = episode_info['chose_true_count'] / (step - 1) * 100
            print(f"True Continuation Pick Rate: {true_rate:.1f}%")
        print_separator()
        print()
    
    return episode_info


def visualize_multiple_episodes(num_episodes: int = 5, 
                                model_path: Optional[str] = None,
                                split: str = 'test',
                                max_stories: Optional[int] = None):
    """
    Visualize multiple episodes with detailed information.
    
    Args:
        num_episodes: Number of episodes to visualize
        model_path: Path to trained model (uses config if None)
        split: Dataset split to use ('train', 'dev', 'test')
        max_stories: Optional limit on stories
    """
    print_separator("=", 80)
    print("  🎬 MODEL BEHAVIOR VISUALIZATION")
    print_separator("=", 80)
    print()
    
    # Load dataset
    if not config or not os.path.exists(config.CSV_STORIES_PATH):
        print("❌ Dataset not found. Cannot visualize.")
        return
    
    csv_path = config.CSV_STORIES_PATH
    json_path = config.JSON_ANNOTATIONS_PATH if config.USE_ANNOTATIONS else None
    
    dataset_manager = DatasetManager(
        csv_path=csv_path,
        json_annotations_path=json_path if config.USE_ANNOTATIONS else None,
        split=split,
        max_stories=max_stories or 50  # Limit for visualization
    )
    
    print(f"📚 Loaded {len(dataset_manager)} stories from {split} split\n")
    
    # Create environment
    reward_weights = config.REWARD_WEIGHTS if config else None
    use_stochastic_emotions = getattr(config, 'USE_STOCHASTIC_EMOTIONS', True) if config else True
    use_generation = getattr(config, 'USE_GENERATION', False) if config else False
    
    env = MultiStoryEnvGym(
        dataset_manager,
        use_enhanced_rewards=config.USE_ANNOTATIONS if config else False,
        reward_weights=reward_weights,
        use_generation=use_generation,
        use_stochastic_emotions=use_stochastic_emotions
    )
    
    state_dim = env.state_dim
    action_size = env.action_space.n
    
    print(f"⚙️  Environment Configuration:")
    print(f"   State dimension: {state_dim}")
    print(f"   Action size: {action_size}")
    print(f"   Generation mode: {use_generation}")
    print(f"   Enhanced rewards: {config.USE_ANNOTATIONS if config else False}")
    print()
    
    # Load trained agent
    model_path = model_path or (config.MODEL_SAVE_PATH if config else "../models/saved_dqn.pt")
    agent = load_trained_agent(model_path, state_dim=state_dim, action_size=action_size)
    
    print(f"✅ Model loaded successfully\n")
    print_separator("=", 80)
    print()
    
    # Visualize episodes
    all_episodes = []
    for ep in range(num_episodes):
        episode_info = visualize_episode(env, agent, episode_num=ep + 1, verbose=True)
        all_episodes.append(episode_info)
        
        if ep < num_episodes - 1:
            print("\n" + "=" * 80 + "\n")
    
    # Summary statistics
    print_separator("=", 80)
    print("  📊 OVERALL STATISTICS")
    print_separator("=", 80)
    
    total_rewards = [ep['total_reward'] for ep in all_episodes]
    true_choice_counts = [ep['chose_true_count'] for ep in all_episodes]
    total_steps = sum(len(ep['steps']) for ep in all_episodes)
    
    print(f"\nEpisodes Visualized: {num_episodes}")
    print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Reward Range: [{np.min(total_rewards):.2f}, {np.max(total_rewards):.2f}]")
    print(f"Total Steps: {total_steps}")
    print(f"Average Steps per Episode: {total_steps / num_episodes:.1f}")
    
    if total_steps > 0:
        total_true_choices = sum(true_choice_counts)
        true_rate = total_true_choices / (total_steps - num_episodes) * 100  # Subtract num_episodes for first steps
        print(f"Overall True Continuation Pick Rate: {true_rate:.1f}%")
    
    print()
    print_separator("=", 80)
    print("✅ Visualization complete!")
    print_separator("=", 80)


def visualize_single_story(story_id: Optional[str] = None,
                           model_path: Optional[str] = None,
                           split: str = 'test'):
    """
    Visualize a single story in detail.
    
    Args:
        story_id: Specific story ID to visualize (random if None)
        model_path: Path to trained model
        split: Dataset split to use
    """
    print_separator("=", 80)
    print("  📖 SINGLE STORY VISUALIZATION")
    print_separator("=", 80)
    print()
    
    # Load dataset
    if not config or not os.path.exists(config.CSV_STORIES_PATH):
        print("❌ Dataset not found.")
        return
    
    csv_path = config.CSV_STORIES_PATH
    json_path = config.JSON_ANNOTATIONS_PATH if config.USE_ANNOTATIONS else None
    
    dataset_manager = DatasetManager(
        csv_path=csv_path,
        json_annotations_path=json_path if config.USE_ANNOTATIONS else None,
        split=split,
        max_stories=1000  # Load enough to find specific story if needed
    )
    
    # Create environment
    reward_weights = config.REWARD_WEIGHTS if config else None
    use_stochastic_emotions = getattr(config, 'USE_STOCHASTIC_EMOTIONS', True) if config else True
    use_generation = getattr(config, 'USE_GENERATION', False) if config else False
    
    env = MultiStoryEnvGym(
        dataset_manager,
        use_enhanced_rewards=config.USE_ANNOTATIONS if config else False,
        reward_weights=reward_weights,
        use_generation=use_generation,
        use_stochastic_emotions=use_stochastic_emotions
    )
    
    # If story_id specified, we'd need to modify environment to load specific story
    # For now, just visualize a random one
    
    state_dim = env.state_dim
    action_size = env.action_space.n
    
    # Load trained agent
    model_path = model_path or (config.MODEL_SAVE_PATH if config else "../models/saved_dqn.pt")
    agent = load_trained_agent(model_path, state_dim=state_dim, action_size=action_size)
    
    print(f"✅ Model loaded successfully\n")
    
    # Visualize the episode
    visualize_episode(env, agent, episode_num=1, verbose=True)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize model behavior')
    parser.add_argument('--episodes', type=int, default=3, 
                       help='Number of episodes to visualize (default: 3)')
    parser.add_argument('--split', type=str, default='test', 
                       choices=['train', 'dev', 'test'],
                       help='Dataset split to use (default: test)')
    parser.add_argument('--single', action='store_true',
                       help='Visualize a single story in detail')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to model file (uses config if not specified)')
    
    args = parser.parse_args()
    
    if args.single:
        visualize_single_story(model_path=args.model_path, split=args.split)
    else:
        visualize_multiple_episodes(
            num_episodes=args.episodes,
            model_path=args.model_path,
            split=args.split
        )

