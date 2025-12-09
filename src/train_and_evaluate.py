"""
Train and Evaluate DQN Model - Complete End-to-End Pipeline
This script trains the model and then evaluates it on dev and test splits.
"""

import torch
import os
import sys
import numpy as np
from typing import Optional

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import config
except ImportError:
    config = None

# Import from dqn_train - we're using existing training code, not duplicating it
from dqn_train import DQNAgent, QNetwork, train_dqn_multi_story
from dataset_manager import DatasetManager
from story_env import MultiStoryEnvGym
from training_utils import evaluate_agent


def load_trained_agent(model_path: str, state_dim: int = 768, action_size: int = None):
    """
    Load trained agent from saved model.
    
    Args:
        model_path: Path to saved model
        state_dim: State dimension (768, 769, 800, or 801)
        action_size: Action size (2 or 3). If None, will try to infer from config or model.
    
    Returns:
        DQNAgent instance with loaded weights
    """
    # Determine action size
    if action_size is None:
        # Try to infer from config (should match training)
        if config:
            use_generation = getattr(config, 'USE_GENERATION', False)
            action_size = 3 if use_generation else 2
        else:
            # Fallback: try to infer from saved model
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                # Check the last layer size
                if 'net.4.weight' in checkpoint:
                    action_size = checkpoint['net.4.weight'].shape[0]
                else:
                    action_size = 2  # Default fallback
            except:
                action_size = 2  # Default fallback
    
    # Create agent with same config as training
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
    agent_config['action_size'] = action_size  # Use determined action size
    
    agent = DQNAgent(**agent_config)
    
    # Load trained weights
    if os.path.exists(model_path):
        agent.q_net.load_state_dict(torch.load(model_path, map_location='cpu'))
        agent.target_net.load_state_dict(agent.q_net.state_dict())
        print(f"Loaded model from {model_path} (action_size={action_size}, state_dim={state_dim})")
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    return agent


def evaluate_on_split(split: str, model_path: str, 
                     num_episodes: int = 100,
                     max_stories: Optional[int] = None,
                     use_annotations: bool = True,
                     verbose: bool = True):
    """
    Evaluate trained model on dev or test split.
    
    Args:
        split: 'dev' or 'test'
        model_path: Path to trained model
        num_episodes: Number of evaluation episodes
        max_stories: Optional limit on stories (for testing)
        use_annotations: Whether to use annotations
        verbose: Whether to print progress
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {split.upper()} split")
    print(f"{'='*60}\n")
    
    # Load dataset for evaluation split
    csv_path = config.CSV_STORIES_PATH if config else "../data/storycommonsense_data/rocstorysubset.csv"
    json_path = config.JSON_ANNOTATIONS_PATH if config and use_annotations else None
    
    dataset_manager = DatasetManager(
        csv_path=csv_path,
        json_annotations_path=json_path if use_annotations else None,
        split=split,
        max_stories=max_stories
    )
    
    print(f"Loaded {len(dataset_manager)} stories from {split} split")
    
    # Create environment
    reward_weights = config.REWARD_WEIGHTS if config else None
    # Get config settings
    use_stochastic_emotions = getattr(config, 'USE_STOCHASTIC_EMOTIONS', True) if config else True
    use_generation = getattr(config, 'USE_GENERATION', False) if config else False
    
    env = MultiStoryEnvGym(
        dataset_manager,
        use_enhanced_rewards=use_annotations,
        reward_weights=reward_weights,
        use_generation=use_generation,
        use_stochastic_emotions=use_stochastic_emotions
    )
    
    # Get state dimension and action size from environment
    state_dim = env.state_dim
    action_size = env.action_space.n
    print(f"State dimension: {state_dim}")
    print(f"Action size: {action_size}")
    
    # Load trained agent (use action size from environment to match training)
    agent = load_trained_agent(model_path, state_dim=state_dim, action_size=action_size)
    
    # Evaluate agent (epsilon = 0, no exploration)
    print(f"\nEvaluating for {num_episodes} episodes (no exploration)...")
    metrics = evaluate_agent(env, agent, num_episodes=num_episodes, verbose=verbose)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"{split.upper()} Split Evaluation Summary")
    print(f"{'='*60}")
    print(f"  Episodes: {metrics['num_episodes']}")
    print(f"  Average Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Reward Range: [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
    print(f"  Average Steps: {metrics['avg_length']:.2f} ± {metrics['std_length']:.2f}")
    print(f"{'='*60}\n")
    
    return metrics


def train_and_evaluate(training_episodes: int = 1000,
                      train_max_stories: Optional[int] = None,
                      dev_episodes: int = 100,
                      test_episodes: int = 100,
                      eval_max_stories: Optional[int] = None,
                      use_annotations: bool = True,
                      skip_training: bool = True):
    """
    Complete workflow: Train model and evaluate on dev/test splits.
    
    Args:
        training_episodes: Number of training episodes
        train_max_stories: Optional limit on training stories
        dev_episodes: Number of evaluation episodes on dev split
        test_episodes: Number of evaluation episodes on test split
        eval_max_stories: Optional limit on evaluation stories
        use_annotations: Whether to use annotations
        skip_training: If True, skip training and only evaluate (model must exist)
    
    Returns:
        Dictionary with dev and test evaluation metrics
    """
    print("="*60)
    print("DQN Training and Evaluation Pipeline")
    print("="*60)
    
    # Get model path
    model_path = config.MODEL_SAVE_PATH if config else "../models/saved_dqn.pt"
    
    # Step 1: Train the model
    train_metrics = None
    if not skip_training:
        print("\n" + "="*60)
        print("STEP 1: Training the Model")
        print("="*60)
        print("This step includes:")
        print("  - Loading dataset from CSV")
        print("  - Loading annotations from JSON (if enabled)")
        print("  - Encoding sentences with DistilBERT")
        print("  - Training DQN agent")
        print("  - Saving trained model")
        print("="*60)
        
        csv_path = config.CSV_STORIES_PATH if config else "../data/storycommonsense_data/rocstorysubset.csv"
        json_path = config.JSON_ANNOTATIONS_PATH if config and use_annotations else None
        split = config.TRAIN_SPLIT if config else 'train'
        
        # Get generation mode setting
        use_generation = getattr(config, 'USE_GENERATION', False) if config else False
        
        # Call the training function from dqn_train.py
        # This executes all the training code: data loading, embedding, training, saving
        training_rewards = train_dqn_multi_story(
            csv_path=csv_path,
            json_annotations_path=json_path if use_annotations else None,
            split=split,
            max_stories=train_max_stories or (config.MAX_STORIES if config else None),
            episodes=training_episodes,
            use_annotations=use_annotations,
            use_generation=use_generation
        )
        
        # Calculate training metrics from rewards
        if training_rewards:
            train_metrics = {
                'num_episodes': len(training_rewards),
                'avg_reward': np.mean(training_rewards),
                'std_reward': np.std(training_rewards),
                'max_reward': np.max(training_rewards),
                'min_reward': np.min(training_rewards),
                'final_avg_reward': np.mean(training_rewards[-10:]) if len(training_rewards) >= 10 else np.mean(training_rewards),
                'avg_length': 5.0  # Stories have 5 sentences, so average length is ~5 steps
            }
        
        print(f"\nTraining complete. Model saved to {model_path}")
    else:
        print("\nSkipping training (using existing model)")
        if not os.path.exists(model_path):
            print(f"Error: Model not found at {model_path}")
            print("Please train the model first or set skip_training=False")
            return None
    
    # Step 1.5: Evaluate on train split (with epsilon=0, no exploration)
    print("\n" + "="*60)
    print("STEP 1.5: Evaluating on TRAIN Split (No Exploration)")
    print("="*60)
    print("This step includes:")
    print("  - Loading train dataset")
    print("  - Loading trained model")
    print("  - Running evaluation (no exploration, epsilon=0)")
    print("  - Computing metrics")
    print("="*60)
    
    train_eval_metrics = evaluate_on_split(
        split='train',
        model_path=model_path,
        num_episodes=dev_episodes,  # Use same number as dev for consistency
        max_stories=eval_max_stories,
        use_annotations=use_annotations,
        verbose=True
    )
    
    # Step 2: Evaluate on dev split
    print("\n" + "="*60)
    print("STEP 2: Evaluating on DEV Split")
    print("="*60)
    print("This step includes:")
    print("  - Loading dev dataset")
    print("  - Loading trained model")
    print("  - Running evaluation (no exploration)")
    print("  - Computing metrics")
    print("="*60)
    
    dev_metrics = evaluate_on_split(
        split='dev',
        model_path=model_path,
        num_episodes=dev_episodes,
        max_stories=eval_max_stories,
        use_annotations=use_annotations,
        verbose=True
    )
    
    # Step 3: Evaluate on test split
    print("\n" + "="*60)
    print("STEP 3: Evaluating on TEST Split")
    print("="*60)
    print("This step includes:")
    print("  - Loading test dataset")
    print("  - Loading trained model")
    print("  - Running evaluation (no exploration)")
    print("  - Computing metrics")
    print("="*60)
    
    test_metrics = evaluate_on_split(
        split='test',
        model_path=model_path,
        num_episodes=test_episodes,
        max_stories=eval_max_stories,
        use_annotations=use_annotations,
        verbose=True
    )
    
    # Step 4: Print final comparison (Train, Dev, Test)
    print("\n" + "="*60)
    print("FINAL RESULTS: Train vs Dev vs Test Comparison")
    print("="*60)
    print(f"  Train - Average Reward: {train_eval_metrics['avg_reward']:.2f} ± {train_eval_metrics['std_reward']:.2f}")
    print(f"  Train - Reward Range: [{train_eval_metrics['min_reward']:.2f}, {train_eval_metrics['max_reward']:.2f}]")
    print(f"  Train - Average Steps: {train_eval_metrics['avg_length']:.2f} ± {train_eval_metrics['std_length']:.2f}")
    print()
    print(f"  Dev   - Average Reward: {dev_metrics['avg_reward']:.2f} ± {dev_metrics['std_reward']:.2f}")
    print(f"  Dev   - Reward Range: [{dev_metrics['min_reward']:.2f}, {dev_metrics['max_reward']:.2f}]")
    print(f"  Dev   - Average Steps: {dev_metrics['avg_length']:.2f} ± {dev_metrics['std_length']:.2f}")
    print()
    print(f"  Test  - Average Reward: {test_metrics['avg_reward']:.2f} ± {test_metrics['std_reward']:.2f}")
    print(f"  Test  - Reward Range: [{test_metrics['min_reward']:.2f}, {test_metrics['max_reward']:.2f}]")
    print(f"  Test  - Average Steps: {test_metrics['avg_length']:.2f} ± {test_metrics['std_length']:.2f}")
    print()
    print(f"  Differences:")
    print(f"    Dev - Train: {dev_metrics['avg_reward'] - train_eval_metrics['avg_reward']:.2f}")
    print(f"    Test - Dev:  {test_metrics['avg_reward'] - dev_metrics['avg_reward']:.2f}")
    print(f"    Test - Train: {test_metrics['avg_reward'] - train_eval_metrics['avg_reward']:.2f}")
    
    # Check for overfitting
    train_dev_diff = abs(dev_metrics['avg_reward'] - train_eval_metrics['avg_reward'])
    dev_test_diff = abs(test_metrics['avg_reward'] - dev_metrics['avg_reward'])
    
    if train_dev_diff < 0.5 and dev_test_diff < 0.5:
        print("  Good generalization (small differences across all splits)")
    elif train_dev_diff > 1.0:
        print("  Warning: Large gap between train and dev - possible overfitting")
    elif dev_test_diff > 0.5:
        print("  Warning: Large difference between dev and test - possible overfitting")
    else:
        print("  Warning: Moderate differences - monitor for overfitting")
    
    print("="*60)
    print("Complete pipeline finished!")
    print("="*60)
    
    return {
        'train': train_eval_metrics,
        'dev': dev_metrics,
        'test': test_metrics,
        'training_metrics': train_metrics  # Training metrics during training (with exploration)
    }


if __name__ == "__main__":
    # Configuration - use getattr with defaults for safety
    TRAINING_EPISODES = getattr(config, 'TRAINING_CONFIG', {}).get('episodes', 1000) if config else 1000
    DEV_EPISODES = 100  # Number of episodes to evaluate on dev split
    TEST_EPISODES = 100  # Number of episodes to evaluate on test split
    
    # Optional: Limit stories for faster testing
    # Set to None to use all stories, or set to a number for faster testing
    TRAIN_MAX_STORIES = None  # None = use all training stories
    EVAL_MAX_STORIES = None   # None = use all evaluation stories
    
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    print(f"  Training episodes: {TRAINING_EPISODES}")
    print(f"  Dev evaluation episodes: {DEV_EPISODES}")
    print(f"  Test evaluation episodes: {TEST_EPISODES}")
    print(f"  Train max stories: {TRAIN_MAX_STORIES if TRAIN_MAX_STORIES else 'All'}")
    print(f"  Eval max stories: {EVAL_MAX_STORIES if EVAL_MAX_STORIES else 'All'}")
    print(f"  Use annotations: {getattr(config, 'USE_ANNOTATIONS', True) if config else True}")
    print(f"  Use generation: {getattr(config, 'USE_GENERATION', False) if config else False}")
    print(f"  Stochastic emotions: {getattr(config, 'USE_STOCHASTIC_EMOTIONS', True) if config else True}")
    print("="*60)
    
    # Run complete pipeline
    results = train_and_evaluate(
        training_episodes=TRAINING_EPISODES,
        train_max_stories=TRAIN_MAX_STORIES,
        dev_episodes=DEV_EPISODES,
        test_episodes=TEST_EPISODES,
        eval_max_stories=EVAL_MAX_STORIES,
        use_annotations=getattr(config, 'USE_ANNOTATIONS', True) if config else True,
        skip_training=False  # Set to False to train, True to skip training and only evaluate
    )
    
    if results:
        print("\nAll steps completed successfully!")
        print(f"   Model saved at: {getattr(config, 'MODEL_SAVE_PATH', '../models/saved_dqn.pt') if config else '../models/saved_dqn.pt'}")
        print(f"\nFinal Performance Summary:")
        print(f"   Train performance: {results['train']['avg_reward']:.2f} ± {results['train']['std_reward']:.2f}")
        print(f"   Dev performance:   {results['dev']['avg_reward']:.2f} ± {results['dev']['std_reward']:.2f}")
        print(f"   Test performance:  {results['test']['avg_reward']:.2f} ± {results['test']['std_reward']:.2f}")
        
        if results.get('training_metrics'):
            print(f"\nTraining Performance (with exploration):")
            print(f"   Average reward during training: {results['training_metrics']['avg_reward']:.2f} ± {results['training_metrics']['std_reward']:.2f}")
            print(f"   Final average (last 10): {results['training_metrics']['final_avg_reward']:.2f}")

