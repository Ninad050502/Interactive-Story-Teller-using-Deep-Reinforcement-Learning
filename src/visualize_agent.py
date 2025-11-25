import os
import sys
import torch
import random
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset_manager import DatasetManager
from story_env import MultiStoryEnvGym
from atomic_continuation_generator import create_atomic_generator
from train_and_evaluate import load_trained_agent

def visualize_agent(num_stories=3, model_path=None):
    """
    Visualize the agent playing through stories.
    """
    print(f"\n{'='*60}")
    print(f"🎬 Visualizing Agent on {num_stories} Stories")
    print(f"{'='*60}\n")
    
    # Path to model
    if model_path is None:
        model_path = config.MODEL_SAVE_PATH
        
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return

    # Load dataset (Test split)
    print("Loading test stories...")
    dataset_manager = DatasetManager(
        csv_path=config.CSV_STORIES_PATH,
        json_annotations_path=config.JSON_ANNOTATIONS_PATH if config.USE_ANNOTATIONS else None,
        split='test',
        max_stories=num_stories * 2 # Load a few more to skip short ones if any
    )
    
    # Create ATOMIC generator
    atomic_generator = None
    if config.USE_ATOMIC:
        atomic_generator = create_atomic_generator(config.ATOMIC_CONFIG)
        
    # Create environment
    env = MultiStoryEnvGym(
        dataset_manager,
        use_enhanced_rewards=config.USE_ANNOTATIONS,
        reward_weights=config.REWARD_WEIGHTS,
        atomic_generator=atomic_generator
    )
    
    # Load agent
    try:
        agent = load_trained_agent(
            model_path=model_path,
            state_dim=env.state_dim,
            action_size=env.action_space.n
        )
    except Exception as e:
        print(f"❌ Failed to load agent: {e}")
        return

    # Run visualization
    for i in range(num_stories):
        print(f"\n📖 Story {i+1}:")
        print(f"{'-'*40}")
        
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        
        # Get story title
        story_title = env.inner_env.story_title
        print(f"Title: {story_title}")
        
        while not done:
            # Get available options (if ATOMIC)
            options = []
            if config.USE_ATOMIC and env.inner_env.current_continuations:
                options = env.inner_env.current_continuations
            
            # Get action from agent
            action = agent.act(state)
            
            # Take step
            next_state, reward, done, _, info = env.step(action)
            
            # Print step info
            print(f"Step {step+1}:")
            
            # Show options if available
            if options:
                print(f"  Options:")
                for idx, (opt_text, relation) in enumerate(options):
                    # Action 0 is usually the "True Next" in this setup if it's the first option
                    # But let's check the relation
                    is_true = (relation == 'true')
                    marker = "✅ (True Next)" if is_true else f"❌ ({relation})"
                    print(f"    [{idx}] {opt_text} [{marker}]")
            
            action_desc = "Selected Option" if options else ("Follow Story" if action == 0 else "Random/Skip")
            print(f"  🤖 Agent Action: {action} ({action_desc})")
            print(f"  🎁 Reward: {reward:.2f}")
            print(f"  📝 Result Line: {info['line']}")
            print()
            
            state = next_state
            total_reward += reward
            step += 1
            
        print(f"✨ Story Complete! Total Reward: {total_reward:.2f}")
        print(f"{'='*60}")

if __name__ == "__main__":
    visualize_agent()
