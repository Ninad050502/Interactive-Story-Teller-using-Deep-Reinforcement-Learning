"""
Test script for story generation functionality
Tests the multi-choice story continuation approach
"""

import sys
import os

# Add parent directory to path for config import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from story_env import StoryEnv, MultiStoryEnvGym
from story_generator import StoryGenerator
from dataset_manager import DatasetManager
import config

def test_story_generator():
    """Test the StoryGenerator class."""
    print("=" * 60)
    print("Testing StoryGenerator")
    print("=" * 60)
    
    generator = StoryGenerator(model_name="gpt2")
    
    # Test context
    context = [
        "Nana came into the room with a puzzled look on her face.",
        "She held up an orange sock and a blue one."
    ]
    
    print(f"\nContext: {context}")
    print("\nGenerating continuations...")
    
    continuations = generator.generate_continuations(context, num_options=2)
    
    for i, cont in enumerate(continuations, 1):
        print(f"  Option {i}: {cont}")
    
    print("\n✅ StoryGenerator test complete!\n")

def test_story_env_with_generation():
    """Test StoryEnv with generation mode."""
    print("=" * 60)
    print("Testing StoryEnv with Generation Mode")
    print("=" * 60)
    
    story_path = os.path.join(os.path.dirname(__file__), "..", "data", "story_sample.json")
    
    generator = StoryGenerator(model_name="gpt2")
    env = StoryEnv(
        story_path=story_path,
        use_generation=True,
        story_generator=generator
    )
    
    state = env.reset()
    done = False
    total_reward = 0
    
    print(f"\nStory: {env.story_title}")
    print(f"Action space: 3 (0=true, 1=generated1, 2=generated2)")
    print("\n🎬 Starting Story Simulation with Generation...\n")
    
    step = 0
    while not done and step < 5:  # Limit to 5 steps for testing
        # Show current state
        current_line = env.generated_story[-1] if env.generated_story else env.story[env.idx]
        print(f"Step {step + 1}:")
        print(f"  Current: {current_line}")
        
        # Generate options (this happens in step, but we can preview)
        if env.idx + 1 < len(env.story):
            true_next = env.story[env.idx + 1]
            print(f"  True continuation: {true_next}")
        
        # Agent chooses action (for testing, we'll try different actions)
        action = step % 3  # Cycle through actions
        print(f"  Agent chooses action: {action}")
        
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if info.get('options'):
            print(f"  Options were:")
            for i, opt in enumerate(info['options']):
                marker = "✓" if i == action else " "
                print(f"    [{marker}] Option {i}: {opt[:80]}...")
        
        print(f"  Selected: {info['line'][:80]}...")
        print(f"  Reward: {reward:.2f}\n")
        
        step += 1
    
    print(f"✅ Simulation complete. Total Reward: {total_reward:.2f}\n")

def test_multi_story_with_generation():
    """Test MultiStoryEnvGym with generation mode."""
    print("=" * 60)
    print("Testing MultiStoryEnvGym with Generation Mode")
    print("=" * 60)
    
    # Check if config has the required attribute
    if not hasattr(config, 'CSV_STORIES_PATH'):
        print("⚠️  Config module doesn't have CSV_STORIES_PATH attribute")
        print("   Skipping multi-story test.")
        return
    
    csv_path = getattr(config, 'CSV_STORIES_PATH', None)
    if not csv_path or not os.path.exists(csv_path):
        print(f"⚠️  Dataset not found at {csv_path}")
        print("   Skipping multi-story test. Run with dataset for full test.")
        return
    
    # Load dataset with limited stories for testing
    csv_path = getattr(config, 'CSV_STORIES_PATH', None)
    json_annotations_path = getattr(config, 'JSON_ANNOTATIONS_PATH', None)
    
    dataset_manager = DatasetManager(
        csv_path=csv_path,
        json_annotations_path=None,  # Skip annotations for faster testing
        split='train',
        max_stories=5  # Just 5 stories for testing
    )
    
    generator = StoryGenerator(model_name="gpt2")
    env = MultiStoryEnvGym(
        dataset_manager,
        use_enhanced_rewards=False,
        use_generation=True,
        story_generator=generator
    )
    
    print(f"\nLoaded {len(dataset_manager)} stories")
    print(f"Action space: {env.action_space.n} actions")
    print(f"State dimension: {env.state_dim}")
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print("\n🎬 Testing Multi-Story Environment with Generation...\n")
    
    story_info = env.get_current_story_info()
    print(f"Story: {story_info['title']} (ID: {story_info['story_id']})\n")
    
    while not done and steps < 3:  # Limit steps for testing
        # Random action for testing
        import random
        action = random.randint(0, 2)
        
        next_state, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1
        
        print(f"Step {steps}:")
        print(f"  Action: {action}")
        print(f"  Line: {info['line'][:80]}...")
        print(f"  Reward: {reward:.2f}")
        if info.get('chose_true') is not None:
            print(f"  Chose true continuation: {info['chose_true']}")
        print()
    
    print(f"✅ Test complete. Total Reward: {total_reward:.2f}\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Story Generation Test Suite")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Story Generator
        test_story_generator()
        
        # Test 2: StoryEnv with generation
        test_story_env_with_generation()
        
        # Test 3: Multi-story with generation
        test_multi_story_with_generation()
        
        print("=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

