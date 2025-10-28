import sys
import os

# Add src folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from story_env import StoryEnv

# Initialize environment
# env = StoryEnv(os.path.join("..", "data", "story_sample.json"))
env = StoryEnv(os.path.join(os.path.dirname(__file__), "..", "data", "story_sample.json"))
# env = StoryEnv("..\data\story_sample.json")

state = env.reset()
done = False
total_reward = 0

print("\n🎬 Starting Story Simulation...\n")

while not done:
    action = 0  # Placeholder since no RL yet
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    print(f"→ {info['line']}   | Reward: {reward}")

print("\n✅ Simulation complete. Total Reward:", total_reward)
