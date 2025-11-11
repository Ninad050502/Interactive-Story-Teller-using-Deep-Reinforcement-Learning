# Interactive Story Teller using Deep Reinforcement Learning

A reinforcement learning project that trains an agent to navigate through story narratives by learning to make sequential decisions that maintain narrative coherence. The project uses Deep Q-Network (DQN) to learn optimal actions for story progression.

## 📋 Project Overview

This project implements an interactive story-telling environment where a reinforcement learning agent learns to navigate through story narratives by learning to make sequential decisions that maintain narrative coherence. The project uses Deep Q-Network (DQN) to learn optimal actions for story progression.

**NEW: Now supports the StoryCommonsense dataset with 14,738 annotated stories!**

The agent uses:
- **DistilBERT** for encoding story sentences into dense vector representations (768-dimensional embeddings)
- **Character Annotations** (optional) for enhanced state representation (emotions, motivations)
- **Deep Q-Network (DQN)** for learning optimal action policies
- **Enhanced Rewards** based on sequence correctness, character consistency, and narrative coherence
- **Gymnasium** for a standardized RL environment interface

### Project Aim

The goal is to train an RL agent that can:
- Understand story context through sentence embeddings
- Leverage character annotations for richer state representation
- Learn to choose actions that maintain narrative coherence
- Navigate through story sequences optimally across multiple stories

## 🏗️ Project Structure

```
Interactive-Story-Teller-using-Deep-Reinforcement-Learning/
├── data/
│   ├── story_sample.json                    # Legacy sample story data
│   └── storycommonsense_data/               # StoryCommonsense dataset
│       ├── rocstorysubset.csv               # ~15,000 stories (CSV format)
│       ├── json_version/
│       │   └── annotations.json            # Character annotations (JSON)
│       └── storyid_partition.txt           # Train/dev/test splits
├── src/
│   ├── dataset_loader.py                   # Story loading utilities (CSV + JSON)
│   ├── dataset_manager.py                  # Dataset management (NEW)
│   ├── state_encoder.py                    # DistilBERT encoder + character features
│   ├── story_env.py                        # RL environment (single + multi-story)
│   ├── reward_calculator.py                # Enhanced reward calculation (NEW)
│   ├── dqn_train.py                        # DQN agent and training script
│   └── training_utils.py                   # Training utilities (NEW)
├── test/
│   └── explore_env.py                      # Environment exploration script
├── models/                                 # Saved model directory
│   └── saved_dqn.pt                        # Trained DQN model weights
├── config.py                               # Configuration file (NEW)
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## 🔍 How It Works

### 1. **Story Loading** (`dataset_loader.py` & `dataset_manager.py`)
- **Legacy Mode**: Loads stories from JSON format (backward compatible)
- **New Mode**: Loads stories from CSV (StoryCommonsense dataset)
  - Supports train/dev/test splits
  - Loads character annotations from JSON when available
  - Supports batch loading of multiple stories
- Preserves sentence order (5 sentences per story)

### 2. **State Encoding** (`state_encoder.py`)
- **Base Mode**: Uses DistilBERT (`distilbert-base-uncased`) to encode sentences
  - Converts text into 768-dimensional dense vectors
  - Mean-pools token embeddings for sentence-level representations
- **Enhanced Mode** (with annotations):
  - Includes character emotion features (8-dim Plutchik emotions)
  - Includes character motivation features (5-dim Maslow + 19-dim Reiss)
  - Total state dimension: 800 (768 + 8 + 24)

### 3. **Environment** (`story_env.py`)
- **StoryEnv**: Core environment that manages story progression
  - Supports both single story (legacy) and multi-story modes
  - Tracks current position in story
  - Uses `next_prob` parameter to control story flow
  - **Enhanced Rewards** (when annotations available):
    - Sequence reward: +1.0 for correct sequence, -1.0 for skipping
    - Character consistency reward: Based on emotion/motivation continuity
    - Narrative coherence reward: Based on semantic similarity
- **MultiStoryEnvGym**: Gymnasium-compatible wrapper for multi-story training
  - Action space: 2 discrete actions
    - Action 0: High probability (0.9) of following story
    - Action 1: Low probability (0.4) of following story
  - Observation space: 768 or 800-dimensional continuous vectors (depending on annotations)

### 4. **Reward Calculation** (`reward_calculator.py`) - NEW
- **Sequence Reward**: +1.0 for correct sequence, -1.0 for skipping
- **Character Consistency Reward**: Measures consistency of character emotions/motivations between consecutive lines
- **Narrative Coherence Reward**: Uses cosine similarity of state embeddings to measure story flow
- **Weighted Sum**: Configurable weights for each component (default: 1.0, 0.5, 0.3)

### 5. **DQN Agent** (`dqn_train.py`)
- **Q-Network**: 3-layer MLP (768/800 → 256 → 256 → 2)
- **Target Network**: Separate network for stable Q-learning
- **Experience Replay**: Stores and samples past experiences (buffer size: 10,000)
- **Epsilon-Greedy**: Balances exploration vs exploitation
  - Starts at ε=1.0 (fully random)
  - Decays to ε=0.1 (mostly greedy)
  - Decay rate: 0.995 per episode
- **Target Update**: Updates target network every 10 episodes

### 6. **Training Process**
- **Multi-Story Training**: Agent trains on multiple stories from dataset
- Collects experiences (state, action, reward, next_state) across stories
- Learns from random batches of past experiences
- Updates target network periodically for stability
- Tracks metrics: average reward, best performance, buffer size
- Saves trained model weights after training


## 🚀 Getting Started

### Prerequisites

- Python 3.11 (or compatible version)
- pip package manager

### Installation Steps

1. **Clone or navigate to the project directory:**
   ```bash
   cd Interactive-Story-Teller-using-Deep-Reinforcement-Learning
   ```

2. **Create a virtual environment:**
   ```bash
   python3.11 -m venv venv
   ```
   
   If Python 3.11 is not available, you can use:
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**
   
   On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
   
   On Windows:
   ```bash
   venv\Scripts\activate
   ```

4. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Running the Project

#### Option 1: Explore the Environment (Test Script)

Run the test script to see how the environment works without training:

```bash
python test/explore_env.py
```

**Expected Output:**
```
🎬 Starting Story Simulation...

→ She held up an orange sock and a blue one.   | Reward: 1.0
→ My daughter jumped up and grabbed the blue one out of her hand.   | Reward: 1.0
→ She took off running down the hall while waving the sock in the air.   | Reward: 1.0
→ Nana chased her down, caught her, and tickled her until she laughed.   | Reward: 1.0

✅ Simulation complete. Total Reward: 4.0
```

#### Option 2: Train the DQN Agent (Legacy - Single Story)

Train the agent on a single story (legacy mode):

```bash
cd src
python dqn_train.py
```

If the StoryCommonsense dataset is not found, it will automatically use legacy mode.

#### Option 3: Train the DQN Agent (Multi-Story Dataset) - RECOMMENDED

Train the agent on the full StoryCommonsense dataset:

```bash
cd src
python dqn_train.py
```

**Expected Output (with annotations):**
```
Loading dataset from .../rocstorysubset.csv...
Loaded 9885 stories from train split
Loaded annotations for 14738 stories

🚀 Starting training on 9885 stories...
Episodes: 1000 | State dim: 800 | Epsilon: 1.000
Enhanced rewards: True | Reward weights: {'sequence': 1.0, 'character_consistency': 0.5, 'narrative_coherence': 0.3}

Episode 0010 | Reward: 5.23 | Avg (last 10): 4.85 | Max: 5.23 | Min: 3.20 | Epsilon: 0.951 | Buffer: 45
Episode 0020 | Reward: 4.50 | Avg (last 10): 4.72 | Max: 5.23 | Min: 3.10 | Epsilon: 0.905 | Buffer: 95
...
✅ Training complete. Model saved to models/saved_dqn.pt

📊 Training Statistics:
  Final epsilon: 0.100
  Total experiences: 5000
  Average reward (all episodes): 4.65
  Best average reward (last 10): 4.85 at episode 10
  Final average reward (last 10): 4.72
  Max reward: 5.23 | Min reward: 2.10
```

The trained model will be saved to `models/saved_dqn.pt`.

#### Option 4: Complete Pipeline (Train + Evaluate) - **BEST FOR END-TO-END**

Run the complete pipeline: train the model, then automatically evaluate on dev and test splits:

```bash
cd src
python train_and_evaluate.py
```

**This will:**
1. ✅ Load dataset from CSV
2. ✅ Load annotations from JSON (if enabled)
3. ✅ Encode sentences with DistilBERT
4. ✅ Train DQN agent on train split
5. ✅ Save trained model
6. ✅ Evaluate on dev split
7. ✅ Evaluate on test split
8. ✅ Print comparison results

**Expected Output:**
```
============================================================
🚀 DQN Training and Evaluation Pipeline
============================================================

============================================================
📚 STEP 1: Training the Model
============================================================
[Training output...]
✅ Training complete. Model saved to models/saved_dqn.pt

============================================================
📊 STEP 2: Evaluating on DEV Split
============================================================
📊 DEV Split Evaluation Summary
  Average Reward: 4.75 ± 0.45
  Reward Range: [3.20, 5.23]

============================================================
📊 STEP 3: Evaluating on TEST Split
============================================================
📊 TEST Split Evaluation Summary
  Average Reward: 4.68 ± 0.52
  Reward Range: [3.10, 5.15]

============================================================
📊 FINAL RESULTS: Dev vs Test Comparison
============================================================
  Dev  - Average Reward: 4.75 ± 0.45
  Test - Average Reward: 4.68 ± 0.52
  Difference (Test - Dev): -0.07
  ✅ Good generalization (small difference between dev and test)
============================================================
✅ Complete pipeline finished!
```

**To skip training and only evaluate** (if model already exists):
Edit `train_and_evaluate.py` and set `skip_training=True` in the main block.

### Understanding the Output

- **Episode**: Training episode number
- **Reward**: Total reward for the episode
  - **Without annotations**: `4.00` = perfect story following (maximum for 5-line story)
  - **With annotations**: Can exceed 4.00 due to enhanced rewards (character consistency + narrative coherence)
  - Lower values indicate mistakes or exploration
- **State dim**: State dimension (768 without annotations, 800 with annotations)
- **Epsilon**: Exploration rate (higher = more random, lower = more greedy)
- **Buffer**: Number of experiences stored in replay buffer
- **Enhanced rewards**: Whether character annotations are being used

### Configuration

The project uses `config.py` for configuration. Key settings:

- **Dataset paths**: CSV stories and JSON annotations
- **Training split**: 'train', 'dev', or 'test'
- **Max stories**: Limit number of stories for testing (None = use all)
- **Use annotations**: Enable/disable character annotations
- **Reward weights**: Weights for sequence, character consistency, and narrative coherence rewards
- **Agent settings**: Learning rate, batch size, buffer size, epsilon decay, etc.

## 📝 Customization

### Modify Training Parameters

Edit `config.py` to change:
- Number of episodes: `TRAINING_CONFIG['episodes']`
- Learning rate: `AGENT_CONFIG['lr']`
- Discount factor: `AGENT_CONFIG['gamma']`
- Batch size: `AGENT_CONFIG['batch_size']`
- Reward weights: `REWARD_WEIGHTS`
- Training split: `TRAIN_SPLIT` ('train', 'dev', or 'test')
- Max stories: `MAX_STORIES` (None = use all, or specify number for testing)

### Use Different Dataset Splits

Edit `config.py`:
```python
TRAIN_SPLIT = 'train'  # or 'dev' or 'test'
MAX_STORIES = 100  # Limit for testing, or None for all
```

### Enable/Disable Enhanced Features

Edit `config.py`:
```python
USE_ANNOTATIONS = True  # Enable character annotations
INCLUDE_CHARACTER_FEATURES = True  # Include in state encoding
```

### Adjust Reward Weights

Edit `config.py`:
```python
REWARD_WEIGHTS = {
    'sequence': 1.0,  # Weight for sequence correctness
    'character_consistency': 0.5,  # Weight for character consistency
    'narrative_coherence': 0.3  # Weight for narrative coherence
}
```

### Add More Stories (Legacy Format)

1. Create new JSON files in `data/` directory following the format:
   ```json
   {
     "lines": {
       "1": {"text": "First sentence."},
       "2": {"text": "Second sentence."},
       ...
     },
     "title": "Story Title",
     "storyid": "unique_id"
   }
   ```

2. Update the path in `config.py` or use directly in training script

## 🔧 Troubleshooting

### Import Errors
- Ensure you're in the correct directory
- Activate the virtual environment
- Check that all dependencies are installed
- If `config` module not found, ensure `config.py` is in the project root

### Path Issues
- Make sure to run scripts from the correct directory
- Use absolute paths if relative paths fail
- Check that `config.py` paths point to correct dataset locations

### Model Saving Issues
- Create `models/` directory if it doesn't exist:
  ```bash
  mkdir -p models
  ```

### Dataset Loading Issues
- Ensure `rocstorysubset.csv` exists in `data/storycommonsense_data/`
- For enhanced rewards, ensure `annotations.json` exists in `data/storycommonsense_data/json_version/`
- Check file permissions and encoding (should be UTF-8)

### State Dimension Mismatch
- If you see dimension errors, ensure state dimension matches between encoder and agent
- With annotations: 800 dimensions (768 + 32 character features)
- Without annotations: 768 dimensions
- The environment automatically detects and uses the correct dimension

## 🆕 New Features

### StoryCommonsense Dataset Integration
- **14,738 annotated stories** from the StoryCommonsense dataset
- Support for train/dev/test splits
- Character-level annotations (emotions, motivations)

### Enhanced State Encoding
- **Character Emotion Features**: 8-dimensional Plutchik emotions
- **Character Motivation Features**: 5-dimensional Maslow + 19-dimensional Reiss motivations
- **Total State Dimension**: 800 (768 sentence + 32 character features)

### Enhanced Rewards
- **Sequence Reward**: Basic reward for following correct sequence
- **Character Consistency Reward**: Measures consistency of character emotions/motivations
- **Narrative Coherence Reward**: Uses semantic similarity for story flow
- **Configurable Weights**: Adjust importance of each reward component

### Multi-Story Training
- Train on multiple stories from the dataset
- Automatic story sampling and switching
- Better generalization across different story types

### Training Utilities
- Metrics tracking and statistics
- Model checkpointing
- Evaluation utilities
- Progress logging

