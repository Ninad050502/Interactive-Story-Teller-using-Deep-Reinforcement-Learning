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
│   ├── dataset_manager.py                  # Dataset management
│   ├── state_encoder.py                    # DistilBERT encoder + character features
│   ├── story_env.py                        # RL environment (single + multi-story)
│   ├── reward_calculator.py                # Enhanced reward calculation
│   ├── dqn_train.py                        # DQN agent and training script
│   ├── train_and_evaluate.py               # Complete training and evaluation pipeline
│   ├── baseline_comparison.py              # Baseline evaluation script
│   ├── story_generator.py                  # GPT-2 story generation
│   ├── emotional_transition.py            # Stochastic emotional transitions
│   └── training_utils.py                   # Training utilities
├── test/
│   └── explore_env.py                      # Environment exploration script
├── models/                                 # Saved model directory
│   └── saved_dqn.pt                        # Trained DQN model weights
├── config.py                               # Configuration file
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+ (or compatible version)
- pip package manager
- ~2GB disk space for dataset and models

### Installation

1. **Navigate to the project directory:**
   ```bash
   cd Interactive-Story-Teller-using-Deep-Reinforcement-Learning
   ```

2. **Create and activate virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Windows)
   venv\Scripts\activate
   
   # Activate (macOS/Linux)
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify dataset is present:**
   ```bash
   # Check dataset files exist
   ls data/storycommonsense_data/rocstorysubset.csv
   ls data/storycommonsense_data/json_version/annotations.json
   ```

## 📖 Usage Guide

### Recommended Workflow

The recommended workflow consists of three main steps:

1. **Train and Evaluate** → Train DQN agent and evaluate on all splits
2. **Baseline Comparison** → Compare DQN against Random and Oracle baselines
3. **Analysis** → Review results and metrics

---

### Step 1: Train and Evaluate Model

**Option A: Complete Pipeline (Train + Evaluate)**

Train the model and automatically evaluate on train/dev/test splits:

```bash
cd src
python train_and_evaluate.py
```

**What this does:**
- ✅ Trains DQN agent for 1,000 episodes on training split
- ✅ Saves model to `models/saved_dqn.pt`
- ✅ Evaluates on train split (no exploration)
- ✅ Evaluates on dev split
- ✅ Evaluates on test split
- ✅ Shows comparison across all splits with generalization analysis

**Expected Output:**
```
============================================================
🚀 DQN Training and Evaluation Pipeline
============================================================

📚 STEP 1: Training the Model
============================================================
Loading dataset from .../rocstorysubset.csv...
Loaded 9885 stories from train split
Loaded annotations for 14738 stories

🚀 Starting training on 9885 stories...
Episodes: 1000 | State dim: 801 | Action size: 3 | Epsilon: 1.000

Episode 0010 | Reward: 5.23 | Avg (last 10): 4.85 | Epsilon: 0.951
Episode 0020 | Reward: 4.50 | Avg (last 10): 4.72 | Epsilon: 0.905
...
✅ Training complete. Model saved to models/saved_dqn.pt

📊 STEP 1.5: Evaluating on TRAIN Split
============================================================
📊 TRAIN Split Evaluation Summary
  Average Reward: 4.85 ± 0.45
  Reward Range: [3.20, 5.23]

📊 STEP 2: Evaluating on DEV Split
============================================================
📊 DEV Split Evaluation Summary
  Average Reward: 4.75 ± 0.45
  Reward Range: [3.20, 5.23]

📊 STEP 3: Evaluating on TEST Split
============================================================
📊 TEST Split Evaluation Summary
  Average Reward: 4.68 ± 0.52
  Reward Range: [3.10, 5.15]

📊 FINAL RESULTS: Train vs Dev vs Test Comparison
============================================================
  Train - Average Reward: 4.85 ± 0.45
  Dev   - Average Reward: 4.75 ± 0.45
  Test  - Average Reward: 4.68 ± 0.52
  ✅ Good generalization (small differences across all splits)
```

**Time:** ~30-60 minutes (depending on episodes and dataset size)

---

**Option B: Evaluation Only (Skip Training)**

If you already have a trained model, the script is configured to skip training by default (`skip_training=True`). It will:
- Load existing model from `models/saved_dqn.pt`
- Evaluate on train/dev/test splits
- Show comparison results

**Note:** Ensure your `config.py` settings match how the model was trained (annotations, generation mode, etc.).

---

### Step 2: Baseline Comparison

Compare your trained DQN agent against Random and Oracle baselines:

```bash
cd src
python baseline_comparison.py --episodes 100
```

**Command Options:**
- `--episodes N`: Number of evaluation episodes (default: 100)
- `--generation`: Use generation mode (if you trained with generation)
- `--stochastic-emotions`: Use stochastic emotions (default: True)

**Example with generation mode:**
```bash
python baseline_comparison.py --episodes 100 --generation
```

**Expected Output:**
```
============================================================
Baseline Comparison: Random vs Oracle vs DQN Agent
============================================================

🎲 Evaluating Random Baseline...
✅ Random baseline complete

🔮 Evaluating Oracle Baseline...
✅ Oracle baseline complete

🤖 Evaluating DQN Agent...
✅ Loaded model from models/saved_dqn.pt
✅ DQN agent evaluation complete

============================================================
📊 BASELINE COMPARISON RESULTS
============================================================

Random:
  Average Reward: 2.34 ± 0.45
  True Continuation Pick Rate: 33.3%

Oracle:
  Average Reward: 5.23 ± 0.12
  True Continuation Pick Rate: 100.0%

DQN Agent:
  Average Reward: 4.87 ± 0.38
  True Continuation Pick Rate: 78.5%

📈 DQN Improvement over Random: +2.53 (+108.1%)
📉 Gap to Oracle: -0.36 (-6.9%)
```

**Time:** ~5-10 minutes (depending on episodes)

---

### Step 3: Explore Environment (Optional)

Test the environment without training:

```bash
python test/explore_env.py
```

This runs a simple story simulation to verify the environment works correctly.

---

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
  - Includes scene index (1-dim normalized position)
  - Total state dimension: 801 (768 + 32 + 1)

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
  - Action space: 2 discrete actions (or 3 with generation mode)
    - Action 0: High probability (0.9) of following story
    - Action 1: Low probability (0.4) of following story
    - Action 2: Generated continuation (if generation mode enabled)
  - Observation space: 769 or 801-dimensional continuous vectors (depending on annotations)

### 4. **Reward Calculation** (`reward_calculator.py`)
- **Sequence Reward**: +1.0 for correct sequence, -1.0 for skipping
- **Character Consistency Reward**: Measures consistency of character emotions/motivations between consecutive lines
- **Narrative Coherence Reward**: Uses cosine similarity of state embeddings to measure story flow
- **Ending Quality Reward**: Up to +5.0 for satisfying story endings
- **Weighted Sum**: Configurable weights for each component (default: 1.0, 0.5, 0.3)

### 5. **DQN Agent** (`dqn_train.py`)
- **Q-Network**: 3-layer MLP (state_dim → 256 → 256 → action_size)
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

## ⚙️ Configuration

All configuration is managed through `config.py`. Key settings:

### Training Settings

```python
TRAINING_CONFIG = {
    'episodes': 1000,              # Number of training episodes
    'eval_frequency': 50,          # Evaluate every N episodes
    'save_frequency': 100,         # Save checkpoint every N episodes
}
```

### Agent Settings

```python
AGENT_CONFIG = {
    'gamma': 0.9,                  # Discount factor
    'lr': 1e-3,                    # Learning rate
    'batch_size': 32,              # Batch size for experience replay
    'buffer_size': 10000,          # Replay buffer size
    'epsilon_decay': 0.995,        # Epsilon decay rate
    'epsilon_min': 0.1,            # Minimum epsilon
    'target_update_frequency': 10  # Update target network every N episodes
}
```

### Feature Flags

```python
USE_ANNOTATIONS = True             # Use character annotations
USE_GENERATION = True              # Use story generation mode
USE_STOCHASTIC_EMOTIONS = True     # Use stochastic emotional transitions
INCLUDE_SCENE_INDEX = True         # Include scene index in state
```

### Reward Weights

```python
REWARD_WEIGHTS = {
    'sequence': 1.0,               # Weight for sequence correctness
    'character_consistency': 0.5,  # Weight for character consistency
    'narrative_coherence': 0.3      # Weight for narrative coherence
}
```

### Quick Configuration Changes

**For faster training (testing):**
```python
TRAINING_CONFIG['episodes'] = 100
MAX_STORIES = 50  # Limit dataset size
USE_GENERATION = False  # Disable generation (faster)
```

**For full features:**
```python
USE_GENERATION = True
USE_STOCHASTIC_EMOTIONS = True
USE_ANNOTATIONS = True
```

## 📊 Understanding Results

### Metrics Explained

- **Average Reward**: Mean episode reward (higher is better)
  - Without annotations: Max ~4.0 (perfect 5-line story)
  - With annotations: Can exceed 4.0 due to enhanced rewards
- **Standard Deviation**: Variability in performance (lower = more stable)
- **True Continuation Pick Rate**: Fraction of times agent selects true continuation
- **Episode Length**: Average number of steps per episode (typically 5 for 5-sentence stories)

### Interpreting Results

**Good Results:**
- ✅ **DQN > Random**: Agent learned meaningful strategies
- ✅ **DQN close to Oracle**: Agent learned well (within 10-15% of oracle)
- ✅ **Small dev-test gap** (< 0.5): Good generalization
- ✅ **High true pick rate** (> 70%): Agent prefers coherent continuations

**Areas for Improvement:**
- ⚠️ **Large train-dev gap** (> 1.0): Possible overfitting
- ⚠️ **DQN ≈ Random**: Agent didn't learn (check hyperparameters, reward function)
- ⚠️ **Large dev-test gap** (> 0.5): Poor generalization
- ⚠️ **Low true pick rate** (< 50%): Agent may need more training or better rewards

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Solution: Ensure you're in the correct directory and virtual environment is activated
cd src
# Verify imports work
python -c "import config; print('Config loaded')"
```

**2. Model Not Found**
```bash
# Solution: Check model path exists
ls models/saved_dqn.pt

# Or update config.py with correct path
MODEL_SAVE_PATH = "path/to/your/model.pt"
```

**3. Dimension Mismatch Errors**
- **Cause**: Config settings don't match how model was trained
- **Solution**: Ensure `USE_ANNOTATIONS`, `USE_GENERATION`, etc. match training configuration
- **Check**: State dimension should be 801 (with annotations) or 769 (without)

**4. Dataset Not Found**
```bash
# Solution: Verify dataset files exist
ls data/storycommonsense_data/rocstorysubset.csv
ls data/storycommonsense_data/json_version/annotations.json

# Check paths in config.py
CSV_STORIES_PATH = "data/storycommonsense_data/rocstorysubset.csv"
```

**5. Out of Memory**
- **Solution**: Reduce batch size or buffer size in `config.py`
- **Alternative**: Limit dataset size with `MAX_STORIES = 100`

**6. Slow Training**
- **Solution**: 
  - Set `USE_GENERATION = False` (faster)
  - Reduce `TRAINING_CONFIG['episodes']`
  - Set `MAX_STORIES = 100` for testing

## 📁 Dataset Information

### StoryCommonsense Dataset

- **Total Stories**: 14,738 annotated stories
- **Format**: CSV (stories) + JSON (annotations)
- **Splits**: 
  - Train: 9,885 stories
  - Dev: 2,427 stories
  - Test: 2,426 stories
- **Story Length**: 5 sentences per story
- **Annotations**: Character emotions (Plutchik) and motivations (Maslow + Reiss)

### Dataset Structure

Each story contains:
- 5 sequential sentences
- Character annotations per sentence (emotions, motivations)
- Story ID and title

## 🎯 Key Components

### State Encoding
- **Base**: DistilBERT embeddings (768-dim)
- **Enhanced**: + Character emotions (8-dim) + Motivations (24-dim) + Scene index (1-dim) = 801-dim

### Action Space
- **Standard Mode**: 2 actions (high/low probability of following story)
- **Generation Mode**: 3 actions (true continuation + 2 generated options)

### Reward Function
- **Sequence Reward**: +1.0 for correct sequence, -1.0 for skipping
- **Character Consistency**: Measures emotion/motivation continuity
- **Narrative Coherence**: Cosine similarity between consecutive states
- **Ending Quality**: Up to +5.0 for satisfying endings

### DQN Architecture
- **Q-Network**: 3-layer MLP (state_dim → 256 → 256 → action_size)
- **Experience Replay**: Buffer size 10,000, batch size 32
- **Target Network**: Updated every 10 episodes
- **Epsilon-Greedy**: Decays from 1.0 to 0.1 over training

## 📚 Additional Resources

- **Project Proposal**: `Project_Proposal_DRL.txt`
- **Literature Survey**: `DRL_Literature_Survey.txt`
- **Workflow Guide**: `RUN_WORKFLOW.md`
- **Implementation Summary**: `IMPLEMENTATION_SUMMARY.md`

## ✅ Quick Reference

**Train and Evaluate:**
```bash
cd src && python train_and_evaluate.py
```

**Compare Baselines:**
```bash
cd src && python baseline_comparison.py --episodes 100
```

**Explore Environment:**
```bash
python test/explore_env.py
```

**Check Configuration:**
```python
# Edit config.py to modify settings
```

---

**Last Updated**: 2025
