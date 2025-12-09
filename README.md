# Interactive Story Teller using Deep Reinforcement Learning

A reinforcement learning project that trains a DQN agent to navigate through story narratives by learning to make sequential decisions that maintain narrative coherence.

## 🚀 Setup

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
   ls data/storycommonsense_data/rocstorysubset.csv
   ls data/storycommonsense_data/json_version/annotations.json
   ```

## 🏃 Running the Project

### Main Training and Evaluation

Run the complete training and evaluation pipeline:

```bash
cd src
python train_and_evaluate.py
```

**What this does:**
- Trains DQN agent for 1,000 episodes on training split
- Saves model to `models/saved_dqn.pt`
- Evaluates on train/dev/test splits
- Shows comparison across all splits with generalization analysis

**Note:** By default, the script trains a new model. To skip training and only evaluate an existing model, set `skip_training=True` in `train_and_evaluate.py` (line 379).

### Getting Results and Metrics

After running `train_and_evaluate.py`, you can get additional results using:

#### 1. Baseline Comparison
Compare your trained DQN agent against Random and Oracle baselines:

```bash
cd src
python baseline_comparison.py --episodes 100
```

**Output:** Shows average rewards, true continuation pick rates, and improvement metrics for Random, Oracle, and DQN agent.

#### 2. Model Visualization
Visualize detailed model behavior and decisions:

```bash
cd src
python visualize_model.py
```

**Output:** Shows detailed episode-by-episode visualization with rewards, actions, and story progression.

## 📁 Folder Structure

```
Interactive-Story-Teller-using-Deep-Reinforcement-Learning/
├── data/
│   ├── story_sample.json                    # Legacy sample story data
│   ├── storycommonsense_data/               # StoryCommonsense dataset
│   │   ├── rocstorysubset.csv               # ~15,000 stories (CSV format)
│   │   ├── json_version/
│   │   │   └── annotations.json             # Character annotations (JSON)
│   │   └── storyid_partition.txt           # Train/dev/test splits
│   └── atomic_data/                         # ATOMIC dataset (if used)
├── src/
│   ├── dataset_loader.py                    # Story loading utilities
│   ├── dataset_manager.py                   # Dataset management
│   ├── state_encoder.py                     # DistilBERT encoder + character features
│   ├── story_env.py                         # RL environment
│   ├── reward_calculator.py                 # Reward calculation
│   ├── dqn_train.py                         # DQN agent and training
│   ├── train_and_evaluate.py               # Main training/evaluation pipeline
│   ├── baseline_comparison.py              # Baseline comparison script
│   ├── visualize_model.py                  # Model visualization script
│   ├── story_generator.py                  # DistilGPT-2 story generation
│   ├── emotional_transition.py            # Stochastic emotional transitions
│   └── training_utils.py                   # Training utilities
├── test/
│   └── explore_env.py                      # Environment exploration script
├── models/                                 # Saved model directory
│   └── saved_dqn.pt                        # Trained DQN model weights
├── figures/                                # Generated figures
├── config.py                               # Configuration file
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## 📄 File Descriptions

### Core Training Files

- **`train_and_evaluate.py`** - Main entry point. Trains the DQN agent and evaluates on train/dev/test splits. Produces performance metrics and generalization analysis.

- **`dqn_train.py`** - Contains the DQN agent implementation (QNetwork, DQNAgent classes) and training loop. Used by `train_and_evaluate.py`.

- **`baseline_comparison.py`** - Compares trained DQN agent against Random and Oracle baselines. Shows performance metrics and improvement statistics.

- **`visualize_model.py`** - Visualizes model behavior with detailed episode-by-episode analysis, showing rewards, actions, and story progression.

### Environment and Data Files

- **`story_env.py`** - Implements the RL environment (StoryEnv and MultiStoryEnvGym). Manages story progression, actions, and rewards.

- **`dataset_manager.py`** - Manages loading and accessing stories from the StoryCommonsense dataset with train/dev/test splits.

- **`dataset_loader.py`** - Low-level utilities for loading stories from CSV and JSON formats.

### State and Reward Files

- **`state_encoder.py`** - Encodes story sentences using DistilBERT. Optionally includes character emotion/motivation features and scene index.

- **`reward_calculator.py`** - Calculates multi-component rewards (sequence, character consistency, narrative coherence, ending quality).

### Additional Features

- **`story_generator.py`** - Generates alternative story continuations using DistilGPT-2 (used when generation mode is enabled).

- **`emotional_transition.py`** - Implements stochastic emotional transitions for character annotations.

- **`training_utils.py`** - Utility functions for training and evaluation (e.g., `evaluate_agent`).

### Configuration

- **`config.py`** - Central configuration file. Contains training settings, agent hyperparameters, feature flags, and dataset paths.

### Test Files

- **`test/explore_env.py`** - Simple script to explore and test the environment without training.

## ⚙️ Configuration

Key settings in `config.py`:

- **`TRAINING_CONFIG`** - Number of episodes, evaluation frequency
- **`AGENT_CONFIG`** - Learning rate, batch size, epsilon decay, etc.
- **`USE_ANNOTATIONS`** - Enable/disable character annotations
- **`USE_GENERATION`** - Enable/disable story generation mode
- **`REWARD_WEIGHTS`** - Weights for different reward components

### Quick Configuration for Faster Training

To quickly test the training and evaluation pipeline with fewer episodes:

1. **Reduce number of training episodes:**
   Edit `config.py` and change:
   ```python
   TRAINING_CONFIG = {
       'episodes': 100,  # Change from 1000 to 100 for faster testing
       ...
   }
   ```

2. **Limit number of stories (optional):**
   Edit `config.py` and change:
   ```python
   MAX_STORIES = 50  # Change from None to limit dataset size
   ```

**Example for quick testing:**
- Set `episodes: 100` in `TRAINING_CONFIG` → Trains for 100 episodes instead of 1000
- Set `MAX_STORIES = 50` → Uses only 50 stories instead of all ~9,885 training stories

**Note:** For full training, use default values: `episodes: 1000`, `MAX_STORIES = None`

## 📊 Understanding Results

### Key Metrics

- **Average Reward**: Mean episode reward (higher is better)
- **Standard Deviation**: Variability in performance (lower = more stable)
- **True Continuation Pick Rate**: Fraction of times agent selects true continuation
- **Episode Length**: Average number of steps per episode (typically 5)

### Good Results Indicators

- ✅ DQN > Random: Agent learned meaningful strategies
- ✅ DQN close to Oracle: Agent learned well (within 10-15% of oracle)
- ✅ Small dev-test gap (< 0.5): Good generalization
- ✅ High true pick rate (> 70%): Agent prefers coherent continuations

---

**Last Updated**: 2025
