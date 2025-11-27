# Story Generation Mode - Quick Start Guide

## Overview

The project now supports **story generation mode**, where the RL agent chooses among multiple continuation options:
- **Option 0**: True continuation (from the dataset)
- **Option 1**: Generated continuation 1 (from language model)
- **Option 2**: Generated continuation 2 (from language model)

This transforms the project from a story navigator to a true story teller that can generate new content.

## How It Works

1. At each step, the environment generates 2 alternative continuations using GPT-2
2. The agent is presented with 3 options: 1 true + 2 generated
3. The agent chooses which continuation to use (action 0, 1, or 2)
4. The selected continuation becomes part of the story
5. Rewards are calculated based on:
   - Whether the agent chose the true continuation (+1.0)
   - Quality of generated continuations (0.0-1.0)
   - Narrative coherence (semantic similarity)
   - Character consistency (if annotations available)

## Setup

### 1. Install Dependencies

The project already includes `transformers` in `requirements.txt`. Make sure it's installed:

```bash
pip install -r requirements.txt
```

### 2. Enable Generation Mode

Edit `config.py`:

```python
USE_GENERATION = True  # Enable multi-choice story continuation
```

### 3. Configure Generation Settings (Optional)

In `config.py`, you can adjust:

```python
GENERATION_CONFIG = {
    'model_name': 'gpt2',  # or 'distilgpt2' for faster/smaller model
    'num_generated_options': 2,
    'max_length': 50,
    'temperature_range': (0.7, 0.9)
}
```

## Running the Code

### Option 1: Test the Generation Functionality

Run the test script to verify everything works:

```bash
cd test
python test_generation.py
```

This will:
- Test the StoryGenerator class
- Test StoryEnv with generation mode
- Test MultiStoryEnvGym with generation mode

### Option 2: Train with Generation Mode

Train the DQN agent with story generation:

```bash
cd src
python dqn_train.py
```

Make sure `USE_GENERATION = True` in `config.py`.

**Expected Output:**
```
Loading dataset from .../rocstorysubset.csv...
Loaded 9885 stories from train split

🚀 Starting training on 9885 stories...
Episodes: 100 | State dim: 768 | Action size: 3 | Epsilon: 1.000
Enhanced rewards: False | Generation mode: True | Reward weights: None

Episode 0010 | Reward: 2.45 | Avg (last 10): 2.30 | ...
```

### Option 3: Quick Test with Single Story

For a quick test without the full dataset:

```bash
cd test
python test_generation.py
```

This uses the sample story from `data/story_sample.json`.

## Key Changes

### Action Space
- **Before**: 2 actions (control probability of following sequence)
- **After**: 3 actions (choose among continuations)

### State Space
- Unchanged: Still uses DistilBERT embeddings (768-dim or 800-dim with annotations)

### Reward Function
- **True continuation chosen**: +1.0 base reward
- **Generated continuation chosen**: 0.0-1.0 based on quality
- **Coherence**: +0.0-0.3 based on semantic similarity
- **Character consistency**: +0.0-0.2 (if annotations available)

## Files Modified

1. **`src/story_generator.py`** (NEW): Language model for generating continuations
2. **`src/story_env.py`**: Added generation mode support
3. **`src/dqn_train.py`**: Updated to handle 3 actions
4. **`config.py`**: Added `USE_GENERATION` flag and `GENERATION_CONFIG`
5. **`test/test_generation.py`** (NEW): Test script

## Troubleshooting

### Issue: "CUDA out of memory"
- Use a smaller model: Set `model_name='distilgpt2'` in `GENERATION_CONFIG`
- Or use CPU: The code automatically falls back to CPU if CUDA unavailable

### Issue: Generation is slow
- Use `distilgpt2` instead of `gpt2` (smaller, faster)
- Reduce `max_length` in `GENERATION_CONFIG`
- Use fewer stories for testing (`MAX_STORIES = 10` in config)

### Issue: Generated text is poor quality
- This is expected initially - the agent learns to choose better options
- The reward function encourages quality through coherence metrics
- Consider fine-tuning the language model on story data (future enhancement)

## Next Steps

1. **Train the agent**: Let it learn which continuations are best
2. **Monitor metrics**: Track how often agent chooses true vs. generated
3. **Adjust rewards**: Tune reward weights in `config.py` to balance exploration
4. **Scale up**: Once working, train on full dataset with more episodes

## Notes

- First run will download GPT-2 model (~500MB) - this is automatic
- Generation mode is slower than navigation mode (generates text each step)
- The agent learns to balance choosing true continuations vs. creative generated ones
- Generated stories are stored in `env.generated_story` for inspection

