# Model Visualization Guide

## Overview

The visualization tool allows you to see exactly how your trained DQN model makes decisions during story generation. You can observe:
- **Q-values** for each action (expected future rewards)
- **Available options** at each step (true continuation + 2 generated options)
- **Action chosen** by the model
- **Reward received** for each step
- **Story progression** with selected continuations
- **Overall statistics** across multiple episodes

## Quick Start

### Visualize Multiple Episodes (Recommended)

```bash
cd src
python visualize_model.py --episodes 5
```

This will show detailed information for 5 episodes, including:
- Q-values for all 3 actions at each step
- Which option the model chose
- Rewards received
- Story progression

### Visualize Single Story in Detail

```bash
cd src
python visualize_model.py --single
```

This shows one complete story with all details.

### Use Different Dataset Split

```bash
# Visualize on test split (default)
python visualize_model.py --episodes 3 --split test

# Visualize on dev split
python visualize_model.py --episodes 3 --split dev

# Visualize on train split
python visualize_model.py --episodes 3 --split train
```

## Output Format

### For Each Step:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📍 STEP 1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 Q-Values (Expected Future Reward):
  👉 Action 0 (True Continuation): 7.234
     Action 1 (Generated Option 1): 5.123
     Action 2 (Generated Option 2): 4.567

✅ Action Taken: 0 (True Continuation)

📋 Available Options:
  ✅ Option 0: She held up an orange sock and a blue one.
     Option 1: The room was filled with colorful decorations.
     Option 2: Everyone gathered around the table.

📝 Selected Continuation:
   She held up an orange sock and a blue one.

💰 Step Reward: 🟢 8.45
📈 Cumulative Reward: 🟢 8.45
```

### Episode Summary:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 EPISODE SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Steps: 4
Total Reward: 🟢 7.89
Average Reward per Step: 🟢 1.97
True Continuation Choices: 3 / 3
Generated Continuation Choices: 0 / 3
True Continuation Pick Rate: 100.0%
```

## Understanding the Output

### Q-Values
- **Higher Q-value** = Model expects higher future reward from that action
- **Action with highest Q-value** is typically chosen (when epsilon=0)
- Compare Q-values to see how confident the model is about each option

### Reward Colors
- 🟢 **Green**: High reward (≥ 7.0) - Good decision
- 🟡 **Yellow**: Medium reward (4.0-7.0) - Acceptable decision
- 🔴 **Red**: Low reward (< 4.0) - Poor decision

### Action Choices
- **Action 0**: True continuation from dataset (ground truth)
- **Action 1**: First generated continuation (from GPT-2)
- **Action 2**: Second generated continuation (from GPT-2)

### True Continuation Pick Rate
- Percentage of times model chose the true continuation
- **Higher is better** (indicates model learned to prefer good continuations)
- 100% = Always chose true continuation (perfect)
- 33% = Random choice (no learning)
- > 50% = Model learned something useful

## Example Use Cases

### 1. Validate Model Learning
```bash
python visualize_model.py --episodes 10
```
Check if the model consistently chooses true continuations (high pick rate).

### 2. Debug Poor Performance
```bash
python visualize_model.py --episodes 3 --split test
```
See which actions lead to low rewards and why.

### 3. Analyze Generated Options
Observe the quality of generated continuations and whether the model correctly identifies good vs. bad options.

### 4. Compare Across Splits
```bash
# Train split
python visualize_model.py --episodes 3 --split train

# Test split
python visualize_model.py --episodes 3 --split test
```
Compare performance to check for overfitting.

## Command Line Options

```
--episodes N       Number of episodes to visualize (default: 3)
--split SPLIT      Dataset split: train, dev, or test (default: test)
--single           Visualize a single story in detail
--model-path PATH  Path to model file (uses config if not specified)
```

## Tips

1. **Start with 3-5 episodes** to get a good overview
2. **Use `--single`** for detailed analysis of one story
3. **Check Q-values** to understand model confidence
4. **Look for patterns** in action choices
5. **Compare rewards** across different action types

## Troubleshooting

### Model Not Found
Make sure you've trained the model first:
```bash
python train_and_evaluate.py
```

### Wrong Action Size Error
Ensure `USE_GENERATION=True` in `config.py` matches your training configuration.

### No Options Shown
Make sure generation mode is enabled in config and the model was trained with generation.

## Next Steps

After visualization, you can:
1. **Adjust training parameters** if model isn't learning well
2. **Modify reward weights** if certain behaviors aren't being rewarded
3. **Improve generation quality** if generated options are poor
4. **Run baseline comparison** to see how your model compares to random/oracle

