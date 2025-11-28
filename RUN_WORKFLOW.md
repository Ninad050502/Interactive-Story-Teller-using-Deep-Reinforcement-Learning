# Complete Workflow Guide

## 🎯 Recommended Workflow

### Step 1: Train and Evaluate Model

Run the complete training and evaluation pipeline:

```bash
cd src
python train_and_evaluate.py
```

**What this does:**
1. ✅ Trains DQN agent on training split
2. ✅ Saves model to `models/saved_dqn.pt`
3. ✅ Evaluates on train split (no exploration)
4. ✅ Evaluates on dev split
5. ✅ Evaluates on test split
6. ✅ Shows comparison across all splits

**Expected Output:**
```
============================================================
🚀 DQN Training and Evaluation Pipeline
============================================================

📚 STEP 1: Training the Model
============================================================
[Training progress...]
✅ Training complete. Model saved to models/saved_dqn.pt

📊 STEP 1.5: Evaluating on TRAIN Split
============================================================
📊 TRAIN Split Evaluation Summary
  Average Reward: 4.85 ± 0.45
  ...

📊 STEP 2: Evaluating on DEV Split
============================================================
📊 DEV Split Evaluation Summary
  Average Reward: 4.75 ± 0.45
  ...

📊 STEP 3: Evaluating on TEST Split
============================================================
📊 TEST Split Evaluation Summary
  Average Reward: 4.68 ± 0.52
  ...

📊 FINAL RESULTS: Train vs Dev vs Test Comparison
============================================================
  Train - Average Reward: 4.85 ± 0.45
  Dev   - Average Reward: 4.75 ± 0.45
  Test  - Average Reward: 4.68 ± 0.52
  ✅ Good generalization
```

**Time:** ~30-60 minutes (depending on episodes and dataset size)

---

### Step 2: Compare with Baselines

After training, compare your DQN agent against baselines:

```bash
cd src
python baseline_comparison.py --episodes 100
```

**Options:**
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
  Avg Ending Reward: 3.2

📈 DQN Improvement over Random: +2.53 (+108.1%)
📉 Gap to Oracle: -0.36 (-6.9%)
```

**Time:** ~5-10 minutes (depending on episodes)

---

## 🔧 Configuration Before Running

### Option A: Default Settings (Recommended for First Run)

Edit `config.py`:
```python
USE_GENERATION = False  # Start without generation (faster)
USE_STOCHASTIC_EMOTIONS = True  # Enable stochastic emotions
INCLUDE_SCENE_INDEX = True  # Include scene index
```

### Option B: Full Features (Slower but Complete)

Edit `config.py`:
```python
USE_GENERATION = True  # Enable generation mode
USE_STOCHASTIC_EMOTIONS = True  # Enable stochastic emotions
INCLUDE_SCENE_INDEX = True  # Include scene index
```

---

## 📊 What You'll Get

### From train_and_evaluate.py:
- ✅ Trained model saved to `models/saved_dqn.pt`
- ✅ Performance metrics on train/dev/test splits
- ✅ Overfitting detection
- ✅ Generalization analysis

### From baseline_comparison.py:
- ✅ Random baseline performance
- ✅ Oracle baseline performance
- ✅ DQN agent performance
- ✅ Improvement metrics
- ✅ True continuation pick rate
- ✅ Ending quality metrics

---

## ⚡ Quick Test (Faster)

For a quick test with limited stories:

```python
# In train_and_evaluate.py, modify:
TRAIN_MAX_STORIES = 50  # Limit training stories
EVAL_MAX_STORIES = 20   # Limit evaluation stories
```

Then run:
```bash
cd src
python train_and_evaluate.py
python baseline_comparison.py --episodes 20
```

---

## 🎓 Understanding the Results

### Good Results:
- **DQN > Random**: Agent learned something
- **DQN close to Oracle**: Agent learned well
- **Small dev-test gap**: Good generalization
- **High true pick rate**: Agent prefers good continuations

### Areas for Improvement:
- **Large train-dev gap**: Possible overfitting
- **DQN ≈ Random**: Agent didn't learn (check hyperparameters)
- **Large dev-test gap**: Poor generalization

---

## ✅ Complete Workflow Summary

1. **Configure** `config.py` (optional - defaults work)
2. **Train & Evaluate**: `python train_and_evaluate.py`
3. **Compare Baselines**: `python baseline_comparison.py --episodes 100`

That's it! You'll have:
- ✅ Trained model
- ✅ Performance metrics
- ✅ Baseline comparisons
- ✅ All proposal requirements met

