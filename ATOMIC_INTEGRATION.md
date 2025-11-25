# ATOMIC Integration Guide

This document explains how to use the ATOMIC integration for generating alternative story continuations.

## Overview

The ATOMIC integration allows the RL agent to select from multiple story continuations:
- **Action 0**: True next line from the dataset
- **Action 1+**: Alternative continuations generated using ATOMIC commonsense relations

## Configuration

Edit `config.py` to enable and configure ATOMIC:

```python
# Enable ATOMIC integration
USE_ATOMIC = True

# ATOMIC configuration
ATOMIC_CONFIG = {
    'mode': 'template',  # Options: 'comet', 'file', or 'template'
    'atomic_file_path': None,  # Path to ATOMIC TSV file (if mode='file')
    'comet_model_name': None,  # HuggingFace model name (if mode='comet')
    'num_alternatives': 2,  # Number of alternative continuations per step
    'relations': ['xEffect', 'xWant', 'xReact']  # ATOMIC relations to use
}
```

## Modes

### 1. Template Mode (Default)
- **Mode**: `'template'`
- **Description**: Uses template-based generation as fallback
- **Pros**: No external dependencies, works immediately
- **Cons**: Less sophisticated than actual ATOMIC data
- **Usage**: Set `mode='template'` (default)

### 2. File Mode
- **Mode**: `'file'`
- **Description**: Loads ATOMIC data from TSV file
- **Requirements**: ATOMIC TSV file with format: `head \t relation \t tail`
- **Usage**:
  ```python
  ATOMIC_CONFIG = {
      'mode': 'file',
      'atomic_file_path': 'path/to/atomic.tsv',
      'num_alternatives': 2,
      'relations': ['xEffect', 'xWant', 'xReact']
  }
  ```

### 3. COMET Mode
- **Mode**: `'comet'`
- **Description**: Uses COMET-ATOMIC model from HuggingFace
- **Requirements**: `transformers` library, model download
- **Usage**:
  ```python
  ATOMIC_CONFIG = {
      'mode': 'comet',
      'comet_model_name': 'microsoft/DialoGPT-medium',  # Or specialized COMET model
      'num_alternatives': 2,
      'relations': ['xEffect', 'xWant', 'xReact']
  }
  ```

## ATOMIC Relations

Available relations (from ATOMIC dataset):
- **xEffect**: What happens as a result of an event
- **xWant**: What someone wants after an event
- **xNeed**: What is needed before an event
- **xIntent**: Why someone does something
- **xReact**: How someone feels/reacts
- **xAttr**: Attributes of a person
- **xWant**: What someone wants
- **oEffect**: Effect on others
- **oReact**: How others react
- **oWant**: What others want

## Action Space

With ATOMIC enabled:
- **Action 0**: Select true next line (from dataset)
- **Action 1**: Select first ATOMIC alternative (e.g., xEffect)
- **Action 2**: Select second ATOMIC alternative (e.g., xWant)
- **Action 3**: Select third ATOMIC alternative (e.g., xReact)
- etc.

Total action space = `1 + num_alternatives`

## Example Usage

### Basic Training with ATOMIC

```python
# In config.py
USE_ATOMIC = True
ATOMIC_CONFIG = {
    'mode': 'template',
    'num_alternatives': 2,
    'relations': ['xEffect', 'xWant']
}

# Training will automatically use ATOMIC
python src/dqn_train.py
```

### With ATOMIC File

1. Download ATOMIC dataset (TSV format)
2. Update config:
   ```python
   ATOMIC_CONFIG = {
       'mode': 'file',
       'atomic_file_path': 'data/atomic.tsv',
       'num_alternatives': 2,
       'relations': ['xEffect', 'xWant', 'xReact']
   }
   ```

### With COMET Model

1. Install transformers (if not already):
   ```bash
   pip install transformers
   ```
2. Update config:
   ```python
   ATOMIC_CONFIG = {
       'mode': 'comet',
       'comet_model_name': 'microsoft/DialoGPT-medium',
       'num_alternatives': 2,
       'relations': ['xEffect', 'xWant']
   }
   ```

## How It Works

1. **At each step**: Environment generates continuations using ATOMIC
2. **Agent selects**: Action chooses which continuation to follow
3. **Story progresses**: Selected continuation becomes the next state
4. **Reward calculated**: 
   - +1 for selecting true next (coherent)
   - +0.5 for ATOMIC alternatives (exploration)
   - +5 bonus for reaching good endings (positive emotions)

## Rewards

With ATOMIC integration:
- **Sequence reward**: +1.0 for true next, 0.0 for alternatives, -1.0 for skipping
- **Character consistency**: Based on emotion/motivation continuity
- **Narrative coherence**: Based on embedding similarity
- **Ending bonus**: +5.0 for reaching satisfying/joyful endings (from proposal)

## Troubleshooting

### Template Mode Not Working
- Check that `USE_ATOMIC = True` in config.py
- Verify `ATOMIC_CONFIG` is properly set

### File Mode Issues
- Ensure ATOMIC file exists and is readable
- Check file format (TSV with columns: head, relation, tail)
- Verify pandas is installed: `pip install pandas`

### COMET Mode Issues
- Install transformers: `pip install transformers`
- Check model name is valid
- Ensure sufficient memory for model loading
- If model fails to load, falls back to template mode automatically

## Next Steps

1. **Get ATOMIC Data**: Download from [ATOMIC dataset](https://allenai.org/data/atomic)
2. **Try Different Relations**: Experiment with different relation combinations
3. **Adjust Alternatives**: Change `num_alternatives` to balance exploration vs. exploitation
4. **Fine-tune Rewards**: Modify reward weights in `config.py`

## Alignment with Proposal

✅ **Action Space**: Multiple continuations (true + alternatives)  
✅ **ATOMIC Integration**: Uses xEffect, xWant, xReact relations  
✅ **Stochastic Transitions**: Same action can lead to different outcomes  
✅ **Ending Bonus**: +5 for good endings  
✅ **Commonsense-Guided**: Alternatives based on commonsense knowledge  

