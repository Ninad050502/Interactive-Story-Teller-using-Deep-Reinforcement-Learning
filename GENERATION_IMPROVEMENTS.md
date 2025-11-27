# Story Generation Quality Improvements

## Overview

The story generator has been significantly improved to address quality and inappropriate content issues.

## Key Improvements

### 1. **Content Filtering** ✅
- Added automatic filtering of inappropriate language
- Filters common profanity and offensive words
- Configurable via `filter_inappropriate` flag in config

### 2. **Better Generation Parameters** ✅
- **Lower temperature range**: Changed from (0.7, 0.9) to (0.6, 0.8) for more focused, coherent generation
- **Improved sampling**: 
  - `top_p`: 0.85 (was 0.9) - more focused
  - `top_k`: 40 (was 50) - better quality
  - `repetition_penalty`: 1.2 - discourages repetition
  - `no_repeat_ngram_size`: 3 - avoids 3-gram repetition

### 3. **Quality Checks** ✅
- **Length validation**: Ensures sentences are 3-30 words
- **Duplicate detection**: Prevents showing identical options
- **Repetition check**: Flags sentences with too many repeated words
- **Structure validation**: Prefers sentences with proper structure

### 4. **Text Cleaning** ✅
- Removes extra whitespace and formatting artifacts
- Fixes punctuation spacing
- Handles quotes and dialogue better
- Cleans up common generation artifacts

### 5. **Better Sentence Extraction** ✅
- Improved handling of quoted dialogue
- Better detection of sentence boundaries
- Handles abbreviations and numbers correctly
- More robust fallback for edge cases

### 6. **Retry Mechanism** ✅
- Generates multiple attempts (up to 5 per option)
- Filters out low-quality or inappropriate content
- Falls back to safe alternatives if generation fails

## Configuration

All settings are configurable in `config.py`:

```python
GENERATION_CONFIG = {
    'model_name': 'gpt2',  # or 'distilgpt2' for faster generation
    'num_generated_options': 2,
    'max_length': 50,
    'temperature_range': (0.6, 0.8),  # Lower = more focused
    'filter_inappropriate': True,  # Enable content filtering
    'max_attempts': 5  # Retry attempts per option
}
```

## How It Works

1. **Generation**: Creates continuations with improved parameters
2. **Filtering**: Removes inappropriate content
3. **Quality Check**: Validates length, structure, and coherence
4. **Deduplication**: Ensures unique options
5. **Cleaning**: Normalizes text formatting
6. **Fallback**: Uses safe alternatives if needed

## Results

### Before:
- ❌ Inappropriate language: "Why did you think you were supposed to be with me, you pathetic bitch?"
- ❌ Off-topic generations
- ❌ Poor coherence

### After:
- ✅ Filtered inappropriate content
- ✅ More contextually appropriate continuations
- ✅ Better coherence and structure
- ✅ Cleaner, more readable text

## Testing

Run the test script to see improvements:

```bash
python test/test_generation.py
```

You should see:
- No inappropriate language in generated options
- More coherent and contextually appropriate continuations
- Better sentence structure and formatting

## Customization

### Add More Inappropriate Words

Edit `src/story_generator.py`:

```python
INAPPROPRIATE_WORDS = [
    'bitch', 'damn', 'hell', 'crap', 'shit', 'fuck', 
    # Add your custom words here
]
```

### Adjust Quality Thresholds

Modify the `_is_quality_continuation()` method in `StoryGenerator` to change:
- Minimum/maximum word count
- Repetition tolerance
- Structure requirements

### Use Different Model

Change in `config.py`:
```python
'model_name': 'distilgpt2'  # Faster, smaller model
# or
'model_name': 'gpt2-medium'  # Larger, potentially better quality
```

## Notes

- First generation may be slower due to quality checks
- Some valid continuations might be filtered if they're too similar to others
- The system prioritizes safety and quality over diversity
- You can disable filtering by setting `filter_inappropriate: False` in config

