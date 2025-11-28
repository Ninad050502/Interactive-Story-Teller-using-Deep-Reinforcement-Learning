"""
Configuration file for StoryCommonsense DQN Training
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Dataset paths
CSV_STORIES_PATH = os.path.join(DATA_DIR, "storycommonsense_data", "rocstorysubset.csv")
JSON_ANNOTATIONS_PATH = os.path.join(DATA_DIR, "storycommonsense_data", "json_version", "annotations.json")
STORY_PARTITION_PATH = os.path.join(DATA_DIR, "storycommonsense_data", "storyid_partition.txt")

# Legacy story path (for backward compatibility)
LEGACY_STORY_PATH = os.path.join(DATA_DIR, "story_sample.json")

# Training settings
TRAIN_SPLIT = 'train'  # 'train', 'dev', or 'test'
MAX_STORIES = None  # None = use all stories, or specify number for testing
STORIES_PER_EPISODE = 1  # Number of stories to use per episode
USE_ANNOTATIONS = True  # Whether to use character annotations
USE_GENERATION = True  # Whether to use story generation (multi-choice mode: 1 true + 2 generated)
USE_STOCHASTIC_EMOTIONS = True  # Whether to use stochastic emotional transitions
INCLUDE_SCENE_INDEX = True  # Whether to include scene index in state encoding

# Reward weights
REWARD_WEIGHTS = {
    'sequence': 1.0,  # Weight for sequence correctness
    'character_consistency': 0.5,  # Weight for character consistency
    'narrative_coherence': 0.3  # Weight for narrative coherence
}

# State encoding settings
STATE_DIM = 768  # Base sentence embedding dimension
INCLUDE_CHARACTER_FEATURES = True  # Whether to include character features in state
CHARACTER_EMOTION_DIM = 8  # Plutchik emotions (8 categories)
CHARACTER_MOTIVATION_DIM = 24  # Maslow (5) + Reiss (19) = 24
SCENE_INDEX_DIM = 1  # Scene index (normalized position in story)
ENHANCED_STATE_DIM = STATE_DIM + CHARACTER_EMOTION_DIM + CHARACTER_MOTIVATION_DIM + SCENE_INDEX_DIM  # 801
BASE_STATE_DIM = STATE_DIM + SCENE_INDEX_DIM  # 769 (without character features)

# DQN Agent settings
AGENT_CONFIG = {
    'state_size': STATE_DIM,  # Will be updated to ENHANCED_STATE_DIM in Phase 2
    'action_size': 2,  # Will be updated to 3 if USE_GENERATION=True
    'gamma': 0.9,
    'lr': 1e-3,
    'batch_size': 32,
    'buffer_size': 10000,  # Increased for multi-story training
    'epsilon_decay': 0.995,
    'epsilon_min': 0.1,
    'target_update_frequency': 10  # Update target network every N episodes
}

# Story generation settings
GENERATION_CONFIG = {
    'model_name': 'gpt2',  # Language model for generation ('gpt2', 'distilgpt2', etc.)
    'num_generated_options': 2,  # Number of generated continuations to create
    'max_length': 50,  # Maximum length of generated text
    'temperature_range': (0.6, 0.8),  # Lower temperature for better quality (was 0.7-0.9)
    'filter_inappropriate': True,  # Filter inappropriate language
    'max_attempts': 5  # Maximum attempts to generate acceptable continuations
}

# Training settings
TRAINING_CONFIG = {
    'episodes': 1000,
    'eval_frequency': 50,  # Evaluate on dev set every N episodes
    'save_frequency': 100,  # Save checkpoint every N episodes
    'early_stopping_patience': 200,  # Stop if no improvement for N episodes
    'min_episodes_before_eval': 100  # Minimum episodes before first evaluation
}

# Model paths
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "saved_dqn.pt")
CHECKPOINT_DIR = os.path.join(MODELS_DIR, "checkpoints")

# Logging
LOG_DIR = os.path.join(BASE_DIR, "logs")
VERBOSE = True  # Print training progress

