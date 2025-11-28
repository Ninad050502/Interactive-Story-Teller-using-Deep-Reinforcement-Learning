"""
Emotional Transition Model for Stochastic Emotional Outcomes
Models probabilistic emotional transitions based on story patterns
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import random
from collections import defaultdict


class EmotionalTransitionModel:
    """
    Models probabilistic emotional transitions for stochastic story outcomes.
    Learns from story patterns or uses fixed probabilities.
    """
    
    def __init__(self, use_learned_probs: bool = False):
        """
        Initialize emotional transition model.
        
        Args:
            use_learned_probs: Whether to learn probabilities from dataset (future)
        """
        self.use_learned_probs = use_learned_probs
        self.transition_probs = {}
        self.emotion_counts = defaultdict(lambda: defaultdict(int))
        
        # Plutchik emotions (8 categories)
        self.plutchik_emotions = ['joy', 'trust', 'fear', 'surprise', 
                                  'sadness', 'disgust', 'anger', 'anticipation']
        
        # Default transition probabilities based on action types
        # These can be learned from data or use ATOMIC relations
        self._initialize_default_transitions()
    
    def _initialize_default_transitions(self):
        """
        Initialize default transition probabilities.
        In a full implementation, these would be learned from story patterns or ATOMIC.
        """
        # Default: transitions tend to maintain or slightly shift emotions
        # High probability of maintaining similar emotion, lower probability of shifts
        
        for emotion in self.plutchik_emotions:
            # Default transition: 70% maintain, 30% shift to related emotions
            self.transition_probs[emotion] = {
                emotion: 0.7,  # Maintain same emotion
            }
            
            # Add probabilities for related emotions
            related_emotions = self._get_related_emotions(emotion)
            prob_per_related = 0.3 / len(related_emotions) if related_emotions else 0.0
            
            for related in related_emotions:
                self.transition_probs[emotion][related] = prob_per_related
    
    def _get_related_emotions(self, emotion: str) -> List[str]:
        """
        Get emotionally related emotions (for transition probabilities).
        """
        # Emotion clusters (can be refined based on Plutchik's wheel)
        emotion_clusters = {
            'joy': ['trust', 'anticipation'],
            'trust': ['joy', 'fear'],
            'fear': ['surprise', 'sadness'],
            'surprise': ['fear', 'disgust'],
            'sadness': ['disgust', 'anger'],
            'disgust': ['anger', 'sadness'],
            'anger': ['disgust', 'fear'],
            'anticipation': ['joy', 'surprise']
        }
        return emotion_clusters.get(emotion.lower(), [])
    
    def sample_emotion_outcome(self, 
                               current_emotion_vector: torch.Tensor,
                               action_type: Optional[str] = None,
                               context: Optional[List[str]] = None) -> torch.Tensor:
        """
        Sample probabilistic emotional outcome based on current emotion and action.
        
        Args:
            current_emotion_vector: Current emotion vector (8-dim Plutchik)
            action_type: Type of action (optional, for action-specific transitions)
            context: Story context (optional, for context-aware transitions)
        
        Returns:
            Sampled emotion vector (8-dim) with probabilistic outcomes
        """
        # Get dominant current emotion
        dominant_emotion_idx = torch.argmax(current_emotion_vector).item()
        dominant_emotion = self.plutchik_emotions[dominant_emotion_idx]
        
        # Get transition probabilities for this emotion
        if dominant_emotion in self.transition_probs:
            probs = self.transition_probs[dominant_emotion]
        else:
            # Fallback: maintain current emotion
            probs = {dominant_emotion: 1.0}
        
        # Sample next emotion probabilistically
        emotions = list(probs.keys())
        probabilities = list(probs.values())
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            # Fallback: maintain current
            emotions = [dominant_emotion]
            probabilities = [1.0]
        
        # Sample emotion
        sampled_emotion = np.random.choice(emotions, p=probabilities)
        sampled_idx = self.plutchik_emotions.index(sampled_emotion)
        
        # Create new emotion vector
        new_emotion_vector = torch.zeros(8)
        new_emotion_vector[sampled_idx] = 1.0  # Set dominant emotion
        
        # Blend with current emotion (smooth transition)
        blend_factor = 0.7  # 70% new, 30% old
        blended = blend_factor * new_emotion_vector + (1 - blend_factor) * current_emotion_vector
        
        # Normalize
        blended = blended / (torch.sum(blended) + 1e-8)
        
        return blended
    
    def sample_emotion_from_annotation(self, 
                                      annotation_emotion: torch.Tensor,
                                      action_type: Optional[str] = None) -> torch.Tensor:
        """
        Sample stochastic emotion outcome from deterministic annotation.
        Adds probabilistic variation to deterministic annotations.
        
        Args:
            annotation_emotion: Emotion vector from annotation (deterministic)
            action_type: Type of action taken
        
        Returns:
            Probabilistically sampled emotion vector
        """
        # Add stochastic variation to annotation
        # 80% follow annotation, 20% probabilistic transition
        if random.random() < 0.8:
            # Mostly follow annotation (with small noise)
            noise = torch.randn(8) * 0.1
            sampled = annotation_emotion + noise
            sampled = torch.clamp(sampled, 0.0, 1.0)
            sampled = sampled / (torch.sum(sampled) + 1e-8)
            return sampled
        else:
            # 20% chance: probabilistic transition
            return self.sample_emotion_outcome(annotation_emotion, action_type)
    
    def update_transition_probs(self, 
                                prev_emotion: str,
                                next_emotion: str,
                                action_type: Optional[str] = None):
        """
        Update transition probabilities based on observed patterns.
        For learning from data.
        
        Args:
            prev_emotion: Previous emotion
            next_emotion: Next emotion
            action_type: Action type (if available)
        """
        if self.use_learned_probs:
            key = (prev_emotion, action_type) if action_type else prev_emotion
            self.emotion_counts[key][next_emotion] += 1
            
            # Update probabilities based on counts
            total = sum(self.emotion_counts[key].values())
            if total > 0:
                self.transition_probs[prev_emotion] = {
                    emo: count / total 
                    for emo, count in self.emotion_counts[key].items()
                }
    
    def get_transition_probability(self, 
                                   from_emotion: str,
                                   to_emotion: str) -> float:
        """
        Get transition probability between two emotions.
        
        Args:
            from_emotion: Source emotion
            to_emotion: Target emotion
        
        Returns:
            Transition probability (0.0 to 1.0)
        """
        if from_emotion in self.transition_probs:
            return self.transition_probs[from_emotion].get(to_emotion, 0.0)
        return 0.0

