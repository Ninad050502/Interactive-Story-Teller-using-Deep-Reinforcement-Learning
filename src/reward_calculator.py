"""
Reward Calculator for Enhanced Story Navigation
Calculates rewards based on sequence correctness, character consistency, and narrative coherence
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple


class RewardCalculator:
    """
    Calculates enhanced rewards for story navigation.
    """
    
    def __init__(self, reward_weights: Optional[Dict[str, float]] = None):
        """
        Initialize reward calculator.
        
        Args:
            reward_weights: Dictionary with weights for different reward components
                Default: {'sequence': 1.0, 'character_consistency': 0.5, 'narrative_coherence': 0.3}
        """
        self.reward_weights = reward_weights or {
            'sequence': 1.0,
            'character_consistency': 0.5,
            'narrative_coherence': 0.3
        }
    
    def calculate_reward(self, prev_idx: int, current_idx: int,
                       prev_state: Optional[torch.Tensor] = None,
                       current_state: Optional[torch.Tensor] = None,
                       prev_char_info: Optional[Dict] = None,
                       current_char_info: Optional[Dict] = None) -> float:
        """
        Calculate total reward for a transition.
        
        Args:
            prev_idx: Previous line index
            current_idx: Current line index
            prev_state: Previous state embedding (optional)
            current_state: Current state embedding (optional)
            prev_char_info: Previous line character annotations (optional)
            current_char_info: Current line character annotations (optional)
        
        Returns:
            Total reward value
        """
        # 1. Sequence reward (always calculated)
        sequence_reward = self._calculate_sequence_reward(prev_idx, current_idx)
        
        # 2. Character consistency reward (if annotations available)
        char_reward = 0.0
        if prev_char_info and current_char_info:
            char_reward = self._calculate_character_consistency(
                prev_char_info, current_char_info
            )
        
        # 3. Narrative coherence reward (if states available)
        coherence_reward = 0.0
        if prev_state is not None and current_state is not None:
            coherence_reward = self._calculate_narrative_coherence(
                prev_state, current_state
            )
        
        # Weighted sum
        total_reward = (
            self.reward_weights['sequence'] * sequence_reward +
            self.reward_weights['character_consistency'] * char_reward +
            self.reward_weights['narrative_coherence'] * coherence_reward
        )
        
        return total_reward
    
    def _calculate_sequence_reward(self, prev_idx: int, current_idx: int) -> float:
        """
        Calculate reward for following correct story sequence.
        
        Args:
            prev_idx: Previous line index
            current_idx: Current line index
        
        Returns:
            +1.0 for correct sequence, -1.0 for skipping
        """
        if current_idx == prev_idx + 1:
            return 1.0
        else:
            return -1.0
    
    def _calculate_character_consistency(self, prev_char_info: Dict, 
                                        current_char_info: Dict) -> float:
        """
        Calculate reward for character consistency between consecutive lines.
        
        Args:
            prev_char_info: Character annotations for previous line
            current_char_info: Character annotations for current line
        
        Returns:
            Reward value between -1.0 and 1.0
        """
        if not prev_char_info or not current_char_info:
            return 0.0
        
        prev_chars = prev_char_info.get('characters', {})
        current_chars = current_char_info.get('characters', {})
        
        if not prev_chars or not current_chars:
            return 0.0
        
        consistency_score = 0.0
        total_comparisons = 0
        
        # Compare emotions and motivations for characters that appear in both lines
        for char_name in prev_chars:
            if char_name not in current_chars:
                continue
            
            prev_char = prev_chars[char_name]
            current_char = current_chars[char_name]
            
            # Check if character appears in both lines
            if not prev_char.get('app', False) or not current_char.get('app', False):
                continue
            
            # Compare emotions
            emotion_consistency = self._compare_emotions(
                prev_char.get('emotion', {}),
                current_char.get('emotion', {})
            )
            
            # Compare motivations
            motivation_consistency = self._compare_motivations(
                prev_char.get('motiv', {}),
                current_char.get('motiv', {})
            )
            
            consistency_score += (emotion_consistency + motivation_consistency) / 2.0
            total_comparisons += 1
        
        if total_comparisons == 0:
            return 0.0
        
        # Normalize to [-1, 1] range
        avg_consistency = consistency_score / total_comparisons
        return avg_consistency * 2.0 - 1.0  # Scale from [0, 1] to [-1, 1]
    
    def _compare_emotions(self, prev_emotions: Dict, current_emotions: Dict) -> float:
        """
        Compare emotion annotations between two lines.
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not prev_emotions or not current_emotions:
            return 0.5  # Neutral if no annotations
        
        # Extract Plutchik emotions
        prev_plutchik = set()
        current_plutchik = set()
        
        for ann_id, ann_data in prev_emotions.items():
            prev_plutchik.update([e.split(':')[0].lower() for e in ann_data.get('plutchik', [])])
        
        for ann_id, ann_data in current_emotions.items():
            current_plutchik.update([e.split(':')[0].lower() for e in ann_data.get('plutchik', [])])
        
        if not prev_plutchik and not current_plutchik:
            return 0.5
        
        # Calculate Jaccard similarity
        intersection = len(prev_plutchik & current_plutchik)
        union = len(prev_plutchik | current_plutchik)
        
        if union == 0:
            return 0.5
        
        return intersection / union
    
    def _compare_motivations(self, prev_motivations: Dict, current_motivations: Dict) -> float:
        """
        Compare motivation annotations between two lines.
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not prev_motivations or not current_motivations:
            return 0.5
        
        # Extract Maslow and Reiss categories
        prev_maslow = set()
        prev_reiss = set()
        current_maslow = set()
        current_reiss = set()
        
        for ann_id, ann_data in prev_motivations.items():
            prev_maslow.update([m.lower() for m in ann_data.get('maslow', [])])
            prev_reiss.update([r.lower() for r in ann_data.get('reiss', [])])
        
        for ann_id, ann_data in current_motivations.items():
            current_maslow.update([m.lower() for m in ann_data.get('maslow', [])])
            current_reiss.update([r.lower() for r in ann_data.get('reiss', [])])
        
        # Calculate similarity for both Maslow and Reiss
        maslow_sim = 0.5
        reiss_sim = 0.5
        
        if prev_maslow or current_maslow:
            intersection = len(prev_maslow & current_maslow)
            union = len(prev_maslow | current_maslow)
            maslow_sim = intersection / union if union > 0 else 0.5
        
        if prev_reiss or current_reiss:
            intersection = len(prev_reiss & current_reiss)
            union = len(prev_reiss | current_reiss)
            reiss_sim = intersection / union if union > 0 else 0.5
        
        return (maslow_sim + reiss_sim) / 2.0
    
    def _calculate_narrative_coherence(self, prev_state: torch.Tensor,
                                      current_state: torch.Tensor) -> float:
        """
        Calculate reward for narrative coherence based on state embeddings.
        
        Args:
            prev_state: Previous state embedding
            current_state: Current state embedding
        
        Returns:
            Coherence score between -1.0 and 1.0
        """
        # Use cosine similarity to measure coherence
        prev_state_norm = prev_state / (torch.norm(prev_state) + 1e-8)
        current_state_norm = current_state / (torch.norm(current_state) + 1e-8)
        
        # Cosine similarity
        cosine_sim = torch.dot(prev_state_norm, current_state_norm).item()
        
        # Scale from [-1, 1] to [0, 1] then to [-1, 1] for reward
        # Higher similarity = better coherence
        coherence_score = cosine_sim
        
        return coherence_score

