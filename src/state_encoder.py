import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Dict, List

class StateEncoder:
    """
    Encodes story sentences into dense vectors using DistilBERT.
    Can optionally include character emotion and motivation features.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", 
                 include_character_features: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # inference mode
        self.include_character_features = include_character_features
        
        # Plutchik emotions (8 categories)
        self.plutchik_emotions = ['joy', 'trust', 'fear', 'surprise', 
                                  'sadness', 'disgust', 'anger', 'anticipation']
        
        # Maslow categories (5)
        self.maslow_categories = ['spiritual growth', 'esteem', 'love', 
                                 'stability', 'physiological']
        
        # Reiss categories (19)
        self.reiss_categories = ['curiosity', 'serenity', 'idealism', 'independence',
                                'competition', 'honor', 'approval', 'power', 'status',
                                'romance', 'belonging', 'family', 'social contact',
                                'health', 'savings', 'order', 'safety', 'food', 'rest']

    def encode(self, text: str, character_info: Optional[Dict] = None) -> torch.Tensor:
        """
        Returns embedding for the input sentence.
        If include_character_features is True and character_info is provided,
        concatenates character features to the sentence embedding.
        
        Args:
            text: Sentence text
            character_info: Optional dictionary with character annotations
        
        Returns:
            Tensor of shape (768,) or (800,) if character features included
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=64
            )
            outputs = self.model(**inputs)
            # mean-pool over tokens
            embedding = outputs.last_hidden_state.mean(dim=1)
            sentence_embedding = embedding.squeeze(0)
        
        if self.include_character_features and character_info:
            char_features = self._encode_character_features(character_info)
            return torch.cat([sentence_embedding, char_features])
        
        return sentence_embedding
    
    def _encode_character_features(self, character_info: Dict) -> torch.Tensor:
        """
        Encode character emotions and motivations into feature vectors.
        
        Args:
            character_info: Dictionary with character annotations for a line
        
        Returns:
            Tensor of shape (32,) - 8 emotion + 24 motivation features
        """
        emotion_vector = self._encode_emotions(character_info)
        motivation_vector = self._encode_motivations(character_info)
        return torch.cat([emotion_vector, motivation_vector])
    
    def _encode_emotions(self, character_info: Dict) -> torch.Tensor:
        """
        Encode Plutchik emotions into 8-dim vector.
        Each dimension represents intensity of that emotion (0-1).
        """
        emotion_vector = torch.zeros(8)
        
        if not character_info or 'characters' not in character_info:
            return emotion_vector
        
        # Aggregate emotions across all characters in the line
        for char_name, char_data in character_info['characters'].items():
            if not char_data.get('app', False):
                continue
            
            emotion_data = char_data.get('emotion', {})
            if not emotion_data:
                continue
            
            # Aggregate across annotators
            emotion_counts = {emotion: 0 for emotion in self.plutchik_emotions}
            total_annotations = 0
            
            for ann_id, ann_data in emotion_data.items():
                plutchik_list = ann_data.get('plutchik', [])
                for emotion_str in plutchik_list:
                    # Format: "joy:2" or "joy:3"
                    if ':' in emotion_str:
                        emotion_name = emotion_str.split(':')[0].lower()
                        intensity = int(emotion_str.split(':')[1])
                        if emotion_name in emotion_counts:
                            emotion_counts[emotion_name] += intensity
                            total_annotations += 1
            
            # Normalize by number of annotations
            if total_annotations > 0:
                for i, emotion in enumerate(self.plutchik_emotions):
                    emotion_vector[i] += emotion_counts[emotion] / (total_annotations * 3.0)
        
        # Normalize to [0, 1] range
        emotion_vector = torch.clamp(emotion_vector, 0.0, 1.0)
        return emotion_vector
    
    def _encode_motivations(self, character_info: Dict) -> torch.Tensor:
        """
        Encode Maslow and Reiss motivations into 24-dim vector.
        First 5 dims: Maslow, next 19 dims: Reiss
        """
        motivation_vector = torch.zeros(24)  # 5 Maslow + 19 Reiss
        
        if not character_info or 'characters' not in character_info:
            return motivation_vector
        
        # Aggregate motivations across all characters
        for char_name, char_data in character_info['characters'].items():
            if not char_data.get('app', False):
                continue
            
            motiv_data = char_data.get('motiv', {})
            if not motiv_data:
                continue
            
            # Aggregate across annotators
            maslow_counts = {cat: 0 for cat in self.maslow_categories}
            reiss_counts = {cat: 0 for cat in self.reiss_categories}
            total_annotations = 0
            
            for ann_id, ann_data in motiv_data.items():
                maslow_list = ann_data.get('maslow', [])
                reiss_list = ann_data.get('reiss', [])
                
                for maslow_cat in maslow_list:
                    if maslow_cat.lower() in maslow_counts:
                        maslow_counts[maslow_cat.lower()] += 1
                        total_annotations += 1
                
                for reiss_cat in reiss_list:
                    if reiss_cat.lower() in reiss_counts:
                        reiss_counts[reiss_cat.lower()] += 1
                        total_annotations += 1
            
            # Normalize by number of annotations
            if total_annotations > 0:
                # Maslow (first 5 dims)
                for i, cat in enumerate(self.maslow_categories):
                    motivation_vector[i] += maslow_counts[cat] / total_annotations
                
                # Reiss (next 19 dims)
                for i, cat in enumerate(self.reiss_categories):
                    motivation_vector[5 + i] += reiss_counts[cat] / total_annotations
        
        # Normalize to [0, 1] range
        motivation_vector = torch.clamp(motivation_vector, 0.0, 1.0)
        return motivation_vector
