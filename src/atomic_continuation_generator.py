"""
ATOMIC Continuation Generator
Generates alternative story continuations using ATOMIC commonsense relations.
Supports multiple access methods: COMET model, ATOMIC file, or template-based generation.
"""

import os
import random
from typing import List, Optional, Dict, Tuple
import torch


class AtomicContinuationGenerator:
    """
    Generates alternative story continuations using ATOMIC commonsense knowledge.
    
    Supports multiple modes:
    1. COMET-ATOMIC model (via transformers)
    2. ATOMIC TSV file lookup
    3. Template-based generation (fallback)
    """
    
    def __init__(self, 
                 mode: str = 'template',
                 atomic_file_path: Optional[str] = None,
                 comet_model_name: Optional[str] = None,
                 num_alternatives: int = 2,
                 relations: Optional[List[str]] = None):
        """
        Initialize ATOMIC continuation generator.
        
        Args:
            mode: 'comet', 'file', or 'template'
            atomic_file_path: Path to ATOMIC TSV file (if mode='file')
            comet_model_name: HuggingFace model name (if mode='comet')
            num_alternatives: Number of alternative continuations to generate
            relations: List of ATOMIC relations to use (e.g., ['xEffect', 'xWant', 'xReact'])
        """
        self.mode = mode
        self.num_alternatives = num_alternatives
        self.relations = relations or ['xEffect', 'xWant', 'xReact']
        self.comet_model = None
        self.comet_tokenizer = None
        self.atomic_data = None
        
        if mode == 'comet':
            self._load_comet_model(comet_model_name)
        elif mode == 'file':
            self._load_atomic_file(atomic_file_path)
        elif mode == 'template':
            print("Using template-based ATOMIC generation (fallback mode)")
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'comet', 'file', or 'template'")
    
    def _load_comet_model(self, model_name: Optional[str]):
        """Load COMET-ATOMIC model from HuggingFace."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            model_name = model_name or "microsoft/DialoGPT-medium"  # Fallback
            print(f"Loading COMET model: {model_name}")
            # Note: For actual ATOMIC, use models like "microsoft/DialoGPT-medium" 
            # or specialized COMET models if available
            self.comet_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.comet_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.comet_model.eval()
            print("✅ COMET model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load COMET model: {e}")
            print("Falling back to template mode")
            self.mode = 'template'
    
    def _load_atomic_file(self, file_path: Optional[str]):
        """Load ATOMIC data from TSV file."""
        if file_path is None or not os.path.exists(file_path):
            print(f"⚠️ ATOMIC file not found: {file_path}")
            print("Falling back to template mode")
            self.mode = 'template'
            return
        
        try:
            import pandas as pd
            print(f"Loading ATOMIC data from: {file_path}")
            # ATOMIC TSV format: head, relation, tail
            self.atomic_data = pd.read_csv(file_path, sep='\t', header=None, 
                                          names=['head', 'relation', 'tail'])
            print(f"✅ Loaded {len(self.atomic_data)} ATOMIC relations")
        except Exception as e:
            print(f"⚠️ Failed to load ATOMIC file: {e}")
            print("Falling back to template mode")
            self.mode = 'template'
    
    def _extract_event(self, sentence: str) -> str:
        """
        Extract the main event/action from a sentence.
        Simple heuristic: clean up the sentence to be used in a template.
        """
        # Clean up: lowercase, remove trailing punctuation
        clean = sentence.strip()
        if clean.endswith('.'):
            clean = clean[:-1]
        return clean.lower()
    
    def _query_atomic_file(self, event: str, relation: str) -> Optional[str]:
        """Query ATOMIC data from loaded file."""
        if self.atomic_data is None:
            return None
        
        # Find matching head and relation
        matches = self.atomic_data[
            (self.atomic_data['head'].str.contains(event, case=False, na=False)) &
            (self.atomic_data['relation'] == relation)
        ]
        
        if len(matches) > 0:
            # Return a random matching tail
            return matches.sample(1).iloc[0]['tail']
        return None
    
    def _query_comet(self, event: str, relation: str) -> Optional[str]:
        """Query COMET model for ATOMIC relation."""
        if self.comet_model is None:
            return None
        
        try:
            # Format: "head [SEP] relation"
            input_text = f"{event} [SEP] {relation}"
            inputs = self.comet_tokenizer.encode(input_text, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.comet_model.generate(
                    inputs, 
                    max_length=50,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
            
            generated = self.comet_tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated
        except Exception as e:
            print(f"Error querying COMET: {e}")
            return None
    
    def _generate_template_continuation(self, sentence: str, relation: str) -> str:
        """
        Generate a template-based continuation (fallback method).
        Creates plausible alternatives based on relation type.
        """
        event = self._extract_event(sentence)
        
        # Template-based generation based on relation type
        # Updated to work better with full sentences as input
        # Removed redundant verbs like "happened" or "occurred" since input is likely a full sentence
        templates = {
            'xEffect': [
                f"Because of that, {event}.",
                f"Consequently, {event}.",
                f"As a result, {event}."
            ],
            'xWant': [
                f"They wanted to make sure that {event}.",
                f"It was their desire that {event}.",
                f"They hoped that {event}."
            ],
            'xReact': [
                f"They felt emotional because {event}.",
                f"It made them react when {event}.",
                f"They responded to the fact that {event}."
            ],
            'xNeed': [
                f"But first, they needed to ensure {event}.",
                f"To do this, it was required that {event}.",
                f"Beforehand, {event} was necessary."
            ],
            'xIntent': [
                f"It was their intention that {event}.",
                f"The goal was that {event}.",
                f"They planned for {event}."
            ]
        }
        
        if relation in templates:
            return random.choice(templates[relation])
        else:
            # Generic fallback
            return f"Then {event}."
    
    def generate_alternatives(self, current_sentence: str, true_next: str) -> List[Tuple[str, str]]:
        """
        Generate alternative continuations for the current sentence.
        
        Args:
            current_sentence: Current story line
            true_next: The true next line from the dataset
        
        Returns:
            List of (continuation_text, relation_type) tuples
            Format: [(true_next, 'true'), (alt1, 'xEffect'), (alt2, 'xWant'), ...]
        """
        alternatives = [(true_next, 'true')]  # Always include true next as first option
        
        event = self._extract_event(current_sentence)
        
        # Generate alternatives for each relation
        for relation in self.relations[:self.num_alternatives]:
            continuation = None
            
            if self.mode == 'comet':
                continuation = self._query_comet(event, relation)
            elif self.mode == 'file':
                continuation = self._query_atomic_file(event, relation)
            
            # Fallback to template if no result
            if continuation is None or len(continuation.strip()) == 0:
                continuation = self._generate_template_continuation(current_sentence, relation)
            
            # Clean up continuation
            continuation = continuation.strip()
            if continuation and continuation != true_next:  # Avoid duplicates
                alternatives.append((continuation, relation))
        
        # Ensure we have at least num_alternatives + 1 (including true)
        while len(alternatives) < self.num_alternatives + 1:
            # Add generic template alternatives
            relation = random.choice(self.relations)
            continuation = self._generate_template_continuation(current_sentence, relation)
            if continuation not in [alt[0] for alt in alternatives]:
                alternatives.append((continuation, relation))
        
        return alternatives[:self.num_alternatives + 1]  # Limit total alternatives


def create_atomic_generator(config: Optional[Dict] = None) -> AtomicContinuationGenerator:
    """
    Factory function to create ATOMIC generator from config.
    
    Args:
        config: Dictionary with generator configuration
    
    Returns:
        AtomicContinuationGenerator instance
    """
    if config is None:
        config = {}
    
    return AtomicContinuationGenerator(
        mode=config.get('mode', 'template'),
        atomic_file_path=config.get('atomic_file_path'),
        comet_model_name=config.get('comet_model_name'),
        num_alternatives=config.get('num_alternatives', 2),
        relations=config.get('relations', ['xEffect', 'xWant', 'xReact'])
    )

