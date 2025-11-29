"""
Story Generator using Language Models
Generates story continuations using GPT-2 or similar models
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Optional
import random
import re


class StoryGenerator:
    """
    Generates story continuations using a pre-trained language model.
    Includes content filtering and quality improvements.
    """
    
    # List of inappropriate words/phrases to filter (basic list - can be expanded)
    INAPPROPRIATE_WORDS = [
        'bitch', 'damn', 'hell', 'crap', 'shit', 'fuck', 'asshole', 
        'bastard', 'idiot', 'stupid', 'moron', 'retard'
    ]
    
    def __init__(self, model_name: str = "gpt2", device: Optional[str] = None,
                 filter_inappropriate: bool = True, use_half_precision: bool = True):
        """
        Initialize the story generator.
        OPTIMIZED: Uses half precision for faster inference.
        
        Args:
            model_name: HuggingFace model name (default: "gpt2")
            device: Device to run on ('cuda', 'cpu', or None for auto)
            filter_inappropriate: Whether to filter inappropriate content
            use_half_precision: Use FP16 for faster inference (GPU only)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Use half precision for faster inference (if CUDA available)
        if use_half_precision and self.device == "cuda":
            self.model = self.model.half()  # FP16 for 2x speedup on GPU
        
        self.model.to(self.device)
        self.model.eval()  # Inference mode
        self.filter_inappropriate = filter_inappropriate
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate_continuations(self, context: List[str], num_options: int = 2, 
                               max_length: int = 50, 
                               temperature_range: tuple = (0.6, 0.8),
                               max_attempts: int = 5,
                               use_batch_generation: bool = True,
                               num_return_sequences: int = 2) -> List[str]:
        """
        Generate multiple continuation options for the given story context.
        Includes quality filtering and content moderation.
        OPTIMIZED: Uses batch generation for faster processing.
        
        Args:
            context: List of sentences representing the story so far
            num_options: Number of different continuations to generate
            max_length: Maximum length of generated text
            temperature_range: (min, max) temperature values for diversity (lower = more focused)
            max_attempts: Maximum attempts to generate acceptable continuations
            use_batch_generation: Whether to generate multiple sequences in parallel (faster)
            num_return_sequences: Number of sequences to generate per batch (when use_batch_generation=True)
        
        Returns:
            List of generated sentence strings (filtered and cleaned)
        """
        # Format context with better prompt engineering
        context_text = self._format_context(context)
        
        # Tokenize context with attention mask (reduced context for speed)
        # Use shorter context window - keep last 300 tokens instead of 400
        inputs = self.tokenizer(context_text, return_tensors="pt", truncation=True, max_length=300)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get input length for generation
        input_length = inputs['input_ids'].shape[1]
        
        continuations = []
        seen_continuations = set()  # Avoid duplicates
        
        with torch.no_grad():
            attempts = 0
            max_total_attempts = max_attempts * num_options
            
            # Use batch generation for faster processing
            if use_batch_generation and num_return_sequences > 1:
                while len(continuations) < num_options and attempts < max_total_attempts:
                    attempts += 1
                    
                    # Vary temperature for diversity
                    temperature = random.uniform(*temperature_range)
                    
                    # Generate multiple sequences in parallel (much faster than sequential)
                    batch_size = min(num_return_sequences, num_options - len(continuations))
                    
                    # Generate continuations in batch (model handles batching internally)
                    # Use optimized generation parameters for speed
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask'),
                        max_length=input_length + max_length,
                        min_length=input_length + 5,  # Minimum length for faster generation
                        temperature=temperature,
                        top_p=0.9,  # Slightly higher for faster convergence
                        top_k=50,   # Slightly higher for faster convergence
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=batch_size,  # Model generates multiple sequences at once
                        repetition_penalty=1.15,  # Slightly lower for speed
                        no_repeat_ngram_size=2  # Reduced from 3 for speed
                        # Note: early_stopping only works with beam search (num_beams>1), so removed
                    )
                    
                    # Process each generated sequence
                    for output in outputs:
                        # Decode the generated text
                        generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                        
                        # Extract only the new part (remove context)
                        new_text = generated_text[len(context_text):].strip()
                        
                        # Extract and clean first sentence
                        sentence = self._extract_first_sentence(new_text)
                        sentence = self._clean_text(sentence)
                        
                        # Quick quality checks (less strict for speed)
                        if not sentence or len(sentence) < 8:  # Reduced from 10
                            continue
                        
                        # Check for inappropriate content
                        if self.filter_inappropriate and self._contains_inappropriate(sentence):
                            continue
                        
                        # Check for duplicates
                        sentence_lower = sentence.lower().strip()
                        if sentence_lower in seen_continuations:
                            continue
                        
                        # Basic quality check (simplified for speed)
                        words = sentence.split()
                        if len(words) < 3 or len(words) > 35:  # Slightly relaxed
                            continue
                        
                        continuations.append(sentence)
                        seen_continuations.add(sentence_lower)
                        
                        if len(continuations) >= num_options:
                            break
            else:
                # Fallback to sequential generation (original method)
                while len(continuations) < num_options and attempts < max_total_attempts:
                    attempts += 1
                    
                    temperature = random.uniform(*temperature_range)
                    
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask'),
                        max_length=input_length + max_length,
                        min_length=input_length + 5,
                        temperature=temperature,
                        top_p=0.9,
                        top_k=50,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        repetition_penalty=1.15,
                        no_repeat_ngram_size=2
                        # Note: early_stopping only works with beam search (num_beams>1), so removed
                    )
                    
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    new_text = generated_text[len(context_text):].strip()
                    sentence = self._extract_first_sentence(new_text)
                    sentence = self._clean_text(sentence)
                    
                    if not sentence or len(sentence) < 8:
                        continue
                    
                    if self.filter_inappropriate and self._contains_inappropriate(sentence):
                        continue
                    
                    sentence_lower = sentence.lower().strip()
                    if sentence_lower in seen_continuations:
                        continue
                    
                    words = sentence.split()
                    if len(words) < 3 or len(words) > 35:
                        continue
                    
                    continuations.append(sentence)
                    seen_continuations.add(sentence_lower)
        
        # If we don't have enough options, add fallbacks
        while len(continuations) < num_options:
            fallback = self._generate_fallback_continuation(context)
            if fallback not in continuations:
                continuations.append(fallback)
        
        return continuations[:num_options]
    
    def _format_context(self, context: List[str]) -> str:
        """
        Format context with better prompt engineering for story continuation.
        
        Args:
            context: List of sentences
        
        Returns:
            Formatted context string
        """
        # Join sentences with proper spacing
        story_text = " ".join(context)
        
        # Add a subtle prompt to guide generation (optional, can be removed if not helpful)
        # The model should naturally continue the story, but we can add context
        return story_text
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize generated text.
        
        Args:
            text: Raw generated text
        
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove quotes at start/end if they don't make sense
        text = text.strip()
        if text.startswith('"') and not text.endswith('"'):
            # Unmatched quote, remove it
            text = text[1:]
        
        # Remove common artifacts
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([,.!?])([^\s])', r'\1 \2', text)  # Add space after punctuation
        
        return text.strip()
    
    def _contains_inappropriate(self, text: str) -> bool:
        """
        Check if text contains inappropriate content.
        
        Args:
            text: Text to check
        
        Returns:
            True if inappropriate content found
        """
        if not self.filter_inappropriate:
            return False
        
        text_lower = text.lower()
        for word in self.INAPPROPRIATE_WORDS:
            # Check for whole word matches (not substrings)
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, text_lower):
                return True
        return False
    
    def _is_quality_continuation(self, sentence: str, context: List[str]) -> bool:
        """
        Check if continuation meets quality standards.
        
        Args:
            sentence: Generated sentence
            context: Story context
        
        Returns:
            True if sentence meets quality criteria
        """
        # Check length (reasonable sentence length)
        words = sentence.split()
        if len(words) < 3 or len(words) > 30:
            return False
        
        # Check for too many repeated words
        word_counts = {}
        for word in words:
            word_lower = word.lower()
            word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            if word_counts[word_lower] > 3:  # Same word repeated too many times
                return False
        
        # Check for proper sentence structure (has subject-verb pattern)
        # Simple heuristic: should have at least one verb-like word
        common_verbs = ['is', 'was', 'are', 'were', 'has', 'had', 'have', 'do', 'did', 
                        'went', 'came', 'said', 'took', 'got', 'made', 'saw', 'looked',
                        'ran', 'walked', 'jumped', 'grabbed', 'held', 'chased']
        words_lower = [w.lower().strip('.,!?;:') for w in words]
        has_verb = any(verb in words_lower for verb in common_verbs)
        
        # Not a hard requirement, but prefer sentences with verbs
        # Allow through if it's a reasonable length and clean
        
        return True
    
    def _generate_fallback_continuation(self, context: List[str]) -> str:
        """
        Generate a simple, safe fallback continuation.
        
        Args:
            context: Story context
        
        Returns:
            Safe fallback sentence
        """
        # Simple, context-appropriate fallbacks
        fallbacks = [
            "The story continued.",
            "Something interesting happened.",
            "The situation developed further.",
            "Events took an unexpected turn.",
            "The moment passed quietly."
        ]
        return random.choice(fallbacks)
    
    def _extract_first_sentence(self, text: str) -> str:
        """
        Extract the first complete sentence from generated text.
        Improved to handle quotes and dialogue better.
        
        Args:
            text: Generated text
        
        Returns:
            First sentence ending with punctuation
        """
        # Remove leading whitespace
        text = text.strip()
        if not text:
            return ""
        
        # Handle quoted dialogue
        if text.startswith('"'):
            # Find closing quote
            quote_end = text.find('"', 1)
            if quote_end != -1:
                # Check if there's punctuation before or after quote
                potential_sentence = text[:quote_end + 1]
                # Look for punctuation after quote
                if len(text) > quote_end + 1:
                    next_char = text[quote_end + 1]
                    if next_char in '.!?':
                        return text[:quote_end + 2].strip()
                return potential_sentence
        
        # Find first sentence ending
        for punct in ['.', '!', '?']:
            idx = text.find(punct)
            if idx != -1:
                # Make sure it's not part of an abbreviation or number
                if idx > 0:
                    prev_char = text[idx - 1]
                    # Skip if it's part of a number or abbreviation
                    if prev_char.isdigit() or (prev_char.isupper() and idx < len(text) - 1):
                        continue
                return text[:idx + 1].strip()
        
        # If no punctuation found, try to find a natural break
        # Look for common sentence-ending patterns
        for pattern in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            idx = text.find(pattern)
            if idx != -1:
                return text[:idx + 1].strip()
        
        # If still no sentence ending, return first 80 words or whole text
        words = text.split()
        if len(words) > 80:
            return ' '.join(words[:80]) + '.'
        return text.strip()
    
    def generate_single(self, context: List[str], temperature: float = 0.8, 
                       max_length: int = 50) -> str:
        """
        Generate a single continuation.
        
        Args:
            context: List of sentences representing the story so far
            temperature: Sampling temperature
            max_length: Maximum length of generated text
        
        Returns:
            Generated sentence string
        """
        continuations = self.generate_continuations(
            context, 
            num_options=1, 
            max_length=max_length,
            temperature_range=(temperature, temperature)
        )
        return continuations[0] if continuations else "The story continued."

