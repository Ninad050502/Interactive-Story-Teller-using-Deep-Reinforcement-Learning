"""
ATOMIC Continuation Generator
Generates alternative story continuations using ATOMIC commonsense relations.
Supports multiple access methods: COMET model, ATOMIC file, or template-based generation.
"""

import os
import random
from typing import List, Optional, Dict, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


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
                 relations: Optional[List[str]] = None,
                 use_similarity: bool = True):
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
        self.word_index = {}  # For fast fuzzy matching
        self.query_cache = {}  # Cache for (event, relation) -> continuation
        self.relation_pools = {}  # Pool of all continuations per relation (for fallback)
        
        # For semantic similarity matching
        self.similarity_model = None
        self.similarity_tokenizer = None
        self.use_similarity = use_similarity  # Enable similarity-based selection (can be disabled for faster training)
        
        if mode == 'comet':
            self._load_comet_model(comet_model_name)
        elif mode == 'file':
            self._load_atomic_file(atomic_file_path)
            # Only load similarity model if enabled (saves time and memory during training)
            if self.use_similarity:
                self._load_similarity_model()  # Load model for semantic similarity
            else:
                print("⚠️ Similarity matching disabled - using random selection for faster training")
        elif mode == 'template':
            print("Using template-based ATOMIC generation (fallback mode)")
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'comet', 'file', or 'template'")
    
    def _load_similarity_model(self):
        """Load DistilBERT model for semantic similarity matching."""
        try:
            print("Loading similarity model (DistilBERT) for semantic matching...")
            model_name = "distilbert-base-uncased"
            self.similarity_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.similarity_model = AutoModel.from_pretrained(model_name)
            self.similarity_model.eval()  # Set to evaluation mode
            print("✅ Similarity model loaded")
        except Exception as e:
            print(f"⚠️ Failed to load similarity model: {e}")
            print("Falling back to random selection")
            self.use_similarity = False
            self.similarity_model = None
            self.similarity_tokenizer = None
    
    def _encode_sentence(self, text: str) -> np.ndarray:
        """Encode a sentence into a vector using DistilBERT."""
        if self.similarity_model is None or self.similarity_tokenizer is None:
            return None
        
        try:
            with torch.no_grad():
                inputs = self.similarity_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=64, padding=True
                )
                outputs = self.similarity_model(**inputs)
                # Use mean pooling of last hidden state
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()
                return embedding
        except Exception as e:
            print(f"Error encoding sentence: {e}")
            return None
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two sentences."""
        emb1 = self._encode_sentence(text1)
        emb2 = self._encode_sentence(text2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Compute cosine similarity
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return float(similarity)
    
    def _select_by_similarity(self, candidates: List[str], target: str, top_k: int = 5) -> str:
        """
        Select the most similar candidate to the target sentence.
        Returns a random choice from top_k most similar candidates.
        """
        if not self.use_similarity or self.similarity_model is None or len(candidates) == 0:
            return random.choice(candidates)
        
        if len(candidates) == 1:
            return candidates[0]
        
        # Compute similarities
        similarities = []
        for candidate in candidates:
            # Format candidate for comparison (remove PersonX, etc.)
            clean_candidate = candidate.replace('PersonX', 'someone').replace('PersonY', 'someone else')
            similarity = self._compute_similarity(clean_candidate, target)
            similarities.append((similarity, candidate))
        
        # Sort by similarity (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Select from top_k most similar
        top_candidates = [cand for _, cand in similarities[:top_k]]
        return random.choice(top_candidates)
    
    def _load_comet_model(self, model_name: Optional[str]):
        """Load COMET-ATOMIC model from HuggingFace."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Try proper COMET models first, then fallback to GPT-2 for story generation
            if model_name is None:
                # Try COMET-ATOMIC models (if available) or use GPT-2 for story continuation
                model_name = "gpt2"  # GPT-2 is good for story continuation
            
            print(f"Loading language model for ATOMIC generation: {model_name}")
            self.comet_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.comet_model = AutoModelForCausalLM.from_pretrained(model_name)
            self.comet_tokenizer.pad_token = self.comet_tokenizer.eos_token
            self.comet_model.eval()
            print("✅ Language model loaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to load language model: {e}")
            print("Falling back to template mode")
            self.mode = 'template'
    
    def _load_atomic_file(self, file_path: Optional[str]):
        """Load ATOMIC data from CSV file (v4 format) with optimizations."""
        if file_path is None or not os.path.exists(file_path):
            print(f"⚠️ ATOMIC file not found: {file_path}")
            print("Falling back to template mode")
            self.mode = 'template'
            return
        
        try:
            import pandas as pd
            import json
            
            print(f"Loading ATOMIC data from: {file_path}")
            
            # OPTIMIZATION: Use train split only for faster loading and queries
            # Check if train split file exists, otherwise use full file
            base_path = os.path.dirname(file_path)
            train_file = os.path.join(base_path, "v4_atomic_trn.csv")
            
            if os.path.exists(train_file) and "all" in os.path.basename(file_path):
                print("   Using train split for faster performance...")
                file_to_load = train_file
            else:
                file_to_load = file_path
            
            # ATOMIC v4 CSV format: event, oEffect, oReact, oWant, xAttr, xEffect, xIntent, xNeed, xReact, xWant, prefix, split
            self.atomic_data = pd.read_csv(file_to_load)
            
            # Parse JSON columns (first 9 columns after 'event' are relation columns)
            relation_columns = ['oEffect', 'oReact', 'oWant', 'xAttr', 'xEffect', 'xIntent', 'xNeed', 'xReact', 'xWant']
            for col in relation_columns:
                if col in self.atomic_data.columns:
                    self.atomic_data[col] = self.atomic_data[col].apply(
                        lambda x: json.loads(x) if pd.notna(x) and str(x) != '[]' and str(x) != '[""none""]' and str(x) != 'nan' else []
                    )
            
            # OPTIMIZATION: Create lookup index for faster queries
            # Create lowercase event column for faster exact matching
            self.atomic_data['event_lower'] = self.atomic_data['event'].str.lower()
            # Reset index to ensure we can use .loc properly
            self.atomic_data.reset_index(drop=True, inplace=True)
            
            # OPTIMIZATION: Pre-filter to only events with relevant relations
            # This reduces search space significantly
            relation_to_column = {
                'xEffect': 'xEffect', 'xWant': 'xWant', 'xReact': 'xReact',
                'xIntent': 'xIntent', 'xNeed': 'xNeed', 'xAttr': 'xAttr',
                'oEffect': 'oEffect', 'oReact': 'oReact', 'oWant': 'oWant'
            }
            relevant_cols = [relation_to_column.get(r, 'xEffect') for r in self.relations if r in relation_to_column]
            
            # Build a pool of all continuations for each relation (for fallback when no match found)
            self.relation_pools = {}
            for rel in self.relations:
                if rel in relation_to_column:
                    col = relation_to_column[rel]
                    if col in self.atomic_data.columns:
                        all_continuations = []
                        for idx in range(len(self.atomic_data)):
                            row = self.atomic_data.iloc[idx]
                            tails = row[col]
                            if isinstance(tails, list) and len(tails) > 0:
                                # Filter: only keep continuations with more than 5 words
                                valid = [t for t in tails if t and t != "none" and t != "" 
                                        and len(str(t).strip()) > 3 
                                        and len(str(t).strip().split()) > 5]
                                all_continuations.extend(valid)
                        self.relation_pools[rel] = all_continuations
                        print(f"   Pool for {rel}: {len(all_continuations)} continuations (filtered: >5 words)")
            
            if relevant_cols:
                # Keep only rows that have at least one relevant relation
                has_data = self.atomic_data[relevant_cols].apply(
                    lambda row: any(len(row[col]) > 0 for col in relevant_cols if col in row), axis=1
                )
                self.atomic_data = self.atomic_data[has_data].copy()
                # Reset index after filtering
                self.atomic_data.reset_index(drop=True, inplace=True)
            
            # OPTIMIZATION: Create word index for faster fuzzy matching
            # Index events by their key words (rebuild after filtering)
            self.word_index = {}
            for idx in range(len(self.atomic_data)):
                row = self.atomic_data.iloc[idx]
                event_words = set(row['event'].lower().split())
                stop_words = {'personx', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', '___'}
                key_words = [w for w in event_words if w not in stop_words and len(w) > 2]
                for word in key_words:
                    if word not in self.word_index:
                        self.word_index[word] = []
                    self.word_index[word].append(idx)  # Use integer position index
            
            print(f"✅ Loaded {len(self.atomic_data)} ATOMIC events (optimized)")
            print(f"   Word index: {len(self.word_index)} unique words")
            
        except Exception as e:
            print(f"⚠️ Failed to load ATOMIC file: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to template mode")
            self.mode = 'template'
            self.word_index = {}
    
    def _extract_event(self, sentence: str) -> str:
        """
        Extract the main event/action from a sentence.
        Converts natural language to ATOMIC-style event format.
        """
        # Clean up: lowercase, remove trailing punctuation
        clean = sentence.strip()
        if clean.endswith('.'):
            clean = clean[:-1]
        
        # Try to convert to ATOMIC format: "PersonX [action]"
        # This is a simple heuristic - could be improved with NLP
        words = clean.lower().split()
        
        # If sentence starts with pronoun/subject, try to convert to PersonX format
        pronouns = ['she', 'he', 'they', 'it', 'personx']
        if words and words[0] in pronouns:
            # Replace first word with PersonX
            words[0] = 'PersonX'
            return ' '.join(words)
        
        # Otherwise, prepend PersonX
        return f"PersonX {clean.lower()}"
    
    def _query_atomic_file(self, event: str, relation: str, true_next: Optional[str] = None) -> Optional[str]:
        """Query ATOMIC data from loaded file with caching and optional similarity matching."""
        if self.atomic_data is None:
            return None
        
        # Check cache first (but note: cache doesn't include true_next, so similarity selection happens after cache)
        cache_key = (event.lower(), relation)
        cached_result = self.query_cache.get(cache_key)
        
        # If we have true_next and similarity enabled, we'll re-select from candidates
        # Otherwise return cached result
        if cached_result is not None and (true_next is None or not self.use_similarity):
            return cached_result
        
        try:
            import pandas as pd
            
            # ATOMIC v4 format: event column contains the event, relation columns contain lists of continuations
            # Map relation names to column names
            relation_to_column = {
                'xEffect': 'xEffect',
                'xWant': 'xWant',
                'xReact': 'xReact',
                'xIntent': 'xIntent',
                'xNeed': 'xNeed',
                'xAttr': 'xAttr',
                'oEffect': 'oEffect',
                'oReact': 'oReact',
                'oWant': 'oWant'
            }
            
            if relation not in relation_to_column:
                return None
            
            column_name = relation_to_column[relation]
            if column_name not in self.atomic_data.columns:
                return None
            
            # OPTIMIZED: Use word index for faster fuzzy matching
            event_lower = event.lower()
            event_words = set(event_lower.split())
            
            # Remove common stop words for better matching
            stop_words = {'personx', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            key_words = [w for w in event_words if w not in stop_words and len(w) > 2]
            
            matches = pd.DataFrame()
            
            # Try exact match first (fast lookup using boolean indexing)
            exact_mask = self.atomic_data['event_lower'] == event_lower
            if exact_mask.any():
                matches = self.atomic_data[exact_mask]
            
            # OPTIMIZED: Use word index for fuzzy matching (much faster than str.contains on full dataset)
            if len(matches) == 0 and key_words and self.word_index:
                # Find events that share key words using index
                candidate_indices = set()
                for word in key_words[:3]:  # Use top 3 key words
                    if word in self.word_index:
                        candidate_indices.update(self.word_index[word])
                
                if candidate_indices:
                    # Get candidates from indices (use iloc for integer positions)
                    candidate_list = list(candidate_indices)
                    if len(candidate_list) > 100:
                        # Limit to first 100 to avoid processing too many
                        candidate_list = candidate_list[:100]
                    matches = self.atomic_data.iloc[candidate_list]
            
            # Fallback: if still no matches and we have word index, try single word
            if len(matches) == 0 and key_words and self.word_index:
                # Use longest key word
                key_word = max(key_words, key=len)
                if key_word in self.word_index:
                    candidate_indices = self.word_index[key_word]
                    if candidate_indices:
                        candidate_list = list(candidate_indices)[:50]  # Limit to 50
                        matches = self.atomic_data.iloc[candidate_list]
            
            if len(matches) > 0:
                # Get a random matching event
                match = matches.sample(1).iloc[0]
                continuations = match[column_name]
                
                # continuations is a list of strings
                if isinstance(continuations, list) and len(continuations) > 0:
                    # Filter out "none" and empty strings, and very short continuations
                    # Filter: only keep continuations with more than 5 words
                    valid_continuations = [
                        c for c in continuations 
                        if c and c != "none" and c != "" and len(c.strip()) > 3
                        and len(str(c).strip().split()) > 5  # Filter: must have more than 5 words
                    ]
                    if valid_continuations:
                        # Use similarity-based selection if true_next is provided
                        if true_next and self.use_similarity and self.similarity_model is not None:
                            continuation = self._select_by_similarity(valid_continuations, true_next, top_k=5)
                        else:
                            # Random selection
                            continuation = random.choice(valid_continuations)
                        
                        # Format as proper sentence
                        continuation = continuation.strip()
                        
                        # Clean up ATOMIC-specific formatting
                        continuation = continuation.replace('PersonX', 'someone').replace('PersonY', 'someone else')
                        continuation = continuation.replace('personx', 'someone').replace('persony', 'someone else')
                        
                        # Ensure proper capitalization and punctuation
                        if continuation and len(continuation) > 0:
                            if not continuation[0].isupper():
                                continuation = continuation[0].upper() + continuation[1:] if len(continuation) > 1 else continuation.upper()
                            if not continuation.endswith('.'):
                                continuation = continuation + '.'
                        
                        # Cache the result
                        self.query_cache[cache_key] = continuation
                        # Limit cache size to avoid memory issues
                        if len(self.query_cache) > 10000:
                            # Remove oldest 20% of cache
                            keys_to_remove = list(self.query_cache.keys())[:2000]
                            for key in keys_to_remove:
                                del self.query_cache[key]
                        
                        return continuation
            
            # If no match found, sample from the relation pool instead of returning None
            # This ensures we use real ATOMIC data, not templates
            # NOTE: Pool is already filtered for >5 words, so no need to filter again
            if relation in self.relation_pools and len(self.relation_pools[relation]) > 0:
                # Pool is already filtered, just convert to strings
                pool_candidates = [str(c).strip() for c in self.relation_pools[relation] 
                                 if c and str(c).strip()]
                
                if pool_candidates:
                    # Use similarity-based selection if true_next is provided
                    if true_next and self.use_similarity and self.similarity_model is not None:
                        # Sample a subset for efficiency (similarity computation can be slow on large pools)
                        sample_size = min(100, len(pool_candidates))
                        sampled = random.sample(pool_candidates, sample_size)
                        continuation = self._select_by_similarity(sampled, true_next, top_k=10)
                    else:
                        continuation = random.choice(pool_candidates)
                else:
                    continuation = None
                
                if continuation:
                    continuation = str(continuation).strip()
                
                # Clean up ATOMIC-specific formatting
                continuation = continuation.replace('PersonX', 'someone').replace('PersonY', 'someone else')
                continuation = continuation.replace('personx', 'someone').replace('persony', 'someone else')
                
                # Format as proper sentence
                words = continuation.split()
                # Only add prefix if it's really incomplete
                needs_expansion = (len(words) == 1) or (len(words) <= 2 and not any(w in continuation.lower() for w in ['is', 'was', 'are', 'were', 'has', 'have', 'had', 'gets', 'got']))
                
                if needs_expansion:
                    # Short phrases need expansion
                    if relation == 'xEffect':
                        # Only add prefix if it doesn't already start with a complete thought
                        if not continuation.lower().startswith(('as a result', 'consequently', 'therefore', 'thus', 'this led')):
                            continuation = f"As a result, {continuation.lower()}"
                    elif relation == 'xWant':
                        # Check if it already starts with "to" (infinitive)
                        if continuation.lower().startswith('to '):
                            continuation = f"Someone wanted {continuation.lower()}"
                        else:
                            # Add "to" if it's a verb without it
                            continuation = f"Someone wanted to {continuation.lower()}"
                    elif relation == 'xReact':
                        continuation = f"People felt {continuation.lower()}"
                
                # Ensure proper capitalization and punctuation
                if continuation and len(continuation) > 0:
                    if not continuation[0].isupper():
                        continuation = continuation[0].upper() + continuation[1:] if len(continuation) > 1 else continuation.upper()
                    if not continuation.endswith('.'):
                        continuation = continuation + '.'
                    
                    # Cache the result
                    self.query_cache[cache_key] = continuation
                    return continuation
            
            # Cache None result too (to avoid repeated failed lookups)
            self.query_cache[cache_key] = None
            return None
        except Exception as e:
            print(f"Error querying ATOMIC file: {e}")
            return None
    
    def _query_comet(self, event: str, relation: str, current_sentence: str) -> Optional[str]:
        """Query language model for ATOMIC-based story continuation."""
        if self.comet_model is None:
            return None
        
        try:
            # Create prompts based on ATOMIC relation type for story continuation
            prompts = {
                'xEffect': [
                    f"{current_sentence} As a result,",
                    f"{current_sentence} This led to",
                    f"{current_sentence} Consequently,",
                ],
                'xWant': [
                    f"{current_sentence} Someone wanted to",
                    f"{current_sentence} They hoped to",
                    f"{current_sentence} The desire was to",
                ],
                'xReact': [
                    f"{current_sentence} People felt",
                    f"{current_sentence} Emotions were",
                    f"{current_sentence} Everyone reacted",
                ],
                'xNeed': [
                    f"{current_sentence} But first, they needed to",
                    f"{current_sentence} Before that, they had to",
                    f"{current_sentence} To do this, they required",
                ],
                'xIntent': [
                    f"{current_sentence} The intention was to",
                    f"{current_sentence} Someone planned to",
                    f"{current_sentence} The goal was to",
                ]
            }
            
            # Select a prompt based on relation
            if relation in prompts:
                prompt = random.choice(prompts[relation])
            else:
                prompt = f"{current_sentence} Then"
            
            # Tokenize and generate
            inputs = self.comet_tokenizer.encode(prompt, return_tensors='pt', max_length=100, truncation=True)
            
            with torch.no_grad():
                outputs = self.comet_model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 30,  # Generate ~30 more tokens
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=self.comet_tokenizer.eos_token_id
                )
            
            # Decode and extract continuation
            generated = self.comet_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the continuation part (after the prompt)
            if prompt in generated:
                continuation = generated[len(prompt):].strip()
            else:
                continuation = generated[len(current_sentence):].strip()
            
            # Clean up: remove incomplete sentences, ensure it ends properly
            if continuation:
                # Remove trailing incomplete words
                continuation = continuation.split('.')[0] + '.' if '.' in continuation else continuation
                # Capitalize first letter
                if len(continuation) > 1:
                    continuation = continuation[0].upper() + continuation[1:]
            
            return continuation if continuation and len(continuation) > 5 else None
        except Exception as e:
            print(f"Error querying language model: {e}")
            return None
    
    def _generate_template_continuation(self, sentence: str, relation: str, true_next: str = None) -> str:
        """
        Generate a template-based continuation (fallback method).
        Creates plausible alternatives based on relation type.
        
        NOTE: This is a FALLBACK method. For proper ATOMIC usage, you should:
        1. Use 'comet' mode with a language model (GPT-2, GPT-2-medium, etc.)
        2. Use 'file' mode with actual ATOMIC dataset file
        3. Template mode is only for testing/fallback
        
        Args:
            sentence: Current sentence
            relation: ATOMIC relation type
            true_next: Optional true next sentence for better generation
        """
        # Try to create contextually relevant alternatives using true_next as inspiration
        # This is better than completely generic templates
        
        # Don't use true_next directly - it creates incomplete sentences
        # Use standalone templates instead
        
        # Fallback to generic templates if no good variation found
        templates = {
            'xEffect': [
                "After that, things changed.",
                "This led to unexpected consequences.",
                "The situation developed further.",
                "Things took a different turn.",
                "Something unexpected happened next.",
            ],
            'xWant': [
                "They wanted something different.",
                "Someone had other plans.",
                "A new desire emerged.",
                "They hoped for a change.",
            ],
            'xReact': [
                "People reacted strongly.",
                "Emotions ran high.",
                "Everyone had a strong response.",
            ],
            'xNeed': [
                "Something else was needed first.",
                "Before that could happen, other things were required.",
            ],
            'xIntent': [
                "Someone had a different plan.",
                "The goal was something else.",
            ]
        }
        
        if relation in templates:
            return random.choice(templates[relation])
        else:
            return self._create_simple_variation(sentence)
    
    def _extract_key_phrase(self, sentence: str) -> str:
        """Extract a key phrase from sentence for template generation."""
        # Simple extraction: take first few words or main clause
        words = sentence.split()
        if len(words) <= 8:
            return ' '.join(words).lower().rstrip('.')
        else:
            # Take first 6-8 words
            return ' '.join(words[:7]).lower().rstrip('.')
    
    def _create_variations_from_true_next(self, true_next: str, relation: str) -> List[str]:
        """
        Create variations of true_next based on relation type.
        NOTE: This method is kept for potential future use but currently not used
        to avoid awkward sentence wrapping. We use standalone templates instead.
        """
        # This method is deprecated in favor of standalone templates
        # to avoid grammatical issues from wrapping sentences
        return []
    
    def _create_simple_variation(self, sentence: str) -> str:
        """Create a simple variation of the sentence."""
        # Simple paraphrasing variations
        variations = [
            f"Meanwhile, something else was happening.",
            f"At the same time, events unfolded differently.",
            f"Elsewhere, things took a different course.",
            f"Simultaneously, another story was developing.",
        ]
        return random.choice(variations)
    
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
                continuation = self._query_comet(event, relation, current_sentence)
            elif self.mode == 'file':
                continuation = self._query_atomic_file(event, relation, true_next=true_next)
            
            # Only fallback to template if we're in template mode or truly have no ATOMIC data
            # In file mode, we should always have ATOMIC data from the pool
            if continuation is None and self.mode == 'file':
                # Try one more time from the pool (should always work)
                if relation in self.relation_pools and len(self.relation_pools[relation]) > 0:
                    pool_candidates = [str(c).strip() for c in self.relation_pools[relation] 
                                     if c and str(c).strip() and len(str(c).strip()) > 3]
                    
                    if pool_candidates:
                        # Use similarity-based selection if true_next is provided
                        if true_next and self.use_similarity and self.similarity_model is not None:
                            # Sample a subset for efficiency
                            sample_size = min(100, len(pool_candidates))
                            sampled = random.sample(pool_candidates, sample_size)
                            continuation = self._select_by_similarity(sampled, true_next, top_k=10)
                        else:
                            continuation = random.choice(pool_candidates)
                    
                    if continuation:
                        continuation = str(continuation).strip()
                        # Format it
                        continuation = continuation.replace('PersonX', 'someone').replace('PersonY', 'someone else')
                        # Ensure proper capitalization and punctuation
                        if continuation and len(continuation) > 0:
                            if not continuation[0].isupper():
                                continuation = continuation[0].upper() + continuation[1:] if len(continuation) > 1 else continuation.upper()
                            if not continuation.endswith('.'):
                                continuation = continuation + '.'
            
            # Only use template as last resort (shouldn't happen in file mode)
            if continuation is None or len(continuation.strip()) == 0:
                if self.mode == 'template':
                    continuation = self._generate_template_continuation(
                        current_sentence, relation, true_next=true_next
                    )
                else:
                    # In file mode, we should have gotten something from the pool
                    # If not, use a simple variation
                    continuation = f"Something happened related to {relation}."
            
            # Clean up continuation
            continuation = continuation.strip()
            # Capitalize first letter
            if continuation:
                continuation = continuation[0].upper() + continuation[1:] if len(continuation) > 1 else continuation.upper()
            if continuation and continuation != true_next:  # Avoid duplicates
                alternatives.append((continuation, relation))
        
        # Ensure we have at least num_alternatives + 1 (including true)
        while len(alternatives) < self.num_alternatives + 1:
            # Add generic template alternatives
            relation = random.choice(self.relations)
            continuation = self._generate_template_continuation(
                current_sentence, relation, true_next=true_next
            )
            if continuation:
                continuation = continuation.strip()
                if len(continuation) > 1:
                    continuation = continuation[0].upper() + continuation[1:]
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
    
    # Disable similarity during training for speed (can enable for evaluation/visualization)
    use_similarity = config.get('use_similarity', False)  # Default to False for training speed
    
    return AtomicContinuationGenerator(
        mode=config.get('mode', 'template'),
        atomic_file_path=config.get('atomic_file_path'),
        comet_model_name=config.get('comet_model_name'),
        num_alternatives=config.get('num_alternatives', 2),
        relations=config.get('relations', ['xEffect', 'xWant', 'xReact']),
        use_similarity=use_similarity
    )

