"""
Dataset Manager for StoryCommonsense Dataset
Handles loading stories from CSV and annotations from JSON
"""

import json
import os
import random
from typing import List, Dict, Optional, Iterator
from dataset_loader import load_stories_batch, load_story_from_csv


class DatasetManager:
    """
    Manages the StoryCommonsense dataset including stories and annotations.
    """
    
    def __init__(self, csv_path: str, json_annotations_path: Optional[str] = None,
                 split: str = 'train', max_stories: Optional[int] = None):
        """
        Initialize the dataset manager.
        
        Args:
            csv_path: Path to rocstorysubset.csv
            json_annotations_path: Optional path to annotations.json
            split: 'train', 'dev', or 'test'
            max_stories: Optional maximum number of stories to load
        """
        self.csv_path = csv_path
        self.json_annotations_path = json_annotations_path
        self.split = split
        self.max_stories = max_stories
        
        # Load stories
        self.stories = load_stories_batch(csv_path, split=split, max_stories=max_stories)
        print(f"Loaded {len(self.stories)} stories from {split} split")
        
        # Load annotations if provided
        self.annotations = {}
        if json_annotations_path and os.path.exists(json_annotations_path):
            self._load_annotations()
        else:
            print("No annotations file provided or file not found. Running without annotations.")
    
    def _load_annotations(self):
        """Load JSON annotations file."""
        try:
            with open(self.json_annotations_path, "r", encoding="utf-8") as f:
                self.annotations = json.load(f)
            print(f"Loaded annotations for {len(self.annotations)} stories")
        except Exception as e:
            print(f"Warning: Could not load annotations: {e}")
            self.annotations = {}
    
    def get_story(self, story_id: Optional[str] = None) -> Dict:
        """
        Get a story by ID or return a random story.
        
        Args:
            story_id: Optional story ID. If None, returns random story.
        
        Returns:
            Story dictionary with annotations if available
        """
        if story_id:
            # Find specific story
            for story in self.stories:
                if story['storyid'] == story_id:
                    return self._add_annotations(story)
            raise ValueError(f"Story ID {story_id} not found")
        else:
            # Return random story
            story = random.choice(self.stories)
            return self._add_annotations(story)
    
    def _add_annotations(self, story: Dict) -> Dict:
        """
        Add annotations to a story if available.
        
        Args:
            story: Story dictionary
        
        Returns:
            Story dictionary with annotations added
        """
        story_id = story['storyid']
        if story_id in self.annotations:
            story['annotations'] = self.annotations[story_id]
        else:
            story['annotations'] = None
        return story
    
    def get_story_iterator(self, shuffle: bool = True) -> Iterator[Dict]:
        """
        Get an iterator over all stories.
        
        Args:
            shuffle: Whether to shuffle stories
        
        Yields:
            Story dictionaries
        """
        stories = self.stories.copy()
        if shuffle:
            random.shuffle(stories)
        
        for story in stories:
            yield self._add_annotations(story)
    
    def __len__(self) -> int:
        """Return number of stories."""
        return len(self.stories)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get story by index."""
        return self._add_annotations(self.stories[idx])
    
    def get_story_count(self) -> int:
        """Get total number of stories."""
        return len(self.stories)
    
    def get_story_ids(self) -> List[str]:
        """Get list of all story IDs."""
        return [story['storyid'] for story in self.stories]

