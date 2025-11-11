import json
import csv
import os
from typing import List, Dict, Optional

def load_story(path: str):
    """
    Loads a single story JSON and returns a list of line texts.
    Maintains backward compatibility with legacy JSON format.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Story file not found: {path}")
    
    print(f"Loading story from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Sort by numeric order of keys to preserve sequence
    story_lines = data.get("lines", {})
    sorted_lines = [story_lines[k]["text"] for k in sorted(story_lines.keys(), key=int)]
    return sorted_lines


def load_story_from_csv(csv_path: str, story_id: Optional[str] = None) -> Dict:
    """
    Load a story from CSV file (rocstorysubset.csv format).
    
    Args:
        csv_path: Path to the CSV file
        story_id: Optional story ID to load specific story. If None, returns first story.
    
    Returns:
        Dictionary with story data:
        {
            'storyid': str,
            'title': str,
            'lines': List[str],  # 5 sentences
            'split': str  # 'train', 'dev', or 'test'
        }
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if story_id is None or row['storyid'] == story_id:
                story_data = {
                    'storyid': row['storyid'],
                    'title': row['storytitle'],
                    'lines': [
                        row['sentence1'],
                        row['sentence2'],
                        row['sentence3'],
                        row['sentence4'],
                        row['sentence5']
                    ],
                    'split': row['split']
                }
                return story_data
    
    if story_id:
        raise ValueError(f"Story ID {story_id} not found in CSV file")
    raise ValueError("No stories found in CSV file")


def load_stories_batch(csv_path: str, split: Optional[str] = None, 
                       max_stories: Optional[int] = None) -> List[Dict]:
    """
    Load multiple stories from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        split: Optional filter by split ('train', 'dev', 'test')
        max_stories: Optional maximum number of stories to load
    
    Returns:
        List of story dictionaries
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    stories = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter by split if specified
            if split and row['split'] != split:
                continue
            
            story_data = {
                'storyid': row['storyid'],
                'title': row['storytitle'],
                'lines': [
                    row['sentence1'],
                    row['sentence2'],
                    row['sentence3'],
                    row['sentence4'],
                    row['sentence5']
                ],
                'split': row['split']
            }
            stories.append(story_data)
            
            # Stop if we've reached max_stories
            if max_stories and len(stories) >= max_stories:
                break
    
    return stories


def convert_story_to_json_format(story_data: Dict) -> Dict:
    """
    Convert story data from CSV format to JSON-like format compatible with existing code.
    
    Args:
        story_data: Story dictionary from load_story_from_csv or load_stories_batch
    
    Returns:
        Dictionary in JSON format:
        {
            'lines': {
                '1': {'text': '...'},
                '2': {'text': '...'},
                ...
            },
            'title': str,
            'storyid': str
        }
    """
    json_format = {
        'lines': {},
        'title': story_data.get('title', ''),
        'storyid': story_data.get('storyid', '')
    }
    
    for idx, line_text in enumerate(story_data['lines'], start=1):
        json_format['lines'][str(idx)] = {'text': line_text}
    
    return json_format
