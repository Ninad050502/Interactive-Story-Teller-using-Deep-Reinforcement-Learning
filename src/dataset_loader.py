import json

def load_story(path: str):
    """
    Loads a single story JSON and returns a list of line texts.
    """
    print(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Sort by numeric order of keys to preserve sequence
    story_lines = data.get("lines", {})
    sorted_lines = [story_lines[k]["text"] for k in sorted(story_lines.keys(), key=int)]
    return sorted_lines
