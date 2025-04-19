import sys
from pathlib import Path
import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from model.app import main as app_analyze_clip

def analyze_clip(clip_path):
    """
    Analyze a video clip and return the result.
    
    Args:
        clip_path (str): Path to the video clip to analyze
        
    Returns:
        dict: Analysis result with metadata
    """
    raw_result = app_analyze_clip(clip_path)
    return {
        "clip_path": clip_path,
        "analysis": raw_result,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        clip_path = sys.argv[1]
        result = analyze_clip(clip_path)
        print(result)
    else:
        print("Please provide a video clip path as an argument")
        print("Usage: python helpers.py /path/to/video.mov")