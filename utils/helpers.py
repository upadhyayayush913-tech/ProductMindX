"""
Shared utility helper functions.
"""

import re
import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

def clean_text(text: Optional[str]) -> str:
    """Basic text cleanup."""
    if not text or not isinstance(text, str):
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def truncate_text(text: str, max_words: int = 100) -> str:
    """Truncate text to a maximum word count with ellipsis."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

def timestamp_to_str(ts: Any) -> str:
    """Convert millisecond or second timestamp to YYYY-MM format."""
    if not ts:
        return "Unknown"
    try:
        # Check if milliseconds
        ts_float = float(ts)
        if ts_float > 1e11:
            ts_float /= 1000.0
        dt = datetime.fromtimestamp(ts_float)
        return dt.strftime("%Y-%m")
    except Exception:
        return str(ts)[:7]

def cosine_similarity_vec(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def timeit(func):
    """Decorator to measure execution time of a function."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = time.perf_counter() - start
        return result, duration
    return wrapper
