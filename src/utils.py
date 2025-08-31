import math
import os
import hashlib
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from src.models import Criterion, EvaluationPlan # Import both Criterion and EvaluationPlan

logger = logging.getLogger(__name__) # Get the logger instance

def normalize_weights(criteria_list: List, total_weight: int = 100) -> None:
    """
    Normalizes the 'weightage' of a list of Criterion objects IN-PLACE,
    scaling their given weightages so that the total equals `total_weight`,
    preserving the relative proportions.

    Args:
        criteria_list: A list of Criterion objects with `weightage` attributes.
        total_weight: Desired total sum of weightages (default is 100).
    """
    if not criteria_list:
        return

    current_weights = [c.weightage if isinstance(c.weightage, (int, float)) else 0 for c in criteria_list]
    sum_current = sum(current_weights)

    if sum_current == 0:
        # Avoid division by zero; assign equal weight
        equal_weight = total_weight // len(criteria_list)
        remainder = total_weight % len(criteria_list)
        for i, c in enumerate(criteria_list):
            c.weightage = equal_weight + (1 if i < remainder else 0)
        return

    # Step 1: Scale weights proportionally
    scaled_weights_float = [(w / sum_current) * total_weight for w in current_weights]
    scaled_weights_int = [math.floor(w) for w in scaled_weights_float]

    # Step 2: Distribute rounding remainder
    remainder = total_weight - sum(scaled_weights_int)
    fractional_parts = [(scaled_weights_float[i] - scaled_weights_int[i], i) for i in range(len(criteria_list))]
    fractional_parts.sort(reverse=True)

    for i in range(remainder):
        _, idx = fractional_parts[i]
        scaled_weights_int[idx] += 1

    # Step 3: Assign back
    for i, c in enumerate(criteria_list):
        c.weightage = scaled_weights_int[i]

    # Optional: Log the final weights for verification
    # final_sum = sum(c.weightage for c in criteria_list)
    # logger.debug(f"Normalization complete. Final weights sum: {final_sum}/{total_weight}")

# --- JD Caching Functions ---

JD_CACHE_DIR = "jd_cache"
os.makedirs(JD_CACHE_DIR, exist_ok=True) # Ensure cache directory exists

def get_jd_cache_key(jd_filepath: str) -> str:
    """Generates a cache key based on the filename and its modification timestamp."""
    try:
        # Using filename and modification time for simplicity.
        # For more robust caching, consider hashing file content.
        mod_time = os.path.getmtime(jd_filepath)
        filename_hash = hashlib.md5(os.path.basename(jd_filepath).encode()).hexdigest()
        return f"{filename_hash}_{int(mod_time)}.json"
    except FileNotFoundError:
        return "" # Return empty if file not found, which will result in cache miss


def load_jd_cache(cache_key: str) -> Optional[Dict[str, Any]]:
    """Loads JD data from the cache."""
    cache_filepath = os.path.join(JD_CACHE_DIR, cache_key)
    if os.path.exists(cache_filepath):
        try:
            with open(cache_filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JD cache file {cache_filepath}: {e}")
            return None
    return None

def save_jd_cache(cache_key: str, jd_text: str, parsed_jd_dict: Dict[str, Any], evaluation_plan_dict: Dict[str, Any]) -> None:
    """Saves JD data to the cache."""
    cache_filepath = os.path.join(JD_CACHE_DIR, cache_key)
    try:
        cache_data = {
            "jd_text": jd_text,
            "parsed_jd": parsed_jd_dict,
            "evaluation_plan": evaluation_plan_dict
        }
        with open(cache_filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, default=str) # Use default=str for any non-serializable types if they sneak in
        print(f"✅ JD cache saved to {cache_filepath}")
    except Exception as e:
        print(f"❌ Error saving JD cache to {cache_filepath}: {e}")

def make_json_serializable(obj):
    if isinstance(obj, EvaluationPlan):
        return obj.model_dump()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj