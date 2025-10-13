import json
import numpy as np

FILE_PATH = "full_benchmark_results.json" 
TIME_KEY = "inference_time"

def find_min_time_value(file_path):
    """
    Loads the benchmark results and finds the absolute minimum value for 
    'inference_time' across all loops and all ensembles.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}")
        return None
    
    # Use a generator to extract all valid time values, then find the minimum
    try:
        max_time = max(
            ensemble_data["metrics"][TIME_KEY]
            for loop_results in data.values()
            for ensemble_data in loop_results.values()
            if "metrics" in ensemble_data and TIME_KEY in ensemble_data["metrics"]
        )
        return float(max_time)
    
    except ValueError:
        # Raised if the generator is empty (no time values found)
        print(f"No valid '{TIME_KEY}' values were found in the file.")
        return None

# --- Execution ---
min_time_value = find_min_time_value(FILE_PATH)

if min_time_value is not None:
    print("\n✅ **Absolute max Inference Time**")
    print(f"The highest time recorded is: {min_time_value:.6f} seconds")