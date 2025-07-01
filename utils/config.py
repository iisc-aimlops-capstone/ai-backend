import yaml
import os

def load_yaml_config(file_path: str) -> dict:
    """
    Load a YAML configuration file.
    Args:
        file_path (str): Path to the YAML config file.
    Returns:
        dict: Configurations as a dictionary.
    """
    try:
        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        # Log or handle error if needed
        print(f"Error loading YAML config: {e}")
        return {}