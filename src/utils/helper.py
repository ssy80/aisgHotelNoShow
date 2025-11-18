import yaml
import pandas as pd
import logging
from typing import Dict, Any
from exception.config_key_error import ConfigKeyError


def setup_logging():
    """Set up basic logging for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger = setup_logging()
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        raise Exception(f"Error loading config: {str(e)}")


def safe_get(config: dict, *keys, default=None, required=False):
    """
    Safely get a nested value from a config dictionary.

    Args:
        config (dict): Configuration dictionary.
        *keys: Keys to traverse.
        default: Value to return if key not found.
        required (bool): If True, raise error when missing.

    Returns:
        Value found or default.

    Raises:
        ConfigKeyError: If required key is missing.
    """
    value = config
    path = []

    for key in keys:
        path.append(key)
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            if required:
                raise ConfigKeyError(path)
            return default
    return value
