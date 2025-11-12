import yaml
import pandas as pd
import logging
from typing import Dict, Any
from exception.config_key_error import ConfigKeyError


def setup_logging():
    """Setup logging configuration"""
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
    """Load configuration from YAML file"""
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
    Safely retrieve nested keys from a config dictionary.
    
    Args:
        config (dict): The config dictionary.
        *keys: Sequence of keys to traverse (e.g. 'preprocessing', 'column_mappings', 'target', 'no_show').
        default: Default value to return if the key is missing (only used if required=False).
        required (bool): If True, raises ConfigKeyError when missing.

    Returns:
        The found value or default.

    Raises:
        ConfigKeyError: If a required key path is missing.
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
