"""Load shared application defaults from the repository configuration."""

from functools import lru_cache
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def load_config():
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read application configuration '{CONFIG_PATH}': {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Application configuration '{CONFIG_PATH}' must contain a YAML mapping.")
    return config


def config_value(section, key):
    config = load_config()
    values = config.get(section)
    if not isinstance(values, dict) or key not in values:
        raise ValueError(f"Missing configuration value '{section}.{key}' in '{CONFIG_PATH}'.")
    return values[key]
