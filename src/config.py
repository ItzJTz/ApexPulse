"""Load shared application defaults from the repository configuration."""

import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
LOCAL_CONFIG_PATH = CONFIG_PATH.with_name("config.local.yaml")

DRIVER_INPUT_PROFILE_KEYS = (
    "brake_release_slope_pct_s",
    "brake_release_min_speed_kmh",
    "throttle_snap_slope_pct_s",
    "loaded_steering_angle_deg",
    "trail_brake_steering_angle_deg",
)


def _merge_local_config(config, local_config):
    if not isinstance(local_config, dict):
        raise ValueError(
            f"Local configuration '{LOCAL_CONFIG_PATH}' must contain a YAML mapping."
        )
    unsupported = set(local_config) - {"driver_inputs"}
    if unsupported:
        raise ValueError(
            "Local configuration only supports driver_inputs overrides; "
            f"unsupported sections: {', '.join(sorted(unsupported))}."
        )
    local_inputs = local_config.get("driver_inputs", {})
    if not isinstance(local_inputs, dict):
        raise ValueError("Local driver_inputs configuration must be a mapping.")
    if not local_inputs:
        return config

    driver_inputs = config.get("driver_inputs")
    if not isinstance(driver_inputs, dict):
        raise ValueError("Base driver_inputs configuration must be a mapping.")
    if "active_profile" in local_inputs:
        driver_inputs["active_profile"] = local_inputs["active_profile"]
    local_profiles = local_inputs.get("profiles", {})
    profiles = driver_inputs.get("profiles")
    if not isinstance(local_profiles, dict) or not isinstance(profiles, dict):
        raise ValueError("Driver input profiles must be mappings.")
    for profile_name, profile_values in local_profiles.items():
        if profile_name not in profiles:
            raise ValueError(
                f"Local input profile '{profile_name}' is not defined in config.yaml."
            )
        if not isinstance(profile_values, dict):
            raise ValueError(
                f"Local input profile '{profile_name}' must be a mapping."
            )
        if not isinstance(profiles[profile_name], dict):
            raise ValueError(
                f"Base input profile '{profile_name}' must be a mapping."
            )
        profiles[profile_name].update(profile_values)
    return config


@lru_cache(maxsize=1)
def load_config():
    try:
        config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read application configuration '{CONFIG_PATH}': {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Application configuration '{CONFIG_PATH}' must contain a YAML mapping.")
    if LOCAL_CONFIG_PATH.exists():
        try:
            local_config = yaml.safe_load(
                LOCAL_CONFIG_PATH.read_text(encoding="utf-8")
            )
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(
                f"Could not read local configuration '{LOCAL_CONFIG_PATH}': {exc}"
            ) from exc
        config = _merge_local_config(config, local_config)
    return config


def config_value(section, key):
    config = load_config()
    values = config.get(section)
    if not isinstance(values, dict) or key not in values:
        raise ValueError(f"Missing configuration value '{section}.{key}' in '{CONFIG_PATH}'.")
    return values[key]


def save_driver_input_settings(active_profile, profiles):
    """Persist validated driver-input settings as a local config override."""
    config = load_config()
    driver_inputs = config.get("driver_inputs")
    if not isinstance(driver_inputs, dict):
        raise ValueError("driver_inputs configuration must be a mapping.")
    configured_profiles = driver_inputs.get("profiles")
    if not isinstance(configured_profiles, dict):
        raise ValueError("driver_inputs.profiles must be a mapping.")
    if (
        not isinstance(active_profile, str)
        or active_profile not in configured_profiles
    ):
        raise ValueError(f"Unknown driver input profile '{active_profile}'.")
    if not isinstance(profiles, dict) or set(profiles) != set(configured_profiles):
        raise ValueError("Settings must include every configured driver input profile.")

    validated_profiles = {}
    for profile_name, profile_values in profiles.items():
        if not isinstance(profile_values, dict):
            raise ValueError(f"Settings for '{profile_name}' must be a mapping.")
        values = {}
        for key in DRIVER_INPUT_PROFILE_KEYS:
            try:
                value = float(profile_values[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{profile_name}.{key} must be a finite number."
                ) from exc
            if not math.isfinite(value) or value < 0:
                raise ValueError(
                    f"{profile_name}.{key} must be finite and non-negative."
                )
            values[key] = value
        for key in (
            "brake_release_slope_pct_s",
            "throttle_snap_slope_pct_s",
            "loaded_steering_angle_deg",
            "trail_brake_steering_angle_deg",
        ):
            if values[key] == 0:
                raise ValueError(f"{profile_name}.{key} must be positive.")
        validated_profiles[profile_name] = values

    local_config = {
        "driver_inputs": {
            "active_profile": active_profile,
            "profiles": validated_profiles,
        }
    }
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=LOCAL_CONFIG_PATH.parent,
            prefix=".config.local.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            temporary.write(yaml.safe_dump(local_config, sort_keys=False))
        os.replace(temp_path, LOCAL_CONFIG_PATH)
    except OSError as exc:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise OSError(
            f"Could not save local driver-input settings to "
            f"'{LOCAL_CONFIG_PATH}': {exc}"
        ) from exc

    load_config.cache_clear()
    return LOCAL_CONFIG_PATH
