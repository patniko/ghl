import os
from typing import Dict, Any, List, Optional

# Default configuration
DEFAULT_CONFIG = {
    # Task enablement
    "TASKS_ENABLED": True,
    # Individual task configuration
    "TASK_NOTIFICATIONS_ENABLED": True,
    "TASK_FILE_PROCESSING_ENABLED": True,
    "TASK_THUMBNAIL_PROCESSING_ENABLED": True,
    # Schedule intervals
    "SCHEDULE_NOTIFICATIONS": "30 seconds",
    "SCHEDULE_FILE_PROCESSING": "10 seconds",
    "SCHEDULE_THUMBNAIL_PROCESSING": "60 seconds",
    # Logging
    "LOG_LEVEL": "INFO",
}


def get_config() -> Dict[str, Any]:
    """
    Get configuration from environment variables falling back to defaults

    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()

    # Override with environment variables
    for key in config.keys():
        env_value = os.getenv(key)
        if env_value is not None:
            # Convert string boolean values
            if env_value.lower() in ("true", "false"):
                config[key] = env_value.lower() == "true"
            # Convert numeric values
            elif env_value.isdigit():
                config[key] = int(env_value)
            else:
                config[key] = env_value

    return config


def get_enabled_tasks() -> List[str]:
    """
    Get list of enabled tasks based on configuration

    Returns:
        List[str]: List of enabled task names
    """
    config = get_config()
    enabled_tasks = []

    # Check if tasks are globally enabled
    if not config["TASKS_ENABLED"]:
        return []

    # Check individual task enablement
    task_mapping = {
        "TASK_NOTIFICATIONS_ENABLED": "notifications",
        "TASK_FILE_PROCESSING_ENABLED": "file_processing",
        "TASK_THUMBNAIL_PROCESSING_ENABLED": "thumbnail_processing",
    }

    for config_key, task_name in task_mapping.items():
        if config[config_key]:
            enabled_tasks.append(task_name)

    return enabled_tasks


def get_task_schedule(task_name: str) -> Optional[str]:
    """
    Get schedule interval for a specific task

    Args:
        task_name: Name of the task

    Returns:
        Optional[str]: Schedule interval or None if not found
    """
    config = get_config()

    task_mapping = {
        "notifications": "SCHEDULE_NOTIFICATIONS",
        "file_processing": "SCHEDULE_FILE_PROCESSING",
        "thumbnail_processing": "SCHEDULE_THUMBNAIL_PROCESSING",
    }

    if task_name in task_mapping and task_mapping[task_name] in config:
        return config[task_mapping[task_name]]

    return None
