#!/usr/bin/env python3
import os
import sys
import argparse
from loguru import logger

# Add parent directory to path to allow imports from the main project
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import these modules after path setup but before they're used
# noqa: E402 tells the linter to ignore the "not at top of file" warning
from tasks.scheduler import TaskScheduler  # noqa: E402
from tasks.config import get_config, get_enabled_tasks  # noqa: E402


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="GHL Progress Hub Task Service")
    parser.add_argument(
        "--disable-tasks",
        type=str,
        help="Comma-separated list of tasks to disable",
        default="",
    )
    parser.add_argument(
        "--enable-tasks",
        type=str,
        help="Comma-separated list of tasks to enable (overrides disable-tasks)",
        default="",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Set the logging level (overrides environment variable)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        help="Path to environment file to load",
        default=None,
    )
    return parser.parse_args()


def load_env_file(env_file_path):
    """Load environment variables from file"""
    if not env_file_path or not os.path.exists(env_file_path):
        return

    logger.info(f"Loading environment from {env_file_path}")
    with open(env_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def configure_logging(log_level=None):
    """Configure logging with loguru"""
    # Use provided log level or get from config
    if log_level is None:
        config = get_config()
        log_level = config.get("LOG_LEVEL", "INFO")

    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


def main():
    """Main entry point for the task service"""
    args = parse_args()

    # Load environment file if specified
    if args.env_file:
        load_env_file(args.env_file)

    # Configure logging
    configure_logging(args.log_level)

    logger.info("Starting GHL Progress Hub Task Service")

    # Get enabled tasks from config then override with command line args
    enabled_tasks = get_enabled_tasks()

    # Override with command line arguments if provided
    if args.disable_tasks:
        disabled_tasks = [t.strip() for t in args.disable_tasks.split(",") if t.strip()]
        enabled_tasks = [t for t in enabled_tasks if t not in disabled_tasks]

    if args.enable_tasks:
        # Replace enabled tasks with command line list
        enabled_tasks = [t.strip() for t in args.enable_tasks.split(",") if t.strip()]

    logger.info(
        f"Enabled tasks: {', '.join(enabled_tasks) if enabled_tasks else 'None'}"
    )

    # Initialize and start the scheduler
    scheduler = TaskScheduler(enabled_tasks=enabled_tasks)

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Task service interrupted, shutting down...")
    except Exception as e:
        logger.error(f"Critical error in task service: {str(e)}")
        sys.exit(1)

    logger.info("Task service shutdown complete")


if __name__ == "__main__":
    main()
