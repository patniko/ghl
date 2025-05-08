import time
import schedule
import asyncio
from typing import List, Dict, Callable
from loguru import logger

# Import configuration
from tasks.config import get_config, get_task_schedule

# Import task handlers
from consumers.notification_consumer import process_notifications
from consumers.evals.dicom_consumer import process_dicom_file
from consumers.evals.csv_consumer import process_csv_file
from consumers.evals.mp4_consumer import process_mp4_file
from consumers.evals.npz_consumer import process_npz_file
from consumers.evals.json_consumer import process_json_file
from consumers.thumbnail_consumer import process_thumbnail_message
from db import SessionLocal
from sqlalchemy import select
from models import File, DicomFile, ProcessingStatus


class Task:
    """Represents a scheduled task with its configuration"""

    def __init__(
        self,
        name: str,
        handler: Callable,
        schedule_interval: str,
        is_async: bool = False,
        enabled: bool = True,
        description: str = "",
    ):
        self.name = name
        self.handler = handler
        self.schedule_interval = schedule_interval
        self.is_async = is_async
        self.enabled = enabled
        self.description = description
        self.job = None  # Will hold the schedule job

    def run(self):
        """Execute the task handler with proper async handling if needed"""
        if not self.enabled:
            logger.debug(f"Task '{self.name}' is disabled")
            return

        try:
            # Execute the task handler
            if self.is_async:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.handler())
                loop.close()
            else:
                self.handler()

            # logger.debug(f"Task '{self.name}' completed successfully")
        except Exception as e:
            logger.error(f"Error in task '{self.name}': {str(e)}")


class TaskScheduler:
    """Manages scheduling of various service tasks"""

    def __init__(
        self, disabled_tasks: List[str] = None, enabled_tasks: List[str] = None
    ):
        """
        Initialize the task scheduler

        Args:
            disabled_tasks: List of task names to disable
            enabled_tasks: List of task names to enable (overrides disabled_tasks)
        """
        self.tasks: Dict[str, Task] = {}

        # Initialize with default task configurations
        self._initialize_tasks()

        # Apply task overrides
        if disabled_tasks:
            for task_name in disabled_tasks:
                self.disable_task(task_name)

        if enabled_tasks:
            # First disable all tasks
            for task_name in self.tasks.keys():
                self.disable_task(task_name)

            # Then enable only the specified tasks
            for task_name in enabled_tasks:
                self.enable_task(task_name)

    async def run_process_notifications(self):
        """Process notifications asynchronously"""
        if not self.tasks["notifications"].enabled:
            logger.debug("Notification processing is disabled")
            return

        try:
            await process_notifications()
        except Exception as e:
            logger.error(f"Error in notification processing: {str(e)}")

    async def run_file_processing(self):
        """Process pending files directly"""
        if not self.tasks["file_processing"].enabled:
            logger.debug("File processing is disabled")
            return

        try:
            db = SessionLocal()
            try:
                # Get files that are in pending status
                pending_files = (
                    db.execute(
                        select(File).where(
                            File.processing_status == ProcessingStatus.PENDING
                        )
                    )
                    .scalars()
                    .all()
                )

                if not pending_files:
                    return

                logger.info(
                    f"Queueing {len(pending_files)} pending files for processing"
                )

                # Process each file directly
                for file in pending_files:
                    try:
                        # Update status to processing
                        file.processing_status = ProcessingStatus.PROCESSING
                        db.commit()

                        # Process based on file type
                        file_type = file.file_type.lower()
                        if file_type in ["dicom", "dcm"]:
                            await process_dicom_file(file.id, file.user_id)
                        elif file_type == "csv":
                            await process_csv_file(file.id, file.user_id)
                        elif file_type == "mp4":
                            await process_mp4_file(file.id, file.user_id)
                        elif file_type == "npz":
                            await process_npz_file(file.id, file.user_id)
                        elif file_type == "json":
                            await process_json_file(file.id, file.user_id)
                        else:
                            error_msg = (
                                f"No processor available for file type: {file_type}"
                            )
                            logger.warning(error_msg)
                            file.processing_status = ProcessingStatus.FAILED
                            file.processing_results = {
                                "message": error_msg,
                                "error": error_msg,
                                "details": {
                                    "file_type": file_type,
                                    "supported_types": [
                                        "dicom",
                                        "dcm",
                                        "csv",
                                        "mp4",
                                        "npz",
                                        "json",
                                    ],
                                },
                            }
                            db.commit()
                    except Exception as e:
                        error_msg = f"Error processing file {file.id}: {str(e)}"
                        logger.error(error_msg)
                        file.processing_status = ProcessingStatus.FAILED
                        file.processing_results = {
                            "message": error_msg,
                            "error": str(e),
                            "details": {"file_type": file_type, "file_id": file.id},
                        }
                        db.commit()
                        continue

                logger.info("File processing completed")

            except Exception as e:
                logger.error(f"Error in file processing: {str(e)}")
                db.rollback()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Critical error in file processing: {str(e)}")

    async def run_thumbnail_processor(self):
        """Process thumbnails directly"""
        if not self.tasks["thumbnail_processing"].enabled:
            logger.debug("Thumbnail processing is disabled")
            return

        try:
            db = SessionLocal()
            try:
                # Get regular files that have been processed successfully but don't have thumbnails yet
                completed_files = (
                    db.execute(
                        select(File).where(
                            File.processing_status == ProcessingStatus.COMPLETED
                        )
                    )
                    .scalars()
                    .all()
                )

                # Filter regular files that don't have thumbnails
                files_without_thumbnails = [
                    file for file in completed_files if not file.has_thumbnail
                ]

                # Get DICOM files that have been processed successfully but don't have thumbnails yet
                completed_dicom_files = (
                    db.execute(
                        select(DicomFile).where(
                            DicomFile.processing_status == ProcessingStatus.COMPLETED
                        )
                    )
                    .scalars()
                    .all()
                )

                # Filter DICOM files that don't have thumbnails
                dicom_files_without_thumbnails = [
                    file for file in completed_dicom_files if not file.has_thumbnail
                ]

                total_files_to_process = len(files_without_thumbnails) + len(
                    dicom_files_without_thumbnails
                )
                if total_files_to_process == 0:
                    return

                logger.info(
                    f"Queueing {total_files_to_process} files for thumbnail generation"
                )

                # Process regular files for thumbnail generation
                for file in files_without_thumbnails:
                    try:
                        message = {
                            "file_id": file.id,
                            "user_id": file.user_id,
                            "file_type": file.file_type,
                            "file_category": "file",
                        }
                        await process_thumbnail_message(message)
                    except Exception as e:
                        logger.error(
                            f"Error generating thumbnail for file {file.id}: {str(e)}"
                        )
                        continue

                # Process DICOM files for thumbnail generation
                for dicom_file in dicom_files_without_thumbnails:
                    try:
                        message = {
                            "file_id": dicom_file.id,
                            "user_id": dicom_file.user_id,
                            "file_type": "dicom",
                            "file_category": "dicom",
                        }
                        await process_thumbnail_message(message)
                    except Exception as e:
                        logger.error(
                            f"Error generating thumbnail for DICOM file {dicom_file.id}: {str(e)}"
                        )
                        continue

                logger.info("Thumbnail processing completed")

            except Exception as e:
                logger.error(f"Error in thumbnail processing: {str(e)}")
                db.rollback()
            finally:
                db.close()
        except Exception as e:
            logger.error(f"Critical error in thumbnail processing: {str(e)}")

    def _initialize_tasks(self):
        """Initialize all available tasks with their default configurations"""
        # Get configuration
        config = get_config()

        # Initialize tasks with configuration
        self.tasks = {
            "notifications": Task(
                name="notifications",
                handler=self.run_process_notifications,
                schedule_interval=get_task_schedule("notifications") or "30 seconds",
                is_async=True,
                enabled=config.get("TASK_NOTIFICATIONS_ENABLED", True),
                description="Process pending notifications",
            ),
            "file_processing": Task(
                name="file_processing",
                handler=self.run_file_processing,
                schedule_interval=get_task_schedule("file_processing") or "30 seconds",
                is_async=True,
                enabled=config.get("TASK_FILE_PROCESSING_ENABLED", True),
                description="Process pending files",
            ),
            "thumbnail_processing": Task(
                name="thumbnail_processing",
                handler=self.run_thumbnail_processor,
                schedule_interval=get_task_schedule("thumbnail_processing")
                or "60 seconds",
                is_async=True,
                enabled=config.get("TASK_THUMBNAIL_PROCESSING_ENABLED", True),
                description="Process thumbnail generation",
            ),
        }

    def enable_task(self, task_name: str) -> bool:
        """Enable a task by name"""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = True
            logger.info(f"Task '{task_name}' enabled")
            return True

        logger.warning(f"Cannot enable unknown task: '{task_name}'")
        return False

    def disable_task(self, task_name: str) -> bool:
        """Disable a task by name"""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = False
            logger.info(f"Task '{task_name}' disabled")
            return True

        logger.warning(f"Cannot disable unknown task: '{task_name}'")
        return False

    def _schedule_tasks(self):
        """Schedule all tasks according to their configurations"""
        for task_name, task in self.tasks.items():
            # Parse the schedule interval
            parts = task.schedule_interval.split()

            if len(parts) < 2:
                logger.error(
                    f"Invalid schedule interval format for task '{task_name}': {task.schedule_interval}"
                )
                continue

            try:
                # Get the time value (first part)
                time_value = int(parts[0])

                # Check for "at" format (e.g. "5 minutes at :00")
                if len(parts) >= 3 and parts[2] == "at":
                    # Format: "X minute(s) at :YY"
                    if parts[1] in ["minute", "minutes"]:
                        at_time = parts[3]
                        task.job = (
                            schedule.every(time_value).minutes.at(at_time).do(task.run)
                        )
                    else:
                        logger.error(
                            f"Unsupported time unit for 'at' format in task '{task_name}': {parts[1]}"
                        )
                        continue
                # Regular interval format
                else:
                    # Format: "X minute(s)"
                    if parts[1] in ["minute", "minutes"]:
                        task.job = schedule.every(time_value).minutes.do(task.run)
                    # Format: "X second(s)"
                    elif parts[1] in ["second", "seconds"]:
                        task.job = schedule.every(time_value).seconds.do(task.run)
                    # Format: "X hour(s)"
                    elif parts[1] in ["hour", "hours"]:
                        task.job = schedule.every(time_value).hours.do(task.run)
                    # Format: "X day(s)"
                    elif parts[1] in ["day", "days"]:
                        task.job = schedule.every(time_value).days.do(task.run)
                    else:
                        logger.error(
                            f"Unsupported time unit in task '{task_name}': {parts[1]}"
                        )
                        continue

                logger.debug(
                    f"Scheduled task '{task_name}' with interval '{task.schedule_interval}'"
                )
            except (ValueError, IndexError) as e:
                logger.error(
                    f"Error parsing schedule interval for task '{task_name}': {str(e)}"
                )
                continue

    def start(self):
        """Start the scheduler with all enabled tasks"""
        try:
            logger.info("Starting task scheduler")

            # Schedule all tasks
            self._schedule_tasks()
            logger.info("All tasks scheduled successfully")

            # Main scheduler loop
            while True:
                try:
                    # Run pending tasks
                    schedule.run_pending()
                    time.sleep(5)  # Sleep to reduce CPU usage
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {str(e)}")
                    time.sleep(5)  # Prevent tight error loop
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted, shutting down...")
        except Exception as e:
            logger.error(f"Critical error in scheduler: {str(e)}")
            raise  # Re-raise the exception to notify the parent thread
