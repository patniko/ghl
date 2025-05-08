import json
from typing import Dict, Any, List
from loguru import logger
from confluent_kafka import Consumer

from consumers.thumbnails.thumbnail_processor import (
    process_file_thumbnail,
    process_dicom_thumbnail,
)

# Supported file types for thumbnail generation
SUPPORTED_FILE_TYPES = {
    "csv",
    "mp4",
    "npz",  # Regular files
    "dicom",
    "dcm",  # DICOM files
}


async def process_thumbnail_message(message: Dict[str, Any]) -> None:
    """
    Process a thumbnail message from Kafka

    Args:
        message: The message payload from Kafka containing file metadata
        Required fields:
            - file_id: int
            - user_id: int
            - file_type: str
            - file_category: str (optional, defaults to "file")
    """
    try:
        # Extract and validate required fields
        try:
            file_id = int(message.get("file_id"))
            user_id = int(message.get("user_id"))
            file_type = str(message.get("file_type", "")).lower()
            file_category = str(message.get("file_category", "file"))
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid field type in message {message}: {str(e)}")
            return

        # Validate required fields
        if not all([file_id > 0, user_id > 0, file_type]):
            logger.error(
                f"Invalid or missing required fields: file_id={file_id}, user_id={user_id}, file_type={file_type}"
            )
            return

        # Validate file type
        if file_type not in SUPPORTED_FILE_TYPES:
            logger.warning(
                f"Unsupported file type for thumbnail generation: {file_type}"
            )
            return

        logger.info(
            f"Processing thumbnail for {file_category} {file_id} of type {file_type}"
        )

        # Process the thumbnail based on file category and type
        if file_category == "dicom" or file_type in {"dicom", "dcm"}:
            await process_dicom_thumbnail(file_id, user_id)
        else:
            await process_file_thumbnail(file_id, user_id)

    except ValueError as e:
        logger.error(f"Invalid value in thumbnail message: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing thumbnail message: {str(e)}", exc_info=True)


async def consume_thumbnail_messages(messages: List[Any]) -> None:
    """
    Consume thumbnail messages from Kafka

    Args:
        messages: List of messages from Kafka
    """
    for message_data in messages:
        try:
            # Parse the message
            try:
                message_value = message_data.value.decode("utf-8")
                message = json.loads(message_value)
            except UnicodeDecodeError as e:
                logger.error(f"Failed to decode message: {str(e)}")
                continue
            except json.JSONDecodeError as e:
                logger.error(
                    f"Invalid JSON in message: {str(e)}, value: {message_data.value}"
                )
                continue
            except AttributeError as e:
                logger.error(f"Invalid message format: {str(e)}")
                continue

            # Process the message
            await process_thumbnail_message(message)

        except Exception as e:
            logger.error(
                f"Error consuming message from partition {message_data.partition}, "
                f"offset {message_data.offset}: {str(e)}",
                exc_info=True,
            )


# This function would be called by the Kafka consumer
async def handle_thumbnail_messages(consumer: Consumer, max_messages: int = 10) -> None:
    """
    Handle thumbnail messages from Kafka

    Args:
        consumer: The Kafka consumer
        max_messages: Maximum number of messages to process at once (1-100)
    """
    try:
        # Validate max_messages
        max_messages = max(1, min(100, max_messages))

        # Poll for messages
        messages = consumer.poll(timeout_ms=1000, max_records=max_messages)

        if not messages:
            logger.debug("No messages received from poll")
            return

        # Process messages for each partition
        for partition, partition_messages in messages.items():
            if not partition_messages:
                continue

            try:
                logger.info(
                    f"Processing {len(partition_messages)} thumbnail messages from partition {partition}"
                )
                await consume_thumbnail_messages(partition_messages)

                # Commit offsets for this partition
                consumer.commit({partition: partition_messages[-1].offset + 1})
                logger.debug(f"Successfully committed offset for partition {partition}")

            except Exception as e:
                logger.error(
                    f"Error processing messages for partition {partition}: {str(e)}",
                    exc_info=True,
                )

    except Exception as e:
        logger.error(f"Error in handle_thumbnail_messages: {str(e)}", exc_info=True)
