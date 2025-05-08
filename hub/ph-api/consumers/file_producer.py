import json
from typing import Dict, Any, Literal
from loguru import logger
from kafka import KafkaProducer
from consumers.kafka_config import get_default_config

# Kafka topics
FILE_PROCESSING_TOPIC = "file_processing"
THUMBNAIL_PROCESSING_TOPIC = "thumbnail_processing"


def get_kafka_producer() -> KafkaProducer:
    """Create and return a Kafka producer"""
    try:
        kafka_config = get_default_config()
        producer = KafkaProducer(
            bootstrap_servers=kafka_config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            # Timeouts and connection settings
            connections_max_idle_ms=30000,
            metadata_max_age_ms=30000,  # Refresh metadata more frequently
            request_timeout_ms=30000,  # Timeout for produce requests
            max_block_ms=30000,  # How long send() will block
            # Retry configuration
            retry_backoff_ms=500,
            reconnect_backoff_ms=500,
            reconnect_backoff_max_ms=5000,
        )
        return producer
    except Exception as e:
        logger.error(f"Failed to create Kafka producer: {str(e)}")
        return None


async def send_message_to_kafka(topic: str, message: Dict[str, Any]) -> None:
    """
    Send a message to a Kafka topic asynchronously

    Args:
        topic: The Kafka topic to send the message to
        message: The message payload to send
    """
    try:
        producer = get_kafka_producer()
        if not producer:
            logger.error("Failed to get Kafka producer")
            return

        # Send the message without waiting for it to complete
        future = producer.send(topic, message)

        # Add callbacks but don't wait for them
        future.add_callback(
            lambda metadata: logger.info(
                f"Sent message to Kafka: topic={metadata.topic}, "
                f"partition={metadata.partition}, offset={metadata.offset}"
            )
        ).add_errback(
            lambda e: logger.error(f"Error sending message to Kafka: {str(e)}")
        )

        # Close the producer without waiting for message completion
        producer.close(timeout=5)

    except Exception as e:
        logger.error(f"Error sending message to Kafka: {str(e)}")


async def send_file_processing_message(
    file_id: int, user_id: int, file_type: str
) -> None:
    """
    Send a message to Kafka for file processing

    Args:
        file_id: The ID of the file to process
        user_id: The ID of the user who owns the file
        file_type: The type of the file (e.g., 'csv', 'dicom')
    """
    # Create the message payload
    message: Dict[str, Any] = {
        "file_id": file_id,
        "user_id": user_id,
        "file_type": file_type,
    }

    await send_message_to_kafka(FILE_PROCESSING_TOPIC, message)


async def send_thumbnail_processing_message(
    file_id: int,
    user_id: int,
    file_type: str,
    file_category: Literal["file", "dicom"] = "file",
) -> None:
    """
    Send a message to Kafka for thumbnail processing

    Args:
        file_id: The ID of the file to process
        user_id: The ID of the user who owns the file
        file_type: The type of the file (e.g., 'csv', 'dicom')
        file_category: The category of the file ('file' or 'dicom')
    """
    # Create the message payload
    message: Dict[str, Any] = {
        "file_id": file_id,
        "user_id": user_id,
        "file_type": file_type,
        "file_category": file_category,
    }

    await send_message_to_kafka(THUMBNAIL_PROCESSING_TOPIC, message)
