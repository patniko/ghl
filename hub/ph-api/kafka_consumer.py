import json
from dataclasses import dataclass
from datetime import datetime
from threading import Thread
import asyncio
from typing import Any, Dict, List, Optional
from loguru import logger
from kafka import KafkaConsumer
from consumers.kafka_config import get_default_config
from consumers.notification_consumer import process_notifications
from consumers.evals.dicom_consumer import process_dicom_file
from consumers.evals.csv_consumer import process_csv_file
from consumers.evals.mp4_consumer import process_mp4_file
from consumers.evals.npz_consumer import process_npz_file
from consumers.thumbnail_consumer import process_thumbnail_message


@dataclass(frozen=True)
class Message:
    topic: str
    value: Dict[str, Any]


def create_kafka_consumer(
    topics: list[str], config: Optional[dict] = None, max_retries: int = 5
) -> Optional[KafkaConsumer]:
    """Create a Kafka consumer for the specified topics with retry mechanism"""
    kafka_config = config or get_default_config()
    retry_count = 0
    while retry_count < max_retries:
        try:
            logger.debug(
                f"Attempting to create Kafka consumer for topics: {topics} (attempt {retry_count + 1}/{max_retries})"
            )
            consumer = KafkaConsumer(
                *topics,
                bootstrap_servers=kafka_config.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset=kafka_config.auto_offset_reset,
                enable_auto_commit=kafka_config.enable_auto_commit,
                group_id=kafka_config.group_id,
                api_version=kafka_config.api_version,
                # Timeouts - session_timeout must be between 6000 and 300000
                session_timeout_ms=10000,
                request_timeout_ms=15000,
                heartbeat_interval_ms=3000,
                # Add connection timeout to prevent hanging
                connections_max_idle_ms=30000,
                max_poll_interval_ms=300000,
                # Retry configuration
                retry_backoff_ms=500,
                reconnect_backoff_ms=500,
                reconnect_backoff_max_ms=5000,
            )
            # Test the connection
            consumer.topics()
            logger.info("Successfully created Kafka consumer")
            return consumer
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                logger.error(
                    f"Failed to create Kafka consumer after {max_retries} attempts: {str(e)}"
                )
                return None
            wait_time = min(
                2**retry_count, 30
            )  # Exponential backoff capped at 30 seconds
            logger.warning(
                f"Failed to create Kafka consumer (attempt {retry_count}/{max_retries}). Retrying in {wait_time} seconds..."
            )
            import time

            time.sleep(wait_time)


async def process_message_batch(messages: List[Message]) -> None:
    """Route messages to appropriate processors"""
    for message in messages:
        try:
            logger.debug(
                f"Processing message from batch: {message.topic} - {message.value}"
            )

            # Example message routing based on topic
            if message.topic == "notifications":
                # Process notification messages
                user_id = message.value.get("user_id")
                event = message.value.get("event")
                content = message.value.get("content")

                if all([user_id, event, content]):
                    # Call your notification processor here
                    await process_notifications()
                else:
                    logger.error("Missing required fields in notification message")
                continue

            # Handle file processing messages
            if message.topic == "file_processing":
                # Process file messages
                file_id = message.value.get("file_id")
                user_id = message.value.get("user_id")
                file_type = message.value.get("file_type")

                if not all([file_id, user_id, file_type]):
                    logger.error("Missing required fields in file processing message")
                    continue

                logger.info(f"Processing file: id={file_id}, type={file_type}")

                # Normalize file type for processing
                file_type_lower = file_type.lower()

                try:
                    # Route to appropriate consumer based on file type
                    if file_type_lower in ["dicom", "dcm"]:
                        await process_dicom_file(file_id, user_id)
                    elif file_type_lower == "csv":
                        await process_csv_file(file_id, user_id)
                    elif file_type_lower == "mp4":
                        await process_mp4_file(file_id, user_id)
                    elif file_type_lower == "npz":
                        await process_npz_file(file_id, user_id)
                    else:
                        logger.warning(
                            f"No processor available for file type: {file_type}"
                        )
                except Exception as e:
                    logger.error(f"Error processing file {file_id}: {str(e)}")
                continue

            # Handle thumbnail processing messages
            if message.topic == "thumbnail_processing":
                # Process thumbnail messages
                file_id = message.value.get("file_id")
                user_id = message.value.get("user_id")
                file_type = message.value.get("file_type")
                file_category = message.value.get("file_category", "file")

                if not all([file_id, user_id, file_type]):
                    logger.error(
                        "Missing required fields in thumbnail processing message"
                    )
                    continue

                logger.info(
                    f"Processing thumbnail for {file_category}: id={file_id}, type={file_type}"
                )

                try:
                    await process_thumbnail_message(message.value)
                except Exception as e:
                    logger.error(
                        f"Error processing thumbnail for file {file_id}: {str(e)}"
                    )
                continue

            # Default case for unhandled topics
            logger.warning(f"Unhandled message topic: {message.topic}")

        except Exception as e:
            logger.error(f"Error processing message in batch: {str(e)}")
            continue


async def consume_messages(consumer: KafkaConsumer) -> None:
    """Consume and batch process messages from Kafka"""
    if not consumer:
        logger.error("No consumer provided")
        return

    # Wait for initial connection
    try:
        consumer.topics()
        logger.debug("Initial Kafka connection successful")
    except Exception as e:
        logger.error(f"Failed to establish initial Kafka connection: {str(e)}")
        return

    BATCH_SIZE = 25  # Process messages in smaller batches
    BATCH_TIMEOUT = 5  # Wait up to 5 seconds to collect messages

    logger.debug("Starting to consume messages")
    while True:
        try:
            # Poll for a batch of messages
            message_batch = []
            start_time = datetime.now()

            while (
                len(message_batch) < BATCH_SIZE
                and (datetime.now() - start_time).seconds < BATCH_TIMEOUT
            ):
                messages = consumer.poll(timeout_ms=1000)
                if not messages:
                    continue

                for topic_partition, records in messages.items():
                    for record in records:
                        message_batch.append(
                            Message(topic=record.topic, value=record.value)
                        )
                        if len(message_batch) >= BATCH_SIZE:
                            break
                    if len(message_batch) >= BATCH_SIZE:
                        break

            if not message_batch:
                await asyncio.sleep(1)  # Prevent tight polling loop
                continue

            # Process the batch of messages
            await process_message_batch(message_batch)
            logger.debug(
                f"Successfully processed batch of {len(message_batch)} messages"
            )

        except Exception as e:
            logger.error(f"Consumer error: {str(e)}")
            await asyncio.sleep(1)  # Prevent tight error loop
            continue


def start_kafka_consumer() -> Optional[Thread]:
    """Start a Kafka consumer in a separate thread"""

    def run_consumer():
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            logger.debug("Starting Kafka consumer")
            consumer = create_kafka_consumer(
                [
                    "notifications",
                    "generic_topic",
                    "file_processing",  # Topic for file processing
                    "thumbnail_processing",  # Topic for thumbnail processing
                    # Add more topics as needed
                ]
            )
            if not consumer:
                logger.error("Failed to create consumer")
                return

            try:
                loop.run_until_complete(consume_messages(consumer))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in consumer thread: {str(e)}")

    try:
        consumer_thread = Thread(target=run_consumer, daemon=True)
        consumer_thread.start()
        logger.debug("Kafka consumer thread started")
        return consumer_thread
    except Exception as e:
        logger.error(f"Failed to start consumer thread: {str(e)}")
        return None


if __name__ == "__main__":
    start_kafka_consumer()
