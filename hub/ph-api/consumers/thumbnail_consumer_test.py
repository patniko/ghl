import json
import pytest
from unittest.mock import MagicMock, patch
from confluent_kafka import Message, TopicPartition

from consumers.thumbnail_consumer import (
    process_thumbnail_message,
    consume_thumbnail_messages,
    handle_thumbnail_messages,
    SUPPORTED_FILE_TYPES,
)


@pytest.fixture
def mock_message():
    """Create a mock Kafka message"""
    message = MagicMock(spec=Message)
    message.value = json.dumps(
        {"file_id": 1, "user_id": 1, "file_type": "npz", "file_category": "file"}
    ).encode("utf-8")
    message.error.return_value = None
    message.partition = 0
    message.offset = 0
    return message


@pytest.fixture
def mock_consumer():
    """Create a mock Kafka consumer"""
    consumer = MagicMock()
    consumer.poll.return_value = {
        TopicPartition("test-topic", 0): [MagicMock(spec=Message)]
    }
    return consumer


@pytest.mark.asyncio
async def test_process_thumbnail_message_valid():
    """Test processing a valid thumbnail message"""
    message = {"file_id": 1, "user_id": 1, "file_type": "npz", "file_category": "file"}

    with patch("consumers.thumbnail_consumer.process_file_thumbnail") as mock_process:
        await process_thumbnail_message(message)
        mock_process.assert_called_once_with(1, 1)


@pytest.mark.asyncio
async def test_process_thumbnail_message_invalid_types():
    """Test processing a message with invalid types"""
    message = {"file_id": "not_an_int", "user_id": "not_an_int", "file_type": "npz"}

    with patch("consumers.thumbnail_consumer.process_file_thumbnail") as mock_process:
        await process_thumbnail_message(message)
        mock_process.assert_not_called()


@pytest.mark.asyncio
async def test_process_thumbnail_message_missing_fields():
    """Test processing a message with missing required fields"""
    message = {
        "file_id": 1,
        # missing user_id
        "file_type": "npz",
    }

    with patch("consumers.thumbnail_consumer.process_file_thumbnail") as mock_process:
        await process_thumbnail_message(message)
        mock_process.assert_not_called()


@pytest.mark.asyncio
async def test_process_thumbnail_message_unsupported_type():
    """Test processing a message with unsupported file type"""
    message = {"file_id": 1, "user_id": 1, "file_type": "unsupported"}

    with patch("consumers.thumbnail_consumer.process_file_thumbnail") as mock_process:
        await process_thumbnail_message(message)
        mock_process.assert_not_called()


@pytest.mark.asyncio
async def test_process_thumbnail_message_dicom():
    """Test processing a DICOM message"""
    message = {
        "file_id": 1,
        "user_id": 1,
        "file_type": "dicom",
        "file_category": "dicom",
    }

    with patch("consumers.thumbnail_consumer.process_dicom_thumbnail") as mock_process:
        await process_thumbnail_message(message)
        mock_process.assert_called_once_with(1, 1)


@pytest.mark.asyncio
async def test_consume_thumbnail_messages(mock_message):
    """Test consuming thumbnail messages from Kafka"""
    messages = [mock_message]

    with patch(
        "consumers.thumbnail_consumer.process_thumbnail_message"
    ) as mock_process:
        await consume_thumbnail_messages(messages)
        mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_consume_thumbnail_messages_invalid_json():
    """Test consuming messages with invalid JSON"""
    message = MagicMock(spec=Message)
    message.value = b"invalid json"
    message.partition = 0
    message.offset = 0

    with patch(
        "consumers.thumbnail_consumer.process_thumbnail_message"
    ) as mock_process:
        await consume_thumbnail_messages([message])
        mock_process.assert_not_called()


@pytest.mark.asyncio
async def test_handle_thumbnail_messages_success(mock_consumer):
    """Test successful handling of thumbnail messages"""
    with patch(
        "consumers.thumbnail_consumer.consume_thumbnail_messages"
    ) as mock_consume:
        await handle_thumbnail_messages(mock_consumer)
        mock_consume.assert_called_once()
        mock_consumer.commit.assert_called_once()


@pytest.mark.asyncio
async def test_handle_thumbnail_messages_no_messages():
    """Test handling when no messages are received"""
    mock_consumer = MagicMock()
    mock_consumer.poll.return_value = {}

    with patch(
        "consumers.thumbnail_consumer.consume_thumbnail_messages"
    ) as mock_consume:
        await handle_thumbnail_messages(mock_consumer)
        mock_consume.assert_not_called()
        mock_consumer.commit.assert_not_called()


@pytest.mark.asyncio
async def test_handle_thumbnail_messages_max_messages():
    """Test handling with max_messages parameter"""
    mock_consumer = MagicMock()

    await handle_thumbnail_messages(mock_consumer, max_messages=50)
    mock_consumer.poll.assert_called_once_with(timeout_ms=1000, max_records=50)


@pytest.mark.asyncio
async def test_handle_thumbnail_messages_max_messages_bounds():
    """Test max_messages parameter bounds"""
    mock_consumer = MagicMock()

    # Test lower bound
    await handle_thumbnail_messages(mock_consumer, max_messages=0)
    mock_consumer.poll.assert_called_with(timeout_ms=1000, max_records=1)

    # Test upper bound
    mock_consumer.reset_mock()
    await handle_thumbnail_messages(mock_consumer, max_messages=200)
    mock_consumer.poll.assert_called_with(timeout_ms=1000, max_records=100)


def test_supported_file_types():
    """Test that all required file types are supported"""
    assert "npz" in SUPPORTED_FILE_TYPES
    assert "csv" in SUPPORTED_FILE_TYPES
    assert "mp4" in SUPPORTED_FILE_TYPES
    assert "dicom" in SUPPORTED_FILE_TYPES
    assert "dcm" in SUPPORTED_FILE_TYPES
