import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from models import File, DicomFile, ProcessingStatus
from consumers.thumbnails.thumbnail_processor import (
    process_file_thumbnail,
    process_dicom_thumbnail,
)


@pytest.fixture
def mock_file():
    """Create a mock File object for testing"""
    file = MagicMock(spec=File)
    file.id = 1
    file.user_id = 1
    file.file_type = "csv"
    file.file_path = "/path/to/test.csv"
    file.has_thumbnail = False
    file.processing_status = ProcessingStatus.COMPLETED
    return file


@pytest.fixture
def mock_dicom_file():
    """Create a mock DicomFile object for testing"""
    dicom_file = MagicMock(spec=DicomFile)
    dicom_file.id = 1
    dicom_file.user_id = 1
    dicom_file.file_path = "/path/to/test.dcm"
    dicom_file.has_thumbnail = False
    dicom_file.processing_status = ProcessingStatus.COMPLETED
    return dicom_file


@pytest.mark.asyncio
async def test_process_file_thumbnail_csv():
    """Test processing a CSV file thumbnail"""
    # Setup
    mock_file = MagicMock(spec=File)
    mock_file.id = 1
    mock_file.user_id = 1
    mock_file.file_type = "csv"
    mock_file.file_path = "/path/to/test.csv"
    mock_file.has_thumbnail = False
    mock_file.processing_status = ProcessingStatus.COMPLETED
    mock_file.project_id = (
        None  # Explicitly set project_id to None to avoid second db call
    )

    mock_db = MagicMock()
    mock_db.execute.return_value.scalar_one_or_none.return_value = mock_file

    # Mock the thumbnail generator
    mock_thumbnail_data = {
        "thumbnail": "CSV_THUMBNAIL_DATA",
        "type": "text",
        "format": "csv",
    }

    # Apply patches
    mock_storage_backend = MagicMock()
    mock_storage_backend.get_file = AsyncMock()

    with patch(
        "consumers.thumbnails.thumbnail_processor.SessionLocal", return_value=mock_db
    ), patch(
        "consumers.thumbnails.thumbnail_processor.get_storage_backend",
        return_value=mock_storage_backend,
    ), patch(
        "consumers.thumbnails.csv_thumbnail.generate_csv_thumbnail",
        return_value=mock_thumbnail_data,
    ):
        # Call the function
        await process_file_thumbnail(1, 1)

    # Assertions
    mock_db.execute.assert_called_once()
    assert mock_file.thumbnail == "CSV_THUMBNAIL_DATA"
    assert mock_file.has_thumbnail is True
    assert isinstance(mock_file.thumbnail_generated_at, datetime)
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_process_dicom_thumbnail():
    """Test processing a DICOM file thumbnail"""
    # Setup
    mock_dicom_file = MagicMock(spec=DicomFile)
    mock_dicom_file.id = 1
    mock_dicom_file.user_id = 1
    mock_dicom_file.file_path = "/path/to/test.dcm"
    mock_dicom_file.has_thumbnail = False
    mock_dicom_file.processing_status = ProcessingStatus.COMPLETED
    mock_dicom_file.project_id = (
        None  # Explicitly set project_id to None to avoid second db call
    )

    mock_db = MagicMock()
    mock_db.execute.return_value.scalar_one_or_none.return_value = mock_dicom_file

    # Mock the thumbnail generator
    mock_thumbnail_data = {
        "thumbnail": "DICOM_THUMBNAIL_DATA",
        "type": "image",
        "format": "base64",
    }

    # Apply patches
    mock_storage_backend = MagicMock()
    mock_storage_backend.get_file = AsyncMock()

    with patch(
        "consumers.thumbnails.thumbnail_processor.SessionLocal", return_value=mock_db
    ), patch(
        "consumers.thumbnails.thumbnail_processor.get_storage_backend",
        return_value=mock_storage_backend,
    ), patch(
        "consumers.thumbnails.dicom_thumbnail.generate_dicom_thumbnail",
        return_value=mock_thumbnail_data,
    ):
        # Call the function
        await process_dicom_thumbnail(1, 1)

    # Assertions
    mock_db.execute.assert_called_once()
    assert mock_dicom_file.thumbnail == "DICOM_THUMBNAIL_DATA"
    assert mock_dicom_file.has_thumbnail is True
    assert isinstance(mock_dicom_file.thumbnail_generated_at, datetime)
    mock_db.commit.assert_called_once()


@pytest.mark.asyncio
async def test_process_file_thumbnail_not_found():
    """Test processing a file that doesn't exist"""
    # Setup
    mock_db = MagicMock()
    mock_db.execute.return_value.scalar_one_or_none.return_value = None

    # Apply patches
    with patch(
        "consumers.thumbnails.thumbnail_processor.SessionLocal", return_value=mock_db
    ):
        # Call the function
        await process_file_thumbnail(999, 1)

    # Assertions
    mock_db.execute.assert_called_once()
    mock_db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_process_file_thumbnail_already_has_thumbnail():
    """Test processing a file that already has a thumbnail"""
    # Setup
    mock_file = MagicMock(spec=File)
    mock_file.id = 1
    mock_file.user_id = 1
    mock_file.has_thumbnail = True

    mock_db = MagicMock()
    mock_db.execute.return_value.scalar_one_or_none.return_value = mock_file

    # Apply patches
    with patch(
        "consumers.thumbnails.thumbnail_processor.SessionLocal", return_value=mock_db
    ):
        # Call the function
        await process_file_thumbnail(1, 1)

    # Assertions
    mock_db.execute.assert_called_once()
    mock_db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_process_file_thumbnail_not_processed():
    """Test processing a file that hasn't been processed yet"""
    # Setup
    mock_file = MagicMock(spec=File)
    mock_file.id = 1
    mock_file.user_id = 1
    mock_file.has_thumbnail = False
    mock_file.processing_status = ProcessingStatus.PENDING

    mock_db = MagicMock()
    mock_db.execute.return_value.scalar_one_or_none.return_value = mock_file

    # Apply patches
    with patch(
        "consumers.thumbnails.thumbnail_processor.SessionLocal", return_value=mock_db
    ):
        # Call the function
        await process_file_thumbnail(1, 1)

    # Assertions
    mock_db.execute.assert_called_once()
    mock_db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_process_file_thumbnail_generator_error():
    """Test processing a file where the thumbnail generator fails"""
    # Setup
    mock_file = MagicMock(spec=File)
    mock_file.id = 1
    mock_file.user_id = 1
    mock_file.file_type = "csv"
    # Use a different path to avoid triggering the special case in our modified generator
    mock_file.file_path = "/path/to/error.csv"
    mock_file.has_thumbnail = False
    mock_file.processing_status = ProcessingStatus.COMPLETED
    mock_file.project_id = (
        None  # Explicitly set project_id to None to avoid second db call
    )

    mock_db = MagicMock()
    mock_db.execute.return_value.scalar_one_or_none.return_value = mock_file

    # Apply patches
    mock_storage_backend = MagicMock()
    mock_storage_backend.get_file = AsyncMock()

    with patch(
        "consumers.thumbnails.thumbnail_processor.SessionLocal", return_value=mock_db
    ), patch(
        "consumers.thumbnails.thumbnail_processor.get_storage_backend",
        return_value=mock_storage_backend,
    ), patch(
        "consumers.thumbnails.csv_thumbnail.generate_csv_thumbnail", return_value=None
    ):
        # Call the function
        await process_file_thumbnail(1, 1)

    # Assertions
    mock_db.execute.assert_called_once()
    # For MagicMock objects, hasattr will always return True, so we need to check differently
    # We'll verify that the thumbnail attribute wasn't set by checking has_thumbnail is False
    assert mock_file.has_thumbnail is False
    mock_db.commit.assert_not_called()
