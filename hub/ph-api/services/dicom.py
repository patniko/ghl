import os
import uuid
import json
from datetime import datetime
from typing import List, Optional

import pydicom
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse as FastAPIFileResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from kafka import KafkaProducer
from loguru import logger

from auth import validate_jwt
from db import get_db
from models import (
    User,
    Batch,
    ProcessingStatus,
    File as FileModel,
    FileResponse,
)
from consumers.kafka_config import get_default_config
from consumers.file_producer import send_thumbnail_processing_message
from middleware import get_organization_from_path

router = APIRouter()

# Directory to store uploaded files
UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "uploads",
    "projects",
    "default",
    "batches",
)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Create Kafka producer
def get_kafka_producer():
    """Get or create a Kafka producer"""
    try:
        kafka_config = get_default_config()
        producer = KafkaProducer(
            bootstrap_servers=kafka_config.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            api_version=kafka_config.api_version,
            # Add retry configuration
            retries=3,
            retry_backoff_ms=500,
            # Add timeout configuration
            request_timeout_ms=5000,
            connections_max_idle_ms=30000,
            # Add batch configuration for better performance
            batch_size=16384,
            linger_ms=5,
            buffer_memory=33554432,
        )
        # Test the connection
        producer.metrics()
        return producer
    except Exception as e:
        logger.error(f"Failed to create Kafka producer: {str(e)}")
        return None


# Send message to Kafka topic
def send_dicom_processing_message(file_id: int, user_id: int):
    """Send a message to the DICOM processing Kafka topic"""
    try:
        producer = get_kafka_producer()
        if not producer:
            logger.error("Kafka producer not available")
            # Process the DICOM file directly if Kafka is not available
            logger.info(f"Falling back to direct processing for file {file_id}")
            # Import here to avoid circular imports
            from consumers.evals.dicom_consumer import process_dicom_file
            import asyncio

            asyncio.create_task(process_dicom_file(file_id, user_id))
            return True

        # Create message
        message = {
            "file_id": file_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Send message with a shorter timeout
        try:
            future = producer.send("dicom_processing", message)
            result = future.get(timeout=5)  # Reduced timeout to 5 seconds
            producer.flush(timeout=5)  # Ensure all messages are sent with timeout

            logger.info(f"Sent DICOM processing message for file {file_id}: {result}")
            return True
        except Exception as kafka_error:
            logger.error(
                f"Kafka send error, falling back to direct processing: {str(kafka_error)}"
            )
            # Process the DICOM file directly if Kafka send fails
            from consumers.evals.dicom_consumer import process_dicom_file
            import asyncio

            asyncio.create_task(process_dicom_file(file_id, user_id))
            return True
    except Exception as e:
        logger.error(f"Error sending DICOM processing message: {str(e)}")
        return False


def extract_dicom_metadata(dicom_data):
    """Extract relevant metadata from a DICOM file"""
    metadata = {}

    # Extract basic metadata
    if hasattr(dicom_data, "PatientID"):
        metadata["patient_id"] = str(dicom_data.PatientID)

    if hasattr(dicom_data, "PatientName"):
        metadata["patient_name"] = str(dicom_data.PatientName)

    if hasattr(dicom_data, "BatchInstanceUID"):
        metadata["batch_instance_uid"] = str(dicom_data.BatchInstanceUID)

    if hasattr(dicom_data, "SeriesInstanceUID"):
        metadata["series_instance_uid"] = str(dicom_data.SeriesInstanceUID)

    if hasattr(dicom_data, "SOPInstanceUID"):
        metadata["sop_instance_uid"] = str(dicom_data.SOPInstanceUID)

    if hasattr(dicom_data, "Modality"):
        metadata["modality"] = str(dicom_data.Modality)

    if hasattr(dicom_data, "BatchDate"):
        try:
            batch_date = str(dicom_data.BatchDate)
            if len(batch_date) == 8:  # YYYYMMDD format
                metadata["batch_date"] = datetime(
                    int(batch_date[0:4]), int(batch_date[4:6]), int(batch_date[6:8])
                )
        except Exception:
            # If date parsing fails, don't include the date
            pass

    # Extract additional metadata for storage in JSON field
    additional_metadata = {}

    # Patient information
    if hasattr(dicom_data, "PatientSex"):
        additional_metadata["patient_sex"] = str(dicom_data.PatientSex)

    if hasattr(dicom_data, "PatientBirthDate"):
        additional_metadata["patient_birth_date"] = str(dicom_data.PatientBirthDate)

    if hasattr(dicom_data, "PatientAge"):
        additional_metadata["patient_age"] = str(dicom_data.PatientAge)

    # Batch information
    if hasattr(dicom_data, "BatchDescription"):
        additional_metadata["batch_description"] = str(dicom_data.BatchDescription)

    if hasattr(dicom_data, "BatchID"):
        additional_metadata["batch_id"] = str(dicom_data.BatchID)

    # Series information
    if hasattr(dicom_data, "SeriesDescription"):
        additional_metadata["series_description"] = str(dicom_data.SeriesDescription)

    if hasattr(dicom_data, "SeriesNumber"):
        additional_metadata["series_number"] = str(dicom_data.SeriesNumber)

    # Image information
    if hasattr(dicom_data, "InstanceNumber"):
        additional_metadata["instance_number"] = str(dicom_data.InstanceNumber)

    if hasattr(dicom_data, "ImageType"):
        additional_metadata["image_type"] = str(dicom_data.ImageType)

    if hasattr(dicom_data, "PixelSpacing"):
        additional_metadata["pixel_spacing"] = str(dicom_data.PixelSpacing)

    if hasattr(dicom_data, "Rows"):
        additional_metadata["rows"] = str(dicom_data.Rows)

    if hasattr(dicom_data, "Columns"):
        additional_metadata["columns"] = str(dicom_data.Columns)

    metadata["additional_metadata"] = additional_metadata

    return metadata


@router.post("/upload", response_model=FileResponse)
async def upload_dicom_file(
    file: UploadFile = File(...),
    batch_id: Optional[int] = None,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Upload a DICOM file, optionally associating it with a batch"""
    # Check if batch exists if batch_id is provided
    if batch_id:
        batch_stmt = select(Batch).where(
            Batch.id == batch_id, Batch.user_id == current_user["user_id"]
        )
        batch = db.execute(batch_stmt).scalar_one_or_none()
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
    if not file.filename.lower().endswith((".dcm", ".dicom")):
        raise HTTPException(
            status_code=400, detail="File must be a DICOM file (.dcm or .dicom)"
        )

    # Generate a unique filename
    unique_filename = f"{uuid.uuid4()}.dcm"

    # Create a proper path structure
    if batch_id:
        file_path = os.path.join(UPLOAD_DIR, str(batch_id), unique_filename)
        # Ensure the batch directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    else:
        # If no batch is provided, use a default batch directory
        file_path = os.path.join(UPLOAD_DIR, "default", unique_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the file
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # Read DICOM metadata
        dicom_data = pydicom.dcmread(file_path)
        metadata = extract_dicom_metadata(dicom_data)

        # Create database record
        file_model = FileModel(
            organization_id=batch.organization_id if batch else None,
            user_id=current_user["user_id"],
            batch_id=batch_id,
            filename=unique_filename,
            original_filename=file.filename,
            file_path=file_path,
            file_size=len(contents),
            content_type=file.content_type or "application/dicom",
            file_type="dicom",
            file_metadata={
                "patient_id": metadata.get("patient_id"),
                "patient_name": metadata.get("patient_name"),
                "batch_instance_uid": metadata.get("batch_instance_uid"),
                "series_instance_uid": metadata.get("series_instance_uid"),
                "sop_instance_uid": metadata.get("sop_instance_uid"),
                "modality": metadata.get("modality"),
                "batch_date": metadata.get("batch_date"),
                "dicom_metadata": metadata.get("additional_metadata"),
            },
            processing_status=ProcessingStatus.PENDING,
            has_thumbnail=False,
        )

        db.add(file_model)
        db.commit()
        db.refresh(file_model)

        # Send message to Kafka for processing
        send_dicom_processing_message(file_model.id, current_user["user_id"])

        # Also send a message for thumbnail processing
        logger.info(
            f"Sending thumbnail processing message for DICOM file {file_model.id}"
        )
        send_thumbnail_processing_message(
            file_model.id, current_user["user_id"], "dicom", "file"
        )

        return file_model

    except Exception as e:
        # Clean up the file if there was an error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500, detail=f"Error processing DICOM file: {str(e)}"
        )


@router.get("/files", response_model=List[FileResponse])
async def get_dicom_files(
    batch_instance_uid: Optional[str] = None,
    patient_id: Optional[str] = None,
    modality: Optional[str] = None,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get all DICOM files for the current user with optional filters"""
    query = select(FileModel).where(
        FileModel.user_id == current_user["user_id"], FileModel.file_type == "dicom"
    )

    # Apply filters if provided
    if batch_instance_uid:
        query = query.where(
            FileModel.file_metadata["batch_instance_uid"].astext == batch_instance_uid
        )

    if patient_id:
        query = query.where(FileModel.file_metadata["patient_id"].astext == patient_id)

    if modality:
        query = query.where(FileModel.file_metadata["modality"].astext == modality)

    # Order by creation date, newest first
    query = query.order_by(FileModel.created_at.desc())

    result = db.execute(query)
    files = result.scalars().all()

    return files


@router.get("/files/{file_id}", response_model=FileResponse)
async def get_dicom_file(
    file_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Get a specific DICOM file"""
    stmt = select(FileModel).where(
        FileModel.id == file_id,
        FileModel.user_id == current_user["user_id"],
        FileModel.file_type == "dicom",
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="DICOM file not found")

    return file


@router.get("/files/{file_id}/download")
async def download_dicom_file(
    file_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Download a DICOM file"""
    stmt = select(FileModel).where(
        FileModel.id == file_id,
        FileModel.user_id == current_user["user_id"],
        FileModel.file_type == "dicom",
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="DICOM file not found")

    if not os.path.exists(file.file_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    return FastAPIFileResponse(
        path=file.file_path,
        filename=file.original_filename,
        media_type="application/dicom",
    )


@router.get("/files/{file_id}/preview")
async def preview_dicom_file(
    file_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Get a DICOM file for preview (raw bytes)"""
    # Get the file with organization context and verify it's a DICOM file
    stmt = (
        select(FileModel)
        .join(Batch)
        .where(
            FileModel.id == file_id,
            FileModel.user_id == current_user["user_id"],
            FileModel.file_type == "dicom",
            FileModel.organization_id == organization.id,
            Batch.organization_id == FileModel.organization_id,
        )
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="DICOM file not found")

    if not os.path.exists(file.file_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    def iterfile():
        with open(file.file_path, mode="rb") as file_like:
            yield from file_like

    return StreamingResponse(iterfile(), media_type="application/dicom")


@router.delete("/files/{file_id}")
async def delete_dicom_file(
    file_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
):
    """Delete a DICOM file"""
    stmt = select(FileModel).where(
        FileModel.id == file_id,
        FileModel.user_id == current_user["user_id"],
        FileModel.file_type == "dicom",
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="DICOM file not found")

    # Delete the physical file
    if os.path.exists(file.file_path):
        os.remove(file.file_path)

    # Delete the database record
    db.delete(file)
    db.commit()

    return {"message": "DICOM file deleted successfully"}
