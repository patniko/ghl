import os
import uuid
import csv
import mimetypes
import io
from typing import List, Optional, BinaryIO
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

# from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from loguru import logger

from auth import validate_jwt
from db import get_db
from models import (
    User,
    Batch,
    Project,
    File as FileModel,
    FileResponse,
    FileUpdate,
    ProcessingStatus,
    FileType,
    Check,
)
from services import storage
from middleware import get_organization_from_path
from services.storage import get_storage_backend, get_project_storage_path

router = APIRouter()


def detect_file_type(filename: str) -> str:
    """Detect file type based on extension"""
    ext = os.path.splitext(filename)[1].lower()

    if ext in [".dcm", ".dicom"]:
        return FileType.DICOM
    elif ext == ".csv":
        return FileType.CSV
    elif ext in [".mp4", ".mpeg4", ".m4v"]:
        return FileType.MP4
    elif ext == ".npz":
        return FileType.NPZ
    elif ext == ".json":
        return FileType.JSON
    else:
        # Default to the extension without the dot
        return ext[1:] if ext.startswith(".") else ext


def extract_csv_headers(file_path: str) -> List[str]:
    """Extract headers from a CSV file"""
    try:
        with open(file_path, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)  # Get the first row as headers
            return headers
    except Exception as e:
        logger.error(f"Error extracting CSV headers: {str(e)}")
        return []


def extract_csv_headers_from_fileobj(file_obj: BinaryIO) -> List[str]:
    """Extract headers from a CSV file-like object"""
    try:
        # Reset file pointer to beginning
        file_obj.seek(0)

        # Read as text
        text_content = io.TextIOWrapper(file_obj, encoding="utf-8")
        reader = csv.reader(text_content)
        headers = next(reader)  # Get the first row as headers

        # Reset file pointer again for future use
        file_obj.seek(0)

        return headers
    except Exception as e:
        logger.error(f"Error extracting CSV headers from file object: {str(e)}")
        return []


@router.post("/upload", response_model=FileResponse)
async def upload_file(
    file: UploadFile = File(...),
    batch_id: int = Form(...),
    project_id: Optional[int] = Form(None),
    file_type: Optional[str] = Form(None),
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Upload a file, associating it with a batch and optionally a project"""
    # Check if batch exists
    batch_stmt = select(Batch).where(
        Batch.id == batch_id,
        Batch.user_id == current_user["user_id"],
        Batch.organization_id == organization.id,
    )
    batch = db.execute(batch_stmt).scalar_one_or_none()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    # Check if project exists if project_id is provided
    project = None
    if project_id:
        project_stmt = select(Project).where(
            Project.id == project_id,
            Project.user_id == current_user["user_id"],
            Project.organization_id == organization.id,
        )
        project = db.execute(project_stmt).scalar_one_or_none()
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

    # Determine file type
    original_filename = file.filename
    detected_file_type = detect_file_type(original_filename)
    final_file_type = file_type or detected_file_type

    # Generate a unique filename to avoid collisions
    unique_filename = f"{uuid.uuid4().hex}_{original_filename}"

    # Determine the storage path based on project and batch
    if project:
        # If project is provided, use project-specific storage path
        storage_path = get_project_storage_path(
            project, f"batches/{batch_id}/{unique_filename}"
        )
    else:
        # If no project is provided but batch has a project_id, use that
        if batch.project_id:
            project_stmt = select(Project).where(Project.id == batch.project_id)
            batch_project = db.execute(project_stmt).scalar_one_or_none()
            if batch_project:
                # Update the project variable so it's used by get_storage_backend later
                project = batch_project
                storage_path = get_project_storage_path(
                    batch_project, f"batches/{batch_id}/{unique_filename}"
                )
            else:
                # Fallback to using the batch's project_id directly
                storage_path = (
                    f"projects/{batch.project_id}/batches/{batch_id}/{unique_filename}"
                )
        else:
            # If no project association at all, raise an error
            # Files must always be associated with a project
            raise HTTPException(
                status_code=400,
                detail="Files must be associated with a project. Please provide a project_id or use a batch that is associated with a project.",
            )

    # Save the file
    try:
        # Get the appropriate storage backend based on project settings
        storage_backend = get_storage_backend(project)

        # Save the file using the storage backend
        file_path, file_size = await storage_backend.save_file(
            file,
            storage_path,
            file.content_type
            or mimetypes.guess_type(original_filename)[0]
            or "application/octet-stream",
        )

        # Create database record
        file_model = FileModel(
            organization_id=batch.organization_id,
            user_id=current_user["user_id"],
            batch_id=batch_id,
            project_id=project_id,  # Associate with project if provided
            filename=unique_filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            content_type=file.content_type
            or mimetypes.guess_type(original_filename)[0]
            or "application/octet-stream",
            file_type=final_file_type,
            file_metadata={},  # Initialize with empty dictionary
            processing_status=ProcessingStatus.PENDING,
            has_thumbnail=False,
        )

        # For CSV files, extract headers
        if final_file_type == FileType.CSV:
            # Reset file position
            await file.seek(0)

            # Read the CSV headers
            content = await file.read()
            file_obj = io.BytesIO(content)
            headers = extract_csv_headers_from_fileobj(file_obj)
            file_model.csv_headers = headers

        db.add(file_model)
        db.commit()
        db.refresh(file_model)

        # NOTE: We are just using a scheduler for now
        # Send message to Kafka for file processing if it's a supported type
        # if final_file_type in [
        #     FileType.CSV,
        #     FileType.DICOM,
        #     FileType.MP4,
        #     FileType.NPZ,
        # ]:
        #     logger.info(
        #         f"Sending file processing message for file {file_model.id} of type {final_file_type}"
        #     )
        #     # Queue the processing tasks in the background
        #     asyncio.create_task(
        #         send_file_processing_message(
        #             file_model.id, current_user["user_id"], final_file_type
        #         )
        #     )
        #     asyncio.create_task(
        #         send_thumbnail_processing_message(
        #             file_model.id, current_user["user_id"], final_file_type, "file"
        #         )
        #     )
        #     logger.info(f"Queued processing tasks for file {file_model.id}")

        return file_model

    except Exception as e:
        # Clean up the file if there was an error
        try:
            if "file_path" in locals():
                await storage_backend.delete_file(file_path)
        except Exception as cleanup_error:
            logger.error(f"Error cleaning up file: {str(cleanup_error)}")

        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/", response_model=List[FileResponse])
async def get_files(
    batch_id: Optional[int] = None,
    file_type: Optional[str] = None,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Get all files for the current user with optional filters"""
    # Query files with organization context
    query = select(FileModel).where(
        FileModel.user_id == current_user["user_id"],
        FileModel.organization_id == organization.id,
    )

    # If batch_id is provided, verify it exists and belongs to the user
    if batch_id is not None:
        batch_stmt = select(Batch).where(
            Batch.id == batch_id,
            Batch.user_id == current_user["user_id"],
            Batch.organization_id == organization.id,
        )
        batch = db.execute(batch_stmt).scalar_one_or_none()
        if not batch:
            raise HTTPException(status_code=404, detail="Batch not found")
        query = query.where(FileModel.batch_id == batch_id)

    # Apply file type filter if provided
    if file_type:
        query = query.where(FileModel.file_type == file_type)

    # Order by creation date, newest first
    query = query.order_by(FileModel.created_at.desc())

    result = db.execute(query)
    files = result.scalars().all()

    return files


@router.get("/{file_id}", response_model=FileResponse)
async def get_file(
    file_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Get a specific file"""
    # Get the file with organization context
    stmt = (
        select(FileModel)
        .join(Batch)
        .where(
            FileModel.id == file_id,
            FileModel.user_id == current_user["user_id"],
            FileModel.organization_id == organization.id,
            Batch.organization_id == FileModel.organization_id,
        )
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    return file


@router.get("/{file_id}/download")
async def download_file(
    file_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Download a file"""
    # Get the file with organization context
    stmt = (
        select(FileModel)
        .join(Batch)
        .where(
            FileModel.id == file_id,
            FileModel.user_id == current_user["user_id"],
            FileModel.organization_id == organization.id,
            Batch.organization_id == FileModel.organization_id,
        )
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Get the project if file is associated with one
        project = None
        if file.project_id:
            project_stmt = select(Project).where(Project.id == file.project_id)
            project = db.execute(project_stmt).scalar_one_or_none()
            # Make sure we're using the correct project data_region

        # Get the appropriate storage backend based on project settings
        storage_backend = get_storage_backend(project)

        # Get the file from storage
        file_obj = await storage_backend.get_file(file.file_path)

        # Create a streaming response
        return StreamingResponse(
            file_obj,
            media_type=file.content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{file.original_filename}"'
            },
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found in storage")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving file: {str(e)}")


@router.put("/{file_id}", response_model=FileResponse)
async def update_file(
    file_id: int,
    file_update: FileUpdate,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Update file metadata and project association"""
    stmt = select(FileModel).where(
        FileModel.id == file_id,
        FileModel.user_id == current_user["user_id"],
        FileModel.organization_id == organization.id,
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    # Check if project exists if project_id is provided
    if file_update.project_id is not None:
        if file_update.project_id > 0:  # Assign to a project
            project_stmt = select(Project).where(
                Project.id == file_update.project_id,
                Project.user_id == current_user["user_id"],
                Project.organization_id == organization.id,
            )
            project = db.execute(project_stmt).scalar_one_or_none()
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")

            # If file is being moved to a different project, we may need to move the file
            if file.project_id != file_update.project_id:
                # Get the old and new storage backends
                old_project = None
                if file.project_id:
                    old_project_stmt = select(Project).where(
                        Project.id == file.project_id
                    )
                    old_project = db.execute(old_project_stmt).scalar_one_or_none()

                old_storage = get_storage_backend(old_project)
                new_storage = get_storage_backend(project)

                # If storage backends are different, we need to move the file
                if old_storage != new_storage:
                    # This would be a complex operation requiring downloading and re-uploading
                    # For now, we'll just update the database record
                    logger.warning(
                        f"File {file_id} is being moved between storage backends. "
                        "Physical file will remain in the original location."
                    )

            # Update the project_id
            file.project_id = file_update.project_id
        else:  # Remove from project
            file.project_id = None

    # Update other fields
    if file_update.file_type is not None:
        file.file_type = file_update.file_type
    if file_update.file_metadata is not None:
        file.file_metadata = file_update.file_metadata
    if file_update.has_thumbnail is not None:
        file.has_thumbnail = file_update.has_thumbnail
    if file_update.thumbnail is not None:
        file.thumbnail = file_update.thumbnail
    if file_update.potential_mappings is not None:
        file.potential_mappings = file_update.potential_mappings

    db.commit()
    db.refresh(file)

    return file


@router.get("/{file_id}/potential-mappings")
async def get_potential_mappings(
    file_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Get potential column-to-check mappings for a CSV file"""
    # Get the file with organization context
    stmt = (
        select(FileModel)
        .join(Batch)
        .where(
            FileModel.id == file_id,
            FileModel.user_id == current_user["user_id"],
            FileModel.file_type == FileType.CSV,
            FileModel.organization_id == organization.id,
            Batch.organization_id == FileModel.organization_id,
        )
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="CSV file not found")

    if not file.potential_mappings:
        # If no potential mappings exist yet, the file might still be processing
        if (
            file.processing_status == ProcessingStatus.PENDING
            or file.processing_status == ProcessingStatus.PROCESSING
        ):
            return {
                "message": "File is still being processed. Potential mappings not available yet.",
                "status": file.processing_status,
                "mappings": None,
            }
        else:
            return {
                "message": "No potential mappings available for this file.",
                "status": file.processing_status,
                "mappings": None,
            }

    # Return the potential mappings
    return {
        "message": "Potential mappings retrieved successfully",
        "status": file.processing_status,
        "mappings": file.potential_mappings,
    }


@router.put("/{file_id}/potential-mappings")
async def update_potential_mappings(
    file_id: int,
    mappings: dict,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Update potential column-to-check mappings for a CSV file"""
    # Get the file with organization context
    stmt = (
        select(FileModel)
        .join(Batch)
        .where(
            FileModel.id == file_id,
            FileModel.user_id == current_user["user_id"],
            FileModel.file_type == FileType.CSV,
            FileModel.organization_id == organization.id,
            Batch.organization_id == FileModel.organization_id,
        )
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="CSV file not found")

    # Update the potential mappings
    file.potential_mappings = mappings
    db.commit()
    db.refresh(file)

    return {
        "message": "Potential mappings updated successfully",
        "mappings": file.potential_mappings,
    }


@router.post("/{file_id}/apply-checks")
async def apply_checks(
    file_id: int,
    column_checks: dict,  # Map of column names to check IDs
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Apply specified checks to a CSV file and generate a data quality report"""
    # Get the file with organization context
    stmt = (
        select(FileModel)
        .join(Batch)
        .where(
            FileModel.id == file_id,
            FileModel.user_id == current_user["user_id"],
            FileModel.file_type == FileType.CSV,
            FileModel.organization_id == organization.id,
            Batch.organization_id == FileModel.organization_id,
        )
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="CSV file not found")

    # Get all checks
    checks_stmt = select(Check).where(
        Check.id.in_(
            [check_id for check_ids in column_checks.values() for check_id in check_ids]
        )
    )
    checks_result = db.execute(checks_stmt)
    checks = {check.id: check for check in checks_result.scalars().all()}

    try:
        # Read the CSV file
        if file.file_path.startswith("s3://"):
            # For S3 storage
            # Get the project if file is associated with one
            project = None
            if file.project_id:
                project_stmt = select(Project).where(Project.id == file.project_id)
                project = db.execute(project_stmt).scalar_one_or_none()
                # Make sure we're using the correct project data_region
                
            # Get the appropriate storage backend
            storage_backend = get_storage_backend(project)
            file_obj = await storage_backend.get_file(file.file_path)
            text_content = io.TextIOWrapper(file_obj, encoding="utf-8")
            reader = csv.DictReader(text_content)
            data = list(reader)
        else:
            # For local storage
            with open(file.file_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                data = list(reader)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")

    if not data:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    # Apply checks to each column
    results = {}
    applied_checks = {}

    for column_name, check_ids in column_checks.items():
        # Extract values for this column
        values = [row.get(column_name) for row in data]

        # Convert values to appropriate type based on check data type
        # This is a simplified version - in a real implementation, you'd need more robust type conversion
        converted_values = []
        for val in values:
            if val is None or val == "":
                converted_values.append(None)
                continue

            try:
                # Try to convert to float for numeric checks
                converted_values.append(float(val))
            except (ValueError, TypeError):
                # Keep as string for non-numeric checks
                converted_values.append(val)

        # Apply each check
        column_results = {}
        column_applied_checks = []

        for check_id in check_ids:
            if check_id not in checks:
                continue

            check = checks[check_id]

            # Get the implementation function
            implementation = CHECK_IMPLEMENTATIONS.get(check.implementation)
            if not implementation:
                continue

            # Apply the check with parameters
            try:
                if check.parameters:
                    result = implementation(converted_values, **check.parameters)
                else:
                    result = implementation(converted_values)

                column_results[check.name] = result
                column_applied_checks.append(
                    {"id": check.id, "name": check.name, "parameters": check.parameters}
                )
            except Exception as e:
                column_results[check.name] = {"error": str(e)}

        results[column_name] = column_results
        applied_checks[column_name] = column_applied_checks

    # Update the file with the results
    file.processing_results = {
        "message": "Data quality checks applied successfully",
        "applied_checks": applied_checks,
        "results": results,
    }
    db.commit()
    db.refresh(file)

    return {
        "message": "Data quality checks applied successfully",
        "applied_checks": applied_checks,
        "results": results,
    }


@router.get("/previews/dicom/{file_id}")
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

    try:
        # Get the project if file is associated with one
        project = None
        if file.project_id:
            project_stmt = select(Project).where(Project.id == file.project_id)
            project = db.execute(project_stmt).scalar_one_or_none()
            # Make sure we're using the correct project data_region
            
        # Get the appropriate storage backend
        storage_backend = get_storage_backend(project)
        
        # Get the file from storage
        file_obj = await storage_backend.get_file(file.file_path)

        # Return streaming response
        return StreamingResponse(file_obj, media_type="application/dicom")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found in storage")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving DICOM file: {str(e)}"
        )


@router.get("/previews/csv/{file_id}")
async def preview_csv_file(
    file_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Get CSV file preview data"""
    # Get the file with organization context and verify it's a CSV file
    stmt = (
        select(FileModel)
        .join(Batch)
        .where(
            FileModel.id == file_id,
            FileModel.user_id == current_user["user_id"],
            FileModel.file_type == "csv",
            FileModel.organization_id == organization.id,
            Batch.organization_id == FileModel.organization_id,
        )
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="CSV file not found")

    try:
        # Read first 1000 rows of CSV file
        data = []
        headers = []
        total_rows = 0

        if file.file_path.startswith("s3://"):
            # For S3 storage
            # Get the project if file is associated with one
            project = None
            if file.project_id:
                project_stmt = select(Project).where(Project.id == file.project_id)
                project = db.execute(project_stmt).scalar_one_or_none()
                # Make sure we're using the correct project data_region
                
            # Get the appropriate storage backend
            storage_backend = get_storage_backend(project)
            file_obj = await storage_backend.get_file(file.file_path)
            text_content = io.TextIOWrapper(file_obj, encoding="utf-8")
            reader = csv.reader(text_content)
            headers = next(reader)  # Get headers

            # Count rows while reading
            for i, row in enumerate(reader):
                if i >= 1000:  # Limit to 1000 rows
                    break
                data.append(row)
                total_rows = i + 1
        else:
            # For local storage
            with open(file.file_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader)  # Get headers
                for i, row in enumerate(reader):
                    if i >= 1000:  # Limit to 1000 rows
                        break
                    data.append(row)

            # Count total rows
            total_rows = sum(1 for _ in open(file.file_path)) - 1  # Subtract header row

        return {
            "headers": headers,
            "data": data,
            "total_rows": total_rows,
            "preview_rows": len(data),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading CSV file: {str(e)}")


@router.get("/previews/npz/{file_id}")
async def preview_npz_file(
    file_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Get NPZ file preview data"""
    # Get the file with organization context and verify it's an NPZ file
    stmt = (
        select(FileModel)
        .join(Batch)
        .where(
            FileModel.id == file_id,
            FileModel.user_id == current_user["user_id"],
            FileModel.file_type == "npz",
            FileModel.organization_id == organization.id,
            Batch.organization_id == FileModel.organization_id,
        )
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="NPZ file not found")

    try:
        import numpy as np

        # Load NPZ file
        if file.file_path.startswith("s3://"):
            # For S3 storage, we need to download to a temporary file first
            # Get the project if file is associated with one
            project = None
            if file.project_id:
                project_stmt = select(Project).where(Project.id == file.project_id)
                project = db.execute(project_stmt).scalar_one_or_none()
                # Make sure we're using the correct project data_region
                
            # Get the appropriate storage backend
            storage_backend = get_storage_backend(project)
            file_obj = await storage_backend.get_file(file.file_path)
            file_bytes = file_obj.read()

            # Create a BytesIO object from the bytes
            bytes_io = io.BytesIO(file_bytes)
            data = np.load(bytes_io)
        else:
            # For local storage
            data = np.load(file.file_path)

        # Convert arrays to lists and get basic info
        preview_data = {}
        for key in data.files:
            array = data[key]
            preview_data[key] = {
                "shape": array.shape,
                "dtype": str(array.dtype),
                "preview": array.flatten()[:100].tolist(),  # First 100 elements
                "min": float(array.min()),
                "max": float(array.max()),
                "mean": float(array.mean()),
            }

        return preview_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading NPZ file: {str(e)}")


@router.delete("/{file_id}")
async def delete_file(
    file_id: int,
    current_user: User = Depends(validate_jwt),
    db: Session = Depends(get_db),
    organization=Depends(get_organization_from_path),
):
    """Delete a file"""
    # Get the file with organization context
    stmt = (
        select(FileModel)
        .join(Batch)
        .where(
            FileModel.id == file_id,
            FileModel.user_id == current_user["user_id"],
            FileModel.organization_id == organization.id,
            Batch.organization_id == FileModel.organization_id,
        )
    )

    result = db.execute(stmt)
    file = result.scalar_one_or_none()

    if not file:
        raise HTTPException(status_code=404, detail="File not found")

    # Get the project if file is associated with one
    project = None
    if file.project_id:
        project_stmt = select(Project).where(Project.id == file.project_id)
        project = db.execute(project_stmt).scalar_one_or_none()

    # Get the appropriate storage backend based on project settings
    storage_backend = get_storage_backend(project)

    # Delete the physical file using the storage backend
    try:
        await storage_backend.delete_file(file.file_path)
    except Exception as e:
        logger.error(f"Error deleting file from storage: {str(e)}")
        # Continue with database deletion even if physical file deletion fails

    # Delete the database record
    db.delete(file)
    db.commit()

    return {"message": "File deleted successfully"}
