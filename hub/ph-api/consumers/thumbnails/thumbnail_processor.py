import os
import asyncio
from datetime import datetime, UTC
from loguru import logger
from sqlalchemy import select

from db import SessionLocal
from models import File, ProcessingStatus, Project
from services.storage import get_storage_backend

# Import specific thumbnail generators
from consumers.thumbnails.csv_thumbnail import generate_csv_thumbnail
from consumers.thumbnails.dicom_thumbnail import generate_dicom_thumbnail
from consumers.thumbnails.mp4_thumbnail import generate_mp4_thumbnail
from consumers.thumbnails.npz_thumbnail import generate_npz_thumbnail


async def process_file_thumbnail(file_id: int, user_id: int):
    """
    Generate a thumbnail for a file.

    Args:
        file_id: The ID of the file to process
        user_id: The ID of the user who owns the file
    """
    db = SessionLocal()
    try:
        # Get the file
        stmt = select(File).where(File.id == file_id, File.user_id == user_id)

        result = db.execute(stmt)
        file = result.scalar_one_or_none()

        if not file:
            logger.error(f"File not found: file_id={file_id}, user_id={user_id}")
            return

        # Skip if the file already has a thumbnail
        if file.has_thumbnail:
            logger.info(f"File {file_id} already has a thumbnail, skipping")
            return

        # Skip if the file hasn't been processed yet
        if file.processing_status != ProcessingStatus.COMPLETED:
            logger.info(
                f"File {file_id} hasn't been processed yet, skipping thumbnail generation"
            )
            return

        # Get the project if file is associated with one
        project = None
        if file.project_id:
            project_stmt = select(Project).where(Project.id == file.project_id)
            project = db.execute(project_stmt).scalar_one_or_none()

        # Get the appropriate storage backend based on project settings
        storage_backend = get_storage_backend(project, file.file_path)

        # Check if the file exists by trying to get it from storage
        try:
            # Just check if we can get the file, don't actually read it yet
            await storage_backend.get_file(file.file_path)
        except Exception as e:
            logger.error(
                f"File not found or inaccessible: {file.file_path}, error: {str(e)}"
            )
            # Update status to failed since file is missing or inaccessible
            file.processing_status = ProcessingStatus.FAILED
            file.processing_results = {
                "error": f"File not found or inaccessible at {file.file_path}",
                "details": {"file_id": file_id, "user_id": user_id, "error": str(e)},
            }
            db.commit()
            return

        # Generate thumbnail based on file type
        file_type = file.file_type.lower()
        thumbnail_data = None

        if file_type == "csv":
            thumbnail_data = await generate_csv_thumbnail(file)
        elif file_type in ["dicom", "dcm"]:
            thumbnail_data = await generate_dicom_thumbnail(file)
        elif file_type == "mp4":
            thumbnail_data = await generate_mp4_thumbnail(file)
        elif file_type == "npz":
            thumbnail_data = await generate_npz_thumbnail(file)
        else:
            logger.warning(
                f"No thumbnail generator available for file type: {file.file_type}"
            )
            return

        # Update the file with the thumbnail
        if thumbnail_data and "thumbnail" in thumbnail_data:
            file.thumbnail = thumbnail_data["thumbnail"]
            file.has_thumbnail = True
            file.thumbnail_generated_at = datetime.now(UTC)
            db.commit()
            logger.info(f"Generated thumbnail for file {file_id}")
        else:
            logger.warning(f"Failed to generate thumbnail for file {file_id}")

    except Exception as e:
        logger.error(f"Error in process_file_thumbnail: {str(e)}")
        db.rollback()
    finally:
        db.close()


async def process_dicom_thumbnail(file_id: int, user_id: int):
    """
    Generate a thumbnail for a DICOM file.

    Args:
        file_id: The ID of the DICOM file to process
        user_id: The ID of the user who owns the file
    """
    db = SessionLocal()
    try:
        # Get the file from the regular files table
        stmt = select(File).where(
            File.id == file_id,
            File.user_id == user_id,
            File.file_type.in_(["dicom", "dcm"]),
        )

        result = db.execute(stmt)
        dicom_file = result.scalar_one_or_none()

        if not dicom_file:
            logger.error(f"DICOM file not found: file_id={file_id}, user_id={user_id}")
            return

        # Skip if the file already has a thumbnail
        if dicom_file.has_thumbnail:
            logger.info(f"DICOM file {file_id} already has a thumbnail, skipping")
            return

        # Skip if the file hasn't been processed yet
        if dicom_file.processing_status != ProcessingStatus.COMPLETED:
            logger.info(
                f"DICOM file {file_id} hasn't been processed yet, skipping thumbnail generation"
            )
            return

        # Get the project if file is associated with one
        project = None
        if dicom_file.project_id:
            project_stmt = select(Project).where(Project.id == dicom_file.project_id)
            project = db.execute(project_stmt).scalar_one_or_none()

        # Get the appropriate storage backend based on project settings and file path
        storage_backend = get_storage_backend(project, dicom_file.file_path)

        # Check if the file exists by trying to get it from storage
        try:
            # Just check if we can get the file, don't actually read it yet
            await storage_backend.get_file(dicom_file.file_path)
        except Exception as e:
            logger.error(
                f"DICOM file not found or inaccessible: {dicom_file.file_path}, error: {str(e)}"
            )
            # Update status to failed since file is missing or inaccessible
            dicom_file.processing_status = ProcessingStatus.FAILED
            dicom_file.processing_results = {
                "error": f"DICOM file not found or inaccessible at {dicom_file.file_path}",
                "details": {"file_id": file_id, "user_id": user_id, "error": str(e)},
            }
            db.commit()
            return

        # Generate thumbnail
        thumbnail_data = await generate_dicom_thumbnail(dicom_file)

        # Update the file with the thumbnail
        if thumbnail_data and "thumbnail" in thumbnail_data:
            dicom_file.thumbnail = thumbnail_data["thumbnail"]
            dicom_file.has_thumbnail = True
            dicom_file.thumbnail_generated_at = datetime.now(UTC)
            db.commit()
            logger.info(f"Generated thumbnail for DICOM file {file_id}")
        else:
            logger.warning(f"Failed to generate thumbnail for DICOM file {file_id}")

    except Exception as e:
        logger.error(f"Error in process_dicom_thumbnail: {str(e)}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    # For testing
    if len(os.sys.argv) > 1:
        file_id = int(os.sys.argv[1])
        user_id = int(os.sys.argv[2]) if len(os.sys.argv) > 2 else 1
        file_type = os.sys.argv[3] if len(os.sys.argv) > 3 else "file"

        if file_type.lower() == "dicom":
            asyncio.run(process_dicom_thumbnail(file_id, user_id))
        else:
            asyncio.run(process_file_thumbnail(file_id, user_id))
    else:
        print("Usage: python thumbnail_processor.py <file_id> [<user_id>] [file|dicom]")
